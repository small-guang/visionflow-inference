"""Real CUDA comparisons against the retained NumPy/OpenCV postprocessors."""
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import cupy as cp
    CUDA_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except (ImportError, RuntimeError):
    cp = None
    CUDA_AVAILABLE = False


ROOT = Path(__file__).resolve().parents[1]
META = np.array([640, 640, 1, 0, 0, 640, 640], dtype=np.float32)


class Tensor:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def is_cpu(self):
        return isinstance(self.value, np.ndarray)

    def as_numpy(self):
        if not self.is_cpu():
            raise AssertionError("GPU inputs must not be copied through as_numpy")
        return self.value

    def to_dlpack(self):
        return self.value.toDlpack()

    @staticmethod
    def from_dlpack(name, capsule):
        return Tensor(name, cp.from_dlpack(capsule))


class Response:
    def __init__(self, output_tensors=None, error=None):
        self.output_tensors = output_tensors or []
        self.error = error


def load_model(name, gpu, host_outputs=False):
    sys.modules.setdefault("triton_python_backend_utils", types.ModuleType("triton_python_backend_utils"))
    path = ROOT / "model_repository" / name / "1" / "model.py"
    spec = importlib.util.spec_from_file_location("gpu_test_" + name + str(gpu), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.pb_utils.Tensor = Tensor
    module.pb_utils.InferenceResponse = Response
    module.pb_utils.TritonError = RuntimeError
    module.pb_utils.get_input_tensor_by_name = lambda request, key: request.get(key)
    model = module.TritonPythonModel()
    config = {"max_batch_size": 4}
    if host_outputs:
        config["parameters"] = {"GPU_OUTPUT_ON_CPU": {"string_value": "yes"}}
    model.initialize({"model_instance_kind": "GPU" if gpu else "CPU",
                      "model_instance_device_id": "0",
                      "model_config": json.dumps(config)})
    return model


def execute(model, values, gpu_inputs):
    request = {name: Tensor(name, cp.asarray(value) if gpu_inputs else value)
               for name, value in values.items()}
    return model.execute([request])[0]


def outputs(response):
    if response.error:
        raise AssertionError(str(response.error))
    return {tensor.name: cp.asnumpy(tensor.value) if not tensor.is_cpu() else tensor.value
            for tensor in response.output_tensors}


@unittest.skipUnless(CUDA_AVAILABLE, "requires CuPy and a CUDA GPU")
class YoloGpuPostprocessTests(unittest.TestCase):
    def compare(self, name, values, host_outputs=False):
        cpu = load_model(name, False)
        gpu = load_model(name, True, host_outputs=host_outputs)
        self.assertIsNotNone(getattr(gpu, "_gpu", None), "GPU instance must select GPU postprocessing")
        self.assertEqual(getattr(gpu._gpu, "host_outputs", False), host_outputs)
        expected = outputs(execute(cpu, values, False))
        for gpu_inputs in (False, True):
            with self.subTest(gpu_inputs=gpu_inputs):
                response = execute(gpu, values, gpu_inputs)
                actual = outputs(response)
                self.assertTrue(all(tensor.is_cpu() == host_outputs for tensor in response.output_tensors))
                self.assertEqual(set(actual), set(expected))
                for key in expected:
                    self.assertEqual(actual[key].dtype, expected[key].dtype)
                    self.assertEqual(actual[key].shape, expected[key].shape)
                    if np.issubdtype(expected[key].dtype, np.floating):
                        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-6, atol=1e-5)
                    else:
                        np.testing.assert_array_equal(actual[key], expected[key])

    def test_pose_stable_top100_and_empty_batch_rows(self):
        raw = np.zeros((4, 120, 57), dtype=np.float32)
        raw[:3, :, :4] = [40, 40, 600, 600]
        raw[:3, :, 0] += np.arange(120)
        raw[:3, :, 4] = 0.75
        raw[:3, :, 6:] = np.arange(51)
        raw[1, 3:, 4] = 0.1
        self.compare("yolo26m_pose_postprocess", {"RAW_OUTPUT": raw, "LETTERBOX_META": np.tile(META, (4, 1))})

    def test_configured_host_transport_keeps_actual_gpu_computation(self):
        for name, width in (("yolov8n_face_postprocess", 5), ("yolo26m_pose_postprocess", 57), ("yolo26s_seg_postprocess", 38)):
            with self.subTest(model=name):
                if width == 5:
                    raw = np.zeros((1, 5, 8), dtype=np.float32)
                    raw[0, :, 0] = [100, 100, 80, 80, 0.9]
                else:
                    raw = np.zeros((1, 8, width), dtype=np.float32)
                    raw[0, 0, :6] = [0, 0, 640, 640, 0.9, 0]
                meta = META[None, :5] if width == 5 else META[None]
                values = {"RAW_OUTPUT": raw, "LETTERBOX_META": meta}
                if width == 38:
                    raw[0, 0, 6] = 1
                    values["MASK_PROTO"] = np.ones((1, 32, 8, 8), dtype=np.float32)
                self.compare(name, values, host_outputs=True)
                model = load_model(name, True, host_outputs=True)
                arguments = [cp.asarray(raw), meta]
                if width == 38:
                    arguments.append(cp.asarray(values["MASK_PROTO"]))
                computed = model._gpu.process(*arguments)
                self.assertTrue(all(isinstance(value, cp.ndarray) for value in computed))

    def test_face_nms_clipping_and_empty_batch_rows(self):
        raw = np.zeros((4, 5, 150), dtype=np.float32)
        raw[0, :, :3] = np.array([[100, 100, 80, 80, 0.9],
                                 [101, 101, 80, 80, 0.8],
                                 [630, 630, 40, 40, 0.7]], dtype=np.float32).T
        raw[1, :, :2] = np.array([[200, 200, 60, 60, 0.6], [300, 300, 60, 60, 0.5]], dtype=np.float32).T
        self.compare("yolov8n_face_postprocess", {"RAW_OUTPUT": raw, "LETTERBOX_META": np.tile(META[:5], (4, 1))})

    def test_face_tied_scores_preserve_numpy_quicksort_order(self):
        raw = np.zeros((1, 5, 120), dtype=np.float32)
        raw[0, 0] = (np.arange(120) % 12) * 50 + 25
        raw[0, 1] = (np.arange(120) // 12) * 50 + 25
        raw[0, 2:4] = 20
        raw[0, 4] = 0.75
        self.compare("yolov8n_face_postprocess", {"RAW_OUTPUT": raw, "LETTERBOX_META": META[None, :5]})

    def test_seg_dynamic_slots_person_filter_stable_ties_and_packed_masks(self):
        raw = np.zeros((4, 120, 38), dtype=np.float32)
        raw[:3, :, :6] = [0, 0, 640, 640, 0.75, 0]
        raw[:3, :, 0] += np.arange(120)
        raw[:3, :, 6] = 1
        raw[1, 2:, 5] = 1
        raw[2, 3:, 4] = 0.1
        proto = np.zeros((4, 32, 16, 16), dtype=np.float32)
        yy, xx = np.indices((16, 16), dtype=np.float32)
        proto[:, 0] = xx + 4 * yy - 30.125
        self.compare("yolo26s_seg_postprocess", {"RAW_OUTPUT": raw, "MASK_PROTO": proto, "LETTERBOX_META": np.tile(META, (4, 1))})

    def test_seg_non_grid_crop_border_replicate_matches_opencv(self):
        raw = np.zeros((1, 3, 38), dtype=np.float32)
        raw[0, :, :6] = [[0, 0, 640, 640, 0.9, 0], [2.3, 1.2, 307.7, 205.8, 0.8, 0], [23.5, 32.5, 400.5, 640, 0.7, 0]]
        raw[0, :, 6] = 1
        proto = np.zeros((1, 32, 11, 20), dtype=np.float32)
        yy, xx = np.indices((11, 20), dtype=np.float32)
        proto[0, 0] = xx + 32 * yy - 105.125
        self.compare("yolo26s_seg_postprocess", {"RAW_OUTPUT": raw, "MASK_PROTO": proto, "LETTERBOX_META": META[None]})

    def test_seg_mixed_coefficients_and_zero_logits(self):
        rng = np.random.default_rng(42)
        raw = np.zeros((1, 8, 38), dtype=np.float32)
        raw[0, :, :6] = [13.25, 4.75, 611.5, 636.25, 0.9, 0]
        raw[0, 1:, 6:] = rng.normal(size=(7, 32)).astype(np.float32)
        proto = rng.normal(size=(1, 32, 160, 160)).astype(np.float32)
        self.compare("yolo26s_seg_postprocess", {"RAW_OUTPUT": raw, "MASK_PROTO": proto, "LETTERBOX_META": META[None]})

    def test_seg_budget_and_rounded_zero_area_contract(self):
        model = load_model("yolo26s_seg_postprocess", True)
        raw = np.zeros((2, 1, 38), dtype=np.float32)
        raw[:, 0, :6] = [3.5, 1, 4.5, 2, 0.9, 0]
        proto = np.zeros((2, 32, 1, 1), dtype=np.float32)
        values = {"RAW_OUTPUT": raw, "MASK_PROTO": proto, "LETTERBOX_META": np.tile(META, (2, 1))}
        result = outputs(execute(model, values, True))
        np.testing.assert_array_equal(result["NUM_DETECTIONS"], [[0], [0]])
        self.assertEqual(result["MASK_CROPS_PACKED"].shape, (2, 1, 160, 20))
        raw[:, 0, :4] = [0, 0, 640, 640]
        old_budget = model._gpu.api["MAX_MASK_OUTPUT_VALUES"]
        try:
            model._gpu.api["MAX_MASK_OUTPUT_VALUES"] = 51_199
            response = execute(model, values, True)
            self.assertRegex(str(response.error), "logical output values exceed budget 51199")
        finally:
            model._gpu.api["MAX_MASK_OUTPUT_VALUES"] = old_budget

    def test_gpu_invalid_sample_identifies_index_and_isolates_next_request(self):
        for name, width in (("yolov8n_face_postprocess", 5), ("yolo26m_pose_postprocess", 57), ("yolo26s_seg_postprocess", 38)):
            with self.subTest(model=name):
                model = load_model(name, True)
                self.assertIsNotNone(getattr(model, "_gpu", None))
                raw = np.zeros((2, width, 8) if width == 5 else (2, 8, width), dtype=np.float32)
                bad_raw = raw.copy()
                bad_raw[1, 0, 0] = np.nan
                meta = np.tile(META[:5] if width == 5 else META, (2, 1))
                good = {"RAW_OUTPUT": Tensor("RAW_OUTPUT", cp.asarray(raw)), "LETTERBOX_META": Tensor("LETTERBOX_META", cp.asarray(meta))}
                bad = dict(good, RAW_OUTPUT=Tensor("RAW_OUTPUT", cp.asarray(bad_raw)))
                if width == 38:
                    good["MASK_PROTO"] = Tensor("MASK_PROTO", cp.zeros((2, 32, 8, 8), dtype=cp.float32))
                    bad["MASK_PROTO"] = good["MASK_PROTO"]
                first, second = model.execute([bad, good])
                self.assertRegex(str(first.error), "INVALID_INPUT:.*batch index 1: RAW_OUTPUT values must be finite")
                self.assertIsNone(second.error)

    def test_invalid_cpu_inputs_are_rejected_before_cuda_upload(self):
        for name, width in (("yolov8n_face_postprocess", 5), ("yolo26m_pose_postprocess", 57), ("yolo26s_seg_postprocess", 38)):
            cpu = load_model(name, False)
            gpu = load_model(name, True)
            raw_shape = (1, width, 8) if width == 5 else (1, 8, width)
            meta = META[None, :5] if width == 5 else META[None]
            good = {"RAW_OUTPUT": np.zeros(raw_shape, dtype=np.float32), "LETTERBOX_META": meta}
            if width == 38:
                good["MASK_PROTO"] = np.zeros((1, 32, 8, 8), dtype=np.float32)
            huge_shape = (1, 5, 100_001) if width == 5 else (1, 100_001, width)
            # A zero-stride view represents an unsafe logical size without allocating it.
            huge = np.broadcast_to(np.zeros((1, 1, 1), dtype=np.float32), huge_shape)
            cases = [
                dict(good, RAW_OUTPUT=huge),
                dict(good, RAW_OUTPUT=good["RAW_OUTPUT"].astype(np.float64)),
                dict(good, RAW_OUTPUT=np.zeros((1,), dtype=np.float32)),
                dict(good, LETTERBOX_META=meta.astype(np.float64)),
                dict(good, LETTERBOX_META=np.zeros((1, 1), dtype=np.float32)),
            ]
            if width == 38:
                huge_proto = np.broadcast_to(np.zeros((1, 1, 1, 1), dtype=np.float32), (1, 32, 640, 640))
                cases.extend((dict(good, MASK_PROTO=huge_proto),
                              dict(good, MASK_PROTO=good["MASK_PROTO"].astype(np.float64))))
            for case_index, values in enumerate(cases):
                with self.subTest(model=name, case=case_index):
                    expected = execute(cpu, values, False)
                    self.assertRegex(str(expected.error), "^INVALID_INPUT:")
                    request = {key: Tensor(key, value) for key, value in values.items()}
                    with mock.patch.object(cp, "asarray", side_effect=AssertionError("CPU upload preceded validation")) as upload:
                        actual = gpu.execute([request])[0]
                    self.assertEqual(str(actual.error), str(expected.error))
                    upload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
