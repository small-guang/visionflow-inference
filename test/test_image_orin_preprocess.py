"""CPU-only contract tests; explicit RUN_ORIN_GPU_TESTS=1 enables real CUDA parity."""

import asyncio
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import types
import unittest
from unittest import mock
import weakref

import cv2
import numpy as np

from scripts import image_orin_preprocess as production


ROOT = Path(__file__).resolve().parents[1]


class FakeTensor:
    def __init__(self, name, value):
        self.name, self.value = name, value

    def as_numpy(self):
        return self.value

    @staticmethod
    def from_dlpack(name, value):
        if not value.flags.c_contiguous:
            raise ValueError("DLPack must be contiguous")
        return FakeTensor(name, value)


PB = types.SimpleNamespace(
    Tensor=FakeTensor, TritonError=str,
    InferenceResponse=lambda **kw: types.SimpleNamespace(**kw),
    get_input_tensor_by_name=lambda request, name: request.get(name),
    get_output_config_by_name=lambda config, name: next(
        item for item in config["output"] if item["name"] == name),
    triton_string_to_numpy=lambda dtype: np.float32 if dtype == "TYPE_FP32" else np.float16,
)


def request(values):
    return {"IMAGE_BYTES": FakeTensor("IMAGE_BYTES", np.asarray(values, dtype=object).reshape(-1, 1))}


class FakeProcessor:
    def __init__(self, fail_first=False):
        self.calls = []
        self.fail_first = fail_first

    def decode(self, value, index=0):
        if value == b"bad":
            raise ValueError("bad image")
        return np.full((2, 3, 3), int(value), dtype=np.uint8)

    def preprocess_batch(self, images):
        self.calls.append([int(image[0, 0, 0]) for image in images])
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("CUDA failed")
        return np.stack([np.full((3, 2, 2), image[0, 0, 0], dtype=np.float32)
                         for image in images])

    def to_host(self, output):
        return output.copy()


class RequestRuntimeTests(unittest.TestCase):
    def test_cross_request_batch_preserves_order_and_bad_request_isolation(self):
        processor = FakeProcessor()
        runtime = production.ImagePreprocessRuntime("clip_image", PB, processor=processor)
        with mock.patch.object(processor, "to_host", wraps=processor.to_host) as bridge, \
                mock.patch.object(FakeTensor, "from_dlpack", side_effect=AssertionError("Orin CUDA IPC must not be used")):
            responses = runtime.execute([request([b"1"]), request([b"2", b"bad"]), request([b"3", b"4"])])
        bridge.assert_called_once()
        self.assertEqual(processor.calls, [[1, 3, 4]])
        self.assertIn("bad image", responses[1].error)
        self.assertEqual(responses[0].output_tensors[0].value.shape, (1, 3, 2, 2))
        np.testing.assert_array_equal(responses[2].output_tensors[0].value[:, 0, 0, 0], [3, 4])
        # Responses keep independent slices alive after the local full batch is gone.
        self.assertTrue(responses[2].output_tensors[0].value.flags.c_contiguous)
        self.assertIsNotNone(responses[2].output_tensors[0].value.base)

    def test_host_bridge_failure_is_an_error_without_cpu_reprocessing(self):
        processor = FakeProcessor()
        runtime = production.ImagePreprocessRuntime("clip_image", PB, processor=processor)
        with mock.patch.object(processor, "to_host", side_effect=RuntimeError("D2H failed")):
            response = runtime.execute([request([b"1"])])[0]
        self.assertIn("D2H failed", response.error)
        self.assertEqual(processor.calls, [[1]])

    def test_groups_do_not_split_requests_or_exceed_batch_four(self):
        processor = FakeProcessor()
        runtime = production.ImagePreprocessRuntime("sbs", PB, processor=processor)
        responses = runtime.execute([request([b"1"] * 3), request([b"2"] * 2), request([b"3"])])
        self.assertEqual(processor.calls, [[1, 1, 1], [2, 2, 3]])
        self.assertEqual([r.output_tensors[0].value.shape[0] for r in responses], [3, 2, 1])

    def test_cuda_failure_is_reported_for_its_group_and_next_group_runs(self):
        processor = FakeProcessor(fail_first=True)
        runtime = production.ImagePreprocessRuntime("sbs", PB, processor=processor)
        responses = runtime.execute([request([b"1"] * 4), request([b"2"])])
        self.assertIn("CUDA failed", responses[0].error)
        self.assertEqual(responses[1].output_tensors[0].value[0, 0, 0, 0], 2)

    def test_invalid_tensor_contract_is_rejected_without_gpu_submission(self):
        processor = FakeProcessor()
        runtime = production.ImagePreprocessRuntime("sbs", PB, processor=processor)
        cases = [{}, request([]), request([b"1"] * 5),
                 {"IMAGE_BYTES": FakeTensor("IMAGE_BYTES", np.ones((1, 2), dtype=object))},
                 {"IMAGE_BYTES": FakeTensor("IMAGE_BYTES", np.ones((1, 1), dtype=np.uint8))}]
        self.assertTrue(all(hasattr(r, "error") for r in runtime.execute(cases)))
        self.assertEqual(processor.calls, [])
        self.assertEqual(runtime.execute([]), [])

    def test_decode_uses_original_opencv_bgr_without_color_conversion(self):
        bgr = np.full((17, 23, 3), [7, 83, 241], dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", bgr)
        self.assertTrue(ok)
        for alias in production.SPECS:
            processor = production.GpuImagePreprocessor.__new__(production.GpuImagePreprocessor)
            processor.alias = alias
            with mock.patch.object(cv2, "cvtColor", side_effect=AssertionError("no CPU color conversion")):
                np.testing.assert_array_equal(processor.decode(encoded.tobytes()), bgr)
            for bad in (b"", b"invalid"):
                with self.assertRaises(ValueError):
                    processor.decode(bad)
        with self.assertRaisesRegex(ValueError, "must be bytes"):
            processor.decode(bytearray(encoded))


class Context:
    def __enter__(self): return self
    def __exit__(self, *_): return False


class FakeCupy:
    float32 = np.float32

    def __init__(self):
        self.next_pointer = 1000
        self.arrays = weakref.WeakValueDictionary()
        self.calls, self.pending = [], []
        self.sync_count = 0
        self.compile_count = 0
        self.raise_kernel = False
        self.stream = Context()
        self.stream.synchronize = self.synchronize
        self.cuda = types.SimpleNamespace(Device=lambda device: Context(),
                                          Stream=lambda **kw: self.stream)

    def asarray(self, value):
        array = types.SimpleNamespace(value=np.array(value), data=types.SimpleNamespace(ptr=self.next_pointer))
        # SimpleNamespace is not weak-referenceable; a small ndarray subclass is.
        class Array(np.ndarray):
            pass
        result = np.array(value).view(Array)
        result.pointer = self.next_pointer
        self.next_pointer += 1000
        # ndarray.data cannot be replaced, so proxy it with a regular object.
        class Proxy:
            pass
        proxy = Proxy()
        proxy.value, proxy.data = result, array.data
        self.arrays[proxy.data.ptr] = proxy
        return proxy

    def empty(self, shape, dtype):
        return self.asarray(np.empty(shape, dtype=dtype))

    def RawKernel(self, source, name, options):
        def launch(grid, block, args):
            records = args[0].value.copy()
            pointers = [int(value) for row in records for value in [row[0], *row[3:]] if value]
            self.pending.extend(pointers)
            self.calls.append((grid, block, records, args[1].value.shape))
            if self.raise_kernel:
                raise RuntimeError("launch failed")
        def compile_kernel():
            self.compile_count += 1
        launch.compile = compile_kernel
        return launch

    def synchronize(self):
        self.sync_count += 1
        if any(pointer not in self.arrays for pointer in self.pending):
            raise AssertionError("device buffer released before stream completion")
        self.pending.clear()


class DeviceSchedulingTests(unittest.TestCase):
    def test_mixed_sizes_use_one_kernel_and_one_output_batch_without_source_pack(self):
        cp = FakeCupy()
        processor = production.GpuImagePreprocessor("clip_image", cp_module=cp)
        self.assertEqual(cp.compile_count, 1)
        images = [np.zeros((h, w, 3), np.uint8) for h, w in ((12, 17), (224, 224), (91, 53), (12, 17))]
        out = processor.preprocess_batch(images)
        self.assertEqual(out.value.shape, (4, 3, 224, 224))
        self.assertEqual(len(cp.calls), 1)
        grid, _, records, _ = cp.calls[0]
        self.assertEqual(grid[1], 4)
        np.testing.assert_array_equal(records[:, 1:3], [[17, 12], [224, 224], [53, 91], [17, 12]])
        np.testing.assert_array_equal(records[1, 3:], 0)
        np.testing.assert_array_equal(records[0, 3:], records[3, 3:])
        self.assertEqual(cp.sync_count, 2)

    def test_launch_failure_still_synchronizes_before_temporaries_are_released(self):
        cp = FakeCupy()
        processor = production.GpuImagePreprocessor("sbs", cp_module=cp)
        cp.raise_kernel = True
        with self.assertRaisesRegex(RuntimeError, "launch failed"):
            processor.preprocess_batch([np.zeros((12, 17, 3), np.uint8)])
        self.assertEqual(cp.sync_count, 2)

    def test_device_coefficient_cache_is_bounded_and_invalid_batch_never_launches(self):
        cp = FakeCupy()
        processor = production.GpuImagePreprocessor("sbs", cp_module=cp)
        for height in range(1, 70):
            processor.preprocess_batch([np.zeros((height, 3, 3), np.uint8)])
        self.assertEqual(len(processor._tables), 64)
        before = len(cp.calls)
        for images in ([], [np.zeros((1, 1, 3), np.uint8)] * 5,
                       [np.zeros((0, 1, 3), np.uint8)], [np.zeros((1, 1, 3), np.float32)]):
            with self.assertRaises(ValueError):
                processor.preprocess_batch(images)
        self.assertEqual(len(cp.calls), before)


class ModelSelectionTests(unittest.TestCase):
    def test_cpu_default_gpu_selection_and_unknown_backend_rejection(self):
        for name, alias in (("clip_image_preprocess", "clip_image"), ("sbs_r50_ibn_image_preprocess", "sbs")):
            path = ROOT / "model_repository" / name / "1/model.py"
            spec = importlib.util.spec_from_file_location(name + "_gpu_tests", path)
            module = importlib.util.module_from_spec(spec)
            with mock.patch.dict(sys.modules, {"triton_python_backend_utils": PB}):
                spec.loader.exec_module(module)
            runtime = mock.Mock()
            runtime.execute.return_value = ["gpu"]
            factory = mock.Mock(return_value=runtime)
            helper = types.SimpleNamespace(ImagePreprocessRuntime=factory)
            for backend in (None, "cpu", "orin_gpu", "invalid"):
                config = {"output": [{"name": "image", "data_type": "TYPE_FP32"}]}
                if backend is not None:
                    config["parameters"] = {"preprocess_backend": {"string_value": backend}}
                args = {"model_config": json.dumps(config), "model_instance_device_id": "2", "model_instance_kind": "GPU"}
                model = module.TritonPythonModel()
                with mock.patch.dict(sys.modules, {"image_orin_preprocess": helper}), mock.patch.object(sys, "path", list(sys.path)):
                    if backend == "invalid":
                        with self.assertRaises(ValueError): model.initialize(args)
                    else:
                        model.initialize(args)
                        if backend == "orin_gpu":
                            factory.assert_called_with(alias, PB, 2)
                            result = model.execute([])
                            if asyncio.iscoroutine(result): result = asyncio.run(result)
                            self.assertEqual(result, ["gpu"])
                        else:
                            self.assertIsNone(model._gpu_runtime)
            config["parameters"]["preprocess_backend"]["string_value"] = "orin_gpu"
            with self.assertRaisesRegex(ValueError, "KIND_GPU"):
                module.TritonPythonModel().initialize({"model_config": json.dumps(config), "model_instance_kind": "CPU"})


class CubicCoefficientTests(unittest.TestCase):
    def test_arm_fused_cubic_coefficients_at_quantization_boundary(self):
        # Independent OpenCV 4.10 interpolateCubic compiled by ARM GCC -O3:
        # these real-crop rows produce tap0=-75, while separate operations give -74.
        for source, rows, expected_indices in (
                (884, (285, 381), ([655, 656, 657, 658], [876, 877, 878, 879])),
                (676, (299,), ([525, 526, 527, 528],))):
            indices, weights = production.opencv_cubic_coefficients(source, 384)
            for row, expected in zip(rows, expected_indices):
                np.testing.assert_array_equal(indices[row], expected)
                np.testing.assert_array_equal(weights[row], [-75, 549, 1790, -217])

    def test_fma_retains_residual_on_both_sides_of_fp32_midpoint(self):
        unit = 2 ** -23
        lower_case = (np.float32((1 + unit) * 2 ** -12),
                      np.float32((1 - unit) * 2 ** -12), np.float32(1 + unit))
        upper_case = (np.nextafter(np.float32(1), np.float32(2)),
                      np.nextafter(np.float32(1), np.float32(0)),
                      np.nextafter(np.float32(2 ** -47), np.float32(np.inf)))
        operands = np.asarray([lower_case, upper_case], dtype=np.float32)
        naive = (operands[:, 0].astype(np.float64) * operands[:, 1].astype(np.float64)
                 + operands[:, 2].astype(np.float64)).astype(np.float32)
        expected = np.asarray([1 + unit, 1 + unit], dtype=np.float32)
        self.assertTrue(np.all(naive != expected))
        actual = production._fma32(*operands.T)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(actual, expected)


@unittest.skipUnless(os.environ.get("RUN_ORIN_GPU_TESTS") == "1", "real GPU tests require explicit opt-in")
class RealOrinParityTests(unittest.TestCase):
    def test_mixed_batch_matches_arm_opencv_pixels_and_normalized_tensor(self):
        self.assertIn(platform.machine().lower(), ("aarch64", "arm64"))
        import cupy as cp
        rng = np.random.default_rng(72)
        for alias, spec in production.SPECS.items():
            processor = production.GpuImagePreprocessor(alias)
            width, height = spec["size"]
            images = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
                      for h, w in ((height, width), (539, 917), (1, 7), (401, 3))]
            actual = cp.asnumpy(processor.preprocess_batch(images))
            for index, image in enumerate(images):
                rgb = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)[..., ::-1]
                expected = ((rgb.astype(np.float32) / 255 - processor.mean) / processor.std).transpose(2, 0, 1)
                np.testing.assert_allclose(actual[index], expected, rtol=0, atol=5e-7)
                pixels = np.rint((actual[index] * processor.std[:, None, None] + processor.mean[:, None, None]) * 255).astype(np.int16)
                np.testing.assert_array_equal(pixels, rgb.transpose(2, 0, 1))

    def test_sbs_fused_coefficient_boundaries_match_rgb_before_resize(self):
        self.assertIn(platform.machine().lower(), ("aarch64", "arm64"))
        import cupy as cp
        rng = np.random.default_rng(172)
        processor = production.GpuImagePreprocessor("sbs")
        images = [rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
                  for height, width in ((884, 764), (676, 779), (884, 764), (676, 779))]
        actual = cp.asnumpy(processor.preprocess_batch(images))
        for index, image in enumerate(images):
            rgb = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (128, 384),
                             interpolation=cv2.INTER_CUBIC)
            expected = ((rgb.astype(np.float32) / 255 - processor.mean)
                        / processor.std).transpose(2, 0, 1)
            np.testing.assert_allclose(actual[index], expected, rtol=0, atol=5e-7)
            pixels = np.rint((actual[index] * processor.std[:, None, None]
                              + processor.mean[:, None, None]) * 255).astype(np.int16)
            np.testing.assert_array_equal(pixels, rgb.transpose(2, 0, 1))


if __name__ == "__main__":
    unittest.main()
