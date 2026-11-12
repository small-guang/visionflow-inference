"""TBPS production wrapper contracts; real CUDA/plan tests run on Orin."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_pipeline():
    path = ROOT / "scripts/tbps_orin_pipeline.py"
    if not path.is_file():
        raise AssertionError("production TBPS Orin pipeline is missing")
    spec = importlib.util.spec_from_file_location("tbps_pipeline_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeviceArray:
    next_pointer = 1024

    def __init__(self, shape, dtype=np.float32):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.flags = types.SimpleNamespace(c_contiguous=True)
        self.data = types.SimpleNamespace(ptr=DeviceArray.next_pointer)
        self.device = types.SimpleNamespace(id=0)
        DeviceArray.next_pointer += int(np.prod(shape)) * self.dtype.itemsize

    def __getitem__(self, index):
        value = DeviceArray((len(range(*index.indices(self.shape[0]))), *self.shape[1:]), self.dtype)
        value.data = self.data
        return value


class Engine:
    def __init__(self):
        self.names = ["image_feat", "image"]  # Deliberately not input-first.
        self.num_io_tensors = 2
        self.num_optimization_profiles = 1
        self.profiles = [((1, 3, 224, 224), (2, 3, 224, 224), (4, 3, 224, 224))]
        self.shapes = {"image": (-1, 3, 224, 224), "image_feat": (-1, 512)}
        self.dtypes = dict.fromkeys(self.names, np.float32)
        self.context = Context()

    def get_tensor_name(self, index): return self.names[index]
    def get_tensor_mode(self, name): return "input" if name == "image" else "output"
    def get_tensor_dtype(self, name): return self.dtypes[name]
    def get_tensor_location(self, name): return "device"
    def get_tensor_format(self, name): return "linear"
    def is_shape_inference_io(self, name): return False
    def get_tensor_shape(self, name): return self.shapes[name]
    def get_tensor_profile_shape(self, name, index): return self.profiles[index]
    def create_execution_context(self): return self.context


class Context:
    def __init__(self):
        self.bound = {}
        self.calls = []
        self.output_width = 512
        self.enqueue = True

    def set_optimization_profile_async(self, index, stream):
        self.calls.append(("profile", index, stream))
        return True

    def set_input_shape(self, name, shape): self.shape = shape; return True
    def infer_shapes(self): return []
    def get_tensor_shape(self, name): return (self.shape[0], self.output_width)
    def set_tensor_address(self, name, address): self.bound[name] = address; return True
    def execute_async_v3(self, stream_handle):
        self.calls.append(("enqueue", stream_handle))
        return self.enqueue


TRT = types.SimpleNamespace(TensorIOMode=types.SimpleNamespace(INPUT="input", OUTPUT="output"),
    TensorLocation=types.SimpleNamespace(DEVICE="device"),
    TensorFormat=types.SimpleNamespace(LINEAR="linear"), nptype=lambda dtype: dtype)
CP = types.SimpleNamespace(empty=lambda shape, dtype: DeviceArray(shape, dtype), ndarray=DeviceArray)


class TbpsRunnerTests(unittest.TestCase):
    def setUp(self): self.module = load_pipeline()

    def runner(self, engine=None):
        return self.module.TensorRtRunner(engine or Engine(), object(), CP, TRT, device_id=0)

    def test_every_batch_is_covered_and_named_device_pointers_bind_on_one_stream(self):
        engine = Engine()
        runner = self.runner(engine)
        for batch in (1, 4, 2, 3):
            image = DeviceArray((batch, 3, 224, 224))
            output = runner.infer(image, 73)
            self.assertEqual(output.shape, (batch, 512))
            self.assertEqual(engine.context.bound["image"], image.data.ptr)
            self.assertEqual(engine.context.bound["image_feat"], output.data.ptr)
        self.assertEqual(engine.context.calls.count(("profile", 0, 73)), 1)
        self.assertEqual(engine.context.calls.count(("enqueue", 73)), 4)

    def test_wrong_plan_contract_or_incomplete_profiles_are_rejected(self):
        cases = []
        engine = Engine(); engine.dtypes["image_feat"] = np.float16; cases.append(engine)
        engine = Engine(); engine.shapes["image_feat"] = (-1, 513); cases.append(engine)
        engine = Engine(); engine.profiles[0] = ((1, 3, 224, 224),) * 3; cases.append(engine)
        engine = Engine(); engine.names = ["images", "image_feat"]; cases.append(engine)
        for engine in cases:
            with self.subTest(engine=engine), self.assertRaises(ValueError):
                self.runner(engine)
            self.assertEqual(engine.context.calls, [])

    def test_invalid_gpu_tensor_or_unresolved_output_never_enqueues(self):
        engine = Engine(); runner = self.runner(engine)
        for shape in ((0, 3, 224, 224), (5, 3, 224, 224), (1, 3, 128, 384)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                runner.infer(DeviceArray(shape), 4)
        engine.context.output_width = -1
        with self.assertRaisesRegex(RuntimeError, "shape"):
            runner.infer(DeviceArray((1, 3, 224, 224)), 4)
        self.assertNotIn(("enqueue", 4), engine.context.calls)
        engine.context.output_width = 512
        engine.context.enqueue = False
        with self.assertRaisesRegex(RuntimeError, "enqueue"):
            runner.infer(DeviceArray((1, 3, 224, 224)), 4)

    def test_plan_paths_handle_both_triton_arguments_and_reject_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "TBPS_clip_image_ensemble").mkdir()
            native = root / "TBPS_clip_image_trt"
            (native / "1").mkdir(parents=True)
            (native / "1/native.plan").write_bytes(b"plan")
            config = native / "config.pbtxt"
            config.write_text('default_model_filename: "native.plan"')
            for path in (root, root / "TBPS_clip_image_ensemble"):
                self.assertEqual(self.module.repository_root(path), root)
                self.assertEqual(self.module.plan_path(root), native / "1/native.plan")
            for bad in ("../native.plan", "..\\native.plan", "/native.plan", "native.onnx"):
                config.write_text(f'default_model_filename: "{bad}"')
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    self.module.plan_path(root)

    def test_formal_gpu_profile_preserves_the_exact_public_contract(self):
        from test.test_batch_model_repository import parse_config, _scalar
        directory = ROOT / "model_repository/TBPS_clip_image_ensemble"
        original = parse_config(directory / "config.pbtxt")
        path = directory / "configs/orin_gpu.pbtxt"
        self.assertTrue(path.is_file(), "formal TBPS GPU config is missing")
        gpu = parse_config(path)
        self.assertEqual(gpu["input"], original["input"])
        self.assertEqual(gpu["output"], original["output"])
        self.assertEqual(gpu["parameters"][0], original["parameters"][0])
        self.assertEqual(_scalar(gpu, "max_batch_size"), 4)
        self.assertEqual(_scalar(gpu, "backend"), "python")
        self.assertNotIn("ensemble_scheduling", gpu)


class TbpsRequestTests(unittest.TestCase):
    def test_byte_objects_preserve_nul_and_invalid_inputs_do_not_reach_gpu(self):
        module = load_pipeline()
        payload = b"jpeg\0\0"
        self.assertEqual(module.validate_encoded_batch(np.array([[payload]], object)), [payload])
        for value in (np.array([payload], object), np.array([[payload]], dtype="S6"),
                      np.empty((0, 1), object), np.full((5, 1), payload, object),
                      np.array([[b""]], object), np.array([["text"]], object)):
            with self.subTest(shape=value.shape), self.assertRaises(module.InputValidationError):
                module.validate_encoded_batch(value)

    def test_requests_are_isolated_and_internal_errors_are_sanitized(self):
        module = load_pipeline()
        class Tensor:
            def __init__(self, name, value): self.name, self.value = name, value
            def is_cpu(self): return True
            def as_numpy(self): return self.value
        class Response:
            def __init__(self, output_tensors=None, error=None): self.output_tensors, self.error = output_tensors, error
        model = module.TritonPythonModel()
        model.pb = types.SimpleNamespace(get_input_tensor_by_name=lambda request, name: request.get(name),
            Tensor=Tensor, InferenceResponse=Response, TritonError=RuntimeError)
        def infer(values):
            if values[0, 0] == b"bad": raise module.InputValidationError("unable to decode image at batch index 0")
            if values[0, 0] == b"driver": raise RuntimeError("private driver details")
            return np.ones((len(values), 512), np.float32)
        model._infer = infer
        requests = [{}, *[{"IMAGE_BYTES": Tensor("IMAGE_BYTES", np.array([[x]], object))}
                          for x in (b"bad", b"driver", b"ok")]]
        with self.assertLogs(module.logger, level="ERROR"):
            responses = model.execute(requests)
        self.assertEqual(len(responses), 4)
        self.assertIn("INVALID_INPUT", str(responses[0].error))
        self.assertIn("batch index 0", str(responses[1].error))
        self.assertNotIn("private driver details", str(responses[2].error))
        self.assertIn("INTERNAL_ERROR", str(responses[2].error))
        self.assertIsNone(responses[3].error)
        self.assertEqual(responses[3].output_tensors[0].name, "image_feat")

    def test_stream_is_synchronized_when_preprocessing_or_inference_fails(self):
        module = load_pipeline()
        model = module.TritonPythonModel()
        model.device = mock.MagicMock()
        model.stream = mock.MagicMock()
        model.pre = types.SimpleNamespace(decode=lambda value, batch_index: np.zeros((1, 1, 3), np.uint8),
                                         preprocess=mock.Mock(side_effect=RuntimeError("CUDA failure")))
        with self.assertRaisesRegex(RuntimeError, "CUDA failure"):
            model._infer(np.array([[b"ok"]], object))
        model.stream.synchronize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
