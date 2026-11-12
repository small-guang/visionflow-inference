"""Contracts for the same-process Orin CUDA/TensorRT pipeline.

CUDA and TensorRT are external here; device integration is exercised separately
on Orin. These fakes expose engine metadata and reject incorrectly bound memory.
"""
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "scripts/yolo_orin_pipeline.py"


def load_pipeline():
    if not PIPELINE_PATH.exists():
        raise AssertionError("same-process Orin pipeline is missing")
    spec = importlib.util.spec_from_file_location("orin_pipeline_tests", PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Engine:
    def __init__(self, family="yolo26m_pose", profiles=None):
        self.names = ["images", "output0"]
        self.shapes = {"images": (-1, 3, -1, -1), "output0": (-1, -1, 57)}
        if family == "yolov8n_face":
            self.shapes["output0"] = (-1, 5, -1)
        if family == "yolo26s_seg":
            self.names.append("output1")
            self.shapes.update(output0=(-1, -1, 38), output1=(-1, 32, -1, -1))
        self.profiles = profiles or [((1, 3, 192, 192), (1, 3, 384, 640), (4, 3, 640, 640))]
        self.num_optimization_profiles = len(self.profiles)
        self.num_io_tensors = len(self.names)
        self.context = Context(self)
        self.dtypes = {name: np.float32 for name in self.names}

    def get_tensor_name(self, index): return self.names[index]
    def get_tensor_mode(self, name): return "input" if name == "images" else "output"
    def get_tensor_dtype(self, name): return self.dtypes[name]
    def get_tensor_location(self, name): return "device"
    def get_tensor_format(self, name): return "linear"
    def is_shape_inference_io(self, name): return False
    def get_tensor_shape(self, name): return self.shapes[name]
    def get_tensor_profile_shape(self, name, profile): return self.profiles[profile]
    def create_execution_context(self): return self.context


class Context:
    def __init__(self, engine):
        self.engine = engine
        self.shape = None
        self.bound = {}
        self.calls = []
        self.invalid_output = False
        self.enqueue_result = True

    def set_optimization_profile_async(self, profile, stream):
        self.calls.append(("profile", profile, stream))
        return True

    def set_input_shape(self, name, shape):
        self.shape = tuple(shape)
        return True

    def infer_shapes(self): return []

    def get_tensor_shape(self, name):
        if self.invalid_output:
            return (-1, 300, 57)
        shape = self.engine.shapes[name]
        if name == "images": return self.shape
        if name == "output1": return (self.shape[0], 32, self.shape[2] // 4, self.shape[3] // 4)
        if shape[1] == 5: return (self.shape[0], 5, 8400)
        return (self.shape[0], 300, shape[2])

    def set_tensor_address(self, name, pointer):
        self.bound[name] = pointer
        return True

    def execute_async_v3(self, stream_handle):
        if set(self.bound) != set(self.engine.names):
            raise AssertionError("all engine tensors must be bound")
        self.calls.append(("enqueue", stream_handle))
        return self.enqueue_result


class Array:
    next_pointer = 1024
    def __init__(self, shape, dtype=np.float32):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.size = int(np.prod(shape))
        self.ndim = len(shape)
        self.flags = types.SimpleNamespace(c_contiguous=True)
        self.data = types.SimpleNamespace(ptr=Array.next_pointer)
        Array.next_pointer += max(1, self.size * self.dtype.itemsize)
    def __getitem__(self, key): return Array((len(range(*key.indices(self.size))),), self.dtype)
    def reshape(self, shape):
        result = Array(shape, self.dtype)
        result.data = self.data
        return result


TRT = types.SimpleNamespace(TensorIOMode=types.SimpleNamespace(INPUT="input", OUTPUT="output"),
    TensorLocation=types.SimpleNamespace(DEVICE="device"),
    TensorFormat=types.SimpleNamespace(LINEAR="linear"), nptype=lambda value: value)
CP = types.SimpleNamespace(empty=lambda shape, dtype: Array(shape, dtype))


class OrinEngineContractTests(unittest.TestCase):
    def setUp(self): self.pipeline = load_pipeline()

    def runner(self, engine=None, family="yolo26m_pose"):
        return self.pipeline.TensorRtRunner(engine or Engine(family), object(), CP, TRT, family)

    def test_profile_coverage_requires_every_rectangle_and_batch_four(self):
        limited = Engine(profiles=[((1, 3, 192, 192), (1, 3, 384, 640), (1, 3, 640, 640))])
        with self.assertRaisesRegex(ValueError, "profile.*4"):
            self.runner(limited)
        rectangles = ((192, 640), (384, 640), (480, 640), (640, 640), (640, 480), (640, 384), (640, 192))
        engine = Engine(profiles=[((1, 3, h, w), (1, 3, h, w), (4, 3, h, w)) for h, w in rectangles])
        runner = self.runner(engine)
        self.assertEqual(runner.profile_for((4, 3, 640, 192)), 6)

    def test_invalid_engine_tensor_contract_is_rejected_before_execution(self):
        cases = []
        bad_type = Engine(); bad_type.dtypes["output0"] = np.float16; cases.append(bad_type)
        bad_width = Engine(); bad_width.shapes["output0"] = (-1, -1, 56); cases.append(bad_width)
        extra = Engine(); extra.names.append("unexpected"); extra.num_io_tensors += 1; cases.append(extra)
        for engine in cases:
            with self.subTest(engine=engine), self.assertRaises(ValueError):
                self.runner(engine)
            self.assertEqual(engine.context.calls, [])

    def test_gpu_input_pointer_is_bound_directly_on_the_supplied_stream(self):
        engine = Engine()
        runner = self.runner(engine)
        images = Array((4, 3, 384, 640))
        outputs = runner.infer(images, 73)
        self.assertEqual(outputs["output0"].shape, (4, 300, 57))
        self.assertEqual(engine.context.bound["images"], images.data.ptr)
        self.assertIn(("enqueue", 73), engine.context.calls)
        self.assertIn(("profile", 0, 73), engine.context.calls)

    def test_unresolved_output_or_failed_enqueue_never_returns_a_tensor(self):
        engine = Engine(); runner = self.runner(engine)
        engine.context.invalid_output = True
        with self.assertRaisesRegex(RuntimeError, "output0"):
            runner.infer(Array((1, 3, 640, 640)), 1)
        self.assertNotIn(("enqueue", 1), engine.context.calls)
        engine.context.invalid_output = False
        engine.context.enqueue_result = False
        with self.assertRaisesRegex(RuntimeError, "enqueue"):
            runner.infer(Array((1, 3, 640, 640)), 1)

    def test_segmentation_allocations_reject_outputs_above_existing_budgets(self):
        engine = Engine("yolo26s_seg")
        runner = self.runner(engine, "yolo26s_seg")
        for name, shape in (("output0", (4, 100_000, 38)), ("output1", (4, 32, 640, 640))):
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "budget"):
                runner._validate_output_shape(name, shape, 4)

    def test_plan_resolution_handles_repository_and_model_directory_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "yolo26m_pose_ensemble"
            model.mkdir()
            inference = root / "yolo26m_pose_onnx"
            (inference / "1").mkdir(parents=True)
            (inference / "config.pbtxt").write_text('default_model_filename: "custom.plan"')
            plan = inference / "1/custom.plan"; plan.write_bytes(b"engine")
            for argument in (root, model):
                actual_root = self.pipeline.repository_root(argument, "yolo26m_pose")
                self.assertEqual(actual_root, root)
                self.assertEqual(self.pipeline.plan_path(actual_root, "yolo26m_pose"), plan)

    def test_python_bindings_must_match_native_tensorrt_major_minor_patch(self):
        class Function:
            def __init__(self, version): self.version = version
            def __call__(self): return self.version
        bindings = types.SimpleNamespace(__version__="10.16.1.11")
        library = types.SimpleNamespace(getInferLibVersion=Function(101601))
        with mock.patch.object(self.pipeline.ctypes.util, "find_library", return_value="libnvinfer.so.10"), \
             mock.patch.object(self.pipeline.ctypes, "CDLL", return_value=library):
            self.assertIs(self.pipeline.validate_tensorrt_version(bindings), library)
            library.getInferLibVersion.version = 101600
            with self.assertRaisesRegex(RuntimeError, "version mismatch"):
                self.pipeline.validate_tensorrt_version(bindings)

    def test_gpu_profiles_preserve_public_tensor_and_api_contracts(self):
        from test.test_batch_model_repository import parse_config, _scalar
        for family in ("yolov8n_face", "yolo26m_pose", "yolo26s_seg"):
            directory = ROOT / "model_repository" / (family + "_ensemble")
            original = parse_config(directory / "config.pbtxt")
            gpu = parse_config(directory / "configs/orin_gpu.pbtxt")
            with self.subTest(family=family):
                self.assertEqual(gpu["input"], original["input"])
                self.assertEqual(gpu["output"], original["output"])
                self.assertEqual(_scalar(gpu, "max_batch_size"), 4)
                self.assertEqual(_scalar(gpu, "backend"), "python")
                self.assertNotIn("ensemble_scheduling", gpu)


class RequestIsolationTests(unittest.TestCase):
    def test_invalid_request_isolated_from_following_success_and_internal_details_hidden(self):
        pipeline = load_pipeline()
        class Tensor:
            def __init__(self, name, value): self.name, self.value = name, value
            def is_cpu(self): return True
            def as_numpy(self): return self.value
        class Response:
            def __init__(self, output_tensors=None, error=None):
                self.output_tensors, self.error = output_tensors or [], error
        model = pipeline.TritonPythonModel()
        model.family = "yolo26m_pose"
        model.pb = types.SimpleNamespace(get_input_tensor_by_name=lambda request, name: request.get(name),
            Tensor=Tensor, InferenceResponse=Response, TritonError=RuntimeError)
        model.input_errors = (pipeline.InputValidationError,)
        model.output_names = ("DETECTIONS", "KEYPOINTS", "NUM_DETECTIONS")
        def infer(images):
            if images[0, 0, 0, 0] == 1: raise RuntimeError("private driver details")
            return (np.ones((1, 100, 6), np.float32), np.ones((1, 100, 17, 3), np.float32), np.ones((1, 1), np.int32))
        model._infer = infer
        with self.assertLogs(pipeline.logger, level="ERROR"):
            responses = model.execute([{}, {"RAW_IMAGE": Tensor("RAW_IMAGE", np.array([0]))},
                {"RAW_IMAGE": Tensor("RAW_IMAGE", np.ones((1, 64, 64, 3), np.uint8))},
                {"RAW_IMAGE": Tensor("RAW_IMAGE", np.full((1, 32, 64, 3), 2, np.uint8))}])
        self.assertEqual(len(responses), 4)
        self.assertEqual(str(responses[0].error), "INVALID_INPUT: missing required input RAW_IMAGE")
        self.assertEqual(str(responses[1].error), "INVALID_INPUT: RAW_IMAGE batch must have shape [B,H,W,3]")
        self.assertNotIn("private driver details", str(responses[2].error))
        self.assertIsNone(responses[3].error)
        self.assertEqual([item.name for item in responses[3].output_tensors], ["DETECTIONS", "KEYPOINTS", "NUM_DETECTIONS"])


if __name__ == "__main__": unittest.main()
