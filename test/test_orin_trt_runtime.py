"""CPU 契约测试：真实 device/plan 数值由目标 Orin 集成验证覆盖。"""
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from scripts import orin_trt_runtime as production
from test.test_tbps_orin_pipeline import DeviceArray, TRT


class Context:
    def __init__(self, engine):
        self.engine = engine
        self.bound, self.calls = {}, []
        self.output_override = None
        self.fail = None

    def set_optimization_profile_async(self, profile, stream):
        self.calls.append(("profile", profile, stream))
        return self.fail != "profile"

    def set_input_shape(self, name, shape):
        self.shape = shape
        return self.fail != "shape"

    def infer_shapes(self): return ["missing"] if self.fail == "infer_shapes" else []

    def get_tensor_shape(self, name):
        return self.output_override or (self.shape[0], *self.engine.output_shapes[name])

    def set_tensor_address(self, name, address):
        self.bound[name] = address
        return self.fail != "bind"

    def execute_async_v3(self, stream_handle):
        self.calls.append(("enqueue", stream_handle))
        return self.fail != "enqueue"


class Engine:
    def __init__(self, input_name="image", shape=(3, 224, 224), outputs=None):
        self.input_name, self.shape = input_name, shape
        self.output_shapes = outputs or {"features": (512,)}
        self.names = [*self.output_shapes, input_name]
        self.num_io_tensors, self.num_optimization_profiles = len(self.names), 1
        self.profiles = [((1, *shape), (2, *shape), (4, *shape))]
        self.shapes = {input_name: (-1, *shape), **{k: (-1, *v) for k, v in self.output_shapes.items()}}
        self.dtypes = dict.fromkeys(self.names, np.float32)
        self.context = Context(self)

    def get_tensor_name(self, index): return self.names[index]
    def get_tensor_mode(self, name): return "input" if name == self.input_name else "output"
    def get_tensor_dtype(self, name): return self.dtypes[name]
    def get_tensor_location(self, name): return "device"
    def get_tensor_format(self, name): return "linear"
    def is_shape_inference_io(self, name): return False
    def get_tensor_shape(self, name): return self.shapes[name]
    def get_tensor_profile_shape(self, name, index): return self.profiles[index]
    def create_execution_context(self): return self.context


class FixedRunnerTests(unittest.TestCase):
    def runner(self, engine=None):
        engine = engine or Engine()
        cp = types.SimpleNamespace(ndarray=DeviceArray, empty=mock.Mock(side_effect=DeviceArray))
        return production.FixedShapeTensorRtRunner(engine, object(), cp, TRT, device_id=0,
            input_name=engine.input_name, input_shape=engine.shape, output_shapes=engine.output_shapes)

    def test_all_five_model_contracts_bind_named_pointers_and_reuse_outputs(self):
        for name, shape, outputs in (
                ("image", (3, 224, 224), {"unnorm_image_features": (512,)}),
                ("image", (3, 384, 128), {"embedding": (2048,)}),
                ("data", (3, 192, 192), {"fc1": (212,)}),
                ("input", (3, 112, 112), {"quality": (10,)}),
                ("input.1", (3, 112, 112), {"683": (512,)})):
            with self.subTest(input=name, shape=shape):
                engine = Engine(name, shape, outputs)
                runner = self.runner(engine)
                pointers = {k: value.data.ptr for k, value in runner.outputs.items()}
                for batch in (1, 4, 2, 3):
                    image = DeviceArray((batch, *shape))
                    result = runner.infer(image, 73)
                    self.assertEqual(engine.context.bound[name], image.data.ptr)
                    for key, template in outputs.items():
                        self.assertEqual(result[key].shape, (batch, *template))
                        self.assertEqual(result[key].data.ptr, pointers[key])
                self.assertEqual(runner.cp.empty.call_count, len(outputs))
                self.assertEqual(engine.context.calls.count(("enqueue", 73)), 4)
                self.assertEqual(engine.context.calls.count(("profile", 0, 73)), 1)

    def test_bad_engine_contract_or_incomplete_batch_profiles_fail_before_enqueue(self):
        cases = []
        engine = Engine(); engine.dtypes["features"] = np.float16; cases.append(engine)
        engine = Engine(); engine.shapes["features"] = (-1, 513); cases.append(engine)
        engine = Engine(); engine.names = ["image", "image"]; cases.append(engine)
        engine = Engine(); engine.profiles = [((1, 3, 224, 224),) * 3]; cases.append(engine)
        engine = Engine(); engine.profiles = [((2, 3, 224, 224), (1, 3, 224, 224), (4, 3, 224, 224))]; cases.append(engine)
        engine = Engine(); engine.get_tensor_location = lambda _: "host"; cases.append(engine)
        engine = Engine(); engine.get_tensor_format = lambda _: "CHW4"; cases.append(engine)
        engine = Engine(); engine.is_shape_inference_io = lambda _: True; cases.append(engine)
        for engine in cases:
            with self.subTest(engine=engine), self.assertRaises(ValueError):
                self.runner(engine)
            self.assertEqual(engine.context.calls, [])

    def test_invalid_device_input_and_shape_binding_failures_never_enqueue(self):
        for transform in (lambda a: setattr(a.device, "id", 1),
                          lambda a: setattr(a.flags, "c_contiguous", False),
                          lambda a: setattr(a, "dtype", np.float16),
                          lambda a: setattr(a, "shape", (5, 3, 224, 224))):
            runner = self.runner(); value = DeviceArray((1, 3, 224, 224)); transform(value)
            with self.assertRaises(ValueError): runner.infer(value, 5)
            self.assertEqual(runner.context.calls, [])
        for failure in ("profile", "shape", "infer_shapes", "bind"):
            runner = self.runner(); runner.context.fail = failure
            with self.subTest(failure=failure), self.assertRaises(RuntimeError):
                runner.infer(DeviceArray((1, 3, 224, 224)), 5)
            self.assertNotIn(("enqueue", 5), runner.context.calls)
        runner = self.runner(); runner.context.output_override = (-1, 512)
        with self.assertRaisesRegex(RuntimeError, "output shape"):
            runner.infer(DeviceArray((1, 3, 224, 224)), 5)
        self.assertNotIn(("enqueue", 5), runner.context.calls)

    def test_enqueue_failure_does_not_return_reusable_output(self):
        runner = self.runner(); runner.context.fail = "enqueue"
        with self.assertRaisesRegex(RuntimeError, "enqueue"):
            runner.infer(DeviceArray((1, 3, 224, 224)), 5)

    def test_native_plan_resolution_uses_config_and_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); native = root / "clip_image_trt"
            (native / "1").mkdir(parents=True)
            (native / "1/custom.plan").write_bytes(b"plan")
            config = native / "config.pbtxt"
            config.write_text('default_model_filename: "custom.plan"')
            self.assertEqual(production.plan_path(root, native.name), native / "1/custom.plan")
            for bad in ("../custom.plan", "..\\custom.plan", "/custom.plan", "custom.onnx"):
                config.write_text(f'default_model_filename: "{bad}"')
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    production.plan_path(root, native.name)
            with self.assertRaises(ValueError): production.plan_path(root, "../native")
            with self.assertRaises(ValueError): production.plan_path(root, native.name, "../1")

    def test_factory_retains_native_library_logger_runtime_and_engine(self):
        engine = Engine()
        native_library, logger = object(), object()
        runtime = types.SimpleNamespace(deserialize_cuda_engine=mock.Mock(return_value=engine))
        trt = types.SimpleNamespace(**vars(TRT),
            Logger=mock.Mock(return_value=logger), Runtime=mock.Mock(return_value=runtime),
            init_libnvinfer_plugins=mock.Mock(return_value=True))
        trt.Logger.WARNING = 2
        cp = types.SimpleNamespace(ndarray=DeviceArray, empty=DeviceArray)
        with tempfile.TemporaryDirectory() as directory:
            plan = Path(directory) / "model.plan"; plan.write_bytes(b"native-plan")
            with mock.patch.object(production, "plan_path", return_value=plan), \
                    mock.patch.object(production, "validate_tensorrt_version", return_value=native_library):
                runner = production.create_runner(directory, "native", "1", cp, trt,
                    device_id=0, input_name="image", input_shape=(3, 224, 224), output_shapes={"features": (512,)})
        self.assertIs(runner.runtime, runtime)
        self.assertIs(runner.engine, engine)
        self.assertIs(runner.native_library, native_library)
        self.assertIs(runner.trt_logger, logger)
        runtime.deserialize_cuda_engine.assert_called_once_with(b"native-plan")


if __name__ == "__main__":
    unittest.main()
