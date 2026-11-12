"""同进程固定形状 TensorRT 执行；不依赖业务模型、Triton 或 CUDA 导入。"""
import ctypes
import ctypes.util
from pathlib import Path
import re

import numpy as np


def validate_tensorrt_version(trt):
    """在反序列化前校验 Python/native major.minor.patch，并保留 native 句柄。"""
    version = tuple(int(value) for value in trt.__version__.split(".")[:3])
    if len(version) != 3 or version[0] < 10:
        raise RuntimeError("Orin GPU pipeline requires TensorRT 10+ Python bindings")
    library_name = ctypes.util.find_library("nvinfer") or f"libnvinfer.so.{version[0]}"
    library = ctypes.CDLL(library_name)
    native_version = library.getInferLibVersion
    native_version.argtypes = []
    native_version.restype = ctypes.c_int32
    actual = native_version()
    expected = version[0] * 10000 + version[1] * 100 + version[2]
    if actual != expected:
        raise RuntimeError(f"TensorRT Python/native version mismatch: Python {trt.__version__}, native {actual}")
    return library


def plan_path(repository, native_model, version="1"):
    if (not re.fullmatch(r"[A-Za-z0-9_.-]+", native_model)
            or native_model in {".", ".."} or not re.fullmatch(r"[0-9]+", str(version))):
        raise ValueError("native model and numeric version must be local path components")
    directory = Path(repository) / native_model
    config = (directory / "config.pbtxt").read_text(encoding="utf-8")
    match = re.search(r'\bdefault_model_filename\s*:\s*"([^"\r\n]+)"', config)
    if not match:
        raise ValueError(f"{native_model} has no default_model_filename")
    filename = match.group(1)
    if "/" in filename or "\\" in filename or not filename.endswith(".plan"):
        raise ValueError("Orin pipeline requires a local TensorRT .plan filename")
    path = directory / str(version) / filename
    if not path.is_file():
        raise FileNotFoundError(f"native TensorRT plan is missing: {path}")
    return path


class FixedShapeTensorRtRunner:
    """一个实例拥有一个 context；调用方持有输入并负责 stream 最终同步。

    只接收 device FP32 NCHW，batch 动态、空间维固定。输出按名称绑定，
    每个输出预分配 B4 存储；返回视图只在下次 infer 前有效。
    """
    def __init__(self, engine, runtime, cp, trt, *, device_id, input_name,
                 input_shape, output_shapes, max_batch_size=4):
        self.engine, self.runtime, self.cp = engine, runtime, cp
        self.device_id, self.input_name = int(device_id), input_name
        self.input_shape = tuple(input_shape)
        self.output_shapes = {name: tuple(shape) for name, shape in output_shapes.items()}
        self.max_batch_size = int(max_batch_size)
        if (self.max_batch_size != 4 or len(self.input_shape) != 3
                or any(type(value) is not int or value < 1 for value in self.input_shape)
                or not self.output_shapes or input_name in self.output_shapes
                or any(not shape or any(type(value) is not int or value < 1 for value in shape)
                       for shape in self.output_shapes.values())):
            raise ValueError("fixed TensorRT contract requires positive fixed shapes and B1–4")
        templates = {input_name: (-1, *self.input_shape),
                     **{name: (-1, *shape) for name, shape in self.output_shapes.items()}}
        names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
        if len(names) != len(set(names)) or set(names) != set(templates):
            raise ValueError(f"unexpected TensorRT tensor names: {names}")
        for name, shape in templates.items():
            mode = trt.TensorIOMode.INPUT if name == input_name else trt.TensorIOMode.OUTPUT
            if (engine.get_tensor_mode(name) != mode
                    or np.dtype(trt.nptype(engine.get_tensor_dtype(name))) != np.dtype(np.float32)
                    or engine.get_tensor_location(name) != trt.TensorLocation.DEVICE
                    or engine.get_tensor_format(name) != trt.TensorFormat.LINEAR
                    or engine.is_shape_inference_io(name)):
                raise ValueError(f"TensorRT {name} must use linear device FP32 data I/O")
            if tuple(engine.get_tensor_shape(name)) != shape:
                raise ValueError(f"TensorRT {name} shape differs from {shape}")
        self.profiles = [tuple(tuple(int(value) for value in shape) for shape in
                               engine.get_tensor_profile_shape(input_name, index))
                         for index in range(engine.num_optimization_profiles)]
        for bounds in self.profiles:
            if (len(bounds) != 3 or any(len(shape) != 4 for shape in bounds)
                    or any(not 1 <= low <= opt <= high for low, opt, high in zip(*bounds))):
                raise ValueError("invalid TensorRT profile bounds")
        for batch in range(1, self.max_batch_size + 1):
            self.profile_for((batch, *self.input_shape))
        self.context = engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("failed to create TensorRT execution context")
        self.active_profile = None
        self.outputs = {name: cp.empty((self.max_batch_size, *shape), dtype=np.float32)
                        for name, shape in self.output_shapes.items()}

    def profile_for(self, shape):
        for index, (minimum, _, maximum) in enumerate(self.profiles):
            if len(shape) == len(minimum) and all(
                    low <= actual <= high for low, actual, high in zip(minimum, shape, maximum)):
                return index
        raise ValueError(f"no TensorRT profile covers input shape {tuple(shape)}")

    def infer(self, images, stream_handle):
        if (not isinstance(images, self.cp.ndarray) or images.dtype != np.float32
                or not images.flags.c_contiguous or images.device.id != self.device_id
                or len(images.shape) != 4 or tuple(images.shape[1:]) != self.input_shape
                or not 1 <= images.shape[0] <= self.max_batch_size):
            raise ValueError(f"TensorRT input must be contiguous device FP32 [B,{self.input_shape}], B1–4")
        shape = tuple(images.shape)
        profile = self.profile_for(shape)
        if self.active_profile != profile:
            if not self.context.set_optimization_profile_async(profile, stream_handle):
                raise RuntimeError("TensorRT profile selection failed")
            self.active_profile = profile
        if not self.context.set_input_shape(self.input_name, shape):
            raise RuntimeError("TensorRT input shape rejected")
        if self.context.infer_shapes():
            raise RuntimeError("TensorRT has unresolved input shapes")
        outputs = {}
        for name, template in self.output_shapes.items():
            output_shape = tuple(self.context.get_tensor_shape(name))
            if output_shape != (shape[0], *template):
                raise RuntimeError(f"invalid TensorRT {name} output shape: {output_shape}")
            outputs[name] = self.outputs[name][:shape[0]]
        for name, value in ((self.input_name, images), *outputs.items()):
            if not self.context.set_tensor_address(name, int(value.data.ptr)):
                raise RuntimeError(f"TensorRT binding failed: {name}")
        if not self.context.execute_async_v3(stream_handle=stream_handle):
            raise RuntimeError("TensorRT enqueue failed")
        return outputs


def create_runner(repository, native_model, version, cp, trt, *, device_id,
                  input_name, input_shape, output_shapes):
    """调用方先选择 device；保留 logger/library/runtime/engine 直到 runner 销毁。"""
    native_library = validate_tensorrt_version(trt)
    path = plan_path(repository, native_model, version)
    trt_logger = trt.Logger(trt.Logger.WARNING)
    if not trt.init_libnvinfer_plugins(trt_logger, ""):
        raise RuntimeError("failed to initialize TensorRT plugins")
    runtime = trt.Runtime(trt_logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError(f"cannot deserialize native TensorRT plan: {path}")
    runner = FixedShapeTensorRtRunner(engine, runtime, cp, trt, device_id=device_id,
        input_name=input_name, input_shape=input_shape, output_shapes=output_shapes)
    runner.native_library, runner.trt_logger, runner.plan = native_library, trt_logger, path
    return runner
