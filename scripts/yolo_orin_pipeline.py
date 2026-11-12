"""Single-process YOLO CUDA → TensorRT → CUDA serving for Jetson Orin.

Only RAW_IMAGE and final public results cross the Python backend boundary.
The serialized TensorRT plan and all intermediate arrays stay in this process.
The default ensemble configurations continue to use their original stages.
"""
import ctypes
import ctypes.util
import importlib.util
import json
import logging
from pathlib import Path
import re
import sys
from time import perf_counter
import uuid

import numpy as np


logger = logging.getLogger("yolo_orin_pipeline")
RECTANGLES = ((192, 640), (384, 640), (480, 640), (640, 640),
              (640, 480), (640, 384), (640, 192))
FAMILIES = {
    "yolov8n_face": (("DETECTIONS", "NUM_DETECTIONS"), 5),
    "yolo26m_pose": (("DETECTIONS", "KEYPOINTS", "NUM_DETECTIONS"), 7),
    "yolo26s_seg": (("DETECTIONS", "MASK_CROPS_PACKED", "NUM_DETECTIONS"), 7),
}


def repository_root(argument, family):
    path = Path(argument)
    if path.name == family + "_ensemble":
        path = path.parent
    if not (path / (family + "_ensemble")).is_dir():
        raise ValueError(f"model repository does not contain {family}_ensemble: {path}")
    return path


def plan_path(repository, family, version="1"):
    directory = Path(repository) / (family + "_onnx")
    config = (directory / "config.pbtxt").read_text(encoding="utf-8")
    match = re.search(r'\bdefault_model_filename\s*:\s*"([^"\r\n]+)"', config)
    if not match:
        raise ValueError(f"{family}_onnx has no default_model_filename")
    filename = match.group(1)
    if Path(filename).name != filename or not filename.endswith(".plan"):
        raise ValueError("Orin GPU pipeline requires a local TensorRT .plan filename")
    if not str(version).isdigit():
        raise ValueError("model_version must be numeric")
    path = directory / str(version) / filename
    if not path.is_file():
        raise FileNotFoundError(f"Orin GPU TensorRT plan is missing: {path}")
    return path


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


batching = _load_module("yolo_orin_request_batching", Path(__file__).with_name("yolo_request_batching.py"))
metrics_api = _load_module("yolo_orin_metrics", Path(__file__).with_name("yolo_orin_metrics.py"))
trt_runtime_api = _load_module("orin_trt_runtime", Path(__file__).with_name("orin_trt_runtime.py"))
InputValidationError = batching.InputValidationError


def validate_tensorrt_version(trt):
    """保留旧导出供现有 YOLO/TBPS 调用；实现由中性 runtime 提供。"""
    return trt_runtime_api.validate_tensorrt_version(trt)


class TensorRtRunner:
    """Named device bindings, validated profiles and reusable device outputs."""
    def __init__(self, engine, runtime, cp, trt, family, metrics=None):
        if family not in FAMILIES:
            raise ValueError(f"unsupported YOLO family: {family}")
        self.engine, self.runtime, self.cp, self.trt, self.family = engine, runtime, cp, trt, family
        self.metrics = metrics if metrics is not None else metrics_api.NullMetrics()
        self.output_names = ("output0", "output1") if family == "yolo26s_seg" else ("output0",)
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        if len(names) != len(set(names)) or set(names) != {"images", *self.output_names}:
            raise ValueError(f"unexpected TensorRT tensor names: {names}")
        templates = {"images": (-1, 3, -1, -1)}
        if family == "yolov8n_face":
            templates["output0"] = (-1, 5, -1)
        elif family == "yolo26m_pose":
            templates["output0"] = (-1, -1, 57)
        else:
            templates.update(output0=(-1, -1, 38), output1=(-1, 32, -1, -1))
        for name in names:
            expected_mode = trt.TensorIOMode.INPUT if name == "images" else trt.TensorIOMode.OUTPUT
            if engine.get_tensor_mode(name) != expected_mode:
                raise ValueError(f"unexpected TensorRT I/O mode for {name}")
            if np.dtype(trt.nptype(engine.get_tensor_dtype(name))) != np.dtype(np.float32):
                raise ValueError(f"TensorRT {name} must expose float32 tensors")
            if engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                raise ValueError(f"TensorRT {name} must use device memory")
            if engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
                raise ValueError(f"TensorRT {name} must use linear format")
            if engine.is_shape_inference_io(name):
                raise ValueError(f"shape-tensor I/O is unsupported: {name}")
            actual = tuple(engine.get_tensor_shape(name))
            expected = templates[name]
            if len(actual) != len(expected) or any(want != -1 and got != want for got, want in zip(actual, expected)):
                raise ValueError(f"TensorRT {name} has unexpected shape {actual}; expected {expected}")
        self.profiles = [tuple(tuple(int(v) for v in shape) for shape in engine.get_tensor_profile_shape("images", index))
                         for index in range(engine.num_optimization_profiles)]
        for batch in range(1, 5):
            for height, width in RECTANGLES:
                self.profile_for((batch, 3, height, width))
        self.context = engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("failed to create TensorRT execution context")
        self._active_profile = None
        self._buffers = {}

    def profile_for(self, shape):
        for index, bounds in enumerate(self.profiles):
            if len(bounds) != 3 or any(len(bound) != len(shape) for bound in bounds):
                continue
            minimum, _, maximum = bounds
            if all(low <= actual <= high for low, actual, high in zip(minimum, shape, maximum)):
                return index
        raise ValueError(f"no TensorRT optimization profile covers input shape {tuple(shape)}")

    def _validate_output_shape(self, name, shape, batch):
        if not shape or any(value < 1 for value in shape) or shape[0] != batch:
            raise RuntimeError(f"unresolved or invalid TensorRT {name} shape: {shape}")
        if name == "output1":
            valid = len(shape) == 4 and shape[1] == 32 and max(shape[2:]) <= 640
            budget = 8_388_608
        elif self.family == "yolov8n_face":
            valid = len(shape) == 3 and shape[1] == 5 and shape[2] <= 100_000 and shape[2] != 5
            budget = 4 * 5 * 100_000
        else:
            width = 57 if self.family == "yolo26m_pose" else 38
            valid = len(shape) == 3 and shape[1] <= 100_000 and shape[2] == width
            budget = 4_194_304 if self.family == "yolo26s_seg" else 4 * 100_000 * width
        if not valid or int(np.prod(shape, dtype=np.int64)) > budget:
            raise RuntimeError(f"TensorRT {name} shape violates YOLO contract or memory budget: {shape}")

    def infer(self, images, stream_handle):
        shape = tuple(images.shape)
        if images.dtype != np.float32 or not images.flags.c_contiguous:
            raise ValueError("TensorRT images must be contiguous device float32")
        if len(shape) != 4 or shape[1] != 3 or shape[0] not in range(1, 5) or shape[2:] not in RECTANGLES:
            raise ValueError(f"unsupported YOLO preprocessed shape: {shape}")
        profile = self.profile_for(shape)
        if self._active_profile != profile:
            if not self.context.set_optimization_profile_async(profile, stream_handle):
                raise RuntimeError(f"failed to select TensorRT profile {profile}")
            self._active_profile = profile
        if not self.context.set_input_shape("images", shape):
            raise RuntimeError(f"TensorRT rejected input shape: {shape}")
        missing = self.context.infer_shapes()
        if missing:
            raise RuntimeError(f"TensorRT shape inference has unspecified tensors: {missing}")
        if not self.context.set_tensor_address("images", int(images.data.ptr)):
            raise RuntimeError("failed to bind TensorRT images device pointer")
        outputs = {}
        for name in self.output_names:
            output_shape = tuple(int(value) for value in self.context.get_tensor_shape(name))
            self._validate_output_shape(name, output_shape, shape[0])
            size = int(np.prod(output_shape, dtype=np.int64))
            buffer = self._buffers.get(name)
            if buffer is None or buffer.size < size:
                buffer = self.cp.empty((size,), dtype=np.float32)
                self._buffers[name] = buffer
            value = buffer[:size].reshape(output_shape)
            if not self.context.set_tensor_address(name, int(value.data.ptr)):
                raise RuntimeError(f"failed to bind TensorRT {name} device pointer")
            outputs[name] = value
        self.metrics.trt_call(shape[0])
        try:
            if not self.context.execute_async_v3(stream_handle=stream_handle):
                raise RuntimeError("TensorRT enqueue failed")
        except Exception:
            self.metrics.trt_failure()
            raise
        return outputs


class TritonPythonModel:
    """Base class used by the three public model-directory wrappers."""
    family = None
    max_batch_size = 4
    metrics = metrics_api.NullMetrics()

    def initialize(self, args):
        if self.family not in FAMILIES:
            raise ValueError(f"unsupported YOLO family: {self.family}")
        if args.get("model_instance_kind") != "GPU":
            raise RuntimeError("orin_gpu requires a KIND_GPU Python instance")
        import cupy as cp
        import tensorrt as trt
        import triton_python_backend_utils as pb_utils

        self.cp, self.pb = cp, pb_utils
        self.metrics = metrics_api.PipelineMetrics(pb_utils, args)
        self.native_library = validate_tensorrt_version(trt)
        config = args.get("model_config", {})
        if isinstance(config, str):
            config = json.loads(config)
        self.max_batch_size = int(config.get("max_batch_size", 4))
        if self.max_batch_size != 4:
            raise ValueError("orin_gpu requires the existing max_batch_size=4 contract")
        root = repository_root(args["model_repository"], self.family)
        version = str(args.get("model_version", "1"))
        path = plan_path(root, self.family, version)
        self.output_names, self.meta_width = FAMILIES[self.family]
        self.device = cp.cuda.Device(int(args.get("model_instance_device_id", 0)))
        with self.device:
            self.stream = cp.cuda.Stream(non_blocking=True)
            self.trt_logger = trt.Logger(trt.Logger.WARNING)
            if not trt.init_libnvinfer_plugins(self.trt_logger, ""):
                raise RuntimeError("failed to initialize TensorRT plugins")
            runtime = trt.Runtime(self.trt_logger)
            engine = runtime.deserialize_cuda_engine(path.read_bytes())
            if engine is None:
                raise RuntimeError(f"failed to deserialize device-native TensorRT plan: {path}")
            self.runner = TensorRtRunner(engine, runtime, cp, trt, self.family, self.metrics)
            pre_dir = root / (self.family + "_preprocess") / version
            post_dir = root / (self.family + "_postprocess") / version
            pre_api = _load_module(self.family + "_orin_pre_api", pre_dir / "model.py")
            post_api = _load_module(self.family + "_orin_post_api", post_dir / "model.py")
            pre_gpu = _load_module(self.family + "_orin_pre_gpu", pre_dir / "gpu.py")
            post_gpu = _load_module(self.family + "_orin_post_gpu", post_dir / "gpu.py")
            cpu_args = dict(args, model_instance_kind="CPU")
            self.pre_model = pre_api.TritonPythonModel()
            self.pre_model.initialize(cpu_args)
            self.post_model = post_api.TritonPythonModel()
            self.post_model.initialize(cpu_args)
            self.pre = pre_gpu.GpuPreprocessor(self.device.id)
            self.post = post_gpu.GpuPostprocessor(self.device.id, self.post_model, vars(post_api))
            self.input_errors = (InputValidationError, pre_gpu.GpuInputValidationError,
                                 pre_api.InputValidationError, post_api.InputValidationError)

    def _infer(self, images):
        batching.validate_images(images, max_batch_size=self.max_batch_size, imgsz=self.pre_model.imgsz)
        with self.device, self.stream:
            # Keep all local device references alive through synchronization even
            # if a downstream validation or enqueue fails partway through a request.
            try:
                processed, meta = self.pre.preprocess(images, meta_width=self.meta_width,
                    imgsz=self.pre_model.imgsz, padding_value=self.pre_model.padding_value)
                self.metrics.copy("input_h2d", images.nbytes)
                raw = self.runner.infer(processed, self.stream.ptr)
                if self.family == "yolo26s_seg":
                    values = self.post.process(raw["output0"], meta, raw["output1"])
                else:
                    values = self.post.process(raw["output0"], meta)
                outputs = []
                for value in values:
                    output = self.cp.asnumpy(value)
                    self.metrics.copy("output_d2h", output.nbytes)
                    outputs.append(output)
                return tuple(outputs)
            finally:
                self.stream.synchronize()

    def execute(self, requests):
        responses = [None] * len(requests)
        items = []
        for index, request in enumerate(requests):
            try:
                if self._cancelled(request):
                    responses[index] = self._cancelled_response()
                    continue
                tensor = self.pb.get_input_tensor_by_name(request, "RAW_IMAGE")
                if tensor is None:
                    raise InputValidationError("missing required input RAW_IMAGE")
                if not tensor.is_cpu():
                    raise InputValidationError("orin_gpu RAW_IMAGE requires CPU transport")
                images = tensor.as_numpy()
                batching.validate_images(images, max_batch_size=self.max_batch_size,
                    imgsz=getattr(getattr(self, "pre_model", None), "imgsz", 640))
                items.append(batching.RequestImage(index, images))
            except self.input_errors as exc:
                responses[index] = self._error_response(f"INVALID_INPUT: {exc}")
            except Exception:
                responses[index] = self._internal_error([index])
        for planned in batching.plan_groups(items, combine=self.family != "yolov8n_face"):
            group = []
            for item in planned:
                if self._cancelled(requests[item.index]):
                    responses[item.index] = self._cancelled_response()
                else:
                    group.append(item)
            if not group:
                continue
            # Release each group's temporary host storage before packing the
            # next group. No request tensor survives execute's return.
            packed = values = outputs = None
            try:
                start = perf_counter()
                packed, copied = batching.pack_group(group)
                self.metrics.copy("host_pack", copied, perf_counter() - start)
                values = self._infer(packed)
                start = perf_counter()
                copied = 0
                for index, outputs, size in batching.split_outputs(values, group,
                        segmentation=self.family == "yolo26s_seg"):
                    copied += size
                    if self._cancelled(requests[index]):
                        responses[index] = self._cancelled_response()
                    else:
                        responses[index] = self.pb.InferenceResponse(output_tensors=[
                            self.pb.Tensor(name, value) for name, value in zip(self.output_names, outputs)
                        ])
                self.metrics.copy("host_split", copied, perf_counter() - start)
            except self.input_errors as exc:
                self.metrics.group_failure()
                for item in group:
                    responses[item.index] = (self._cancelled_response() if self._cancelled(requests[item.index])
                                            else self._error_response(f"INVALID_INPUT: {exc}"))
            except Exception:
                self.metrics.group_failure()
                incident = uuid.uuid4().hex[:12]
                indices = [item.index for item in group]
                logger.exception("%s Orin GPU failure request_indices=%s incident=%s", self.family, indices, incident)
                for item in group:
                    responses[item.index] = (self._cancelled_response() if self._cancelled(requests[item.index])
                        else self._error_response(f"INTERNAL_ERROR: {self.family}_ensemble failed; incident={incident}"))
            finally:
                packed = values = outputs = None
        return responses

    @staticmethod
    def _cancelled(request):
        check = getattr(request, "is_cancelled", None)
        return bool(check and check())

    def _error_response(self, message):
        return self.pb.InferenceResponse(output_tensors=[], error=self.pb.TritonError(message))

    def _cancelled_response(self):
        return self.pb.InferenceResponse(output_tensors=[],
            error=self.pb.TritonError("request cancelled", self.pb.TritonError.CANCELLED))

    def _internal_error(self, indices):
        incident = uuid.uuid4().hex[:12]
        logger.exception("%s Orin GPU failure request_indices=%s incident=%s", self.family, indices, incident)
        return self._error_response(f"INTERNAL_ERROR: {self.family}_ensemble failed; incident={incident}")

    def finalize(self):
        if hasattr(self, "stream"):
            with self.device:
                self.stream.synchronize()
        self.runner = None
        self.metrics.close()
