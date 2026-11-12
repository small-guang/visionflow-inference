"""Opt-in TBPS CPU decode → CUDA Pillow resize → native TensorRT on Orin.

Only encoded bytes and the final image_feat cross the Python backend boundary.
The default ensemble and every other model retain their existing execution path.
"""
import importlib.util
import json
import logging
from pathlib import Path
import re
import uuid

import numpy as np


logger = logging.getLogger("tbps_orin_pipeline")
PUBLIC_MODEL = "TBPS_clip_image_ensemble"


class InputValidationError(ValueError):
    pass


def repository_root(argument):
    root = Path(argument)
    if root.name == PUBLIC_MODEL:
        root = root.parent
    if not (root / PUBLIC_MODEL).is_dir():
        raise ValueError(f"model repository does not contain {PUBLIC_MODEL}: {root}")
    return root


def plan_path(repository, version="1"):
    directory = Path(repository) / "TBPS_clip_image_trt"
    config = (directory / "config.pbtxt").read_text(encoding="utf-8")
    match = re.search(r'\bdefault_model_filename\s*:\s*"([^"\r\n]+)"', config)
    if not match:
        raise ValueError("TBPS_clip_image_trt has no default_model_filename")
    filename = match.group(1)
    if "/" in filename or "\\" in filename or not filename.endswith(".plan"):
        raise ValueError("TBPS orin_gpu requires a local TensorRT .plan filename")
    if not re.fullmatch(r"[0-9]+", str(version)):
        raise ValueError("model_version must be numeric")
    path = directory / str(version) / filename
    if not path.is_file():
        raise FileNotFoundError(f"TBPS TensorRT plan is missing: {path}")
    return path


def validate_encoded_batch(values):
    if (not isinstance(values, np.ndarray) or values.dtype != np.object_
            or values.ndim != 2 or values.shape[1] != 1 or not 1 <= values.shape[0] <= 4):
        raise InputValidationError("IMAGE_BYTES must be object/BYTES [B,1], B in 1..4")
    payloads = []
    for index, payload in enumerate(values[:, 0]):
        if not isinstance(payload, bytes) or not payload:
            raise InputValidationError(f"image at batch index {index} must be nonempty bytes")
        payloads.append(payload)
    return payloads


class TensorRtRunner:
    """TBPS named device bindings with complete B1–4 profile validation."""
    def __init__(self, engine, runtime, cp, trt, device_id):
        self.engine, self.runtime, self.cp, self.device_id = engine, runtime, cp, device_id
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        if len(names) != 2 or set(names) != {"image", "image_feat"}:
            raise ValueError(f"unexpected TBPS TensorRT tensor names: {names}")
        for name, template in (("image", (-1, 3, 224, 224)), ("image_feat", (-1, 512))):
            mode = trt.TensorIOMode.INPUT if name == "image" else trt.TensorIOMode.OUTPUT
            if (engine.get_tensor_mode(name) != mode
                    or np.dtype(trt.nptype(engine.get_tensor_dtype(name))) != np.dtype(np.float32)
                    or engine.get_tensor_location(name) != trt.TensorLocation.DEVICE
                    or engine.get_tensor_format(name) != trt.TensorFormat.LINEAR
                    or engine.is_shape_inference_io(name)):
                raise ValueError(f"invalid TBPS TensorRT format for {name}")
            if tuple(engine.get_tensor_shape(name)) != template:
                raise ValueError(f"invalid TBPS TensorRT shape for {name}")
        self.profiles = [tuple(tuple(int(value) for value in shape) for shape in
                               engine.get_tensor_profile_shape("image", index))
                         for index in range(engine.num_optimization_profiles)]
        for bounds in self.profiles:
            if (len(bounds) != 3 or any(len(shape) != 4 for shape in bounds)
                    or any(not 1 <= low <= opt <= high for low, opt, high in zip(*bounds))):
                raise ValueError("invalid TBPS TensorRT profile bounds")
        for batch in range(1, 5):
            self.profile_for((batch, 3, 224, 224))
        self.context = engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("failed to create TBPS TensorRT execution context")
        self.active_profile = None
        self.output = cp.empty((4, 512), dtype=np.float32)

    def profile_for(self, shape):
        for index, (minimum, _, maximum) in enumerate(self.profiles):
            if all(low <= actual <= high for low, actual, high in zip(minimum, shape, maximum)):
                return index
        raise ValueError(f"no TBPS TensorRT profile covers {shape}")

    def infer(self, images, stream_handle):
        if (not isinstance(images, self.cp.ndarray) or images.dtype != np.float32
                or not images.flags.c_contiguous or images.device.id != self.device_id
                or len(images.shape) != 4 or images.shape[1:] != (3, 224, 224)
                or not 1 <= images.shape[0] <= 4):
            raise ValueError("TBPS TensorRT input must be contiguous device FP32 [B,3,224,224], B1–4")
        shape = tuple(images.shape)
        profile = self.profile_for(shape)
        if self.active_profile != profile:
            if not self.context.set_optimization_profile_async(profile, stream_handle):
                raise RuntimeError("TBPS TensorRT profile selection failed")
            self.active_profile = profile
        if not self.context.set_input_shape("image", shape):
            raise RuntimeError("TBPS TensorRT input shape rejected")
        if self.context.infer_shapes():
            raise RuntimeError("TBPS TensorRT has unresolved input shapes")
        output_shape = tuple(self.context.get_tensor_shape("image_feat"))
        if output_shape != (shape[0], 512):
            raise RuntimeError(f"invalid TBPS TensorRT output shape: {output_shape}")
        output = self.output[:shape[0]]
        for name, value in (("image", images), ("image_feat", output)):
            if not self.context.set_tensor_address(name, int(value.data.ptr)):
                raise RuntimeError(f"TBPS TensorRT binding failed: {name}")
        if not self.context.execute_async_v3(stream_handle=stream_handle):
            raise RuntimeError("TBPS TensorRT enqueue failed")
        return output


class TritonPythonModel:
    def initialize(self, args):
        if args.get("model_instance_kind") != "GPU":
            raise RuntimeError("TBPS orin_gpu requires a KIND_GPU Python instance")
        config = args.get("model_config", {})
        if isinstance(config, str):
            config = json.loads(config)
        if int(config.get("max_batch_size", 0)) != 4:
            raise ValueError("TBPS orin_gpu requires max_batch_size=4")
        for key, expected in (("input", ("IMAGE_BYTES", "TYPE_STRING", [1])),
                              ("output", ("image_feat", "TYPE_FP32", [512]))):
            rows = config.get(key, [])
            if len(rows) != 1 or (rows[0].get("name"), rows[0].get("data_type"), rows[0].get("dims")) != expected:
                raise ValueError(f"TBPS orin_gpu {key} violates the public contract")
        import cupy as cp
        import tensorrt as trt
        import triton_python_backend_utils as pb
        from yolo_orin_pipeline import validate_tensorrt_version

        self.cp, self.pb = cp, pb
        self.native_library = validate_tensorrt_version(trt)
        root = repository_root(args["model_repository"])
        version = str(args.get("model_version", "1"))
        path = plan_path(root, version)
        helper = root / "TBPS_clip_image_preprocess" / version / "gpu.py"
        spec = importlib.util.spec_from_file_location("tbps_orin_gpu_preprocess", helper)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import TBPS GPU preprocessing: {helper}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.device = cp.cuda.Device(int(args.get("model_instance_device_id", 0)))
        with self.device:
            self.stream = cp.cuda.Stream(non_blocking=True)
            self.trt_logger = trt.Logger(trt.Logger.WARNING)
            if not trt.init_libnvinfer_plugins(self.trt_logger, ""):
                raise RuntimeError("failed to initialize TensorRT plugins")
            runtime = trt.Runtime(self.trt_logger)
            engine = runtime.deserialize_cuda_engine(path.read_bytes())
            if engine is None:
                raise RuntimeError(f"cannot deserialize native TBPS TensorRT plan: {path}")
            self.runner = TensorRtRunner(engine, runtime, cp, trt, self.device.id)
            self.pre = module.TbpsGpuPreprocessor(cp_module=cp)

    def _infer(self, values):
        payloads = validate_encoded_batch(values)
        images, tensors = [], []
        with self.device, self.stream:
            try:
                for index, payload in enumerate(payloads):
                    try:
                        rgb = self.pre.decode(payload, batch_index=index)
                    except ValueError as exc:
                        raise InputValidationError(f"unable to decode image at batch index {index}") from exc
                    images.append(rgb)
                    tensors.append(self.pre.preprocess(rgb))
                tensor = self.cp.stack(tensors)
                output = self.runner.infer(tensor, self.stream.ptr)
                return output.get(stream=self.stream, blocking=True)
            finally:
                # Retain decoded hosts, per-image tensors and bindings through
                # completion, including partial batches and failed enqueues.
                self.stream.synchronize()

    def execute(self, requests):
        responses = []
        for index, request in enumerate(requests):
            try:
                tensor = self.pb.get_input_tensor_by_name(request, "IMAGE_BYTES")
                if tensor is None:
                    raise InputValidationError("missing required input IMAGE_BYTES")
                if not tensor.is_cpu():
                    raise InputValidationError("IMAGE_BYTES requires CPU transport")
                value = self._infer(tensor.as_numpy())
                responses.append(self.pb.InferenceResponse(output_tensors=[self.pb.Tensor("image_feat", value)]))
            except InputValidationError as exc:
                responses.append(self.pb.InferenceResponse(output_tensors=[],
                    error=self.pb.TritonError(f"INVALID_INPUT: {exc}")))
            except Exception:
                incident = uuid.uuid4().hex[:12]
                logger.exception("TBPS Orin GPU failure request_index=%d incident=%s", index, incident)
                responses.append(self.pb.InferenceResponse(output_tensors=[], error=self.pb.TritonError(
                    f"INTERNAL_ERROR: {PUBLIC_MODEL} failed; incident={incident}")))
        return responses

    def finalize(self):
        if hasattr(self, "stream"):
            with self.device:
                self.stream.synchronize()
        self.runner = None

