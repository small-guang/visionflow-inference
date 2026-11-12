"""Isolated image GPU experiment server; never selected by normal startup.

The CPU fused control calls the existing CPU model implementation verbatim.
This module is also the repository generator for disposable evaluation models.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np


MODELS = {
    "clip_image": ("clip_ensemble_trt", "clip_image_preprocess", "clip_image_trt", "unnorm_image_features", 512, (224, 224)),
    "tbps_clip_image": ("TBPS_clip_image_ensemble", "TBPS_clip_image_preprocess", "TBPS_clip_image_trt", "image_feat", 512, (224, 224)),
    "sbs": ("sbs_r50_ibn_image_ensemble", "sbs_r50_ibn_image_preprocess", "sbs_r50_ibn_onnx", "embedding", 2048, (384, 128)),
}
VARIANTS = ("cpu_fused", "gpu_normalize", "gpu_resize")
OPTIONAL_VARIANTS = ("gpu_resize_neon",)


def validate_selection(models, variants):
    models, variants = list(dict.fromkeys(models)), list(dict.fromkeys(variants))
    if not models or any(alias not in MODELS for alias in models):
        raise ValueError(f"Unknown or empty model selection: {models}")
    if not variants or any(variant not in VARIANTS + OPTIONAL_VARIANTS for variant in variants):
        raise ValueError(f"Unknown or empty variant selection: {variants}")
    if "tbps_clip_image" in models and "gpu_resize_neon" in variants:
        raise ValueError("tbps_clip_image does not support gpu_resize_neon")
    return models, variants


def resolve_model(name):
    for alias in MODELS:
        for variant in VARIANTS + OPTIONAL_VARIANTS:
            if name == f"benchmark_{alias}_{variant}":
                validate_selection([alias], [variant])
                return alias, variant
    raise ValueError(f"Unknown evaluation model: {name}")


def choose_profile(profiles, shape):
    for index, (minimum, _, maximum) in enumerate(profiles):
        if len(minimum) == len(maximum) == len(shape) and all(
            low <= actual <= high for low, actual, high in zip(minimum, shape, maximum)
        ):
            return index
    raise ValueError(f"no TensorRT profile covers {shape}")


def validate_encoded_batch(values):
    if (values.dtype != np.object_ or values.ndim != 2 or values.shape[1] != 1
            or not 1 <= values.shape[0] <= 4):
        raise ValueError("IMAGE_BYTES must be object/BYTES [B,1], B in 1..4")
    result = []
    for index, value in enumerate(values[:, 0]):
        if not isinstance(value, bytes) or not value:
            raise ValueError(f"image at batch index {index} must be nonempty bytes")
        result.append(value)
    return result


def load_cpu_preprocessor(root, alias):
    directory = MODELS[alias][1]
    path = Path(root) / directory / "1" / "model.py"
    spec = importlib.util.spec_from_file_location(f"eval_cpu_{alias}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.TritonPythonModel()
    model.initialize({"model_config": json.dumps({"output": [{"name": "image", "data_type": "TYPE_FP32"}]})})
    return model._preprocess_image if alias == "tbps_clip_image" else model._preprocess


class ImageGpuEvalModel:
    """One disposable Python backend process owns its native TRT context."""
    def initialize(self, args):
        name = args["model_name"]
        self.alias, self.variant = resolve_model(name)
        import cupy as cp
        import tensorrt as trt
        import triton_python_backend_utils as pb
        from image_gpu_candidate import ImageGpuPreprocessor
        from yolo_orin_pipeline import validate_tensorrt_version

        if args.get("model_instance_kind") != "GPU":
            raise ValueError("evaluation requires KIND_GPU")
        self.cp, self.trt, self.pb = cp, trt, pb
        self.native_library = validate_tensorrt_version(trt)
        self.device = cp.cuda.Device(int(args.get("model_instance_device_id", 0)))
        root = Path(args["model_repository"])
        if root.name == name:
            root = root.parent
        _, _, native, self.output_name, self.width, self.hw = MODELS[self.alias]
        self.cpu_pre = load_cpu_preprocessor(root, self.alias)
        with self.device:
            self.stream = cp.cuda.Stream(non_blocking=True)
            if self.variant == "gpu_resize_neon":
                self.pre = ImageGpuPreprocessor(self.alias, cp_module=cp, opencv_rounding="neon")
            else:
                self.pre = ImageGpuPreprocessor(self.alias, cp_module=cp)
            self.logger = trt.Logger(trt.Logger.WARNING)
            if not trt.init_libnvinfer_plugins(self.logger, ""):
                raise RuntimeError("cannot initialize TensorRT plugins")
            self.runtime = trt.Runtime(self.logger)
            config = (root / native / "config.pbtxt").read_text()
            filename = re.search(r'default_model_filename\s*:\s*"([^"/\\]+\.plan)"', config).group(1)
            plan = root / native / "1" / filename
            self.engine = self.runtime.deserialize_cuda_engine(plan.read_bytes())
            if self.engine is None:
                raise RuntimeError(f"cannot deserialize {plan}")
            names = {self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)}
            if names != {"image", self.output_name}:
                raise ValueError(f"unexpected plan I/O {names}")
            for tensor_name in names:
                mode = trt.TensorIOMode.INPUT if tensor_name == "image" else trt.TensorIOMode.OUTPUT
                if (self.engine.get_tensor_mode(tensor_name) != mode
                    or np.dtype(trt.nptype(self.engine.get_tensor_dtype(tensor_name))) != np.dtype("float32")
                    or self.engine.get_tensor_location(tensor_name) != trt.TensorLocation.DEVICE
                    or self.engine.get_tensor_format(tensor_name) != trt.TensorFormat.LINEAR):
                    raise ValueError(f"unexpected plan format for {tensor_name}")
            self.profiles = [self.engine.get_tensor_profile_shape("image", i)
                             for i in range(self.engine.num_optimization_profiles)]
            for batch in range(1, 5):
                choose_profile(self.profiles, (batch, 3, *self.hw))
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("cannot create TensorRT context")
            self.active_profile = None
            self.output = cp.empty((4, self.width), dtype=cp.float32)

    def infer_device(self, tensor):
        shape = tuple(tensor.shape)
        if tensor.dtype != self.cp.float32 or not tensor.flags.c_contiguous or shape[1:] != (3, *self.hw):
            raise ValueError(f"bad GPU tensor {shape}")
        profile = choose_profile(self.profiles, shape)
        if self.active_profile != profile:
            if not self.context.set_optimization_profile_async(profile, self.stream.ptr):
                raise RuntimeError("profile selection failed")
            self.active_profile = profile
        if not self.context.set_input_shape("image", shape):
            raise RuntimeError("TensorRT rejected input shape")
        if self.context.infer_shapes():
            raise RuntimeError("unresolved TensorRT input shapes")
        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        if output_shape != (shape[0], self.width):
            raise ValueError(f"unexpected output shape {output_shape}")
        if not self.context.set_tensor_address("image", int(tensor.data.ptr)):
            raise RuntimeError("input binding failed")
        output = self.output[:shape[0]]
        if not self.context.set_tensor_address(self.output_name, int(output.data.ptr)):
            raise RuntimeError("output binding failed")
        if not self.context.execute_async_v3(stream_handle=self.stream.ptr):
            raise RuntimeError("TensorRT enqueue failed")
        return output.get(stream=self.stream, blocking=True)

    def execute(self, requests):
        results = []
        for request in requests:
            retained = []
            try:
                input_tensor = self.pb.get_input_tensor_by_name(request, "IMAGE_BYTES")
                if input_tensor is None:
                    raise ValueError("IMAGE_BYTES is required")
                payloads = validate_encoded_batch(input_tensor.as_numpy())
                with self.device, self.stream:
                    if self.variant == "cpu_fused":
                        cpu = np.stack([self.cpu_pre(payload, i) for i, payload in enumerate(payloads)])
                        tensor = self.cp.asarray(cpu)
                        retained.append(cpu)
                    else:
                        mode = "normalize" if self.variant == "gpu_normalize" else "resize"
                        images = [self.pre.decode(payload) for payload in payloads]
                        retained.extend(images)
                        tensors = [self.pre.gpu_tensor(image, mode=mode) for image in images]
                        retained.extend(tensors)
                        tensor = self.cp.stack(tensors)
                    retained.append(tensor)
                    value = self.infer_device(tensor)
                results.append(self.pb.InferenceResponse(output_tensors=[self.pb.Tensor(self.output_name, value)]))
            except Exception as exc:
                with self.device:
                    self.stream.synchronize()
                results.append(self.pb.InferenceResponse(error=self.pb.TritonError(str(exc))))
        return results

    def finalize(self):
        with self.device:
            self.stream.synchronize()


def prepare_repository(source, output, native_root, *, models=None, variants=None):
    """Copy source-only baselines and symlink the same read-only native plans."""
    models, variants = validate_selection(MODELS if models is None else models,
                                         VARIANTS if variants is None else variants)
    source, output, native_root = Path(source), Path(output), Path(native_root)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("evaluation repository must be empty")
    manifest = {"models": {}, "source_hashes": {}}
    for alias in models:
        public, pre, native, name, width, hw = MODELS[alias]
        for directory in (public, pre, native):
            shutil.copytree(source / "model_repository" / directory, output / directory,
                            ignore=shutil.ignore_patterns("*.plan", "*.onnx", "__pycache__"))
        # Source-only archives omit empty directories, but Triton ensembles
        # still require a numeric version directory even without a model file.
        (output / public / "1").mkdir(exist_ok=True)
        plan_files = sorted((native_root / native / "1").glob("*.plan"))
        if not plan_files:
            raise FileNotFoundError(f"no native plans for {native}")
        target_version = output / native / "1"
        target_version.mkdir(exist_ok=True)
        for plan in plan_files:
            (target_version / plan.name).symlink_to(plan)
        selected_config = (output / native / "config.pbtxt").read_text()
        selected_name = re.search(r'default_model_filename\s*:\s*"([^"/\\]+\.plan)"', selected_config).group(1)
        plan = target_version / selected_name
        manifest["models"][alias] = {"plan": str(plan.resolve()), "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest()}
        for variant in variants:
            model_name = f"benchmark_{alias}_{variant}"
            directory = output / model_name
            (directory / "1").mkdir(parents=True)
            (directory / "1" / "model.py").write_text(
                "from image_gpu_eval_server import ImageGpuEvalModel as TritonPythonModel\n")
            (directory / "config.pbtxt").write_text(f'''name: "{model_name}"
backend: "python"
max_batch_size: 4
parameters {{ key: "FORCE_CPU_ONLY_INPUT_TENSORS" value {{ string_value: "yes" }} }}
input [ {{ name: "IMAGE_BYTES" data_type: TYPE_STRING dims: [1] }} ]
output [ {{ name: "{name}" data_type: TYPE_FP32 dims: [{width}] }} ]
instance_group [ {{ count: 1 kind: KIND_GPU }} ]
dynamic_batching {{ max_queue_delay_microseconds: 2000 }}
''')
    for directory in ("tools", "scripts", "model_repository"):
        for path in sorted((source / directory).rglob("*.py")):
            if "__pycache__" not in path.parts:
                manifest["source_hashes"][path.relative_to(source).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output.parent / "provenance.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS + OPTIONAL_VARIANTS)
    options = parser.parse_args()
    try:
        prepare_repository(options.source, options.output, options.native_root,
                           models=options.models, variants=options.variants)
    except ValueError as exc:
        parser.error(str(exc))
