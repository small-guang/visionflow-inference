"""CLIP/SBS 同进程 decode → CUDA preprocess → TensorRT → embedding。

只在公开模型的 orin_gpu profile 中启用。预处理 tensor 直接绑定 device
指针；跨 Python backend 边界的输出只剩最终特征，无中间图像 D2H/H2D。
"""
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import NamedTuple
import uuid

import numpy as np

if __package__:
    from .image_orin_preprocess import GpuImagePreprocessor
    from .orin_trt_runtime import create_runner
    from .orin_pipeline_metrics import PipelineMetrics, NullMetrics
else:
    from image_orin_preprocess import GpuImagePreprocessor
    from orin_trt_runtime import create_runner
    from orin_pipeline_metrics import PipelineMetrics, NullMetrics


logger = logging.getLogger("image_orin_pipeline")
SPECS = {
    "clip_ensemble_trt": {"alias": "clip_image", "native": "clip_image_trt",
        "shape": (3, 224, 224), "output": "unnorm_image_features", "width": 512, "version": "1"},
    "sbs_r50_ibn_image_ensemble": {"alias": "sbs", "native": "sbs_r50_ibn_onnx",
        "shape": (3, 384, 128), "output": "embedding", "width": 2048, "version": "2"},
}


class InputValidationError(ValueError):
    pass


class DecodeItem(NamedTuple):
    payload: object
    batch_index: int


def repository_root(argument, public_model):
    root = Path(argument)
    if root.name == public_model:
        root = root.parent
    if not (root / public_model).is_dir():
        raise ValueError(f"model repository does not contain {public_model}: {root}")
    return root


class TritonPythonModel:
    metrics = NullMetrics()

    def initialize(self, args):
        self.public_model = args["model_name"]
        if self.public_model not in SPECS:
            raise ValueError(f"unsupported image pipeline: {self.public_model}")
        self.spec = SPECS[self.public_model]
        config = args.get("model_config", {})
        if isinstance(config, str):
            config = json.loads(config)
        parameters = config.get("parameters", {})
        if parameters.get("pipeline_backend", {}).get("string_value") != "orin_gpu":
            raise ValueError("image pipeline requires pipeline_backend=orin_gpu")
        if args.get("model_instance_kind") != "GPU":
            raise RuntimeError("image pipeline requires a KIND_GPU Python instance")
        if int(config.get("max_batch_size", 0)) != 4:
            raise ValueError("image pipeline requires max_batch_size=4")
        if parameters.get("api_contract_version", {}).get("string_value") != self.spec["version"]:
            raise ValueError("image pipeline API contract version mismatch")
        for key, expected in (("input", ("IMAGE_BYTES", "TYPE_STRING", [1])),
                              ("output", (self.spec["output"], "TYPE_FP32", [self.spec["width"]]))):
            rows = config.get(key, [])
            if len(rows) != 1 or (rows[0].get("name"), rows[0].get("data_type"), rows[0].get("dims")) != expected:
                raise ValueError(f"image pipeline {key} violates the public contract")
        workers = parameters.get("jpeg_decode_workers", {}).get("string_value", "2")
        if workers not in ("1", "2", "3", "4"):
            raise ValueError("jpeg_decode_workers must be an integer between 1 and 4")
        self.jpeg_decode_workers = int(workers)
        import cupy as cp
        import tensorrt as trt
        import triton_python_backend_utils as pb
        if __package__:
            from .image_decode import BoundedImageDecoder
        else:
            from image_decode import BoundedImageDecoder
        self.cp, self.pb = cp, pb
        self.metrics = PipelineMetrics(pb, args, mode=self.spec["alias"])
        self.device = cp.cuda.Device(int(args.get("model_instance_device_id", 0)))
        root = repository_root(args["model_repository"], self.public_model)
        with self.device:
            self.stream = cp.cuda.Stream(non_blocking=True)
            self.pre = GpuImagePreprocessor(self.spec["alias"], self.device.id, cp_module=cp)
            self.runner = create_runner(root, self.spec["native"], str(args.get("model_version", "1")),
                cp, trt, device_id=self.device.id, input_name="image", input_shape=self.spec["shape"],
                output_shapes={self.spec["output"]: (self.spec["width"],)})
        self.decoder = BoundedImageDecoder(self._decode_one, workers=self.jpeg_decode_workers, max_batch_size=4)

    def _decode_one(self, item):
        # Only this CPU OpenCV method executes in decoder workers.
        return self.pre.decode(item.payload, batch_index=item.batch_index)

    def _request_items(self, request):
        tensor = self.pb.get_input_tensor_by_name(request, "IMAGE_BYTES")
        if tensor is None:
            raise InputValidationError("missing required input IMAGE_BYTES")
        if not tensor.is_cpu():
            raise InputValidationError("IMAGE_BYTES requires CPU transport")
        values = tensor.as_numpy()
        if (not isinstance(values, np.ndarray) or values.dtype != np.object_
                or values.ndim != 2 or values.shape[1] != 1 or not 1 <= values.shape[0] <= 4):
            raise InputValidationError("IMAGE_BYTES must be object/BYTES [B,1], B in 1..4")
        return [DecodeItem(values[index, 0], index) for index in range(values.shape[0])]

    def _infer(self, images, request_count):
        processed = output = None
        with self.device, self.stream:
            try:
                start = perf_counter()
                try:
                    # Stable helper owns/synchronizes its preprocessing stream.
                    # The completed device result can then be bound on this stream.
                    processed = self.pre.preprocess_batch(images)
                finally:
                    self.metrics.stage("preprocess", perf_counter() - start)
                self.metrics.copy("input_h2d", sum(image.nbytes for image in images))
                start = perf_counter()
                try:
                    output = self.runner.infer(processed, self.stream.ptr)
                    self.metrics.batch(len(images), request_count)
                    host = self.cp.asnumpy(output[self.spec["output"]], stream=self.stream)
                    self.metrics.copy("output_d2h", host.nbytes)
                    return host
                finally:
                    self.metrics.stage("trt_download", perf_counter() - start)
            finally:
                # Source/device inputs and persistent output views stay alive even
                # if shape binding, enqueue or D2H raises after submitting work.
                self.stream.synchronize()

    def _run_group(self, group, requests, responses):
        group = [(index, items) for index, items in group
                 if not self._set_cancelled(index, requests[index], responses)]
        if not group:
            return
        start = perf_counter()
        try:
            decoded = self.decoder.decode([item for _, items in group for item in items])
        except Exception:
            self._fail_group(group, responses)
            return
        finally:
            self.metrics.stage("decode", perf_counter() - start)
        valid, images, offset = [], [], 0
        for index, items in group:
            slots = decoded[offset:offset + len(items)]
            offset += len(items)
            failures = [slot.error for slot in slots if slot.error is not None]
            failure = next((error for error in failures if not isinstance(error, ValueError)),
                           failures[0] if failures else None)
            if failure is not None:
                if isinstance(failure, ValueError):
                    self.metrics.error("invalid")
                    responses[index] = self._error(f"INVALID_INPUT: {failure}")
                else:
                    self._fail_group([(index, None)], responses, error=failure)
            elif not self._set_cancelled(index, requests[index], responses):
                valid.append((index, len(items)))
                images.extend(slot.value for slot in slots)
        if not valid:
            return
        try:
            values = self._infer(images, len(valid))
        except Exception:
            self._fail_group(valid, responses)
            return
        offset = 0
        for index, size in valid:
            try:
                value = values[offset:offset + size]
                if not self._set_cancelled(index, requests[index], responses):
                    responses[index] = self.pb.InferenceResponse(output_tensors=[
                        self.pb.Tensor(self.spec["output"], value)])
            except Exception:
                self._fail_group([(index, None)], responses)
            finally:
                offset += size

    def execute(self, requests):
        self.metrics.requests(len(requests))
        responses = [None] * len(requests)
        group, samples = [], 0
        for index, request in enumerate(requests):
            try:
                if self._set_cancelled(index, request, responses):
                    continue
                items = self._request_items(request)
            except InputValidationError as exc:
                self.metrics.error("invalid")
                responses[index] = self._error(f"INVALID_INPUT: {exc}")
                continue
            except Exception:
                self._fail_group([(index, None)], responses)
                continue
            if samples + len(items) > 4:
                self._run_group(group, requests, responses)
                group, samples = [], 0
            group.append((index, items))
            samples += len(items)
        if group:
            self._run_group(group, requests, responses)
        return responses

    def _error(self, message):
        return self.pb.InferenceResponse(output_tensors=[], error=self.pb.TritonError(message))

    def _set_cancelled(self, index, request, responses):
        check = getattr(request, "is_cancelled", None)
        if not (check and check()):
            return False
        self.metrics.error("cancelled")
        responses[index] = self.pb.InferenceResponse(output_tensors=[], error=self.pb.TritonError(
            "request cancelled", self.pb.TritonError.CANCELLED))
        return True

    def _fail_group(self, group, responses, *, error=None):
        indices = [index for index, _ in group if responses[index] is None]
        if not indices:
            return
        incident = uuid.uuid4().hex[:12]
        logger.error("%s Orin failure request_indices=%s incident=%s", self.public_model, indices, incident,
                     exc_info=error if error is not None else True)
        self.metrics.error("internal", len(indices))
        for index in indices:
            responses[index] = self._error(f"INTERNAL_ERROR: {self.public_model} failed; incident={incident}")

    def finalize(self):
        if hasattr(self, "decoder"):
            self.decoder.close()
        if hasattr(self, "stream"):
            with self.device:
                self.stream.synchronize()
        self.runner = None
        self.metrics.close()
