import importlib.util
import json
import logging
from pathlib import Path
import uuid

import cv2
import numpy as np
import triton_python_backend_utils as pb_utils


MAX_SIDE = 8192
MAX_PIXELS = 8_388_608
LETTERBOX_BUCKET_SIZES = (192, 384, 480, 640)
logger = logging.getLogger("yolo26m_pose_preprocess")


class InputValidationError(ValueError):
    pass


def _validate_batch(images):
    if not isinstance(images, np.ndarray):
        raise InputValidationError("RAW_IMAGE must be a numpy array")
    if images.ndim != 4 or images.shape[3] != 3:
        raise InputValidationError("RAW_IMAGE batch must have shape [B,H,W,3]")
    if images.dtype != np.uint8:
        raise InputValidationError("RAW_IMAGE must have dtype uint8")
    if images.shape[0] == 0:
        raise InputValidationError("RAW_IMAGE batch must be non-empty")


class TritonPythonModel:
    def initialize(self, args):
        self.imgsz = LETTERBOX_BUCKET_SIZES[-1]
        self.padding_value = 114
        self._gpu = None
        if args.get("model_instance_kind") == "GPU":
            path = Path(__file__).with_name("gpu.py")
            spec = importlib.util.spec_from_file_location("yolo26m_pose_gpu_preprocess", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._gpu_validation_error = module.GpuInputValidationError
            config = json.loads(args.get("model_config", "{}"))
            host_outputs = config.get("parameters", {}).get("GPU_OUTPUT_ON_CPU", {}).get("string_value") == "yes"
            self._gpu = module.GpuPreprocessor(args["model_instance_device_id"], host_outputs=host_outputs)

    def _letterbox(self, image):
        if not isinstance(image, np.ndarray):
            raise InputValidationError("RAW_IMAGE sample must be a numpy array")
        if image.dtype != np.uint8:
            raise InputValidationError("RAW_IMAGE must have dtype uint8")
        if image.ndim != 3 or image.shape[2] != 3:
            raise InputValidationError("RAW_IMAGE sample must have shape [H,W,3]")
        orig_h, orig_w = image.shape[:2]
        if orig_h <= 0 or orig_w <= 0:
            raise InputValidationError("RAW_IMAGE height and width must be positive")
        if orig_h > MAX_SIDE or orig_w > MAX_SIDE:
            raise InputValidationError(f"RAW_IMAGE exceeds max side {MAX_SIDE}")
        if orig_h * orig_w > MAX_PIXELS:
            raise InputValidationError(f"RAW_IMAGE exceeds max pixels {MAX_PIXELS}")

        ratio = min(self.imgsz / orig_h, self.imgsz / orig_w)
        resized_w = round(orig_w * ratio)
        resized_h = round(orig_h * ratio)
        if resized_w < 1 or resized_h < 1:
            raise InputValidationError(
                "RAW_IMAGE aspect ratio produces a resize dimension "
                "smaller than 1 pixel"
            )
        # Keep the original content scale and pad to the smallest fixed bucket.
        target_h = next(size for size in LETTERBOX_BUCKET_SIZES if size >= resized_h)
        target_w = next(size for size in LETTERBOX_BUCKET_SIZES if size >= resized_w)
        pad_w = (target_w - resized_w) / 2
        pad_h = (target_h - resized_h) / 2
        if (orig_w, orig_h) != (resized_w, resized_h):
            image = cv2.resize(
                image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
            )

        top = round(pad_h - 0.1)
        bottom = round(pad_h + 0.1)
        left = round(pad_w - 0.1)
        right = round(pad_w + 0.1)
        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(self.padding_value,) * 3,
        )
        input_h, input_w = image.shape[:2]
        meta = np.array(
            [orig_h, orig_w, ratio, left, top, input_h, input_w],
            dtype=np.float32,
        )
        return image, meta

    @staticmethod
    def _to_model_input(image):
        image = image[..., ::-1].transpose(2, 0, 1)
        return np.ascontiguousarray(image, dtype=np.float32) / np.float32(255.0)

    def execute(self, requests):
        responses = []
        for request_index, request in enumerate(requests):
            try:
                tensor = pb_utils.get_input_tensor_by_name(request, "RAW_IMAGE")
                if tensor is None:
                    raise InputValidationError("missing required input RAW_IMAGE")
                if self._gpu is not None:
                    try:
                        images = self._gpu.from_tensor(tensor)
                        output, meta_output = self._gpu.preprocess(
                            images, meta_width=7, imgsz=self.imgsz,
                            padding_value=self.padding_value,
                        )
                    except self._gpu_validation_error as exc:
                        raise InputValidationError(str(exc)) from exc
                    output_tensor = (
                        pb_utils.Tensor("PREPROCESSED_IMAGE", self._gpu.to_host(output))
                        if self._gpu.host_outputs else
                        pb_utils.Tensor.from_dlpack("PREPROCESSED_IMAGE", output)
                    )
                    responses.append(pb_utils.InferenceResponse(output_tensors=[
                        output_tensor,
                        pb_utils.Tensor("LETTERBOX_META", meta_output),
                    ]))
                    continue
                images = tensor.as_numpy()
                _validate_batch(images)

                processed = []
                metas = []
                for batch_index, image in enumerate(images):
                    try:
                        letterboxed, meta = self._letterbox(image)
                        processed.append(self._to_model_input(letterboxed))
                        metas.append(meta)
                    except InputValidationError as exc:
                        raise InputValidationError(
                            f"batch index {batch_index}: {exc}"
                        ) from exc
                    except Exception:
                        incident = uuid.uuid4().hex[:12]
                        logger.exception(
                            "yolo26m_pose_preprocess internal failure "
                            "request_index=%d batch_index=%d incident=%s",
                            request_index,
                            batch_index,
                            incident,
                        )
                        responses.append(
                            pb_utils.InferenceResponse(
                                output_tensors=[],
                                error=pb_utils.TritonError(
                                    "INTERNAL_ERROR: yolo26m_pose_preprocess "
                                    f"failed; incident={incident}"
                                ),
                            )
                        )
                        break
                else:
                    output = np.ascontiguousarray(
                        np.stack(processed, axis=0), dtype=np.float32
                    )
                    meta_output = np.ascontiguousarray(
                        np.stack(metas, axis=0), dtype=np.float32
                    )
                    responses.append(
                        pb_utils.InferenceResponse(
                            output_tensors=[
                                pb_utils.Tensor("PREPROCESSED_IMAGE", output),
                                pb_utils.Tensor("LETTERBOX_META", meta_output),
                            ]
                        )
                    )
                    continue
                continue
            except InputValidationError as exc:
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(f"INVALID_INPUT: {exc}"),
                    )
                )
            except Exception:
                incident = uuid.uuid4().hex[:12]
                logger.exception(
                    "yolo26m_pose_preprocess internal failure "
                    "request_index=%d incident=%s",
                    request_index,
                    incident,
                )
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(
                            "INTERNAL_ERROR: yolo26m_pose_preprocess failed; "
                            f"incident={incident}"
                        ),
                    )
                )
        return responses

    def finalize(self):
        print("yolo26m_pose_preprocess finalized.", flush=True)
