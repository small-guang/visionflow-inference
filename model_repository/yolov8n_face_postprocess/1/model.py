import logging
import uuid

import numpy as np
import triton_python_backend_utils as pb_utils


MAX_SIDE = 8192
MAX_PIXELS = 8_388_608
MAX_PREDICTIONS = 100_000
MAX_RAW_VALUES = 500_000
LETTERBOX_SIZE = 640
LETTERBOX_BUCKET_SIZES = (192, 384, 480, 640)
META_RTOL = 1e-6
META_ATOL = 1e-5
logger = logging.getLogger("yolov8n_face_postprocess")


class InputValidationError(ValueError):
    pass


def xywh_to_xyxy(boxes):
    xyxy = np.empty_like(boxes, dtype=np.float32)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return xyxy


def nms(boxes, scores, iou_threshold, max_detections):
    if boxes.size == 0:
        return []

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < max_detections:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[index], x1[rest])
        yy1 = np.maximum(y1[index], y1[rest])
        xx2 = np.minimum(x2[index], x2[rest])
        yy2 = np.minimum(y2[index], y2[rest])
        intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(
            0.0, yy2 - yy1
        )
        union = areas[index] + areas[rest] - intersection
        iou = intersection / np.maximum(union, 1e-7)
        order = rest[iou <= iou_threshold]
    return keep


def _validate_meta(meta):
    if not isinstance(meta, np.ndarray) or meta.shape != (5,):
        raise InputValidationError("LETTERBOX_META sample must have shape [5]")
    if meta.dtype != np.float32:
        raise InputValidationError("LETTERBOX_META must have dtype float32")
    if not np.isfinite(meta).all():
        raise InputValidationError("LETTERBOX_META values must be finite")
    orig_h, orig_w, ratio, left, top = (float(value) for value in meta)
    if orig_h <= 0 or orig_w <= 0:
        raise InputValidationError("LETTERBOX_META original dimensions must be positive")
    if orig_h > MAX_SIDE or orig_w > MAX_SIDE:
        raise InputValidationError(
            f"LETTERBOX_META original dimensions exceed max side {MAX_SIDE}"
        )
    if orig_h * orig_w > MAX_PIXELS:
        raise InputValidationError(
            f"LETTERBOX_META original image exceeds max pixels {MAX_PIXELS}"
        )
    rounded_h = round(orig_h)
    rounded_w = round(orig_w)
    if not np.isclose(orig_h, rounded_h, rtol=0.0, atol=META_ATOL) or not np.isclose(
        orig_w, rounded_w, rtol=0.0, atol=META_ATOL
    ):
        raise InputValidationError(
            "LETTERBOX_META original dimensions must be integer-valued"
        )
    original_h = int(rounded_h)
    original_w = int(rounded_w)
    if original_h < 1 or original_w < 1:
        raise InputValidationError(
            "LETTERBOX_META original dimensions must be at least 1 pixel"
        )
    expected_ratio = min(
        LETTERBOX_SIZE / original_h, LETTERBOX_SIZE / original_w
    )
    resized_w = round(original_w * expected_ratio)
    resized_h = round(original_h * expected_ratio)
    if resized_w < 1 or resized_h < 1:
        raise InputValidationError(
            "LETTERBOX_META aspect ratio produces a resize dimension "
            "smaller than 1 pixel"
        )
    target_h = next(size for size in LETTERBOX_BUCKET_SIZES if size >= resized_h)
    target_w = next(size for size in LETTERBOX_BUCKET_SIZES if size >= resized_w)
    pad_w = (target_w - resized_w) / 2
    pad_h = (target_h - resized_h) / 2
    expected_left = round(pad_w - 0.1)
    expected_top = round(pad_h - 0.1)
    if not np.isclose(
        ratio, expected_ratio, rtol=META_RTOL, atol=META_ATOL
    ):
        raise InputValidationError(
            "LETTERBOX_META ratio is inconsistent with original dimensions"
        )
    if not np.isclose(left, expected_left, rtol=0.0, atol=META_ATOL):
        raise InputValidationError(
            "LETTERBOX_META left padding is inconsistent with letterbox geometry"
        )
    if not np.isclose(top, expected_top, rtol=0.0, atol=META_ATOL):
        raise InputValidationError(
            "LETTERBOX_META top padding is inconsistent with letterbox geometry"
        )


def _validate_batch_inputs(raw_output, meta):
    if not isinstance(raw_output, np.ndarray):
        raise InputValidationError("RAW_OUTPUT must be a numpy array")
    if (
        raw_output.ndim != 3
        or raw_output.shape[1] != 5
        or raw_output.shape[2] == 5
    ):
        raise InputValidationError(
            "RAW_OUTPUT batch must have unambiguous shape [B,5,N]"
        )
    if raw_output.dtype != np.float32:
        raise InputValidationError("RAW_OUTPUT must have dtype float32")
    if raw_output.shape[0] == 0:
        raise InputValidationError("input batch must be non-empty")
    if not isinstance(meta, np.ndarray):
        raise InputValidationError("LETTERBOX_META must be a numpy array")
    if meta.ndim != 2 or meta.shape[1] != 5:
        raise InputValidationError("LETTERBOX_META batch must have shape [B,5]")
    if meta.dtype != np.float32:
        raise InputValidationError("LETTERBOX_META must have dtype float32")
    if raw_output.shape[0] != meta.shape[0]:
        raise InputValidationError(
            "RAW_OUTPUT and LETTERBOX_META batch size must match"
        )


class TritonPythonModel:
    def initialize(self, args):
        self.conf_threshold = 0.5
        self.iou_threshold = 0.45
        self.max_detections = 100
        self._gpu = None
        if (args or {}).get("model_instance_kind") == "GPU":
            import importlib.util
            import json
            from pathlib import Path

            gpu_config = args.get("model_config") or {}
            if isinstance(gpu_config, str):
                gpu_config = json.loads(gpu_config)
            host_outputs = gpu_config.get("parameters", {}).get(
                "GPU_OUTPUT_ON_CPU", {}
            ).get("string_value") == "yes"
            spec = importlib.util.spec_from_file_location(
                __name__ + "_gpu", Path(__file__).with_name("gpu.py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._gpu = module.GpuPostprocessor(
                int(args.get("model_instance_device_id", "0")), self, globals(),
                host_outputs=host_outputs,
            )

    def _prepare_predictions(self, raw_output):
        if not isinstance(raw_output, np.ndarray):
            raise InputValidationError("RAW_OUTPUT sample must be a numpy array")
        if raw_output.dtype != np.float32:
            raise InputValidationError("RAW_OUTPUT must have dtype float32")
        if (
            raw_output.ndim != 2
            or raw_output.shape[0] != 5
            or raw_output.shape[1] == 5
        ):
            raise InputValidationError(
                "RAW_OUTPUT sample must have unambiguous shape [5,N]"
            )
        pred = raw_output.T
        if pred.shape[0] > MAX_PREDICTIONS:
            raise InputValidationError(
                f"RAW_OUTPUT exceeds max predictions {MAX_PREDICTIONS}"
            )
        if raw_output.size > MAX_RAW_VALUES:
            raise InputValidationError(
                f"RAW_OUTPUT exceeds max total values {MAX_RAW_VALUES}"
            )
        if not np.isfinite(pred).all():
            raise InputValidationError("RAW_OUTPUT values must be finite")
        return pred

    def _postprocess(self, raw_output, meta):
        _validate_meta(meta)
        detections = np.zeros((self.max_detections, 6), dtype=np.float32)
        num_detections = np.array([0], dtype=np.int32)
        pred = self._prepare_predictions(raw_output)
        if pred.size == 0:
            return detections, num_detections

        if pred.shape[1] == 5:
            scores = pred[:, 4]
            classes = np.zeros(pred.shape[0], dtype=np.float32)
        else:
            class_scores = pred[:, 4:]
            class_ids = np.argmax(class_scores, axis=1)
            scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
            classes = class_ids.astype(np.float32)

        keep_mask = scores >= self.conf_threshold
        if not np.any(keep_mask):
            return detections, num_detections
        boxes = xywh_to_xyxy(pred[keep_mask, :4])
        scores = scores[keep_mask].astype(np.float32)
        classes = classes[keep_mask].astype(np.float32)

        orig_h, orig_w, ratio, left, top = meta
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, scores, classes = boxes[valid], scores[valid], classes[valid]
        if boxes.size == 0:
            return detections, num_detections

        keep = nms(boxes, scores, self.iou_threshold, self.max_detections)
        count = min(len(keep), self.max_detections)
        if count:
            keep = keep[:count]
            detections[:count, :4] = boxes[keep]
            detections[:count, 4] = scores[keep]
            detections[:count, 5] = classes[keep]
            num_detections[0] = count
        return detections, num_detections

    def execute(self, requests):
        responses = []
        for request_index, request in enumerate(requests):
            try:
                if getattr(self, "_gpu", None) is not None:
                    responses.append(self._gpu.execute(request, pb_utils, request_index))
                    continue
                raw_tensor = pb_utils.get_input_tensor_by_name(request, "RAW_OUTPUT")
                meta_tensor = pb_utils.get_input_tensor_by_name(
                    request, "LETTERBOX_META"
                )
                if raw_tensor is None:
                    raise InputValidationError("missing required input RAW_OUTPUT")
                if meta_tensor is None:
                    raise InputValidationError(
                        "missing required input LETTERBOX_META"
                    )
                raw_output = raw_tensor.as_numpy()
                meta = meta_tensor.as_numpy()
                _validate_batch_inputs(raw_output, meta)

                batch_detections = []
                batch_counts = []
                for batch_index, (prediction, sample_meta) in enumerate(
                    zip(raw_output, meta)
                ):
                    try:
                        detections, count = self._postprocess(
                            prediction, sample_meta
                        )
                        batch_detections.append(detections)
                        batch_counts.append(count)
                    except InputValidationError as exc:
                        raise InputValidationError(
                            f"batch index {batch_index}: {exc}"
                        ) from exc
                    except Exception:
                        incident = uuid.uuid4().hex[:12]
                        logger.exception(
                            "yolov8n_face_postprocess internal failure "
                            "request_index=%d batch_index=%d incident=%s",
                            request_index,
                            batch_index,
                            incident,
                        )
                        responses.append(
                            pb_utils.InferenceResponse(
                                output_tensors=[],
                                error=pb_utils.TritonError(
                                    "INTERNAL_ERROR: yolov8n_face_postprocess "
                                    f"failed; incident={incident}"
                                ),
                            )
                        )
                        break
                else:
                    detections = np.ascontiguousarray(
                        np.stack(batch_detections), dtype=np.float32
                    )
                    counts = np.ascontiguousarray(
                        np.stack(batch_counts), dtype=np.int32
                    )
                    responses.append(
                        pb_utils.InferenceResponse(
                            output_tensors=[
                                pb_utils.Tensor("DETECTIONS", detections),
                                pb_utils.Tensor("NUM_DETECTIONS", counts),
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
                    "yolov8n_face_postprocess internal failure "
                    "request_index=%d incident=%s",
                    request_index,
                    incident,
                )
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(
                            "INTERNAL_ERROR: yolov8n_face_postprocess failed; "
                            f"incident={incident}"
                        ),
                    )
                )
        return responses

    def finalize(self):
        print("yolov8n_face_postprocess finalized.", flush=True)
