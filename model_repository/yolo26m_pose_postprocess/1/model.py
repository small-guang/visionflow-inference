import logging
import uuid

import numpy as np
import triton_python_backend_utils as pb_utils


MAX_SIDE = 8192
MAX_PIXELS = 8_388_608
MAX_PREDICTIONS = 100_000
LETTERBOX_SIZE = 640
LETTERBOX_BUCKET_SIZES = (192, 384, 480, 640)
LETTERBOX_STRIDE = 32
META_RTOL = 1e-6
META_ATOL = 1e-5
logger = logging.getLogger("yolo26m_pose_postprocess")


class InputValidationError(ValueError):
    pass


def _validate_meta(meta):
    if not isinstance(meta, np.ndarray) or meta.shape != (7,):
        raise InputValidationError("LETTERBOX_META sample must have shape [7]")
    if meta.dtype != np.float32:
        raise InputValidationError("LETTERBOX_META must have dtype float32")
    if not np.isfinite(meta).all():
        raise InputValidationError("LETTERBOX_META values must be finite")
    orig_h, orig_w, ratio, left, top, input_h, input_w = (
        float(value) for value in meta
    )
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
    half_pad_w = (target_w - resized_w) / 2
    half_pad_h = (target_h - resized_h) / 2
    expected_left = round(half_pad_w - 0.1)
    expected_right = round(half_pad_w + 0.1)
    expected_top = round(half_pad_h - 0.1)
    expected_bottom = round(half_pad_h + 0.1)
    expected_input_h = resized_h + expected_top + expected_bottom
    expected_input_w = resized_w + expected_left + expected_right
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
    if not np.isclose(
        input_h, expected_input_h, rtol=0.0, atol=META_ATOL
    ) or not np.isclose(
        input_w, expected_input_w, rtol=0.0, atol=META_ATOL
    ):
        raise InputValidationError(
            "LETTERBOX_META input dimensions are inconsistent with letterbox geometry"
        )
    if (
        expected_input_h > LETTERBOX_SIZE
        or expected_input_w > LETTERBOX_SIZE
        or expected_input_h % LETTERBOX_STRIDE != 0
        or expected_input_w % LETTERBOX_STRIDE != 0
    ):
        raise InputValidationError(
            "LETTERBOX_META reconstructed input dimensions are unsafe"
        )


def _validate_batch_inputs(raw_output, meta):
    if not isinstance(raw_output, np.ndarray):
        raise InputValidationError("RAW_OUTPUT must be a numpy array")
    if (
        raw_output.ndim != 3
        or raw_output.shape[2] != 57
    ):
        raise InputValidationError(
            "RAW_OUTPUT batch must have shape [B,N,57]"
        )
    if raw_output.dtype != np.float32:
        raise InputValidationError("RAW_OUTPUT must have dtype float32")
    if raw_output.shape[0] == 0:
        raise InputValidationError("input batch must be non-empty")
    if not isinstance(meta, np.ndarray):
        raise InputValidationError("LETTERBOX_META must be a numpy array")
    if meta.ndim != 2 or meta.shape[1] != 7:
        raise InputValidationError("LETTERBOX_META batch must have shape [B,7]")
    if meta.dtype != np.float32:
        raise InputValidationError("LETTERBOX_META must have dtype float32")
    if raw_output.shape[0] != meta.shape[0]:
        raise InputValidationError(
            "RAW_OUTPUT and LETTERBOX_META batch size must match"
        )


class TritonPythonModel:
    def initialize(self, args):
        self.conf_threshold = 0.5
        self.max_detections = 100
        self.num_keypoints = 17
        self.keypoint_dims = 3
        self.keypoint_values = self.num_keypoints * self.keypoint_dims
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
        expected_width = 6 + self.keypoint_values
        if (
            raw_output.ndim != 2
            or raw_output.shape[1] != expected_width
        ):
            raise InputValidationError(
                "YOLO pose output must use the end-to-end layout "
                f"[N, {expected_width}], got {raw_output.shape}"
            )
        if raw_output.shape[0] > MAX_PREDICTIONS:
            raise InputValidationError(
                f"RAW_OUTPUT exceeds max predictions {MAX_PREDICTIONS}"
            )
        if not np.isfinite(raw_output).all():
            raise InputValidationError("RAW_OUTPUT values must be finite")
        return raw_output

    def _decode_predictions(self, pred):
        boxes = pred[:, :4]
        scores = pred[:, 4]
        classes = np.round(pred[:, 5]).astype(np.float32)
        keypoints = pred[:, 6 : 6 + self.keypoint_values]
        return boxes, scores, classes, keypoints

    def _empty_outputs(self):
        detections = np.zeros((self.max_detections, 6), dtype=np.float32)
        keypoints = np.zeros(
            (self.max_detections, self.num_keypoints, self.keypoint_dims),
            dtype=np.float32,
        )
        num_detections = np.array([0], dtype=np.int32)
        return detections, keypoints, num_detections

    def _scale_boxes_to_original(self, boxes, meta):
        orig_h, orig_w, ratio, left, top = meta[:5]
        scaled = boxes.astype(np.float32, copy=True)
        scaled[:, [0, 2]] = (scaled[:, [0, 2]] - left) / ratio
        scaled[:, [1, 3]] = (scaled[:, [1, 3]] - top) / ratio
        scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]], 0, orig_w)
        scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]], 0, orig_h)
        return scaled

    def _scale_keypoints_to_original(self, keypoints, meta):
        orig_h, orig_w, ratio, left, top = meta[:5]
        scaled = keypoints.reshape(
            -1, self.num_keypoints, self.keypoint_dims
        ).astype(np.float32, copy=True)
        scaled[:, :, 0] = (scaled[:, :, 0] - left) / ratio
        scaled[:, :, 1] = (scaled[:, :, 1] - top) / ratio
        scaled[:, :, 0] = np.clip(scaled[:, :, 0], 0, orig_w)
        scaled[:, :, 1] = np.clip(scaled[:, :, 1], 0, orig_h)
        return scaled

    def _postprocess(self, raw_output, meta):
        _validate_meta(meta)
        detections, output_keypoints, num_detections = self._empty_outputs()
        pred = self._prepare_predictions(raw_output)
        if pred.size == 0:
            return detections, output_keypoints, num_detections

        boxes, scores, classes, keypoints = self._decode_predictions(pred)
        keep_mask = scores >= self.conf_threshold
        if not np.any(keep_mask):
            return detections, output_keypoints, num_detections
        boxes = boxes[keep_mask].astype(np.float32)
        scores = scores[keep_mask].astype(np.float32)
        classes = classes[keep_mask].astype(np.float32)
        keypoints = keypoints[keep_mask].astype(np.float32)

        boxes = self._scale_boxes_to_original(boxes, meta)
        keypoints = self._scale_keypoints_to_original(keypoints, meta)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, scores, classes, keypoints = (
            boxes[valid],
            scores[valid],
            classes[valid],
            keypoints[valid],
        )
        if boxes.size == 0:
            return detections, output_keypoints, num_detections

        keep = np.argsort(-scores, kind="stable")[: self.max_detections]
        count = keep.size
        if count:
            detections[:count, :4] = boxes[keep]
            detections[:count, 4] = scores[keep]
            detections[:count, 5] = classes[keep]
            output_keypoints[:count] = keypoints[keep]
            num_detections[0] = count
        return detections, output_keypoints, num_detections

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
                batch_keypoints = []
                batch_counts = []
                for batch_index, (prediction, sample_meta) in enumerate(
                    zip(raw_output, meta)
                ):
                    try:
                        detections, keypoints, count = self._postprocess(
                            prediction, sample_meta
                        )
                        batch_detections.append(detections)
                        batch_keypoints.append(keypoints)
                        batch_counts.append(count)
                    except InputValidationError as exc:
                        raise InputValidationError(
                            f"batch index {batch_index}: {exc}"
                        ) from exc
                    except Exception:
                        incident = uuid.uuid4().hex[:12]
                        logger.exception(
                            "yolo26m_pose_postprocess internal failure "
                            "request_index=%d batch_index=%d incident=%s",
                            request_index,
                            batch_index,
                            incident,
                        )
                        responses.append(
                            pb_utils.InferenceResponse(
                                output_tensors=[],
                                error=pb_utils.TritonError(
                                    "INTERNAL_ERROR: yolo26m_pose_postprocess "
                                    f"failed; incident={incident}"
                                ),
                            )
                        )
                        break
                else:
                    detections = np.ascontiguousarray(
                        np.stack(batch_detections), dtype=np.float32
                    )
                    keypoints = np.ascontiguousarray(
                        np.stack(batch_keypoints), dtype=np.float32
                    )
                    counts = np.ascontiguousarray(
                        np.stack(batch_counts), dtype=np.int32
                    )
                    responses.append(
                        pb_utils.InferenceResponse(
                            output_tensors=[
                                pb_utils.Tensor("DETECTIONS", detections),
                                pb_utils.Tensor("KEYPOINTS", keypoints),
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
                    "yolo26m_pose_postprocess internal failure "
                    "request_index=%d incident=%s",
                    request_index,
                    incident,
                )
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(
                            "INTERNAL_ERROR: yolo26m_pose_postprocess failed; "
                            f"incident={incident}"
                        ),
                    )
                )
        return responses

    def finalize(self):
        print("yolo26m_pose_postprocess finalized.", flush=True)
