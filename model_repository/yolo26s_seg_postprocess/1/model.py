import json
import logging
import uuid
from typing import NamedTuple

import cv2
import numpy as np
import triton_python_backend_utils as pb_utils


MAX_SIDE = 8192
MAX_PIXELS = 8_388_608
MAX_PREDICTIONS = 100_000
MAX_MASK_CHANNELS = 512
MAX_PROTO_SIDE = 640
MAX_RAW_OUTPUT_VALUES = 4_194_304
MAX_MASK_PROTO_VALUES = 8_388_608
MAX_MASK_OUTPUT_VALUES = 268_435_456
MASK_CROP_SIZE = 160
MASK_PACKED_WIDTH = MASK_CROP_SIZE // 8
PERSON_CLASS_ID = 0
LETTERBOX_SIZE = 640
LETTERBOX_BUCKET_SIZES = (192, 384, 480, 640)
LETTERBOX_STRIDE = 32
META_RTOL = 1e-6
META_ATOL = 1e-5
DEFAULT_TEST_MAX_BATCH_SIZE = 2
logger = logging.getLogger("yolo26s_seg_postprocess")


class InputValidationError(ValueError):
    pass


class CanonicalGeometry(NamedTuple):
    original_h: int
    original_w: int
    ratio: float
    resized_h: int
    resized_w: int
    left: int
    right: int
    top: int
    bottom: int
    input_h: int
    input_w: int


class SampleSelection(NamedTuple):
    """Variable-length detection and mask-coefficient selection for one sample."""

    detections: np.ndarray
    coefficients: np.ndarray


def _allocate_zeros(shape, dtype):
    return np.zeros(shape, dtype=dtype)


def _shape_value_count(shape):
    value_count = 1
    for dimension in shape:
        value_count *= int(dimension)
    return value_count


def _validate_meta(meta):
    if not isinstance(meta, np.ndarray) or meta.shape != (7,):
        raise InputValidationError("LETTERBOX_META sample must have shape [7]")
    if meta.dtype != np.float32:
        raise InputValidationError("LETTERBOX_META must have dtype float32")
    if not np.isfinite(meta).all():
        raise InputValidationError("LETTERBOX_META values must be finite")

    original_h_value = float(meta[0])
    original_w_value = float(meta[1])
    if original_h_value <= 0 or original_w_value <= 0:
        raise InputValidationError("LETTERBOX_META original dimensions must be positive")
    if original_h_value > MAX_SIDE or original_w_value > MAX_SIDE:
        raise InputValidationError(
            f"LETTERBOX_META original dimensions exceed max side {MAX_SIDE}"
        )
    if original_h_value * original_w_value > MAX_PIXELS:
        raise InputValidationError(
            f"LETTERBOX_META original image exceeds max pixels {MAX_PIXELS}"
        )

    original_h = round(original_h_value)
    original_w = round(original_w_value)
    if not np.isclose(
        original_h_value, original_h, rtol=0.0, atol=META_ATOL
    ) or not np.isclose(
        original_w_value, original_w, rtol=0.0, atol=META_ATOL
    ):
        raise InputValidationError(
            "LETTERBOX_META original dimensions must be integer-valued"
        )
    original_h = int(original_h)
    original_w = int(original_w)
    if original_h < 1 or original_w < 1:
        raise InputValidationError(
            "LETTERBOX_META original dimensions must be at least 1 pixel"
        )

    ratio = min(LETTERBOX_SIZE / original_h, LETTERBOX_SIZE / original_w)
    resized_w = round(original_w * ratio)
    resized_h = round(original_h * ratio)
    if resized_w < 1 or resized_h < 1:
        raise InputValidationError(
            "LETTERBOX_META aspect ratio produces a resize dimension smaller than 1 pixel"
        )
    target_h = next(size for size in LETTERBOX_BUCKET_SIZES if size >= resized_h)
    target_w = next(size for size in LETTERBOX_BUCKET_SIZES if size >= resized_w)
    half_pad_w = (target_w - resized_w) / 2
    half_pad_h = (target_h - resized_h) / 2
    left = round(half_pad_w - 0.1)
    right = round(half_pad_w + 0.1)
    top = round(half_pad_h - 0.1)
    bottom = round(half_pad_h + 0.1)
    input_h = resized_h + top + bottom
    input_w = resized_w + left + right

    if not np.isclose(float(meta[2]), ratio, rtol=META_RTOL, atol=META_ATOL):
        raise InputValidationError(
            "LETTERBOX_META ratio is inconsistent with original dimensions"
        )
    if not np.isclose(float(meta[3]), left, rtol=0.0, atol=META_ATOL):
        raise InputValidationError(
            "LETTERBOX_META left padding is inconsistent with letterbox geometry"
        )
    if not np.isclose(float(meta[4]), top, rtol=0.0, atol=META_ATOL):
        raise InputValidationError(
            "LETTERBOX_META top padding is inconsistent with letterbox geometry"
        )
    if not np.isclose(
        float(meta[5]), input_h, rtol=0.0, atol=META_ATOL
    ) or not np.isclose(float(meta[6]), input_w, rtol=0.0, atol=META_ATOL):
        raise InputValidationError(
            "LETTERBOX_META input dimensions are inconsistent with letterbox geometry"
        )
    if (
        input_h > LETTERBOX_SIZE
        or input_w > LETTERBOX_SIZE
        or input_h % LETTERBOX_STRIDE != 0
        or input_w % LETTERBOX_STRIDE != 0
    ):
        raise InputValidationError(
            "LETTERBOX_META reconstructed input dimensions are unsafe"
        )
    return CanonicalGeometry(
        original_h=original_h,
        original_w=original_w,
        ratio=ratio,
        resized_h=resized_h,
        resized_w=resized_w,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        input_h=input_h,
        input_w=input_w,
    )


def _validate_proto_sample(raw_proto):
    if not isinstance(raw_proto, np.ndarray):
        raise InputValidationError("MASK_PROTO sample must be a numpy array")
    if raw_proto.dtype != np.float32:
        raise InputValidationError("MASK_PROTO must have dtype float32")
    if raw_proto.ndim != 3:
        raise InputValidationError("MASK_PROTO sample must have shape [C,H,W]")
    channels, height, width = raw_proto.shape
    if channels < 1 or channels > MAX_MASK_CHANNELS:
        raise InputValidationError(
            f"MASK_PROTO channels must be between 1 and {MAX_MASK_CHANNELS}"
        )
    if height < 1 or width < 1:
        raise InputValidationError("MASK_PROTO spatial dimensions must be positive")
    if height > MAX_PROTO_SIDE or width > MAX_PROTO_SIDE:
        raise InputValidationError(f"MASK_PROTO exceeds max side {MAX_PROTO_SIDE}")
    if not np.isfinite(raw_proto).all():
        raise InputValidationError("MASK_PROTO values must be finite")
    return channels


def _validate_prediction_sample(raw_output, mask_channels):
    if not isinstance(raw_output, np.ndarray):
        raise InputValidationError("RAW_OUTPUT sample must be a numpy array")
    if raw_output.dtype != np.float32:
        raise InputValidationError("RAW_OUTPUT must have dtype float32")
    expected_width = 6 + mask_channels
    if raw_output.ndim != 2 or raw_output.shape[1] != expected_width:
        raise InputValidationError(
            "YOLO segment output must use the end-to-end layout "
            f"[N, 6 + mask_channels] with width {expected_width}, got {raw_output.shape}"
        )
    if raw_output.shape[0] > MAX_PREDICTIONS:
        raise InputValidationError(
            f"RAW_OUTPUT exceeds max predictions {MAX_PREDICTIONS}"
        )
    if not np.isfinite(raw_output).all():
        raise InputValidationError("RAW_OUTPUT values must be finite")


class TritonPythonModel:
    def initialize(self, args):
        self.conf_threshold = 0.5
        self.max_detections = 100
        self.max_batch_size = DEFAULT_TEST_MAX_BATCH_SIZE
        model_config_value = (args or {}).get("model_config")
        if model_config_value:
            if isinstance(model_config_value, str):
                model_config = json.loads(model_config_value)
            elif isinstance(model_config_value, dict):
                model_config = model_config_value
            else:
                raise ValueError("model_config must be a JSON string or mapping")
            configured_max = int(model_config.get("max_batch_size", self.max_batch_size))
            if configured_max < 1:
                raise ValueError("model_config max_batch_size must be positive")
            self.max_batch_size = configured_max
        self._gpu = None
        if (args or {}).get("model_instance_kind") == "GPU":
            import importlib.util
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

    def _preflight_batch(self, raw_output, raw_proto, meta):
        if not isinstance(raw_output, np.ndarray):
            raise InputValidationError("RAW_OUTPUT must be a numpy array")
        if raw_output.ndim != 3:
            raise InputValidationError(
                "RAW_OUTPUT batch must have shape [B,N,6 + mask_channels]"
            )
        if raw_output.dtype != np.float32:
            raise InputValidationError("RAW_OUTPUT must have dtype float32")
        batch_size = int(raw_output.shape[0])
        if batch_size == 0:
            raise InputValidationError("input batch must be non-empty")
        if batch_size > self.max_batch_size:
            raise InputValidationError(
                f"batch size {batch_size} exceeds configured maximum {self.max_batch_size}"
            )
        if raw_output.shape[1] > MAX_PREDICTIONS:
            raise InputValidationError(
                f"RAW_OUTPUT exceeds max predictions {MAX_PREDICTIONS}"
            )
        raw_value_count = _shape_value_count(raw_output.shape)
        if raw_value_count > MAX_RAW_OUTPUT_VALUES:
            raise InputValidationError(
                f"RAW_OUTPUT total values exceed {MAX_RAW_OUTPUT_VALUES}"
            )

        if not isinstance(raw_proto, np.ndarray):
            raise InputValidationError("MASK_PROTO must be a numpy array")
        if raw_proto.ndim != 4:
            raise InputValidationError("MASK_PROTO batch must have shape [B,C,H,W]")
        if raw_proto.dtype != np.float32:
            raise InputValidationError("MASK_PROTO must have dtype float32")
        proto_value_count = _shape_value_count(raw_proto.shape)
        if proto_value_count > MAX_MASK_PROTO_VALUES:
            raise InputValidationError(
                f"MASK_PROTO total values exceed {MAX_MASK_PROTO_VALUES}"
            )

        if not isinstance(meta, np.ndarray):
            raise InputValidationError("LETTERBOX_META must be a numpy array")
        if meta.ndim != 2 or meta.shape[1] != 7:
            raise InputValidationError("LETTERBOX_META batch must have shape [B,7]")
        if meta.dtype != np.float32:
            raise InputValidationError("LETTERBOX_META must have dtype float32")
        if raw_proto.shape[0] != batch_size or meta.shape[0] != batch_size:
            raise InputValidationError(
                "RAW_OUTPUT, MASK_PROTO, and LETTERBOX_META batch size must match"
            )

        geometries = []
        expected_size = None
        for batch_index in range(batch_size):
            try:
                geometry = _validate_meta(meta[batch_index])
                mask_channels = _validate_proto_sample(raw_proto[batch_index])
                _validate_prediction_sample(raw_output[batch_index], mask_channels)
                sample_size = (geometry.original_h, geometry.original_w)
                if expected_size is None:
                    expected_size = sample_size
                elif sample_size != expected_size:
                    raise InputValidationError(
                        "all samples in a request must have the same original height and width"
                    )
                geometries.append(geometry)
            except InputValidationError as exc:
                raise InputValidationError(f"batch index {batch_index}: {exc}") from exc

        return tuple(geometries)

    def _validate_sample(self, raw_output, raw_proto, meta):
        geometry = _validate_meta(meta)
        mask_channels = _validate_proto_sample(raw_proto)
        _validate_prediction_sample(raw_output, mask_channels)
        return geometry

    @staticmethod
    def _empty_sample_selection(mask_channels):
        return SampleSelection(
            detections=_allocate_zeros((0, 6), np.float32),
            coefficients=_allocate_zeros((0, mask_channels), np.float32),
        )

    @staticmethod
    def _scale_boxes_to_original(boxes, geometry):
        scaled = boxes.astype(np.float32, copy=True)
        scaled[:, [0, 2]] = (scaled[:, [0, 2]] - geometry.left) / geometry.ratio
        scaled[:, [1, 3]] = (scaled[:, [1, 3]] - geometry.top) / geometry.ratio
        scaled[:, [0, 2]] = np.clip(
            scaled[:, [0, 2]], 0, geometry.original_w
        )
        scaled[:, [1, 3]] = np.clip(
            scaled[:, [1, 3]], 0, geometry.original_h
        )
        return scaled

    @staticmethod
    def _render_mask_crops_into(
        output_crops, proto, coefficients, boxes_original, geometry
    ):
        """Render one binary crop at a time into big-endian packed rows."""
        count = int(output_crops.shape[0])
        if count == 0:
            return

        channels, proto_h, proto_w = proto.shape
        flattened_proto = proto.reshape(channels, -1)
        boxes_proto = boxes_original.astype(np.float32, copy=True)
        boxes_proto[:, [0, 2]] = (
            boxes_proto[:, [0, 2]] * geometry.ratio + geometry.left
        ) * (proto_w / geometry.input_w)
        boxes_proto[:, [1, 3]] = (
            boxes_proto[:, [1, 3]] * geometry.ratio + geometry.top
        ) * (proto_h / geometry.input_h)

        binary_mask = np.empty((MASK_CROP_SIZE, MASK_CROP_SIZE), dtype=np.uint8)
        for index in range(count):
            logit = (coefficients[index] @ flattened_proto).reshape(
                proto_h, proto_w
            )
            x1, y1, x2, y2 = boxes_proto[index]
            scale_x = (x2 - x1) / MASK_CROP_SIZE
            scale_y = (y2 - y1) / MASK_CROP_SIZE
            destination_to_source = np.array(
                [
                    [scale_x, 0.0, x1 + 0.5 * scale_x - 0.5],
                    [0.0, scale_y, y1 + 0.5 * scale_y - 0.5],
                ],
                dtype=np.float32,
            )
            sampled_logit = cv2.warpAffine(
                logit,
                destination_to_source,
                (MASK_CROP_SIZE, MASK_CROP_SIZE),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
            np.greater(sampled_logit, 0.0, out=binary_mask)
            output_crops[index] = np.packbits(binary_mask, axis=-1, bitorder="big")

    def _process_sample(self, raw_output, geometry):
        mask_channels = int(raw_output.shape[1] - 6)
        if raw_output.size == 0:
            return self._empty_sample_selection(mask_channels)
        boxes_letterbox = raw_output[:, :4]
        scores = raw_output[:, 4]
        classes = np.round(raw_output[:, 5]).astype(np.float32)
        coefficients = raw_output[:, 6:]
        # Only people may consume top-k slots or require mask rendering.
        confident_persons = (scores >= self.conf_threshold) & (
            classes == PERSON_CLASS_ID
        )
        if not np.any(confident_persons):
            return self._empty_sample_selection(mask_channels)

        boxes_letterbox = boxes_letterbox[confident_persons].astype(np.float32)
        scores = scores[confident_persons].astype(np.float32)
        classes = classes[confident_persons].astype(np.float32)
        coefficients = coefficients[confident_persons].astype(np.float32)
        boxes_original = self._scale_boxes_to_original(boxes_letterbox, geometry)
        # Match restore_mask_crops: Python round (ties-to-even), then clip.
        boxes_pixels = np.fromiter(
            (round(float(coordinate)) for coordinate in boxes_original.flat),
            dtype=np.int64,
            count=boxes_original.size,
        ).reshape(boxes_original.shape)
        boxes_pixels[:, [0, 2]] = np.clip(
            boxes_pixels[:, [0, 2]], 0, geometry.original_w
        )
        boxes_pixels[:, [1, 3]] = np.clip(
            boxes_pixels[:, [1, 3]], 0, geometry.original_h
        )
        valid_float_area = (boxes_original[:, 2] > boxes_original[:, 0]) & (
            boxes_original[:, 3] > boxes_original[:, 1]
        )
        valid_pixel_area = (boxes_pixels[:, 2] > boxes_pixels[:, 0]) & (
            boxes_pixels[:, 3] > boxes_pixels[:, 1]
        )
        valid = valid_float_area & valid_pixel_area
        boxes_original = boxes_original[valid]
        scores = scores[valid]
        classes = classes[valid]
        coefficients = coefficients[valid]
        if boxes_original.size == 0:
            return self._empty_sample_selection(mask_channels)

        keep = np.argsort(-scores, kind="stable")[: self.max_detections]
        count = int(keep.size)
        detections = _allocate_zeros((count, 6), np.float32)
        detections[:, :4] = boxes_original[keep]
        detections[:, 4] = scores[keep]
        detections[:, 5] = classes[keep]
        return SampleSelection(
            detections=detections,
            coefficients=np.ascontiguousarray(coefficients[keep]),
        )

    def _pack_batch_outputs(self, protos, geometries, sample_selections):
        batch_size = len(sample_selections)
        output_slots = max(
            1,
            max(
                int(selection.detections.shape[0])
                for selection in sample_selections
            ),
        )
        # Keep the original logical pixel budget despite the smaller payload.
        output_value_count = (
            batch_size * output_slots * MASK_CROP_SIZE * MASK_CROP_SIZE
        )
        if output_value_count > MAX_MASK_OUTPUT_VALUES:
            raise InputValidationError(
                "MASK_CROPS_PACKED logical output values exceed budget "
                f"{MAX_MASK_OUTPUT_VALUES}"
            )

        detections = _allocate_zeros(
            (batch_size, output_slots, 6), np.float32
        )
        mask_crops = _allocate_zeros(
            (batch_size, output_slots, MASK_CROP_SIZE, MASK_PACKED_WIDTH),
            np.uint8,
        )
        counts = _allocate_zeros((batch_size, 1), np.int32)
        for batch_index, selection in enumerate(sample_selections):
            count = int(selection.detections.shape[0])
            if count:
                detections[batch_index, :count] = selection.detections
                self._render_mask_crops_into(
                    mask_crops[batch_index, :count],
                    protos[batch_index],
                    selection.coefficients,
                    selection.detections[:, :4],
                    geometries[batch_index],
                )
            counts[batch_index, 0] = count
        return detections, mask_crops, counts

    def _postprocess(self, raw_output, raw_proto, meta):
        geometry = self._validate_sample(raw_output, raw_proto, meta)
        selection = self._process_sample(raw_output, geometry)
        detections, mask_crops, num_detections = self._pack_batch_outputs(
            (raw_proto,), (geometry,), (selection,)
        )
        return detections[0], mask_crops[0], num_detections[0]

    def execute(self, requests):
        responses = []
        for request_index, request in enumerate(requests):
            try:
                if getattr(self, "_gpu", None) is not None:
                    responses.append(self._gpu.execute(request, pb_utils, request_index))
                    continue
                output_tensor = pb_utils.get_input_tensor_by_name(request, "RAW_OUTPUT")
                proto_tensor = pb_utils.get_input_tensor_by_name(request, "MASK_PROTO")
                meta_tensor = pb_utils.get_input_tensor_by_name(request, "LETTERBOX_META")
                if output_tensor is None:
                    raise InputValidationError("missing required input RAW_OUTPUT")
                if proto_tensor is None:
                    raise InputValidationError("missing required input MASK_PROTO")
                if meta_tensor is None:
                    raise InputValidationError("missing required input LETTERBOX_META")

                raw_output = output_tensor.as_numpy()
                raw_proto = proto_tensor.as_numpy()
                meta = meta_tensor.as_numpy()
                geometries = self._preflight_batch(raw_output, raw_proto, meta)
                sample_selections = tuple(
                    self._process_sample(
                        raw_output[batch_index],
                        geometries[batch_index],
                    )
                    for batch_index in range(int(raw_output.shape[0]))
                )
                detections, mask_crops, counts = self._pack_batch_outputs(
                    raw_proto, geometries, sample_selections
                )
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor(
                                "DETECTIONS", np.ascontiguousarray(detections)
                            ),
                            pb_utils.Tensor(
                                "MASK_CROPS_PACKED", mask_crops
                            ),
                            pb_utils.Tensor(
                                "NUM_DETECTIONS", np.ascontiguousarray(counts)
                            ),
                        ]
                    )
                )
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
                    "yolo26s_seg_postprocess internal failure "
                    "request_index=%d incident=%s",
                    request_index,
                    incident,
                )
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(
                            "INTERNAL_ERROR: yolo26s_seg_postprocess failed; "
                            f"incident={incident}"
                        ),
                    )
                )
        return responses

    def finalize(self):
        print("yolo26s_seg_postprocess finalized.", flush=True)
