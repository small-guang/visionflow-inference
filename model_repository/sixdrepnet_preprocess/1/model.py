import logging
import uuid

import numpy as np
import triton_python_backend_utils as pb_utils
from PIL import Image


OUTPUT_SIZE = 224
MAX_IMAGE_SIDE = 8192
MAX_IMAGE_PIXELS = 8_388_608
MAX_ASPECT_RATIO = 20
MAX_RESIZED_LONG_EDGE = 8192
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
logger = logging.getLogger("sixdrepnet_preprocess")


class InputValidationError(ValueError):
    pass


class _UnexpectedSampleError(RuntimeError):
    def __init__(self, batch_index):
        self.batch_index = batch_index
        super().__init__(f"unexpected preprocessing failure at batch index {batch_index}")


def _validated_resize_dimensions(height, width):
    long_side = max(height, width)
    short_side = min(height, width)
    if long_side > MAX_IMAGE_SIDE:
        raise InputValidationError(
            f"RAW_FACE_IMAGE maximum side must not exceed {MAX_IMAGE_SIDE}"
        )
    if height * width > MAX_IMAGE_PIXELS:
        raise InputValidationError(
            "RAW_FACE_IMAGE pixel count must not exceed "
            f"{MAX_IMAGE_PIXELS}"
        )
    if long_side > short_side * MAX_ASPECT_RATIO:
        raise InputValidationError(
            "RAW_FACE_IMAGE aspect ratio must not exceed "
            f"{MAX_ASPECT_RATIO}:1"
        )

    resized_long_edge = OUTPUT_SIZE * long_side // short_side
    if resized_long_edge > MAX_RESIZED_LONG_EDGE:
        raise InputValidationError(
            "RAW_FACE_IMAGE resized long edge must not exceed "
            f"{MAX_RESIZED_LONG_EDGE}"
        )

    if width >= height:
        return resized_long_edge, OUTPUT_SIZE
    return OUTPUT_SIZE, resized_long_edge


def preprocess(image):
    if not isinstance(image, np.ndarray):
        raise InputValidationError("RAW_FACE_IMAGE must be a numpy array")
    if image.dtype != np.uint8:
        raise InputValidationError("RAW_FACE_IMAGE must have dtype uint8")
    if image.ndim != 3 or image.shape[2] != 3:
        raise InputValidationError("RAW_FACE_IMAGE must have shape [H,W,3]")

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise InputValidationError(
            "RAW_FACE_IMAGE height and width must be positive"
        )

    resized_width, resized_height = _validated_resize_dimensions(height, width)
    rgb = np.ascontiguousarray(image[:, :, ::-1])
    resized = Image.fromarray(rgb).resize(
        (resized_width, resized_height),
        resample=Image.Resampling.BILINEAR,
    )
    crop_top = int(round((resized_height - OUTPUT_SIZE) / 2.0))
    crop_left = int(round((resized_width - OUTPUT_SIZE) / 2.0))
    cropped = np.asarray(
        resized.crop(
            (
                crop_left,
                crop_top,
                crop_left + OUTPUT_SIZE,
                crop_top + OUTPUT_SIZE,
            )
        )
    )

    normalized = (
        cropped.astype(np.float32) / np.float32(255.0) - MEAN
    ) / STD
    chw = normalized.transpose(2, 0, 1)
    return np.ascontiguousarray(chw, dtype=np.float32)


def _validate_batch_input(images):
    if not isinstance(images, np.ndarray):
        raise InputValidationError("RAW_FACE_IMAGE must be a numpy array")
    if images.ndim != 4 or images.shape[3] != 3:
        raise InputValidationError(
            "RAW_FACE_IMAGE batch must have shape [B,H,W,3]"
        )
    if images.dtype != np.uint8:
        raise InputValidationError("RAW_FACE_IMAGE must have dtype uint8")
    if images.shape[0] == 0:
        raise InputValidationError("input batch must be non-empty")


def preprocess_batch(images):
    _validate_batch_input(images)
    outputs = []
    for batch_index, image in enumerate(images):
        try:
            outputs.append(preprocess(image))
        except InputValidationError as exc:
            raise InputValidationError(
                f"batch index {batch_index}: {exc}"
            ) from exc
        except Exception as exc:
            raise _UnexpectedSampleError(batch_index) from exc
    return np.ascontiguousarray(np.stack(outputs), dtype=np.float32)


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request_index, request in enumerate(requests):
            try:
                input_tensor = pb_utils.get_input_tensor_by_name(
                    request, "RAW_FACE_IMAGE"
                )
                if input_tensor is None:
                    raise InputValidationError(
                        "missing required input RAW_FACE_IMAGE"
                    )
                value = preprocess_batch(input_tensor.as_numpy())
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor("PREPROCESSED_IMAGE", value)
                        ]
                    )
                )
            except InputValidationError as error:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(
                            "INVALID_INPUT: sixdrepnet_preprocess: "
                            f"{error}"
                        )
                    )
                )
            except _UnexpectedSampleError as error:
                incident = uuid.uuid4().hex[:12]
                logger.error(
                    "sixdrepnet_preprocess internal failure "
                    "request_index=%d batch_index=%d incident=%s",
                    request_index,
                    error.batch_index,
                    incident,
                )
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(
                            "INTERNAL_ERROR: sixdrepnet_preprocess failed; "
                            f"incident={incident}"
                        )
                    )
                )
            except Exception:
                incident = uuid.uuid4().hex[:12]
                logger.error(
                    "sixdrepnet_preprocess internal failure "
                    "request_index=%d batch_index=unknown incident=%s",
                    request_index,
                    incident,
                )
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(
                            "INTERNAL_ERROR: sixdrepnet_preprocess failed; "
                            f"incident={incident}"
                        )
                    )
                )
        return responses
