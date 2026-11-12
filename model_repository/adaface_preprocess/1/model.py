import logging
import uuid

import cv2
import numpy as np
from skimage import transform as trans
import triton_python_backend_utils as pb_utils


REFERENCE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)
OUTPUT_SIZE = 112
MAX_SIDE = 8192
MAX_PIXELS = 8_388_608
LANDMARK_MARGIN_RATIO = 0.25
MIN_LANDMARK_RMS_SPREAD = 1.0
# Broad bounds reject pathological transforms while allowing large-image crops.
MIN_SIMILARITY_SCALE = 1.0 / 256.0
MAX_SIMILARITY_SCALE = 256.0
logger = logging.getLogger("adaface_preprocess")


class InputValidationError(ValueError):
    pass


def align_face(image, landmark):
    height, width = image.shape[:2]
    x_coordinates = landmark[:, 0]
    y_coordinates = landmark[:, 1]
    if (
        np.any(x_coordinates < -LANDMARK_MARGIN_RATIO * width)
        or np.any(x_coordinates > (1.0 + LANDMARK_MARGIN_RATIO) * width)
        or np.any(y_coordinates < -LANDMARK_MARGIN_RATIO * height)
        or np.any(y_coordinates > (1.0 + LANDMARK_MARGIN_RATIO) * height)
    ):
        raise InputValidationError(
            "LANDMARK coordinates exceed the 25% image margin"
        )

    landmark_float64 = landmark.astype(np.float64, copy=False)
    centered = landmark_float64 - landmark_float64.mean(axis=0)
    rms_spread = float(
        np.sqrt(np.mean(np.sum(centered * centered, axis=1)))
    )
    if rms_spread < MIN_LANDMARK_RMS_SPREAD:
        raise InputValidationError(
            "LANDMARK RMS spread for alignment transform must be at least "
            f"{MIN_LANDMARK_RMS_SPREAD} px"
        )

    transform = trans.SimilarityTransform()
    if not transform.estimate(landmark, REFERENCE):
        raise InputValidationError("unable estimate face alignment transform")
    matrix = transform.params[:2, :]
    if not np.isfinite(matrix).all():
        raise InputValidationError(
            "face alignment transform matrix must be finite"
        )
    scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))
    if not MIN_SIMILARITY_SCALE <= scale <= MAX_SIMILARITY_SCALE:
        raise InputValidationError(
            "face alignment transform scale must be within "
            f"[{MIN_SIMILARITY_SCALE}, {MAX_SIMILARITY_SCALE}]"
        )
    return cv2.warpAffine(
        image,
        matrix,
        (OUTPUT_SIZE, OUTPUT_SIZE),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def preprocess(image, landmark):
    if not isinstance(image, np.ndarray):
        raise InputValidationError("RAW_IMAGE must be a numpy array")
    if image.dtype != np.uint8:
        raise InputValidationError("RAW_IMAGE must have dtype uint8")
    if image.ndim != 3 or image.shape[2] != 3:
        raise InputValidationError("RAW_IMAGE must have shape [H,W,3]")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise InputValidationError(
            "RAW_IMAGE height and width must be positive"
        )
    height, width = image.shape[:2]
    if height > MAX_SIDE or width > MAX_SIDE:
        raise InputValidationError(f"RAW_IMAGE exceeds max side {MAX_SIDE}")
    if height * width > MAX_PIXELS:
        raise InputValidationError(
            f"RAW_IMAGE exceeds max pixels {MAX_PIXELS}"
        )

    landmark = np.asarray(landmark)
    if landmark.shape != (5, 2):
        raise InputValidationError("LANDMARK must have shape [5,2]")
    if landmark.dtype != np.float32:
        raise InputValidationError("LANDMARK must use float32")
    landmark = landmark.astype(np.float32, copy=False)
    if not np.isfinite(landmark).all():
        raise InputValidationError("LANDMARK values must be finite")

    aligned = align_face(image, landmark)
    normalized = (aligned.astype(np.float32) - 127.5) / 127.5
    return np.ascontiguousarray(normalized.transpose(2, 0, 1), dtype=np.float32)


def _validate_batch_inputs(images, landmarks):
    if not isinstance(images, np.ndarray):
        raise InputValidationError("RAW_IMAGE must be a numpy array")
    if images.ndim != 4 or images.shape[3] != 3:
        raise InputValidationError(
            "RAW_IMAGE batch must have shape [B,H,W,3]; "
            "each sample must have shape [H,W,3]"
        )
    if not isinstance(landmarks, np.ndarray):
        raise InputValidationError("LANDMARK must be a numpy array")
    if landmarks.ndim != 3 or landmarks.shape[1:] != (5, 2):
        raise InputValidationError(
            "LANDMARK batch must have shape [B,5,2]; "
            "each sample must have shape [5,2]"
        )
    if images.shape[0] == 0:
        raise InputValidationError("input batch must be non-empty")
    if images.shape[0] != landmarks.shape[0]:
        raise InputValidationError(
            "RAW_IMAGE and LANDMARK batch size must match"
        )


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request_index, request in enumerate(requests):
            try:
                image_tensor = pb_utils.get_input_tensor_by_name(
                    request, "RAW_IMAGE"
                )
                landmark_tensor = pb_utils.get_input_tensor_by_name(
                    request, "LANDMARK"
                )
                if image_tensor is None:
                    raise InputValidationError(
                        "missing required input RAW_IMAGE"
                    )
                if landmark_tensor is None:
                    raise InputValidationError(
                        "missing required input LANDMARK"
                    )
                images = image_tensor.as_numpy()
                landmarks = landmark_tensor.as_numpy()
                _validate_batch_inputs(images, landmarks)
                processed = []
                for batch_index, (image, landmark) in enumerate(
                    zip(images, landmarks)
                ):
                    try:
                        processed.append(preprocess(image, landmark))
                    except InputValidationError as exc:
                        raise InputValidationError(
                            f"batch index {batch_index}: {exc}"
                        ) from exc
                    except Exception:
                        incident = uuid.uuid4().hex[:12]
                        logger.exception(
                            "adaface_preprocess internal failure "
                            "request_index=%d batch_index=%d incident=%s",
                            request_index,
                            batch_index,
                            incident,
                        )
                        responses.append(
                            pb_utils.InferenceResponse(
                                output_tensors=[],
                                error=pb_utils.TritonError(
                                    "INTERNAL_ERROR: adaface_preprocess failed; "
                                    f"incident={incident}"
                                ),
                            )
                        )
                        break
                else:
                    output = np.ascontiguousarray(
                        np.stack(processed, axis=0), dtype=np.float32
                    )
                    responses.append(
                        pb_utils.InferenceResponse(
                            output_tensors=[pb_utils.Tensor("input", output)]
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
                    "adaface_preprocess internal failure "
                    "request_index=%d incident=%s",
                    request_index,
                    incident,
                )
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(
                            "INTERNAL_ERROR: adaface_preprocess failed; "
                            f"incident={incident}"
                        ),
                    )
                )
        return responses

    def finalize(self):
        print("adaface_preprocess finalized.", flush=True)
