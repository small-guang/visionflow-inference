import logging
import uuid

import cv2
import numpy as np
import triton_python_backend_utils as pb_utils


MIN_AFFINE_SCALE = 1.0 / 256.0
MAX_AFFINE_SCALE = 256.0
MAX_AFFINE_CONDITION_NUMBER = 1.01
logger = logging.getLogger("face_keypoint_2d106_post")


class InputValidationError(ValueError):
    pass


def trans_points2d(points, matrix):
    """Apply one affine matrix to one array of 2D points."""
    new_points = np.zeros(shape=points.shape, dtype=np.float32)
    for index, point in enumerate(points):
        homogeneous = np.array(
            [point[0], point[1], 1.0], dtype=np.float64
        )
        transformed = np.dot(matrix, homogeneous)
        new_points[index] = transformed[0:2]
    return new_points


def postprocess(raw_output, affine_matrix, *, input_size):
    """Map one normalized landmark prediction to source-image coordinates."""
    raw_output = np.asarray(raw_output)
    if raw_output.shape != (212,):
        raise InputValidationError("RAW_OUTPUT sample must have shape [212]")
    if raw_output.dtype != np.float32:
        raise InputValidationError("RAW_OUTPUT must have dtype float32")
    if not np.isfinite(raw_output).all():
        raise InputValidationError("RAW_OUTPUT values must be finite")

    affine_matrix = np.asarray(affine_matrix)
    if affine_matrix.shape != (2, 3):
        raise InputValidationError("AFFINE_MATRIX sample must have shape [2,3]")
    if affine_matrix.dtype != np.float64:
        raise InputValidationError("AFFINE_MATRIX must have dtype float64")
    if not np.isfinite(affine_matrix).all():
        raise InputValidationError("AFFINE_MATRIX values must be finite")
    linear = affine_matrix[:, :2].astype(np.float64, copy=False)
    try:
        singular_values = np.linalg.svd(linear, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise InputValidationError(
            "AFFINE_MATRIX singular values could not be computed"
        ) from exc
    max_scale = float(singular_values[0])
    min_scale = float(singular_values[-1])
    if (
        min_scale < MIN_AFFINE_SCALE
        or max_scale > MAX_AFFINE_SCALE
    ):
        raise InputValidationError(
            "AFFINE_MATRIX transform scale must be within "
            f"[{MIN_AFFINE_SCALE}, {MAX_AFFINE_SCALE}]"
        )
    condition_number = max_scale / min_scale
    if condition_number > MAX_AFFINE_CONDITION_NUMBER:
        raise InputValidationError(
            "AFFINE_MATRIX transform condition number must be at most "
            f"{MAX_AFFINE_CONDITION_NUMBER}"
        )

    prediction = raw_output.astype(np.float32, copy=True).reshape(106, 2)
    prediction += np.float32(1.0)
    prediction *= np.float32(input_size // 2)

    inverse = cv2.invertAffineTransform(
        affine_matrix.astype(np.float64, copy=False)
    )
    if inverse.shape != (2, 3) or not np.isfinite(inverse).all():
        raise InputValidationError(
            "AFFINE_MATRIX inverse transform must be finite"
        )
    landmarks = trans_points2d(prediction, inverse)
    if not np.isfinite(landmarks).all():
        raise InputValidationError("transformed landmarks must be finite")
    return np.ascontiguousarray(landmarks, dtype=np.float32)


def _validate_batch_inputs(raw_output, affine_matrices):
    if not isinstance(raw_output, np.ndarray):
        raise InputValidationError("RAW_OUTPUT must be a numpy array")
    if raw_output.ndim != 2 or raw_output.shape[1] != 212:
        raise InputValidationError(
            "RAW_OUTPUT batch must have shape [B,212]"
        )
    if raw_output.dtype != np.float32:
        raise InputValidationError("RAW_OUTPUT must have dtype float32")
    if not isinstance(affine_matrices, np.ndarray):
        raise InputValidationError("AFFINE_MATRIX must be a numpy array")
    if affine_matrices.ndim != 3 or affine_matrices.shape[1:] != (2, 3):
        raise InputValidationError(
            "AFFINE_MATRIX batch must have shape [B,2,3]"
        )
    if affine_matrices.dtype != np.float64:
        raise InputValidationError("AFFINE_MATRIX must have dtype float64")
    if raw_output.shape[0] == 0:
        raise InputValidationError("input batch must be non-empty")
    if raw_output.shape[0] != affine_matrices.shape[0]:
        raise InputValidationError(
            "RAW_OUTPUT and AFFINE_MATRIX batch size must match"
        )


class TritonPythonModel:
    def initialize(self, args):
        self.input_size = 192
        self.lmk_dim = 2
        self.lmk_num = 106

    def execute(self, requests):
        responses = []
        for request_index, request in enumerate(requests):
            try:
                raw_tensor = pb_utils.get_input_tensor_by_name(
                    request, "RAW_OUTPUT"
                )
                affine_tensor = pb_utils.get_input_tensor_by_name(
                    request, "AFFINE_MATRIX"
                )
                if raw_tensor is None:
                    raise InputValidationError(
                        "missing required input RAW_OUTPUT"
                    )
                if affine_tensor is None:
                    raise InputValidationError(
                        "missing required input AFFINE_MATRIX"
                    )

                raw_output = raw_tensor.as_numpy()
                affine_matrices = affine_tensor.as_numpy()
                _validate_batch_inputs(raw_output, affine_matrices)

                batch_landmarks = []
                for batch_index, (prediction, matrix) in enumerate(
                    zip(raw_output, affine_matrices)
                ):
                    try:
                        batch_landmarks.append(
                            postprocess(
                                prediction,
                                matrix,
                                input_size=self.input_size,
                            )
                        )
                    except InputValidationError as exc:
                        raise InputValidationError(
                            f"batch index {batch_index}: {exc}"
                        ) from exc
                    except Exception:
                        incident = uuid.uuid4().hex[:12]
                        logger.exception(
                            "face_keypoint_2d106_post internal failure "
                            "request_index=%d batch_index=%d incident=%s",
                            request_index,
                            batch_index,
                            incident,
                        )
                        responses.append(
                            pb_utils.InferenceResponse(
                                output_tensors=[],
                                error=pb_utils.TritonError(
                                    "INTERNAL_ERROR: "
                                    "face_keypoint_2d106_post failed; "
                                    f"incident={incident}"
                                ),
                            )
                        )
                        break
                else:
                    landmarks = np.ascontiguousarray(
                        np.stack(batch_landmarks, axis=0), dtype=np.float32
                    )
                    responses.append(
                        pb_utils.InferenceResponse(
                            output_tensors=[
                                pb_utils.Tensor(
                                    "LANDMARKS_106", landmarks
                                )
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
                    "face_keypoint_2d106_post internal failure "
                    "request_index=%d incident=%s",
                    request_index,
                    incident,
                )
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(
                            "INTERNAL_ERROR: "
                            "face_keypoint_2d106_post failed; "
                            f"incident={incident}"
                        ),
                    )
                )
        return responses

    def finalize(self):
        print("face_keypoint_2d106_post finalized.", flush=True)
