import logging
import uuid

import numpy as np
import triton_python_backend_utils as pb_utils


ORTHOGONALITY_TOLERANCE = 1e-3
DETERMINANT_TOLERANCE = 1e-3
SINGULAR_TOLERANCE = 1e-6
logger = logging.getLogger("sixdrepnet_postprocess")


class InputValidationError(ValueError):
    pass


class _UnexpectedSampleError(RuntimeError):
    def __init__(self, batch_index):
        self.batch_index = batch_index
        super().__init__(f"unexpected postprocessing failure at batch index {batch_index}")


def decode_head_pose(rotation):
    rotation = np.asarray(rotation)
    if rotation.shape != (3, 3):
        raise InputValidationError("ROTATION_MATRIX must have shape [3,3]")
    if rotation.dtype != np.float32:
        raise InputValidationError("ROTATION_MATRIX must have dtype float32")
    if not np.isfinite(rotation).all():
        raise InputValidationError(
            "ROTATION_MATRIX must contain only finite values"
        )

    matrix = rotation.astype(np.float64, copy=False)
    identity_error = np.max(
        np.abs(matrix.T @ matrix - np.eye(3, dtype=np.float64))
    )
    if identity_error > ORTHOGONALITY_TOLERANCE:
        raise InputValidationError("ROTATION_MATRIX must be orthogonal")

    determinant = np.linalg.det(matrix)
    if abs(determinant - 1.0) > DETERMINANT_TOLERANCE:
        raise InputValidationError("ROTATION_MATRIX determinant must be 1")

    sy = np.hypot(matrix[0, 0], matrix[1, 0])
    if sy < SINGULAR_TOLERANCE:
        pitch = np.arctan2(-matrix[1, 2], matrix[1, 1])
        yaw = np.arctan2(-matrix[2, 0], sy)
        roll = 0.0
    else:
        pitch = np.arctan2(matrix[2, 1], matrix[2, 2])
        yaw = np.arctan2(-matrix[2, 0], sy)
        roll = np.arctan2(matrix[1, 0], matrix[0, 0])

    head_pose = np.rad2deg(
        np.asarray([pitch, yaw, roll], dtype=np.float64)
    ).astype(np.float32)
    if not np.isfinite(head_pose).all():
        raise InputValidationError(
            "decoded HEAD_POSE must contain only finite values"
        )
    return np.ascontiguousarray(head_pose, dtype=np.float32)


def _validate_batch_input(rotations):
    if not isinstance(rotations, np.ndarray):
        raise InputValidationError("ROTATION_MATRIX must be a numpy array")
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise InputValidationError(
            "ROTATION_MATRIX batch must have shape [B,3,3]"
        )
    if rotations.dtype != np.float32:
        raise InputValidationError("ROTATION_MATRIX must have dtype float32")
    if rotations.shape[0] == 0:
        raise InputValidationError("input batch must be non-empty")


def decode_batch(rotations):
    _validate_batch_input(rotations)
    outputs = []
    for batch_index, rotation in enumerate(rotations):
        try:
            outputs.append(decode_head_pose(rotation))
        except InputValidationError as exc:
            raise InputValidationError(f"batch index {batch_index}: {exc}") from exc
        except Exception as exc:
            raise _UnexpectedSampleError(batch_index) from exc
    return np.ascontiguousarray(np.stack(outputs), dtype=np.float32)


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request_index, request in enumerate(requests):
            try:
                input_tensor = pb_utils.get_input_tensor_by_name(
                    request, "ROTATION_MATRIX"
                )
                if input_tensor is None:
                    raise InputValidationError(
                        "missing required input ROTATION_MATRIX"
                    )
                head_pose = decode_batch(input_tensor.as_numpy())
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor("HEAD_POSE", head_pose)
                        ]
                    )
                )
            except InputValidationError as error:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(
                            "INVALID_INPUT: sixdrepnet_postprocess: "
                            f"{error}"
                        )
                    )
                )
            except _UnexpectedSampleError as error:
                incident = uuid.uuid4().hex[:12]
                logger.error(
                    "sixdrepnet_postprocess internal failure "
                    "request_index=%d batch_index=%d incident=%s",
                    request_index,
                    error.batch_index,
                    incident,
                )
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(
                            "INTERNAL_ERROR: sixdrepnet_postprocess failed; "
                            f"incident={incident}"
                        )
                    )
                )
            except Exception:
                incident = uuid.uuid4().hex[:12]
                logger.error(
                    "sixdrepnet_postprocess internal failure "
                    "request_index=%d batch_index=unknown incident=%s",
                    request_index,
                    incident,
                )
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(
                            "INTERNAL_ERROR: sixdrepnet_postprocess failed; "
                            f"incident={incident}"
                        )
                    )
                )
        return responses
