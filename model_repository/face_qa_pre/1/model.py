import json
import logging
import sys
import uuid

import cv2
import numpy as np
import triton_python_backend_utils as pb_utils


MAX_SIDE = 8192
MAX_PIXELS = 8_388_608
LANDMARK_MARGIN_RATIO = 0.25
MIN_LANDMARK_RMS_SPREAD = 1.0
MIN_SIMILARITY_SCALE = 1.0 / 256.0
MAX_SIMILARITY_SCALE = 256.0
FACE_QUALITY_REFERENCE_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)
logger = logging.getLogger("face_qa_pre")


class InputValidationError(ValueError):
    pass


def _alignment_matrix(landmark):
    source = landmark.astype(np.float64, copy=False)
    destination = FACE_QUALITY_REFERENCE_LANDMARKS.astype(
        np.float64, copy=False
    )
    source_mean = source.mean(axis=0)
    destination_mean = destination.mean(axis=0)
    source_centered = source - source_mean
    destination_centered = destination - destination_mean
    covariance = destination_centered.T @ source_centered / source.shape[0]
    reflection = np.ones(2, dtype=np.float64)
    if np.linalg.det(covariance) < 0:
        reflection[-1] = -1
    try:
        left, singular_values, right = np.linalg.svd(covariance)
    except np.linalg.LinAlgError as exc:
        raise InputValidationError(
            "unable to estimate face alignment transform"
        ) from exc
    rank = np.linalg.matrix_rank(covariance)
    if rank == 0:
        raise InputValidationError("unable to estimate face alignment transform")
    if rank == 1:
        if np.linalg.det(left) * np.linalg.det(right) > 0:
            rotation = left @ right
        else:
            previous_reflection = reflection[-1]
            reflection[-1] = -1
            rotation = left @ np.diag(reflection) @ right
            reflection[-1] = previous_reflection
    else:
        rotation = left @ np.diag(reflection) @ right
    source_variance = source_centered.var(axis=0).sum()
    if source_variance <= 0:
        raise InputValidationError("unable to estimate face alignment transform")
    scale = float(singular_values @ reflection / source_variance)
    matrix = np.empty((2, 3), dtype=np.float64)
    matrix[:, :2] = rotation * scale
    matrix[:, 2] = destination_mean - matrix[:, :2] @ source_mean
    if not np.isfinite(matrix).all():
        raise InputValidationError("face alignment transform must be finite")
    affine_scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))
    if not MIN_SIMILARITY_SCALE <= affine_scale <= MAX_SIMILARITY_SCALE:
        raise InputValidationError(
            "face alignment transform scale must be within "
            f"[{MIN_SIMILARITY_SCALE}, {MAX_SIMILARITY_SCALE}]"
        )
    return matrix


def prepare_affine(face_image, landmark):
    """Validate one sample and estimate its alignment on the CPU."""
    if not isinstance(face_image, np.ndarray):
        raise InputValidationError("RAW_FACE_IMAGE must be a numpy array")
    if face_image.dtype != np.uint8:
        raise InputValidationError("RAW_FACE_IMAGE must have dtype uint8")
    if face_image.ndim != 3 or face_image.shape[2] != 3:
        raise InputValidationError(
            "RAW_FACE_IMAGE sample must have shape [H,W,3]"
        )
    if face_image.shape[0] <= 0 or face_image.shape[1] <= 0:
        raise InputValidationError(
            "RAW_FACE_IMAGE height and width must be positive"
        )
    height, width = face_image.shape[:2]
    if height > MAX_SIDE or width > MAX_SIDE:
        raise InputValidationError(
            f"RAW_FACE_IMAGE exceeds max side {MAX_SIDE}"
        )
    if height * width > MAX_PIXELS:
        raise InputValidationError(
            f"RAW_FACE_IMAGE exceeds max pixels {MAX_PIXELS}"
        )
    if not isinstance(landmark, np.ndarray):
        raise InputValidationError("LANDMARK must be a numpy array")
    if landmark.dtype != np.float32:
        raise InputValidationError("LANDMARK must have dtype float32")
    if landmark.shape != (5, 2):
        raise InputValidationError("LANDMARK sample must have shape [5,2]")
    if not np.isfinite(landmark).all():
        raise InputValidationError("LANDMARK values must be finite")

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
    rms_spread = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    if rms_spread < MIN_LANDMARK_RMS_SPREAD:
        raise InputValidationError(
            "LANDMARK RMS spread for alignment transform must be at least "
            f"{MIN_LANDMARK_RMS_SPREAD} px"
        )

    return _alignment_matrix(landmark)


def preprocess(face_image, landmark):
    """CPU reference: preserve its division/subtraction normalization order."""
    matrix = prepare_affine(face_image, landmark)
    aligned = cv2.warpAffine(face_image, matrix, (112, 112), borderValue=0.0)
    image = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (112, 112))
    image = (image.astype(np.float32) / 255.0 - 0.5) / 0.5
    return np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)


def _validate_batch(images, landmarks):
    if not isinstance(images, np.ndarray):
        raise InputValidationError("RAW_FACE_IMAGE must be a numpy array")
    if images.ndim != 4 or images.shape[3] != 3:
        raise InputValidationError(
            "RAW_FACE_IMAGE batch must have shape [B,H,W,3]"
        )
    if images.shape[0] == 0:
        raise InputValidationError("RAW_FACE_IMAGE batch must be non-empty")
    if not isinstance(landmarks, np.ndarray):
        raise InputValidationError("LANDMARK must be a numpy array")
    if landmarks.ndim != 3 or landmarks.shape[1:] != (5, 2):
        raise InputValidationError("LANDMARK batch must have shape [B,5,2]")
    if images.shape[0] != landmarks.shape[0]:
        raise InputValidationError(
            "RAW_FACE_IMAGE and LANDMARK batch size must match"
        )


class TritonPythonModel:
    def initialize(self, args):
        config = args.get("model_config", {})
        if isinstance(config, str):
            config = json.loads(config)
        parameters = config.get("parameters", {})
        backend = parameters.get("preprocess_backend", {}).get("string_value", "cpu")
        if backend not in ("cpu", "orin_gpu"):
            raise ValueError("preprocess_backend must be cpu or orin_gpu")
        self.gpu = None
        if backend == "orin_gpu":
            if args.get("model_instance_kind") != "GPU":
                raise RuntimeError("orin_gpu preprocessing requires KIND_GPU")
            sys.path.insert(0, "/usr/local/bin")
            from face_orin_preprocess import FaceGpuPreprocessor
            self.gpu = FaceGpuPreprocessor(
                int(args.get("model_instance_device_id", 0)), "qa"
            )

    def execute(self, requests):
        if getattr(self, "gpu", None) is not None:
            return self.gpu.execute(
                requests, pb_utils, prepare_affine, _validate_batch,
                InputValidationError, logger,
            )
        responses = []
        for request_index, request in enumerate(requests):
            try:
                tensor = pb_utils.get_input_tensor_by_name(
                    request, "RAW_FACE_IMAGE"
                )
                if tensor is None:
                    raise InputValidationError(
                        "missing required input RAW_FACE_IMAGE"
                    )
                landmark_tensor = pb_utils.get_input_tensor_by_name(
                    request, "LANDMARK"
                )
                if landmark_tensor is None:
                    raise InputValidationError(
                        "missing required input LANDMARK"
                    )
                images = tensor.as_numpy()
                landmarks = landmark_tensor.as_numpy()
                _validate_batch(images, landmarks)

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
                            "face_qa_pre internal failure "
                            "request_index=%d batch_index=%d incident=%s",
                            request_index,
                            batch_index,
                            incident,
                        )
                        responses.append(
                            pb_utils.InferenceResponse(
                                output_tensors=[],
                                error=pb_utils.TritonError(
                                    "INTERNAL_ERROR: face_qa_pre failed; "
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
                            output_tensors=[
                                pb_utils.Tensor(
                                    "QA_PREPROCESSED_IMAGE", output
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
                    "face_qa_pre internal failure "
                    "request_index=%d incident=%s",
                    request_index,
                    incident,
                )
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(
                            "INTERNAL_ERROR: face_qa_pre failed; "
                            f"incident={incident}"
                        ),
                    )
                )
        return responses

    def finalize(self):
        print("Cleaning up face_qa_pre...")
