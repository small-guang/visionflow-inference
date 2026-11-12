import json
import logging
import sys
import numpy as np
import triton_python_backend_utils as pb_utils
import uuid
from skimage import transform as trans
import cv2


class InputValidationError(ValueError):
    pass


LANDMARK_MARGIN_RATIO = 0.25
MIN_LANDMARK_RMS_SPREAD = 1.0
MIN_SIMILARITY_SCALE = 1.0 / 256.0
MAX_SIMILARITY_SCALE = 256.0
logger = logging.getLogger("face_rec_pre")


arcface_dst = np.array([
    [38.2946, 51.6963],  
    [73.5318, 51.5014],  
    [56.0252, 71.7366],  
    [41.5493, 92.3655],  
    [70.7299, 92.2041]   
], dtype=np.float32)

def estimate_norm(lmk, image_size=112, mode='arcface'):
    if lmk.shape != (5, 2):
        raise InputValidationError("landmark sample must have shape [5,2]")
    if not (image_size % 112 == 0 or image_size % 128 == 0):
        raise ValueError("image_size must be divisible by 112 or 128")
    if image_size%112==0:
        ratio = float(image_size)/112.0
        diff_x = 0
    else:
        ratio = float(image_size)/128.0
        diff_x = 8.0*ratio
    dst = arcface_dst * ratio
    dst[:,0] += diff_x
    tform = trans.SimilarityTransform()
    if not tform.estimate(lmk, dst):
        raise InputValidationError("unable to estimate face alignment transform")
    M = tform.params[0:2, :]
    if not np.isfinite(M).all():
        raise InputValidationError(
            "face alignment transform matrix must be finite"
        )
    scale = float(np.hypot(M[0, 0], M[0, 1]))
    if not MIN_SIMILARITY_SCALE <= scale <= MAX_SIMILARITY_SCALE:
        raise InputValidationError(
            "face alignment transform scale must be within "
            f"[{MIN_SIMILARITY_SCALE}, {MAX_SIMILARITY_SCALE}]"
        )
    return M

def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


def _validate_sample(image, landmark):
    """Validate one face without changing its existing alignment policy."""
    if not isinstance(image, np.ndarray):
        raise InputValidationError("raw_image must be a numpy array")
    if image.dtype != np.uint8:
        raise InputValidationError("raw_image must have dtype uint8")
    if image.ndim != 3 or image.shape[2] != 3:
        raise InputValidationError("raw_image sample must have shape [H,W,3]")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise InputValidationError("raw_image height and width must be positive")
    if not isinstance(landmark, np.ndarray):
        raise InputValidationError("landmark must be a numpy array")
    if landmark.dtype != np.float32:
        raise InputValidationError("landmark must have dtype float32")
    if landmark.shape != (5, 2):
        raise InputValidationError("landmark sample must have shape [5,2]")
    if not np.isfinite(landmark).all():
        raise InputValidationError("landmark values must be finite")

    height, width = image.shape[:2]
    x_coordinates = landmark[:, 0]
    y_coordinates = landmark[:, 1]
    if (
        np.any(x_coordinates < -LANDMARK_MARGIN_RATIO * width)
        or np.any(
            x_coordinates > (1.0 + LANDMARK_MARGIN_RATIO) * width
        )
        or np.any(y_coordinates < -LANDMARK_MARGIN_RATIO * height)
        or np.any(
            y_coordinates > (1.0 + LANDMARK_MARGIN_RATIO) * height
        )
    ):
        raise InputValidationError(
            "landmark coordinates exceed the 25% image margin"
        )

    landmark_float64 = landmark.astype(np.float64, copy=False)
    centered = landmark_float64 - landmark_float64.mean(axis=0)
    rms_spread = float(
        np.sqrt(np.mean(np.sum(centered * centered, axis=1)))
    )
    if rms_spread < MIN_LANDMARK_RMS_SPREAD:
        raise InputValidationError(
            "landmark RMS spread for alignment transform must be at least "
            f"{MIN_LANDMARK_RMS_SPREAD} px"
        )


def prepare_affine(image, landmark, image_size=112):
    _validate_sample(image, landmark)
    return np.ascontiguousarray(estimate_norm(landmark, image_size), dtype=np.float64)


def preprocess(image, landmark, image_size=112):
    """CPU reference: retain OpenCV's subtract-then-multiply normalization."""
    _validate_sample(image, landmark)
    aligned_face = norm_crop(image, landmark, image_size=image_size, mode='arcface')
    blob = cv2.dnn.blobFromImages(
        [aligned_face],
        1.0 / 127.5,
        (image_size, image_size),
        (127.5, 127.5, 127.5),
        swapRB=True,
        crop=False,
    )
    return np.ascontiguousarray(blob[0], dtype=np.float32)


def _validate_batch_inputs(images, landmarks):
    if not isinstance(images, np.ndarray):
        raise InputValidationError("raw_image must be a numpy array")
    if images.ndim != 4 or images.shape[3] != 3:
        raise InputValidationError(
            "raw_image batch must have shape [B,H,W,3]; "
            "each sample must have shape [H,W,3]"
        )
    if not isinstance(landmarks, np.ndarray):
        raise InputValidationError("landmark must be a numpy array")
    if landmarks.ndim != 3 or landmarks.shape[1:] != (5, 2):
        raise InputValidationError(
            "landmark batch must have shape [B,5,2]; "
            "each sample must have shape [5,2]"
        )
    if images.shape[0] == 0:
        raise InputValidationError("input batch must be non-empty")
    if images.shape[0] != landmarks.shape[0]:
        raise InputValidationError(
            "raw_image and landmark batch size must match"
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
                int(args.get("model_instance_device_id", 0)), "recognition"
            )

        self.input_mean = 127.5
        self.input_std = 127.5
        self.output_size = 112

    def execute(self, requests):
        if getattr(self, "gpu", None) is not None:
            return self.gpu.execute(
                requests, pb_utils, prepare_affine, _validate_batch_inputs,
                InputValidationError, logger,
            )
        responses = []
        for request_index, request in enumerate(requests):
            try:
                img_tensor = pb_utils.get_input_tensor_by_name(request, "raw_image")
                landmark_tensor = pb_utils.get_input_tensor_by_name(request, "landmark")
                if img_tensor is None:
                    raise InputValidationError("missing required input raw_image")
                if landmark_tensor is None:
                    raise InputValidationError("missing required input landmark")

                images = img_tensor.as_numpy()
                landmarks = landmark_tensor.as_numpy()
                _validate_batch_inputs(images, landmarks)
                processed = []
                for batch_index, (image, landmark) in enumerate(
                    zip(images, landmarks)
                ):
                    try:
                        processed.append(
                            preprocess(image, landmark, self.output_size)
                        )
                    except InputValidationError as exc:
                        raise InputValidationError(
                            f"batch index {batch_index}: {exc}"
                        ) from exc
                    except Exception:
                        incident = uuid.uuid4().hex[:12]
                        logger.exception(
                            "face_rec_pre internal failure "
                            "request_index=%d batch_index=%d incident=%s",
                            request_index,
                            batch_index,
                            incident,
                        )
                        responses.append(
                            pb_utils.InferenceResponse(
                                output_tensors=[],
                                error=pb_utils.TritonError(
                                    "INTERNAL_ERROR: face_rec_pre failed; "
                                    f"incident={incident}"
                                ),
                            )
                        )
                        break
                else:
                    processed_input = np.ascontiguousarray(
                        np.stack(processed, axis=0), dtype=np.float32
                    )
                    out_tensor = pb_utils.Tensor(
                        "processed_input", processed_input
                    )
                    responses.append(
                        pb_utils.InferenceResponse(
                            output_tensors=[out_tensor]
                        )
                    )
                    continue
                continue
            except InputValidationError as e:
                error_message = f"INVALID_INPUT: {e}"
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(error_message)
                ))
            except Exception:
                incident = uuid.uuid4().hex[:12]
                logger.exception(
                    "face_rec_pre internal failure "
                    "request_index=%d incident=%s",
                    request_index,
                    incident,
                )
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(
                        "INTERNAL_ERROR: face_rec_pre failed; "
                        f"incident={incident}"
                    )
                ))
        return responses

    def finalize(self):
        print("face_rec_pre finalized.", flush=True)
