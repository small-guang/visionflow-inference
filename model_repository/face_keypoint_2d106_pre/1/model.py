import json
import logging
import sys
import uuid

import cv2
import numpy as np
from skimage import transform as trans
import triton_python_backend_utils as pb_utils


MAX_SIDE = 8192
MAX_PIXELS = 8_388_608
BBOX_MARGIN_RATIO = 0.25
MIN_BBOX_SIDE = 1.0
MIN_SIMILARITY_SCALE = 1.0 / 256.0
MAX_SIMILARITY_SCALE = 256.0
logger = logging.getLogger("face_keypoint_2d106_pre")


class InputValidationError(ValueError):
    pass


def _transform_matrix(center, output_size, scale, rotation=0):
    rot = float(rotation) * np.pi / 180.0
    t1 = trans.SimilarityTransform(scale=scale)
    cx = center[0] * scale
    cy = center[1] * scale
    t2 = trans.SimilarityTransform(translation=(-cx, -cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(
        translation=(output_size / 2, output_size / 2)
    )
    transform = t1 + t2 + t3 + t4
    return transform.params[0:2]


def transform_face(data, center, output_size, scale, rotation=0):
    """Crop and align one face using an affine transform."""
    matrix = _transform_matrix(center, output_size, scale, rotation)
    cropped = cv2.warpAffine(
        data, matrix, (output_size, output_size), borderValue=0.0
    )
    return cropped, matrix


def _validate_sample(image, bbox, *, input_size):
    """Validate one sample and calculate its center and scale on the CPU."""
    if not isinstance(image, np.ndarray):
        raise InputValidationError("RAW_IMAGE must be a numpy array")
    if image.dtype != np.uint8:
        raise InputValidationError("RAW_IMAGE must have dtype uint8")
    if image.ndim != 3 or image.shape[2] != 3:
        raise InputValidationError("RAW_IMAGE sample must have shape [H,W,3]")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise InputValidationError("RAW_IMAGE dimensions must be positive")
    image_height, image_width = image.shape[:2]
    if image_height > MAX_SIDE or image_width > MAX_SIDE:
        raise InputValidationError(f"RAW_IMAGE exceeds max side {MAX_SIDE}")
    if image_height * image_width > MAX_PIXELS:
        raise InputValidationError(
            f"RAW_IMAGE exceeds max pixels {MAX_PIXELS}"
        )

    bbox = np.asarray(bbox)
    if bbox.shape != (4,):
        raise InputValidationError("BBOX sample must have shape [4]")
    if bbox.dtype != np.float32:
        raise InputValidationError("BBOX must have dtype float32")
    if not np.isfinite(bbox).all():
        raise InputValidationError("BBOX values must be finite")

    x1, y1, x2, y2 = bbox
    width = float(x2 - x1)
    height = float(y2 - y1)
    if width < MIN_BBOX_SIDE or height < MIN_BBOX_SIDE:
        raise InputValidationError(
            "BBOX width and height must be at least 1 px"
        )
    if (
        x1 < -BBOX_MARGIN_RATIO * image_width
        or x2 > (1.0 + BBOX_MARGIN_RATIO) * image_width
        or y1 < -BBOX_MARGIN_RATIO * image_height
        or y2 > (1.0 + BBOX_MARGIN_RATIO) * image_height
    ):
        raise InputValidationError(
            "BBOX coordinates exceed the 25% image margin"
        )
    center = (
        (float(x2) + float(x1)) / 2.0,
        (float(y2) + float(y1)) / 2.0,
    )
    scale = input_size / (max(width, height) * 1.5)
    if not np.isfinite(scale):
        raise InputValidationError("BBOX transform scale must be finite")
    if not MIN_SIMILARITY_SCALE <= scale <= MAX_SIMILARITY_SCALE:
        raise InputValidationError(
            "BBOX transform scale must be within "
            f"[{MIN_SIMILARITY_SCALE}, {MAX_SIMILARITY_SCALE}]"
        )

    return center, scale


def prepare_affine(image, bbox, *, input_size=192):
    center, scale = _validate_sample(image, bbox, input_size=input_size)
    matrix = np.asarray(_transform_matrix(center, input_size, scale), dtype=np.float64)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise InputValidationError("face transform must produce a finite [2,3] affine matrix")
    return np.ascontiguousarray(matrix)


def preprocess(image, bbox, *, input_size, input_mean, input_std):
    """CPU reference: preserve warpAffine and blobFromImage arithmetic."""
    center, scale = _validate_sample(image, bbox, input_size=input_size)
    aligned_image, matrix = transform_face(
        image, center, input_size, scale, rotation=0
    )
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (2, 3):
        raise InputValidationError(
            "face transform must produce a [2,3] affine matrix"
        )
    if not np.isfinite(matrix).all():
        raise InputValidationError("affine matrix values must be finite")

    blob = cv2.dnn.blobFromImage(
        aligned_image,
        1.0 / input_std,
        (input_size, input_size),
        (input_mean, input_mean, input_mean),
        swapRB=True,
    )
    return (
        np.ascontiguousarray(blob[0], dtype=np.float32),
        np.ascontiguousarray(matrix, dtype=np.float64),
    )


def _validate_batch_inputs(images, boxes):
    if not isinstance(images, np.ndarray):
        raise InputValidationError("RAW_IMAGE must be a numpy array")
    if images.ndim != 4 or images.shape[3] != 3:
        raise InputValidationError(
            "RAW_IMAGE batch must have shape [B,H,W,3]"
        )
    if not isinstance(boxes, np.ndarray):
        raise InputValidationError("BBOX must be a numpy array")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise InputValidationError("BBOX batch must have shape [B,4]")
    if boxes.dtype != np.float32:
        raise InputValidationError("BBOX must have dtype float32")
    if images.shape[0] == 0:
        raise InputValidationError("input batch must be non-empty")
    if images.shape[0] != boxes.shape[0]:
        raise InputValidationError(
            "RAW_IMAGE and BBOX batch size must match"
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
                int(args.get("model_instance_device_id", 0)), "keypoint"
            )
        self.input_size = 192
        # ONNX graph contains its own Sub/Mul normalization nodes.
        self.input_mean = 0.0
        self.input_std = 1.0

    def execute(self, requests):
        if getattr(self, "gpu", None) is not None:
            return self.gpu.execute(
                requests, pb_utils, prepare_affine, _validate_batch_inputs,
                InputValidationError, logger,
            )
        responses = []
        for request_index, request in enumerate(requests):
            try:
                image_tensor = pb_utils.get_input_tensor_by_name(
                    request, "RAW_IMAGE"
                )
                bbox_tensor = pb_utils.get_input_tensor_by_name(
                    request, "BBOX"
                )
                if image_tensor is None:
                    raise InputValidationError(
                        "missing required input RAW_IMAGE"
                    )
                if bbox_tensor is None:
                    raise InputValidationError("missing required input BBOX")

                images = image_tensor.as_numpy()
                boxes = bbox_tensor.as_numpy()
                _validate_batch_inputs(images, boxes)

                processed_images = []
                affine_matrices = []
                for batch_index, (image, bbox) in enumerate(
                    zip(images, boxes)
                ):
                    try:
                        processed, matrix = preprocess(
                            image,
                            bbox,
                            input_size=self.input_size,
                            input_mean=self.input_mean,
                            input_std=self.input_std,
                        )
                    except InputValidationError as exc:
                        raise InputValidationError(
                            f"batch index {batch_index}: {exc}"
                        ) from exc
                    except Exception:
                        incident = uuid.uuid4().hex[:12]
                        logger.exception(
                            "face_keypoint_2d106_pre internal failure "
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
                                    "face_keypoint_2d106_pre failed; "
                                    f"incident={incident}"
                                ),
                            )
                        )
                        break
                    processed_images.append(processed)
                    affine_matrices.append(matrix)
                else:
                    image_batch = np.ascontiguousarray(
                        np.stack(processed_images, axis=0), dtype=np.float32
                    )
                    matrix_batch = np.ascontiguousarray(
                        np.stack(affine_matrices, axis=0), dtype=np.float64
                    )
                    responses.append(
                        pb_utils.InferenceResponse(
                            output_tensors=[
                                pb_utils.Tensor(
                                    "PREPROCESSED_IMAGE", image_batch
                                ),
                                pb_utils.Tensor(
                                    "AFFINE_MATRIX", matrix_batch
                                ),
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
                    "face_keypoint_2d106_pre internal failure "
                    "request_index=%d incident=%s",
                    request_index,
                    incident,
                )
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(
                            "INTERNAL_ERROR: "
                            "face_keypoint_2d106_pre failed; "
                            f"incident={incident}"
                        ),
                    )
                )
        return responses

    def finalize(self):
        print("face_keypoint_2d106_pre finalized.", flush=True)
