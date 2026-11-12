"""Manual Triton HTTP smoke client for the AdaFace recognition pipeline.

Example:
    python test/test_adaface_image.py --image face.jpg --landmarks face.json

The landmarks file must contain five ``[x, y]`` points in this order: left
eye, right eye, nose tip, left mouth corner, right mouth corner. Coordinates
use the decoded image coordinate system. The URL format is
``host:port[/base]`` without a URL scheme.
"""

import argparse
import importlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


MODEL_NAME = "adaface_recognition_pipeline"
OUTPUT_NAME = "EMBEDDING"
EMBEDDING_SIZE = 512
_LANDMARK_ERROR = "LANDMARK must be finite FP32 JSON with shape [5,2]"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run AdaFace face-recognition inference through Triton HTTP."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to a decodable color image",
    )
    parser.add_argument(
        "--landmarks",
        required=True,
        help=(
            "Path to strict JSON containing five [x, y] points in order: "
            "left eye, right eye, nose tip, left mouth corner, right mouth "
            "corner"
        ),
    )
    parser.add_argument(
        "--url",
        default="localhost:8000",
        help=(
            "Triton HTTP endpoint as host:port[/base] without a URL scheme "
            "(default: localhost:8000)"
        ),
    )
    return parser.parse_args(argv)


def _reject_nonstandard_json_constant(value):
    raise ValueError(f"non-standard JSON number: {value}")


def _load_landmarks(path):
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FileNotFoundError(f"unable to read LANDMARK JSON: {path}") from exc

    try:
        value = json.loads(
            text,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{_LANDMARK_ERROR}: {path}") from exc

    if (
        not isinstance(value, list)
        or len(value) != 5
        or any(not isinstance(point, list) or len(point) != 2 for point in value)
    ):
        raise ValueError(f"{_LANDMARK_ERROR}: {path}")

    fp32_limit = float(np.finfo(np.float32).max)
    for point in value:
        for coordinate in point:
            if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
                raise ValueError(f"{_LANDMARK_ERROR}: {path}")
            try:
                numeric_coordinate = float(coordinate)
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError(f"{_LANDMARK_ERROR}: {path}") from exc
            if (
                not math.isfinite(numeric_coordinate)
                or abs(numeric_coordinate) > fp32_limit
            ):
                raise ValueError(f"{_LANDMARK_ERROR}: {path}")

    landmarks = np.asarray(value, dtype=np.float32)
    if landmarks.shape != (5, 2) or not np.isfinite(landmarks).all():
        raise ValueError(f"{_LANDMARK_ERROR}: {path}")
    return np.ascontiguousarray(landmarks)


def _validate_raw_inputs(image, landmarks):
    if not isinstance(image, np.ndarray):
        raise ValueError("RAW_IMAGE must be a numpy array")
    if image.dtype != np.uint8:
        raise ValueError("RAW_IMAGE must have dtype UINT8")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("RAW_IMAGE must have shape [H,W,3]")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("RAW_IMAGE height and width must be positive")
    if not isinstance(landmarks, np.ndarray):
        raise ValueError(_LANDMARK_ERROR)
    if landmarks.dtype != np.float32 or landmarks.shape != (5, 2):
        raise ValueError(_LANDMARK_ERROR)
    if not np.isfinite(landmarks).all():
        raise ValueError(_LANDMARK_ERROR)
    return np.ascontiguousarray(image), np.ascontiguousarray(landmarks)


def load_inputs(image_path, landmarks_path):
    """Decode a BGR image and load a strict finite FP32 landmark array."""
    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(
            f"unable to decode color image in BGR mode: {image_path}"
        )
    landmarks = _load_landmarks(landmarks_path)
    return _validate_raw_inputs(image, landmarks)


def load_httpclient():
    """Import Triton's optional HTTP client only when an inference is requested."""
    try:
        return importlib.import_module("tritonclient.http")
    except ModuleNotFoundError as exc:
        missing_name = exc.name or ""
        if missing_name == "tritonclient" or missing_name.startswith("tritonclient."):
            raise RuntimeError(
                "tritonclient[http] is required; install it with "
                "`python -m pip install tritonclient[http]`"
            ) from exc
        raise


def validate_embedding(embedding):
    if embedding is None:
        raise RuntimeError(f"Triton response does not contain {OUTPUT_NAME!r}")
    if not isinstance(embedding, np.ndarray):
        raise AssertionError("EMBEDDING must be a numpy array")
    if embedding.shape != (1, EMBEDDING_SIZE):
        raise AssertionError(
            "expected EMBEDDING shape "
            f"(1, {EMBEDDING_SIZE}), got {embedding.shape}"
        )
    if embedding.dtype != np.float32:
        raise AssertionError(f"expected FP32 EMBEDDING, got {embedding.dtype}")
    if not np.isfinite(embedding).all():
        raise AssertionError("EMBEDDING must contain only finite values")

    norm = float(np.linalg.norm(embedding))
    if not np.isclose(norm, 1.0, rtol=0.0, atol=1e-3):
        raise AssertionError(
            f"EMBEDDING is not L2-normalized within atol=1e-3; norm={norm}"
        )
    return embedding[0]


def run_inference(url, image, landmarks, *, httpclient_module=None):
    """Call the fixed AdaFace ensemble and return its validated embedding."""
    image, landmarks = _validate_raw_inputs(image, landmarks)
    image_batch = np.ascontiguousarray(image[None, ...])
    landmark_batch = np.ascontiguousarray(landmarks[None, ...])
    httpclient = httpclient_module or load_httpclient()
    triton = httpclient.InferenceServerClient(url=url)
    primary_error = None
    try:
        if not triton.is_model_ready(MODEL_NAME):
            raise RuntimeError(f"Triton model is not ready: {MODEL_NAME}")

        raw_image_input = httpclient.InferInput(
            "RAW_IMAGE",
            image_batch.shape,
            "UINT8",
        )
        raw_image_input.set_data_from_numpy(image_batch)
        landmark_input = httpclient.InferInput(
            "LANDMARK",
            landmark_batch.shape,
            "FP32",
        )
        landmark_input.set_data_from_numpy(landmark_batch)
        result = triton.infer(
            MODEL_NAME,
            inputs=[raw_image_input, landmark_input],
            outputs=[httpclient.InferRequestedOutput(OUTPUT_NAME)],
        )
        return validate_embedding(result.as_numpy(OUTPUT_NAME))
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            triton.close()
        except BaseException as close_error:
            if primary_error is None:
                raise
            primary_error.add_note(
                "Triton HTTP client close failed: "
                f"{type(close_error).__name__}: {close_error}"
            )


def main(argv=None):
    args = parse_args(argv)
    image, landmarks = load_inputs(args.image, args.landmarks)
    embedding = run_inference(args.url, image, landmarks)
    norm = float(np.linalg.norm(embedding))
    print(f"model: {MODEL_NAME}")
    print(f"image_shape: {image.shape}")
    print(f"embedding_shape: {embedding.shape}")
    print(f"embedding_dtype: {embedding.dtype}")
    print(f"embedding_norm: {norm:.6f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, AssertionError) as exc:
        raise SystemExit(f"error: {exc}") from exc
