import argparse
import time
from pathlib import Path

import numpy as np

try:
    import tritonclient.http as httpclient
except ModuleNotFoundError:
    httpclient = None


MODEL_NAME = "sbs_r50_ibn_image_ensemble"
EMBEDDING_SIZE = 2048


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run and validate SBS R50-IBN image inference through Triton."
    )
    parser.add_argument("--image", required=True, help="Path to a JPEG or PNG image")
    parser.add_argument("--url", default="localhost:8000", help="Triton HTTP endpoint")
    parser.add_argument("--model", default=MODEL_NAME, help="Triton ensemble name")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        choices=(1,),
        metavar="1",
        help="This single-image smoke client uses B=1; the server supports B1-4",
    )
    parser.add_argument(
        "--warmup", type=int, default=0, help="Number of untimed warm-up requests"
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="Number of timed requests"
    )
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    return args


def make_input(image_path, batch_size):
    if batch_size != 1:
        raise ValueError("this single-image smoke client requires batch_size=1")
    encoded = Path(image_path).read_bytes()
    if not encoded:
        raise ValueError(f"image file is empty: {image_path}")
    return np.array([[encoded]], dtype=object)


def infer(client, model_name, image_batch):
    infer_input = httpclient.InferInput("IMAGE_BYTES", image_batch.shape, "BYTES")
    infer_input.set_data_from_numpy(image_batch)
    result = client.infer(
        model_name,
        inputs=[infer_input],
        outputs=[httpclient.InferRequestedOutput("embedding")],
    )
    embedding = result.as_numpy("embedding")
    if embedding is None:
        raise RuntimeError("Triton response does not contain 'embedding'")
    return embedding


def validate_embedding(embedding):
    expected_shape = (1, EMBEDDING_SIZE)
    if embedding.shape != expected_shape:
        raise AssertionError(
            f"expected embedding shape {expected_shape}, got {embedding.shape}"
        )
    if embedding.dtype != np.float32:
        raise AssertionError(f"expected FP32 embedding, got {embedding.dtype}")
    if not np.isfinite(embedding).all():
        raise AssertionError("embedding contains NaN or infinite values")

    norms = np.linalg.norm(embedding, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-3, atol=1e-3):
        raise AssertionError(f"embedding is not L2-normalized; norms={norms}")
    return embedding[0]


def main():
    args = parse_args()
    if httpclient is None:
        raise RuntimeError(
            "tritonclient is required for inference; install tritonclient[http]"
        )
    image_batch = make_input(args.image, args.batch_size)
    client = httpclient.InferenceServerClient(url=args.url)
    if not client.is_model_ready(args.model):
        raise RuntimeError(f"Triton model is not ready: {args.model}")

    for _ in range(args.warmup):
        validate_embedding(infer(client, args.model, image_batch))

    durations_ms = []
    response = None
    embedding = None
    for _ in range(args.repeat):
        started = time.perf_counter()
        response = infer(client, args.model, image_batch)
        durations_ms.append((time.perf_counter() - started) * 1000.0)
        embedding = validate_embedding(response)
        norm = np.linalg.norm(embedding)

    print(f"model: {args.model}")
    print(f"batch_size: {args.batch_size}")
    print(f"response_shape: {response.shape}")
    print(f"selected_embedding_shape: {embedding.shape}")
    print(f"embedding_dtype: {embedding.dtype}")
    print(f"embedding_norm: {norm:.6f}")
    print(f"finite: {bool(np.isfinite(embedding).all())}")
    print(f"requests: {args.repeat}")
    print(f"latency_ms_mean: {np.mean(durations_ms):.2f}")
    print(f"latency_ms_min: {np.min(durations_ms):.2f}")
    print(f"latency_ms_max: {np.max(durations_ms):.2f}")


if __name__ == "__main__":
    main()
