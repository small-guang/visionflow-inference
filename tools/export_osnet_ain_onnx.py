"""Export the official OSNet-AIN x1.0 MSMT17 checkpoint to normalized ONNX."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torch import nn

try:
    from .onnx_export_validation import publish_onnx_atomically
except ImportError:
    from onnx_export_validation import publish_onnx_atomically


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_NAME = (
    "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_"
    "coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
)
TRUSTED_CHECKPOINT_SHA256 = (
    "8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f"
)
DEFAULT_CHECKPOINT = ROOT / "tools" / "models" / "osnet" / CHECKPOINT_NAME
DEFAULT_OUTPUT = (
    ROOT / "model_repository" / "osnet_ain_x1_0_onnx" / "1" / "model.onnx"
)
EXPECTED_NUM_CLASSES = 4101
EXPECTED_EMBEDDING_SIZE = 512
OPSET_VERSION = 17
MINIMUM_COSINE_SIMILARITY = 0.9999
VENDORED_ARCHITECTURE = ROOT / "tools" / "vendor" / "torchreid" / "osnet_ain.py"


class NormalizedOSNet(nn.Module):
    def __init__(self, model: nn.Module, embedding_size: int | None = None):
        super().__init__()
        self.model = model
        self.embedding_size = embedding_size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(self.model(image), p=2, dim=1)
        if self.embedding_size is None:
            return normalized
        return normalized.reshape(image.shape[0], self.embedding_size)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path, expected_sha256: str):
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256.strip().lower():
        raise RuntimeError(
            f"Checkpoint SHA256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state dictionary")
    return {
        name.removeprefix("module."): value for name, value in checkpoint.items()
    }


def validate_checkpoint_identity(
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    weight = state_dict.get("classifier.weight")
    bias = state_dict.get("classifier.bias")
    if weight is None or tuple(weight.shape) != (
        EXPECTED_NUM_CLASSES,
        EXPECTED_EMBEDDING_SIZE,
    ):
        raise RuntimeError(
            "Expected MSMT17 combineall classifier shape (4101, 512)"
        )
    if bias is None or tuple(bias.shape) != (EXPECTED_NUM_CLASSES,):
        raise RuntimeError(
            "Expected MSMT17 combineall classifier bias shape (4101,)"
        )


def load_architecture_module():
    spec = importlib.util.spec_from_file_location(
        "_vendored_torchreid_osnet_ain", VENDORED_ARCHITECTURE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load architecture: {VENDORED_ARCHITECTURE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(state_dict: Mapping[str, torch.Tensor]) -> NormalizedOSNet:
    validate_checkpoint_identity(state_dict)
    factory = load_architecture_module().osnet_ain_x1_0
    model = factory(
        num_classes=EXPECTED_NUM_CLASSES,
        pretrained=False,
        loss="softmax",
    ).eval()
    model.load_state_dict(state_dict, strict=True)
    return NormalizedOSNet(model, EXPECTED_EMBEDDING_SIZE).eval()


def validate_torch_output(
    wrapper: nn.Module,
    example: torch.Tensor,
    embedding_size: int,
) -> np.ndarray:
    with torch.no_grad():
        output = wrapper(example)
    expected_shape = (example.shape[0], embedding_size)
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected PyTorch output: expected {expected_shape}, "
            f"got {tuple(output.shape)}"
        )
    if output.dtype != torch.float32 or not torch.isfinite(output).all():
        raise RuntimeError("PyTorch output must be finite FP32")
    if not torch.allclose(
        output.norm(p=2, dim=1),
        torch.ones(example.shape[0]),
        rtol=0.0,
        atol=1e-5,
    ):
        raise RuntimeError("PyTorch output is not L2-normalized")
    return output.cpu().numpy()


def graph_shape(value_info) -> tuple[str | int, ...]:
    shape = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.dim_param:
            shape.append(dimension.dim_param)
        else:
            shape.append(dimension.dim_value)
    return tuple(shape)


def validate_candidate(
    candidate_path: Path,
    wrapper: nn.Module,
    examples: tuple[torch.Tensor, ...],
    embedding_size: int = EXPECTED_EMBEDDING_SIZE,
) -> None:
    model = onnx.load(str(candidate_path))
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise RuntimeError("OSNet ONNX must have exactly one input and one output")
    input_info = model.graph.input[0]
    output_info = model.graph.output[0]
    if input_info.name != "image" or graph_shape(input_info) != (
        "batch",
        3,
        256,
        128,
    ):
        raise RuntimeError("Unexpected OSNet ONNX input signature")
    if output_info.name != "embedding" or graph_shape(output_info) != (
        "batch",
        embedding_size,
    ):
        raise RuntimeError("Unexpected OSNet ONNX output signature")
    if input_info.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise RuntimeError("OSNet ONNX input must be FLOAT")
    if output_info.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise RuntimeError("OSNet ONNX output must be FLOAT")

    session = ort.InferenceSession(
        str(candidate_path), providers=["CPUExecutionProvider"]
    )
    for example in examples:
        expected = validate_torch_output(wrapper, example, embedding_size)
        actual = session.run(
            ["embedding"], {"image": example.cpu().numpy()}
        )[0]
        if actual.shape != expected.shape or actual.dtype != np.float32:
            raise RuntimeError("Unexpected ONNX Runtime output shape or dtype")
        if not np.isfinite(actual).all():
            raise RuntimeError("ONNX Runtime output contains non-finite values")
        norms = np.linalg.norm(actual, axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-5):
            raise RuntimeError(f"ONNX Runtime output is not normalized: {norms}")
        cosine = np.sum(expected * actual, axis=1) / (
            np.linalg.norm(expected, axis=1) * norms
        )
        if float(np.min(cosine)) < MINIMUM_COSINE_SIMILARITY:
            raise RuntimeError(
                f"PyTorch/ONNX cosine similarity is too low: {cosine}"
            )


def export_model(
    wrapper: nn.Module,
    output_path: Path,
    embedding_size: int = EXPECTED_EMBEDDING_SIZE,
) -> str:
    output_path = Path(output_path)
    generator = torch.Generator().manual_seed(20260805)
    examples = (
        torch.randn((1, 3, 256, 128), generator=generator),
        torch.randn((4, 3, 256, 128), generator=generator),
    )

    def build_and_validate(candidate_path: Path) -> None:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                examples[0],
                str(candidate_path),
                input_names=["image"],
                output_names=["embedding"],
                dynamic_axes={
                    "image": {0: "batch"},
                    "embedding": {0: "batch"},
                },
                opset_version=OPSET_VERSION,
                do_constant_folding=True,
                dynamo=False,
            )
        validate_candidate(candidate_path, wrapper, examples, embedding_size)

    return publish_onnx_atomically(output_path, build_and_validate)


def export_release(
    wrapper: nn.Module,
    output_path: Path,
    embedding_size: int = EXPECTED_EMBEDDING_SIZE,
) -> str:
    return export_model(wrapper, output_path, embedding_size)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export OSNet-AIN x1.0 MSMT17 to validated ONNX."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expected-sha256",
        default=TRUSTED_CHECKPOINT_SHA256,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing OSNet-AIN checkpoint: {checkpoint}")
    state_dict = load_checkpoint(checkpoint, args.expected_sha256)
    wrapper = build_model(state_dict)
    digest = export_release(wrapper, output, EXPECTED_EMBEDDING_SIZE)
    print(f"Checkpoint SHA256: {file_sha256(checkpoint)}")
    print(f"ONNX SHA256: {digest}")
    print("Input: image [batch,3,256,128] FLOAT")
    print("Output: embedding [batch,512] FLOAT, L2-normalized")
    print(f"Exported: {output}")


if __name__ == "__main__":
    main()
