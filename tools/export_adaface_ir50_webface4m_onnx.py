"""Export the official AdaFace IR-50 WebFace4M checkpoint to ONNX."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

try:
    from .onnx_export_validation import (
        publish_file_atomically,
        publish_onnx_atomically,
    )
except ImportError:
    from onnx_export_validation import (
        publish_file_atomically,
        publish_onnx_atomically,
    )


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_NAME = "adaface_ir50_webface4m.ckpt"
DEFAULT_CHECKPOINT = ROOT / "tools" / "models" / "adaface" / CHECKPOINT_NAME
DEFAULT_IDENTITY = ROOT / "tools" / "adaface_ir50_webface4m_identity.json"
DEFAULT_OUTPUT = (
    ROOT
    / "model_repository"
    / "adaface_ir50_webface4m_onnx"
    / "1"
    / "model.onnx"
)
VENDORED_ARCHITECTURE = ROOT / "tools" / "vendor" / "adaface" / "net.py"
OFFICIAL_REPOSITORY = "https://github.com/mk-minchul/AdaFace"
OFFICIAL_FILE_ID = "1BmDRrhPsHSbXcWZoYFPJg2KJn1sd3QpN"
EXPECTED_EMBEDDING_SIZE = 512
OPSET_VERSION = 17
PARITY_RTOL = 1e-4
PARITY_ATOL = 1e-5
NORM_ATOL = 1e-5


class BatchedAdaFace(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        embedding, _ = self.model(image)
        return embedding


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs):
    identity = {}
    for key, value in pairs:
        if key in identity:
            raise RuntimeError(f"Identity JSON contains duplicate key: {key}")
        identity[key] = value
    return identity


def load_identity(path: Path) -> dict[str, object]:
    try:
        identity = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unable to load identity file: {path}") from error

    if not isinstance(identity, Mapping):
        raise RuntimeError("Identity JSON must contain an object")

    expected_fields = {
        "schema_version",
        "model",
        "source_repository",
        "source_file_id",
        "checkpoint_filename",
        "checkpoint_size",
        "checkpoint_sha256",
    }
    actual_fields = set(identity)
    if actual_fields != expected_fields:
        raise RuntimeError(
            "Identity fields mismatch: "
            f"expected {sorted(expected_fields)}, got {sorted(actual_fields)}"
        )

    if type(identity["schema_version"]) is not int or identity["schema_version"] != 1:
        raise RuntimeError("Identity schema_version must be 1")
    if identity["model"] != "adaface_ir50_webface4m":
        raise RuntimeError("Identity model must be 'adaface_ir50_webface4m'")
    if identity["source_repository"] != OFFICIAL_REPOSITORY:
        raise RuntimeError(
            f"Identity source_repository must be '{OFFICIAL_REPOSITORY}'"
        )
    if identity["source_file_id"] != OFFICIAL_FILE_ID:
        raise RuntimeError(f"Identity source_file_id must be '{OFFICIAL_FILE_ID}'")
    if identity["checkpoint_filename"] != CHECKPOINT_NAME:
        raise RuntimeError(
            f"Identity checkpoint_filename must be '{CHECKPOINT_NAME}'"
        )

    checkpoint_size = identity["checkpoint_size"]
    if type(checkpoint_size) is not int or checkpoint_size <= 0:
        raise RuntimeError("Identity checkpoint_size must be a positive integer")

    checkpoint_sha256 = identity["checkpoint_sha256"]
    if (
        not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
        or any(
            character not in "0123456789abcdefABCDEF"
            for character in checkpoint_sha256
        )
    ):
        raise RuntimeError(
            "Identity checkpoint_sha256 must be a 64-character hexadecimal string"
        )

    normalized_identity = dict(identity)
    normalized_identity["checkpoint_sha256"] = checkpoint_sha256.lower()
    return normalized_identity


def load_checkpoint(
    path: Path,
    expected_size: int,
    expected_sha256: str,
) -> dict[str, torch.Tensor]:
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"Checkpoint size mismatch: expected {expected_size} bytes, "
            f"got {actual_size} bytes"
        )

    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256.strip().lower():
        raise RuntimeError(
            f"Checkpoint SHA256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )

    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("Checkpoint must contain a mapping")

    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise RuntimeError("Checkpoint state_dict must contain a mapping")

    model_state = {}
    for name, value in state_dict.items():
        if not isinstance(name, str) or not name.startswith("model."):
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Checkpoint model parameter {name!r} must be a torch.Tensor"
            )
        model_state[name.removeprefix("model.")] = value
    if not model_state:
        raise RuntimeError("Checkpoint state_dict contains no model parameters")
    return model_state


def load_architecture_module():
    spec = importlib.util.spec_from_file_location(
        "_vendored_adaface_net", VENDORED_ARCHITECTURE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load architecture: {VENDORED_ARCHITECTURE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(state_dict: Mapping[str, torch.Tensor]) -> BatchedAdaFace:
    model = load_architecture_module().build_model("ir_50").eval()
    model.load_state_dict(state_dict, strict=True)
    return BatchedAdaFace(model).eval()


def graph_shape(value_info) -> tuple[int, ...]:
    dimensions = []
    for index, dimension in enumerate(value_info.type.tensor_type.shape.dim):
        if not dimension.HasField("dim_value"):
            raise RuntimeError(
                f"ONNX value {value_info.name!r} dimension {index} must have "
                "a static dim_value"
            )
        dimensions.append(dimension.dim_value)
    return tuple(dimensions)


def graph_dynamic_batch_sample_shape(value_info) -> tuple[int, ...]:
    dimensions = value_info.type.tensor_type.shape.dim
    if not dimensions:
        raise RuntimeError(
            f"ONNX value {value_info.name!r} must have a batch dimension"
        )
    batch_dimension = dimensions[0]
    if batch_dimension.dim_param != "batch" or batch_dimension.HasField("dim_value"):
        raise RuntimeError(
            f"ONNX value {value_info.name!r} dimension 0 must be symbolic 'batch'"
        )
    sample_dimensions = []
    for index, dimension in enumerate(dimensions[1:], start=1):
        if not dimension.HasField("dim_value"):
            raise RuntimeError(
                f"ONNX value {value_info.name!r} dimension {index} must have "
                "a static dim_value"
            )
        sample_dimensions.append(dimension.dim_value)
    return tuple(sample_dimensions)


def validate_torch_output(
    wrapper: nn.Module,
    example: torch.Tensor,
    embedding_size: int,
) -> np.ndarray:
    with torch.no_grad():
        output = wrapper(example)

    if not isinstance(output, torch.Tensor):
        raise RuntimeError("PyTorch output must be a torch.Tensor")
    expected_shape = (example.shape[0], embedding_size)
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"PyTorch output shape must be {expected_shape}, "
            f"got {tuple(output.shape)}"
        )
    if output.dtype != torch.float32:
        raise RuntimeError(
            f"PyTorch output dtype must be torch.float32, got {output.dtype}"
        )
    if not torch.isfinite(output).all().item():
        raise RuntimeError("PyTorch output contains non-finite values")

    norms = torch.linalg.vector_norm(output, dim=1)
    if not torch.allclose(
        norms,
        torch.ones_like(norms),
        rtol=0.0,
        atol=NORM_ATOL,
    ):
        raise RuntimeError(
            "PyTorch output must be L2-normalized per row, "
            f"got norms={norms.detach().cpu().tolist()}"
        )
    return output.detach().cpu().numpy()


def _make_distinct_b2(example: torch.Tensor) -> torch.Tensor:
    second = example.clone()
    flattened = second.reshape(1, -1)
    position = min(1, flattened.shape[1] - 1)
    delta = max(float(second.detach().abs().max().item()), 1.0)
    flattened[0, position] += delta
    return torch.cat((example, second), dim=0)


def validate_candidate(
    candidate_path: Path,
    wrapper: nn.Module,
    example: torch.Tensor,
    embedding_size: int,
) -> None:
    model = onnx.load(candidate_path)
    onnx.checker.check_model(model)

    if len(model.opset_import) != 1 or model.opset_import[0].version != OPSET_VERSION:
        imports = [
            (opset.domain, opset.version)
            for opset in model.opset_import
        ]
        raise RuntimeError(
            f"ONNX model must have exactly opset {OPSET_VERSION}, got {imports}"
        )
    if model.opset_import[0].domain != "":
        raise RuntimeError(
            "ONNX opset must use the default ONNX domain (''), "
            f"got {model.opset_import[0].domain!r}"
        )
    if len(model.graph.input) != 1:
        raise RuntimeError("ONNX model must have exactly one input")
    if len(model.graph.output) != 1:
        raise RuntimeError("ONNX model must have exactly one output")

    input_info = model.graph.input[0]
    output_info = model.graph.output[0]
    if input_info.name != "input":
        raise RuntimeError(f"ONNX input name must be 'input', got {input_info.name!r}")
    if graph_dynamic_batch_sample_shape(input_info) != tuple(example.shape[1:]):
        raise RuntimeError(
            f"ONNX input sample shape must be {tuple(example.shape[1:])}, "
            f"got {graph_dynamic_batch_sample_shape(input_info)}"
        )
    if input_info.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise RuntimeError("ONNX input dtype must be FLOAT")
    if output_info.name != "embedding":
        raise RuntimeError(
            f"ONNX output name must be 'embedding', got {output_info.name!r}"
        )
    if graph_dynamic_batch_sample_shape(output_info) != (embedding_size,):
        raise RuntimeError(
            f"ONNX output sample shape must be {(embedding_size,)}, "
            f"got {graph_dynamic_batch_sample_shape(output_info)}"
        )
    if output_info.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise RuntimeError("ONNX output dtype must be FLOAT")

    session = ort.InferenceSession(
        str(candidate_path),
        providers=["CPUExecutionProvider"],
    )
    validation_batches = (
        example,
        _make_distinct_b2(example),
    )
    for batch in validation_batches:
        batched_expected = validate_torch_output(wrapper, batch, embedding_size)
        with torch.no_grad():
            row_expected = torch.cat(
                [wrapper(row) for row in batch.split(1)],
                dim=0,
            ).detach().cpu().numpy()
        if batch.shape[0] == 2 and np.allclose(
            row_expected[0],
            row_expected[1],
            rtol=PARITY_RTOL,
            atol=PARITY_ATOL,
        ):
            raise RuntimeError(
                "B2 validation samples must produce distinct embeddings"
            )
        if not np.allclose(
            batched_expected,
            row_expected,
            rtol=PARITY_RTOL,
            atol=PARITY_ATOL,
        ):
            max_abs_error = float(np.max(np.abs(batched_expected - row_expected)))
            raise RuntimeError(
                "PyTorch batch output differs from independent B1 outputs: "
                f"max_abs_error={max_abs_error}"
            )

        actual = session.run(
            ["embedding"],
            {"input": batch.cpu().numpy()},
        )[0]
        expected_shape = (batch.shape[0], embedding_size)
        if actual.shape != expected_shape:
            raise RuntimeError(
                f"ONNX Runtime output shape must be {expected_shape}, "
                f"got {actual.shape}"
            )
        if actual.dtype != np.float32:
            raise RuntimeError(
                f"ONNX Runtime output dtype must be float32, got {actual.dtype}"
            )
        if not np.isfinite(actual).all():
            raise RuntimeError("ONNX Runtime output contains non-finite values")
        if not np.allclose(
            actual,
            row_expected,
            rtol=PARITY_RTOL,
            atol=PARITY_ATOL,
        ):
            max_abs_error = float(np.max(np.abs(actual - row_expected)))
            raise RuntimeError(
                f"ONNX numerical mismatch: max_abs_error={max_abs_error}"
            )

        norms = np.linalg.norm(actual, axis=1)
        if not np.allclose(norms, np.ones_like(norms), rtol=0.0, atol=NORM_ATOL):
            raise RuntimeError(
                "ONNX Runtime output must be L2-normalized per row, "
                f"got norms={norms.tolist()}"
            )
    del session


def export_release(
    wrapper: nn.Module,
    output_path: Path,
    embedding_size: int = EXPECTED_EMBEDDING_SIZE,
    input_shape: tuple[int, int, int, int] = (1, 3, 112, 112),
) -> str:
    generator = torch.Generator().manual_seed(20260807)
    example = torch.randn(
        input_shape,
        generator=generator,
        dtype=torch.float32,
    )

    def build_and_validate(candidate_path: Path) -> None:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                example,
                str(candidate_path),
                input_names=["input"],
                output_names=["embedding"],
                opset_version=OPSET_VERSION,
                do_constant_folding=True,
                dynamic_axes={
                    "input": {0: "batch"},
                    "embedding": {0: "batch"},
                },
                dynamo=False,
            )
        validate_candidate(candidate_path, wrapper, example, embedding_size)

    return publish_onnx_atomically(output_path, build_and_validate)


def _require_distinct_paths(named_paths):
    resolved_paths = {}
    seen = {}
    for name, path in named_paths:
        resolved = Path(path).expanduser().resolve()
        if resolved in seen:
            raise RuntimeError(
                "Publication paths must be different: "
                f"{seen[resolved]} and {name} both resolve to {resolved}"
            )
        seen[resolved] = name
        resolved_paths[name] = resolved
    return resolved_paths


def write_official_identity(
    checkpoint_path: Path,
    identity_path: Path = DEFAULT_IDENTITY,
    *,
    confirmed_source_file_id: str,
    overwrite: bool = False,
) -> dict[str, object]:
    """Record TOFU identity after an explicit caller source assertion.

    ``confirmed_source_file_id`` is a TOFU caller assertion; it does not prove
    transport provenance for the checkpoint bytes.
    """
    if confirmed_source_file_id != OFFICIAL_FILE_ID:
        raise ValueError(
            "TOFU caller assertion confirmed_source_file_id must exactly match "
            f"the official file ID: {OFFICIAL_FILE_ID}"
        )

    resolved = _require_distinct_paths(
        (
            ("checkpoint", checkpoint_path),
            ("identity", identity_path),
        )
    )
    checkpoint_path = resolved["checkpoint"]
    identity_path = resolved["identity"]
    if identity_path.exists() and not overwrite:
        raise FileExistsError(
            f"Identity trust root already exists: {identity_path}; "
            "explicit overwrite authorization is required"
        )

    checkpoint_size = checkpoint_path.stat().st_size
    checkpoint_sha256 = file_sha256(checkpoint_path)
    identity = {
        "schema_version": 1,
        "model": "adaface_ir50_webface4m",
        "source_repository": OFFICIAL_REPOSITORY,
        "source_file_id": OFFICIAL_FILE_ID,
        "checkpoint_filename": CHECKPOINT_NAME,
        "checkpoint_size": checkpoint_size,
        "checkpoint_sha256": checkpoint_sha256,
    }

    state_dict = load_checkpoint(
        checkpoint_path,
        expected_size=checkpoint_size,
        expected_sha256=checkpoint_sha256,
    )
    build_model(state_dict)

    def build_identity_candidate(candidate_path):
        candidate_path.write_text(
            json.dumps(identity, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    publish_file_atomically(identity_path, build_identity_candidate)
    return load_identity(identity_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Export the official AdaFace IR-50 WebFace4M checkpoint to ONNX."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the official AdaFace checkpoint.",
    )
    parser.add_argument(
        "--identity",
        type=Path,
        default=DEFAULT_IDENTITY,
        help="Path to the pinned checkpoint identity JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination model.onnx path.",
    )
    parser.add_argument(
        "--record-official-identity",
        action="store_true",
        help=(
            "Record a Trust-On-First-Use identity after checkpoint parsing and "
            "a strict model load succeed. This records a caller assertion and "
            "does not prove transport provenance."
        ),
    )
    parser.add_argument(
        "--confirm-official-file-id",
        help=(
            "Required with --record-official-identity. TOFU caller assertion of "
            "the official Drive file ID; does not prove transport provenance."
        ),
    )
    parser.add_argument(
        "--force-record-official-identity",
        action="store_true",
        help=(
            "Allow record mode to overwrite an existing identity trust root; "
            "valid only with --record-official-identity."
        ),
    )
    arguments = parser.parse_args(argv)
    if arguments.record_official_identity:
        if arguments.confirm_official_file_id is None:
            parser.error(
                "--record-official-identity requires "
                "--confirm-official-file-id"
            )
        if arguments.confirm_official_file_id != OFFICIAL_FILE_ID:
            parser.error(
                "--confirm-official-file-id must exactly match the official "
                f"file ID: {OFFICIAL_FILE_ID}"
            )
    else:
        if arguments.confirm_official_file_id is not None:
            parser.error(
                "--confirm-official-file-id is valid only with "
                "--record-official-identity"
            )
        if arguments.force_record_official_identity:
            parser.error(
                "--force-record-official-identity is valid only with "
                "--record-official-identity"
            )
    return arguments


def main(argv=None) -> None:
    arguments = parse_args(argv)

    if arguments.record_official_identity:
        resolved = _require_distinct_paths(
            (
                ("checkpoint", arguments.checkpoint),
                ("identity", arguments.identity),
            )
        )
        checkpoint_path = resolved["checkpoint"]
        identity_path = resolved["identity"]
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"AdaFace checkpoint not found: {checkpoint_path}")
        identity = write_official_identity(
            checkpoint_path,
            identity_path,
            confirmed_source_file_id=arguments.confirm_official_file_id,
            overwrite=arguments.force_record_official_identity,
        )
        print(json.dumps(identity, indent=2))
        return

    output_path = Path(arguments.output).expanduser().resolve()
    resolved = _require_distinct_paths(
        (
            ("checkpoint", arguments.checkpoint),
            ("identity", arguments.identity),
            ("output", output_path),
        )
    )
    checkpoint_path = resolved["checkpoint"]
    identity_path = resolved["identity"]
    output_path = resolved["output"]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"AdaFace checkpoint not found: {checkpoint_path}")

    identity = load_identity(identity_path)
    state_dict = load_checkpoint(
        checkpoint_path,
        expected_size=identity["checkpoint_size"],
        expected_sha256=identity["checkpoint_sha256"],
    )
    wrapper = build_model(state_dict)
    onnx_digest = export_release(wrapper, output_path)

    print(f"Checkpoint SHA256: {identity['checkpoint_sha256']}")
    print(f"ONNX SHA256: {onnx_digest}")
    print("Input: input [batch,3,112,112] FLOAT, aligned BGR")
    print("Output: embedding [batch,512] FLOAT, per-row L2-normalized")
    print(f"Exported: {output_path}")


if __name__ == "__main__":
    main()
