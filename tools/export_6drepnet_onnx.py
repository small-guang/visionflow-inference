from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

if __package__:
    from .onnx_export_validation import (
        file_sha256,
        publish_onnx_atomically,
    )
    from .vendor.sixdrepnet.repvgg import create_RepVGG_B1g2
else:
    from onnx_export_validation import (
        file_sha256,
        publish_onnx_atomically,
    )
    from vendor.sixdrepnet.repvgg import create_RepVGG_B1g2


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_NAME = "6DRepNet_300W_LP_AFLW2000.pth"
CHECKPOINT_SIZE = 157_315_191
CHECKPOINT_SHA256 = "9bdcd0e072df0d21e92288b40ea1d1b3cf7173c44728ab9c45c4d673f361e018"
DEFAULT_CHECKPOINT = ROOT / "tools" / "models" / "6DRepNet" / CHECKPOINT_NAME
DEFAULT_OUTPUT = ROOT / "model_repository" / "sixdrepnet_onnx" / "1" / "model.onnx"
OPSET_VERSION = 17
NORMALIZATION_EPS = 1e-8
COLLINEAR_EPS = 1e-6
CHECKPOINT_HASH_CHUNK_SIZE = 1024 * 1024


def _coordinate_axes(vector: torch.Tensor) -> tuple[torch.Tensor, ...]:
    ones = torch.ones_like(vector[:, :1])
    zeros = torch.zeros_like(vector[:, :1])
    x_axis = torch.cat((ones, zeros, zeros), dim=1)
    y_axis = torch.cat((zeros, ones, zeros), dim=1)
    z_axis = torch.cat((zeros, zeros, ones), dim=1)
    return x_axis, y_axis, z_axis


def _scale_by_row_max(vector: torch.Tensor) -> torch.Tensor:
    max_magnitude = torch.amax(torch.abs(vector), dim=1, keepdim=True)
    safe_scale = torch.where(
        max_magnitude > 0,
        max_magnitude,
        torch.ones_like(max_magnitude),
    )
    return vector / safe_scale


def _normalize_vector(
    vector: torch.Tensor,
    fallback: torch.Tensor,
) -> torch.Tensor:
    scaled = _scale_by_row_max(vector)
    magnitude = torch.sqrt(torch.sum(scaled * scaled, dim=1, keepdim=True))
    is_nonzero = magnitude > NORMALIZATION_EPS
    safe_magnitude = torch.where(is_nonzero, magnitude, torch.ones_like(magnitude))
    normalized = scaled / safe_magnitude
    return torch.where(is_nonzero, normalized, fallback)


def _cross_product(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    x = first[:, 1] * second[:, 2] - first[:, 2] * second[:, 1]
    y = first[:, 2] * second[:, 0] - first[:, 0] * second[:, 2]
    z = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    return torch.stack((x, y, z), dim=1)


def ortho6d_to_rotation_matrix(poses: torch.Tensor) -> torch.Tensor:
    if poses.ndim != 2 or poses.shape[1] != 6:
        raise ValueError(f"poses must have shape [N, 6], got {tuple(poses.shape)}")

    x_axis, y_axis, z_axis = _coordinate_axes(poses[:, :3])
    x = _normalize_vector(poses[:, :3], x_axis)

    second = _scale_by_row_max(poses[:, 3:])
    candidate_z = _cross_product(x, second)
    candidate_magnitude = torch.sqrt(
        torch.sum(candidate_z * candidate_z, dim=1, keepdim=True)
    )

    absolute_x = torch.abs(x)
    use_x_axis = (absolute_x[:, :1] <= absolute_x[:, 1:2]) & (
        absolute_x[:, :1] <= absolute_x[:, 2:3]
    )
    use_y_axis = absolute_x[:, 1:2] <= absolute_x[:, 2:3]
    reference = torch.where(
        use_x_axis,
        x_axis,
        torch.where(use_y_axis, y_axis, z_axis),
    )
    fallback_z = _cross_product(x, reference)
    selected_z = torch.where(
        candidate_magnitude > COLLINEAR_EPS,
        candidate_z,
        fallback_z,
    )
    z = _normalize_vector(selected_z, z_axis)
    y = _cross_product(z, x)
    return torch.stack((x, y, z), dim=2)


class SixDRepNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = create_RepVGG_B1g2(deploy=True)
        self.layer0 = backbone.stage0
        self.layer1 = backbone.stage1
        self.layer2 = backbone.stage2
        self.layer3 = backbone.stage3
        self.layer4 = backbone.stage4
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear_reg = nn.Linear(2048, 6)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.layer0(images)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)
        features = self.gap(features)
        features = torch.flatten(features, 1)
        poses = self.linear_reg(features)
        return ortho6d_to_rotation_matrix(poses)


def load_checkpoint(
    path: str | Path,
    expected_size: int,
    expected_sha256: str,
) -> dict[str, torch.Tensor]:
    path = Path(path)
    with path.open("rb") as checkpoint_handle:
        actual_size = os.fstat(checkpoint_handle.fileno()).st_size
        if actual_size != expected_size:
            raise RuntimeError(
                "Checkpoint identity mismatch: "
                f"expected size={expected_size}; actual size={actual_size}"
            )

        with tempfile.TemporaryFile(mode="w+b") as checkpoint_snapshot:
            digest = hashlib.sha256()
            copied_size = 0
            for chunk in iter(
                lambda: checkpoint_handle.read(CHECKPOINT_HASH_CHUNK_SIZE),
                b"",
            ):
                written_size = checkpoint_snapshot.write(chunk)
                if written_size != len(chunk):
                    raise RuntimeError(
                        "Unable to create complete private checkpoint snapshot: "
                        f"expected write={len(chunk)}; actual write={written_size}"
                    )
                copied_size += written_size
                digest.update(chunk)

            if copied_size != expected_size:
                raise RuntimeError(
                    "Checkpoint identity mismatch: "
                    f"expected copied size={expected_size}; "
                    f"actual copied size={copied_size}"
                )
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    "Checkpoint identity mismatch: "
                    f"expected sha256={expected_sha256}; "
                    f"actual sha256={actual_sha256}"
                )

            checkpoint_snapshot.flush()
            checkpoint_snapshot.seek(0)
            state_dict = torch.load(
                checkpoint_snapshot,
                map_location="cpu",
                weights_only=True,
            )
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint must contain a mapping of tensor names to tensors")
    if not all(isinstance(key, str) for key in state_dict):
        raise TypeError("Checkpoint keys must all be strings")
    if not all(isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise TypeError("Checkpoint values must all be torch.Tensor instances")
    return dict(state_dict)


def build_model(state_dict: Mapping[str, torch.Tensor]) -> SixDRepNet:
    model = SixDRepNet().eval()
    expected_state = model.state_dict()
    expected_keys = set(expected_state)
    actual_keys = set(state_dict)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if len(state_dict) != 58 or missing or unexpected:
        raise RuntimeError(
            "Checkpoint must contain exactly 58 tensors with the model's exact keys: "
            f"got {len(state_dict)}; missing={missing}; unexpected={unexpected}"
        )

    shape_mismatches = [
        f"{key}: expected {tuple(expected_state[key].shape)}, "
        f"got {tuple(state_dict[key].shape)}"
        for key in sorted(expected_keys)
        if tuple(state_dict[key].shape) != tuple(expected_state[key].shape)
    ]
    if shape_mismatches:
        raise RuntimeError(
            "Checkpoint tensor shape mismatch: " + "; ".join(shape_mismatches)
        )

    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def graph_shape(value_info) -> tuple[int | str, ...]:
    """Return a graph value's positive static or non-empty symbolic tensor shape."""
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        raise RuntimeError(f"{value_info.name!r} tensor shape is unset")

    shape = []
    for index, dimension in enumerate(tensor_type.shape.dim):
        dimension_kind = dimension.WhichOneof("value")
        if dimension_kind == "dim_param":
            if not dimension.dim_param:
                raise RuntimeError(
                    f"{value_info.name!r} dimension {index} has an empty symbol"
                )
            shape.append(dimension.dim_param)
            continue
        if dimension_kind != "dim_value":
            raise RuntimeError(
                f"{value_info.name!r} dimension {index} is unset"
            )
        dimension_value = int(dimension.dim_value)
        if dimension_value <= 0:
            raise RuntimeError(
                f"{value_info.name!r} dimension {index} must be positive, "
                f"got {dimension_value}"
            )
        shape.append(dimension_value)
    return tuple(shape)


def validate_rotation_matrices(rotation) -> None:
    """Require finite float32 proper 3D rotation matrices for a non-empty batch."""
    rotation = np.asarray(rotation)
    if rotation.ndim != 3 or rotation.shape[0] <= 0 or rotation.shape[1:] != (3, 3):
        raise RuntimeError(
            f"rotation matrix shape must be [B, 3, 3] with B > 0, got {rotation.shape}"
        )
    if rotation.dtype != np.dtype(np.float32):
        raise RuntimeError(
            f"rotation matrix dtype must be float32, got {rotation.dtype}"
        )

    finite_by_batch = np.isfinite(rotation).all(axis=(1, 2))
    if not finite_by_batch.all():
        batch_index = int(np.flatnonzero(~finite_by_batch)[0])
        raise RuntimeError(
            "rotation matrix must contain only finite values: "
            f"batch index {batch_index}"
        )

    identity = np.eye(3, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        orthogonality_errors = np.max(
            np.abs(np.swapaxes(rotation, 1, 2) @ rotation - identity),
            axis=(1, 2),
        )
    invalid_orthogonality = (
        ~np.isfinite(orthogonality_errors) | (orthogonality_errors > 1e-3)
    )
    if invalid_orthogonality.any():
        batch_index = int(np.flatnonzero(invalid_orthogonality)[0])
        raise RuntimeError(
            "rotation matrix orthogonality error exceeds 1e-3: "
            f"batch index {batch_index}; "
            f"max_abs_error={orthogonality_errors[batch_index]}"
        )

    determinants = np.linalg.det(rotation)
    determinant_errors = np.abs(determinants - 1.0)
    invalid_determinants = (
        ~np.isfinite(determinant_errors) | (determinant_errors > 1e-3)
    )
    if invalid_determinants.any():
        batch_index = int(np.flatnonzero(invalid_determinants)[0])
        raise RuntimeError(
            "rotation matrix determinant must be within 1e-3 of 1: "
            f"batch index {batch_index}; determinant={determinants[batch_index]}; "
            f"abs_error={determinant_errors[batch_index]}"
        )


def _validate_graph_value(value_info, *, name, shape) -> None:
    if value_info.name != name:
        raise RuntimeError(
            f"ONNX value name mismatch: expected {name!r}, got {value_info.name!r}"
        )
    element_type = value_info.type.tensor_type.elem_type
    if element_type != onnx.TensorProto.FLOAT:
        actual_type = onnx.TensorProto.DataType.Name(element_type)
        raise RuntimeError(
            f"ONNX value {name!r} dtype must be FLOAT, got {actual_type}"
        )
    actual_shape = graph_shape(value_info)
    if actual_shape != shape:
        raise RuntimeError(
            f"ONNX value {name!r} shape must be {shape}, got {actual_shape}"
        )


def _validate_runtime_value(value, *, name, shape) -> None:
    if value.name != name:
        raise RuntimeError(
            f"ONNX Runtime value name mismatch: expected {name!r}, got {value.name!r}"
        )
    if value.type != "tensor(float)":
        raise RuntimeError(
            f"ONNX Runtime value {name!r} dtype must be tensor(float), "
            f"got {value.type!r}"
        )
    if tuple(value.shape) != shape:
        raise RuntimeError(
            f"ONNX Runtime value {name!r} shape must be {shape}, "
            f"got {tuple(value.shape)}"
        )


def _iter_tensor_protos(message):
    """Yield every TensorProto recursively, including sparse and nested values."""
    if isinstance(message, onnx.TensorProto):
        yield message
        return

    for field, value in message.ListFields():
        if field.message_type is None:
            continue
        if field.is_repeated:
            for child in value:
                yield from _iter_tensor_protos(child)
        else:
            yield from _iter_tensor_protos(value)


def _validate_opset_imports(onnx_model) -> None:
    actual_imports = [
        (item.domain, int(item.version))
        for item in onnx_model.opset_import
    ]
    expected_imports = [("", OPSET_VERSION)]
    if actual_imports != expected_imports:
        raise RuntimeError(
            f"ONNX opset imports must be exactly {expected_imports}, "
            f"got {actual_imports}"
        )


def _validate_self_contained(onnx_model) -> None:
    external_tensors = []
    for index, tensor in enumerate(_iter_tensor_protos(onnx_model)):
        if (
            tensor.data_location == onnx.TensorProto.EXTERNAL
            or tensor.external_data
        ):
            external_tensors.append(tensor.name or f"<unnamed tensor {index}>")
    if external_tensors:
        raise RuntimeError(
            "ONNX candidate must be self-contained; external tensors: "
            f"{external_tensors}"
        )


def validate_candidate(path: str | Path, model: nn.Module, examples) -> None:
    """Validate an ONNX candidate's graph, runtime signature, and numerics."""
    path = Path(path)
    onnx_model = onnx.load(path, load_external_data=False)
    _validate_opset_imports(onnx_model)
    _validate_self_contained(onnx_model)
    onnx.checker.check_model(onnx_model)

    graph_inputs = onnx_model.graph.input
    graph_outputs = onnx_model.graph.output
    if len(graph_inputs) != 1:
        raise RuntimeError(
            f"ONNX graph must have exactly one input, got {len(graph_inputs)}"
        )
    if len(graph_outputs) != 1:
        raise RuntimeError(
            f"ONNX graph must have exactly one output, got {len(graph_outputs)}"
        )
    _validate_graph_value(
        graph_inputs[0],
        name="image",
        shape=("batch", 3, 224, 224),
    )
    _validate_graph_value(
        graph_outputs[0],
        name="rotation_matrix",
        shape=("batch", 3, 3),
    )

    session = ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )
    runtime_inputs = session.get_inputs()
    runtime_outputs = session.get_outputs()
    if len(runtime_inputs) != 1:
        raise RuntimeError(
            f"ONNX Runtime session must have exactly one input, got {len(runtime_inputs)}"
        )
    if len(runtime_outputs) != 1:
        raise RuntimeError(
            "ONNX Runtime session must have exactly one output, "
            f"got {len(runtime_outputs)}"
        )
    _validate_runtime_value(
        runtime_inputs[0],
        name="image",
        shape=("batch", 3, 224, 224),
    )
    _validate_runtime_value(
        runtime_outputs[0],
        name="rotation_matrix",
        shape=("batch", 3, 3),
    )

    for example_index, example in enumerate(examples):
        example_array = example.detach().cpu().numpy()
        with torch.no_grad():
            expected = model(example).detach().cpu().numpy()
        expected_shape = (example.shape[0], 3, 3)
        if expected.shape != expected_shape:
            raise RuntimeError(
                f"Torch output {example_index} shape must be {expected_shape}, "
                f"got {expected.shape}"
            )
        if expected.dtype != np.dtype(np.float32):
            raise RuntimeError(
                f"Torch output {example_index} dtype must be float32, "
                f"got {expected.dtype}"
            )
        if not np.isfinite(expected).all():
            raise RuntimeError(f"Torch output {example_index} contains non-finite values")

        actual = session.run(
            ["rotation_matrix"],
            {"image": example_array},
        )[0]
        if actual.shape != expected_shape:
            raise RuntimeError(
                f"ONNX output {example_index} shape must be {expected_shape}, "
                f"got {actual.shape}"
            )
        try:
            validate_rotation_matrices(actual)
        except RuntimeError as error:
            raise RuntimeError(
                f"ONNX output {example_index} is not a proper rotation matrix: {error}"
            ) from error
        if not np.allclose(actual, expected, rtol=1e-4, atol=1e-5):
            max_abs_error = float(np.max(np.abs(actual - expected)))
            raise RuntimeError(
                f"ONNX output {example_index} numerical mismatch: "
                f"max_abs_error={max_abs_error}"
            )


def _validate_export_model(model: nn.Module) -> None:
    training_modules = [
        name or "<root>"
        for name, module in model.named_modules()
        if module.training
    ]
    if training_modules:
        raise RuntimeError(
            "ONNX export requires evaluation mode for every module; "
            f"training modules={training_modules}"
        )

    for state_kind, named_tensors in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, tensor in named_tensors:
            if tensor.device.type != "cpu":
                raise RuntimeError(
                    f"ONNX export requires every model {state_kind} on CPU; "
                    f"{name!r} is on {tensor.device}"
                )
            if tensor.is_floating_point() and tensor.dtype != torch.float32:
                raise RuntimeError(
                    "ONNX export requires every floating model "
                    f"{state_kind} to be float32; {name!r} has dtype={tensor.dtype}"
                )


def export_release(model: nn.Module, output: str | Path) -> str:
    """Export, validate, and rollback-safely publish an ONNX release."""
    _validate_export_model(model)
    output = Path(output)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260824)
    examples = (
        torch.randn(
            (1, 3, 224, 224),
            dtype=torch.float32,
            generator=generator,
        ),
        torch.linspace(
            -1.0,
            1.0,
            4 * 3 * 224 * 224,
            dtype=torch.float32,
        ).reshape(4, 3, 224, 224),
    )

    def build_and_validate(candidate_path):
        with torch.no_grad():
            torch.onnx.export(
                model,
                examples[0],
                candidate_path,
                input_names=["image"],
                output_names=["rotation_matrix"],
                opset_version=OPSET_VERSION,
                do_constant_folding=True,
                dynamo=False,
                dynamic_axes={
                    "image": {0: "batch"},
                    "rotation_matrix": {0: "batch"},
                },
            )
        validate_candidate(candidate_path, model, examples)

    return publish_onnx_atomically(output, build_and_validate)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Export and validate a trusted 6DRepNet checkpoint as ONNX."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"trusted PyTorch checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"published ONNX model path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args(argv)


def _paths_are_aliases(first: Path, second: Path) -> bool:
    try:
        if first.resolve(strict=False) == second.resolve(strict=False):
            return True
        if first.exists() and second.exists():
            return os.path.samefile(first, second)
    except OSError as error:
        raise RuntimeError(
            f"Unable to validate CLI release paths {first!s} and {second!s}: {error}"
        ) from error
    return False


def _validate_cli_paths(checkpoint: Path, output: Path) -> None:
    release_paths = (
        ("checkpoint", checkpoint),
        ("output", output),
    )
    for first_index, (first_name, first_path) in enumerate(release_paths):
        for second_name, second_path in release_paths[first_index + 1 :]:
            if _paths_are_aliases(first_path, second_path):
                raise RuntimeError(
                    "CLI path collision: "
                    f"{first_name}={first_path} aliases "
                    f"{second_name}={second_path}"
                )


def main(argv=None) -> int:
    arguments = parse_args(argv)
    checkpoint = Path(arguments.checkpoint)
    output = Path(arguments.output)
    _validate_cli_paths(checkpoint, output)
    state_dict = load_checkpoint(
        checkpoint,
        CHECKPOINT_SIZE,
        CHECKPOINT_SHA256,
    )
    model = build_model(state_dict)
    digest = export_release(model, output)

    print(f"Checkpoint: {checkpoint}")
    print(f"Checkpoint size: {CHECKPOINT_SIZE}")
    print(f"Checkpoint SHA256: {CHECKPOINT_SHA256}")
    print(f"ONNX: {output}")
    print(f"ONNX size: {output.stat().st_size}")
    print(f"ONNX SHA256: {digest}")
    print("Input: image FLOAT [batch,3,224,224]")
    print("Output: rotation_matrix FLOAT [batch,3,3]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

