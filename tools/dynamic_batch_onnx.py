"""Publish ONNX models whose public boundary accepts a symbolic batch size."""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto

if __package__:
    from .onnx_export_validation import publish_onnx_atomically
else:
    from onnx_export_validation import publish_onnx_atomically


def set_symbolic_batch(value_info: onnx.ValueInfoProto, name: str = "batch") -> None:
    """Set a tensor boundary's first dimension to the supplied symbolic name."""
    shape = value_info.type.tensor_type.shape
    if len(shape.dim) < 1:
        raise ValueError(f"{value_info.name!r} must be a tensor with rank at least 1")
    first_dimension = shape.dim[0]
    first_dimension.ClearField("dim_value")
    first_dimension.dim_param = name


def _validate_model_contract(
    model: onnx.ModelProto,
    *,
    input_name: str,
    output_names: tuple[str, ...],
    sample_shape: tuple[int, ...],
    output_sample_shapes: tuple[tuple[int, ...], ...],
) -> None:
    graph = model.graph
    actual_input_names = tuple(value.name for value in graph.input)
    actual_output_names = tuple(value.name for value in graph.output)
    if actual_input_names != (input_name,):
        raise ValueError(
            f"Unexpected ONNX input names: {actual_input_names!r}; "
            f"expected {(input_name,)!r}"
        )
    if actual_output_names != output_names:
        raise ValueError(
            f"Unexpected ONNX output names: {actual_output_names!r}; "
            f"expected {output_names!r}"
        )
    if len(output_sample_shapes) != len(output_names):
        raise ValueError(
            "output_sample_shapes length must match output_names length: "
            f"{len(output_sample_shapes)} != {len(output_names)}"
        )

    input_info = graph.input[0]
    if input_info.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise ValueError("Dynamic batch publication requires a float input tensor")
    dimensions = input_info.type.tensor_type.shape.dim
    if len(dimensions) != len(sample_shape) + 1:
        raise ValueError(
            f"sample_shape {sample_shape!r} does not match input rank {len(dimensions)}"
        )
    if any(not isinstance(dimension, int) or dimension <= 0 for dimension in sample_shape):
        raise ValueError(f"sample_shape must contain positive integer dimensions: {sample_shape!r}")
    for index, (dimension, expected) in enumerate(zip(dimensions[1:], sample_shape), 1):
        if not dimension.HasField("dim_value"):
            actual = (
                f"symbolic {dimension.dim_param!r}"
                if dimension.dim_param
                else "unknown"
            )
            raise ValueError(
                f"input dimension {index} must be static {expected}; got {actual}"
            )
        if dimension.dim_value != expected:
            raise ValueError(
                f"sample_shape[{index - 1}]={expected} does not match "
                f"input dimension {index}={dimension.dim_value}"
            )
    for output, output_sample_shape in zip(graph.output, output_sample_shapes):
        if output.type.tensor_type.elem_type != TensorProto.FLOAT:
            raise ValueError(f"output {output.name!r} must be a float tensor")
        output_dimensions = output.type.tensor_type.shape.dim
        if len(output_dimensions) != len(output_sample_shape) + 1:
            raise ValueError(
                f"output {output.name!r} sample shape {output_sample_shape!r} "
                f"does not match output rank {len(output_dimensions)}"
            )
        if any(
            not isinstance(dimension, int) or dimension <= 0
            for dimension in output_sample_shape
        ):
            raise ValueError(
                f"output {output.name!r} sample shape must contain positive integer "
                f"dimensions: {output_sample_shape!r}"
            )
        for index, (dimension, expected) in enumerate(
            zip(output_dimensions[1:], output_sample_shape), 1
        ):
            if not dimension.HasField("dim_value"):
                actual = (
                    f"symbolic {dimension.dim_param!r}"
                    if dimension.dim_param
                    else "unknown"
                )
                raise ValueError(
                    f"output {output.name!r} dimension {index} must be static "
                    f"{expected}; got {actual}"
                )
            if dimension.dim_value != expected:
                raise ValueError(
                    f"output {output.name!r} dimension {index} must be {expected}; "
                    f"got {dimension.dim_value}"
                )


def _run_source_batch_one(
    session: ort.InferenceSession,
    *,
    input_name: str,
    output_names: tuple[str, ...],
    batch: np.ndarray,
) -> list[np.ndarray]:
    per_output: list[list[np.ndarray]] = [[] for _ in output_names]
    for sample in batch:
        outputs = session.run(list(output_names), {input_name: sample[None, ...]})
        for output_name, output, samples in zip(output_names, outputs, per_output):
            if output.ndim < 1 or output.shape[0] != 1:
                raise RuntimeError(
                    f"source output {output_name!r} must retain a leading B1 dimension; "
                    f"got shape {output.shape}"
                )
            samples.append(output[0])
    return [np.stack(samples, axis=0) for samples in per_output]


def validate_dynamic_batch_model(
    source_path,
    candidate_path,
    *,
    input_name: str,
    output_names: tuple[str, ...],
    sample_shape: tuple[int, ...],
    output_sample_shapes: tuple[tuple[int, ...], ...],
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    """Confirm candidate B2 values match two independent source B1 runs."""
    source_path = Path(source_path)
    candidate_path = Path(candidate_path)
    output_names = tuple(output_names)
    output_sample_shapes = tuple(tuple(shape) for shape in output_sample_shapes)
    if len(output_sample_shapes) != len(output_names):
        raise ValueError(
            "output_sample_shapes length must match output_names length: "
            f"{len(output_sample_shapes)} != {len(output_names)}"
        )
    batch = np.random.default_rng(2026).standard_normal(
        (2, *sample_shape), dtype=np.float32
    )
    source_session = None
    candidate_session = None
    try:
        source_session = ort.InferenceSession(
            str(source_path), providers=["CPUExecutionProvider"]
        )
        candidate_session = ort.InferenceSession(
            str(candidate_path), providers=["CPUExecutionProvider"]
        )
        expected_outputs = _run_source_batch_one(
            source_session,
            input_name=input_name,
            output_names=output_names,
            batch=batch,
        )
        actual_outputs = candidate_session.run(list(output_names), {input_name: batch})
        for output_name, expected, actual, output_sample_shape in zip(
            output_names, expected_outputs, actual_outputs, output_sample_shapes
        ):
            expected_shape = (2, *output_sample_shape)
            if expected.shape != expected_shape:
                raise RuntimeError(
                    f"output {output_name}: source output shape mismatch: "
                    f"expected {expected_shape}, got {expected.shape}"
                )
            if actual.shape != expected_shape:
                raise RuntimeError(
                    f"output {output_name}: candidate output shape mismatch: "
                    f"expected {expected_shape}, got {actual.shape}"
                )
            if actual.shape != expected.shape:
                raise RuntimeError(
                    f"output {output_name}: shape mismatch: expected {expected.shape}, "
                    f"got {actual.shape}"
                )
            if expected.dtype != np.float32:
                raise RuntimeError(
                    f"output {output_name}: expected FLOAT output dtype, "
                    f"got {expected.dtype}"
                )
            if actual.dtype != np.float32:
                raise RuntimeError(
                    f"output {output_name}: expected FLOAT output dtype, "
                    f"got {actual.dtype}"
                )
            if actual.dtype != expected.dtype:
                raise RuntimeError(
                    f"output {output_name}: dtype mismatch: expected {expected.dtype}, "
                    f"got {actual.dtype}"
                )
            if not np.isfinite(expected).all():
                raise RuntimeError(f"output {output_name}: source output contains non-finite values")
            if not np.isfinite(actual).all():
                raise RuntimeError(f"output {output_name}: candidate output contains non-finite values")
            if not np.allclose(actual, expected, rtol=rtol, atol=atol):
                max_abs_error = np.max(np.abs(actual - expected))
                raise RuntimeError(
                    f"output {output_name}: numerical mismatch: "
                    f"max_abs_error={max_abs_error}"
                )
    finally:
        if candidate_session is not None:
            del candidate_session
        if source_session is not None:
            del source_session


def publish_dynamic_batch_model(
    source_path,
    target_path,
    *,
    input_name: str,
    output_names: tuple[str, ...],
    sample_shape: tuple[int, ...],
    output_sample_shapes: tuple[tuple[int, ...], ...],
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    """Atomically publish a metadata-only symbolic-batch version of a model."""
    source_path = Path(source_path)
    target_path = Path(target_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source ONNX: {source_path}")
    output_names = tuple(output_names)
    sample_shape = tuple(sample_shape)
    output_sample_shapes = tuple(tuple(shape) for shape in output_sample_shapes)
    model = onnx.load(str(source_path))
    try:
        _validate_model_contract(
            model,
            input_name=input_name,
            output_names=output_names,
            sample_shape=sample_shape,
            output_sample_shapes=output_sample_shapes,
        )
        set_symbolic_batch(model.graph.input[0])
        for output in model.graph.output:
            set_symbolic_batch(output)
        onnx.checker.check_model(model)

        def build_and_validate(candidate_path: Path) -> None:
            onnx.save(model, str(candidate_path))
            validate_dynamic_batch_model(
                source_path,
                candidate_path,
                input_name=input_name,
                output_names=output_names,
                sample_shape=sample_shape,
                output_sample_shapes=output_sample_shapes,
                rtol=rtol,
                atol=atol,
            )

        publish_onnx_atomically(target_path, build_and_validate)
    finally:
        del model
