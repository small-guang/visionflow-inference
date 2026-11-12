import hashlib
import os
import stat
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def publication_mode(target_path):
    """Return an existing artifact's mode, or a readable default for a new one."""
    target_path = Path(target_path)
    if target_path.exists():
        return stat.S_IMODE(target_path.stat().st_mode)
    return 0o644


def _make_cleanup_writable(path):
    path = Path(path)
    if path.exists():
        current_mode = stat.S_IMODE(path.stat().st_mode)
        os.chmod(path, current_mode | stat.S_IWUSR)


def _add_exception_note(error, note):
    if hasattr(error, "add_note"):
        error.add_note(note)


def _cleanup_publication_file(path, primary_error=None):
    path = Path(path)
    try:
        if not path.exists():
            return
        _make_cleanup_writable(path)
        path.unlink()
    except Exception as cleanup_error:
        if primary_error is None:
            raise
        _add_exception_note(
            primary_error,
            f"Cleanup failed for {path.resolve()}: {cleanup_error}",
        )


def replace_publication_file(candidate_path, target_path, desired_mode):
    """Replace a publication file safely, including read-only Windows targets."""
    candidate_path = Path(candidate_path)
    target_path = Path(target_path)
    original_destination_mode = None
    destination_made_writable = False
    try:
        os.chmod(candidate_path, desired_mode)
        if os.name == "nt" and target_path.exists():
            original_destination_mode = stat.S_IMODE(target_path.stat().st_mode)
            if not original_destination_mode & stat.S_IWUSR:
                os.chmod(
                    target_path,
                    original_destination_mode | stat.S_IWUSR,
                )
                destination_made_writable = True
        os.replace(candidate_path, target_path)
    except BaseException as replace_error:
        if destination_made_writable and target_path.exists():
            try:
                os.chmod(target_path, original_destination_mode)
            except Exception as mode_error:
                _add_exception_note(
                    replace_error,
                    "Unable to restore publication mode for "
                    f"{target_path.resolve()}: {mode_error}",
                )
        raise


def publish_file_atomically(target_path, build_candidate, *, desired_mode=None):
    """Build a sibling candidate and publish it with controlled replacement."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if desired_mode is None:
        desired_mode = publication_mode(target_path)
    file_descriptor, candidate_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.",
        suffix=".tmp",
        dir=target_path.parent,
    )
    os.close(file_descriptor)
    candidate_path = Path(candidate_name)
    primary_error = None
    try:
        build_candidate(candidate_path)
        replace_publication_file(candidate_path, target_path, desired_mode)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        _cleanup_publication_file(candidate_path, primary_error)


def publish_onnx_atomically(target_path, build_and_validate) -> str:
    """Validate and hash a sibling candidate, then atomically publish its bytes."""
    digest = ""

    def build_nonempty_candidate(candidate_path):
        nonlocal digest
        build_and_validate(candidate_path)
        if candidate_path.stat().st_size == 0:
            raise RuntimeError("validated ONNX candidate is empty")
        digest = file_sha256(candidate_path)

    publish_file_atomically(target_path, build_nonempty_candidate)
    return digest


def file_sha256(path, chunk_size=1024 * 1024):
    """Return the lowercase SHA256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_onnx_output(
    model_path,
    input_name,
    output_name,
    input_value,
    expected_output,
    *,
    rtol=1e-4,
    atol=1e-5,
):
    """Run a checked ONNX model and compare its sole output to an expectation."""
    model_path = Path(model_path)
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    del model

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    if len(inputs) != 1 or inputs[0].name != input_name:
        raise RuntimeError(f"input name must be the model's sole input: {input_name!r}")
    if len(outputs) != 1 or outputs[0].name != output_name:
        raise RuntimeError(f"output name must be the model's sole output: {output_name!r}")

    input_array = np.asarray(input_value)
    expected_array = np.asarray(expected_output)
    actual = session.run([output_name], {input_name: input_array})[0]

    if actual.shape != expected_array.shape:
        raise RuntimeError(
            f"output shape mismatch: expected {expected_array.shape}, got {actual.shape}"
        )
    if not np.isfinite(expected_array).all():
        raise RuntimeError("expected output contains non-finite values")
    if actual.dtype != expected_array.dtype:
        raise RuntimeError(
            f"dtype mismatch: expected {expected_array.dtype}, got {actual.dtype}"
        )
    if not np.isfinite(actual).all():
        raise RuntimeError("actual output contains non-finite values")
    if not np.allclose(actual, expected_array, rtol=rtol, atol=atol):
        max_abs_error = np.max(np.abs(actual - expected_array))
        raise RuntimeError(f"numerical mismatch: max_abs_error={max_abs_error}")

    return actual
