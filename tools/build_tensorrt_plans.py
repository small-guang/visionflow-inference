"""Build target-device TensorRT plans while preserving deployed tensor contracts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

YOLO_MODELS = ("yolov8n_face_onnx", "yolo26m_pose_onnx", "yolo26s_seg_onnx")
FIXED_INPUTS = {
    "clip_image_trt": ("image", (3, 224, 224)),
    "clip_text_trt": ("text", (52,)),
    "TBPS_clip_image_trt": ("image", (3, 224, 224)),
    "TBPS_clip_text_trt": ("text", (77,)),
    "sbs_r50_ibn_onnx": ("image", (3, 384, 128)),
    "w600k_r50_trt": ("input.1", (3, 112, 112)),
    "face_qa": ("input", (3, 112, 112)),
    "face_keypoint_2d106": ("data", (3, 192, 192)),
}
MODELS = (*YOLO_MODELS, *FIXED_INPUTS)
CANDIDATE_FIXED_INPUTS = {**FIXED_INPUTS,
    "adaface_ir50_webface4m_onnx": ("input", (3, 112, 112)),
    "osnet_ain_x1_0_onnx": ("image", (3, 256, 128)),
    "sixdrepnet_onnx": ("image", (3, 224, 224)),
}
PLAN_FILENAME = "model_throughput.plan"
# FP16 exceeded quality-score / pixel-coordinate tolerances on RTX 3090 with
# TensorRT 10.7. Keep these small models at FP32 unless explicitly requalified.
DEFAULT_PRECISIONS = {"face_qa": "fp32", "face_keypoint_2d106": "fp32"}


def resolve_precision(model: str, precision: str | None) -> str:
    selected = DEFAULT_PRECISIONS.get(model, "fp16") if precision is None else precision
    if selected not in {"fp16", "fp32"}:
        raise ValueError(f"Unsupported precision: {selected}")
    return selected


def build_command(trtexec: str, source: Path, output: Path,
                  *, model: str, precision: str | None = None) -> list[str]:
    if model not in MODELS:
        raise ValueError(f"Unknown native TensorRT model: {model}")
    return _batch_command(trtexec, source, output, model=model,
                          max_batch=4, precision=precision)


def _batch_command(trtexec: str, source: Path, output: Path, *, model: str,
                   max_batch: int, opt_batch: int = 1, precision: str | None = None,
                   workspace_mib: int = 1024) -> list[str]:
    if model not in (*YOLO_MODELS, *CANDIDATE_FIXED_INPUTS):
        raise ValueError(f"Unknown native TensorRT model: {model}")
    for name, value in (("max_batch", max_batch), ("opt_batch", opt_batch),
                        ("workspace_mib", workspace_mib)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if opt_batch > max_batch:
        raise ValueError("opt_batch must not exceed max_batch")
    precision = resolve_precision(model, precision)
    if model in YOLO_MODELS:
        minimum, optimum, maximum = ("images:1x3x192x192", f"images:{opt_batch}x3x384x640",
                                     f"images:{max_batch}x3x640x640")
    else:
        name, shape = CANDIDATE_FIXED_INPUTS[model]
        dimensions = "x".join(map(str, shape))
        minimum = f"{name}:1x{dimensions}"
        optimum = f"{name}:{opt_batch}x{dimensions}"
        maximum = f"{name}:{max_batch}x{dimensions}"
    return [
        str(trtexec), f"--onnx={source}", f"--saveEngine={output}",
        "--fp16" if precision == "fp16" else "--noTF32",
        f"--minShapes={minimum}", f"--optShapes={optimum}",
        f"--maxShapes={maximum}", f"--memPoolSize=workspace:{workspace_mib}",
        "--skipInference",
    ]


def _validate_engine_paths(source: Path, output: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Export the source ONNX first: {source}")
    if source.resolve() == output.resolve() or (output.exists() and source.samefile(output)):
        raise ValueError("Source ONNX and output must be different files")


def engine_command(trtexec: str, source: Path, output: Path, *, model: str,
                   max_batch: int, opt_batch: int = 1, precision: str | None = None,
                   workspace_mib: int = 1024) -> list[str]:
    """Return argv for an explicit candidate without writing files or changing I/O types.

    Unlike the deployment command preview, this API requires an existing source file.
    Batch settings affect only the leading dimension; text lengths and YOLO's
    spatial range retain their existing contracts.
    """
    command = _batch_command(trtexec, source, output, model=model, max_batch=max_batch,
                             opt_batch=opt_batch, precision=precision, workspace_mib=workspace_mib)
    _validate_engine_paths(Path(source), Path(output))
    return command


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_paths(repository: Path, model: str) -> tuple[Path, Path]:
    if model not in MODELS:
        raise ValueError(f"Unsupported native TensorRT model: {model}")
    repository = repository.resolve()
    version = (repository / model / "1").resolve()
    if not version.is_relative_to(repository):
        raise ValueError("Model version directory must be inside the repository")
    source = version / "model.onnx"
    if not source.is_file():
        raise FileNotFoundError(f"Export the source ONNX first: {source}")
    return source, version / PLAN_FILENAME


def build_plan(repository: Path, model: str, trtexec: str,
               *, precision: str | None = None) -> Path:
    source, target = model_paths(repository, model)
    precision = resolve_precision(model, precision)
    logical_command = build_command(trtexec, source, target, model=model, precision=precision)
    return _build_artifacts(
        source, target, model=model, profile="throughput", precision=precision,
        logical_command=logical_command,
        command_for_output=lambda output: build_command(trtexec, source, output,
                                                        model=model, precision=precision),
    )


def build_engine(source: Path, destination: Path, *, model: str, max_batch: int,
                 opt_batch: int = 1, precision: str | None = None,
                 workspace_mib: int = 1024, trtexec: str = "trtexec") -> Path:
    """Build an independent candidate at destination and return its resolved path.

    Create missing parent directories after validation. Publish the engine with
    appended .json and .build.log sidecars, restoring previous artifacts if a
    replacement fails. Provenance uses batchN and records the final-path command;
    unique attempt logs are retained, including for failed builds.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    logical_command = engine_command(trtexec, source, destination, model=model, max_batch=max_batch,
                                     opt_batch=opt_batch, precision=precision, workspace_mib=workspace_mib)
    precision = resolve_precision(model, precision)
    # Explicit filenames must not allow a sidecar to overwrite the source graph.
    for suffix in (".json", ".build.log"):
        _validate_engine_paths(source, destination.with_name(destination.name + suffix))
    destination.parent.mkdir(parents=True, exist_ok=True)
    return _build_artifacts(
        source, destination, model=model, profile=f"batch{max_batch}", precision=precision,
        logical_command=logical_command,
        command_for_output=lambda output: engine_command(trtexec, source, output, model=model,
                                                         max_batch=max_batch, opt_batch=opt_batch,
                                                         precision=precision, workspace_mib=workspace_mib),
        extra_provenance={"max_batch": max_batch, "opt_batch": opt_batch, "workspace_mib": workspace_mib},
    )


def _build_artifacts(source: Path, target: Path, *, model: str, profile: str, precision: str,
                     logical_command, command_for_output, extra_provenance=None) -> Path:
    source_hash = sha256(source)
    log_path = target.with_name(target.name + ".build.log")
    attempt_log = target.with_name(f"{target.name}.build-{uuid.uuid4().hex}.log")
    print(f"Building {model}/{profile}; TensorRT log: {attempt_log}", flush=True)
    # Stage beside the destination so each replacement stays on one filesystem.
    with tempfile.TemporaryDirectory(prefix=".trt-build-", dir=target.parent) as temporary:
        candidate = Path(temporary) / target.name
        command = command_for_output(candidate)
        with attempt_log.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode or not candidate.is_file() or candidate.stat().st_size == 0:
            raise RuntimeError(f"TensorRT build failed; existing plan was kept. See {attempt_log}")
        if sha256(source) != source_hash:
            raise RuntimeError("Source ONNX changed during build; existing plan was kept")
        provenance = {
            "model": model, "profile": profile, "precision": precision.upper(),
            "source_sha256": source_hash, "plan_sha256": sha256(candidate),
            "build_command": logical_command,
            "build_log": log_path.name,
        }
        if extra_provenance:
            provenance.update(extra_provenance)
        metadata = Path(temporary) / (target.name + ".json")
        metadata.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        successful_log = Path(temporary) / log_path.name
        shutil.copyfile(attempt_log, successful_log)
        artifacts = ((candidate, target), (successful_log, log_path),
                     (metadata, target.with_name(target.name + ".json")))
        backups = {}
        for _, destination in artifacts:
            backup = Path(temporary) / (destination.name + ".previous")
            if destination.exists():
                shutil.copyfile(destination, backup)
                backups[destination] = backup
            else:
                backups[destination] = None
        published = []
        try:
            for prepared, destination in artifacts:
                os.replace(prepared, destination)
                published.append(destination)
        except OSError:
            # Per-file replacements are atomic; restore prior files if publication
            # fails so a new engine is never left with an old provenance hash.
            for destination in reversed(published):
                if backups[destination] is None:
                    destination.unlink()
                else:
                    os.replace(backups[destination], destination)
            raise
    print(f"Built {target}", flush=True)
    return target


def main(argv=None, *, default_models=MODELS, model_choices=MODELS) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-repository", type=Path, default=Path("model_repository"))
    parser.add_argument("--models", nargs="+", choices=model_choices, default=list(default_models))
    parser.add_argument("--trtexec", default="trtexec", help="trtexec from the deployment's matching TensorRT SDK")
    parser.add_argument("--precision", choices=("fp16", "fp32"), help="Override model defaults: face_qa and face_keypoint_2d106 FP32 (TF32 disabled), other models FP16; requalify on the target GPU")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    # Fail before compiling anything if one of the requested sources is missing.
    for model in args.models:
        model_paths(args.model_repository, model)
    for model in args.models:
        if args.dry_run:
            source, target = model_paths(args.model_repository, model)
            print(shlex.join(build_command(args.trtexec, source, target, model=model, precision=args.precision)))
        else:
            build_plan(args.model_repository, model, args.trtexec, precision=args.precision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
