#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
PATH_COMPONENT_NAMES = frozenset({".", ".."})
CONFLICTING_FLAGS = (
    "--model-control-mode",
    "--load-model",
    "--model-config-name",
)
ORIN_FUSED_PREPROCESSING_ENTRY_POINTS = frozenset({
    "clip_ensemble_trt", "sbs_r50_ibn_image_ensemble",
    "face_keypoint_2d106_ensemble", "face_qa_ensemble", "face_recognition_pipeline",
})
ORIN_GPU_ENTRY_POINTS = frozenset({
    "yolov8n_face_ensemble", "yolo26m_pose_ensemble", "yolo26s_seg_ensemble",
    "TBPS_clip_image_ensemble",
}) | ORIN_FUSED_PREPROCESSING_ENTRY_POINTS
ORIN_GPU_PREPROCESSORS = frozenset({
    "clip_image_preprocess", "sbs_r50_ibn_image_preprocess",
    "face_keypoint_2d106_pre", "face_qa_pre", "face_rec_pre",
})


class StartupConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ModelSetDefinition:
    entry_point: str
    models: tuple[str, ...]


@dataclass(frozen=True)
class Manifest:
    schema_version: int
    model_sets: Mapping[str, ModelSetDefinition]


def load_manifest(path: Path) -> Manifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StartupConfigError(
            f"cannot load model-set manifest {path}: {exc}"
        ) from exc
    if type(raw) is not dict:
        raise StartupConfigError("model-set manifest root must be an object")
    if set(raw) != {"schema_version", "model_sets"}:
        raise StartupConfigError(
            "model-set manifest must contain only schema_version and model_sets"
        )
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise StartupConfigError("unsupported model-set manifest schema_version")
    raw_sets = raw["model_sets"]
    if type(raw_sets) is not dict or not raw_sets:
        raise StartupConfigError("model_sets must be a non-empty object")

    definitions = {}
    for alias, raw_definition in raw_sets.items():
        if (type(alias) is not str or not SAFE_NAME.fullmatch(alias)
                or alias in PATH_COMPONENT_NAMES):
            raise StartupConfigError(f"unsafe model-set alias: {alias!r}")
        if alias == "all":
            raise StartupConfigError("all is reserved and cannot be a manifest alias")
        if type(raw_definition) is not dict or set(raw_definition) != {
            "entry_point", "models"
        }:
            raise StartupConfigError(
                f"model-set {alias!r} must contain only entry_point and models"
            )
        entry_point = raw_definition["entry_point"]
        raw_models = raw_definition["models"]
        if (type(entry_point) is not str or not SAFE_NAME.fullmatch(entry_point)
                or entry_point in PATH_COMPONENT_NAMES):
            raise StartupConfigError(f"unsafe entry point for model-set {alias!r}")
        if type(raw_models) is not list or not raw_models:
            raise StartupConfigError(f"models for model-set {alias!r} must be a list")
        if any(
            type(model) is not str or not SAFE_NAME.fullmatch(model)
            or model in PATH_COMPONENT_NAMES
            for model in raw_models
        ):
            raise StartupConfigError(f"unsafe model name in model-set {alias!r}")
        if len(raw_models) != len(set(raw_models)):
            raise StartupConfigError(f"duplicate model in model-set {alias!r}")
        if raw_models[-1] != entry_point:
            raise StartupConfigError(
                f"entry point must be the final model in model-set {alias!r}"
            )
        definitions[alias] = ModelSetDefinition(
            entry_point=entry_point,
            models=tuple(raw_models),
        )
    return Manifest(schema_version=1, model_sets=definitions)


def resolve_model_set(value: str, manifest: Manifest) -> tuple[str, ...]:
    raw_aliases = value.split(",")
    aliases = tuple(alias.strip() for alias in raw_aliases)
    if any(not alias for alias in aliases):
        raise StartupConfigError("TRITON_MODEL_SET contains an empty alias")
    if len(aliases) != len(set(aliases)):
        raise StartupConfigError("TRITON_MODEL_SET contains duplicate aliases")
    if "all" in aliases and aliases != ("all",):
        raise StartupConfigError("TRITON_MODEL_SET cannot combine all with another alias")
    if aliases == ("all",):
        aliases = tuple(manifest.model_sets)
    unknown = tuple(alias for alias in aliases if alias not in manifest.model_sets)
    if unknown:
        raise StartupConfigError(f"unknown model-set alias: {unknown[0]}")
    ordered = []
    for alias in aliases:
        for model in manifest.model_sets[alias].models:
            if model not in ordered:
                ordered.append(model)
    return tuple(ordered)


def validate_repository(
    repository: Path,
    models: Sequence[str],
) -> None:
    if not repository.is_dir():
        raise StartupConfigError(f"model repository is missing: {repository}")
    for model in models:
        model_dir = repository / model
        config = model_dir / "config.pbtxt"
        if not model_dir.is_dir():
            raise StartupConfigError(f"model directory is missing: {model_dir}")
        if not config.is_file():
            raise StartupConfigError(
                f"model config is missing: {config}"
            )


def _reject_conflicts(base_command: Sequence[str]) -> None:
    for argument in base_command[1:]:
        for flag in CONFLICTING_FLAGS:
            if argument == flag or argument.startswith(flag + "="):
                raise StartupConfigError(
                    f"conflicting Triton argument already supplied: {argument}"
                )


def resolve_runtime_models(
    model_set_value: str,
    manifest: Manifest,
    model_config_name: str = "",
) -> tuple[str, ...]:
    """Resolve loaded nodes while the original manifest keeps packaged resources."""
    if model_config_name not in {"", "orin_gpu"}:
        raise StartupConfigError("TRITON_MODEL_CONFIG must be empty or orin_gpu")
    if not model_config_name:
        return resolve_model_set(model_set_value, manifest)
    runtime_manifest = Manifest(manifest.schema_version, {
        alias: ModelSetDefinition(definition.entry_point, (definition.entry_point,))
        if definition.entry_point in ORIN_GPU_ENTRY_POINTS else definition
        for alias, definition in manifest.model_sets.items()
    })
    return resolve_model_set(model_set_value, runtime_manifest)


def build_triton_command(
    base_command: Sequence[str],
    model_set_value: str,
    manifest: Manifest,
    repository: Path,
    model_config_name: str = "",
) -> tuple[str, ...]:
    if not base_command or not base_command[0]:
        raise StartupConfigError("a Triton command is required after --")
    _reject_conflicts(base_command)
    models = resolve_runtime_models(model_set_value, manifest, model_config_name)
    # The fused wrappers read engines and helpers from the original dependency
    # directories even though Triton must not load those separate instances.
    validate_repository(repository, resolve_model_set(model_set_value, manifest))
    injected = []
    if model_config_name:
        for model in models:
            if model in ORIN_GPU_PREPROCESSORS:
                for relative in ("configs/orin_gpu.pbtxt", "1/model.py"):
                    path = repository / model / relative
                    if not path.is_file():
                        raise StartupConfigError(f"Orin GPU preprocessing resource is missing: {path}")
            if model in ORIN_GPU_ENTRY_POINTS:
                for relative in ("configs/orin_gpu.pbtxt", "1/model.py"):
                    path = repository / model / relative
                    if not path.is_file():
                        raise StartupConfigError(f"Orin GPU resource is missing: {path}")
                if model == "TBPS_clip_image_ensemble":
                    path = repository / "TBPS_clip_image_preprocess" / "1" / "gpu.py"
                    if not path.is_file():
                        raise StartupConfigError(f"Orin GPU resource is missing: {path}")
        injected.append(f"--model-config-name={model_config_name}")
    if model_set_value.strip() != "all" or model_config_name:
        injected.append("--model-control-mode=explicit")
        injected.extend(f"--load-model={model}" for model in models)
    return (base_command[0], *injected, *base_command[1:])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-repository", type=Path, required=True)
    parser.add_argument("--log-wrapper", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("triton_command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = tuple(args.triton_command)
    if command[:1] == ("--",):
        command = command[1:]
    try:
        manifest = load_manifest(args.manifest)
        resolved = build_triton_command(
            command,
            os.environ.get("TRITON_MODEL_SET", "all"),
            manifest,
            args.model_repository,
            model_config_name=os.environ.get("TRITON_MODEL_CONFIG", ""),
        )
        if not args.log_wrapper.is_file():
            raise StartupConfigError(f"log wrapper is missing: {args.log_wrapper}")
    except StartupConfigError as exc:
        print(f"start_triton: ERROR: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps(resolved, ensure_ascii=False))
        return 0
    os.execv(
        sys.executable,
        [sys.executable, str(args.log_wrapper), "--", *resolved],
    )
    raise AssertionError("os.execv returned unexpectedly")


if __name__ == "__main__":
    raise SystemExit(main())
