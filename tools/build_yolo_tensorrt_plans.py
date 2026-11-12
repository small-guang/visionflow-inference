"""Compatibility entry point for building only the three YOLO model plans."""
import argparse
import shlex
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import build_tensorrt_plans as _builder

MODELS = _builder.YOLO_MODELS
PLAN_FILENAME = _builder.PLAN_FILENAME
sha256 = _builder.sha256
os = _builder.os
subprocess = _builder.subprocess


def build_command(trtexec, source, output, *, precision="fp16"):
    return _builder.build_command(trtexec, source, output, model=MODELS[0], precision=precision)


def model_paths(repository, model):
    if model not in MODELS:
        raise ValueError(f"Unsupported YOLO model: {model}")
    return _builder.model_paths(repository, model)


def build_plan(repository, model, trtexec, *, precision="fp16"):
    model_paths(repository, model)
    return _builder.build_plan(repository, model, trtexec, precision=precision)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-repository", type=Path, default=Path("model_repository"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--trtexec", default="trtexec", help="trtexec from the deployment's matching TensorRT SDK")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16", help="FP32 also disables TF32; qualify the selected precision before deployment")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    # Keep the YOLO entry point's model restriction and default precision.
    build_options = {} if args.precision == "fp16" else {"precision": args.precision}
    for model in args.models:
        model_paths(args.model_repository, model)
    for model in args.models:
        if args.dry_run:
            source, target = model_paths(args.model_repository, model)
            print(shlex.join(build_command(args.trtexec, source, target, **build_options)))
        else:
            build_plan(args.model_repository, model, args.trtexec, **build_options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
