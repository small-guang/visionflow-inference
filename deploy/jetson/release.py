#!/usr/bin/env python3
"""Build, verify and explicitly publish frozen Jetson inference images."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid


ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "deploy" / "jetson"
RUNTIME = "registry.cn-hangzhou.aliyuncs.com/eapil/tritonserver:26.04-py3-arm64-runtime"
sys.path.insert(0, str(ROOT / "scripts"))
import start_triton  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def included(path):
    return not (any(part.startswith(".") or part == "__pycache__" or part.startswith("trt_cache_")
                    for part in path.parts) or path.suffix in {".pyc", ".log"})


def release_aliases(root):
    """Read the build-time service allowlist; runtime selection is a subset of it."""
    selection = json.loads((root / "deploy/jetson/release_models.json").read_text())
    if (type(selection) is not dict or set(selection) != {"schema_version", "aliases"}
            or type(selection["schema_version"]) is not int or selection["schema_version"] != 1):
        raise ValueError("invalid release_models.json schema")
    aliases = selection["aliases"]
    if (type(aliases) is not list or not aliases
            or any(type(alias) is not str or not start_triton.SAFE_NAME.fullmatch(alias) for alias in aliases)
            or "all" in aliases or len(aliases) != len(set(aliases))):
        raise ValueError("release aliases must be a nonempty list of unique explicit model-set names")
    manifest = start_triton.load_manifest(root / "profiles/model_sets.json")
    start_triton.resolve_model_set(",".join(aliases), manifest)
    return aliases


def inventory(root, aliases=None):
    """Validate every shipped model and fingerprint all runtime resources."""
    repository = root / "model_repository"
    manifest = start_triton.load_manifest(root / "profiles/model_sets.json")
    models = start_triton.resolve_model_set("all" if aliases is None else ",".join(aliases), manifest)
    start_triton.validate_repository(repository, models)
    directories = sorted(repository / model for model in models)
    files = {}
    for directory in directories:
        if directory.is_symlink():
            raise ValueError(f"symlink is not allowed in release input: {directory}")
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(root)
            if not included(relative):
                continue
            if path.is_symlink():
                raise ValueError(f"symlink is not allowed in release input: {relative}")
            if path.is_file():
                files[relative.as_posix()] = sha256(path)
    missing = []
    reference_onnx = {}
    for directory in directories:
        config = directory / "config.pbtxt"
        if not config.is_file():
            raise ValueError(f"missing model config: {config}")
        content = re.sub(r"#.*", "", config.read_text())
        versions = [p for p in directory.iterdir() if p.is_dir() and p.name.isdigit()]
        if not versions:
            raise ValueError(f"missing numeric model version: {directory}")
        if re.search(r'platform\s*:\s*"ensemble"', content):
            continue
        backend = re.search(r'(?:backend|platform)\s*:\s*"([^"]+)"', content)
        defaults = {"tensorrt": "model.plan", "onnxruntime": "model.onnx",
                    "onnxruntime_onnx": "model.onnx", "python": "model.py"}
        if backend is None or backend[1] not in defaults:
            raise ValueError(f"unsupported backend in {config}")
        filename = re.search(r'default_model_filename\s*:\s*"([^"]+)"', content)
        filename = filename[1] if filename else defaults[backend[1]]
        if Path(filename).name != filename or filename in {".", ".."}:
            raise ValueError(f"invalid model filename in {config}: {filename}")
        for version in versions:
            required = [version / filename]
            if backend[1] == "tensorrt":
                required.append(version / "model.onnx")
            for path in required:
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f"missing or empty model artifact: {path}")
            if backend[1] == "tensorrt":
                reference = (version / "model.onnx").relative_to(root).as_posix()
                reference_onnx[reference] = files[reference]
                sidecar = version / (filename + ".json")
                if not sidecar.is_file():
                    missing.append(sidecar.relative_to(repository).as_posix())
                    continue
                metadata = json.loads(sidecar.read_text())
                for key, artifact in (("plan_sha256", version / filename),
                                      ("source_sha256", version / "model.onnx")):
                    if metadata.get(key) != files[artifact.relative_to(root).as_posix()]:
                        raise ValueError(f"{sidecar}: {key} does not match artifact")
                if not metadata.get("build_command") or not metadata.get("precision"):
                    raise ValueError(f"incomplete build provenance: {sidecar}")
    for relative in ("profiles/model_sets.json", "scripts/start_triton.py",
                     "scripts/run_triton_with_rotating_logs.py", "scripts/yolo_orin_pipeline.py",
                     "scripts/tbps_orin_pipeline.py", "scripts/yolo_request_batching.py", "scripts/yolo_orin_metrics.py",
                     "scripts/image_orin_preprocess.py", "scripts/face_orin_preprocess.py",
                     "scripts/orin_trt_runtime.py", "scripts/orin_pipeline_metrics.py", "scripts/image_decode.py", "scripts/image_orin_pipeline.py", "scripts/face_orin_pipeline.py",
                     "deploy/jetson/smoke.py"):
        path = root / relative
        if path.is_file():
            files[relative] = sha256(path)
    return {"files": files, "model_count": len(directories), "missing_provenance": missing,
            "reference_onnx": reference_onnx,
            "model_sets": list(manifest.model_sets) if aliases is None else list(aliases)}


def run(*command, capture=False, **kwargs):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE if capture else None,
                            check=True, **kwargs)
    return result.stdout.strip() if capture else None


def inspect_image(image):
    return json.loads(run("docker", "image", "inspect", image, capture=True))[0]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_env(path, image, model_set="all", model_config="orin_gpu"):
    content = (DEPLOY / ".env.example").read_text()
    content = content.replace("TRITON_IMAGE=\n", f"TRITON_IMAGE={image}\n")
    content = content.replace("TRITON_MODEL_SET=all\n", f"TRITON_MODEL_SET={model_set}\n")
    content = content.replace("TRITON_MODEL_CONFIG=orin_gpu\n", f"TRITON_MODEL_CONFIG={model_config}\n")
    path.write_text(content)


def snapshot(root, context):
    """Copy only release inputs, including empty numeric model directories."""
    aliases = release_aliases(root)
    manifest = start_triton.load_manifest(root / "profiles/model_sets.json")
    models = set(start_triton.resolve_model_set(",".join(aliases), manifest))
    start_triton.validate_repository(root / "model_repository", models)
    for source_dir in ("model_repository", "scripts", "deploy/jetson"):
        for path in sorted((root / source_dir).rglob("*")):
            relative = path.relative_to(root)
            if source_dir == "model_repository" and relative.parts[1] not in models:
                continue
            if source_dir == "scripts" and path.name not in {
                "start_triton.py", "run_triton_with_rotating_logs.py", "yolo_orin_pipeline.py",
                "tbps_orin_pipeline.py", "yolo_request_batching.py", "yolo_orin_metrics.py",
                "image_orin_preprocess.py", "face_orin_preprocess.py",
                "orin_trt_runtime.py", "orin_pipeline_metrics.py", "image_decode.py", "image_orin_pipeline.py", "face_orin_pipeline.py",
            }:
                continue
            if source_dir == "deploy/jetson" and path.name not in {
                "Dockerfile", "Dockerfile.dockerignore", "smoke.py", "release.py", "release.sh", "compose.yaml",
                "compose.gpu.yaml",
                "release_models.json",
            }:
                continue
            if not included(relative):
                continue
            if path.is_symlink():
                raise ValueError(f"symlink is not allowed in release input: {relative}")
            destination = context / relative
            if path.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            elif path.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
    # all must mean all services included in this image, not all development models.
    packaged_manifest = {
        "schema_version": manifest.schema_version,
        "model_sets": {alias: {"entry_point": manifest.model_sets[alias].entry_point,
                               "models": list(manifest.model_sets[alias].models)} for alias in aliases},
    }
    (context / "profiles").mkdir(parents=True, exist_ok=True)
    write_json(context / "profiles/model_sets.json", packaged_manifest)


def archive_reference_onnx(context, archive, release):
    """Keep TensorRT reference graphs with the publisher, outside the image context."""
    for relative in release["reference_onnx"]:
        destination = archive / Path(relative).relative_to("model_repository")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(context / relative), destination)
        del release["files"][relative]


def build(args):
    if platform.machine() not in {"aarch64", "arm64"}:
        raise ValueError("build on the target Jetson ARM64 environment")
    if not re.fullmatch(r"[a-z0-9][a-z0-9./:_-]*:[A-Za-z0-9_][A-Za-z0-9_.-]*", args.image):
        raise ValueError("--image must be an explicit repository:version reference")
    version = args.image.rsplit(":", 1)[1]
    if version in {"latest", "stable"}:
        raise ValueError("use a unique version tag, not latest/stable")
    if args.output.exists():
        raise ValueError(f"output already exists; choose a new release directory: {args.output}")
    run("docker", "info", capture=True)
    if subprocess.run(["docker", "image", "inspect", args.image],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        raise ValueError(f"image tag already exists locally: {args.image}")
    runtime = inspect_image(args.runtime_image)
    if runtime["Architecture"] != "arm64" or runtime["Os"] != "linux":
        raise ValueError("runtime image must be linux/arm64")
    # Locally imported images can have stale/unresolvable RepoDigests. Pin their
    # actual local content rather than requiring another registry download.
    runtime_ref = "jetson-release-runtime:" + runtime["Id"].removeprefix("sha256:")
    run("docker", "tag", runtime["Id"], runtime_ref)
    status = run("git", "-C", str(ROOT), "status", "--porcelain", capture=True)
    commit = run("git", "-C", str(ROOT), "rev-parse", "HEAD", capture=True)
    for relative in ("model_repository", "scripts", "profiles", "deploy"):
        if args.output.is_relative_to(ROOT / relative):
            raise ValueError("release output must be outside source/model directories; use artifacts/releases")
    # Freeze the inputs before hashing/building; never change the live model repository.
    args.output.mkdir(parents=True)
    context = args.output / "context"
    context.mkdir()
    snapshot(ROOT, context)
    release = inventory(context)
    manifest = start_triton.load_manifest(context / "profiles/model_sets.json")
    default_runtime_count = len(start_triton.resolve_runtime_models("all", manifest, "orin_gpu"))
    archive_reference_onnx(context, args.output / "reference_onnx", release)
    release.update({
        "schema_version": 1, "created_at": now(), "version": version,
        "git_commit": commit, "git_dirty": bool(status), "git_status": status,
        "runtime_image": args.runtime_image, "runtime_image_id": runtime["Id"],
        "runtime_repository_digests": runtime.get("RepoDigests", []),
        "architecture": "linux/arm64",
        "default_model_config": "orin_gpu", "default_runtime_model_count": default_runtime_count,
        "l4t_release": Path("/etc/nv_tegra_release").read_text() if Path("/etc/nv_tegra_release").exists() else "unknown",
        "packaging_files": {p.name: sha256(p) for p in (context / "deploy/jetson").iterdir() if p.is_file()},
    })
    write_json(context / "release.json", release)
    write_json(args.output / "release.json", release)
    if release["missing_provenance"]:
        print(f"Candidate only: {len(release['missing_provenance'])} plan provenance records missing.", flush=True)
    run("docker", "build", "--platform", "linux/arm64", "--pull=false", "--network", "none",
        "--build-arg", f"TRITON_RUNTIME_IMAGE={runtime_ref}",
        "--build-arg", f"RELEASE_VERSION={version}", "--build-arg", f"SOURCE_REVISION={commit}",
        "-f", str(context / "deploy/jetson/Dockerfile"), "-t", args.image, str(context))
    image = inspect_image(args.image)
    base_layers = runtime["RootFS"]["Layers"]
    if image["RootFS"]["Layers"][:len(base_layers)] != base_layers:
        raise ValueError("built image does not contain the expected runtime layers")
    write_json(args.output / "build.json", {
        "image": args.image, "image_id": image["Id"], "built_at": now(),
        "model_count": release["model_count"],
        "default_model_config": release["default_model_config"],
        "default_runtime_model_count": release["default_runtime_model_count"],
        "compose_sha256": sha256(context / "deploy/jetson/compose.yaml"),
        "compose_gpu_sha256": sha256(context / "deploy/jetson/compose.gpu.yaml"),
    })
    delivery = args.output / "deploy"
    delivery.mkdir()
    shutil.copyfile(context / "deploy/jetson/compose.yaml", delivery / "compose.yaml")
    shutil.copyfile(context / "deploy/jetson/compose.gpu.yaml", delivery / "compose.gpu.yaml")
    write_env(delivery / ".env", args.image)
    print(f"Candidate built: {args.image}\nDelivery files: {delivery}\nNext: release.sh verify --output {args.output}")


def load_build(output):
    record = json.loads((output / "build.json").read_text())
    if inspect_image(record["image"])["Id"] != record["image_id"]:
        raise ValueError("local image tag changed since build; refusing to use another image")
    return record


def validate_verification(build_record, report):
    if (build_record.get("default_model_config") != "orin_gpu"
            or type(build_record.get("default_runtime_model_count")) is not int
            or build_record["default_runtime_model_count"] < 1):
        raise ValueError("candidate lacks the default GPU runtime contract; build a new candidate")
    if not report or not report.get("passed") or report.get("image_id") != build_record["image_id"]:
        raise ValueError("successful verification for this exact image is required")
    if report.get("model_set") != "all":
        raise ValueError("full-model verification is required to publish the full image")
    smoke_report = report.get("smoke", {})
    if report.get("model_config") != "orin_gpu" or smoke_report.get("model_config") != "orin_gpu":
        raise ValueError("default orin_gpu verification is required to publish the full image")
    if (smoke_report.get("model_set") != "all"
            or smoke_report.get("ready_models") != build_record["default_runtime_model_count"]):
        raise ValueError("actual smoke coverage does not include the full default GPU runtime")


def verify(args):
    record = load_build(args.output)
    args.model_set = ",".join(alias.strip() for alias in args.model_set.split(","))
    manifest = start_triton.load_manifest(args.output / "context/profiles/model_sets.json")
    model_config = args.model_config
    models = start_triton.resolve_runtime_models(args.model_set, manifest, model_config)
    report_name = "verification-orin_gpu" if model_config else "verification"
    report_path = args.output / (report_name + ".json")
    project = "jetson-release-check-" + uuid.uuid4().hex[:12]
    report = {"image_id": record["image_id"], "model_set": args.model_set,
              "model_config": model_config,
              "started_at": now(), "passed": False}
    write_json(report_path, report)
    if sha256(args.output / "deploy/compose.yaml") != record["compose_sha256"]:
        raise ValueError("delivery Compose changed since build; build a new candidate")
    if model_config:
        overlay = args.output / "deploy/compose.gpu.yaml"
        if not overlay.is_file() or sha256(overlay) != record.get("compose_gpu_sha256"):
            raise ValueError("delivery GPU Compose changed or was not frozen at build; build a new candidate")
    env = {key: value for key, value in os.environ.items() if not key.startswith(("TRITON_", "COMPOSE_"))}
    with tempfile.TemporaryDirectory(prefix=project + "-") as temporary:
        folder = Path(temporary)
        shutil.copyfile(args.output / "deploy/compose.yaml", folder / "compose.yaml")
        write_env(folder / ".env", record["image"], args.model_set, model_config)
        with (folder / ".env").open("a") as stream:
            stream.write("\nTRITON_BIND_ADDRESS=127.0.0.1\nTRITON_HTTP_PORT=0\nTRITON_GRPC_PORT=0\nTRITON_METRICS_PORT=0\n")
        compose = ["docker", "compose", "--project-name", project,
                   "--env-file", str(folder / ".env"), "-f", str(folder / "compose.yaml")]
        # Exercise the actual delivered base Compose. An overlay must not mask
        # a missing profile setting in that default deployment path.
        try:
            run(*compose, "up", "-d", "--pull", "never", "--wait", "--wait-timeout", str(args.timeout), env=env)
            container = run(*compose, "ps", "-q", "triton-server", capture=True, env=env)
            actual = json.loads(run("docker", "inspect", container, capture=True))[0]
            if actual["Image"] != record["image_id"] or any(m["Type"] == "bind" for m in actual["Mounts"]):
                raise ValueError("verification must use the built image without host bind mounts")
            result = run("docker", "exec", container, "python3", "/opt/triton/release/smoke.py", capture=True)
            report["smoke"] = json.loads(result)
            if (report["smoke"]["model_set"] != args.model_set
                    or report["smoke"].get("model_config", "") != model_config
                    or report["smoke"]["ready_models"] != len(models)):
                raise ValueError("actual smoke coverage differs from the requested model set/config")
        except BaseException as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                with (args.output / (report_name + ".log")).open("w") as log:
                    subprocess.run([*compose, "logs", "--no-color"], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            except BaseException as exc:
                report.update(passed=False, log_error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                cleanup = subprocess.run([*compose, "down", "--volumes", "--remove-orphans"], env=env)
                report["cleanup_passed"] = cleanup.returncode == 0
                report["passed"] = report["passed"] and report["cleanup_passed"]
                report["finished_at"] = now()
                write_json(report_path, report)
            if cleanup.returncode:
                raise ValueError(f"verification cleanup failed; inspect Compose project {project}")
    # Publish success only after logs, cleanup and temporary-directory removal all
    # complete normally. Interruptions leave the initial failed report in place.
    report["passed"] = True
    write_json(report_path, report)
    print(f"Verification passed: {report_path}")


def publish(args):
    record = load_build(args.output)
    if sha256(args.output / "deploy/compose.yaml") != record["compose_sha256"]:
        raise ValueError("delivery Compose changed since build; build a new candidate")
    if record.get("compose_gpu_sha256"):
        overlay = args.output / "deploy/compose.gpu.yaml"
        if not overlay.is_file() or sha256(overlay) != record["compose_gpu_sha256"]:
            raise ValueError("delivery GPU Compose changed since build; build a new candidate")
    # Read the embedded manifest so edits to the local record cannot hide missing provenance.
    release = json.loads(run("docker", "run", "--rm", "--network", "none", "--entrypoint", "cat",
                             record["image_id"], "/opt/triton/release.json", capture=True))
    if release["git_dirty"]:
        raise ValueError("candidate was built from a dirty worktree; commit reviewed sources and build a new release")
    if release["missing_provenance"]:
        raise ValueError("plan build provenance is missing; recover records or rebuild in the target environment")
    for key in ("default_model_config", "default_runtime_model_count"):
        if key not in release or record.get(key) != release[key]:
            raise ValueError("default runtime contract differs from the embedded release; build a new candidate")
    report_path = args.output / "verification-orin_gpu.json"
    validate_verification(record, json.loads(report_path.read_text()) if report_path.exists() else None)
    if not args.acceptance_report.is_file() or args.acceptance_report.stat().st_size == 0:
        raise ValueError("a nonempty target-device numerical/business acceptance report is required")
    acceptance = args.output / "acceptance-report.txt"
    if args.acceptance_report.resolve() != acceptance.resolve():
        shutil.copyfile(args.acceptance_report, acceptance)
    acceptance_hash = sha256(acceptance)
    remote = subprocess.run(["docker", "manifest", "inspect", record["image"]], text=True, capture_output=True)
    if remote.returncode == 0:
        raise ValueError("remote version tag already exists; publish a new version")
    if not any(message in (remote.stderr + remote.stdout).lower()
               for message in ("no such manifest", "manifest unknown")):
        raise ValueError(f"cannot establish that the remote tag is absent: {remote.stderr.strip()}")
    run("docker", "push", record["image"])
    image = inspect_image(record["image"])
    repository = record["image"].rsplit(":", 1)[0]
    digest = next((ref for ref in image.get("RepoDigests", []) if ref.startswith(repository + "@")), None)
    if digest is None:
        raise ValueError("push completed but no matching repository digest was returned")
    write_env(args.output / "deploy/.env", digest)
    write_json(args.output / "published.json", {
        "image": record["image"], "digest": digest, "image_id": record["image_id"],
        "published_at": now(), "acceptance_sha256": acceptance_hash,
    })
    print(f"Published: {digest}\nDeliver only: {args.output / 'deploy'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="check current model inputs without Docker")
    build_parser = commands.add_parser("build", help="freeze inputs and build a local candidate")
    build_parser.add_argument("--image", required=True)
    build_parser.add_argument("--runtime-image", default=RUNTIME)
    build_parser.add_argument("--output", type=Path, required=True)
    verify_parser = commands.add_parser("verify", help="test an isolated two-file deployment on this Jetson")
    verify_parser.add_argument("--output", type=Path, required=True)
    verify_parser.add_argument("--model-set", default="all")
    verify_parser.add_argument("--model-config", choices=("", "orin_gpu"), default="orin_gpu",
                               help="verify the delivered orin_gpu default; pass an empty value for the CPU baseline")
    verify_parser.add_argument("--timeout", type=int, default=600)
    publish_parser = commands.add_parser("publish", help="explicitly push a verified release")
    publish_parser.add_argument("--output", type=Path, required=True)
    publish_parser.add_argument("--acceptance-report", type=Path, required=True)
    args = parser.parse_args()
    if hasattr(args, "output"):
        args.output = args.output.resolve()
    try:
        if args.command == "check":
            result = inventory(ROOT, aliases=release_aliases(ROOT))
            print(json.dumps({key: value for key, value in result.items() if key != "files"}, ensure_ascii=False, indent=2))
        else:
            {"build": build, "verify": verify, "publish": publish}[args.command](args)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"release: ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    sys.exit(main())
