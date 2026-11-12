"""Shared validation and read-only pose collection for the final GPU profile.

The default local path reuses the existing public pose preprocessing, a known
ONNX, and an explicitly fingerprinted local TensorRT engine. It never writes to
the supplied source repository or the production model_repository.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import re
import tempfile


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
NATIVE = "yolo26m_pose_onnx"
POSE_MODELS = ("yolo26m_pose_preprocess", NATIVE, "yolo26m_pose_postprocess", "yolo26m_pose_ensemble")
PLAN_SHA256 = "955294679779502dbd8598f627b05f4041d804f95b393d71f1f5205f3ed36238"
PLAN_SIZE = 48933260
AGPL_LICENSE_SHA256 = "0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0"
AGPL_LICENSE_SIZE = 34523
SAFE_MODEL = re.compile(r"^[a-zA-Z0-9_-]+$")
DTYPES = {1: "FP32", 2: "UINT8", 3: "INT8", 4: "UINT16", 5: "INT16", 6: "INT32", 7: "INT64", 8: "BYTES", 9: "BOOL", 10: "FP16", 11: "FP64", 12: "UINT32", 13: "UINT64"}


def checked_destination(root, relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
            or any(part in {".", ".."} for part in relative.split("/"))):
        raise ValueError("unsafe relative path")
    portable = PurePosixPath(relative)
    if portable.is_absolute() or str(portable) != relative:
        raise ValueError("unsafe relative path")
    root = Path(root).resolve()
    destination = (root / relative).resolve()
    if destination == root or not destination.is_relative_to(root):
        raise ValueError("output path escapes root")
    return destination


def verify_blob(data, expected_sha256, expected_size):
    if len(data) != expected_size:
        raise ValueError("artifact size mismatch")
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("artifact SHA256 mismatch")
    return data


def verify_agpl_license(data):
    return verify_blob(data, AGPL_LICENSE_SHA256, AGPL_LICENSE_SIZE)


def validate_manifest(manifest):
    """Validate frozen OpenCV Zoo candidates before any future remote acquisition."""
    if manifest.get("schema_version") != 1 or not re.fullmatch(r"[a-f0-9]{40}", manifest.get("source_revision", "")):
        raise ValueError("source revision must be a frozen commit")
    if manifest.get("source_repository") != "https://github.com/opencv/opencv_zoo":
        raise ValueError("unapproved source repository")
    if not manifest.get("models"):
        raise ValueError("no models")
    for name, spec in manifest["models"].items():
        if not SAFE_MODEL.fullmatch(name):
            raise ValueError("unsafe model name")
        if spec.get("weights_license") not in {"MIT", "Apache-2.0"}:
            raise ValueError("unknown weight license")
        for field in ("source_path", "license_path"):
            checked_destination(ROOT, spec[field])
        for digest in ("sha256", "license_sha256"):
            if not re.fullmatch(r"[a-f0-9]{64}", spec.get(digest, "")):
                raise ValueError("invalid source/license SHA256")
        for field in ("size", "license_size"):
            if type(spec.get(field)) is not int or spec[field] <= 0:
                raise ValueError("invalid source/license size")
        prefix = manifest["source_repository"] + "/blob/" + manifest["source_revision"] + "/"
        if not spec.get("license_scope_url", "").startswith(prefix):
            raise ValueError("license scope evidence must use frozen source revision")


def inspect_onnx(data):
    """Read actual ONNX signatures/metadata; this performs no model inference."""
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("Preparation requires onnx (install gpu_demo/requirements.txt)") from exc
    try:
        model = onnx.load_model_from_string(data)
        def describe(value):
            tensor = value.type.tensor_type
            if tensor.elem_type not in DTYPES:
                raise ValueError("unsupported ONNX tensor dtype")
            return {"name": value.name, "dtype": DTYPES[tensor.elem_type],
                    "shape": [dim.dim_value if dim.HasField("dim_value") else dim.dim_param or -1
                              for dim in tensor.shape.dim]}
        initializer_names = {value.name for value in model.graph.initializer}
        inputs = [describe(value) for value in model.graph.input if value.name not in initializer_names]
        outputs = [describe(value) for value in model.graph.output]
        if not inputs or not outputs:
            raise ValueError("ONNX graph has no tensor contract")
        # Do not expose exporter host paths from description metadata.
        metadata = {item.key: item.value for item in model.metadata_props
                    if item.key in {"author", "license", "version", "task", "stride", "end2end", "kpt_shape", "names"}}
        return {"inputs": inputs, "outputs": outputs, "metadata": metadata}
    except Exception as exc:
        raise ValueError("invalid ONNX protobuf or tensor contract") from exc


def render_config(name, signature):
    if not SAFE_MODEL.fullmatch(name):
        raise ValueError("unsafe model name")
    lines = [f'name: "{name}"', 'backend: "onnxruntime"', 'default_model_filename: "model.onnx"',
             "max_batch_size: 0", "instance_group [{ count: 1 kind: KIND_GPU gpus: [0] }]"]
    for direction in ("input", "output"):
        lines.append(direction + " [")
        for index, spec in enumerate(signature[direction + "s"]):
            if index:
                lines[-1] += ","
            if not re.fullmatch(r"[A-Za-z0-9_.:/-]+", spec["name"]) or spec["dtype"] not in DTYPES.values():
                raise ValueError("unsafe tensor contract")
            shape = [dim if type(dim) is int and dim > 0 else -1 for dim in spec["shape"]]
            lines.append('  { name: "' + spec["name"] + '" data_type: TYPE_' + spec["dtype"]
                         + " dims: [" + ", ".join(map(str, shape)) + "] }")
        lines.append("]")
    return "\n".join(lines) + "\n"


def validate_native_config(config):
    if (not re.search(r'backend\s*:\s*"tensorrt"', config)
            or "KIND_GPU" not in config or "KIND_CPU" in config
            or not re.search(r'default_model_filename\s*:\s*"model_throughput.plan"', config)):
        raise ValueError("native configuration must use TensorRT GPU execution")


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".prepare-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def collect_pose(source_repository):
    """Validate and collect the pose closure without publishing a staging tree."""
    source = Path(source_repository).resolve()
    native_onnx = checked_destination(source, NATIVE + "/1/model.onnx")
    catalog = json.loads((PROJECT / "tools/model_pipeline/catalog.json").read_text(encoding="utf-8"))
    spec = catalog["models"][NATIVE]
    onnx_data = verify_blob(native_onnx.read_bytes(), spec["onnx_sha256"], spec["onnx_size"])
    signature = inspect_onnx(onnx_data)
    if "AGPL-3.0" not in signature["metadata"].get("license", ""):
        raise ValueError("known model must retain its embedded AGPL-3.0 weight license")
    import onnx
    onnx.checker.check_model(onnx_data, full_check=True)
    plan = checked_destination(source, NATIVE + "/1/model_throughput.plan")
    plan_data = verify_blob(plan.read_bytes(), PLAN_SHA256, PLAN_SIZE)
    license_data = verify_agpl_license((ROOT / "licenses/Ultralytics-AGPL-3.0.txt").read_bytes())
    # Collect and verify everything before publishing any generated file.
    payloads = {NATIVE + "/1/model.onnx": onnx_data,
                NATIVE + "/1/model_throughput.plan": plan_data}
    for model in POSE_MODELS:
        model_root = PROJECT / "model_repository" / model
        config_data = (model_root / "config.pbtxt").read_bytes()
        if model == NATIVE:
            validate_native_config(config_data.decode("utf-8"))
        payloads[model + "/config.pbtxt"] = config_data
        if model != NATIVE:
            for script in (model_root / "1").glob("*.py"):
                payloads[model + "/1/" + script.name] = script.read_bytes()
    files = {}
    for relative, data in payloads.items():
        files["model_repository/" + relative] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    source_identity = catalog["sources"][spec["source"]]
    manifest = {
        "schema_version": 1, "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "preparation_toolchain": {"python": platform.python_version(), "onnx": onnx.__version__,
                                  "catalog_sha256": hashlib.sha256((PROJECT / "tools/model_pipeline/catalog.json").read_bytes()).hexdigest()},
        "mode": "local-authorized-tensorrt-gpu", "entry_point": "yolo26m_pose_ensemble",
        "runtime": {"backend": "tensorrt", "gpu_required": True, "cpu_fallback": False,
                    "engine_build_provenance": "not available; exact local engine fingerprint preserved",
                    "engine_target_compatibility": "requires actual GPU loading and inference verification"},
        "models": {NATIVE: {"source_model": spec["source"], "source_urls": source_identity["urls"],
                             "source_checkpoint_sha256": source_identity["sha256"],
                             "onnx_sha256": spec["onnx_sha256"], "plan_sha256": PLAN_SHA256,
                             "onnx_validation": "onnx.checker full_check plus graph contract/embedded license",
                             "weights_license": "AGPL-3.0", "upstream_code_license": "AGPL-3.0",
                             "project_pipeline_code_license": "Apache-2.0", "contract": signature}},
        "model_closure": list(POSE_MODELS), "files": files,
        "license_evidence": {"embedded_onnx": signature["metadata"]["license"],
                             "official_url": "https://github.com/ultralytics/ultralytics/blob/v8.4.61/LICENSE",
                             "local_file": "licenses/Ultralytics-AGPL-3.0.txt",
                             "license_sha256": hashlib.sha256(license_data).hexdigest()},
        "scope": "GPU inference assets only; preparation does not claim runtime success or business accuracy",
    }
    return payloads, manifest
