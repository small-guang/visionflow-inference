"""Prepare the complete GPU identity repository directly from approved sources.

No CPU/GPU inference or Docker mutation occurs here. All source bytes and
license evidence are checked before any runtime model file is published.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile

import onnx

try:
    from .prepare import ROOT, atomic_write, checked_destination, collect_pose, inspect_onnx, render_config, validate_manifest, verify_blob
except ImportError:
    from prepare import ROOT, atomic_write, checked_destination, collect_pose, inspect_onnx, render_config, validate_manifest, verify_blob


OSNET_ONNX_SHA256 = "e766a2c3bdc0200ce20b9be0a05c784d4d81f342c3db40d8e054980b87d47c08"
OSNET_CHECKPOINT_SHA256 = "8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f"
OSNET_REVISION = "01af85e82a9db4f3a4f6ed3a72ed9150bd416d04"


def download(url, path, *, local_ports="18100-18120", proxy=None):
    if not url.startswith("https://") or not re.fullmatch(r"[0-9]{1,5}-[0-9]{1,5}", local_ports):
        raise ValueError("download requires HTTPS and a numeric local port range")
    executable = shutil.which("curl.exe") or shutil.which("curl")
    if not executable:
        raise RuntimeError("curl is required to acquire source files")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, prefix=".download-") as stream:
        temporary = Path(stream.name)
    command = [executable, "-4", "--local-port", local_ports, "--location", "--fail", "--silent", "--show-error",
               "--connect-timeout", "15", "--max-time", "90", "--output", str(temporary)]
    if platform.system() == "Windows":
        command.append("--ssl-revoke-best-effort")
    if proxy:
        command += ["--proxy", proxy]
    try:
        subprocess.run([*command, url], check=True, timeout=100)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def acquire(path, digest, size, urls, *, offline, local_ports, proxy, encoded_contents=False, candidates=()):
    for candidate in (path, *candidates):
        if candidate.is_file():
            return verify_blob(candidate.read_bytes(), digest, size)
    if offline:
        raise FileNotFoundError(path)
    errors = []
    for url in urls:
        try:
            download(url, path, local_ports=local_ports, proxy=proxy)
            data = path.read_bytes()
            if encoded_contents:
                data = base64.b64decode(json.loads(data)["content"], validate=False)
            data = verify_blob(data, digest, size)
            atomic_write(path, data)
            return data
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            path.unlink(missing_ok=True)
            errors.append(type(exc).__name__)
    raise RuntimeError("source acquisition failed: " + ", ".join(errors))


def add_pose(payloads, records, source_repository):
    collected, manifest = collect_pose(source_repository)
    payloads.update(collected)
    records.update(manifest["models"])
    return {key: value for key, value in manifest.items() if key not in {"models", "files"}}


def add_osnet(payloads, records, model_path, evidence_path):
    evidence_path = Path(evidence_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (evidence.get("weights_license") != "MIT" or evidence.get("weight_repository") != "kaiyangzhou/osnet"
            or evidence.get("weight_revision") != OSNET_REVISION
            or evidence.get("source_checkpoint", {}).get("sha256") != OSNET_CHECKPOINT_SHA256):
        raise ValueError("OSNet weight-license evidence does not match the approved source")
    for spec in evidence["files"]:
        verify_blob(checked_destination(evidence_path.parent, spec["file"]).read_bytes(), spec["sha256"], spec["bytes"])
    data = verify_blob(Path(model_path).read_bytes(), OSNET_ONNX_SHA256, 8752792)
    signature = inspect_onnx(data)
    onnx.checker.check_model(data, full_check=True)
    if (signature["inputs"] != [{"name": "image", "dtype": "FP32", "shape": ["batch", 3, 256, 128]}]
            or signature["outputs"][0]["name"] != "embedding"
            or signature["outputs"][0]["shape"][-1] != 512):
        raise ValueError("OSNet graph differs from the expected image/embedding contract")
    config = ('name: "osnet_gpu"\nbackend: "onnxruntime"\nmax_batch_size: 4\n'
              'default_model_filename: "model.onnx"\n'
              'input [{ name: "image" data_type: TYPE_FP32 dims: [3,256,128] }]\n'
              'output [{ name: "embedding" data_type: TYPE_FP32 dims: [512] }]\n'
              'instance_group [{ count: 1 kind: KIND_GPU gpus: [0] }]\n'
              'dynamic_batching { max_queue_delay_microseconds: 2000 }\n')
    payloads["osnet_gpu/1/model.onnx"] = data
    payloads["osnet_gpu/config.pbtxt"] = config.encode()
    records["osnet_gpu"] = {"onnx_sha256": OSNET_ONNX_SHA256, "weights_license": "MIT",
                            "source_checkpoint_sha256": OSNET_CHECKPOINT_SHA256, "source_revision": OSNET_REVISION,
                            "license_evidence": "licenses/osnet/evidence.json", "contract": signature,
                            "embedding_space": "osnet-ain-msmt17-256x128-512-l2", "training_data": "MSMT17; separate terms; not bundled"}


def prepare_identity(sources, cache_root, output_root=ROOT, *, offline=False, local_ports="18100-18120", proxy=None,
                     source_repository=None, model_repository=None, osnet_model=None, osnet_evidence=None):
    validate_manifest(sources)
    cache_root, output_root = Path(cache_root), Path(output_root)
    model_repository = Path(model_repository) if model_repository is not None else output_root / "identity_model_repository"
    if source_repository is not None:
        source, target = Path(source_repository).resolve(), (output_root / "identity_model_repository").resolve()
        if source == target or source.is_relative_to(target) or target.is_relative_to(source):
            raise ValueError("read-only pose source and output model repository must not overlap")
    payloads, models, notices = {}, {}, {}
    for name, spec in sources["models"].items():
        model_cache = checked_destination(cache_root, spec["cache_filename"])
        license_cache = checked_destination(cache_root, spec["license_cache_filename"])
        data = acquire(model_cache, spec["sha256"], spec["size"], spec.get("download_urls", []),
                       offline=offline, local_ports=local_ports, proxy=proxy,
                       candidates=(checked_destination(model_repository, name + "/1/model.onnx"),))
        license_url = ("https://api.github.com/repos/opencv/opencv_zoo/contents/" + spec["license_path"]
                       + "?ref=" + sources["source_revision"])
        license_data = acquire(license_cache, spec["license_sha256"], spec["license_size"], [license_url],
                               offline=offline, local_ports=local_ports, proxy=proxy, encoded_contents=True,
                               candidates=(checked_destination(output_root, "licenses/identity/" + name + "/LICENSE"),
                                           checked_destination(ROOT, "licenses/identity/" + name + "/LICENSE")))
        signature = inspect_onnx(data)
        onnx.checker.check_model(data, full_check=True)
        graph = onnx.load_model_from_string(data)
        payloads[name + "/1/model.onnx"] = data
        payloads[name + "/config.pbtxt"] = render_config(name, signature).encode("utf-8")
        notices[name + "/LICENSE"] = license_data
        models[name] = {"source_repository": sources["source_repository"], "source_revision": sources["source_revision"],
                        "source_path": spec["source_path"], "source_sha256": spec["sha256"],
                        "weights_license": spec["weights_license"], "license_sha256": spec["license_sha256"],
                        "license_scope_url": spec["license_scope_url"], "contract": signature,
                        "operators": dict(Counter(node.op_type for node in graph.graph.node)),
                        "input_semantics": spec.get("input_semantics"), "output_semantics": spec.get("output_semantics")}
    pose_provenance = None
    if source_repository is not None:
        pose_provenance = add_pose(payloads, models, source_repository)
    if osnet_model is not None:
        add_osnet(payloads, models, osnet_model, osnet_evidence or ROOT / "licenses/osnet/evidence.json")
    files = {}
    for relative, data in payloads.items():
        atomic_write(checked_destination(output_root, "identity_model_repository/" + relative), data)
        files[relative] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    for relative, data in notices.items():
        atomic_write(checked_destination(output_root, "licenses/identity/" + relative), data)
    manifest = {"schema_version": 1, "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
                "gpu_verified": False, "runtime": "Triton native ONNX Runtime CUDA EP; pose native TensorRT",
                "placement": "KIND_GPU, GPU device0; validate loaded providers and actual requests on target",
                "cpu_fallback_claim": "No claim of disabled ORT operator fallback; stock24.12 has no generic session.disable_cpu_ep_fallback parameter",
                "models": models, "files": files, "scope": "Preparation and graph checks only; not inference accuracy or identity proof",
                "toolchain": {"python": platform.python_version(), "onnx": onnx.__version__}}
    if pose_provenance is not None:
        manifest["pose_provenance"] = pose_provenance
    atomic_write(checked_destination(output_root, "identity-model-manifest.json"),
                 (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode())
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "identity-source-manifest.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/identity")
    parser.add_argument("--source-repository", type=Path, required=True,
                        help="Read-only approved pose model_repository; no pose-only staging tree is needed")
    parser.add_argument("--model-repository", type=Path, default=ROOT / "identity_model_repository",
                        help="Existing final identity model repository to reuse after hash verification")
    parser.add_argument("--output-dir", type=Path, default=ROOT,
                        help="Destination containing identity_model_repository and its manifest")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--local-ports", default="18100-18120")
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--osnet-onnx", type=Path)
    args = parser.parse_args(argv)
    result = prepare_identity(json.loads(args.manifest.read_text(encoding="utf-8")), args.cache_dir,
                              args.output_dir,
                              offline=args.offline, local_ports=args.local_ports, proxy=args.proxy,
                              source_repository=args.source_repository, model_repository=args.model_repository,
                              osnet_model=args.osnet_onnx or args.model_repository / "osnet_gpu/1/model.onnx")
    print(json.dumps({"models": list(result["models"]), "gpu_verified": False,
                      "repository": str(args.output_dir / "identity_model_repository")}))


if __name__ == "__main__":
    main()
