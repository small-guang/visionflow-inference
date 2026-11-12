#!/usr/bin/env python3
"""Check embedded bytes and exercise public Triton chains on the target GPU."""

import argparse
import hashlib
from importlib import metadata as packages
import io
import json
import os
from pathlib import Path
import struct
import sys
import urllib.error
import urllib.request

import numpy as np
from PIL import Image


BASE = "http://127.0.0.1:8000"
DTYPES = {"FP32": "<f4", "FP64": "<f8", "INT64": "<i8", "INT32": "<i4",
          "INT16": "<i2", "INT8": "i1", "UINT8": "u1", "UINT32": "<u4", "BOOL": "?"}
YOLO_SHAPES = ((192, 640), (384, 640), (480, 640), (640, 640),
               (640, 480), (640, 384), (640, 192))


def integrity():
    release = json.loads(Path("/opt/triton/release.json").read_text())
    for source in release.get("reference_onnx", {}):
        target = Path("/models") / Path(source).relative_to("model_repository")
        if target.exists():
            raise ValueError(f"TensorRT reference ONNX must not be shipped: {target}")
    for source, expected in release["files"].items():
        if source.startswith("model_repository/"):
            target = Path("/models") / source.removeprefix("model_repository/")
        elif source.startswith("scripts/"):
            target = Path("/usr/local/bin") / Path(source).name
        elif source == "profiles/model_sets.json":
            target = Path("/opt/triton/model_sets.json")
        elif source == "deploy/jetson/smoke.py":
            target = Path("/opt/triton/release/smoke.py")
        else:
            raise ValueError(f"unrecognized release file: {source}")
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"embedded file hash mismatch: {target}")
    return len(release["files"])


def request(path, data=None, headers=None):
    req = urllib.request.Request(BASE + path, data=data, headers=headers or {})
    try:
        return urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as exc:
        raise ValueError(f"{path}: HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc


def get_json(path):
    with request(path) as response:
        return json.load(response)


def probe_input(spec, batch, hw):
    """Use valid synthetic images/landmarks; preserve each public tensor contract."""
    name, datatype = spec["name"], spec["datatype"]
    height, width = hw
    pixels = (np.arange(height * width * 3, dtype=np.uint32) % 251).astype(np.uint8).reshape(height, width, 3)
    if name in {"RAW_IMAGE", "RAW_FACE_IMAGE"}:
        return np.repeat(pixels[None], batch, axis=0)
    if name == "LANDMARK":
        points = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                           [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
        points *= np.array([width / 112, height / 112], dtype=np.float32)
        return np.repeat(points[None], batch, axis=0)
    if name == "BBOX":
        return np.tile(np.array([0, 0, width - 1, height - 1], dtype=np.float32), (batch, 1))
    if name == "IMAGE_BYTES":
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        encoded = buffer.getvalue()
        if datatype == "BYTES":
            return np.array([[encoded]] * batch, dtype=object)
        return np.repeat(np.frombuffer(encoded, dtype=np.uint8)[None], batch, axis=0)
    if datatype == "BYTES":
        return np.array([["a person walking".encode()]] * batch, dtype=object)
    shape = [batch, *spec["shape"][1:]]
    if any(dimension < 0 for dimension in shape):
        raise ValueError(f"no probe shape defined for {name}: {shape}")
    return np.zeros(shape, dtype=DTYPES[datatype])


def validate_outputs(header, body, expected, batch):
    outputs = header.get("outputs", [])
    contract = {spec["name"]: spec for spec in expected}
    if len(outputs) != len(contract) or {output["name"] for output in outputs} != set(contract):
        raise ValueError("output names differ from model metadata")
    offset = 0
    for output in outputs:
        spec = contract[output["name"]]
        shape = output["shape"]
        if not shape or shape[0] != batch:
            raise ValueError(f"output batch mismatch: {output['name']}")
        if (output["datatype"] != spec["datatype"] or len(shape) != len(spec["shape"])
                or any(want >= 0 and got != want for got, want in zip(shape, spec["shape"]))):
            raise ValueError(f"output schema mismatch: {output['name']}")
        size = output["parameters"]["binary_data_size"]
        dtype = np.dtype(DTYPES[output["datatype"]])
        if size != int(np.prod(shape)) * dtype.itemsize or offset + size > len(body):
            raise ValueError(f"output byte size mismatch: {output['name']}")
        array = np.frombuffer(body[offset:offset + size], dtype=dtype)
        if not np.isfinite(array).all():
            raise ValueError(f"output contains non-finite values: {output['name']}")
        offset += size
    if offset != len(body):
        raise ValueError("unexpected trailing output byte size")


def infer(name, metadata, batch, hw):
    header = {"inputs": [], "outputs": [
        {"name": item["name"], "parameters": {"binary_data": True}}
        for item in metadata["outputs"]
    ]}
    buffers = []
    for spec in metadata["inputs"]:
        array = probe_input(spec, batch, hw)
        if spec["datatype"] == "BYTES":
            content = b"".join(struct.pack("<I", len(value)) + value for value in array.flat)
        else:
            content = array.astype(DTYPES[spec["datatype"]], copy=False).tobytes()
        buffers.append(content)
        header["inputs"].append({"name": spec["name"], "datatype": spec["datatype"],
                                 "shape": list(array.shape),
                                 "parameters": {"binary_data_size": len(content)}})
    encoded = json.dumps(header).encode()
    with request(f"/v2/models/{name}/infer", encoded + b"".join(buffers), {
        "Content-Type": "application/octet-stream", "Inference-Header-Content-Length": str(len(encoded)),
    }) as response:
        data = response.read()
        length = int(response.headers["Inference-Header-Content-Length"])
    validate_outputs(json.loads(data[:length]), data[length:], metadata["outputs"], batch)


def smoke():
    sys.path.insert(0, "/usr/local/bin")
    import start_triton
    manifest = start_triton.load_manifest(Path("/opt/triton/model_sets.json"))
    model_set = ",".join(alias.strip() for alias in os.environ.get("TRITON_MODEL_SET", "all").split(","))
    model_config = os.environ.get("TRITON_MODEL_CONFIG", "")
    models = start_triton.resolve_runtime_models(model_set, manifest, model_config)
    with request("/v2/health/ready"):
        pass
    for name in models:
        with request(f"/v2/models/{name}/ready"):
            pass
        config = get_json(f"/v2/models/{name}/config")
        if model_config == "orin_gpu" and name in start_triton.ORIN_GPU_PREPROCESSORS:
            validate_gpu_preprocessing(name, config)
        if model_config == "orin_gpu" and name in start_triton.ORIN_FUSED_PREPROCESSING_ENTRY_POINTS:
            validate_fused_pipeline(name, config)
        if model_config == "orin_gpu" and name in start_triton.ORIN_GPU_ENTRY_POINTS:
            if config.get("backend") != "python" or not any(
                group.get("kind") == "KIND_GPU" for group in config.get("instance_group", [])
            ):
                raise ValueError(f"Orin GPU wrapper is not configured for Python GPU execution: {name}")
            if name in {"yolo26m_pose_ensemble", "yolo26s_seg_ensemble"}:
                batching = config.get("dynamic_batching")
                groups = config.get("instance_group", [])
                if (config.get("max_batch_size") != 4 or not isinstance(batching, dict)
                        or int(batching.get("max_queue_delay_microseconds", 0)) != 0
                        or batching.get("preferred_batch_size")
                        or any(group.get("kind") != "KIND_GPU" for group in groups)
                        or sum(group.get("count", 1) for group in groups) != 2):
                    raise ValueError(f"Orin cross-request batching requires B4, two GPU instances and no added wait: {name}")
        if config.get("backend") in {"tensorrt", "onnxruntime"}:
            if not any(group.get("kind") == "KIND_GPU" for group in config.get("instance_group", [])):
                raise ValueError(f"inference node is not configured for GPU: {name}")
    probes = []
    aliases = manifest.model_sets if model_set == "all" else [alias.strip() for alias in model_set.split(",")]
    for alias in aliases:
        name = manifest.model_sets[alias].entry_point
        metadata = get_json(f"/v2/models/{name}")
        shapes = YOLO_SHAPES if alias.startswith("yolo") else ((224, 224),)
        for batch in (1, 2, 3, 4):
            for hw in shapes:
                infer(name, metadata, batch, hw)
                probes.append({"model": name, "batch": batch, "image_hw": hw})
    versions = {}
    for distribution in ("numpy", "pillow", "regex", "ftfy", "opencv-python-headless", "scikit-image"):
        versions[distribution] = packages.version(distribution)
    loaded_libraries = set()
    for process in Path("/proc").glob("[0-9]*"):
        try:
            if (process / "comm").read_text().strip() == "tritonserver":
                for line in (process / "maps").read_text().splitlines():
                    if "libnvinfer.so" in line:
                        loaded_libraries.add(line.split()[-1])
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return {"model_set": model_set, "model_config": model_config,
            "ready_models": len(models), "probes": probes,
            "packages": versions, "triton": get_json("/v2"),
            "loaded_tensorrt_libraries": sorted(loaded_libraries),
            "cuda_version": os.environ.get("CUDA_VERSION"),
            "scope": "synthetic inference, batch/shape and finite outputs; numerical/business acceptance is separate"}


def validate_gpu_preprocessing(name, config):
    """Reject a CPU fallback even when the selected instance is KIND_GPU."""
    groups = config.get("instance_group", [])
    backend = config.get("parameters", {}).get("preprocess_backend", {}).get("string_value")
    if (config.get("backend") != "python" or backend != "orin_gpu"
            or config.get("max_batch_size") != 4 or not groups
            or any(group.get("kind") != "KIND_GPU" for group in groups)):
        raise ValueError(f"Orin GPU preprocessing is not enabled: {name}")


def validate_fused_pipeline(name, config):
    """Require the actual fused implementation, not an old ensemble profile."""
    groups = config.get("instance_group", [])
    if (config.get("backend") != "python" or config.get("max_batch_size") != 4
            or config.get("parameters", {}).get("pipeline_backend", {}).get("string_value") != "orin_gpu"
            or not groups or any(group.get("kind") != "KIND_GPU" for group in groups)):
        raise ValueError(f"Orin fused GPU pipeline is not enabled: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integrity-only", action="store_true")
    args = parser.parse_args()
    result = {"verified_files": integrity()}
    if not args.integrity_only:
        result.update(smoke())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
