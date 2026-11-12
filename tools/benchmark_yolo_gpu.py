"""Compare local real-image YOLO CPU/GPU ensembles over Triton binary HTTP.

Images are read locally. The JSON report contains paths, hashes and numeric
parity summaries, never image pixels or rendered face crops.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

import numpy as np
from PIL import Image


MODELS = {
    "face": "yolov8n_face_ensemble",
    "pose": "yolo26m_pose_ensemble",
    "seg": "yolo26s_seg_ensemble",
}
OUTPUTS = {
    MODELS["face"]: ("DETECTIONS", "NUM_DETECTIONS"),
    MODELS["pose"]: ("DETECTIONS", "KEYPOINTS", "NUM_DETECTIONS"),
    MODELS["seg"]: ("DETECTIONS", "MASK_CROPS_PACKED", "NUM_DETECTIONS"),
}
TOLERANCES = {"coordinate_atol_pixels": 0.25, "confidence_atol": 0.0001,
              "class_and_count_exact": True, "mask_bits_exact": True,
              "relative_tolerance": 0}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sample_images(dataset_root, limit=64):
    """Take all test images, then sorted first/middle/last originals per size."""
    if not 1 <= limit <= 64:
        raise ValueError("limit must be between 1 and 64")
    root = Path(dataset_root).resolve()
    if any("snapshot" in part.lower() for part in root.parts):
        raise ValueError("Snapshot dataset roots are excluded")
    candidates = []
    test_dir = root / "test"
    if test_dir.is_dir():
        candidates.extend(sorted(path for path in test_dir.iterdir()
                                 if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS))

    original_root = (root / "face_person_ab_v1" / "pool" / "images" / "original").resolve()
    manifest = root / "face_person_ab_v1" / "pool" / "samples.jsonl"
    originals = {}
    if manifest.is_file():
        with manifest.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                path = (root / "face_person_ab_v1" / row["original_path"]).resolve()
                if (not path.is_relative_to(original_root)
                        or any("snapshot" in part.lower() for part in path.parts)):
                    raise ValueError(f"Original image path escapes the original pool: {path}")
                dimensions = (int(row["original_width"]), int(row["original_height"]))
                if path in originals and originals[path] != dimensions:
                    raise ValueError(f"Conflicting original image dimensions: {path}")
                originals[path] = dimensions
        groups = defaultdict(list)
        for path, dimensions in originals.items():
            groups[dimensions].append(path)
        for dimensions in sorted(groups):
            paths = sorted(groups[dimensions])
            for index in sorted({0, len(paths) // 2, len(paths) - 1}):
                candidates.append(paths[index])

    samples = []
    for path in dict.fromkeys(candidates):
        path = path.resolve()
        if (not path.is_relative_to(root)
                or any("snapshot" in part.lower() for part in path.parts)):
            raise ValueError(f"Sample path escapes the dataset root: {path}")
        with Image.open(path) as image:
            width, height = image.size
        if path in originals and (width, height) != originals[path]:
            raise ValueError(f"Original image dimensions differ from the manifest: {path}")
        samples.append({"id": path.relative_to(root).as_posix(), "path": str(path),
                        "width": width, "height": height,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        if len(samples) == limit:
            break
    if not samples:
        raise ValueError("No test or original pool images were found")
    return samples


def build_batches(samples, batch_size):
    """Group equal raw dimensions and explicitly record repeated padding slots."""
    if batch_size not in (1, 4):
        raise ValueError("Only B1 and B4 are supported")
    groups = defaultdict(list)
    for sample in samples:
        groups[(sample["width"], sample["height"])].append(sample)
    batches = []
    for (width, height), group in groups.items():
        for offset in range(0, len(group), batch_size):
            selected = group[offset:offset + batch_size]
            original_count = len(selected)
            selected = [selected[index % original_count] for index in range(batch_size)]
            batches.append({"samples": selected,
                            "shape": [batch_size, height, width, 3],
                            "repeated_slots": list(range(original_count, batch_size))})
    return batches


def compare_outputs(cpu, gpu):
    """Compare complete output tensors using the existing stable slot order."""
    result = {"passed": True, "errors": [], "outputs": {}, "masks": []}

    def fail(message):
        result["passed"] = False
        result["errors"].append(message)

    if set(cpu) != set(gpu) or not {"DETECTIONS", "NUM_DETECTIONS"}.issubset(cpu):
        fail("Output names differ or required detection outputs are missing")
        return result
    for name in cpu:
        a, b = np.asarray(cpu[name]), np.asarray(gpu[name])
        result["outputs"][name] = {"cpu_shape": list(a.shape), "gpu_shape": list(b.shape),
                                    "cpu_dtype": str(a.dtype), "gpu_dtype": str(b.dtype)}
        if a.shape != b.shape or a.dtype != b.dtype:
            fail(f"{name}: shape or dtype differs")
        elif not np.isfinite(a).all() or not np.isfinite(b).all():
            fail(f"{name}: contains nonfinite values")
    if result["errors"]:
        return result

    counts_a, counts_b = cpu["NUM_DETECTIONS"], gpu["NUM_DETECTIONS"]
    detections_a, detections_b = cpu["DETECTIONS"], gpu["DETECTIONS"]
    if (detections_a.ndim != 3 or detections_a.shape[-1] != 6
            or counts_a.shape != (detections_a.shape[0], 1)
            or not 1 <= detections_a.shape[1] <= 100
            or np.any(counts_a < 0) or np.any(counts_b < 0)
            or np.any(counts_a > detections_a.shape[1]) or np.any(counts_b > detections_b.shape[1])):
        fail("Invalid detection/count output contract")
        return result
    result["counts"] = {"cpu": counts_a[:, 0].tolist(), "gpu": counts_b[:, 0].tolist(),
                        "equal": bool(np.array_equal(counts_a, counts_b))}
    if not result["counts"]["equal"]:
        fail("Detection counts differ")

    def numeric(name, left, right, tolerance, key):
        difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
        metric = {"max_abs_error": float(difference.max(initial=0)),
                  "mismatched_values": int(np.count_nonzero(difference > tolerance)),
                  "atol": tolerance}
        result["outputs"][name][key] = metric
        if metric["mismatched_values"]:
            fail(f"{name}.{key}: exceeds absolute tolerance")

    numeric("DETECTIONS", detections_a[..., :4], detections_b[..., :4],
            TOLERANCES["coordinate_atol_pixels"], "boxes")
    numeric("DETECTIONS", detections_a[..., 4], detections_b[..., 4],
            TOLERANCES["confidence_atol"], "confidence")
    numeric("DETECTIONS", detections_a[..., 5], detections_b[..., 5], 0, "classes")
    if "KEYPOINTS" in cpu:
        a, b = cpu["KEYPOINTS"], gpu["KEYPOINTS"]
        if a.shape != (*detections_a.shape[:2], 17, 3):
            fail("Invalid KEYPOINTS output contract")
        else:
            numeric("KEYPOINTS", a[..., :2], b[..., :2],
                    TOLERANCES["coordinate_atol_pixels"], "coordinates")
            numeric("KEYPOINTS", a[..., 2], b[..., 2],
                    TOLERANCES["confidence_atol"], "confidence")
    if "MASK_CROPS_PACKED" in cpu:
        a, b = cpu["MASK_CROPS_PACKED"], gpu["MASK_CROPS_PACKED"]
        if a.shape != (*detections_a.shape[:2], 160, 20) or a.dtype != np.uint8:
            fail("Invalid MASK_CROPS_PACKED output contract")
        else:
            left = np.unpackbits(a, axis=-1, count=160, bitorder="big").astype(bool)
            right = np.unpackbits(b, axis=-1, count=160, bitorder="big").astype(bool)
            mismatch = np.count_nonzero(left != right, axis=(-2, -1))
            intersection = np.count_nonzero(left & right, axis=(-2, -1))
            union = np.count_nonzero(left | right, axis=(-2, -1))
            result["outputs"]["MASK_CROPS_PACKED"]["total_bit_mismatch"] = int(mismatch.sum())
            if mismatch.any():
                fail("MASK_CROPS_PACKED: unpacked bits differ")
            for batch_index in range(left.shape[0]):
                for slot in range(max(int(counts_a[batch_index, 0]), int(counts_b[batch_index, 0]))):
                    bits = int(mismatch[batch_index, slot])
                    pixel_union = int(union[batch_index, slot])
                    result["masks"].append({"batch_index": batch_index, "slot": slot,
                                            "bit_mismatch": bits,
                                            "bit_mismatch_fraction": bits / (160 * 160),
                                            "iou": (float(intersection[batch_index, slot]) / pixel_union
                                                    if pixel_union else 1.0)})
    return result


def processing_statistics(before, after):
    """Subtract Triton's server-side timing counters for measured requests."""
    old = {(row["name"], row.get("version")): row for row in before["model_stats"]}
    phases = ("compute_input", "compute_infer", "compute_output")

    def difference(previous, current):
        values = {}
        for phase in phases:
            a, b = previous.get(phase, {}), current.get(phase, {})
            count = int(b.get("count", 0)) - int(a.get("count", 0))
            ns = int(b.get("ns", 0)) - int(a.get("ns", 0))
            if count < 0 or ns < 0:
                raise ValueError("Triton processing counters reset during measurement")
            values[phase] = {"count": count, "total_ms": ns / 1e6,
                             "mean_ms": ns / (count * 1e6) if count else None}
        return values

    result = []
    for row in after["model_stats"]:
        previous = old.get((row["name"], row.get("version")), {})
        values = {"name": row["name"], "version": row.get("version"),
                  **difference(previous.get("inference_stats", {}), row.get("inference_stats", {})),
                  "batch_stats": []}
        old_batches = {int(value["batch_size"]): value for value in previous.get("batch_stats", [])}
        for batch in row.get("batch_stats", []):
            batch_size = int(batch["batch_size"])
            values["batch_stats"].append({"batch_size": batch_size,
                                          **difference(old_batches.get(batch_size, {}), batch)})
        result.append(values)
    return result


class HttpInference:
    def __init__(self, urls, models):
        import tritonclient.http as httpclient

        self.httpclient = httpclient
        self.clients = {}
        self.metadata = {}
        self.model_configurations = {}
        for url in dict.fromkeys(urls):
            parsed = urlsplit(url if "://" in url else "http://" + url)
            if parsed.path not in ("", "/") or not parsed.hostname:
                raise ValueError("Triton URL must contain only host and port")
            client = httpclient.InferenceServerClient(url=parsed.netloc, ssl=parsed.scheme == "https",
                                                      connection_timeout=30, network_timeout=120)
            self.clients[url] = client
            if not client.is_server_ready():
                raise RuntimeError(f"Triton server is not ready: {url}")
            for model in models:
                if not client.is_model_ready(model):
                    raise RuntimeError(f"Triton model is not ready: {url}/{model}")
            self.metadata[url] = client.get_server_metadata()
            names = [model.removesuffix("_ensemble") + "_" + suffix
                     for model in models for suffix in ("ensemble", "preprocess", "onnx", "postprocess")]
            # An Orin pipeline loads only its public model: its CUDA helpers and
            # TensorRT plan execute in-process rather than as child models.
            self.model_configurations[url] = {
                name: client.get_model_config(name) for name in names
                if client.is_model_ready(name)
            }

    def __call__(self, url, model, image_batch, output_names):
        tensor = self.httpclient.InferInput("RAW_IMAGE", image_batch.shape, "UINT8")
        tensor.set_data_from_numpy(image_batch, binary_data=True)
        result = self.clients[url].infer(
            model, inputs=[tensor],
            outputs=[self.httpclient.InferRequestedOutput(name, binary_data=True)
                     for name in output_names],
        )
        outputs = {name: result.as_numpy(name) for name in output_names}
        if any(value is None for value in outputs.values()):
            raise RuntimeError(f"Missing requested output from {url}/{model}")
        return outputs

    def statistics(self, url, model):
        prefix = model.removesuffix("_ensemble")
        names = {prefix + "_" + suffix for suffix in ("ensemble", "preprocess", "onnx", "postprocess")}
        stats = self.clients[url].get_inference_statistics()
        return {"model_stats": [row for row in stats["model_stats"] if row["name"] in names]}

    def close(self):
        for client in self.clients.values():
            client.close()


def run_benchmark(args, infer=None):
    import cv2

    samples = sample_images(args.dataset_root, args.limit)
    report = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "cpu_url": args.cpu_url, "gpu_url": args.gpu_url,
              "dataset_root": str(Path(args.dataset_root).resolve()), "samples": samples,
              "models": args.models, "warmup": args.warmup, "repeats": args.repeats,
              "timing_scope": "Client binary HTTP request/response including serialization; excludes image decode",
              "server_processing_scope": "Triton counter deltas during measured requests; excludes warmup and HTTP transit",
              "input_color_order": "BGR", "slot_matching": "existing stable output order",
              "tolerances": TOLERANCES, "passed": True, "runs": []}
    runner = HttpInference((args.cpu_url, args.gpu_url), args.models) if infer is None else None
    infer = runner if runner is not None else infer
    if runner is not None:
        report["servers"] = runner.metadata
        report["server_model_configurations"] = runner.model_configurations
    try:
        for model in args.models:
            for batch_size in (1, 4):
                for batch_index, batch in enumerate(build_batches(samples, batch_size)):
                    images = []
                    for sample in batch["samples"]:
                        image = cv2.imread(sample["path"], cv2.IMREAD_COLOR)
                        if image is None or image.shape != (sample["height"], sample["width"], 3):
                            raise ValueError(f"Cannot decode the expected BGR image: {sample['path']}")
                        images.append(image)
                    image_batch = np.ascontiguousarray(np.stack(images))
                    output_names = OUTPUTS[model]
                    for _ in range(args.warmup):
                        infer(args.cpu_url, model, image_batch, output_names)
                        infer(args.gpu_url, model, image_batch, output_names)
                    before = ({side: runner.statistics(url, model)
                               for side, url in (("cpu", args.cpu_url), ("gpu", args.gpu_url))}
                              if runner is not None else None)
                    timings = {"cpu": [], "gpu": []}
                    comparisons = []
                    for repeat in range(args.repeats):
                        outputs = {}
                        # Alternate which server is measured first to reduce systematic drift.
                        order = ("cpu", "gpu") if repeat % 2 == 0 else ("gpu", "cpu")
                        for side in order:
                            url = args.cpu_url if side == "cpu" else args.gpu_url
                            started = time.perf_counter()
                            outputs[side] = infer(url, model, image_batch, output_names)
                            timings[side].append((time.perf_counter() - started) * 1000)
                        comparisons.append(compare_outputs(outputs["cpu"], outputs["gpu"]))
                    passed = all(value["passed"] for value in comparisons)
                    run = {"model": model, "batch_size": batch_size, "batch_index": batch_index,
                           "shape": batch["shape"],
                           "sample_ids": [sample["id"] for sample in batch["samples"]],
                           "repeated_slots": batch["repeated_slots"],
                           "parity": {"passed": passed, "comparisons": len(comparisons),
                                      "results": comparisons}}
                    if runner is not None:
                        run["server_processing"] = {
                            side: processing_statistics(before[side], runner.statistics(url, model))
                            for side, url in (("cpu", args.cpu_url), ("gpu", args.gpu_url))
                        }
                    for side, values in timings.items():
                        run[side] = {"latency_ms": values, "p50_ms": float(np.percentile(values, 50)),
                                     "p95_ms": float(np.percentile(values, 95))}
                    run["gpu_speedup_p50"] = run["cpu"]["p50_ms"] / max(run["gpu"]["p50_ms"], 1e-12)
                    report["runs"].append(run)
                    report["passed"] &= passed
                    print(f"{model} B{batch_size} batch {batch_index + 1}: "
                          f"CPU {run['cpu']['p50_ms']:.2f} ms / GPU {run['gpu']['p50_ms']:.2f} ms; "
                          f"parity {'PASS' if passed else 'FAIL'}", flush=True)
    finally:
        if runner is not None:
            runner.close()
    report["summary"] = []
    for model in args.models:
        for batch_size in (1, 4):
            runs = [run for run in report["runs"] if run["model"] == model and run["batch_size"] == batch_size]
            summary = {"model": model, "batch_size": batch_size,
                       "requests_per_server": len(runs) * args.repeats,
                       "repeated_image_slots": sum(len(run["repeated_slots"]) for run in runs) * args.repeats,
                       "passed": all(run["parity"]["passed"] for run in runs)}
            for side in ("cpu", "gpu"):
                timings = [value for run in runs for value in run[side]["latency_ms"]]
                summary[side] = {"p50_ms": float(np.percentile(timings, 50)),
                                 "p95_ms": float(np.percentile(timings, 95))}
            summary["gpu_speedup_p50"] = summary["cpu"]["p50_ms"] / max(summary["gpu"]["p50_ms"], 1e-12)
            report["summary"].append(summary)
    return report


def main(argv=None, *, infer=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/hdg/camtellsvi_alog/datasets"))
    parser.add_argument("--cpu-url", default="localhost:8000")
    parser.add_argument("--gpu-url", default="localhost:18000")
    parser.add_argument("--models", nargs="+", default=list(MODELS), help="face pose seg or full ensemble names")
    parser.add_argument("--limit", type=int, default=64, help="Maximum unique images (1..64)")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup requests per batch per server (at least 3)")
    parser.add_argument("--repeats", type=int, default=10, help="Measured requests per batch per server (at least 10)")
    parser.add_argument("--output", type=Path, default=Path("logs/yolo_gpu_benchmark.json"))
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= 64 or args.warmup < 3 or args.repeats < 10:
        parser.error("Require 1 <= limit <= 64, warmup >= 3 and repeats >= 10")
    models = []
    for value in args.models:
        for name in value.split(","):
            model = MODELS.get(name, name)
            if model not in OUTPUTS:
                parser.error(f"Unknown YOLO model: {name}")
            if model not in models:
                models.append(model)
    args.models = models
    try:
        report = run_benchmark(args, infer=infer)
    except Exception as error:
        report = {"schema_version": 1, "passed": False,
                  "error": {"type": type(error).__name__, "message": str(error)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(f"Report: {args.output.resolve()} — {'PASS' if report['passed'] else 'FAIL'}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
