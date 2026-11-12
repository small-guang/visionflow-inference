"""Compare real encoded images through baseline and experimental Triton models.

The report contains paths, hashes, dimensions and numerical summaries, never
image pixels or encoded image contents. No model repository files are changed.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

import numpy as np


MODELS = {
    "clip_image": {"baseline": "clip_ensemble_trt", "output": "unnorm_image_features", "dimension": 512},
    "tbps_clip_image": {"baseline": "TBPS_clip_image_ensemble", "output": "image_feat", "dimension": 512},
    "sbs": {"baseline": "sbs_r50_ibn_image_ensemble", "output": "embedding", "dimension": 2048},
}
VARIANTS = ("cpu_fused", "gpu_normalize", "gpu_resize")
OPTIONAL_VARIANTS = ("gpu_resize_neon",)
TOLERANCES = {"minimum_cosine_similarity": 0.99999, "maximum_relative_l2": 0.001,
              "relative_error_denominator_floor": 1e-12,
              "required_dtype": "float32", "shape_exact": True, "finite_required": True}


def candidate_model(alias, variant):
    if alias not in MODELS or variant not in VARIANTS + OPTIONAL_VARIANTS:
        raise ValueError(f"Unknown model or variant: {alias}/{variant}")
    if alias == "tbps_clip_image" and variant == "gpu_resize_neon":
        raise ValueError("tbps_clip_image does not support gpu_resize_neon")
    return f"benchmark_{alias}_{variant}"


def sample_images(dataset_root, limit=64):
    """Reuse the reviewed test/original sampler, keeping distinct encoded hashes."""
    path = Path(__file__).with_name("benchmark_yolo_gpu.py")
    spec = importlib.util.spec_from_file_location("_image_benchmark_dataset_sampler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    samples = module.sample_images(dataset_root, limit=64)
    if not 1 <= limit <= 64:
        raise ValueError("limit must be between 1 and 64")
    unique = {}
    for sample in samples:
        unique.setdefault(sample["sha256"], sample)
    return list(unique.values())[:limit]


def build_batches(samples, batch_size):
    """Keep sequential mixed-size samples and explicitly mark repeated tail slots."""
    if batch_size not in (1, 2, 3, 4):
        raise ValueError("Batch size must be between 1 and 4")
    batches = []
    for offset in range(0, len(samples), batch_size):
        selected = samples[offset:offset + batch_size]
        count = len(selected)
        slots = [selected[index % count] for index in range(batch_size)]
        batches.append({"samples": slots, "input_shape": [batch_size, 1],
                        "repeated_slots": list(range(count, batch_size))})
    return batches


def encoded_batch(batch):
    """Read original bytes into object cells; fixed-width strings lose trailing NULs."""
    values = np.empty((len(batch["samples"]), 1), dtype=object)
    for index, sample in enumerate(batch["samples"]):
        data = Path(sample["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != sample["sha256"]:
            raise ValueError(f"Sample bytes changed after manifest creation: {sample['path']}")
        values[index, 0] = data
    return values


def compare_vectors(baseline, candidate, *, expected_batch_size=None, expected_dimension=None):
    """Apply declared embedding tolerances; these do not establish task accuracy."""
    left, right = np.asarray(baseline), np.asarray(candidate)
    result = {"passed": False, "errors": [], "baseline_shape": list(left.shape),
              "candidate_shape": list(right.shape), "baseline_dtype": str(left.dtype),
              "candidate_dtype": str(right.dtype), "vectors": []}
    if left.dtype != np.float32 or right.dtype != np.float32:
        result["errors"].append("Both outputs must have dtype float32")
    if (left.shape != right.shape or left.ndim != 2 or not all(left.shape)
            or (expected_batch_size is not None and left.shape[0] != expected_batch_size)
            or (expected_dimension is not None and left.shape[-1] != expected_dimension)):
        result["errors"].append("Output shape must match the exact [B, embedding_dimension] contract")
    if result["errors"]:
        return result
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        result["errors"].append("Output contains nonfinite values")
        return result
    a, b = left.astype(np.float64), right.astype(np.float64)
    difference = np.abs(a - b)
    floor = TOLERANCES["relative_error_denominator_floor"]
    norms_a, norms_b = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    result.update(exact_equal=bool(np.array_equal(left, right)),
                  max_abs_error=float(difference.max()),
                  max_relative_error=float((difference / np.maximum(np.abs(a), floor)).max()))
    for index in range(a.shape[0]):
        relative_l2 = float(np.linalg.norm(a[index] - b[index]) / max(norms_a[index], floor))
        cosine = (float(np.clip(np.dot(a[index], b[index]) / (norms_a[index] * norms_b[index]), -1, 1))
                  if norms_a[index] > 0 and norms_b[index] > 0 else None)
        passed = (cosine is not None and cosine >= TOLERANCES["minimum_cosine_similarity"]
                  and relative_l2 <= TOLERANCES["maximum_relative_l2"])
        result["vectors"].append({"batch_slot": index, "baseline_norm": float(norms_a[index]),
                                  "candidate_norm": float(norms_b[index]),
                                  "cosine_similarity": cosine, "relative_l2": relative_l2,
                                  "max_abs_error": float(difference[index].max()),
                                  "max_relative_error": float((difference[index] / np.maximum(np.abs(a[index]), floor)).max()),
                                  "passed": passed})
    result["passed"] = all(vector["passed"] for vector in result["vectors"])
    if not result["passed"]:
        result["errors"].append("At least one vector failed cosine or relative L2 tolerance (or has zero norm)")
    return result


def processing_statistics(before, after):
    """Subtract cumulative request, inference, execution and batch timing counters."""
    old = {(row["name"], row.get("version")): row for row in before.get("model_stats", [])}

    def counter(previous, current):
        value = int(current) - int(previous)
        if value < 0:
            raise ValueError("Triton counters reset during measurement")
        return value

    def timings(previous, current):
        values = {}
        for phase, value in current.items():
            if not isinstance(value, dict) or "count" not in value or "ns" not in value:
                continue
            count = counter(previous.get(phase, {}).get("count", 0), value["count"])
            ns = counter(previous.get(phase, {}).get("ns", 0), value["ns"])
            values[phase] = {"count": count, "total_ms": ns / 1e6,
                             "mean_ms": ns / (count * 1e6) if count else None}
        return values

    result = []
    for row in after.get("model_stats", []):
        previous = old.get((row["name"], row.get("version")), {})
        value = {"name": row["name"], "version": row.get("version"),
                 "inference_count": counter(previous.get("inference_count", 0), row.get("inference_count", 0)),
                 "execution_count": counter(previous.get("execution_count", 0), row.get("execution_count", 0)),
                 "inference_stats": timings(previous.get("inference_stats", {}), row.get("inference_stats", {})),
                 "batch_stats": []}
        old_batches = {int(batch["batch_size"]): batch for batch in previous.get("batch_stats", [])}
        for batch in row.get("batch_stats", []):
            size = int(batch["batch_size"])
            value["batch_stats"].append({"batch_size": size, **timings(old_batches.get(size, {}), batch)})
        result.append(value)
    return result


class HttpInference:
    def __init__(self, endpoint_models):
        import tritonclient.http as httpclient

        self.httpclient = httpclient
        self.clients = {}
        self.metadata = {}
        self.model_metadata = {}
        self.model_configurations = {}
        try:
            for url, models in endpoint_models.items():
                parsed = urlsplit(url if "://" in url else "http://" + url)
                if (parsed.path not in ("", "/") or not parsed.hostname
                        or parsed.scheme not in ("http", "https") or parsed.query or parsed.fragment
                        or parsed.username or parsed.password):
                    raise ValueError("Triton URL must contain only HTTP(S) host and port")
                client = httpclient.InferenceServerClient(url=parsed.netloc, ssl=parsed.scheme == "https",
                                                          connection_timeout=30, network_timeout=120)
                self.clients[url] = client
                if not client.is_server_ready():
                    raise RuntimeError(f"Triton server is not ready: {url}")
                self.metadata[url] = client.get_server_metadata()
                self.model_metadata[url] = {}
                self.model_configurations[url] = {}
                for model in dict.fromkeys(models):
                    if not client.is_model_ready(model):
                        raise RuntimeError(f"Triton model is not ready: {url}/{model}")
                    self.model_metadata[url][model] = client.get_model_metadata(model)
                    self.model_configurations[url][model] = client.get_model_config(model)
        except Exception:
            self.close()
            raise

    def __call__(self, url, model, image_batch, output_name):
        if image_batch.dtype != object or image_batch.ndim != 2 or image_batch.shape[1] != 1:
            raise ValueError("IMAGE_BYTES must be an object array of shape [B,1]")
        if not all(isinstance(value, bytes) for value in image_batch[:, 0]):
            raise ValueError("Each IMAGE_BYTES object cell must contain original bytes")
        tensor = self.httpclient.InferInput("IMAGE_BYTES", image_batch.shape, "BYTES")
        tensor.set_data_from_numpy(image_batch, binary_data=True)
        response = self.clients[url].infer(
            model, inputs=[tensor],
            outputs=[self.httpclient.InferRequestedOutput(output_name, binary_data=True)],
        )
        value = response.as_numpy(output_name)
        if value is None:
            raise RuntimeError(f"Missing requested output {output_name} from {url}/{model}")
        return value

    def statistics(self, url):
        return self.clients[url].get_inference_statistics()

    def close(self):
        for client in self.clients.values():
            client.close()


def run_benchmark(args, infer=None):
    samples = sample_images(args.dataset_root, args.limit)
    variants = ["baseline", *args.variants]
    endpoints = defaultdict(list)
    for alias in args.models:
        endpoints[args.baseline_url].append(MODELS[alias]["baseline"])
        endpoints[args.candidate_url].extend(candidate_model(alias, variant) for variant in args.variants)
    report = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "dataset_root": str(Path(args.dataset_root).resolve()), "samples": samples,
              "baseline_url": args.baseline_url, "candidate_url": args.candidate_url,
              "models": args.models, "model_contracts": {alias: MODELS[alias] for alias in args.models},
              "variants": variants, "warmup": args.warmup, "repeats": args.repeats,
              "batch_sizes": args.batch_sizes,
              "input_contract": {"name": "IMAGE_BYTES", "datatype": "BYTES", "numpy_dtype": "object", "shape": "[B,1]"},
              "timing_scope": "Synchronous binary HTTP request/response including serialization and output materialization; excludes reading image bytes, warmup, correctness calculations and statistics calls",
              "server_processing_scope": "All server model counter deltas across all measured variants; excludes warmup. Shared model counters include all callers and concurrent external traffic, if any.",
              "request_order": "Sequential paired requests; rotate baseline and candidate order by repeat plus batch group index",
              "tolerances": dict(TOLERANCES),
              "accuracy_scope": "Embedding numerical agreement on this sample does not establish downstream retrieval or task accuracy parity",
              "passed": True, "runs": [], "requests": [], "summary": [], "errors": []}
    runner = None
    owned_runner = infer is None
    before = {}
    try:
        runner = HttpInference(endpoints) if owned_runner else infer
        for key, attribute in (("servers", "metadata"), ("server_model_metadata", "model_metadata"),
                               ("server_model_configurations", "model_configurations")):
            report[key] = getattr(runner, attribute, {})
        groups = []
        for alias in args.models:
            for size in args.batch_sizes:
                for index, batch in enumerate(build_batches(samples, size)):
                    groups.append((alias, size, index, batch, encoded_batch(batch)))

        def target(alias, variant):
            return ((args.baseline_url, MODELS[alias]["baseline"]) if variant == "baseline"
                    else (args.candidate_url, candidate_model(alias, variant)))

        # Warm every concrete batch before capturing any measured server counters.
        for group_index, (alias, size, index, batch, payload) in enumerate(groups):
            for repeat in range(args.warmup):
                shift = (group_index + repeat) % len(variants)
                for variant in variants[shift:] + variants[:shift]:
                    url, model = target(alias, variant)
                    runner(url, model, payload, MODELS[alias]["output"])
        if hasattr(runner, "statistics"):
            for url in endpoints:
                before[url] = runner.statistics(url)
            report["server_statistics_before"] = before
        for group_index, (alias, size, index, batch, payload) in enumerate(groups):
            run = {"model": alias, "batch_size": size, "batch_index": index,
                   "input_shape": batch["input_shape"], "sample_ids": [s["id"] for s in batch["samples"]],
                   "sample_dimensions": [[s["width"], s["height"]] for s in batch["samples"]],
                   "repeated_slots": batch["repeated_slots"],
                   "variants": {variant: {"latency_ms": [], "comparisons": [], "passed": True}
                                for variant in variants}}
            report["runs"].append(run)
            for repeat in range(args.repeats):
                outputs = {}
                shift = (group_index + repeat) % len(variants)
                order = variants[shift:] + variants[:shift]
                for position, variant in enumerate(order):
                    url, model = target(alias, variant)
                    request = {"model": alias, "batch_size": size, "batch_index": index,
                               "repeat": repeat, "order_index": position, "variant": variant,
                               "url": url, "triton_model": model, "success": False}
                    started = time.perf_counter()
                    try:
                        outputs[variant] = runner(url, model, payload, MODELS[alias]["output"])
                        request["success"] = True
                    except Exception as error:
                        request["error"] = {"type": type(error).__name__, "message": str(error)}
                    request["latency_ms"] = (time.perf_counter() - started) * 1000
                    report["requests"].append(request)
                    if request["success"]:
                        run["variants"][variant]["latency_ms"].append(request["latency_ms"])
                for variant in variants:
                    if "baseline" in outputs and variant in outputs:
                        comparison = compare_vectors(outputs["baseline"], outputs[variant],
                                                     expected_batch_size=size,
                                                     expected_dimension=MODELS[alias]["dimension"])
                    else:
                        comparison = {"passed": False, "errors": ["Baseline or candidate inference failed"]}
                    comparison["repeat"] = repeat
                    run["variants"][variant]["comparisons"].append(comparison)
                    run["variants"][variant]["passed"] &= comparison["passed"]
                    report["passed"] &= comparison["passed"]
            print(f"{alias} B{size} batch {index + 1}: "
                  + ", ".join(f"{v} {'PASS' if run['variants'][v]['passed'] else 'FAIL'}" for v in variants), flush=True)
    except Exception as error:
        report["passed"] = False
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
    finally:
        if runner is not None and before:
            report["server_statistics_after"] = {}
            report["server_processing"] = {}
            for url in before:
                try:
                    after = runner.statistics(url)
                    report["server_statistics_after"][url] = after
                    report["server_processing"][url] = processing_statistics(before[url], after)
                except Exception as error:
                    report["passed"] = False
                    report["errors"].append({"scope": "server_statistics", "url": url,
                                              "type": type(error).__name__, "message": str(error)})
        if owned_runner and runner is not None:
            runner.close()
    for alias in args.models:
        for size in args.batch_sizes:
            runs = [run for run in report["runs"] if run["model"] == alias and run["batch_size"] == size]
            baseline_times = [t for run in runs for t in run["variants"]["baseline"]["latency_ms"]]
            expected_requests = len(build_batches(samples, size)) * args.repeats
            baseline_valid = (len(baseline_times) == expected_requests
                              and all(run["variants"]["baseline"]["passed"] for run in runs))
            for variant in variants:
                values = [t for run in runs for t in run["variants"][variant]["latency_ms"]]
                passed = (len(values) == expected_requests and baseline_valid
                          and all(run["variants"][variant]["passed"] for run in runs))
                p50 = float(np.percentile(values, 50)) if values else None
                p95 = float(np.percentile(values, 95)) if values else None
                report["summary"].append({"model": alias, "batch_size": size, "variant": variant,
                                          "successful_requests": len(values), "expected_requests": expected_requests,
                                          "unique_images": len(samples),
                                          "repeated_image_slots": sum(len(r["repeated_slots"]) for r in runs) * args.repeats,
                                          "p50_ms": p50, "p95_ms": p95, "passed": passed,
                                          "performance_claim_valid": passed,
                                          "baseline_speedup_p50": (float(np.percentile(baseline_times, 50)) / max(p50, 1e-12)
                                                                   if passed else None)})
    return report


def main(argv=None, *, infer=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/hdg/camtellsvi_alog/datasets"))
    parser.add_argument("--baseline-url", default="localhost:8000")
    parser.add_argument("--candidate-url", default="localhost:18000")
    parser.add_argument("--models", nargs="+", default=list(MODELS), help="clip_image tbps_clip_image sbs")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4], help="B1/B4 performance; B2/B3 optional smoke")
    parser.add_argument("--limit", type=int, default=64, help="Maximum distinct image SHA256s (1..64)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("logs/image_gpu_benchmark.json"))
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= 64 or args.warmup < 3 or args.repeats < 10:
        parser.error("Require 1 <= limit <= 64, warmup >= 3 and repeats >= 10")
    for attribute, allowed in (("models", MODELS), ("variants", VARIANTS + OPTIONAL_VARIANTS)):
        values = list(dict.fromkeys(name for value in getattr(args, attribute) for name in value.split(",")))
        if any(value not in allowed for value in values):
            parser.error(f"Unknown {attribute}: {values}; allowed: {list(allowed)}")
        setattr(args, attribute, values)
    try:
        for alias in args.models:
            for variant in args.variants:
                candidate_model(alias, variant)
    except ValueError as exc:
        parser.error(str(exc))
    args.batch_sizes = list(dict.fromkeys(args.batch_sizes))
    if any(size not in (1, 2, 3, 4) for size in args.batch_sizes):
        parser.error("Batch sizes must be 1, 2, 3 or 4")
    try:
        report = run_benchmark(args, infer=infer)
    except Exception as error:
        report = {"schema_version": 1, "passed": False,
                  "errors": [{"type": type(error).__name__, "message": str(error)}]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Report: {args.output.resolve()} — {'PASS' if report['passed'] else 'FAIL'}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
