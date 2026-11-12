"""Isolated preprocessing wall-time profile; no inference or HTTP requests.

Compressed image files are read before measurement. Every measured invocation
decodes those bytes again. GPU-ready timings end at stream synchronization;
downloads happen only in the separate numerical parity pass.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import types

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_image_gpu import sample_images
from image_gpu_candidate import ImageGpuPreprocessor, SPECS, parity_metrics
from image_gpu_eval_server import load_cpu_preprocessor


VARIANTS = ("cpu_decode", "cpu_preprocess", "cpu_gpu_ready", "gpu_normalize", "gpu_resize")


def load_original(root, alias):
    """Supply initialization helpers when running outside a Triton stub.

    The original preprocessing function executes verbatim. It does not call the
    serving API, so only config lookup and FP32 dtype conversion are needed.
    """
    name = "triton_python_backend_utils"
    previous = sys.modules.get(name)
    shim = types.ModuleType(name)
    shim.get_output_config_by_name = lambda config, output: next(
        row for row in config["output"] if row["name"] == output)
    shim.triton_string_to_numpy = lambda dtype: {"TYPE_FP32": np.float32}[dtype]
    sys.modules[name] = shim
    try:
        return load_cpu_preprocessor(root, alias)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def compare(reference, candidate, spec):
    metrics = parity_metrics(reference, candidate)
    delta = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    mean = np.asarray(spec["mean"], np.float64)[:, None, None]
    std = np.asarray(spec["std"], np.float64)[:, None, None]
    def recover(tensor):
        return np.clip(np.rint((tensor.astype(np.float64) * std + mean) * 255), 0, 255).astype(np.int16)
    pixel_delta = np.abs(recover(reference) - recover(candidate))
    metrics.update(normalization_atol=1e-6, elements_over_atol=int(np.count_nonzero(delta > 1e-6)),
                   recovered_uint8_different_channels=int(np.count_nonzero(pixel_delta)),
                   recovered_uint8_different_pixels=int(np.count_nonzero(np.any(pixel_delta, axis=0))),
                   recovered_uint8_max_abs=int(pixel_delta.max()),
                   same_uint8_and_normalization_within_atol=bool(not pixel_delta.any() and not (delta > 1e-6).any()))
    return metrics


def statistics(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["alias"], row["variant"], row["width"], row["height"])].append(row["wall_ms"])
    summaries = []
    for (alias, variant, width, height), values in grouped.items():
        summaries.append({"alias": alias, "variant": variant, "width": width, "height": height,
                          "count": len(values), "p50_ms": float(np.percentile(values, 50)),
                          "p95_ms": float(np.percentile(values, 95)), "mean_ms": float(np.mean(values))})
    pooled = defaultdict(list)
    for row in rows:
        pooled[(row["alias"], row["variant"])].append(row["wall_ms"])
    overall = [{"alias": alias, "variant": variant, "count": len(values),
                "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)),
                "mean_ms": float(np.mean(values))} for (alias, variant), values in pooled.items()]
    return {"by_image_dimensions": summaries, "overall": overall}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--source-models", required=True, help="Model repository containing original preprocess directories")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--opencv-rounding", choices=("scalar", "neon"), default="scalar")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    import cupy as cp
    import cv2
    import PIL
    samples = sample_images(args.dataset_root, limit=64)
    if len(samples) != 25:
        raise ValueError(f"expected the reviewed 25 unique images, got {len(samples)}")
    payloads = []
    for sample in samples:
        payload = Path(sample["path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != sample["sha256"]:
            raise ValueError("dataset bytes changed after sampling")
        payloads.append(payload)
    report = {"kind": "isolated_preprocessing_wall_time_no_inference_no_http",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "versions": {"cupy": cp.__version__, "opencv": cv2.__version__, "pillow": PIL.__version__, "numpy": np.__version__},
              "source_models": str(Path(args.source_models).resolve()), "samples": samples,
              "repeats": args.repeats, "warmups_per_image_variant": 3,
              "opencv_rounding": args.opencv_rounding,
              "decode_timing_scope": "cpu_decode includes RGB NumPy materialization; for Pillow this differs from the original PIL-through-resize path, so stage subtraction is not exact",
              "normalization_atol": 1e-6, "parity": [], "raw_rows": [],
              "measurement": "perf_counter_ns wall time; compressed bytes resident; each call re-decodes; GPU paths synchronize current dedicated stream; parity downloads excluded"}
    for alias in SPECS:
        original = load_original(args.source_models, alias)
        pre = ImageGpuPreprocessor(alias, cp_module=cp, opencv_rounding=args.opencv_rounding)
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            for sample, payload in zip(samples, payloads):
                reference = original(payload, 0)
                rgb = pre.decode(payload)
                cpu = pre.cpu_tensor(rgb)
                cpu_metrics = compare(reference, cpu, pre.spec)
                cpu_metrics["exact_equal"] = bool(np.array_equal(reference, cpu))
                report["parity"].append({"alias": alias, "image_id": sample["id"], "variant": "candidate_cpu", **cpu_metrics})
                if not cpu_metrics["exact_equal"]:
                    raise ValueError(f"candidate CPU differs from original for {alias}/{sample['id']}")
                for mode in ("normalize", "resize"):
                    device = pre.gpu_tensor(rgb, mode=mode)
                    host = device.get(stream=stream, blocking=True)
                    report["parity"].append({"alias": alias, "image_id": sample["id"], "variant": "gpu_" + mode,
                                             **compare(reference, host, pre.spec)})

            def invoke(variant, payload):
                if variant == "cpu_decode":
                    return pre.decode(payload)
                if variant == "cpu_preprocess":
                    return original(payload, 0)
                if variant == "cpu_gpu_ready":
                    cpu = original(payload, 0)
                    device = cp.asarray(cpu)
                else:
                    rgb = pre.decode(payload)
                    device = pre.gpu_tensor(rgb, mode=variant.removeprefix("gpu_"))
                stream.synchronize()
                return device

            for _ in range(3):
                for payload in payloads:
                    for variant in VARIANTS:
                        invoke(variant, payload)
            for repeat in range(args.repeats):
                for image_index, (sample, payload) in enumerate(zip(samples, payloads)):
                    rotation = (repeat + image_index) % len(VARIANTS)
                    ordered = VARIANTS[rotation:] + VARIANTS[:rotation]
                    for order, variant in enumerate(ordered):
                        start = time.perf_counter_ns()
                        value = invoke(variant, payload)
                        elapsed = (time.perf_counter_ns() - start) / 1e6
                        report["raw_rows"].append({"alias": alias, "variant": variant, "image_id": sample["id"],
                            "width": sample["width"], "height": sample["height"], "repeat": repeat,
                            "order_in_pair": order, "wall_ms": elapsed})
                        del value
        print(f"finished {alias}: {len(samples)} images, {args.repeats} repeats", flush=True)
    report["statistics"] = statistics(report["raw_rows"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(str(args.output), flush=True)


if __name__ == "__main__":
    main()
