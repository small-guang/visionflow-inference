"""验证已验收 GPU-host 基线与融合 Triton 管线；数值通过后才允许连续性能窗口。

只连接调用方准备的隔离服务，不启动 Docker、不消费 Kafka。输入提前准备，
每个负载线程复用独立客户端；/proc、Triton stats、Prometheus 只在窗口边界读取。
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlsplit
from urllib.request import urlopen

import numpy as np


_SPEC = importlib.util.spec_from_file_location(
    "_orin_fused_shared_validation", Path(__file__).with_name("validate_orin_preprocessing.py"))
V = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(V)
_STATS_SPEC = importlib.util.spec_from_file_location(
    "_orin_fused_triton_statistics", Path(__file__).with_name("benchmark_image_gpu.py"))
_STATS = importlib.util.module_from_spec(_STATS_SPEC)
_STATS_SPEC.loader.exec_module(_STATS)

ACCEPTED_IMAGE = "sha256:c213eee77b9af813e9d1d102306f615a5a93ec17a713d6befdff9a61e4bc6e78"
METRIC_PREFIX = "orin_pipeline_"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def identity(path):
    value = json.loads(Path(path).read_text())
    image_id = value.get("image_id", value.get("target_image_id", value.get("after", {}).get("image_id")))
    return {"path": str(Path(path).resolve()), "sha256": file_digest(path), "image_id": image_id,
            "provenance": "caller-supplied immutable launch/source evidence; this tool does not inspect Docker"}


def load_samples(args):
    import cv2

    if not args.face_manifest.is_file() or not args.person_manifest.is_file():
        raise ValueError("reuse existing face/person manifests; this tool never resamples")
    inventory = V.inventory(args.dataset)
    if inventory["errors"] and not args.exclude_invalid_images:
        raise ValueError("invalid images present; inspect inventory and explicitly use --exclude-invalid-images")
    originals = []
    for item in V.select_samples(inventory["samples"], len(inventory["samples"])):
        encoded = Path(item["path"]).read_bytes()
        if hashlib.sha256(encoded).hexdigest() != item["sha256"]:
            raise ValueError("dataset changed after inventory")
        image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV {cv2.__version__} cannot decode inventoried image {item['sha256']}")
        originals.append(item | {"image": image, "encoded": encoded})
    if len(originals) != args.expected_images:
        raise ValueError(f"expected {args.expected_images} valid original images, found {len(originals)}")
    faces, _ = V.sample_manifest(args.face_manifest, originals, kind="face")
    people, _ = V.sample_manifest(args.person_manifest, originals, kind="person")
    if len(faces) != args.expected_faces or len(people) != args.expected_people:
        raise ValueError("sample manifest counts do not match the declared complete cohort")
    selected = {alias: originals if alias == "clip" else people if alias == "sbs" else faces
                for alias in args.models}
    metadata = {"inventory": inventory, "opencv": cv2.__version__, "numpy": np.__version__,
                "face_manifest": {"path": str(args.face_manifest.resolve()), "sha256": file_digest(args.face_manifest)},
                "person_manifest": {"path": str(args.person_manifest.resolve()), "sha256": file_digest(args.person_manifest)}}
    return selected, metadata


def make_case(alias, samples, *, identifier, repeated_slots=(), expected_error=False):
    return {"id": identifier, "alias": alias, "batch_size": len(samples),
            "sample_ids": [s.get("encoded_sha256", s["sha256"]) for s in samples],
            "repeated_slots": list(repeated_slots), "expected_error": expected_error,
            "inputs": V.inputs_for(alias, samples, "ensemble")}


def make_batches(selected, sizes):
    cases = []
    for alias, samples in selected.items():
        for size in sizes:
            for index, batch in enumerate(V.groups_of(samples, size, same_shape=alias not in {"clip", "sbs"})):
                cases.append(make_case(alias, batch["samples"], identifier=f"{alias}/B{size}/{index}",
                                       repeated_slots=batch["repeated_slots"]))
    return cases


def stress_cases(alias, samples, kind):
    # Each raw face request keeps its original H/W; padding would hide ragged scheduling bugs.
    selected = [samples[i % len(samples)] for i in range(4)]
    if kind == "b1_pressure":
        return [make_case(alias, [sample], identifier=f"{alias}/stress/B1/{i}")
                for i, sample in enumerate(selected)]
    if alias in {"clip", "sbs"}:
        slots = (selected[:1], selected[1:4], selected[:2], selected[2:4])
        cases = [make_case(alias, values, identifier=f"{alias}/stress/mixed/{i}")
                 for i, values in enumerate(slots)]
    else:
        cases = [make_case(alias, [sample] * size, identifier=f"{alias}/stress/mixed/{i}",
                           repeated_slots=range(1, size))
                 for i, (sample, size) in enumerate(zip(selected, (1, 3, 2, 2)))]
    invalid = cases[0] | {"id": f"{alias}/stress/invalid", "expected_error": True,
                         "inputs": {k: value.copy() for k, value in cases[0]["inputs"].items()}}
    if alias in {"clip", "sbs"}:
        invalid["inputs"]["IMAGE_BYTES"][0, 0] = b"not-an-image"
    else:
        invalid["inputs"]["BBOX" if alias == "keypoint" else "LANDMARK"].flat[0] = np.nan
    return [cases[0], invalid, *cases[1:]]


def case_metadata(case):
    return {key: value for key, value in case.items() if key != "inputs"}


def infer_case(endpoint, case):
    alias = case["alias"]
    output = V.MODELS[alias][3]
    return endpoint.infer(V.MODELS[alias][1], case["inputs"], [output])[output]


def compare_case(case, reference, actual):
    if isinstance(reference, dict):
        if case["alias"] != "clip" or reference.get("case_id") != case["id"]:
            raise ValueError("native reference bank is restricted to its original CLIP request")
        serial = compare_case(case, reference["serial"], actual)
        comparisons = []
        for variant in reference["variants"]:
            # A single native execution slice must match every row of this request.
            comparison = compare_case(case, variant["output"], actual)
            comparisons.append({key: value for key, value in variant.items() if key != "output"}
                               | {"comparison": comparison})
        return {"passed": any(row["comparison"]["passed"] for row in comparisons),
                "policy": "clip_original_native_B1_B4_whole_request", "serial_comparison": serial,
                "matched_native_variants": [row for row in comparisons if row["comparison"]["passed"]],
                "native_variant_comparisons": comparisons,
                "scope": "compatible native batch/slot references, not an observation of actual request grouping; no per-row or per-element selection"}
    return V.compare_arrays(reference, actual, V.MODELS[case["alias"]][4], V.DEFAULT_TOLERANCES)


def clip_batch_evidence(path, endpoints, model_metadata):
    """Require the frozen real native diagnostic before enabling the CLIP-only reference bank."""
    path = Path(path)
    diagnostic = json.loads(path.read_text())
    summary_path = path.with_name(path.stem + "-summary.json")
    summary = json.loads(summary_path.read_text())
    raw_path = Path(diagnostic["raw_arrays"]["path"])
    hashes = {"diagnostic_sha256": file_digest(path), "summary_sha256": file_digest(summary_path),
              "raw_arrays_sha256": file_digest(raw_path)}
    if (not diagnostic.get("completed") or not summary.get("completed")
            or diagnostic.get("tolerances_unchanged") != V.DEFAULT_TOLERANCES
            or summary.get("tolerances_unchanged") != V.DEFAULT_TOLERANCES
            or summary.get("source_json_sha256") != hashes["diagnostic_sha256"]
            or summary.get("raw_npz_sha256") != hashes["raw_arrays_sha256"]
            or diagnostic["raw_arrays"].get("sha256") != hashes["raw_arrays_sha256"]
            or not summary.get("all_preprocessed_equal")):
        raise ValueError("CLIP native batch diagnostic is incomplete, changed or uses different tolerances")
    for key in ("neighbor_comparisons", "repeated_slot_comparisons", "slot_reordering_comparisons",
                "batch_2_3_4_comparisons"):
        if not summary.get(key) or not all(row.get("exact_equal") and row.get("passed") for row in summary[key]):
            raise ValueError(f"CLIP native diagnostic does not establish {key}")
    paired = summary.get("same_batch_comparisons", {})
    for key in ("candidate_vs_native_same_batch", "baseline_vs_native_same_batch", "candidate_vs_baseline_same_batch"):
        row = paired.get(key, {})
        if not row.get("comparisons") or not row.get("all_exact_equal") or not row.get("all_passed"):
            raise ValueError("CLIP same-batch diagnostic did not pass")
    if not summary.get("native_B1_failures"):
        raise ValueError("no demonstrated native batch dependence; a CLIP reference bank is not justified")
    native_config = endpoints["baseline"].client.get_model_config("clip_image_trt")
    expected = {"baseline_public": model_metadata["baseline"]["models"]["clip_ensemble_trt"],
                "baseline_pre": model_metadata["baseline"]["models"]["clip_image_preprocess"],
                "candidate_public": model_metadata["candidate"]["models"]["clip_ensemble_trt"],
                "baseline_native": native_config}
    if diagnostic.get("model_configs") != expected:
        raise ValueError("CLIP diagnostic model configuration does not match the current endpoints")
    return {"diagnostic_path": str(path.resolve()), **hashes, "native_config": native_config,
            "reason": "original FP16 native plan changes output between B1 and B2/B3/B4; real fixed-input diagnostic proves same-batch equivalence and slot/neighbor independence",
            "tolerances": V.DEFAULT_TOLERANCES}


class ClipNativeReferences:
    """Finite original-native references, prepared serially outside stress/performance windows."""

    def __init__(self, endpoint, audit):
        self.endpoint = endpoint
        self.audit = audit
        self.preprocessed = {}
        self.groups = {}

    def wrap(self, case, serial):
        if case["alias"] != "clip" or case["batch_size"] == 4:
            return serial
        encoded = list(case["inputs"]["IMAGE_BYTES"][:, 0])
        ids = [hashlib.sha256(value).hexdigest() for value in encoded]
        for identifier, value in zip(ids, encoded):
            if identifier not in self.preprocessed:
                inputs = np.empty((1, 1), object)
                inputs[0, 0] = value
                output = self.endpoint.infer("clip_image_preprocess", {"IMAGE_BYTES": inputs}, ["image"])["image"]
                if output.shape != (1, 3, 224, 224) or output.dtype != np.float32 or not np.isfinite(output).all():
                    raise ValueError("invalid original CLIP preprocessing reference")
                self.preprocessed[identifier] = output[0].copy()
        variants = []
        for size in range(len(ids), 5):
            for offset in range(size - len(ids) + 1):
                layout = [ids[0]] * size
                layout[offset:offset + len(ids)] = ids
                key = tuple(layout)
                if key not in self.groups:
                    tensor = np.stack([self.preprocessed[identifier] for identifier in layout])
                    output = self.endpoint.infer("clip_image_trt", {"image": tensor}, ["unnorm_image_features"])["unnorm_image_features"]
                    repeat = self.endpoint.infer("clip_image_trt", {"image": tensor}, ["unnorm_image_features"])["unnorm_image_features"]
                    stability = V.compare_arrays(output, repeat, "embedding", V.DEFAULT_TOLERANCES)
                    if not stability["passed"]:
                        raise ValueError("original CLIP native references are not repeatable")
                    self.groups[key] = (output.copy(), stability)
                output, stability = self.groups[key]
                sliced = output[offset:offset + len(ids)].copy()
                variants.append({"total_batch_size": size, "request_start_slot": offset,
                                 "ordered_native_sample_sha256": layout,
                                 "output_sha256": hashlib.sha256(sliced.tobytes()).hexdigest(),
                                 "repeat_comparison": stability, "output": sliced})
        direct = compare_case(case, serial, variants[0]["output"])
        if not direct["passed"]:
            raise ValueError("original native reference does not match the serial public request at the same batch")
        metadata = {"case_id": case["id"], "serial_vs_native_same_batch": direct,
                    "serial_output_sha256": hashlib.sha256(serial.tobytes()).hexdigest(),
                    "variants": [{key: value for key, value in row.items() if key != "output"} for row in variants]}
        self.audit[case["id"]] = metadata
        return {"case_id": case["id"], "serial": serial, "variants": variants}


def metadata(endpoints, aliases):
    result = {}
    for side, endpoint in endpoints.items():
        if not endpoint.client.is_server_ready():
            raise ValueError(f"{side} is not ready")
        configurations = {}
        for alias in aliases:
            model = V.MODELS[alias][1]
            if not endpoint.client.is_model_ready(model):
                raise ValueError(f"{side}/{model} is not ready")
            configurations[model] = endpoint.client.get_model_config(model)
            if side == "baseline":
                pre = V.MODELS[alias][0]
                config = endpoint.client.get_model_config(pre)
                if config.get("parameters", {}).get("preprocess_backend", {}).get("string_value") != "orin_gpu":
                    raise ValueError(f"baseline {pre} is not the accepted GPU-host path")
                configurations[pre] = config
        result[side] = {"server": endpoint.client.get_server_metadata(), "models": configurations}
    return result


def parse_metrics(payload):
    """只解析约定的低基数累积指标；拒绝非有限值、重复series和非标准labels。"""
    result = {}
    pattern = re.compile(r'^(\w+)(?:\{(.*)\})?\s+([^\s]+)(?:\s+\d+)?$')
    label_pattern = re.compile(r'([a-zA-Z_]\w*)="((?:[^"\\]|\\.)*)"')
    for line in payload.splitlines():
        if not line.startswith(METRIC_PREFIX):
            continue
        match = pattern.fullmatch(line)
        if not match:
            raise ValueError("malformed pipeline metric")
        name, raw_labels, raw_value = match.groups()
        labels, cursor = {}, 0
        for label in label_pattern.finditer(raw_labels or ""):
            gap = (raw_labels or "")[cursor:label.start()].strip()
            if gap not in ({""} if cursor == 0 else {","}) or label[1] in labels:
                raise ValueError("malformed or duplicated metric label")
            labels[label[1]] = json.loads('"' + label[2] + '"')
            cursor = label.end()
        if (raw_labels or "")[cursor:].strip():
            raise ValueError("malformed metric labels")
        value = float(raw_value)
        key = (name, tuple(sorted(labels.items())))
        if not math.isfinite(value) or value < 0 or key in result:
            raise ValueError("invalid or duplicated pipeline counter")
        result[key] = value
    return result


def fetch_metrics(url):
    if not url:
        return None
    parsed = urlsplit(url)
    if parsed.path != "/metrics" or parsed.port in {8000, 8001, 8002, 8003}:
        raise ValueError("metrics URL must use a disposable port and /metrics")
    V.validate_url(parsed._replace(path="").geturl())
    with urlopen(url, timeout=15) as response:
        return parse_metrics(response.read().decode())


def metrics_delta(before, after):
    if before is None or after is None:
        return None
    if before.keys() - after.keys():
        raise ValueError("pipeline metric series disappeared during window")
    rows = []
    for (name, labels), value in after.items():
        delta = value - before.get((name, labels), 0)
        if delta < 0:
            raise ValueError("pipeline counters reset during window")
        rows.append({"name": name, "labels": dict(labels), "delta": delta})
    return rows


def summarize_metrics(rows, alias):
    if rows is None:
        return {"available": False, "reason": "metrics_url_not_supplied"}
    values = [r for r in rows if r["labels"].get("model") in {alias, V.MODELS[alias][1]}]
    sums = {}
    stages, batches, transfers = {}, {}, {}
    for row in values:
        suffix = row["name"].removeprefix(METRIC_PREFIX)
        sums[suffix] = sums.get(suffix, 0) + row["delta"]
        if suffix in {"stage_seconds_total", "stage_calls_total"}:
            stage = row["labels"]["stage"]
            stages.setdefault(stage, {})["seconds" if suffix == "stage_seconds_total" else "calls"] = (
                stages.setdefault(stage, {}).get("seconds" if suffix == "stage_seconds_total" else "calls", 0) + row["delta"])
        elif suffix == "trt_batches_total":
            size = int(row["labels"]["batch_size"])
            batches[size] = batches.get(size, 0) + row["delta"]
        elif suffix == "copy_bytes_total":
            direction = row["labels"]["direction"]
            transfers[direction] = transfers.get(direction, 0) + row["delta"]
    return {"available": bool(values), "counters": sums, "stages_host_wall": stages,
            "trt_batches": [{"batch_size": k, "executions": v} for k, v in sorted(batches.items()) if v],
            "logical_copy_bytes": transfers,
            "stage_scope": "host wall; trt_download includes GPU waiting and final download; stages can overlap across instances and are not additive request latency"}


def capture(endpoint, metrics_url, *, before):
    # CPU is last before the window and first after it; stats/metrics are outside CPU/wall timing.
    start = utc_now()
    cpu = None if before else V.cpu_tree(endpoint.pid)
    statistics = endpoint.client.get_inference_statistics()
    metrics = fetch_metrics(metrics_url)
    if before:
        cpu = V.cpu_tree(endpoint.pid)
    return {"cpu": cpu, "statistics": statistics, "metrics": metrics,
            "capture_started_utc": start, "capture_finished_utc": utc_now()}


def server_delta(before, after, alias):
    pipeline_delta = metrics_delta(before["metrics"], after["metrics"])
    return {"cpu": V.cpu_delta(before["cpu"], after["cpu"]),
            "triton": _STATS.processing_statistics(before["statistics"], after["statistics"]),
            "pipeline": summarize_metrics(pipeline_delta, alias), "pipeline_deltas": pipeline_delta,
            "snapshot_times": {"before": [before.get("capture_started_utc"), before.get("capture_finished_utc")],
                               "after": [after.get("capture_started_utc"), after.get("capture_finished_utc")]}}


def pipeline_accounting_errors(window):
    """候选的真实runner计数必须与本窗口完成的RPC/图片一致，不能只检查HTTP成功。"""
    pipeline = window["server"]["pipeline"]
    if not pipeline["available"] or not pipeline.get("trt_batches"):
        return ["candidate pipeline execution counters are missing"]
    counters = pipeline["counters"]
    errors = []
    for name in ("requests_total", "group_requests_total"):
        if counters.get(name) != window["successful_requests"]:
            errors.append(f"{name} does not match successful client requests")
    images = sum(row["batch_size"] * row["executions"] for row in pipeline["trt_batches"])
    if images != window["successful_images_including_repeated_slots"]:
        errors.append("enqueued TensorRT images do not match successful client images")
    if counters.get("errors_total", 0):
        errors.append("pipeline reported errors during the performance window")
    if not pipeline.get("stages_host_wall", {}).get("trt_download", {}).get("calls", 0):
        errors.append("trt_download stage timing is missing")
    return errors


def stress_check(endpoint, cases, references, *, requests, concurrency, rounds, metrics_url):
    alias = cases[0]["alias"]
    jobs = [cases[i % len(cases)] for i in range(requests * rounds)]
    start = threading.Barrier(concurrency)
    stop = threading.Event()
    before = capture(endpoint, metrics_url, before=True)

    def worker(worker_id):
        client = V.Endpoint(endpoint.url)
        rows = []
        try:
            start.wait(timeout=60)
            for index in range(worker_id, len(jobs), concurrency):
                if stop.is_set():
                    break
                case = jobs[index]
                try:
                    output = infer_case(client, case)
                    comparison = (compare_case(case, references[case["id"]], output)
                                  if not case["expected_error"] else {"passed": False, "reason": "bad_request_accepted"})
                    rows.append({"index": index, "case": case["id"], "comparison": comparison,
                                 "passed": comparison["passed"]})
                    if not comparison["passed"]:
                        stop.set()
                except Exception as exc:
                    rows.append({"index": index, "case": case["id"], "error_class": type(exc).__name__,
                                 "passed": case["expected_error"]})
                    if not case["expected_error"]:
                        stop.set()
            return rows
        finally:
            client.close()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = [row for worker_rows in pool.map(worker, range(concurrency)) for row in worker_rows]
    after = capture(endpoint, metrics_url, before=False)
    server = server_delta(before, after, alias)
    cross_groups = server["pipeline"].get("counters", {}).get("cross_request_groups_total")
    cross_evidence = {"groups": cross_groups, "observed": None if cross_groups is None else cross_groups > 0,
                      "note": ("cross-request counter missing; numerical success does not establish cross-request batching"
                               if cross_groups is None else
                               "no cross-request groups observed; numerical success does not establish cross-request batching"
                               if cross_groups == 0 else
                               "actual runner reported cross-request groups; numerical outputs checked against baseline")}
    return {"passed": len(results) == len(jobs) and all(row["passed"] for row in results),
            "requests": len(jobs), "concurrency": concurrency, "rounds": rounds,
            "expected_errors": sum(case["expected_error"] for case in jobs),
            "cases": [case_metadata(case) | {"input_shapes": {key: list(value.shape)
                                                               for key, value in case["inputs"].items()}}
                      for case in cases],
            "results": sorted(results, key=lambda row: row["index"]),
            "cross_request_batching": cross_evidence, "server": server}


def correctness(args, endpoints, selected, cases, report=None, clip_references=None):
    report = {} if report is None else report
    report.update(passed=False, batch_checks=[], stress=[], stress_concurrency=args.stress_concurrency,
                  stress_requests=args.stress_requests, stress_rounds=args.stress_rounds)
    references = {}
    for case in cases:
        try:
            side = "baseline"
            reference = infer_case(endpoints["baseline"], case)
            references[case["id"]] = reference
            side = "candidate"
            for repeat in range(args.correctness_repeats):
                result = compare_case(case, reference, infer_case(endpoints["candidate"], case))
                report["batch_checks"].append(case_metadata(case) | {"repeat": repeat, "comparison": result,
                                                                   "passed": result["passed"]})
        except Exception as exc:
            report["batch_checks"].append(case_metadata(case) | {"passed": False, "side": side,
                                                               "error_class": type(exc).__name__})
            return report, references
    if not all(row["passed"] for row in report["batch_checks"]):
        return report, references
    if clip_references is not None:
        references = {case["id"]: clip_references.wrap(case, references[case["id"]]) for case in cases}
    for alias, samples in selected.items():
        for kind in ("b1_pressure", "mixed_errors"):
            requests = stress_cases(alias, samples, kind)
            refs = {case["id"]: infer_case(endpoints["baseline"], case)
                    for case in requests if not case["expected_error"]}
            if clip_references is not None:
                refs = {case["id"]: clip_references.wrap(case, refs[case["id"]])
                        for case in requests if not case["expected_error"]}
            for side, endpoint in endpoints.items():
                result = stress_check(endpoint, requests, refs, requests=args.stress_requests,
                                      concurrency=args.stress_concurrency, rounds=args.stress_rounds,
                                      metrics_url=getattr(args, f"{side}_metrics_url"))
                report["stress"].append({"alias": alias, "side": side, "kind": kind, **result})
    report["passed"] = bool(report["batch_checks"]) and all(row["passed"] for row in report["batch_checks"] + report["stress"])
    return report, references


def check_gate(gate, binding, concurrency):
    if not gate.get("passed") or not gate.get("correctness", {}).get("passed"):
        raise ValueError("numerical correctness gate has not passed")
    if gate.get("binding_sha256") != digest(binding):
        raise ValueError("correctness evidence does not match current datasets, identities or model configurations")
    if gate["correctness"].get("stress_concurrency", 0) < max(concurrency):
        raise ValueError("performance concurrency exceeds the validated numerical stress concurrency")


def run_window(endpoint, cases, references, *, duration, concurrency, warmup, metrics_url,
               endpoint_factory=None, capture_fn=None):
    """窗口内没有/proc/statistics抓取、文件I/O或数值比较；最后响应排空后计总wall。"""
    factory = endpoint_factory or V.Endpoint
    snapshot = capture_fn or capture
    ready = threading.Barrier(concurrency + 1)
    begin = threading.Event()
    cancel = threading.Event()
    control = {}

    def worker(worker_id):
        client, warm_error = None, None
        result = {"latencies": [], "requests": 0, "images": 0, "errors": [], "outputs": {}, "sample_ids": set()}
        try:
            try:
                client = factory(endpoint.url)
                for index in range(warmup):
                    infer_case(client, cases[(worker_id + index) % len(cases)])
            except Exception as exc:
                warm_error = type(exc).__name__
                cancel.set()
            ready.wait(timeout=180)
            begin.wait(timeout=180)
            if warm_error or cancel.is_set():
                result["warmup_error"] = warm_error
                return result
            index = worker_id
            while not cancel.is_set() and time.perf_counter() < control["deadline"]:
                case = cases[index % len(cases)]
                tick = time.perf_counter()
                try:
                    output = infer_case(client, case)
                    elapsed = time.perf_counter() - tick
                    result["latencies"].append(elapsed)
                    result["requests"] += 1
                    result["images"] += case["batch_size"]
                    result["outputs"][case["id"]] = output
                    result["sample_ids"].update(case["sample_ids"])
                except Exception as exc:
                    result["errors"].append(type(exc).__name__)
                    cancel.set()
                index += concurrency
            return result
        finally:
            if client is not None:
                client.close()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker, index) for index in range(concurrency)]
        try:
            ready.wait(timeout=180)
            before = snapshot(endpoint, metrics_url, before=True)
            start_utc = utc_now()
            process_start, started = time.process_time(), time.perf_counter()
            control["deadline"] = started + duration
            begin.set()
            results = [future.result() for future in futures]
            finished = time.perf_counter()
            client_cpu = time.process_time() - process_start
            end_utc = utc_now()
            after = snapshot(endpoint, metrics_url, before=False)
        finally:
            cancel.set()
            begin.set()
    elapsed = finished - started
    alias = cases[0]["alias"]
    comparisons = []
    by_id = {case["id"]: case for case in cases}
    for result in results:
        for case_id, output in result["outputs"].items():
            comparisons.append({"case": case_id, **compare_case(by_id[case_id], references[case_id], output)})
    latencies = [value * 1000 for result in results for value in result["latencies"]]
    requests = sum(result["requests"] for result in results)
    images = sum(result["images"] for result in results)
    errors = [error for result in results for error in result["errors"]]
    warm_errors = [result["warmup_error"] for result in results if result.get("warmup_error")]
    server = server_delta(before, after, alias)
    cpu = server["cpu"]
    passed = bool(requests) and not errors and not warm_errors and all(x["passed"] for x in comparisons)
    return {"passed": passed, "start_utc": start_utc, "end_utc": end_utc,
            "requested_seconds": duration, "wall_seconds_including_drain": elapsed,
            "drain_seconds": max(0, finished - control["deadline"]), "concurrency": concurrency,
            "successful_requests": requests, "successful_images_including_repeated_slots": images,
            "requests_per_second": requests / elapsed, "images_per_second": images / elapsed,
            "latency_ms": {"mean": float(np.mean(latencies)), "p50": float(np.percentile(latencies, 50)),
                           "p95": float(np.percentile(latencies, 95)), "p99": float(np.percentile(latencies, 99))} if latencies else None,
            "latency_scope": "successful requests only; any request failure invalidates the performance window",
            "attempted_requests": requests + len(errors), "error_count": len(errors),
            "errors": errors, "warmup_errors": warm_errors, "client_cpu_seconds": client_cpu,
            "server_cpu_seconds_per_image": cpu["seconds"] / images if images and cpu["complete_process_set"] else None,
            "unique_samples_seen": len(set().union(*(result["sample_ids"] for result in results))),
            "post_window_comparisons": comparisons, "server": server,
            "verification_scope": "full numerical gate first; timed window retains last response per case/worker for post-window comparison, not every repeated response"}


def run(args):
    report = {"schema_version": 1, "created_at_utc": utc_now(), "passed": False,
              "phase": args.phase, "tolerances": V.DEFAULT_TOLERANCES, "windows": [],
              "scope": "Direct Triton endpoints, not ALog/Kafka throughput. Baseline is the accepted GPU-host bridge; TensorRT runs on GPU on both sides."}
    endpoints = {}
    try:
        selected, dataset = load_samples(args)
        report["dataset"] = dataset
        identities = {side: identity(getattr(args, f"{side}_identity")) for side in ("baseline", "candidate")}
        if identities["baseline"]["image_id"] != args.expected_baseline_image:
            raise ValueError("baseline identity is not the explicitly accepted GPU-host image")
        if args.baseline_url == args.candidate_url:
            raise ValueError("baseline and candidate must be distinct disposable endpoints")
        for side in ("baseline", "candidate"):
            endpoints[side] = V.Endpoint(getattr(args, f"{side}_url"), getattr(args, f"{side}_pid"))
        model_metadata = metadata(endpoints, args.models)
        clip_evidence = (clip_batch_evidence(args.clip_batch_diagnostic, endpoints, model_metadata)
                         if args.clip_batch_diagnostic else None)
        report["arguments"] = {key: str(value) if isinstance(value, Path) else value
                               for key, value in vars(args).items()}
        binding = {"identities": identities, "models": model_metadata, "cohort": {
            "source_sha256": [x["sha256"] for x in dataset["inventory"]["samples"]],
            "face_manifest": dataset["face_manifest"]["sha256"], "person_manifest": dataset["person_manifest"]["sha256"]},
            "aliases": args.models, "batch_sizes": args.batch_sizes, "tolerances": V.DEFAULT_TOLERANCES,
            "tool_sha256": file_digest(__file__), "shared_tool_sha256": file_digest(V.__file__),
            "statistics_tool_sha256": file_digest(_STATS.__file__), "clip_batch_evidence": clip_evidence}
        report.update(binding=binding, binding_sha256=digest(binding), endpoint_metadata=model_metadata)
        report["clip_native_references"] = {}
        clip_references = ClipNativeReferences(endpoints["baseline"], report["clip_native_references"]) if clip_evidence else None
        cases = make_batches(selected, args.batch_sizes)
        report["cases"] = [case_metadata(case) for case in cases]
        references = {}
        if args.phase in {"correctness", "all"}:
            report["correctness"] = {}
            _, references = correctness(args, endpoints, selected, cases, report=report["correctness"],
                                        clip_references=clip_references)
            report["passed"] = report["correctness"]["passed"]
            if not report["passed"] or args.phase == "correctness":
                return report
            gate = report
        else:
            gate = json.loads(args.correctness_report.read_text())
            report["correctness_evidence"] = {"path": str(args.correctness_report.resolve()),
                                               "sha256": file_digest(args.correctness_report)}
        check_gate(gate, binding, args.concurrency)
        # Reconstruct reference arrays outside all performance windows; reports never save embeddings.
        if not references:
            references = {case["id"]: infer_case(endpoints["baseline"], case) for case in cases}
            if clip_references is not None:
                references = {case["id"]: clip_references.wrap(case, references[case["id"]]) for case in cases}
                prior = gate.get("clip_native_references", {})
                if any(prior.get(key) != value for key, value in report["clip_native_references"].items()):
                    raise ValueError("native CLIP reference bank changed since the numerical gate")
        for alias in args.performance_models or args.models:
            for size in args.performance_batch_sizes or args.batch_sizes:
                cohort = [case for case in cases if case["alias"] == alias and case["batch_size"] == size]
                for concurrency in args.concurrency:
                    for repeat in range(args.window_repeats):
                        order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
                        for side in order:
                            value = run_window(endpoints[side], cohort, references, duration=args.seconds,
                                               concurrency=concurrency, warmup=args.warmup,
                                               metrics_url=getattr(args, f"{side}_metrics_url"))
                            if side == "candidate":
                                accounting_errors = pipeline_accounting_errors(value)
                                if accounting_errors:
                                    value.update(passed=False, stage_evidence_errors=accounting_errors)
                            report["windows"].append({"alias": alias, "batch_size": size, "side": side,
                                                       "repeat": repeat, **value})
                            print(json.dumps({"alias": alias, "B": size, "C": concurrency, "side": side,
                                              "repeat": repeat, "passed": value["passed"],
                                              "images_per_second": value["images_per_second"]}), flush=True)
                            if not value["passed"]:
                                report["passed"] = False
                                return report
        report["passed"] = bool(report["windows"]) and all(window["passed"] for window in report["windows"])
        report["measurement_notes"] = [
            "Inputs preloaded; each worker reuses a client. No /proc, metrics, statistics, file IO or array comparisons in timed loops.",
            "Window throughput divides all successful images/requests by actual wall including draining outstanding requests; warmup excluded.",
            "Server process CPU captures only window boundaries. Linux ticks are quantized; incomplete process sets produce null CPU/image.",
            "Stage counters measure host wall and can overlap; trt_download includes waiting plus final D2H, not pure GPU inference.",
            "Copy byte counters cover logical input pixels/final output only, not metadata or all runtime allocations.",
            "This is a closed-loop threaded HTTP load generator. Client CPU/serialization can limit offered load; no resource saturation is assumed."]
    except Exception as exc:
        report.update(passed=False, error_class=type(exc).__name__, error=str(exc))
    finally:
        for endpoint in endpoints.values():
            endpoint.close()
        report["finished_at_utc"] = utc_now()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "face-manifest", "person-manifest", "output", "baseline-identity", "candidate-identity"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--phase", choices=("correctness", "performance", "all"), default="correctness")
    parser.add_argument("--correctness-report", type=Path)
    parser.add_argument("--clip-batch-diagnostic", type=Path,
                        help="explicitly enable CLIP native batch references after validating this real diagnostic, sibling -summary.json and bound NPZ")
    parser.add_argument("--exclude-invalid-images", action="store_true")
    parser.add_argument("--expected-baseline-image", default=ACCEPTED_IMAGE)
    for side, port in (("baseline", 18300), ("candidate", 18400)):
        parser.add_argument(f"--{side}-url", default=f"http://127.0.0.1:{port}")
        parser.add_argument(f"--{side}-metrics-url", default=f"http://127.0.0.1:{port + 2}/metrics")
        parser.add_argument(f"--{side}-pid", type=int)
    parser.add_argument("--models", nargs="+", choices=list(V.MODELS), default=list(V.MODELS))
    parser.add_argument("--batch-sizes", nargs="+", type=int, choices=(1, 4), default=[1, 4])
    parser.add_argument("--performance-models", nargs="+", choices=list(V.MODELS),
                        help="performance-only subset of the numerically gated --models")
    parser.add_argument("--performance-batch-sizes", nargs="+", type=int, choices=(1, 4),
                        help="performance-only subset of the numerically gated --batch-sizes")
    parser.add_argument("--expected-images", type=int, default=86)
    parser.add_argument("--expected-faces", type=int, default=16)
    parser.add_argument("--expected-people", type=int, default=32)
    parser.add_argument("--correctness-repeats", type=int, default=2)
    parser.add_argument("--stress-requests", type=int, default=32)
    parser.add_argument("--stress-rounds", type=int, default=2)
    parser.add_argument("--stress-concurrency", type=int, default=16)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--window-repeats", type=int, default=2)
    args = parser.parse_args()
    if args.phase == "performance" and not args.correctness_report:
        parser.error("performance requires a passed --correctness-report")
    if args.clip_batch_diagnostic and "clip" not in args.models:
        parser.error("--clip-batch-diagnostic requires clip in --models")
    if args.output.exists():
        parser.error("output exists; preserve earlier evidence")
    if (min(args.concurrency + [args.expected_images, args.expected_faces, args.expected_people,
                               args.correctness_repeats, args.stress_requests, args.stress_rounds,
                               args.stress_concurrency, args.window_repeats]) < 1
            or not math.isfinite(args.seconds) or args.seconds <= 0 or args.warmup < 1):
        parser.error("counts, durations and warmup must be positive")
    if args.stress_requests * args.stress_rounds < args.stress_concurrency:
        parser.error("stress must contain at least one request per worker")
    if (not set(args.performance_models or args.models) <= set(args.models)
            or not set(args.performance_batch_sizes or args.batch_sizes) <= set(args.batch_sizes)):
        parser.error("performance selection must be a subset of the numerical gate")
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        os.chmod(args.output, 0o600)
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output.resolve()), "passed": report["passed"],
                      "windows": len(report["windows"]), "error_class": report.get("error_class")}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
