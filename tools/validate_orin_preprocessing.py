"""真实图片的隔离 CPU/GPU Triton 预处理验证；不写业务存储、不管理服务。

两个端点必须由调用方预先启动。HTTP wall 包含传输与输出物化；可选 PID
采集服务进程树 CPU 秒，不能用客户端 CPU 秒代替。报告不包含图片或特征原值。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

import numpy as np


MODELS = {
    "clip": ("clip_image_preprocess", "clip_ensemble_trt", "image", "unnorm_image_features", "embedding"),
    "sbs": ("sbs_r50_ibn_image_preprocess", "sbs_r50_ibn_image_ensemble", "image", "embedding", "embedding"),
    "keypoint": ("face_keypoint_2d106_pre", "face_keypoint_2d106_ensemble", "PREPROCESSED_IMAGE", "LANDMARKS_106", "keypoint"),
    "qa": ("face_qa_pre", "face_qa_ensemble", "QA_PREPROCESSED_IMAGE", "quality_score", "qa"),
    "recognition": ("face_rec_pre", "face_recognition_pipeline", "processed_input", "EMBEDDING", "embedding"),
}
DEFAULT_TOLERANCES = {"tensor_atol": 1e-5, "tensor_rtol": 1e-5,
                      "affine_atol": 1e-10, "keypoint_atol": 1e-3, "qa_atol": 1e-5,
                      "minimum_cosine": 0.99999, "maximum_relative_l2": 0.001}


def inventory(directory):
    from PIL import Image, ImageFile

    samples, errors, files = [], [], []
    previous_truncated = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        for path in sorted(Path(directory).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            info = {"path": str(path.resolve())}
            files.append(info)
            try:
                payload = path.read_bytes()
                info.update(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
                with Image.open(path) as image:
                    info.update(width=image.width, height=image.height, format=image.format, mode=image.mode)
                    image.verify()
                # JPEG verify() can accept a truncated scan. Decode pixels with strict loading.
                with Image.open(path) as image:
                    image.load()
                samples.append(info)
            except Exception as exc:
                errors.append(info | {"error_class": type(exc).__name__, "error": str(exc)})
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous_truncated
    return {"files": len(files), "valid_files": len(samples), "invalid_files": len(errors),
            "unique_sha256": len({s["sha256"] for s in files if "sha256" in s}),
            "bytes": sum(s.get("bytes", 0) for s in files),
            "validation": "Pillow verify plus full load; LOAD_TRUNCATED_IMAGES=False",
            "dimensions": dict(Counter(f"{s['width']}x{s['height']}" for s in samples)),
            "formats": dict(Counter(s["format"] for s in samples)), "samples": samples, "errors": errors}


def select_samples(samples, limit):
    """尺寸分层轮转选择，不用目录排序偷偷只覆盖一种分辨率。"""
    if limit < 1:
        raise ValueError("limit must be positive")
    groups = defaultdict(list)
    seen = set()
    for sample in samples:
        if sample["sha256"] not in seen:
            groups[(sample["height"], sample["width"])].append(sample)
            seen.add(sample["sha256"])
    chosen = []
    while any(groups.values()) and len(chosen) < limit:
        for key in sorted(groups):
            if groups[key] and len(chosen) < limit:
                chosen.append(groups[key].pop(0))
    return chosen


def groups_of(samples, size, *, same_shape):
    groups = defaultdict(list)
    for sample in samples:
        groups[(sample["height"], sample["width"]) if same_shape else ()].append(sample)
    result = []
    for group in groups.values():
        for offset in range(0, len(group), size):
            values = group[offset:offset + size]
            count = len(values)
            result.append({"samples": [values[i % count] for i in range(size)],
                           "repeated_slots": list(range(count, size))})
    return result


def compare_arrays(left, right, kind, tolerances):
    a, b = np.asarray(left), np.asarray(right)
    report = {"shape": list(a.shape), "candidate_shape": list(b.shape),
              "dtype": str(a.dtype), "candidate_dtype": str(b.dtype), "passed": False}
    expected_dtype = np.float64 if kind == "affine" else np.float32
    if a.shape != b.shape or not a.size or a.dtype != expected_dtype or b.dtype != expected_dtype:
        report["reason"] = "shape_or_dtype_contract"
        return report
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        report["reason"] = "nonfinite"
        return report
    x, y = a.astype(np.float64), b.astype(np.float64)
    delta = np.abs(x - y)
    report.update(max_abs=float(delta.max()), mean_abs=float(delta.mean()),
                  rmse=float(np.sqrt(np.mean(delta ** 2))), exact_equal=bool(np.array_equal(a, b)))
    if kind == "embedding":
        if x.ndim != 2:
            report["reason"] = "embedding_must_be_BxD"
            return report
        na, nb = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1)
        if np.any(na == 0) or np.any(nb == 0):
            report["reason"] = "zero_norm"
            return report
        cosine = np.clip(np.sum(x * y, axis=1) / (na * nb), -1, 1)
        relative_l2 = np.linalg.norm(x - y, axis=1) / na
        report.update(minimum_cosine=float(cosine.min()), maximum_relative_l2=float(relative_l2.max()))
        report["passed"] = bool(np.all(cosine >= tolerances["minimum_cosine"])
                                and np.all(relative_l2 <= tolerances["maximum_relative_l2"]))
    else:
        atol = tolerances[f"{kind}_atol"]
        rtol = tolerances["tensor_rtol"] if kind == "tensor" else 0
        report.update(atol=atol, rtol=rtol, passed=bool(np.allclose(x, y, atol=atol, rtol=rtol)))
    return report


def cpu_tree(pid):
    """返回 PID/start-time 身份键；PID 不可见时报告缺失，不伪造 CPU=0。"""
    if pid is None:
        return None
    processes = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            try:
                fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
                processes[int(path.name)] = (int(fields[1]), int(fields[19]), int(fields[11]) + int(fields[12]))
            except (OSError, ValueError, IndexError):
                pass
    if pid not in processes:
        raise ValueError(f"server PID {pid} is not visible in this PID namespace")
    admitted = {pid}
    while True:
        children = {key for key, value in processes.items() if value[0] in admitted} - admitted
        if not children:
            break
        admitted.update(children)
    return {(key, processes[key][1]): processes[key][2] for key in admitted}


def cpu_delta(before, after):
    if before is None or after is None:
        return {"seconds": None, "complete_process_set": False, "reason": "server_pid_not_supplied"}
    common = before.keys() & after.keys()
    value = sum(after[key] - before[key] for key in common)
    if value < 0:
        raise ValueError("CPU counter moved backwards")
    return {"seconds": value / os.sysconf("SC_CLK_TCK"),
            "complete_process_set": before.keys() == after.keys(), "processes": len(common),
            "missing_or_new_processes": len(before.keys() ^ after.keys())}


def validate_url(url):
    parsed = urlsplit(url if "://" in url else "http://" + url)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password
            or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        raise ValueError("URL must be HTTP(S) host:port without credentials or path")
    if parsed.port in {8000, 8001, 8003}:
        raise ValueError("use disposable evaluation ports, not the production 8000/8001/8003 ports")
    return parsed


class Endpoint:
    def __init__(self, url, pid=None):
        import tritonclient.http as http

        parsed = validate_url(url)
        self.url, self.pid, self.http = url, pid, http
        self.client = http.InferenceServerClient(parsed.netloc, ssl=parsed.scheme == "https",
                                                 connection_timeout=30, network_timeout=120)

    def close(self):
        self.client.close()

    def infer(self, model, inputs, output_names):
        from tritonclient.utils import np_to_triton_dtype

        tensors = []
        for name, value in inputs.items():
            value = np.ascontiguousarray(value)
            tensor = self.http.InferInput(name, value.shape, np_to_triton_dtype(value.dtype))
            tensor.set_data_from_numpy(value, binary_data=True)
            tensors.append(tensor)
        result = self.client.infer(model, tensors, outputs=[self.http.InferRequestedOutput(n) for n in output_names])
        outputs = {name: result.as_numpy(name) for name in output_names}
        if any(value is None for value in outputs.values()):
            raise ValueError("required output missing")
        return outputs


def inputs_for(alias, samples, phase):
    if alias in {"clip", "sbs"}:
        encoded = np.empty((len(samples), 1), dtype=object)
        for index, sample in enumerate(samples):
            encoded[index, 0] = sample["encoded"]
        return {"IMAGE_BYTES": encoded}
    images = np.stack([sample["image"] for sample in samples])
    if alias == "keypoint":
        return {"RAW_IMAGE": images, "BBOX": np.stack([s["bbox"] for s in samples])}
    if alias == "qa":
        return {"RAW_FACE_IMAGE": images, "LANDMARK": np.stack([s["landmark"] for s in samples])}
    names = ("raw_image", "landmark") if phase == "pre" else ("RAW_IMAGE", "LANDMARK")
    return {names[0]: images, names[1]: np.stack([s["landmark"] for s in samples])}


def output_kinds(alias, phase):
    _, _, pre_output, output, kind = MODELS[alias]
    if phase == "pre":
        return {pre_output: "tensor", **({"AFFINE_MATRIX": "affine"} if alias == "keypoint" else {})}
    return {output: kind}


def landmarks5(points):
    if points.shape != (106, 2) or not np.isfinite(points).all():
        raise ValueError("invalid detector keypoints")
    return np.asarray([points[33:43].mean(axis=0, dtype=np.float64),
                       points[87:97].mean(axis=0, dtype=np.float64),
                       points[86], points[52], points[61]], dtype=np.float32)


def crop_sample(sample, bbox, margin=1.0):
    height, width = sample["image"].shape[:2]
    center = (bbox[:2] + bbox[2:]) / 2
    half = (bbox[2:] - bbox[:2]) * margin / 2
    x1, y1 = np.maximum(np.floor(center - half), 0).astype(int)
    x2, y2 = np.minimum(np.ceil(center + half), [width, height]).astype(int)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("empty detected crop")
    image = np.ascontiguousarray(sample["image"][y1:y2, x1:x2])
    return sample | {"image": image, "height": int(y2 - y1), "width": int(x2 - x1),
                     "crop_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                     "bbox": (bbox - np.array([x1, y1, x1, y1], dtype=np.float32)).astype(np.float32)}


def collect_faces(endpoint, samples, limit):
    faces, diagnostics = [], []
    for sample in samples:
        if len(faces) >= limit:
            break
        result = endpoint.infer("yolov8n_face_ensemble", {"RAW_IMAGE": sample["image"][None]},
                                ["DETECTIONS", "NUM_DETECTIONS"])
        count = int(result["NUM_DETECTIONS"].reshape(-1)[0])
        detections = result["DETECTIONS"][0, :count]
        accepted = False
        for box in sorted(detections, key=lambda row: -float(row[4])):
            bbox = box[:4].astype(np.float32)
            if not np.isfinite(bbox).all() or min(bbox[2:] - bbox[:2]) < 8:
                continue
            cropped = crop_sample(sample, bbox, margin=1.5)
            result = endpoint.infer("face_keypoint_2d106_ensemble",
                                    {"RAW_IMAGE": cropped["image"][None], "BBOX": cropped["bbox"][None]}, ["LANDMARKS_106"])
            points = landmarks5(result["LANDMARKS_106"][0])
            if np.sqrt(np.mean(np.sum((points - points.mean(axis=0)) ** 2, axis=1))) < 1:
                continue
            faces.append(cropped | {"landmark": points})
            accepted = True
            break
        diagnostics.append({"sha256": sample["sha256"], "detections": count, "accepted": accepted})
    if not faces:
        raise ValueError("no real detected faces available; synthetic landmarks are not substituted")
    return faces, diagnostics


def collect_people(endpoint, samples, limit):
    import cv2

    people, diagnostics = [], []
    for sample in samples:
        if len(people) >= limit:
            break
        result = endpoint.infer("yolo26m_pose_ensemble", {"RAW_IMAGE": sample["image"][None]},
                                ["DETECTIONS", "NUM_DETECTIONS"])
        count = int(result["NUM_DETECTIONS"].reshape(-1)[0])
        accepted = 0
        for detection in sorted(result["DETECTIONS"][0, :count], key=lambda row: -float(row[4])):
            if len(people) >= limit:
                break
            bbox = detection[:4].astype(np.float32)
            if not np.isfinite(bbox).all() or min(bbox[2:] - bbox[:2]) < 16 or int(detection[5]) != 0:
                continue
            cropped = crop_sample(sample, bbox)
            ok, encoded = cv2.imencode(".jpg", cropped["image"], [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise ValueError("person JPEG encode failed")
            payload = encoded.tobytes()
            people.append(cropped | {"encoded": payload, "encoded_sha256": hashlib.sha256(payload).hexdigest(),
                                     "crop_encoding": "JPEG_quality95", "opencv_version": cv2.__version__})
            accepted += 1
        diagnostics.append({"sha256": sample["sha256"], "detections": count, "accepted": accepted})
    if not people:
        raise ValueError("no real person crops detected")
    return people, diagnostics


def sample_manifest(path, samples, *, kind, diagnostics=None):
    """只保存原图hash、裁图坐标和几何；图片从原图重建，人物编码hash复核。"""
    import cv2

    path = Path(path)
    if path.exists():
        data = json.loads(path.read_text())
        if data["kind"] != kind:
            raise ValueError("sample manifest kind mismatch")
        source = {s["sha256"]: s for s in samples}
        loaded = []
        for item in data["samples"]:
            original = source[item["sha256"]]
            x1, y1, x2, y2 = item["crop_xyxy"]
            image = np.ascontiguousarray(original["image"][y1:y2, x1:x2])
            value = original | item | {"image": image, "bbox": np.asarray(item["bbox"], dtype=np.float32)}
            if kind == "face":
                value["landmark"] = np.asarray(item["landmark"], dtype=np.float32)
            else:
                ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok or hashlib.sha256(encoded.tobytes()).hexdigest() != item["encoded_sha256"]:
                    raise ValueError("person crop encoding changed; keep the sampling OpenCV version")
                value["encoded"] = encoded.tobytes()
            loaded.append(value)
        return loaded, data["diagnostics"]
    serializable = []
    for sample in samples:
        item = {key: sample[key] for key in ("sha256", "path", "crop_xyxy", "width", "height")}
        item["bbox"] = sample["bbox"].tolist()
        if kind == "face":
            item["landmark"] = sample["landmark"].tolist()
        else:
            item.update({key: sample[key] for key in ("encoded_sha256", "crop_encoding", "opencv_version")})
        serializable.append(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        os.chmod(path, 0o600)
        json.dump({"kind": kind, "samples": serializable, "diagnostics": diagnostics}, stream, indent=2)
    return samples, diagnostics


def cross_samples(samples):
    """跨请求专项用不同真实样本；raw只在右/下补零到同尺寸，不计性能。"""
    selected = [samples[i % len(samples)] for i in range(4)]
    height = max(s["image"].shape[0] for s in selected)
    width = max(s["image"].shape[1] for s in selected)
    return [s | {"image": np.pad(s["image"], ((0, height - s["image"].shape[0]),
                                                 (0, width - s["image"].shape[1]), (0, 0)))} for s in selected]


def measured_calls(endpoint, model, inputs, outputs, repeats):
    before = cpu_tree(endpoint.pid)
    cpu_started, started = time.process_time(), time.perf_counter()
    latencies = []
    last = None
    for _ in range(repeats):
        tick = time.perf_counter()
        last = endpoint.infer(model, inputs, outputs)
        latencies.append(time.perf_counter() - tick)
    elapsed, client_cpu = time.perf_counter() - started, time.process_time() - cpu_started
    after = cpu_tree(endpoint.pid)
    return last, {"requests": repeats, "wall_seconds": elapsed, "client_cpu_seconds": client_cpu,
                  "server_cpu": cpu_delta(before, after), "latency_ms": [v * 1000 for v in latencies]}


def concurrent_requests(endpoint, model, cases, outputs):
    """每线程持独立 HTTP client，保留返回的 request 顺序。"""
    def call(inputs):
        client = Endpoint(endpoint.url)
        try:
            return {"outputs": client.infer(model, inputs, outputs), "error": None}
        except Exception as exc:
            return {"outputs": None, "error": type(exc).__name__}
        finally:
            client.close()
    with ThreadPoolExecutor(max_workers=len(cases)) as pool:
        return list(pool.map(call, cases))


def counter_delta(before, after):
    old = {int(x["batch_size"]): x for x in before.get("batch_stats", [])}
    batches = []
    for row in after.get("batch_stats", []):
        previous = old.get(int(row["batch_size"]), {})
        executions = int(row.get("compute_infer", {}).get("count", 0)) - int(previous.get("compute_infer", {}).get("count", 0))
        if executions:
            batches.append({"batch_size": int(row["batch_size"]), "executions": executions})
    return {"inferences": int(after.get("inference_count", 0)) - int(before.get("inference_count", 0)),
            "executions": int(after.get("execution_count", 0)) - int(before.get("execution_count", 0)),
            "observed_batches": batches}


def cross_request_checks(endpoint, alias, batch, tolerances):
    model = MODELS[alias][0]
    kinds = output_kinds(alias, "pre")
    cases = [inputs_for(alias, batch[:1], "pre"), inputs_for(alias, batch[1:4], "pre")]
    references = [endpoint.infer(model, case, kinds) for case in cases]
    before = endpoint.client.get_inference_statistics(model_name=model)["model_stats"][0]
    valid = concurrent_requests(endpoint, model, cases, kinds)
    after = endpoint.client.get_inference_statistics(model_name=model)["model_stats"][0]
    comparisons = [{name: compare_arrays(reference[name], result["outputs"][name], kind, tolerances)
                    for name, kind in kinds.items()} if result["outputs"] is not None else {"request_error": result["error"]}
                   for reference, result in zip(references, valid)]
    invalid = {name: value.copy() for name, value in cases[0].items()}
    if alias in {"clip", "sbs"}:
        invalid["IMAGE_BYTES"][0, 0] = b"not-an-image"
    else:
        key = "BBOX" if alias == "keypoint" else "LANDMARK" if alias == "qa" else "landmark"
        invalid[key].flat[0] = np.nan
    mixed = concurrent_requests(endpoint, model, [cases[0], invalid, cases[1]], kinds)
    isolation = []
    for index, reference in zip((0, 2), references):
        result = mixed[index]
        isolation.append({name: compare_arrays(reference[name], result["outputs"][name], kind, tolerances)
                          for name, kind in kinds.items()} if result["outputs"] is not None else {"request_error": result["error"]})
    valid_pass = all(all(value.get("passed", False) for value in group.values()) for group in comparisons if "request_error" not in group)
    isolation_pass = all(all(value.get("passed", False) for value in group.values()) for group in isolation if "request_error" not in group)
    passed = valid_pass and isolation_pass and not any("request_error" in group for group in comparisons + isolation) and mixed[1]["error"] is not None
    return {"passed": passed, "valid_B1_B3": comparisons, "batch_statistics": counter_delta(before, after),
            "invalid_request_rejected": mixed[1]["error"] is not None, "valid_neighbors": isolation,
            "scope": "concurrent HTTP requests; batch statistics show whether Triton actually coalesced them"}


def validate_endpoints(endpoints, aliases):
    metadata = {}
    for side, endpoint in endpoints.items():
        if not endpoint.client.is_server_ready():
            raise ValueError(f"{side} server is not ready")
        models = {}
        for alias in aliases:
            pre, ensemble, *_ = MODELS[alias]
            for model in (pre, ensemble):
                if not endpoint.client.is_model_ready(model):
                    raise ValueError(f"{side}/{model} is not ready")
            config = endpoint.client.get_model_config(pre)
            backend = config.get("parameters", {}).get("preprocess_backend", {}).get("string_value", "cpu")
            if backend != ("orin_gpu" if side == "candidate" else "cpu"):
                raise ValueError(f"{side}/{pre} has unexpected preprocess_backend={backend}")
            models[pre] = config
        metadata[side] = {"server": endpoint.client.get_server_metadata(), "models": models}
    return metadata


def timing_summary(runs):
    groups = defaultdict(list)
    for run in runs:
        groups[(run["alias"], run["phase"], run["batch_size"])].append(run)
    summary = []
    for (alias, phase, size), values in groups.items():
        sides = {}
        for side in ("baseline", "candidate"):
            timings = [value["timing"][side] for value in values]
            requests = sum(t["requests"] for t in timings)
            wall = sum(t["wall_seconds"] for t in timings)
            latencies = [v for t in timings for v in t["latency_ms"]]
            complete = all(t["server_cpu"]["seconds"] is not None and t["server_cpu"]["complete_process_set"] for t in timings)
            sides[side] = {"requests": requests, "images_including_tail_repeats": requests * size,
                           "wall_seconds": wall, "http_images_per_second": requests * size / wall,
                           "client_cpu_seconds": sum(t["client_cpu_seconds"] for t in timings),
                           "server_cpu_seconds": sum(t["server_cpu"]["seconds"] for t in timings) if complete else None,
                           "server_cpu_complete": complete,
                           "latency_mean_ms": float(np.mean(latencies)), "latency_p95_ms": float(np.percentile(latencies, 95))}
        summary.append({"alias": alias, "phase": phase, "batch_size": size, **sides})
    return summary


def run(args):
    import cv2

    dataset = inventory(args.dataset)
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "dataset": dataset, "passed": False, "tolerances": dict(DEFAULT_TOLERANCES), "runs": [], "cross_request": []}
    report["client_runtime"] = {"numpy": np.__version__, "opencv": cv2.__version__,
                                 "tritonclient": importlib.metadata.version("tritonclient"),
                                 "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.inventory_only:
        report["passed"] = bool(dataset["samples"]) and not dataset["errors"]
        return report
    if args.baseline_url == args.candidate_url:
        raise ValueError("CPU and GPU endpoints must be distinct disposable servers")
    endpoints = {}
    context = {"stage": "endpoint_initialization"}
    try:
        report["invalid_image_policy"] = "explicitly_exclude_and_report" if args.exclude_invalid_images else "fail"
        if dataset["errors"] and not args.exclude_invalid_images:
            raise ValueError("dataset contains invalid images; inspect dataset.errors and explicitly use --exclude-invalid-images to evaluate valid images")
        if not dataset["samples"]:
            raise ValueError("dataset has no valid images")
        endpoints["baseline"] = Endpoint(args.baseline_url, args.baseline_pid)
        if not args.collect_samples_only:
            endpoints["candidate"] = Endpoint(args.candidate_url, args.candidate_pid)
            report["endpoint_metadata"] = validate_endpoints(endpoints, args.models)
        samples = []
        report["local_decode_diagnostics"] = []
        for item in select_samples(dataset["samples"], len(dataset["samples"]) if args.collect_samples_only or args.face_manifest or args.person_manifest else args.limit):
            payload = Path(item["path"]).read_bytes()
            if hashlib.sha256(payload).hexdigest() != item["sha256"]:
                raise ValueError("sample changed after inventory")
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                report["local_decode_diagnostics"].append({"sha256": item["sha256"], "path": item["path"],
                                                            "error": "OpenCV decode failed", "opencv_version": cv2.__version__})
                raise ValueError(f"OpenCV {cv2.__version__} decode failed for {item['path']}; use the runtime-matching decoder, do not silently exclude samples")
            samples.append(item | {"encoded": payload, "image": image})
        faces, people = [], []
        if any(alias not in {"clip", "sbs"} for alias in args.models):
            if args.face_manifest and args.face_manifest.exists():
                faces, report["face_sampling"] = sample_manifest(args.face_manifest, samples, kind="face")
            else:
                faces, report["face_sampling"] = collect_faces(endpoints["baseline"], samples, args.face_limit)
                if args.face_manifest:
                    sample_manifest(args.face_manifest, faces, kind="face", diagnostics=report["face_sampling"])
            faces = faces[:args.face_limit]
            report["face_sample_sha256"] = [s["sha256"] for s in faces]
        if "sbs" in args.models:
            if args.person_manifest and args.person_manifest.exists():
                people, report["person_sampling"] = sample_manifest(args.person_manifest, samples, kind="person")
            else:
                people, report["person_sampling"] = collect_people(endpoints["baseline"], samples, args.person_limit)
                if args.person_manifest:
                    sample_manifest(args.person_manifest, people, kind="person", diagnostics=report["person_sampling"])
            people = people[:args.person_limit]
            report["person_crop_sha256"] = [s["encoded_sha256"] for s in people]
        report["sample_manifests"] = {kind: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                        for kind, path in (("face", args.face_manifest), ("person", args.person_manifest)) if path and path.exists()}
        samples = select_samples(samples, args.limit)
        report["selected_sha256"] = [s["sha256"] for s in samples]
        if args.collect_samples_only:
            report["passed"] = True
            return report
        report["timing_scope"] = "HTTP binary request/response including output materialization; excludes sampling, warmup, correctness, CPU snapshots; process CPU includes server descendants and any unrelated work in that tree. GPU preprocessing uses explicit reduced-tensor D2H for the Orin backend boundary; ensemble timing includes that D2H, native TensorRT H2D and inference. GPU arithmetic does not imply zero-copy. These are sequential model requests, not whole ALog throughput."
        report["face_sampling_scope"] = "baseline real face detector; 1.5x bbox crop; crop-relative bbox and 106-to-5 landmarks; raw B4 grouped by exact crop H/W, tail repeats explicit; cross-request-only samples right/bottom zero-padded"
        report["sbs_sampling_scope"] = "baseline real pose detector person boxes; JPEG quality95 crops, encoded SHA256 verified; no segmentation mask (not a complete ALog benchmark)"
        for alias in args.models:
            selected = samples if alias == "clip" else people if alias == "sbs" else faces
            for size in args.batch_sizes:
                for batch_index, batch in enumerate(groups_of(selected, size, same_shape=alias not in {"clip", "sbs"})):
                    for phase in ("pre", "ensemble"):
                        model = MODELS[alias][0 if phase == "pre" else 1]
                        inputs = inputs_for(alias, batch["samples"], phase)
                        kinds = output_kinds(alias, phase)
                        values, timing = {}, {}
                        order = ("baseline", "candidate") if batch_index % 2 == 0 else ("candidate", "baseline")
                        for side in order:
                            endpoint = endpoints[side]
                            context = {"stage": "warmup", "side": side, "alias": alias, "phase": phase,
                                       "model": model, "batch_size": size, "batch_index": batch_index,
                                       "sample_sha256": [s["sha256"] for s in batch["samples"]]}
                            for _ in range(args.warmup):
                                endpoint.infer(model, inputs, kinds)
                            context = context | {"stage": "measurement"}
                            values[side], timing[side] = measured_calls(endpoint, model, inputs, kinds, args.repeats)
                        comparisons = {name: compare_arrays(values["baseline"][name], values["candidate"][name], kind, report["tolerances"])
                                       for name, kind in kinds.items()}
                        report["runs"].append({"alias": alias, "phase": phase, "model": model, "batch_size": size,
                                               "sample_sha256": [s["sha256"] for s in batch["samples"]],
                                               "repeated_slots": batch["repeated_slots"], "timing": timing,
                                               "comparisons": comparisons, "passed": all(c["passed"] for c in comparisons.values())})
            for side, endpoint in endpoints.items():
                context = {"stage": "cross_request", "side": side, "alias": alias}
                report["cross_request"].append({"alias": alias, "side": side,
                                                 **cross_request_checks(endpoint, alias, cross_samples(selected), report["tolerances"])})
        report["passed"] = all(r["passed"] for r in report["runs"] + report["cross_request"])
        report["timing_summary"] = timing_summary(report["runs"])
    except Exception as exc:
        report.update(passed=False, error_class=type(exc).__name__, error=str(exc), failure_context=context)
    finally:
        for endpoint in endpoints.values():
            endpoint.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--exclude-invalid-images", action="store_true",
                        help="explicitly exclude fully decoded inventory failures, preserving their hashes and errors in the report")
    parser.add_argument("--collect-samples-only", action="store_true")
    parser.add_argument("--face-manifest", type=Path)
    parser.add_argument("--person-manifest", type=Path)
    parser.add_argument("--baseline-url")
    parser.add_argument("--candidate-url")
    parser.add_argument("--baseline-pid", type=int)
    parser.add_argument("--candidate-pid", type=int)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--batch-sizes", nargs="+", type=int, choices=(1, 4), default=[1, 4])
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--face-limit", type=int, default=8)
    parser.add_argument("--person-limit", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not args.inventory_only and (not args.baseline_url or (not args.collect_samples_only and not args.candidate_url)):
        parser.error("benchmark requires --baseline-url and --candidate-url")
    if min(args.limit, args.face_limit, args.person_limit, args.repeats) < 1 or args.warmup < 0:
        parser.error("limits/repeats must be positive and warmup nonnegative")
    if args.output.exists():
        parser.error("output already exists; do not overwrite prior evidence")
    try:
        report = run(args)
    except Exception as exc:
        report = {"passed": False, "error_class": type(exc).__name__, "error": str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        os.chmod(args.output, 0o600)
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(args.output.resolve()), "passed": report["passed"],
                      "runs": len(report.get("runs", [])), "error_class": report.get("error_class")}, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
