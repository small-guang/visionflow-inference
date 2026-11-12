import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image


BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "tools" / "benchmark_yolo_gpu.py"


def load_benchmark():
    assert BENCHMARK_PATH.is_file(), "The CPU/GPU benchmark tool has not been implemented"
    spec = importlib.util.spec_from_file_location("yolo_gpu_benchmark", BENCHMARK_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_dataset(root):
    test_dir = root / "test"
    test_dir.mkdir()
    for name in ("02.jpg", "01.jpg"):
        Image.new("RGB", (8, 4), (20, 40, 60)).save(test_dir / name)
    original = root / "face_person_ab_v1" / "pool" / "images" / "original"
    original.mkdir(parents=True)
    records = []
    for names, size in ((["a4", "a1", "a3", "a0", "a2"], (8, 4)),
                        (["s3", "s0", "s2", "s1"], (6, 6))):
        for name in names:
            path = original / f"{name}.jpg"
            Image.new("RGB", size, (30, 50, 70)).save(path)
            records.append({"original_path": f"pool/images/original/{name}.jpg",
                            "original_width": size[0], "original_height": size[1]})
    (original.parents[1] / "samples.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )
    snapshot = root / "face_person_ab_v1_snapshot_old" / "test"
    snapshot.mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(snapshot / "snapshot.jpg")


def outputs(batch=1, capacity=2, *, pose=False, seg=False):
    result = {"NUM_DETECTIONS": np.ones((batch, 1), dtype=np.int32),
              "DETECTIONS": np.zeros((batch, capacity, 6), dtype=np.float32)}
    result["DETECTIONS"][:, 0] = [1, 2, 3, 4, 0.9, 0]
    if pose:
        result["KEYPOINTS"] = np.zeros((batch, capacity, 17, 3), dtype=np.float32)
    if seg:
        result["MASK_CROPS_PACKED"] = np.zeros((batch, capacity, 160, 20), dtype=np.uint8)
    return result


class YoloGpuBenchmarkTests(unittest.TestCase):
    def test_manifest_is_deterministic_and_uses_test_then_original_dimension_extremes(self):
        module = load_benchmark()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_dataset(root)
            first = module.sample_images(root, limit=64)
            second = module.sample_images(root, limit=64)
            self.assertEqual(first, second)
            self.assertEqual([Path(row["path"]).name for row in first],
                             ["01.jpg", "02.jpg", "s0.jpg", "s2.jpg", "s3.jpg",
                              "a0.jpg", "a2.jpg", "a4.jpg"])
            self.assertTrue(all("snapshot" not in row["path"] for row in first))
            self.assertTrue(all(len(row["sha256"]) == 64 for row in first))
            self.assertEqual([row["id"] for row in module.sample_images(root, limit=2)],
                             ["test/01.jpg", "test/02.jpg"])

    def test_manifest_rejects_original_metadata_escaping_into_snapshot(self):
        module = load_benchmark()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_dataset(root)
            manifest = root / "face_person_ab_v1" / "pool" / "samples.jsonl"
            with manifest.open("a", encoding="utf-8") as stream:
                stream.write('\n{"original_path":"../face_person_ab_v1_snapshot_old/test/snapshot.jpg",'
                             '"original_width":2,"original_height":2}')
            with self.assertRaises(ValueError):
                module.sample_images(root, limit=64)

    def test_b4_batches_group_dimensions_and_record_each_repeated_slot(self):
        module = load_benchmark()
        samples = [{"id": name, "width": width, "height": height}
                   for name, width, height in [("a", 8, 4), ("s", 6, 6),
                                              ("b", 8, 4), ("c", 8, 4), ("d", 8, 4)]]
        batches = module.build_batches(samples, 4)
        self.assertEqual(len(batches), 2)
        self.assertEqual([row["id"] for row in batches[0]["samples"]], ["a", "b", "c", "d"])
        self.assertEqual(batches[0]["repeated_slots"], [])
        self.assertEqual([row["id"] for row in batches[1]["samples"]], ["s"] * 4)
        self.assertEqual(batches[1]["repeated_slots"], [1, 2, 3])
        for batch in batches:
            self.assertEqual(len({(row["width"], row["height"]) for row in batch["samples"]}), 1)

    def test_parity_checks_padding_classes_and_pose_keypoints(self):
        module = load_benchmark()
        cpu = outputs(pose=True)
        gpu = {name: value.copy() for name, value in cpu.items()}
        self.assertTrue(module.compare_outputs(cpu, gpu)["passed"])
        gpu["KEYPOINTS"][0, 0, 0, 0] = 1
        self.assertFalse(module.compare_outputs(cpu, gpu)["passed"])
        gpu = {name: value.copy() for name, value in cpu.items()}
        gpu["DETECTIONS"][0, 1, 4] = 0.1
        self.assertFalse(module.compare_outputs(cpu, gpu)["passed"])
        gpu = {name: value.copy() for name, value in cpu.items()}
        gpu["DETECTIONS"][0, 0, 5] = 1
        self.assertFalse(module.compare_outputs(cpu, gpu)["passed"])

    def test_masks_report_unpacked_bit_mismatch_and_iou_per_stable_slot(self):
        module = load_benchmark()
        cpu = outputs(seg=True)
        gpu = {name: value.copy() for name, value in cpu.items()}
        cpu["MASK_CROPS_PACKED"][0, 0, 0, 0] = 128
        gpu["MASK_CROPS_PACKED"][0, 0, 0, 0] = 192
        parity = module.compare_outputs(cpu, gpu)
        self.assertFalse(parity["passed"])
        mask = parity["masks"][0]
        self.assertEqual(mask["slot"], 0)
        self.assertEqual(mask["bit_mismatch"], 1)
        self.assertEqual(mask["iou"], 0.5)
        self.assertTrue(module.compare_outputs(outputs(seg=True), outputs(seg=True))["passed"])

    def test_count_shape_and_nonfinite_changes_fail_parity(self):
        module = load_benchmark()
        cpu = outputs()
        gpu = outputs()
        gpu["NUM_DETECTIONS"][0, 0] = 0
        self.assertFalse(module.compare_outputs(cpu, gpu)["passed"])
        self.assertFalse(module.compare_outputs(cpu, outputs(capacity=3))["passed"])
        gpu = outputs()
        gpu["DETECTIONS"][0, 0, 0] = np.nan
        self.assertFalse(module.compare_outputs(cpu, gpu)["passed"])

    def test_server_processing_delta_excludes_previous_requests_and_zero_counts(self):
        module = load_benchmark()
        before = {"model_stats": [{"name": "face_preprocess", "version": "1",
                  "inference_stats": {"compute_infer": {"count": "5", "ns": "20000000"},
                                      "compute_input": {"count": "0", "ns": "0"}},
                  "batch_stats": []}]}
        after = {"model_stats": [{"name": "face_preprocess", "version": "1",
                 "inference_stats": {"compute_infer": {"count": "7", "ns": "24000000"},
                                     "compute_input": {"count": "0", "ns": "0"}},
                 "batch_stats": []}]}
        values = module.processing_statistics(before, after)
        self.assertEqual(values[0]["name"], "face_preprocess")
        self.assertEqual(values[0]["compute_infer"],
                         {"count": 2, "total_ms": 4.0, "mean_ms": 2.0})
        self.assertIsNone(values[0]["compute_input"]["mean_ms"])

    def test_real_report_and_cli_exit_fail_when_external_gpu_outputs_differ(self):
        module = load_benchmark()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_dataset(root)
            report_path = root / "benchmark.json"

            def infer(url, model, image_batch, output_names):
                result = outputs(batch=image_batch.shape[0], capacity=100)
                if url == "gpu:8000":
                    result["DETECTIONS"][:, 0, 0] += 1
                return result

            status = module.main(["--dataset-root", str(root), "--models", "face",
                                  "--cpu-url", "cpu:8000", "--gpu-url", "gpu:8000",
                                  "--limit", "1", "--warmup", "3", "--repeats", "10",
                                  "--output", str(report_path)], infer=infer)
            self.assertEqual(status, 1)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertEqual(len(report["samples"]), 1)
            self.assertEqual({row["batch_size"] for row in report["runs"]}, {1, 4})
            self.assertEqual([row["requests_per_server"] for row in report["summary"]], [10, 10])
            self.assertEqual([row["repeated_image_slots"] for row in report["summary"]], [0, 30])
            for run in report["runs"]:
                self.assertEqual(len(run["cpu"]["latency_ms"]), 10)
                self.assertIn("p95_ms", run["gpu"])
                self.assertFalse(run["parity"]["passed"])
                self.assertEqual(run["parity"]["comparisons"], 10)


if __name__ == "__main__":
    unittest.main()
