"""验证评测口径，单测不访问 Triton 或 CUDA。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "tools/validate_orin_preprocessing.py"
SPEC = importlib.util.spec_from_file_location("validate_orin_preprocessing", PATH)
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)


class ValidationAccountingTests(unittest.TestCase):
    def test_inventory_fully_decodes_and_reports_truncated_jpeg(self):
        from PIL import Image, ImageFile

        with tempfile.TemporaryDirectory() as directory:
            valid = Path(directory) / "valid.jpg"
            Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(valid)
            truncated = Path(directory) / "truncated.jpg"
            truncated.write_bytes(valid.read_bytes()[:-2])
            with patch.object(ImageFile, "LOAD_TRUNCATED_IMAGES", True):
                report = validation.inventory(directory)
                self.assertTrue(ImageFile.LOAD_TRUNCATED_IMAGES)
        self.assertEqual(report["files"], 2)
        self.assertEqual(report["valid_files"], 1)
        self.assertEqual(report["invalid_files"], 1)
        self.assertEqual(report["errors"][0]["path"], str(truncated))
        self.assertEqual(report["errors"][0]["error_class"], "OSError")
        self.assertIn("truncated", report["errors"][0]["error"])
        self.assertEqual(len(report["errors"][0]["sha256"]), 64)

    def test_sampler_covers_dimensions_and_deduplicates_bytes(self):
        samples = [{"sha256": "a", "height": 10, "width": 20},
                   {"sha256": "a", "height": 10, "width": 20},
                   {"sha256": "b", "height": 10, "width": 20},
                   {"sha256": "c", "height": 20, "width": 20}]
        self.assertEqual([s["sha256"] for s in validation.select_samples(samples, 2)], ["a", "c"])

    def test_raw_batches_do_not_mix_shape_and_disclose_repeated_tail(self):
        samples = [{"sha256": "a", "height": 10, "width": 20},
                   {"sha256": "b", "height": 10, "width": 20},
                   {"sha256": "c", "height": 20, "width": 20}]
        batches = validation.groups_of(samples, 4, same_shape=True)
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0]["repeated_slots"], [2, 3])
        self.assertEqual([s["sha256"] for s in batches[0]["samples"]], ["a", "b", "a", "b"])

    def test_embedding_cosine_alone_does_not_hide_scale_error(self):
        value = np.ones((2, 8), np.float32)
        result = validation.compare_arrays(value, value * 2, "embedding", validation.DEFAULT_TOLERANCES)
        self.assertGreater(result["minimum_cosine"], 0.99999)
        self.assertFalse(result["passed"])
        self.assertFalse(validation.compare_arrays(value, value.astype(np.float64), "embedding", validation.DEFAULT_TOLERANCES)["passed"])
        self.assertFalse(validation.compare_arrays(value * 0, value * 0, "embedding", validation.DEFAULT_TOLERANCES)["passed"])

    def test_nonfinite_and_affine_precision_are_checked(self):
        affine = np.zeros((1, 2, 3), np.float64)
        self.assertTrue(validation.compare_arrays(affine, affine.copy(), "affine", validation.DEFAULT_TOLERANCES)["passed"])
        changed = affine.copy()
        changed[0, 0, 0] = 1e-8
        self.assertFalse(validation.compare_arrays(affine, changed, "affine", validation.DEFAULT_TOLERANCES)["passed"])
        changed[0, 0, 0] = np.nan
        self.assertEqual(validation.compare_arrays(affine, changed, "affine", validation.DEFAULT_TOLERANCES)["reason"], "nonfinite")

    def test_cpu_missing_is_not_zero_and_pid_reuse_not_counted(self):
        self.assertIsNone(validation.cpu_delta(None, None)["seconds"])
        with patch.object(validation.os, "sysconf", return_value=100):
            delta = validation.cpu_delta({(42, 10): 100, (43, 11): 20}, {(42, 10): 150, (43, 99): 900})
        self.assertEqual(delta["seconds"], 0.5)
        self.assertFalse(delta["complete_process_set"])
        self.assertEqual(delta["missing_or_new_processes"], 2)

    def test_production_ports_and_credentials_rejected(self):
        for url in ("localhost:8000", "http://user:password@localhost:18100", "http://localhost:18100/path"):
            with self.assertRaises(ValueError):
                validation.validate_url(url)
        self.assertEqual(validation.validate_url("127.0.0.1:18100").port, 18100)

    def test_crop_coordinates_relative_and_cross_padding_is_bottom_right(self):
        original = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
        sample = {"image": original, "sha256": "a", "width": 30, "height": 20}
        crop = validation.crop_sample(sample, np.array([10, 5, 20, 15], np.float32), margin=1.5)
        self.assertEqual(crop["crop_xyxy"], [7, 2, 23, 18])
        np.testing.assert_array_equal(crop["bbox"], [3, 3, 13, 13])
        padded = validation.cross_samples([crop, sample])
        self.assertEqual(padded[0]["image"].shape, original.shape)
        np.testing.assert_array_equal(padded[0]["image"][:16, :16], original[2:18, 7:23])
        self.assertFalse(padded[0]["image"][16:].any())

    def test_face_manifest_roundtrip_does_not_resample(self):
        image = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
        sample = {"image": image, "sha256": "a", "width": 30, "height": 20, "path": "image.jpg"}
        crop = validation.crop_sample(sample, np.array([10, 5, 20, 15], np.float32))
        crop["landmark"] = np.arange(10, dtype=np.float32).reshape(5, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "faces.json"
            validation.sample_manifest(path, [crop], kind="face", diagnostics=[{"accepted": True}])
            stored = json.loads(path.read_text())
            self.assertNotIn("image", stored["samples"][0])
            loaded, diagnostics = validation.sample_manifest(path, [sample], kind="face")
        np.testing.assert_array_equal(loaded[0]["image"], crop["image"])
        np.testing.assert_array_equal(loaded[0]["bbox"], crop["bbox"])
        np.testing.assert_array_equal(loaded[0]["landmark"], crop["landmark"])
        self.assertEqual(diagnostics, [{"accepted": True}])

    def test_observed_batch_counts_do_not_claim_unobserved_coalescing(self):
        before = {"inference_count": "4", "execution_count": "4", "batch_stats": [{"batch_size": "1", "compute_infer": {"count": "4"}}]}
        after = {"inference_count": "8", "execution_count": "5", "batch_stats": [
            {"batch_size": "1", "compute_infer": {"count": "4"}}, {"batch_size": "4", "compute_infer": {"count": "1"}}]}
        self.assertEqual(validation.counter_delta(before, after), {"inferences": 4, "executions": 1,
                                                                   "observed_batches": [{"batch_size": 4, "executions": 1}]})

    def test_summary_divides_total_images_by_total_wall_and_refuses_partial_cpu(self):
        runs = []
        for requests, wall, complete in ((1, 1.0, True), (3, 2.0, False)):
            timing = {"requests": requests, "wall_seconds": wall, "client_cpu_seconds": 0.1,
                      "server_cpu": {"seconds": 0.2, "complete_process_set": complete},
                      "latency_ms": [wall / requests * 1000] * requests}
            runs.append({"alias": "clip", "phase": "ensemble", "batch_size": 4,
                         "timing": {"baseline": timing, "candidate": timing}})
        report = validation.timing_summary(runs)[0]["candidate"]
        self.assertEqual(report["http_images_per_second"], 16 / 3)
        self.assertIsNone(report["server_cpu_seconds"])
        self.assertFalse(report["server_cpu_complete"])


if __name__ == "__main__":
    unittest.main()
