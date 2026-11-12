import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "tools" / "benchmark_image_gpu.py"


def load_benchmark():
    assert BENCHMARK_PATH.is_file(), "The encoded image GPU benchmark has not been implemented"
    spec = importlib.util.spec_from_file_location("image_gpu_benchmark", BENCHMARK_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_dataset(root, count=5):
    folder = root / "test"
    folder.mkdir()
    for index in range(count):
        path = folder / f"{index:02}.png"
        Image.new("RGB", (5 + index, 4 + index), (index * 20, 30, 40)).save(path)
        # Decoders permit trailing data. These bytes must survive the HTTP input.
        path.write_bytes(path.read_bytes() + b"\x00original\x00")


def stats(count, ns, *, batch_size=4):
    timing = {"count": str(count), "ns": str(ns)}
    return {"model_stats": [{"name": "example", "version": "1",
                             "inference_count": str(count * batch_size),
                             "execution_count": str(count),
                             "inference_stats": {"success": timing, "compute_infer": timing},
                             "batch_stats": [{"batch_size": str(batch_size),
                                              "compute_infer": timing}]}]}


class ImageGpuBenchmarkTests(unittest.TestCase):
    def test_neon_variant_is_opt_in_and_rejects_tbps(self):
        module = load_benchmark()
        self.assertEqual(module.VARIANTS, ("cpu_fused", "gpu_normalize", "gpu_resize"))
        for alias in ("clip_image", "sbs"):
            self.assertEqual(module.candidate_model(alias, "gpu_resize_neon"),
                             f"benchmark_{alias}_gpu_resize_neon")
        with self.assertRaisesRegex(ValueError, "tbps_clip_image.*gpu_resize_neon"):
            module.candidate_model("tbps_clip_image", "gpu_resize_neon")
        with self.assertRaises(SystemExit) as error:
            module.main(["--models", "tbps_clip_image", "--variants", "gpu_resize_neon"])
        self.assertEqual(error.exception.code, 2)

    def test_explicit_neon_benchmark_requests_only_selected_variant(self):
        module = load_benchmark()
        calls = []

        def infer(url, model, payload, output):
            calls.append(model)
            width = 2048 if model.startswith("sbs_") or model.startswith("benchmark_sbs_") else 512
            return np.ones((payload.shape[0], width), np.float32)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_dataset(root, 1)
            output = root / "report.json"
            code = module.main(["--dataset-root", str(root), "--models", "clip_image", "sbs",
                                "--variants", "gpu_resize_neon", "--batch-sizes", "1", "--repeats", "10",
                                "--output", str(output)], infer=infer)
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(report["variants"], ["baseline", "gpu_resize_neon"])
        self.assertEqual(set(calls), {"clip_ensemble_trt", "sbs_r50_ibn_image_ensemble",
                                     "benchmark_clip_image_gpu_resize_neon", "benchmark_sbs_gpu_resize_neon"})
        self.assertTrue(all(row["passed"] for row in report["summary"]))

    def test_exact_public_output_contracts(self):
        module = load_benchmark()
        self.assertEqual(module.MODELS["clip_image"],
                         {"baseline": "clip_ensemble_trt", "output": "unnorm_image_features", "dimension": 512})
        self.assertEqual(module.MODELS["tbps_clip_image"]["output"], "image_feat")
        self.assertEqual(module.MODELS["sbs"],
                         {"baseline": "sbs_r50_ibn_image_ensemble", "output": "embedding", "dimension": 2048})
        self.assertEqual(module.candidate_model("sbs", "gpu_resize"), "benchmark_sbs_gpu_resize")

    def test_batches_preserve_mixed_dimensions_and_record_final_padding(self):
        module = load_benchmark()
        samples = [{"id": str(i), "width": 5 + i, "height": 4 + i} for i in range(5)]
        batches = module.build_batches(samples, 4)
        self.assertEqual([[s["id"] for s in b["samples"]] for b in batches],
                         [["0", "1", "2", "3"], ["4"] * 4])
        self.assertEqual(batches[0]["repeated_slots"], [])
        self.assertEqual(batches[1]["repeated_slots"], [1, 2, 3])
        self.assertEqual(batches[0]["input_shape"], [4, 1])
        self.assertEqual(len(module.build_batches(samples, 1)), 5)
        with self.assertRaises(ValueError):
            module.build_batches(samples, 0)

    def test_original_bytes_and_sha256_are_preserved_in_object_array(self):
        module = load_benchmark()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_dataset(root, 2)
            # Same encoded contents at another path do not add an independent sample.
            (root / "test" / "duplicate.png").write_bytes((root / "test" / "00.png").read_bytes())
            samples = module.sample_images(root)
            self.assertEqual(len(samples), 2)
            batch = module.encoded_batch(module.build_batches(samples, 4)[0])
            self.assertEqual(batch.dtype, np.dtype(object))
            self.assertEqual(batch.shape, (4, 1))
            for index, sample in enumerate([samples[0], samples[1]] * 2):
                original = Path(sample["path"]).read_bytes()
                self.assertEqual(batch[index, 0], original)
                self.assertTrue(batch[index, 0].endswith(b"\x00original\x00"))
                self.assertEqual(sample["sha256"], hashlib.sha256(original).hexdigest())
            Path(samples[0]["path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                module.encoded_batch(module.build_batches(samples, 1)[0])

    def test_vector_metrics_reject_norm_drift_even_with_perfect_cosine(self):
        module = load_benchmark()
        reference = np.array([[1, 2, 3], [3, 2, 1]], dtype=np.float32)
        exact = module.compare_vectors(reference, reference.copy(), expected_batch_size=2, expected_dimension=3)
        self.assertTrue(exact["passed"])
        self.assertTrue(exact["exact_equal"])
        self.assertEqual(exact["max_abs_error"], 0)
        self.assertEqual(exact["vectors"][0]["relative_l2"], 0)
        self.assertAlmostEqual(exact["vectors"][0]["baseline_norm"], np.sqrt(14))
        scaled = module.compare_vectors(reference, reference * 1.002)
        self.assertFalse(scaled["passed"])
        self.assertAlmostEqual(scaled["vectors"][0]["cosine_similarity"], 1)
        self.assertGreater(scaled["vectors"][0]["relative_l2"], 0.001)
        self.assertTrue(module.compare_vectors(reference, reference * 1.0005)["passed"])
        turned = reference.copy()
        turned[0, 0] += 0.1
        self.assertFalse(module.compare_vectors(reference, turned)["passed"])

    def test_vector_comparison_strictly_checks_contract_and_finiteness(self):
        module = load_benchmark()
        good = np.ones((2, 3), dtype=np.float32)
        for bad in (good.astype(np.float64), good[:, :2], good[0],
                    np.full_like(good, np.nan), np.full_like(good, np.inf), np.zeros_like(good)):
            with self.subTest(shape=bad.shape, dtype=bad.dtype):
                self.assertFalse(module.compare_vectors(good, bad)["passed"])
        self.assertFalse(module.compare_vectors(good, good, expected_batch_size=1)["passed"])
        self.assertFalse(module.compare_vectors(good, good, expected_dimension=4)["passed"])

    def test_processing_statistics_include_counts_batches_and_detect_resets(self):
        module = load_benchmark()
        delta = module.processing_statistics(stats(2, 8_000_000), stats(5, 20_000_000))[0]
        self.assertEqual(delta["inference_count"], 12)
        self.assertEqual(delta["execution_count"], 3)
        self.assertEqual(delta["inference_stats"]["compute_infer"],
                         {"count": 3, "total_ms": 12, "mean_ms": 4})
        self.assertEqual(delta["batch_stats"][0]["batch_size"], 4)
        self.assertEqual(delta["batch_stats"][0]["compute_infer"]["count"], 3)
        with self.assertRaisesRegex(ValueError, "reset"):
            module.processing_statistics(stats(5, 20_000_000), stats(2, 8_000_000))

    def test_http_uses_binary_bytes_and_deduplicates_same_endpoint(self):
        module = load_benchmark()
        observed = {}

        class Input:
            def __init__(self, name, shape, datatype):
                observed.update(name=name, shape=shape, datatype=datatype)

            def set_data_from_numpy(self, data, binary_data):
                observed.update(data=data, binary_data=binary_data)

        class Client:
            def __init__(self, **kwargs):
                observed["client_count"] = observed.get("client_count", 0) + 1

            def is_server_ready(self):
                return True

            def is_model_ready(self, model):
                return True

            def get_server_metadata(self):
                return {"version": "test"}

            def get_model_metadata(self, model):
                return {"name": model, "versions": ["1"]}

            def get_model_config(self, model):
                return {"name": model}

            def infer(self, model, *, inputs, outputs):
                observed.update(model=model, outputs=outputs)
                return types.SimpleNamespace(as_numpy=lambda name: np.ones((1, 512), np.float32))

            def close(self):
                observed["closed"] = True

        http = types.ModuleType("tritonclient.http")
        http.InferInput = Input
        http.InferenceServerClient = Client
        http.InferRequestedOutput = lambda name, binary_data: (name, binary_data)
        package = types.ModuleType("tritonclient")
        package.http = http
        with patch.dict(sys.modules, {"tritonclient": package, "tritonclient.http": http}):
            runner = module.HttpInference({"localhost:8000": ["clip_ensemble_trt", "benchmark_clip_image_cpu_fused"]})
            payload = np.array([[b"a\x00b\x00"]], dtype=object)
            result = runner("localhost:8000", "clip_ensemble_trt", payload, "unnorm_image_features")
            runner.close()
        self.assertEqual(observed["client_count"], 1)
        self.assertEqual((observed["name"], observed["shape"], observed["datatype"]),
                         ("IMAGE_BYTES", (1, 1), "BYTES"))
        self.assertTrue(observed["binary_data"])
        self.assertEqual(observed["data"][0, 0], b"a\x00b\x00")
        self.assertEqual(observed["outputs"], [("unnorm_image_features", True)])
        self.assertEqual(result.shape, (1, 512))

    def test_run_rotates_order_saves_failed_parity_and_suppresses_speedup(self):
        module = load_benchmark()
        calls = []

        class FakeInference:
            metadata = {"same:8000": {"version": "test"}}
            model_metadata = {}
            model_configurations = {}

            def __call__(self, url, model, payload, output):
                calls.append((model, payload.shape[0]))
                factor = 1.01 if model.endswith("gpu_resize") else 1.0
                return np.full((payload.shape[0], 512), factor, np.float32)

            def statistics(self, url):
                return stats(len(calls), len(calls) * 1_000_000)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_dataset(root, 1)
            output = root / "report.json"
            code = module.main(["--dataset-root", str(root), "--baseline-url", "same:8000",
                                "--candidate-url", "same:8000", "--models", "clip_image",
                                "--warmup", "3", "--repeats", "10", "--output", str(output)],
                               infer=FakeInference())
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(code, 1)
        self.assertFalse(report["passed"])
        self.assertEqual(len(report["requests"]), 80)  # B1+B4, ten repeats, four variants
        self.assertEqual(len(calls), 104)  # includes three warmups for each group
        for batch_size in (1, 4):
            requests = [r for r in report["requests"] if r["batch_size"] == batch_size]
            starts = [next(r["variant"] for r in requests if r["repeat"] == i and r["order_index"] == 0)
                      for i in range(4)]
            self.assertEqual(len(set(starts)), 4)
            bad = next(s for s in report["summary"] if s["batch_size"] == batch_size and s["variant"] == "gpu_resize")
            self.assertIsNone(bad["baseline_speedup_p50"])
            self.assertFalse(bad["performance_claim_valid"])
            good = next(s for s in report["summary"] if s["batch_size"] == batch_size and s["variant"] == "cpu_fused")
            self.assertTrue(good["passed"])
        self.assertEqual(report["server_processing"]["same:8000"][0]["execution_count"], 80)
        self.assertNotIn("pixels", report)


if __name__ == "__main__":
    unittest.main()
