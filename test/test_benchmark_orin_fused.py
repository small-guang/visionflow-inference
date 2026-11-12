"""融合评测工具的口径与门禁单测；假客户端只用于单测，不作为GPU性能证据。"""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "tools/benchmark_orin_fused.py"
SPEC = importlib.util.spec_from_file_location("fused_benchmark_tests_target", PATH)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class FusedBenchmarkAccountingTests(unittest.TestCase):
    def clip_bank(self, serial, variants):
        return {"case_id": "clip-test", "serial": serial,
                "variants": [{"total_batch_size": index + len(serial), "request_start_slot": 0,
                              "output": value} for index, value in enumerate(variants)]}

    def test_clip_native_bank_keeps_original_failure_and_all_matching_variants(self):
        serial = np.ones((1, 512), np.float32)
        native = serial * np.float32(1.007)
        case = {"id": "clip-test", "alias": "clip"}
        bank = self.clip_bank(serial, [serial, native, native.copy(), native.copy()])
        self.assertFalse(benchmark.compare_case(case, serial, native)["passed"])
        result = benchmark.compare_case(case, bank, native)
        self.assertTrue(result["passed"])
        self.assertFalse(result["serial_comparison"]["passed"])
        self.assertEqual([row["total_batch_size"] for row in result["matched_native_variants"]], [2, 3, 4])

    def test_clip_native_bank_cannot_mix_rows_from_different_variants(self):
        first = np.asarray([[1, 0], [0, 1]], np.float32)
        second = first + np.float32(0.02)
        mixed = np.stack([first[0], second[1]])
        case = {"id": "clip-test", "alias": "clip"}
        result = benchmark.compare_case(case, self.clip_bank(first, [first, second]), mixed)
        self.assertFalse(result["passed"])
        self.assertEqual(result["matched_native_variants"], [])

    def test_clip_native_bank_rejects_amplitude_order_and_contract_corruption(self):
        reference = np.asarray([[1, 0], [0, 1]], np.float32)
        case = {"id": "clip-test", "alias": "clip"}
        bank = self.clip_bank(reference, [reference])
        bad_values = [reference * np.float32(1.002), reference[::-1], np.zeros_like(reference),
                      np.full_like(reference, np.nan), reference.astype(np.float64), reference[:1]]
        for value in bad_values:
            with self.subTest(value=value):
                self.assertFalse(benchmark.compare_case(case, bank, value)["passed"])
        self.assertAlmostEqual(benchmark.compare_case(case, reference, bad_values[0])["minimum_cosine"], 1)
        for changed in (case | {"alias": "sbs"}, case | {"id": "different-sample"}):
            with self.assertRaises(ValueError):
                benchmark.compare_case(changed, bank, reference)

    def test_explicit_paired_gate_cannot_be_rescued_by_native_variants(self):
        sample = {"sha256": "a", "encoded": b"jpeg"}
        case = benchmark.make_case("clip", [sample], identifier="clip-test")
        serial = np.ones((1, 512), np.float32)
        changed = serial * np.float32(1.007)
        args = SimpleNamespace(stress_concurrency=4, stress_requests=8, stress_rounds=2, correctness_repeats=1)

        class ForbiddenBank:
            def wrap(self, case, reference):
                raise AssertionError("explicit paired failure must stop before alternative references")

        with patch.object(benchmark, "infer_case", side_effect=[serial, changed]):
            report, _ = benchmark.correctness(args, {"baseline": object(), "candidate": object()},
                                               {"clip": [sample]}, [case], clip_references=ForbiddenBank())
        self.assertFalse(report["passed"])
        self.assertFalse(report["batch_checks"][0]["passed"])
        self.assertEqual(report["stress"], [])

    def test_native_reference_builder_preserves_complete_request_slots_and_reuses_groups(self):
        calls = []

        class Native:
            def infer(self, model, inputs, outputs):
                calls.append(model)
                if model == "clip_image_preprocess":
                    value = inputs["IMAGE_BYTES"][0, 0][0]
                    return {"image": np.full((1, 3, 224, 224), value, np.float32)}
                values = inputs["image"][:, 0, 0, 0]
                return {"unnorm_image_features": np.repeat(values[:, None], 512, axis=1)}

        samples = [{"sha256": str(i), "encoded": bytes([i])} for i in (1, 2)]
        case = benchmark.make_case("clip", samples, identifier="clip-test")
        serial = np.repeat(np.asarray([[1], [2]], np.float32), 512, axis=1)
        audit = {}
        builder = benchmark.ClipNativeReferences(Native(), audit)
        bank = builder.wrap(case, serial)
        self.assertEqual([(v["total_batch_size"], v["request_start_slot"]) for v in bank["variants"]],
                         [(2, 0), (3, 0), (3, 1), (4, 0), (4, 1), (4, 2)])
        for variant in bank["variants"]:
            np.testing.assert_array_equal(variant["output"], serial)
        self.assertEqual(calls.count("clip_image_preprocess"), 2)
        self.assertEqual(calls.count("clip_image_trt"), 12)
        before = len(calls)
        builder.wrap(case, serial)
        self.assertEqual(len(calls), before)
        self.assertEqual(len(audit["clip-test"]["variants"]), 6)
        for alias, size in (("sbs", 2), ("clip", 4)):
            self.assertIs(builder.wrap(case | {"alias": alias, "batch_size": size}, serial), serial)

    def test_clip_evidence_requires_bound_real_diagnostic_and_unchanged_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "diagnostic.npz"
            raw.write_bytes(b"unit-test-only-array-provenance")
            path = root / "diagnostic.json"
            configs = {key: {"name": key} for key in ("baseline_public", "baseline_pre", "candidate_public", "baseline_native")}
            diagnostic = {"completed": True, "tolerances_unchanged": benchmark.V.DEFAULT_TOLERANCES,
                          "raw_arrays": {"path": str(raw), "sha256": benchmark.file_digest(raw)},
                          "model_configs": configs}
            path.write_text(json.dumps(diagnostic))
            summary = {"completed": True, "all_preprocessed_equal": True,
                       "tolerances_unchanged": benchmark.V.DEFAULT_TOLERANCES,
                       "source_json_sha256": benchmark.file_digest(path), "raw_npz_sha256": benchmark.file_digest(raw),
                       "native_B1_failures": [{"passed": False}],
                       "same_batch_comparisons": {key: {"comparisons": 40, "all_exact_equal": True, "all_passed": True}
                            for key in ("candidate_vs_native_same_batch", "baseline_vs_native_same_batch", "candidate_vs_baseline_same_batch")}}
            for key in ("neighbor_comparisons", "repeated_slot_comparisons", "slot_reordering_comparisons", "batch_2_3_4_comparisons"):
                summary[key] = [{"exact_equal": True, "passed": True}]
            summary_path = root / "diagnostic-summary.json"
            summary_path.write_text(json.dumps(summary))
            endpoints = {"baseline": SimpleNamespace(client=SimpleNamespace(get_model_config=lambda _: configs["baseline_native"]))}
            metadata = {"baseline": {"models": {"clip_ensemble_trt": configs["baseline_public"],
                                                 "clip_image_preprocess": configs["baseline_pre"]}},
                        "candidate": {"models": {"clip_ensemble_trt": configs["candidate_public"]}}}
            self.assertIn("native_config", benchmark.clip_batch_evidence(path, endpoints, metadata))
            summary["tolerances_unchanged"] = benchmark.V.DEFAULT_TOLERANCES | {"maximum_relative_l2": 0.1}
            summary_path.write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                benchmark.clip_batch_evidence(path, endpoints, metadata)

    def test_pipeline_metrics_accumulate_instances_and_preserve_host_timing(self):
        before = benchmark.parse_metrics('orin_pipeline_stage_seconds_total{model="clip",instance="0",stage="decode"} 1\n')
        after = benchmark.parse_metrics('''
orin_pipeline_stage_seconds_total{model="clip",instance="0",stage="decode"} 2
orin_pipeline_stage_seconds_total{model="clip",instance="1",stage="decode"} 0.5
orin_pipeline_stage_calls_total{model="clip",instance="0",stage="decode"} 4
orin_pipeline_trt_batches_total{model="clip",batch_size="4",instance="0"} 3
orin_pipeline_cross_request_groups_total{model="clip",instance="0"} 2
orin_pipeline_copy_bytes_total{model="clip",direction="output_d2h"} 24576
''')
        value = benchmark.summarize_metrics(benchmark.metrics_delta(before, after), "clip")
        self.assertEqual(value["stages_host_wall"]["decode"], {"seconds": 1.5, "calls": 4})
        self.assertEqual(value["trt_batches"], [{"batch_size": 4, "executions": 3}])
        self.assertEqual(value["counters"]["cross_request_groups_total"], 2)
        self.assertEqual(value["logical_copy_bytes"]["output_d2h"], 24576)
        self.assertIn("not additive", value["stage_scope"])

    def test_missing_metrics_do_not_become_zero_measurements(self):
        self.assertIsNone(benchmark.metrics_delta(None, {}))
        self.assertFalse(benchmark.summarize_metrics(None, "clip")["available"])
        self.assertFalse(benchmark.summarize_metrics([], "clip")["available"])

    def test_counter_reset_and_disappearance_invalidate_window(self):
        first = benchmark.parse_metrics('orin_pipeline_requests_total{model="clip"} 4')
        reset = benchmark.parse_metrics('orin_pipeline_requests_total{model="clip"} 1')
        for after in (reset, {}):
            with self.assertRaises(ValueError):
                benchmark.metrics_delta(first, after)

    def test_invalid_metric_values_and_duplicate_series_are_rejected(self):
        for payload in ('orin_pipeline_requests_total NaN', 'orin_pipeline_requests_total -1',
                        'orin_pipeline_requests_total{model="a",model="b"} 1',
                        'orin_pipeline_requests_total 1\norin_pipeline_requests_total 2'):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                benchmark.parse_metrics(payload)

    def test_gate_binds_models_data_identities_and_validated_concurrency(self):
        binding = {"identity": "accepted", "dataset": "sha", "model": {"backend": "python"}}
        gate = {"passed": True, "binding_sha256": benchmark.digest(binding),
                "correctness": {"passed": True, "stress_concurrency": 8}}
        benchmark.check_gate(gate, binding, [1, 8])
        for changed, concurrency in ((binding | {"dataset": "different"}, [1]), (binding, [16])):
            with self.assertRaises(ValueError):
                benchmark.check_gate(gate, changed, concurrency)
        with self.assertRaises(ValueError):
            benchmark.check_gate(gate | {"passed": False}, binding, [1])

    def test_runner_accounting_requires_real_execution_and_stage_evidence(self):
        pipeline = {"available": True, "trt_batches": [{"batch_size": 4, "executions": 2}],
                    "counters": {"requests_total": 8, "group_requests_total": 8, "errors_total": 0},
                    "stages_host_wall": {"trt_download": {"calls": 2, "seconds": 0.01}}}
        window = {"server": {"pipeline": pipeline}, "successful_requests": 8,
                  "successful_images_including_repeated_slots": 8}
        self.assertEqual(benchmark.pipeline_accounting_errors(window), [])
        window["successful_images_including_repeated_slots"] = 12
        self.assertTrue(benchmark.pipeline_accounting_errors(window))
        window["server"]["pipeline"] = {"available": False}
        self.assertIn("missing", benchmark.pipeline_accounting_errors(window)[0])

    def test_correctness_retains_failed_case_and_does_not_enter_stress(self):
        sample = {"sha256": "a", "encoded": b"jpeg", "image": np.zeros((2, 2, 3), np.uint8)}
        case = benchmark.make_case("clip", [sample], identifier="a")
        report = {}
        args = SimpleNamespace(stress_concurrency=4, stress_requests=8, stress_rounds=2, correctness_repeats=2)
        with patch.object(benchmark, "infer_case", side_effect=[np.ones((1, 512), np.float32), RuntimeError("failed")]):
            result, references = benchmark.correctness(args, {"baseline": object(), "candidate": object()},
                                                       {"clip": [sample]}, [case], report=report)
        self.assertIs(result, report)
        self.assertFalse(result["passed"])
        self.assertEqual(result["stress"], [])
        self.assertEqual(result["batch_checks"][0]["side"], "candidate")
        self.assertEqual(result["batch_checks"][0]["error_class"], "RuntimeError")
        self.assertIn("a", references)

    def test_encoded_batches_preserve_tail_bytes_and_explicit_repeats(self):
        samples = [{"sha256": str(i), "encoded": b"bytes\0", "image": np.zeros((4, 6, 3), np.uint8),
                    "height": 4, "width": 6} for i in range(3)]
        case = benchmark.make_batches({"clip": samples}, [4])[0]
        self.assertEqual(case["repeated_slots"], [3])
        self.assertEqual(case["inputs"]["IMAGE_BYTES"].dtype, object)
        self.assertEqual(case["inputs"]["IMAGE_BYTES"][0, 0], b"bytes\0")
        self.assertNotIn("inputs", benchmark.case_metadata(case))

    def test_mixed_face_stress_preserves_valid_neighbors(self):
        samples = [{"sha256": str(i), "image": np.full((4 + i, 6, 3), i, np.uint8),
                    "height": 4 + i, "width": 6, "bbox": np.ones(4, np.float32),
                    "landmark": np.ones((5, 2), np.float32)} for i in range(4)]
        with patch.object(benchmark.V, "cross_samples", side_effect=AssertionError("padding forbidden")):
            cases = benchmark.stress_cases("recognition", samples, "mixed_errors")
        self.assertTrue(np.isfinite(cases[0]["inputs"]["LANDMARK"]).all())
        self.assertTrue(np.isnan(cases[1]["inputs"]["LANDMARK"]).any())
        self.assertTrue(cases[1]["expected_error"])
        self.assertEqual(cases[2]["batch_size"], 3)
        for case, sample, size in zip([cases[0], *cases[2:]], samples, (1, 3, 2, 2)):
            self.assertEqual(case["inputs"]["RAW_IMAGE"].shape, (size, *sample["image"].shape))
            self.assertEqual(case["repeated_slots"], list(range(1, size)))
            for image in case["inputs"]["RAW_IMAGE"]:
                np.testing.assert_array_equal(image, sample["image"])
        self.assertTrue(all(np.isfinite(sample["landmark"]).all() for sample in samples))

    def test_b1_face_stress_keeps_each_original_shape_and_pixels(self):
        samples = [{"sha256": str(i), "image": np.full((4 + i, 6 + i, 3), i + 1, np.uint8),
                    "height": 4 + i, "width": 6 + i, "bbox": np.ones(4, np.float32),
                    "landmark": np.ones((5, 2), np.float32)} for i in range(4)]
        for alias in ("keypoint", "qa", "recognition"):
            with self.subTest(alias=alias), patch.object(benchmark.V, "cross_samples", side_effect=AssertionError("padding forbidden")):
                cases = benchmark.stress_cases(alias, samples, "b1_pressure")
            self.assertEqual(len(cases), 4)
            image_name = "RAW_FACE_IMAGE" if alias == "qa" else "RAW_IMAGE"
            for case, sample in zip(cases, samples):
                self.assertEqual(case["inputs"][image_name].shape, (1, *sample["image"].shape))
                np.testing.assert_array_equal(case["inputs"][image_name][0], sample["image"])
                self.assertEqual(case["repeated_slots"], [])

    def test_real_request_stress_accounting_keeps_expected_errors_separate(self):
        samples = [{"sha256": str(i), "encoded": bytes([i]), "image": np.zeros((2, 2, 3), np.uint8)}
                   for i in range(4)]
        cases = benchmark.stress_cases("clip", samples, "mixed_errors")
        references = {case["id"]: np.ones((case["batch_size"], 512), np.float32)
                      for case in cases if not case["expected_error"]}

        class Client:
            def __init__(self, url):
                pass

            def infer(self, model, inputs, outputs):
                if inputs["IMAGE_BYTES"][0, 0] == b"not-an-image":
                    raise RuntimeError("invalid")
                return {outputs[0]: np.ones((len(inputs["IMAGE_BYTES"]), 512), np.float32)}

            def close(self):
                pass

        def snapshot(endpoint, url, *, before):
            return {"cpu": None, "statistics": {"model_stats": []}, "metrics": None}

        with patch.object(benchmark.V, "Endpoint", Client), patch.object(benchmark, "capture", snapshot):
            result = benchmark.stress_check(SimpleNamespace(url="fake"), cases, references,
                                            requests=20, concurrency=4, rounds=1, metrics_url=None)
        self.assertTrue(result["passed"])
        self.assertEqual(result["expected_errors"], 4)
        self.assertEqual(sum("error_class" in row for row in result["results"]), 4)
        self.assertEqual([row["index"] for row in result["results"]], list(range(20)))
        self.assertIsNone(result["cross_request_batching"]["observed"])
        self.assertIn("missing", result["cross_request_batching"]["note"])
        self.assertEqual(len(result["cases"]), 5)

    def test_window_reuses_clients_and_only_captures_at_boundaries(self):
        sample = {"sha256": "a", "encoded": b"jpeg", "image": np.zeros((2, 2, 3), np.uint8)}
        case = benchmark.make_case("clip", [sample] * 4, identifier="a")
        expected = np.ones((4, 512), np.float32)
        clients, snapshots, calls = [], [], []

        class Client:
            def __init__(self, url):
                self.closed = False
                clients.append(self)

            def infer(self, model, inputs, outputs):
                calls.append(threading.get_ident())
                time.sleep(0.001)
                return {outputs[0]: expected.copy()}

            def close(self):
                self.closed = True

        def snapshot(endpoint, url, *, before):
            snapshots.append((before, len(calls), threading.get_ident()))
            return {"cpu": {(42, 1): 0 if before else 10}, "statistics": {"model_stats": []}, "metrics": {}}

        value = benchmark.run_window(SimpleNamespace(url="fake"), [case], {"a": expected},
                                     duration=0.02, concurrency=2, warmup=2, metrics_url=None,
                                     endpoint_factory=Client, capture_fn=snapshot)
        self.assertTrue(value["passed"])
        self.assertEqual(len(clients), 2)
        self.assertTrue(all(client.closed for client in clients))
        self.assertEqual([row[0] for row in snapshots], [True, False])
        self.assertEqual(snapshots[0][1], 4)
        self.assertTrue(all(row[2] == threading.get_ident() for row in snapshots))
        self.assertEqual(value["successful_images_including_repeated_slots"], value["successful_requests"] * 4)
        self.assertAlmostEqual(value["images_per_second"], value["successful_images_including_repeated_slots"] / value["wall_seconds_including_drain"])
        self.assertEqual(len(calls), value["successful_requests"] + 4)
        self.assertEqual(value["unique_samples_seen"], 1)

    def test_window_warmup_failure_is_not_a_success_or_deadlock(self):
        sample = {"sha256": "a", "encoded": b"jpeg", "image": np.zeros((2, 2, 3), np.uint8)}
        case = benchmark.make_case("clip", [sample], identifier="a")

        class Broken:
            def __init__(self, url):
                raise RuntimeError("unavailable")

        def snapshot(endpoint, url, *, before):
            return {"cpu": None, "statistics": {"model_stats": []}, "metrics": None}

        value = benchmark.run_window(SimpleNamespace(url="fake"), [case], {}, duration=0.01,
                                     concurrency=2, warmup=1, metrics_url=None,
                                     endpoint_factory=Broken, capture_fn=snapshot)
        self.assertFalse(value["passed"])
        self.assertEqual(value["successful_requests"], 0)
        self.assertEqual(len(value["warmup_errors"]), 2)
        self.assertIsNone(value["server_cpu_seconds_per_image"])


if __name__ == "__main__":
    unittest.main()
