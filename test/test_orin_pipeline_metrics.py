"""CPU-only checks for fused-pipeline counter meaning and metric lifetime."""

import gc
import types
import unittest
import weakref

from scripts.orin_pipeline_metrics import DIRECTIONS, STAGES, NullMetrics, PipelineMetrics


class PipelineMetricsTests(unittest.TestCase):
    def make_metrics(self):
        events, families, counters = [], [], []

        class Counter:
            def __init__(self, labels):
                self.labels = labels
                self.value = 0
                counters.append(weakref.ref(self))

            def increment(self, value):
                self.value += value

            def __del__(self):
                events.append("counter")

        class Family:
            COUNTER = "counter"

            def __init__(self, name, description, kind):
                self.name, self.description, self.kind = name, description, kind
                families.append(weakref.ref(self))

            def Metric(self, labels):
                return Counter(labels)

            def __del__(self):
                events.append("family")

        metrics = PipelineMetrics(types.SimpleNamespace(MetricFamily=Family), {
            "model_name": "face_qa_ensemble", "model_version": "1",
            "model_instance_name": "face_qa_ensemble_0_0",
        }, "qa")
        return metrics, events, families, counters

    def test_fixed_cardinality_and_no_request_identifying_labels(self):
        metrics, _, families, counters = self.make_metrics()
        self.assertEqual(len(families), 8)
        self.assertEqual(len(counters), 27)
        expected_base = {"model": "face_qa_ensemble", "version": "1",
                         "instance": "face_qa_ensemble_0_0", "mode": "qa"}
        for ref in counters:
            labels = ref().labels
            self.assertEqual({key: labels[key] for key in expected_base}, expected_base)
            self.assertLessEqual(set(labels) - set(expected_base), {"batch_size", "stage", "kind", "direction"})
            self.assertLessEqual(len(labels), 5)
        for ref in families:
            self.assertEqual(ref().kind, "counter")
            self.assertTrue(ref().name.startswith("orin_pipeline_"))
        metrics.close()

    def test_counter_values_distinguish_requests_batches_and_cross_request_groups(self):
        metrics, _, _, _ = self.make_metrics()
        metrics.requests(7)
        metrics.batch(4, 2)
        metrics.batch(4, 1)
        metrics.batch(1, 1)
        metrics.stage("warp", 0.125)
        metrics.stage("warp", 0.25)
        metrics.copy("input_h2d", 1024)
        metrics.copy("output_d2h", 40)
        metrics.error("invalid", 2)
        metrics.error("internal")
        metrics.error("cancelled")
        self.assertEqual(metrics.metrics["requests"].value, 7)
        self.assertEqual(metrics.metrics["cross_groups"].value, 1)
        self.assertEqual(metrics.metrics[("batch", 4)].value, 2)
        self.assertEqual(metrics.metrics[("group_requests", 4)].value, 3)
        self.assertEqual(metrics.metrics[("batch", 1)].value, 1)
        self.assertEqual(metrics.metrics[("seconds", "warp")].value, 0.375)
        self.assertEqual(metrics.metrics[("calls", "warp")].value, 2)
        self.assertEqual(metrics.metrics[("copy", "input_h2d")].value, 1024)
        self.assertEqual(metrics.metrics[("copy", "output_d2h")].value, 40)
        self.assertEqual(metrics.metrics[("errors", "invalid")].value, 2)
        self.assertEqual(metrics.metrics[("errors", "internal")].value, 1)
        self.assertEqual(metrics.metrics[("errors", "cancelled")].value, 1)
        metrics.close()

    def test_close_releases_metrics_before_families_and_is_idempotent(self):
        metrics, events, families, counters = self.make_metrics()
        metrics.close()
        gc.collect()
        self.assertTrue(all(ref() is None for ref in counters + families))
        self.assertEqual(events, ["counter"] * 27 + ["family"] * 8)
        metrics.close()
        self.assertEqual(len(events), 35)

    def test_null_metrics_supports_every_public_operation(self):
        metrics = NullMetrics()
        metrics.requests(3)
        metrics.batch(4, 3)
        for stage in STAGES:
            metrics.stage(stage, 0.1)
        for direction in DIRECTIONS:
            metrics.copy(direction, 128)
        for kind in ("invalid", "internal", "cancelled"):
            metrics.error(kind, 2)
        metrics.close()
        metrics.close()


if __name__ == "__main__":
    unittest.main()
