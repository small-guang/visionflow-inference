"""Cross-RPC scheduler contracts without CUDA, a server or input datasets."""
import gc
import types
import unittest
import weakref
from unittest import mock

import numpy as np

from test.test_yolo_orin_pipeline import Array, CP, Engine, TRT, load_pipeline


class Tensor:
    def __init__(self, name, value, cpu=True):
        self.name, self.value, self.cpu = name, value, cpu

    def is_cpu(self): return self.cpu
    def as_numpy(self): return self.value


class Request(dict):
    def __init__(self, images=None, *, cancelled=False):
        super().__init__({} if images is None else {"RAW_IMAGE": Tensor("RAW_IMAGE", images)})
        self.cancelled = cancelled

    def is_cancelled(self): return self.cancelled


class TritonError(Exception):
    CANCELLED = "cancelled"
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


class Response:
    def __init__(self, output_tensors=None, error=None):
        self.output_tensors, self.error = output_tensors or [], error


class RecordingMetrics:
    def __init__(self): self.calls = []
    def trt_call(self, batch): self.calls.append(("trt", batch))
    def trt_failure(self): self.calls.append(("trt_failure",))
    def group_failure(self): self.calls.append(("group_failure",))
    def copy(self, direction, size, seconds=0): self.calls.append((direction, size))
    def close(self): self.calls.append(("close",))


def image(batch=1, height=16, width=32, marker=0):
    return np.full((batch, height, width, 3), marker, np.uint8)


def pose_outputs(images):
    batch = len(images)
    markers = images[:, 0, 0, 0]
    detections = np.zeros((batch, 100, 6), np.float32)
    detections[:, :, 0] = markers[:, None]
    return detections, np.zeros((batch, 100, 17, 3), np.float32), np.ones((batch, 1), np.int32)


class BatchPlannerTests(unittest.TestCase):
    def setUp(self): self.api = load_pipeline().batching

    def items(self, batches):
        return [self.api.RequestImage(index, image(batch, marker=index)) for index, batch in enumerate(batches)]

    def test_batch_patterns_preserve_rpc_boundaries_and_never_exceed_four(self):
        for batches, expected in [([1, 1, 1, 1], [4]), ([1, 3], [4]), ([2, 2], [4]),
                                  ([4], [4]), ([3, 2, 1], [3, 3]), ([4, 1, 4], [4, 1, 4])]:
            with self.subTest(batches=batches):
                groups = self.api.plan_groups(self.items(batches))
                self.assertEqual([sum(item.batch_size for item in group) for group in groups], expected)
                self.assertEqual([item.index for group in groups for item in group], list(range(len(batches))))

    def test_shapes_group_independently_but_responses_keep_original_indices(self):
        items = [self.api.RequestImage(0, image()), self.api.RequestImage(1, image(width=16)),
                 self.api.RequestImage(2, image(marker=2))]
        groups = self.api.plan_groups(items)
        self.assertEqual([[item.index for item in group] for group in groups], [[0, 2], [1]])
        packed, copied = self.api.pack_group(groups[0])
        self.assertEqual(copied, packed.nbytes)
        split = list(self.api.split_outputs(pose_outputs(packed), groups[0]))
        self.assertEqual([index for index, _, _ in split], [0, 2])
        self.assertEqual([values[0][0, 0, 0] for _, values, _ in split], [0, 2])
        self.assertTrue(all(value.flags.owndata and value.flags.c_contiguous
                            for _, values, _ in split for value in values))

    def test_validation_rejects_budget_and_aspect_failures_without_allocating_images(self):
        pixel = np.zeros(1, np.uint8)
        for shape in ((0, 2, 2, 3), (5, 2, 2, 3), (1, 0, 1, 3), (1, 8193, 1, 3),
                      (1, 4096, 4096, 3), (1, 1, 8192, 3), (1, 2, 2, 4)):
            array = np.lib.stride_tricks.as_strided(pixel, shape=shape, strides=(0,) * 4)
            with self.subTest(shape=shape), self.assertRaises(self.api.InputValidationError):
                self.api.validate_images(array)
        with self.assertRaises(self.api.InputValidationError): self.api.validate_images(image().astype(np.float32))
        with self.assertRaises(self.api.InputValidationError): self.api.validate_images(None)

    def test_maximum_legal_single_rpc_uses_no_pack_allocation(self):
        array = np.broadcast_to(np.zeros(1, np.uint8), (4, 2048, 4096, 3))
        self.api.validate_images(array)
        self.assertEqual(array.nbytes, self.api.MAX_PACK_BYTES)
        with mock.patch.object(self.api.np, "concatenate", side_effect=AssertionError("unnecessary pack")):
            packed, copied = self.api.pack_group([self.api.RequestImage(0, array)])
        self.assertIs(packed, array)
        self.assertEqual(copied, 0)

    def test_pack_budget_and_shape_are_defended_at_execution_boundary(self):
        for items in (self.items([4, 1]), [self.api.RequestImage(0, image()),
                                         self.api.RequestImage(1, image(width=16))]):
            with self.assertRaises(ValueError): self.api.pack_group(items)

    def test_segmentation_restores_each_rpc_canonical_slots_and_exact_mask_bytes(self):
        group = self.items([1, 2, 1])
        counts = np.array([[0], [2], [1], [3]], np.int32)
        detections = np.zeros((4, 3, 6), np.float32)
        masks = np.zeros((4, 3, 160, 20), np.uint8)
        for batch, count in enumerate(counts[:, 0]):
            detections[batch, :count] = batch + 1
            masks[batch, :count] = batch + 10
        values = (detections, masks, counts)
        results = list(self.api.split_outputs(values, group, segmentation=True))
        self.assertEqual([part[0].shape for _, part, _ in results], [(1, 1, 6), (2, 2, 6), (1, 3, 6)])
        self.assertFalse(results[0][1][0].any())
        self.assertFalse(results[0][1][1].any())
        np.testing.assert_array_equal(results[1][1][1], masks[1:3, :2])
        self.assertFalse(results[1][1][1][1, 1].any())
        values[1][:] = 255
        self.assertEqual(results[1][1][1][0, 0, 0, 0], 11)

    def test_single_rpc_outputs_remain_zero_copy(self):
        group = self.items([4])
        values = pose_outputs(group[0].images)
        _, outputs, copied = next(self.api.split_outputs(values, group))
        self.assertIs(outputs, values)
        self.assertEqual(copied, 0)


class ExecuteBatchTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = load_pipeline()
        self.model = self.pipeline.TritonPythonModel()
        self.model.family = "yolo26m_pose"
        self.model.output_names = self.pipeline.FAMILIES[self.model.family][0]
        self.model.input_errors = (self.pipeline.InputValidationError,)
        self.model.pb = types.SimpleNamespace(get_input_tensor_by_name=lambda req, name: req.get(name),
            Tensor=Tensor, TritonError=TritonError, InferenceResponse=Response)
        self.model.metrics = RecordingMetrics()
        self.executed = []
        def infer(images):
            self.executed.append(images.copy())
            return pose_outputs(images)
        self.model._infer = infer

    def test_four_rpcs_share_one_infer_with_original_order(self):
        responses = self.model.execute([Request(image(marker=index)) for index in range(4)])
        self.assertEqual(len(self.executed), 1)
        self.assertEqual(self.executed[0].shape[0], 4)
        self.assertEqual([response.output_tensors[0].value[0, 0, 0] for response in responses], [0, 1, 2, 3])

    def test_invalid_or_cancelled_input_does_not_poison_valid_group(self):
        cancelled = Request(image(), cancelled=True)
        responses = self.model.execute([Request(np.ones((1, 2, 2, 3), np.float32)),
                                        Request(image(marker=1)), cancelled, Request(image(marker=2))])
        self.assertIn("INVALID_INPUT", str(responses[0].error))
        self.assertEqual(responses[2].error.code, TritonError.CANCELLED)
        self.assertEqual(self.executed[0].shape[0], 2)
        self.assertIsNone(responses[1].error)
        self.assertIsNone(responses[3].error)

    def test_cancellation_between_validation_and_pack_excludes_that_rpc(self):
        request = Request(image())
        calls = iter((False, True))
        request.is_cancelled = lambda: next(calls)
        responses = self.model.execute([request, Request(image(marker=1))])
        self.assertEqual(responses[0].error.code, TritonError.CANCELLED)
        self.assertEqual(len(self.executed[0]), 1)

    def test_cancellation_after_launch_preserves_other_responses(self):
        requests = [Request(image()), Request(image(marker=1))]
        def infer(images):
            requests[0].cancelled = True
            return pose_outputs(images)
        self.model._infer = infer
        responses = self.model.execute(requests)
        self.assertEqual(responses[0].error.code, TritonError.CANCELLED)
        self.assertIsNone(responses[1].error)
        self.assertEqual(responses[1].output_tensors[0].value[0, 0, 0], 1)

    def test_internal_failure_affects_only_its_group_and_hides_driver_details(self):
        def infer(images):
            if images.shape[2] == 32: raise RuntimeError("private driver details")
            return pose_outputs(images)
        self.model._infer = infer
        with self.assertLogs(self.pipeline.logger, "ERROR"):
            responses = self.model.execute([Request(image()), Request(image(width=16)), Request(image())])
        self.assertIsNone(responses[1].error)
        for index in (0, 2):
            self.assertIn("INTERNAL_ERROR", str(responses[index].error))
            self.assertNotIn("private", str(responses[index].error))
        self.assertEqual(self.model.metrics.calls.count(("group_failure",)), 1)

    def test_face_retains_independent_rpcs_and_two_outputs(self):
        self.model.family = "yolov8n_face"
        self.model.output_names = self.pipeline.FAMILIES[self.model.family][0]
        batches = []
        def infer(images):
            batches.append(len(images))
            return np.zeros((len(images), 100, 6), np.float32), np.zeros((len(images), 1), np.int32)
        self.model._infer = infer
        responses = self.model.execute([Request(image(2)), Request(image())])
        self.assertEqual(batches, [2, 1])
        self.assertTrue(all(response.error is None and len(response.output_tensors) == 2 for response in responses))

    def test_previous_pack_is_released_before_next_group_allocation(self):
        original = self.pipeline.batching.pack_group
        refs = []
        def pack(group):
            gc.collect()
            self.assertTrue(all(reference() is None for reference in refs))
            packed, size = original(group)
            refs.append(weakref.ref(packed))
            return packed, size
        with mock.patch.object(self.pipeline.batching, "pack_group", side_effect=pack):
            responses = self.model.execute([Request(image(marker=i)) for i in range(8)])
        self.assertTrue(all(response.error is None for response in responses))
        self.assertEqual(len(refs), 2)


class NativeMetricsTests(unittest.TestCase):
    def test_trt_counter_tracks_only_native_attempts_including_failed_enqueue(self):
        pipeline = load_pipeline()
        metrics, engine = RecordingMetrics(), Engine()
        runner = pipeline.TensorRtRunner(engine, object(), CP, TRT, "yolo26m_pose", metrics)
        engine.context.invalid_output = True
        with self.assertRaises(RuntimeError): runner.infer(Array((1, 3, 640, 640)), 1)
        self.assertEqual(metrics.calls, [])
        engine.context.invalid_output = False
        for batch in range(1, 5): runner.infer(Array((batch, 3, 640, 640)), 1)
        engine.context.enqueue_result = False
        with self.assertRaises(RuntimeError): runner.infer(Array((2, 3, 640, 640)), 1)
        self.assertEqual(metrics.calls, [("trt", 1), ("trt", 2), ("trt", 3), ("trt", 4),
                                         ("trt", 2), ("trt_failure",)])

    def test_metrics_cardinality_labels_and_lifetime(self):
        pipeline = load_pipeline()
        events = []
        class Metric:
            def __init__(self, labels): self.labels, self.value = labels, 0
            def increment(self, amount): self.value += amount
            def __del__(self): events.append("metric")
        class Family:
            COUNTER = "counter"
            def __init__(self, **kwargs): self.name = kwargs["name"]
            def Metric(self, labels): return Metric(labels)
            def __del__(self): events.append("family")
        metrics = pipeline.metrics_api.PipelineMetrics(types.SimpleNamespace(MetricFamily=Family),
            {"model_name": "pose", "model_version": "1", "model_instance_name": "pose_0"})
        self.assertEqual(len(metrics.families), 5)
        self.assertEqual(len(metrics.metrics), 12)
        metrics.trt_call(4)
        self.assertEqual(metrics.metrics[("trt", 4)].value, 1)
        self.assertEqual(metrics.metrics[("trt", 4)].labels,
                         {"model": "pose", "version": "1", "instance": "pose_0", "batch_size": "4"})
        metrics.copy("host_pack", 1024, .1)
        self.assertEqual(metrics.metrics[("bytes", "host_pack")].value, 1024)
        metrics.close()
        self.assertEqual(events[:12], ["metric"] * 12)
        self.assertEqual(events[12:], ["family"] * 5)


class StreamLifetimeTests(unittest.TestCase):
    def test_synchronizes_on_each_gpu_stage_failure_and_keeps_outputs_alive(self):
        for failure in ("pre", "trt", "post", "d2h", None):
            with self.subTest(failure=failure):
                pipeline = load_pipeline()
                model = pipeline.TritonPythonModel()
                model.family, model.meta_width = "yolo26m_pose", 7
                model.pre_model = types.SimpleNamespace(imgsz=640, padding_value=114)
                live, events = [], []
                class Context:
                    ptr = 7
                    def __enter__(self): return self
                    def __exit__(self, *args): pass
                    def synchronize(self):
                        self_check = all(reference() is not None for reference in live)
                        if not self_check: raise AssertionError("released device output before synchronization")
                        events.append("sync")
                def stage(name, value):
                    if failure == name: raise RuntimeError(name)
                    return value
                def preprocess(*args, **kwargs):
                    stage("pre", None)
                    result = np.zeros((1, 3, 640, 640), np.float32)
                    live.append(weakref.ref(result))
                    return result, np.zeros((1, 7))
                def infer(*args):
                    stage("trt", None)
                    result = np.zeros((1, 300, 57), np.float32)
                    live.append(weakref.ref(result))
                    return {"output0": result}
                def post(*args):
                    stage("post", None)
                    result = pose_outputs(image())
                    live.extend(weakref.ref(value) for value in result)
                    return result
                model.device, model.stream = Context(), Context()
                model.pre = types.SimpleNamespace(preprocess=preprocess)
                model.runner = types.SimpleNamespace(infer=infer)
                model.post = types.SimpleNamespace(process=post)
                model.cp = types.SimpleNamespace(asnumpy=lambda value: stage("d2h", value.copy()))
                if failure:
                    with self.assertRaisesRegex(RuntimeError, failure): model._infer(image())
                else:
                    model._infer(image())
                self.assertEqual(events, ["sync"])


if __name__ == "__main__": unittest.main()
