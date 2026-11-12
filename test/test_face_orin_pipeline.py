"""CPU geometry/postprocessing with fake device ownership; not CUDA parity tests."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest import mock
import weakref

import cv2
import numpy as np

from scripts import orin_pipeline_metrics

with mock.patch.dict(sys.modules, {"orin_pipeline_metrics": orin_pipeline_metrics}):
    from scripts import face_orin_pipeline as production


ROOT = Path(__file__).resolve().parents[1]
HAS_SKIMAGE = importlib.util.find_spec("skimage") is not None
NAMES = {"keypoint": "face_keypoint_2d106_ensemble", "qa": "face_qa_ensemble",
         "recognition": "face_recognition_pipeline"}


class Tensor:
    def __init__(self, name, value, cpu=True):
        self.name, self.value, self.cpu = name, value, cpu

    def as_numpy(self):
        return self.value

    def is_cpu(self):
        return self.cpu


class Request(dict):
    cancelled = False

    def is_cancelled(self):
        return self.cancelled


class Response:
    def __init__(self, output_tensors=None, error=None):
        self.output_tensors = output_tensors or []
        self.error = error


class TritonError(RuntimeError):
    CANCELLED = "CANCELLED"

    def __init__(self, message, code="INTERNAL"):
        super().__init__(message)
        self.code = code


PB = types.SimpleNamespace(Tensor=Tensor, InferenceResponse=Response, TritonError=TritonError,
                          get_input_tensor_by_name=lambda request, name: request.get(name))


class Context:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class Stream(Context):
    ptr = 73

    def __init__(self, state):
        self.state = state
        self.synchronizations = 0

    def synchronize(self):
        self.synchronizations += 1
        for ref in self.state.pending_refs:
            if ref() is None:
                self.state.lifetime_failures += 1
                raise AssertionError("buffer released before stream synchronization")
        self.state.pending_refs.clear()


class DeviceTensor:
    def __init__(self, state, markers):
        self.state, self.markers = state, markers

    def get(self, **_):
        raise AssertionError("preprocessed image must never be downloaded")


class FakeCupy:
    def __init__(self, state):
        self.state = state

    def concatenate(self, tensors, axis=0):
        if axis != 0 or not all(isinstance(tensor, DeviceTensor) for tensor in tensors):
            raise AssertionError("mixed shapes must concatenate preprocessed device batches")
        markers = np.concatenate([tensor.markers for tensor in tensors])
        self.state.concatenations.append(list(markers))
        if self.state.fail == "concat":
            self.state.fail = None
            raise RuntimeError("private concatenate failure details")
        output = DeviceTensor(self.state, markers)
        self.state.pending_refs.append(weakref.ref(output))
        return output


class DeviceOutput:
    def __init__(self, state, values):
        self.state, self.values = state, values

    def get(self, *, stream, blocking):
        if stream is not self.state.stream or blocking is not True:
            raise AssertionError("final output download must use the execution stream and block")
        self.state.downloads += 1
        if self.state.fail == "download":
            self.state.fail = None
            raise RuntimeError("private download failure details")
        host = self.values.copy()
        self.state.pending_refs.append(weakref.ref(host))
        return host


class FakeGpu:
    def __init__(self, state):
        self.state = state

    def warp(self, images, matrices):
        self.state.warps.append((images.copy(), matrices.copy()))
        self.state.pending_refs.extend([weakref.ref(images), weakref.ref(matrices)])
        if self.state.fail == "warp" or (self.state.fail == "warp_second" and len(self.state.warps) == 2):
            self.state.fail = None
            raise RuntimeError("private warp failure details")
        output = DeviceTensor(self.state, images[:, 0, 0, 0].copy())
        self.state.pending_refs.append(weakref.ref(output))
        return output


class FakeRunner:
    def __init__(self, state, spec):
        self.state, self.spec = state, spec

    def infer(self, tensor, stream_ptr):
        if stream_ptr != self.state.stream.ptr:
            raise AssertionError("wrong execution stream")
        self.state.batches.append(list(tensor.markers))
        if self.state.fail == "infer":
            self.state.fail = None
            raise RuntimeError("private engine failure details")
        width = self.spec["raw_shape"][0]
        values = np.repeat(tensor.markers.astype(np.float32)[:, None] / np.float32(100), width, axis=1)
        if self.state.nonfinite_marker is not None:
            values[tensor.markers == self.state.nonfinite_marker, 0] = np.nan
        output = DeviceOutput(self.state, values)
        self.state.pending_refs.append(weakref.ref(output))
        return {self.spec["output"]: output}


def config_args(mode):
    # Public contract is stated independently from the production SPECS table.
    definitions = {
        "keypoint": ("RAW_IMAGE", "BBOX", [4], "LANDMARKS_106", [106, 2], "1"),
        "qa": ("RAW_FACE_IMAGE", "LANDMARK", [5, 2], "quality_score", [1], "2"),
        "recognition": ("RAW_IMAGE", "LANDMARK", [5, 2], "EMBEDDING", [512], "1"),
    }
    image, auxiliary, aux_shape, output, output_shape, version = definitions[mode]
    return {"model_name": NAMES[mode], "model_instance_kind": "GPU", "model_version": "1",
            "model_repository": str(ROOT / "model_repository"),
            "model_config": {"max_batch_size": 4,
                "parameters": {"pipeline_backend": {"string_value": "orin_gpu"},
                               "api_contract_version": {"string_value": version}},
                "input": [{"name": image, "data_type": "TYPE_UINT8", "dims": [-1, -1, 3]},
                          {"name": auxiliary, "data_type": "TYPE_FP32", "dims": aux_shape}],
                "output": [{"name": output, "data_type": "TYPE_FP32", "dims": output_shape}]}}


class FacePipelineTests(unittest.TestCase):
    def make_pipeline(self, mode="qa"):
        if mode != "qa" and not HAS_SKIMAGE:
            self.skipTest("real keypoint/recognition geometry requires scikit-image")
        spec = production.SPECS[NAMES[mode]]
        runtime = production.FaceOrinPipeline()
        runtime.spec, runtime.mode, runtime.name, runtime.pb = spec, mode, NAMES[mode], PB
        with mock.patch.dict(sys.modules, {"triton_python_backend_utils": PB}):
            runtime.pre = production.load_helper(ROOT / "model_repository", spec["pre"], "1")
            runtime.post = production.load_helper(ROOT / "model_repository", spec["post"], "1") if spec["post"] else None
        runtime.validate_batch = runtime.pre._validate_batch if mode == "qa" else runtime.pre._validate_batch_inputs
        state = types.SimpleNamespace(warps=[], batches=[], downloads=0, pending_refs=[], concatenations=[],
                                      lifetime_failures=0,
                                      fail=None, nonfinite_marker=None)
        runtime.device, runtime.stream = Context(), Stream(state)
        state.stream = runtime.stream
        runtime.gpu, runtime.runner = FakeGpu(state), FakeRunner(state, spec)
        runtime.cp = FakeCupy(state)
        runtime.metrics = mock.Mock(spec=orin_pipeline_metrics.NullMetrics)
        return runtime, state

    def request(self, runtime, markers, shape=(112, 112)):
        images = np.stack([np.full((*shape, 3), value, np.uint8) for value in markers])
        if runtime.mode == "keypoint":
            auxiliary = np.repeat(np.array([[10, 12, 95, 100]], dtype=np.float32), len(markers), axis=0)
        else:
            reference = (runtime.pre.FACE_QUALITY_REFERENCE_LANDMARKS if runtime.mode == "qa"
                         else runtime.pre.arcface_dst)
            auxiliary = np.repeat(reference[None], len(markers), axis=0)
        return Request({runtime.spec["image"]: Tensor("", images), runtime.spec["aux"]: Tensor("", auxiliary)})

    def test_qa_b1_and_b3_make_one_trt_call_and_keep_real_geometry_and_postprocessing(self):
        runtime, state = self.make_pipeline()
        requests = [self.request(runtime, [10]), self.request(runtime, [20, 30, 40])]
        expected_matrices = np.stack([runtime.pre.prepare_affine(
            request[runtime.spec["image"]].value[index], request[runtime.spec["aux"]].value[index])
            for request in requests for index in range(len(request[runtime.spec["image"]].value))])
        with mock.patch.object(runtime.pre, "prepare_affine", wraps=runtime.pre.prepare_affine) as geometry, \
                mock.patch.object(runtime.post, "_validate_quality", wraps=runtime.post._validate_quality) as post, \
                mock.patch.object(cv2, "warpAffine", side_effect=AssertionError("CPU pixels forbidden")):
            responses = runtime.execute(requests)
        self.assertEqual(geometry.call_count, 4)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(state.batches, [[10, 20, 30, 40]])
        self.assertEqual(state.concatenations, [])
        self.assertEqual(state.downloads, 1)
        self.assertEqual(runtime.stream.synchronizations, 1)
        self.assertEqual(state.warps[0][1].dtype, np.float64)
        np.testing.assert_array_equal(state.warps[0][1], expected_matrices)
        for response, markers in zip(responses, ([10], [20, 30, 40])):
            self.assertIsNone(response.error)
            raw = np.repeat(np.asarray(markers, np.float32)[:, None] / np.float32(100), 10, axis=1)
            reference = runtime.post.TritonPythonModel().execute([{"quality": Tensor("quality", raw)}])[0]
            np.testing.assert_array_equal(response.output_tensors[0].value, reference.output_tensors[0].value)
            self.assertEqual(response.output_tensors[0].name, "quality_score")
        runtime.metrics.batch.assert_called_once_with(4, 2)
        runtime.metrics.requests.assert_called_once_with(2)
        runtime.metrics.copy.assert_has_calls([
            mock.call("input_h2d", 4 * 112 * 112 * 3), mock.call("output_d2h", 4 * 10 * 4)])

    def test_keypoint_keeps_fp64_geometry_and_calls_existing_inverse_transform(self):
        runtime, state = self.make_pipeline("keypoint")
        request = self.request(runtime, [10, 20])
        matrices = [runtime.pre.prepare_affine(image, bbox) for image, bbox in zip(
            request["RAW_IMAGE"].value, request["BBOX"].value)]
        expected = np.stack([runtime.post.postprocess(np.full(212, marker / 100, np.float32), matrix, input_size=192)
                             for marker, matrix in zip((10, 20), matrices)])
        with mock.patch.object(runtime.post, "postprocess", wraps=runtime.post.postprocess) as post:
            response = runtime.execute([request])[0]
        self.assertIsNone(response.error)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(state.warps[0][1].dtype, np.float64)
        np.testing.assert_array_equal(state.warps[0][1], matrices)
        np.testing.assert_array_equal(response.output_tensors[0].value, expected)
        self.assertEqual(response.output_tensors[0].value.shape, (2, 106, 2))

    def test_recognition_keeps_original_alignment_and_final_host_output_alive(self):
        runtime, state = self.make_pipeline("recognition")
        request = self.request(runtime, [11])
        expected_matrix = runtime.pre.estimate_norm(request["LANDMARK"].value[0])
        responses = runtime.execute([request, self.request(runtime, [22, 33, 44])])
        self.assertEqual(state.batches, [[11, 22, 33, 44]])
        self.assertEqual(state.downloads, 1)
        np.testing.assert_array_equal(state.warps[0][1][0], expected_matrix)
        for response, markers in zip(responses, ([11], [22, 33, 44])):
            self.assertIsNone(response.error)
            expected = np.repeat(np.asarray(markers, np.float32)[:, None] / np.float32(100), 512, axis=1)
            np.testing.assert_array_equal(response.output_tensors[0].value, expected)
            self.assertTrue(response.output_tensors[0].value.flags.c_contiguous)
            self.assertIsNotNone(response.output_tensors[0].value.base)

    def test_mixed_shapes_share_trt_with_device_concat_and_keep_batch_limit_and_order(self):
        runtime, state = self.make_pipeline()
        requests = [self.request(runtime, [10, 11, 12]), self.request(runtime, [20], shape=(128, 128)),
                    self.request(runtime, [30, 31]), self.request(runtime, [40]),
                    self.request(runtime, [50], shape=(128, 128))]
        responses = runtime.execute(requests)
        self.assertEqual(state.batches, [[10, 11, 12, 20], [30, 31, 40, 50]])
        self.assertEqual(state.concatenations, state.batches)
        self.assertEqual([tuple(images.shape[1:3]) for images, _ in state.warps],
                         [(112, 112), (128, 128), (112, 112), (112, 112), (128, 128)])
        self.assertEqual(state.downloads, 2)
        self.assertEqual(runtime.stream.synchronizations, 2)
        self.assertTrue(all(response.error is None for response in responses))
        self.assertEqual([r.output_tensors[0].value.shape[0] for r in responses], [3, 1, 2, 1, 1])
        for response, expected in zip(responses, ([1, 1.1, 1.2], [2], [3, 3.1], [4], [5])):
            np.testing.assert_allclose(response.output_tensors[0].value[:, 0], expected, rtol=1e-6)

    def test_pack_memory_budget_starts_another_group(self):
        runtime, state = self.make_pipeline()
        request = self.request(runtime, [10, 20])
        size = request[runtime.spec["image"]].value.nbytes
        with mock.patch.object(production, "MAX_PACK_BYTES", size):
            responses = runtime.execute([request, self.request(runtime, [30])])
        self.assertEqual(state.batches, [[10, 20], [30]])
        self.assertTrue(all(response.error is None for response in responses))

    def test_single_request_over_pack_memory_budget_is_rejected_before_gpu(self):
        runtime, state = self.make_pipeline()
        with mock.patch.object(production, "MAX_PACK_BYTES", 1):
            response = runtime.execute([self.request(runtime, [10])])[0]
        self.assertIn("INVALID_INPUT", str(response.error))
        self.assertEqual(state.batches, [])
        self.assertEqual(state.warps, [])

    def test_invalid_sample_rejects_only_its_whole_request(self):
        runtime, state = self.make_pipeline()
        bad = self.request(runtime, [20, 21])
        bad["LANDMARK"].value[1, 2, 0] = np.nan
        responses = runtime.execute([self.request(runtime, [10]), bad, self.request(runtime, [30])])
        self.assertEqual(state.batches, [[10, 30]])
        self.assertRegex(str(responses[1].error), "INVALID_INPUT: batch index 1: LANDMARK values must be finite")
        self.assertIsNone(responses[0].error)
        self.assertIsNone(responses[2].error)
        runtime.metrics.error.assert_called_once_with("invalid")
        self.assertEqual(sum(call.args[0] == "prepare" for call in runtime.metrics.stage.call_args_list), 3)

    def test_missing_device_transport_oversized_and_invalid_dtype_do_no_gpu_work(self):
        runtime, state = self.make_pipeline()
        missing = self.request(runtime, [10])
        del missing["LANDMARK"]
        device = self.request(runtime, [10])
        device["LANDMARK"].cpu = False
        oversized = self.request(runtime, [10] * 5)
        wrong_dtype = self.request(runtime, [10])
        wrong_dtype["RAW_FACE_IMAGE"].value = wrong_dtype["RAW_FACE_IMAGE"].value.astype(np.float32)
        empty = self.request(runtime, [10])
        empty["RAW_FACE_IMAGE"].value = empty["RAW_FACE_IMAGE"].value[:0]
        empty["LANDMARK"].value = empty["LANDMARK"].value[:0]
        responses = runtime.execute([missing, device, oversized, wrong_dtype, empty])
        self.assertTrue(all("INVALID_INPUT" in str(response.error) for response in responses))
        self.assertEqual(state.batches, [])
        self.assertEqual(state.warps, [])
        self.assertEqual(runtime.execute([]), [])

    def test_cancelled_request_does_no_prepare_or_device_work(self):
        runtime, state = self.make_pipeline()
        request = self.request(runtime, [10])
        request.cancelled = True
        with mock.patch.object(runtime, "_prepare", side_effect=AssertionError("cancelled work was prepared")):
            response = runtime.execute([request])[0]
        self.assertIn("CANCELLED", str(response.error))
        self.assertEqual(response.error.code, TritonError.CANCELLED)
        self.assertEqual(state.batches, [])
        runtime.metrics.error.assert_called_once_with("cancelled")

    def test_cancellation_during_prepare_avoids_gpu_and_counts_once(self):
        runtime, state = self.make_pipeline()
        request = self.request(runtime, [10])
        prepare = runtime.pre.prepare_affine

        def cancel_after_geometry(*args, **kwargs):
            matrix = prepare(*args, **kwargs)
            request.cancelled = True
            return matrix

        with mock.patch.object(runtime.pre, "prepare_affine", side_effect=cancel_after_geometry):
            responses = runtime.execute([request])
        self.assertEqual(responses[0].error.code, TritonError.CANCELLED)
        self.assertEqual(state.batches, [])
        self.assertTrue(runtime._set_cancelled(0, request, responses))
        runtime.metrics.error.assert_called_once_with("cancelled")

    def test_cancellation_while_neighbor_prepares_removes_only_cancelled_request(self):
        runtime, state = self.make_pipeline()
        cancelled, valid = self.request(runtime, [10]), self.request(runtime, [20])
        prepare = runtime.pre.prepare_affine

        def cancel_previous(image, auxiliary):
            matrix = prepare(image, auxiliary)
            if image[0, 0, 0] == 20:
                cancelled.cancelled = True
            return matrix

        with mock.patch.object(runtime.pre, "prepare_affine", side_effect=cancel_previous):
            responses = runtime.execute([cancelled, valid])
        self.assertEqual(state.batches, [[20]])
        self.assertEqual(responses[0].error.code, TritonError.CANCELLED)
        self.assertIsNone(responses[1].error)
        np.testing.assert_allclose(responses[1].output_tensors[0].value, [[2]], rtol=1e-6)
        runtime.metrics.error.assert_called_once_with("cancelled")
        runtime.metrics.batch.assert_called_once_with(1, 1)

    def test_cancellation_during_inference_keeps_neighbor_offsets_and_response(self):
        runtime, state = self.make_pipeline()
        cancelled, valid = self.request(runtime, [10]), self.request(runtime, [20, 30])
        infer = runtime.runner.infer

        def cancel_after_infer(*args, **kwargs):
            output = infer(*args, **kwargs)
            cancelled.cancelled = True
            return output

        with mock.patch.object(runtime.runner, "infer", side_effect=cancel_after_infer):
            responses = runtime.execute([cancelled, valid])
        self.assertEqual(state.batches, [[10, 20, 30]])
        self.assertEqual(responses[0].error.code, TritonError.CANCELLED)
        self.assertIsNone(responses[1].error)
        np.testing.assert_allclose(responses[1].output_tensors[0].value[:, 0], [2, 3], rtol=1e-6)
        runtime.metrics.error.assert_called_once_with("cancelled")
        self.assertEqual(state.lifetime_failures, 0)

    def test_cancellation_during_postprocessing_is_checked_before_response_creation(self):
        runtime, _ = self.make_pipeline()
        request = self.request(runtime, [10])
        postprocess = runtime._postprocess

        def cancel_after_post(*args, **kwargs):
            value = postprocess(*args, **kwargs)
            request.cancelled = True
            return value

        with mock.patch.object(runtime, "_postprocess", side_effect=cancel_after_post), \
                mock.patch.object(runtime.pb, "Tensor", side_effect=AssertionError("cancelled output must not be emitted")):
            response = runtime.execute([request])[0]
        self.assertEqual(response.error.code, TritonError.CANCELLED)
        runtime.metrics.error.assert_called_once_with("cancelled")

    def test_later_gpu_failure_does_not_overwrite_request_cancelled_before_group(self):
        runtime, state = self.make_pipeline()
        cancelled, valid = self.request(runtime, [10]), self.request(runtime, [20])
        prepare = runtime.pre.prepare_affine

        def cancel_previous(image, auxiliary):
            matrix = prepare(image, auxiliary)
            if image[0, 0, 0] == 20:
                cancelled.cancelled = True
            return matrix

        state.fail = "infer"
        with mock.patch.object(runtime.pre, "prepare_affine", side_effect=cancel_previous), \
                self.assertLogs(production.logger, level="ERROR"):
            responses = runtime.execute([cancelled, valid])
        self.assertEqual(responses[0].error.code, TritonError.CANCELLED)
        self.assertEqual(responses[1].error.code, "INTERNAL")
        self.assertEqual(runtime.metrics.error.call_args_list, [mock.call("cancelled"), mock.call("internal")])

    def test_gpu_failures_synchronize_keep_buffers_alive_and_do_not_poison_other_groups(self):
        for stage in ("warp", "warp_second", "concat", "infer", "download"):
            with self.subTest(stage=stage):
                runtime, state = self.make_pipeline()
                state.fail = stage
                with self.assertLogs(production.logger, level="ERROR"):
                    responses = runtime.execute([self.request(runtime, [10]),
                                                 self.request(runtime, [20, 21, 22], shape=(128, 128)),
                                                 self.request(runtime, [30])])
                for response in responses[:2]:
                    self.assertRegex(str(response.error), "INTERNAL_ERROR: face_qa_ensemble failed; incident=[0-9a-f]{12}")
                    self.assertNotIn("private", str(response.error))
                self.assertIsNone(responses[2].error)
                self.assertEqual(runtime.stream.synchronizations, 2)
                self.assertEqual(state.lifetime_failures, 0)
                self.assertEqual(runtime.metrics.error.call_count, 2)
                stages = [call.args[0] for call in runtime.metrics.stage.call_args_list]
                self.assertEqual(stages.count("warp"), 2)
                self.assertEqual(stages.count("trt_download"), 2 if stage in ("infer", "download") else 1)

    def test_nonfinite_model_result_is_isolated_after_shared_execution(self):
        runtime, state = self.make_pipeline()
        state.nonfinite_marker = 20
        with self.assertLogs(production.logger, level="ERROR"):
            responses = runtime.execute([self.request(runtime, [10]), self.request(runtime, [20]),
                                         self.request(runtime, [30])])
        self.assertEqual(state.batches, [[10, 20, 30]])
        self.assertIsNone(responses[0].error)
        self.assertIn("INTERNAL_ERROR", str(responses[1].error))
        self.assertIsNone(responses[2].error)
        self.assertEqual(sum(call.args[0] == "postprocess" for call in runtime.metrics.stage.call_args_list), 3)

    def test_finalize_synchronizes_before_releasing_gpu_objects(self):
        runtime, _ = self.make_pipeline()
        runtime.finalize()
        self.assertEqual(runtime.stream.synchronizations, 1)
        self.assertIsNone(runtime.runner)
        self.assertIsNone(runtime.gpu)
        runtime.metrics.close.assert_called_once()

    def test_public_configs_require_original_tensor_contract_and_api_version(self):
        for mode in NAMES:
            with self.subTest(mode=mode):
                args = config_args(mode)
                self.assertEqual(production.validate_config(args)["mode"], mode)
                encoded = copy.deepcopy(args)
                encoded["model_config"] = json.dumps(encoded["model_config"])
                self.assertEqual(production.validate_config(encoded)["mode"], mode)
                for field, value in (("data_type", "TYPE_FP64"), ("dims", [3]), ("name", "OTHER")):
                    invalid = copy.deepcopy(args)
                    invalid["model_config"]["input"][1][field] = value
                    with self.assertRaises(ValueError):
                        production.validate_config(invalid)
                for field, value in (("pipeline_backend", "cpu"), ("api_contract_version", "99")):
                    invalid = copy.deepcopy(args)
                    invalid["model_config"]["parameters"][field]["string_value"] = value
                    with self.assertRaises(ValueError):
                        production.validate_config(invalid)
                invalid = copy.deepcopy(args)
                invalid["model_instance_kind"] = "CPU"
                with self.assertRaises(ValueError):
                    production.validate_config(invalid)
                invalid = copy.deepcopy(args)
                invalid["model_config"]["max_batch_size"] = 8
                with self.assertRaises(ValueError):
                    production.validate_config(invalid)


if __name__ == "__main__":
    unittest.main()
