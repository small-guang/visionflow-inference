"""融合入口的 request/batch/生命周期契约；不替代真实 Orin 数值验收。"""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from scripts import image_orin_pipeline as production


ROOT = Path(__file__).resolve().parents[1]


class Tensor:
    def __init__(self, name, value): self.name, self.value = name, value
    def as_numpy(self): return self.value
    def is_cpu(self): return True


class Error(Exception):
    CANCELLED = "cancelled"


class Stream:
    ptr = 73
    def __init__(self): self.syncs = 0
    def __enter__(self): return self
    def __exit__(self, *_): return False
    def synchronize(self): self.syncs += 1


class Processor:
    def __init__(self): self.calls = []; self.last_tensor = None
    def decode(self, payload, batch_index=0):
        if payload == b"bad": raise ValueError(f"unable to decode image at batch index {batch_index}")
        value = int(payload)
        return np.full((2 + value % 2, 3, 3), value, np.uint8)
    def preprocess_batch(self, images):
        self.calls.append([int(image[0, 0, 0]) for image in images])
        self.last_tensor = np.asarray(self.calls[-1], np.float32)[:, None]
        return self.last_tensor
    def to_host(self, value): raise AssertionError("preprocessed tensor must never return to host")


class Decoder:
    def __init__(self, decode_one): self.decode_one = decode_one; self.calls = []
    def decode(self, items):
        self.calls.append(list(items))
        values = []
        for item in items:
            try: values.append(types.SimpleNamespace(value=self.decode_one(item), error=None))
            except Exception as exc: values.append(types.SimpleNamespace(value=None, error=exc))
        return values
    def close(self): pass


class Runner:
    def __init__(self, spec): self.spec = spec; self.calls = []; self.fail_first = False
    def infer(self, tensor, stream_handle):
        self.calls.append((tensor, stream_handle))
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("private TRT driver details")
        return {self.spec["output"]: np.repeat(tensor, self.spec["width"], axis=1)}


PB = types.SimpleNamespace(Tensor=Tensor, TritonError=Error,
    InferenceResponse=lambda **values: types.SimpleNamespace(error=None, **values) if "error" not in values
        else types.SimpleNamespace(**values),
    get_input_tensor_by_name=lambda request, name: request.get(name))


def request(*values):
    return {"IMAGE_BYTES": Tensor("IMAGE_BYTES", np.asarray(values, dtype=object).reshape(-1, 1))}


def model(name="clip_ensemble_trt"):
    value = production.TritonPythonModel()
    value.public_model, value.spec = name, production.SPECS[name]
    value.pb, value.pre = PB, Processor()
    value.decoder = Decoder(value._decode_one)
    value.runner = Runner(value.spec)
    value.device = Stream(); value.stream = Stream()
    value.cp = types.SimpleNamespace(asnumpy=mock.Mock(side_effect=lambda output, stream: output.copy()))
    value.metrics = mock.Mock(spec=production.NullMetrics)
    return value


class ImageRequestTests(unittest.TestCase):
    def test_three_requests_share_one_b4_preprocess_and_trt_and_split_in_original_order(self):
        for name in production.SPECS:
            with self.subTest(model=name):
                pipeline = model(name)
                result = pipeline.execute([request(b"1"), request(b"2", b"3"), request(b"4")])
                self.assertEqual(pipeline.pre.calls, [[1, 2, 3, 4]])
                self.assertEqual(len(pipeline.runner.calls), 1)
                tensor, stream = pipeline.runner.calls[0]
                self.assertIs(tensor, pipeline.pre.last_tensor)
                self.assertEqual(stream, 73)
                self.assertEqual(pipeline.cp.asnumpy.call_count, 1)
                self.assertEqual([r.output_tensors[0].value.shape for r in result],
                    [(1, pipeline.spec["width"]), (2, pipeline.spec["width"]), (1, pipeline.spec["width"])])
                np.testing.assert_array_equal(result[1].output_tensors[0].value[:, 0], [2, 3])
                self.assertTrue(result[1].output_tensors[0].value.flags.c_contiguous)
                self.assertIsNotNone(result[1].output_tensors[0].value.base)
                pipeline.metrics.batch.assert_called_once_with(4, 3)
                self.assertEqual(pipeline.stream.syncs, 1)

    def test_bad_image_discards_only_its_whole_request_then_runs_valid_neighbors_together(self):
        pipeline = model()
        result = pipeline.execute([request(b"1", b"bad"), request(b"3"), request(b"4")])
        self.assertIn("batch index 1", str(result[0].error))
        self.assertEqual(pipeline.pre.calls, [[3, 4]])
        pipeline.metrics.batch.assert_called_once_with(2, 2)
        self.assertIsNone(result[1].error)
        self.assertIsNone(result[2].error)

    def test_real_bounded_decoder_keeps_request_error_indices_and_reuses_pool(self):
        from scripts.image_decode import BoundedImageDecoder
        pipeline = model()
        pipeline.decoder = BoundedImageDecoder(pipeline._decode_one, workers=4, max_batch_size=4)
        try:
            for _ in range(2):
                result = pipeline.execute([request(b"1", b"bad"), request(b"3"), request(b"4")])
                self.assertIn("batch index 1", str(result[0].error))
                self.assertEqual(pipeline.pre.calls[-1], [3, 4])
                np.testing.assert_array_equal(result[1].output_tensors[0].value[:, 0], [3])
        finally:
            pipeline.finalize()

    def test_unexpected_decode_errors_are_private_and_valid_neighbors_still_share_gpu_work(self):
        for error in (MemoryError("private allocation details"), RuntimeError("private decoder details")):
            with self.subTest(error=type(error).__name__):
                pipeline = model()
                decode = pipeline.pre.decode
                def decode_with_failure(payload, batch_index=0):
                    if payload == b"2":
                        raise error
                    return decode(payload, batch_index)
                pipeline.pre.decode = decode_with_failure
                with self.assertLogs(production.logger, "ERROR") as logs:
                    result = pipeline.execute([request(b"1"), request(b"2"), request(b"3")])
                self.assertIn("INTERNAL_ERROR", str(result[1].error))
                self.assertIn("incident=", str(result[1].error))
                self.assertNotIn("private", str(result[1].error))
                self.assertIn(str(error), "\n".join(logs.output))
                self.assertIsNone(result[0].error)
                self.assertIsNone(result[2].error)
                self.assertEqual(pipeline.pre.calls, [[1, 3]])
                pipeline.metrics.error.assert_called_once_with("internal", 1)

    def test_response_failure_preserves_cancelled_and_successful_neighbors(self):
        class LateCancelled(dict):
            def __init__(self):
                super().__init__(request(b"1"))
                self.checks = 0
            def is_cancelled(self):
                self.checks += 1
                return self.checks == 4
        for first_cancelled in (False, True):
            with self.subTest(first_cancelled=first_cancelled):
                pipeline = model()
                def tensor(name, value):
                    if value[0, 0] == 2:
                        raise RuntimeError("private response construction details")
                    return Tensor(name, value)
                pipeline.pb = types.SimpleNamespace(**{**vars(PB), "Tensor": tensor})
                first = LateCancelled() if first_cancelled else request(b"1")
                with self.assertLogs(production.logger, "ERROR"):
                    result = pipeline.execute([first, request(b"2"), request(b"3")])
                if first_cancelled:
                    self.assertEqual(result[0].error.args, ("request cancelled", Error.CANCELLED))
                    self.assertEqual(pipeline.metrics.error.call_args_list,
                                     [mock.call("cancelled"), mock.call("internal", 1)])
                else:
                    self.assertIsNone(result[0].error)
                    pipeline.metrics.error.assert_called_once_with("internal", 1)
                self.assertIn("INTERNAL_ERROR", str(result[1].error))
                self.assertNotIn("private", str(result[1].error))
                self.assertIsNone(result[2].error)
                np.testing.assert_array_equal(result[2].output_tensors[0].value[:, 0], [3])

    def test_internal_decode_failure_is_not_hidden_by_another_invalid_slot_in_the_request(self):
        pipeline = model()
        decode = pipeline.pre.decode
        def decode_with_failure(payload, batch_index=0):
            if payload == b"2":
                raise MemoryError("private allocation details")
            return decode(payload, batch_index)
        pipeline.pre.decode = decode_with_failure
        with self.assertLogs(production.logger, "ERROR"):
            result = pipeline.execute([request(b"bad", b"2"), request(b"3")])
        self.assertIn("INTERNAL_ERROR", str(result[0].error))
        self.assertNotIn("private", str(result[0].error))
        self.assertIsNone(result[1].error)
        self.assertEqual(pipeline.pre.calls, [[3]])
        pipeline.metrics.error.assert_called_once_with("internal", 1)

    def test_group_failure_only_terminates_unresolved_responses(self):
        pipeline = model()
        success = PB.InferenceResponse(output_tensors=[Tensor("value", np.zeros(1))])
        cancelled = PB.InferenceResponse(output_tensors=[], error=Error("request cancelled", Error.CANCELLED))
        responses = [success, cancelled, None]
        with self.assertLogs(production.logger, "ERROR"):
            pipeline._fail_group([(0, None), (1, None), (2, None)], responses,
                                 error=RuntimeError("driver details"))
        self.assertIs(responses[0], success)
        self.assertIs(responses[1], cancelled)
        self.assertIn("INTERNAL_ERROR", str(responses[2].error))
        pipeline.metrics.error.assert_called_once_with("internal", 1)
        pipeline._fail_group([(0, None), (1, None), (2, None)], responses)
        pipeline.metrics.error.assert_called_once_with("internal", 1)

    def test_request_is_never_split_across_gpu_groups(self):
        pipeline = model()
        result = pipeline.execute([request(b"1", b"2", b"3"), request(b"4", b"5"), request(b"6")])
        self.assertEqual(pipeline.pre.calls, [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(len(result), 3)
        self.assertEqual([len(items) for items in pipeline.decoder.calls], [3, 3])

    def test_invalid_shape_type_transport_and_empty_batch_never_submit_work(self):
        pipeline = model()
        device_input = Tensor("IMAGE_BYTES", np.array([[b"1"]], object))
        device_input.is_cpu = lambda: False
        cases = [{}, request(), request(*([b"1"] * 5)),
                 {"IMAGE_BYTES": Tensor("IMAGE_BYTES", np.array([b"1"], object))},
                 {"IMAGE_BYTES": Tensor("IMAGE_BYTES", np.array([[b"1"]], dtype="S1"))},
                 {"IMAGE_BYTES": device_input}]
        result = pipeline.execute(cases)
        self.assertTrue(all("INVALID_INPUT" in str(r.error) for r in result))
        self.assertEqual(pipeline.runner.calls, [])
        self.assertEqual(pipeline.decoder.calls, [])
        self.assertEqual(pipeline.execute([]), [])

    def test_trt_failure_isolated_to_group_and_stream_syncs_before_following_group(self):
        pipeline = model(); pipeline.runner.fail_first = True
        with self.assertLogs(production.logger, "ERROR"):
            result = pipeline.execute([request(b"1", b"2", b"3", b"4"), request(b"5")])
        self.assertIn("INTERNAL_ERROR", str(result[0].error))
        self.assertNotIn("private TRT driver", str(result[0].error))
        self.assertIsNone(result[1].error)
        self.assertEqual(pipeline.stream.syncs, 2)
        pipeline.metrics.batch.assert_called_once_with(1, 1)

    def test_preprocess_or_download_error_always_synchronizes(self):
        for stage in ("preprocess", "download"):
            pipeline = model()
            if stage == "preprocess":
                pipeline.pre.preprocess_batch = mock.Mock(side_effect=RuntimeError("CUDA"))
            else:
                pipeline.cp.asnumpy.side_effect = RuntimeError("D2H")
            with self.assertLogs(production.logger, "ERROR"):
                result = pipeline.execute([request(b"1")])
            self.assertIn("INTERNAL_ERROR", str(result[0].error))
            self.assertEqual(pipeline.stream.syncs, 1)

    def test_cancelled_request_is_removed_before_decode_or_output(self):
        class Request(dict):
            def __init__(self, when): super().__init__(request(b"2")); self.calls = 0; self.when = when
            def is_cancelled(self): self.calls += 1; return self.calls >= self.when
        pipeline = model()
        early, late = Request(1), Request(4)
        results = pipeline.execute([early, request(b"1"), late])
        self.assertEqual(pipeline.pre.calls, [[1, 2]])
        self.assertIn("request cancelled", str(results[0].error))
        self.assertIn("request cancelled", str(results[2].error))
        self.assertIsNone(results[1].error)

    def test_finalize_closes_decoder_before_synchronizing_and_clears_runner(self):
        pipeline = model(); calls = []
        pipeline.decoder.close = lambda: calls.append("decoder")
        pipeline.stream.synchronize = lambda: calls.append("stream")
        pipeline.finalize()
        self.assertEqual(calls, ["decoder", "stream"])
        self.assertIsNone(pipeline.runner)
        pipeline.metrics.close.assert_called_once()


class ConfigurationTests(unittest.TestCase):
    def config(self, name):
        spec = production.SPECS[name]
        return {"max_batch_size": 4,
            "parameters": {"pipeline_backend": {"string_value": "orin_gpu"},
                "api_contract_version": {"string_value": spec["version"]}},
            "input": [{"name": "IMAGE_BYTES", "data_type": "TYPE_STRING", "dims": [1]}],
            "output": [{"name": spec["output"], "data_type": "TYPE_FP32", "dims": [spec["width"]]}]}

    def test_invalid_backend_instance_batch_and_public_contract_fail_before_cuda_import(self):
        name = "clip_ensemble_trt"
        for change in (lambda c: c.update(max_batch_size=8),
                       lambda c: c["parameters"].update(pipeline_backend={"string_value": "cpu"}),
                       lambda c: c["parameters"].update(jpeg_decode_workers={"string_value": "20"}),
                       lambda c: c["parameters"].update(api_contract_version={"string_value": "3"}),
                       lambda c: c["output"][0].update(dims=[513])):
            config = self.config(name); change(config)
            with self.assertRaises(ValueError):
                production.TritonPythonModel().initialize({"model_name": name,
                    "model_instance_kind": "GPU", "model_config": json.dumps(config)})
        with self.assertRaisesRegex(RuntimeError, "KIND_GPU"):
            production.TritonPythonModel().initialize({"model_name": name,
                "model_instance_kind": "CPU", "model_config": self.config(name)})

    def test_initialize_selects_original_engine_and_same_device_then_closes_real_decoder(self):
        class Device(Stream):
            def __init__(self, device_id): super().__init__(); self.id = device_id
        cp = types.SimpleNamespace(cuda=types.SimpleNamespace(Device=Device, Stream=lambda **_: Stream()))
        for name, spec in production.SPECS.items():
            with self.subTest(model=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); (root / name).mkdir()
                config = self.config(name)
                config["parameters"]["jpeg_decode_workers"] = {"string_value": "4" if spec["alias"] == "clip_image" else "2"}
                processor, runner = Processor(), Runner(spec)
                with mock.patch.dict(sys.modules, {"cupy": cp, "tensorrt": object(),
                        "triton_python_backend_utils": PB}), \
                        mock.patch.object(production, "GpuImagePreprocessor", return_value=processor) as pre_factory, \
                        mock.patch.object(production, "create_runner", return_value=runner) as factory, \
                        mock.patch.object(production, "PipelineMetrics", return_value=production.NullMetrics()):
                    instance = production.TritonPythonModel()
                    instance.initialize({"model_name": name, "model_instance_kind": "GPU",
                        "model_instance_device_id": "2", "model_repository": str(root / name),
                        "model_version": "1", "model_config": json.dumps(config)})
                    self.assertEqual(instance.device.id, 2)
                    pre_factory.assert_called_once_with(spec["alias"], 2, cp_module=cp)
                    self.assertEqual(factory.call_args.args[:3], (root, spec["native"], "1"))
                    self.assertEqual(factory.call_args.kwargs, {"device_id": 2, "input_name": "image",
                        "input_shape": spec["shape"], "output_shapes": {spec["output"]: (spec["width"],)}})
                    instance.finalize()

    def test_fused_profiles_preserve_public_contract_and_use_bounded_instances(self):
        from test.test_batch_model_repository import parse_config, _repeated_messages, _scalar
        for name, spec in production.SPECS.items():
            directory = ROOT / "model_repository" / name
            original = parse_config(directory / "config.pbtxt")
            fused = parse_config(directory / "configs/orin_gpu.pbtxt")
            for field in ("input", "output"):
                self.assertEqual(_repeated_messages(fused, field), _repeated_messages(original, field))
            self.assertEqual(_scalar(fused, "backend"), "python")
            self.assertEqual(_scalar(fused, "max_batch_size"), 4)
            self.assertNotIn("ensemble_scheduling", fused)
            group = _repeated_messages(fused, "instance_group")[0]
            self.assertEqual(_scalar(group, "count"), 2)
            self.assertEqual(_scalar(group, "kind"), "KIND_GPU")


if __name__ == "__main__":
    unittest.main()
