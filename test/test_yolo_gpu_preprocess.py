"""Real CUDA checks for YOLO preprocessing; run in the CuPy runtime image."""

import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import cv2
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

from test.test_batch_python_models import make_fake_pb_utils, FakeTensor


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("yolov8n_face", "yolo26m_pose", "yolo26s_seg")


class DeviceTensor(FakeTensor):
    def is_cpu(self):
        return isinstance(self.value, np.ndarray)

    def to_dlpack(self):
        return self.value.toDlpack()

    @classmethod
    def from_dlpack(cls, name, value):
        return cls(name, cp.from_dlpack(value))


def load_model(name, kind="GPU", *, host_outputs=False):
    path = ROOT / "model_repository" / (name + "_preprocess") / "1" / "model.py"
    spec = importlib.util.spec_from_file_location("gpu_test_" + name, path)
    module = importlib.util.module_from_spec(spec)
    utils = make_fake_pb_utils()
    utils.Tensor = DeviceTensor
    with mock.patch.dict(sys.modules, {"triton_python_backend_utils": utils}):
        spec.loader.exec_module(module)
    model = module.TritonPythonModel()
    config = {"parameters": {"GPU_OUTPUT_ON_CPU": {"string_value": "yes" if host_outputs else "no"}}}
    model.initialize({"model_instance_kind": kind, "model_instance_device_id": "0", "model_config": json.dumps(config)})
    return module, model


def output(response, name):
    return next(t.value for t in response.output_tensors if t.name == name)


class YoloGpuInitializationTests(unittest.TestCase):
    def test_cpu_instances_load_without_cupy(self):
        with mock.patch.dict(sys.modules, {"cupy": None}):
            for name in MODELS:
                _, model = load_model(name, "CPU")
                self.assertIsNone(model._gpu)

    def test_gpu_instances_fail_loading_without_cupy(self):
        with mock.patch.dict(sys.modules, {"cupy": None}):
            for name in MODELS:
                with self.subTest(model=name), self.assertRaises(ImportError):
                    load_model(name)


@unittest.skipUnless(cp is not None, "requires CuPy and a CUDA device")
class YoloGpuPreprocessTests(unittest.TestCase):
    def test_host_transport_keeps_real_gpu_computation(self):
        images = np.random.default_rng(719).integers(0, 256, size=(2, 101, 200, 3), dtype=np.uint8)
        for name in MODELS:
            with self.subTest(model=name):
                _, gpu = load_model(name, host_outputs=True)
                raw, metas = gpu._gpu.preprocess(images, meta_width=5 if name == "yolov8n_face" else 7)
                self.assertIsInstance(raw, cp.ndarray)
                response = gpu.execute([{"RAW_IMAGE": DeviceTensor("RAW_IMAGE", images)}])[0]
                self.assertIsNone(response.error)
                actual = output(response, "PREPROCESSED_IMAGE")
                self.assertIsInstance(actual, np.ndarray)
                np.testing.assert_array_equal(actual, cp.asnumpy(raw))
                np.testing.assert_array_equal(output(response, "LETTERBOX_META"), metas)

    def test_gpu_instance_returns_device_output_for_cpu_input(self):
        image = np.full((1, 180, 640, 3), [10, 20, 30], np.uint8)
        for name in MODELS:
            with self.subTest(model=name):
                _, model = load_model(name)
                self.assertIsNotNone(getattr(model, "_gpu", None))
                response = model.execute([{"RAW_IMAGE": DeviceTensor("RAW_IMAGE", image)}])[0]
                self.assertIsNone(response.error)
                pixels = output(response, "PREPROCESSED_IMAGE")
                self.assertIsInstance(pixels, cp.ndarray)
                self.assertTrue(pixels.flags.c_contiguous)
                self.assertEqual(pixels.shape, (1, 3, 192, 640))
                np.testing.assert_allclose(cp.asnumpy(pixels[0, :, 6, 0]), [30/255, 20/255, 10/255], atol=1e-7, rtol=0)
                np.testing.assert_allclose(cp.asnumpy(pixels[0, :, 0, 0]), np.full(3, 114/255), atol=1e-7, rtol=0)

    def test_seven_buckets_match_opencv_pixels_and_metadata(self):
        cases = ((180, 640), (101, 200), (480, 640), (513, 640), (120, 80), (200, 101), (640, 180), (901, 1601), (1081, 1921), (19, 31), (1, 1), (1, 640), (640, 1), (3, 7), (7, 3), (1280, 1280), (2560, 1280), (1024, 1024))
        rng = np.random.default_rng(917)
        for name in MODELS:
            _, cpu = load_model(name, "CPU")
            _, gpu = load_model(name)
            self.assertIsNotNone(getattr(gpu, "_gpu", None))
            for height, width in cases:
                with self.subTest(model=name, shape=(height, width)):
                    images = rng.integers(0, 256, size=(2, height, width, 3), dtype=np.uint8)
                    reference = cpu.execute([{"RAW_IMAGE": DeviceTensor("RAW_IMAGE", images)}])[0]
                    response = gpu.execute([{"RAW_IMAGE": DeviceTensor("RAW_IMAGE", images)}])[0]
                    self.assertIsNone(response.error)
                    actual = output(response, "PREPROCESSED_IMAGE")
                    np.testing.assert_allclose(cp.asnumpy(actual), output(reference, "PREPROCESSED_IMAGE"), rtol=0, atol=1e-7)
                    np.testing.assert_array_equal(output(response, "LETTERBOX_META"), output(reference, "LETTERBOX_META"))

    def test_dlpack_input_and_noncontiguous_input_preserve_values(self):
        _, model = load_model("yolo26m_pose")
        self.assertIsNotNone(getattr(model, "_gpu", None))
        host = np.random.default_rng(51).integers(0, 256, (4, 101, 400, 3), dtype=np.uint8)[:, :, ::2]
        device = cp.asarray(host)[:, ::-1]
        original = device.copy()
        _, cpu = load_model("yolo26m_pose", "CPU")
        reference = cpu.execute([{"RAW_IMAGE": DeviceTensor("RAW_IMAGE", cp.asnumpy(device))}])[0]
        response = model.execute([{"RAW_IMAGE": DeviceTensor("RAW_IMAGE", device)}])[0]
        self.assertIsNone(response.error)
        np.testing.assert_allclose(cp.asnumpy(output(response, "PREPROCESSED_IMAGE")), output(reference, "PREPROCESSED_IMAGE"), rtol=0, atol=1e-7)
        np.testing.assert_array_equal(cp.asnumpy(device), cp.asnumpy(original))

    def test_invalid_gpu_request_is_isolated_from_next_request(self):
        _, model = load_model("yolo26s_seg")
        self.assertIsNotNone(getattr(model, "_gpu", None))
        good = cp.zeros((1, 19, 31, 3), cp.uint8)
        bad_values = (good.astype(cp.float32), good[0], cp.empty((0, 19, 31, 3), cp.uint8), cp.empty((1, 0, 31, 3), cp.uint8), cp.empty((1, 2000, 1, 3), cp.uint8))
        for bad in bad_values:
            with self.subTest(shape=bad.shape, dtype=bad.dtype):
                responses = model.execute([{"RAW_IMAGE": DeviceTensor("RAW_IMAGE", bad)}, {"RAW_IMAGE": DeviceTensor("RAW_IMAGE", good)}])
                self.assertEqual(len(responses), 2)
                self.assertIn("INVALID_INPUT", str(responses[0].error))
                self.assertIsNone(responses[1].error)

    def test_bucket_boundaries_and_odd_padding_match_cpu(self):
        _, cpu = load_model("yolov8n_face", "CPU")
        _, gpu = load_model("yolov8n_face")
        self.assertIsNotNone(getattr(gpu, "_gpu", None))
        for short in (191, 192, 193, 383, 384, 385, 479, 480, 481, 639, 640):
            for height, width in ((short, 640), (640, short)):
                with self.subTest(shape=(height, width)):
                    images = np.full((1, height, width, 3), [255, 0, 127], np.uint8)
                    a = cpu.execute([{"RAW_IMAGE": DeviceTensor("RAW_IMAGE", images)}])[0]
                    b = gpu.execute([{"RAW_IMAGE": DeviceTensor("RAW_IMAGE", images)}])[0]
                    self.assertIsNone(b.error)
                    np.testing.assert_allclose(cp.asnumpy(output(b, "PREPROCESSED_IMAGE")), output(a, "PREPROCESSED_IMAGE"), atol=1e-7, rtol=0)
                    np.testing.assert_array_equal(output(b, "LETTERBOX_META"), output(a, "LETTERBOX_META"))

    def test_host_resource_limits_are_rejected_before_upload(self):
        _, model = load_model("yolo26s_seg")
        self.assertIsNotNone(getattr(model, "_gpu", None))
        pixel = np.zeros((1, 1, 1, 3), dtype=np.uint8)
        for shape in ((1, 8193, 1, 3), (1, 4096, 4096, 3)):
            value = np.lib.stride_tricks.as_strided(pixel, shape=shape, strides=(0, 0, 0, 1))
            with mock.patch.object(cp, "asarray", side_effect=AssertionError("uploaded invalid image")):
                response = model.execute([{"RAW_IMAGE": DeviceTensor("RAW_IMAGE", value)}])[0]
            self.assertIn("INVALID_INPUT", str(response.error))
            self.assertIn("batch index 0", str(response.error))


if __name__ == "__main__":
    unittest.main()
