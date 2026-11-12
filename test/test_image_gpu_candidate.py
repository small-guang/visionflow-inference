"""Parity checks for the optional image benchmark; production models stay separate."""

import importlib
import io
import os
import platform
import sys
import unittest
from unittest import mock

import cv2
import numpy as np
from PIL import Image


def candidate_module():
    try:
        return importlib.import_module("tools.image_gpu_candidate")
    except ModuleNotFoundError as exc:
        if exc.name != "tools.image_gpu_candidate":
            raise
        return None


class ImageGpuCandidateCpuTests(unittest.TestCase):
    def setUp(self):
        self.module = candidate_module()
        self.assertIsNotNone(self.module, "benchmark candidate module is missing")

    def test_cpu_tensor_preserves_rgb_scale_and_model_normalization(self):
        rgb = np.full((9, 17, 3), [240, 80, 10], dtype=np.uint8)
        for alias, shape, expected in (
            ("clip_image", (3, 224, 224), [1.7113599, -0.5514756, -1.3380191]),
            ("tbps_clip_image", (3, 224, 224), [1.9920369, -0.6351540, -1.6301525]),
            ("sbs", (3, 384, 128), [1.9920369, -0.6351540, -1.6301525]),
        ):
            with self.subTest(alias=alias), mock.patch.dict(sys.modules, {"cupy": None}):
                actual = self.module.ImageGpuPreprocessor(alias).cpu_tensor(rgb)
                self.assertEqual(actual.shape, shape)
                self.assertEqual(actual.dtype, np.float32)
                self.assertTrue(actual.flags.c_contiguous)
                np.testing.assert_allclose(actual[:, 0, 0], expected, rtol=0, atol=2e-6)

    def test_decode_preserves_original_library_color_and_jpeg_semantics(self):
        rng = np.random.default_rng(7283)
        for fmt in ("PNG", "JPEG"):
            for source in (rng.integers(0, 256, (17, 21, 3), dtype=np.uint8),
                           rng.integers(0, 256, (17, 21), dtype=np.uint8)):
                encoded = io.BytesIO()
                Image.fromarray(source).save(encoded, format=fmt)
                payload = encoded.getvalue()
                pil_expected = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
                cv_expected = cv2.cvtColor(cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                for alias in ("clip_image", "tbps_clip_image", "sbs"):
                    with self.subTest(alias=alias, fmt=fmt, ndim=source.ndim):
                        actual = self.module.ImageGpuPreprocessor(alias).decode(payload)
                        expected = pil_expected if alias == "tbps_clip_image" else cv_expected
                        np.testing.assert_array_equal(actual, expected)

    def test_cpu_resize_matches_each_original_resize_on_texture_and_edges(self):
        rgb = np.random.default_rng(42).integers(0, 256, (517, 301, 3), dtype=np.uint8)
        for alias, size in (("clip_image", (224, 224)), ("tbps_clip_image", (224, 224)), ("sbs", (128, 384))):
            with self.subTest(alias=alias):
                processor = self.module.ImageGpuPreprocessor(alias)
                expected = (np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.BICUBIC))
                            if alias == "tbps_clip_image" else cv2.resize(rgb, size, interpolation=cv2.INTER_CUBIC))
                np.testing.assert_array_equal(processor.cpu_resize(rgb), expected)

    def test_invalid_input_has_no_silent_fallback(self):
        with self.assertRaises(ValueError):
            self.module.ImageGpuPreprocessor("unknown")
        for alias in ("clip_image", "tbps_clip_image", "sbs"):
            processor = self.module.ImageGpuPreprocessor(alias)
            for payload in (b"", b"not an image"):
                with self.subTest(alias=alias, payload=payload), self.assertRaises(ValueError):
                    processor.decode(payload)
            with self.assertRaises(ValueError):
                processor.cpu_tensor(np.zeros((4, 5), np.uint8))
            with self.assertRaises(ValueError):
                processor.gpu_tensor(np.zeros((4, 5, 3), np.uint8), mode="invalid")

    def test_parity_metrics_exposes_mismatch_and_rejects_bad_shape(self):
        metrics = self.module.parity_metrics(np.array([0, 1, 2], np.float32), np.array([0, 2, 2], np.float32))
        self.assertEqual(metrics["max_abs"], 1.0)
        self.assertAlmostEqual(metrics["mean_abs"], 1 / 3)
        self.assertEqual(metrics["different_elements"], 1)
        with self.assertRaises(ValueError):
            self.module.parity_metrics(np.ones(2), np.ones(3))

    def test_neon_rounding_is_opt_in_and_rejects_unknown_arithmetic(self):
        rgb = np.full((5, 7, 3), 128, dtype=np.uint8)
        default = self.module.ImageGpuPreprocessor("clip_image")
        neon = self.module.ImageGpuPreprocessor("clip_image", opencv_rounding="neon")
        np.testing.assert_array_equal(default.cpu_tensor(rgb), neon.cpu_tensor(rgb))
        with self.assertRaises(ValueError):
            self.module.ImageGpuPreprocessor("clip_image", opencv_rounding="fast")


def cuda_module():
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() > 0:
            return cupy
    except (ImportError, RuntimeError):
        pass
    return None


CP = cuda_module()


@unittest.skipUnless(CP is not None, "requires CuPy and CUDA")
class ImageGpuCandidateCudaTests(unittest.TestCase):
    @unittest.skipUnless(platform.machine().lower() in ("aarch64", "arm64"), "requires ARM NEON OpenCV reference")
    def test_neon_resize_matches_arm_opencv_pixels_and_normalized_tensor(self):
        module = candidate_module()
        for alias in ("clip_image", "sbs"):
            processor = module.ImageGpuPreprocessor(alias, cp_module=CP, opencv_rounding="neon")
            images = [np.random.default_rng(sum(shape)).integers(0, 256, (*shape, 3), dtype=np.uint8)
                      for shape in ((19, 31), (224, 224), (539, 917), (1, 7), (401, 3), (997, 53))]
            # Root supplies the same reviewed real dataset for target validation.
            if os.environ.get("IMAGE_GPU_PARITY_DATASET"):
                from tools.benchmark_image_gpu import sample_images
                from pathlib import Path
                images.extend(processor.decode(Path(sample["path"]).read_bytes())
                              for sample in sample_images(os.environ["IMAGE_GPU_PARITY_DATASET"]))
            stream = CP.cuda.Stream(non_blocking=True)
            for index, rgb in enumerate(images):
                with self.subTest(alias=alias, image=index, shape=rgb.shape), stream:
                    actual = processor.gpu_tensor(rgb, mode="resize").get(stream=stream, blocking=True)
                    cpu = processor.cpu_tensor(rgb)
                    mean = processor.mean[:, None, None]
                    std = processor.std[:, None, None]
                    recovered = np.rint((actual * std + mean) * 255).astype(np.int16)
                    expected_pixels = processor.cpu_resize(rgb).transpose(2, 0, 1).astype(np.int16)
                    np.testing.assert_array_equal(recovered, expected_pixels)
                    np.testing.assert_allclose(actual, cpu, rtol=0, atol=5e-7)

    def test_normalize_mode_matches_cpu_exact_resize_on_nondefault_stream(self):
        module = candidate_module()
        rgb = np.random.default_rng(55).integers(0, 256, (539, 917, 3), dtype=np.uint8)
        stream = CP.cuda.Stream(non_blocking=True)
        for alias in ("clip_image", "tbps_clip_image", "sbs"):
            with self.subTest(alias=alias), stream:
                processor = module.ImageGpuPreprocessor(alias, cp_module=CP)
                actual = processor.gpu_tensor(rgb, mode="normalize")
                stream.synchronize()
                np.testing.assert_allclose(CP.asnumpy(actual), processor.cpu_tensor(rgb), rtol=0, atol=5e-7)

    def test_resize_mode_preserves_pillow_antialias_and_uint8_pass_rounding(self):
        module = candidate_module()
        processor = module.ImageGpuPreprocessor("tbps_clip_image", cp_module=CP)
        for shape in ((19, 31), (224, 224), (224, 533), (537, 224), (539, 917), (1, 7), (401, 3)):
            with self.subTest(shape=shape):
                rgb = np.random.default_rng(sum(shape)).integers(0, 256, (*shape, 3), dtype=np.uint8)
                actual = CP.asnumpy(processor.gpu_tensor(rgb, mode="resize"))
                np.testing.assert_allclose(actual, processor.cpu_tensor(rgb), rtol=0, atol=5e-7)

    def test_opencv_resize_candidate_reports_at_most_one_uint8_rounding_step(self):
        module = candidate_module()
        for alias in ("clip_image", "sbs"):
            processor = module.ImageGpuPreprocessor(alias, cp_module=CP)
            for shape in ((19, 31), (224, 224), (539, 917), (1, 7), (401, 3)):
                with self.subTest(alias=alias, shape=shape):
                    rgb = np.random.default_rng(sum(shape)).integers(0, 256, (*shape, 3), dtype=np.uint8)
                    actual = CP.asnumpy(processor.gpu_tensor(rgb, mode="resize"))
                    cpu = processor.cpu_tensor(rgb)
                    # OpenCV's SIMD path can use float/FMA vertical rounding.
                    limit = 1.0 / (255.0 * min(processor.spec["std"])) + 1e-6
                    self.assertLessEqual(module.parity_metrics(cpu, actual)["max_abs"], limit)


if __name__ == "__main__":
    unittest.main(verbosity=2)
