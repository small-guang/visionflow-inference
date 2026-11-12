"""Pillow parity and real-CUDA checks for the TBPS production GPU helper."""

import importlib.util
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "model_repository" / "TBPS_clip_image_preprocess" / "1" / "gpu.py"
# H/W from failing real person crops, plus both sides of Pillow's strict
# pass-order thresholds and unchanged axes. Pixel expectations come from Pillow.
PORTRAIT_AND_BOUNDARY_SHAPES = (
    (263, 201), (229, 221), (848, 467),
    (287, 256), (288, 256), (289, 256),
    (223, 223), (224, 223), (225, 223), (225, 224), (224, 225),
    (300, 3), (301, 3),
)


def load_helper():
    spec = importlib.util.spec_from_file_location("tbps_gpu_production_test", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_tensor(rgb):
    resized = np.asarray(Image.fromarray(rgb).resize((224, 224), Image.Resampling.BICUBIC), np.float32)
    pixels = resized / 255.0
    pixels = (pixels - np.asarray([0.485, 0.456, 0.406], np.float32)) / np.asarray([0.229, 0.224, 0.225], np.float32)
    return np.ascontiguousarray(pixels.transpose(2, 0, 1))


def coefficient_resize(module, rgb):
    """Exercise production coefficients on the CPU; expected pixels use Pillow."""
    def axis(image, dimension):
        source_size = image.shape[dimension]
        if source_size == 224:
            return image
        bounds, weights = module._pillow_coefficients(source_size, 224)
        source = np.moveaxis(image, dimension, 0)
        output = np.empty((224, *source.shape[1:]), np.uint8)
        for index, (start, count) in enumerate(bounds):
            values = source[start:start + count].astype(np.int64)
            total = (values * weights[index, :count][:, None, None]).sum(axis=0)
            output[index] = np.clip((total + (1 << 21)) >> 22, 0, 255)
        return np.moveaxis(output, 0, dimension)
    vertical_first = module.TbpsGpuPreprocessor()._vertical_first(*rgb.shape[:2])
    for dimension in ((0, 1) if vertical_first else (1, 0)):
        rgb = axis(rgb, dimension)
    return rgb


def portrait_and_boundary_images():
    """Use encoded JPEGs as in the person-crop path, without a private dataset."""
    for height, width in PORTRAIT_AND_BOUNDARY_SHAPES:
        rgb = np.random.default_rng(height * 1000 + width).integers(
            0, 256, (height, width, 3), dtype=np.uint8)
        payload = io.BytesIO()
        Image.fromarray(rgb).save(payload, format="JPEG", quality=95)
        with Image.open(io.BytesIO(payload.getvalue())) as decoded:
            yield np.asarray(decoded.convert("RGB"))


class TbpsGpuPreprocessorCpuTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(HELPER.is_file(), "TBPS production GPU helper is missing")
        self.module = load_helper()

    def test_decode_preserves_original_pillow_rgb_conversion(self):
        processor = self.module.TbpsGpuPreprocessor()
        rng = np.random.default_rng(20)
        for fmt, channels in (("PNG", 3), ("PNG", 4), ("PNG", 1), ("JPEG", 3)):
            shape = (13, 17, channels) if channels != 1 else (13, 17)
            image = Image.fromarray(rng.integers(0, 256, shape, np.uint8))
            encoded = io.BytesIO()
            image.save(encoded, format=fmt)
            payload = encoded.getvalue()
            with Image.open(io.BytesIO(payload)) as decoded:
                expected = np.asarray(decoded.convert("RGB"))
            actual = processor.decode(payload, batch_index=2)
            np.testing.assert_array_equal(actual, expected)
            self.assertTrue(actual.flags.c_contiguous)

    def test_bad_encoded_and_decoded_inputs_are_rejected_before_cuda(self):
        processor = self.module.TbpsGpuPreprocessor()
        for payload in (None, bytearray(b"abc"), b"", b"bad image"):
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "batch index 3"):
                processor.decode(payload, batch_index=3)
        for rgb in (None, np.zeros((3, 4), np.uint8), np.zeros((3, 4, 4), np.uint8),
                    np.zeros((0, 4, 3), np.uint8), np.zeros((3, 4, 3), np.float32)):
            with self.subTest(shape=getattr(rgb, "shape", None)), self.assertRaises(ValueError):
                processor.preprocess(rgb)

    def test_22bit_coefficients_reproduce_pillow_pixels_across_axes_and_edges(self):
        for shape in ((19, 31), (224, 224), (224, 533), (537, 224), (539, 917), (1, 7), (401, 3)):
            with self.subTest(shape=shape):
                rgb = np.random.default_rng(sum(shape)).integers(0, 256, (*shape, 3), dtype=np.uint8)
                expected = np.asarray(Image.fromarray(rgb).resize((224, 224), Image.Resampling.BICUBIC))
                np.testing.assert_array_equal(coefficient_resize(self.module, rgb), expected)

    def test_portrait_jpegs_and_pass_order_boundaries_match_real_pillow(self):
        for rgb in portrait_and_boundary_images():
            with self.subTest(shape=rgb.shape, pillow=PILLOW_VERSION):
                expected = np.asarray(Image.fromarray(rgb).resize((224, 224), Image.Resampling.BICUBIC))
                np.testing.assert_array_equal(coefficient_resize(self.module, rgb), expected)

    def test_pass_order_retains_older_pillow_behavior_and_strict_boundaries(self):
        # Expected order for independently documented version transitions;
        # full pixel tests above compare the active version with real Pillow.
        cases = (
            ((263, 201), (False, False, False, True)),
            ((288, 256), (False, False, False, False)),
            ((289, 256), (False, False, False, True)),
            ((300, 3), (False, False, False, True)),
            ((301, 3), (False, True, True, True)),
            ((224, 1), (False, False, False, False)),
            ((223, 223), (False, False, False, False)),
            ((225, 224), (False, False, False, True)),
            ((224, 225), (False, False, False, False)),
        )
        for version_index, version in enumerate(("12.1.0", "12.2.0", "12.3.0", "13.0.0")):
            with patch.object(self.module, "PILLOW_VERSION", version):
                processor = self.module.TbpsGpuPreprocessor()
            for shape, expected in cases:
                with self.subTest(version=version, shape=shape):
                    self.assertEqual(processor._vertical_first(*shape), expected[version_index])


try:
    import cupy as CP
    if CP.cuda.runtime.getDeviceCount() < 1:
        CP = None
except (ImportError, RuntimeError):
    CP = None


@unittest.skipUnless(CP is not None, "requires CuPy and CUDA")
class TbpsGpuPreprocessorCudaTests(unittest.TestCase):
    def test_current_nondefault_stream_matches_pillow_pixels_and_fp32_tensor(self):
        module = load_helper()
        processor = module.TbpsGpuPreprocessor(cp_module=CP)
        images = [np.random.default_rng(sum(shape)).integers(0, 256, (*shape, 3), dtype=np.uint8)
                  for shape in ((19, 31), (224, 224), (224, 533), (537, 224), (539, 917), (1, 7), (401, 3))]
        images.extend(portrait_and_boundary_images())
        if os.environ.get("IMAGE_GPU_PARITY_DATASET"):
            sys.path.insert(0, str(ROOT))
            from tools.benchmark_image_gpu import sample_images
            images.extend(processor.decode(Path(sample["path"]).read_bytes())
                          for sample in sample_images(os.environ["IMAGE_GPU_PARITY_DATASET"]))
        stream = CP.cuda.Stream(non_blocking=True)
        with stream:
            for index, rgb in enumerate(images):
                with self.subTest(image=index, shape=rgb.shape):
                    output = processor.preprocess(rgb)
                    self.assertEqual(output.shape, (3, 224, 224))
                    self.assertEqual(output.dtype, CP.float32)
                    self.assertTrue(output.flags.c_contiguous)
                    actual = output.get(stream=stream, blocking=True)
                    np.testing.assert_allclose(actual, reference_tensor(rgb), rtol=0, atol=5e-7)
                    recovered = np.rint((actual * np.asarray([0.229, 0.224, 0.225], np.float32)[:, None, None]
                                        + np.asarray([0.485, 0.456, 0.406], np.float32)[:, None, None]) * 255)
                    expected = np.asarray(Image.fromarray(rgb).resize((224, 224), Image.Resampling.BICUBIC))
                    np.testing.assert_array_equal(recovered.astype(np.uint8), expected.transpose(2, 0, 1))

    def test_queued_inputs_remain_valid_through_coefficient_cache_eviction(self):
        processor = load_helper().TbpsGpuPreprocessor(cp_module=CP)
        stream = CP.cuda.Stream(non_blocking=True)
        with stream:
            first = np.random.default_rng(415).integers(0, 256, (17, 19, 3), dtype=np.uint8)
            expected = reference_tensor(first)
            result = processor.preprocess(first)
            del first
            for width in range(20, 150):
                processor.preprocess(np.full((9, width, 3), width % 256, np.uint8))
            actual = result.get(stream=stream, blocking=True)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
