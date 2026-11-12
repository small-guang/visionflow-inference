import re
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
from torchvision import transforms

from test.sixdrepnet_backend_test_utils import (
    FakeResponse,
    FakeTensor,
    FakeTritonError,
    load_backend,
    make_request,
)


class SixDRepNetPreprocessTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    MODEL_PATH = (
        ROOT
        / "model_repository"
        / "sixdrepnet_preprocess"
        / "1"
        / "model.py"
    )

    def setUp(self):
        self.module, self.model = load_backend(
            "sixdrepnet_preprocess", self.MODEL_PATH
        )
        self.request = lambda value: make_request(RAW_FACE_IMAGE=value)

    def assert_error(self, response, expected_message=None):
        self.assertIsInstance(response, FakeResponse)
        self.assertTrue(response.has_error())
        self.assertIsInstance(response.error(), FakeTritonError)
        if expected_message is not None:
            self.assertEqual(str(response.error()), expected_message)
        return response.error()

    def assert_success(self, response, batch_size=1):
        self.assertIsInstance(response, FakeResponse)
        self.assertFalse(response.has_error())
        tensors = response.output_tensors()
        self.assertEqual(len(tensors), 1)
        self.assertIsInstance(tensors[0], FakeTensor)
        self.assertEqual(tensors[0].name(), "PREPROCESSED_IMAGE")
        value = tensors[0].as_numpy()
        self.assertEqual(value.shape, (batch_size, 3, 224, 224))
        self.assertEqual(value.dtype, np.float32)
        self.assertTrue(value.flags.c_contiguous)
        return value

    def test_constant_bgr_becomes_imagenet_normalized_rgb(self):
        image = np.empty((1, 224, 224, 3), dtype=np.uint8)
        image[:] = [10, 80, 240]

        actual = self.assert_success(
            self.model.execute([self.request(image)])[0]
        )
        expected_pixel = (
            np.array([240, 80, 10], dtype=np.float32) / np.float32(255.0)
            - np.array([0.485, 0.456, 0.406], dtype=np.float32)
        ) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        expected = np.broadcast_to(
            expected_pixel[None, :, None, None], actual.shape
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)

    @staticmethod
    def strided_image(height, width):
        pixel = np.zeros(3, dtype=np.uint8)
        return np.lib.stride_tricks.as_strided(
            pixel,
            shape=(height, width, 3),
            strides=(0, 0, pixel.strides[0]),
            writeable=False,
        )

    @staticmethod
    def official_reference(image_bgr):
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        transform = transforms.Compose(
            [
                transforms.Resize(
                    224,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        return transform(Image.fromarray(rgb)).numpy()

    def test_resize_dimensions_use_short_edge_and_integer_floor(self):
        self.assertTrue(hasattr(self.module, "_validated_resize_dimensions"))
        cases = (
            ((100, 200), (448, 224)),
            ((200, 100), (224, 448)),
            ((224, 227), (227, 224)),
        )
        for (height, width), expected_target in cases:
            with self.subTest(height=height, width=width):
                self.assertEqual(
                    self.module._validated_resize_dimensions(height, width),
                    expected_target,
                )

    def test_matches_official_torchvision_pixels_for_real_images(self):
        cases = (
            ("downsample_landscape", (300, 600, 3)),
            ("downsample_portrait", (600, 300, 3)),
            ("upsample_landscape", (100, 200, 3)),
            ("upsample_portrait", (200, 100, 3)),
            ("odd_crop_margin", (224, 227, 3)),
        )
        for seed, (name, shape) in enumerate(cases):
            with self.subTest(name=name):
                image = np.random.default_rng(seed).integers(
                    0, 256, size=shape, dtype=np.uint8
                )
                expected = self.official_reference(image)
                actual = self.module.preprocess(image)
                np.testing.assert_array_equal(actual, expected)

    def test_pure_preprocess_returns_single_chw_image(self):
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        output = self.module.preprocess(image)
        self.assertEqual(output.shape, (3, 224, 224))
        self.assertEqual(output.dtype, np.float32)
        self.assertTrue(output.flags.c_contiguous)

    def test_batch_of_four_equals_four_single_item_requests(self):
        images = np.random.default_rng(42).integers(
            0, 256, size=(4, 224, 224, 3), dtype=np.uint8
        )
        batched = self.assert_success(
            self.model.execute([self.request(images)])[0], batch_size=4
        )
        singles = np.concatenate(
            [
                self.assert_success(
                    self.model.execute([self.request(images[index:index + 1])])[0]
                )
                for index in range(4)
            ],
            axis=0,
        )
        np.testing.assert_array_equal(batched, singles)

    def test_geometry_boundaries_are_accepted(self):
        self.assertEqual(getattr(self.module, "MAX_IMAGE_SIDE", None), 8192)
        self.assertEqual(getattr(self.module, "MAX_IMAGE_PIXELS", None), 8_388_608)
        self.assertEqual(getattr(self.module, "MAX_ASPECT_RATIO", None), 20)
        self.assertEqual(getattr(self.module, "MAX_RESIZED_LONG_EDGE", None), 8192)
        cases = (
            ((8192, 1024), (224, 1792)),
            ((4096, 2048), (224, 448)),
            ((20, 400), (4480, 224)),
        )
        for (height, width), expected_target in cases:
            with self.subTest(height=height, width=width):
                self.assertEqual(
                    self.module._validated_resize_dimensions(height, width),
                    expected_target,
                )
        with mock.patch.object(self.module, "MAX_ASPECT_RATIO", 100):
            self.assertEqual(
                self.module._validated_resize_dimensions(7, 256),
                (8192, 224),
            )

    def test_over_limit_geometry_rejected_and_later_request_succeeds(self):
        cases = (
            (
                "max_side",
                self.strided_image(8193, 400),
                "RAW_FACE_IMAGE maximum side must not exceed 8192",
            ),
            (
                "max_pixels",
                self.strided_image(4097, 2048),
                "RAW_FACE_IMAGE pixel count must not exceed 8388608",
            ),
            (
                "max_aspect_ratio",
                self.strided_image(20, 401),
                "RAW_FACE_IMAGE aspect ratio must not exceed 20:1",
            ),
        )
        good = np.zeros((1, 224, 224, 3), dtype=np.uint8)
        requests = [self.request(image[None, ...]) for _, image, _ in cases]
        requests.append(self.request(good))
        with mock.patch.object(
            self.module.Image.Image, "resize", autospec=True,
            side_effect=self.module.Image.Image.resize,
        ) as resize:
            responses = self.model.execute(requests)
        self.assertEqual(len(responses), len(requests))
        for response, (_, _, message) in zip(responses, cases):
            self.assert_error(
                response,
                "INVALID_INPUT: sixdrepnet_preprocess: batch index 0: "
                f"{message}",
            )
        self.assert_success(responses[-1])
        self.assertEqual(resize.call_count, 1)

    def test_validation_errors_reject_non_batch_contract(self):
        cases = (
            ("missing", make_request(), "missing required input RAW_FACE_IMAGE"),
            (
                "non_array", self.request([1, 2, 3]),
                "RAW_FACE_IMAGE must be a numpy array",
            ),
            (
                "float32", self.request(np.zeros((1, 10, 10, 3), dtype=np.float32)),
                "RAW_FACE_IMAGE must have dtype uint8",
            ),
            (
                "unbatched", self.request(np.zeros((10, 10, 3), dtype=np.uint8)),
                "RAW_FACE_IMAGE batch must have shape [B,H,W,3]",
            ),
            (
                "nchw", self.request(np.zeros((1, 3, 224, 224), dtype=np.uint8)),
                "RAW_FACE_IMAGE batch must have shape [B,H,W,3]",
            ),
            (
                "four_channels", self.request(np.zeros((1, 10, 10, 4), dtype=np.uint8)),
                "RAW_FACE_IMAGE batch must have shape [B,H,W,3]",
            ),
            (
                "empty_batch", self.request(np.zeros((0, 10, 10, 3), dtype=np.uint8)),
                "input batch must be non-empty",
            ),
            (
                "zero_height", self.request(np.zeros((1, 0, 10, 3), dtype=np.uint8)),
                "batch index 0: RAW_FACE_IMAGE height and width must be positive",
            ),
            (
                "zero_width", self.request(np.zeros((1, 10, 0, 3), dtype=np.uint8)),
                "batch index 0: RAW_FACE_IMAGE height and width must be positive",
            ),
        )
        for name, request, detail in cases:
            with self.subTest(name=name):
                self.assert_error(
                    self.model.execute([request])[0],
                    f"INVALID_INPUT: sixdrepnet_preprocess: {detail}",
                )

    def test_bad_second_sample_is_whole_request_error_with_exact_batch_index(self):
        images = np.zeros((2, 224, 224, 3), dtype=np.uint8)
        good = np.zeros((1, 224, 224, 3), dtype=np.uint8)
        good_value = self.module.preprocess(images[0])
        with mock.patch.object(
            self.module,
            "preprocess",
            side_effect=[
                good_value,
                self.module.InputValidationError(
                    "RAW_FACE_IMAGE height and width must be positive"
                ),
                self.module.preprocess(good[0]),
            ],
        ):
            responses = self.model.execute(
                [self.request(images), self.request(good)]
            )
        self.assertEqual(len(responses), 2)
        self.assert_error(
            responses[0],
            "INVALID_INPUT: sixdrepnet_preprocess: batch index 1: "
            "RAW_FACE_IMAGE height and width must be positive",
        )
        self.assert_success(responses[1])

    def test_internal_error_is_logged_redacted_with_request_and_batch_indices(self):
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        good_value = self.module.preprocess(image)
        private_detail = r"runtime boom at C:\private\opencv\resize.cpp:77"
        with mock.patch.object(
            self.module,
            "preprocess",
            side_effect=[good_value, RuntimeError(private_detail), good_value],
        ), self.assertLogs("sixdrepnet_preprocess", level="ERROR") as captured:
            responses = self.model.execute(
                [
                    self.request(np.zeros((2, 224, 224, 3), dtype=np.uint8)),
                    self.request(np.zeros((1, 224, 224, 3), dtype=np.uint8)),
                ]
            )
        self.assertEqual(len(responses), 2)
        error = self.assert_error(responses[0])
        match = re.fullmatch(
            r"INTERNAL_ERROR: sixdrepnet_preprocess failed; incident=([0-9a-f]{12})",
            str(error),
        )
        self.assertIsNotNone(match)
        incident = match.group(1)
        self.assertNotIn("private", str(error).lower())
        self.assertNotIn("opencv", str(error).lower())
        self.assert_success(responses[1])
        self.assertEqual(len(captured.records), 1)
        record = captured.records[0]
        self.assertIn("request_index=0", record.getMessage())
        self.assertIn("batch_index=1", record.getMessage())
        self.assertIn(f"incident={incident}", record.getMessage())
        self.assertIsNone(record.exc_info)
        self.assertNotIn(private_detail, captured.output[0])
        self.assertNotIn("private", captured.output[0].lower())
        self.assertNotIn("opencv", captured.output[0].lower())

    def test_pure_preprocess_rejects_non_array(self):
        with self.assertRaisesRegex(
            self.module.InputValidationError,
            r"^RAW_FACE_IMAGE must be a numpy array$",
        ):
            self.module.preprocess([1, 2, 3])

    def test_accepts_non_contiguous_input_and_returns_contiguous_fp32(self):
        backing = np.empty((1, 224, 448, 3), dtype=np.uint8)
        backing[:] = [10, 80, 240]
        images = backing[:, :, ::2, :]
        self.assertFalse(images.flags.c_contiguous)
        actual = self.assert_success(
            self.model.execute([self.request(images)])[0]
        )
        self.assertTrue(actual.flags.c_contiguous)


if __name__ == "__main__":
    unittest.main(verbosity=2)
