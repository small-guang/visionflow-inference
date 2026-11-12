import importlib.util
import re
import sys
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model_repository" / "adaface_preprocess" / "1" / "model.py"
REFERENCE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)
_MISSING = object()


class FakeTensor:
    def __init__(self, name, value):
        self.__name = name
        self.value = value

    def name(self):
        return self.__name

    def as_numpy(self):
        return self.value


class FakeRequest:
    def __init__(self, inputs):
        self.__inputs = tuple(inputs.values())

    def _input_tensor_by_name(self, name):
        return next(
            (tensor for tensor in self.__inputs if tensor.name() == name),
            None,
        )


class FakeTritonError(RuntimeError):
    pass


class FakeResponse:
    def __init__(self, output_tensors=None, error=None):
        if output_tensors is None:
            output_tensors = []
        if not isinstance(output_tensors, (list, tuple)):
            raise TypeError("output_tensors must be a list or tuple")
        if not all(isinstance(tensor, FakeTensor) for tensor in output_tensors):
            raise TypeError("output_tensors must contain only FakeTensor values")
        if error is not None and not isinstance(error, FakeTritonError):
            raise TypeError("error must be None or FakeTritonError")

        self.__output_tensors = list(output_tensors)
        self.__error = error

    def has_error(self):
        return self.__error is not None

    def error(self):
        return self.__error

    def output_tensors(self):
        return list(self.__output_tensors)


def _get_input_tensor_by_name(inference_request, name):
    if not isinstance(inference_request, FakeRequest):
        raise TypeError("inference_request must be a FakeRequest")
    return inference_request._input_tensor_by_name(name)


def load_preprocessor():
    pb_utils = types.ModuleType("triton_python_backend_utils")
    pb_utils.Tensor = FakeTensor
    pb_utils.InferenceResponse = FakeResponse
    pb_utils.TritonError = FakeTritonError
    pb_utils.get_input_tensor_by_name = _get_input_tensor_by_name

    previous_pb_utils = sys.modules.get("triton_python_backend_utils", _MISSING)
    sys.modules["triton_python_backend_utils"] = pb_utils
    try:
        spec = importlib.util.spec_from_file_location(
            "adaface_preprocess", MODEL_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous_pb_utils is _MISSING:
            sys.modules.pop("triton_python_backend_utils", None)
        else:
            sys.modules["triton_python_backend_utils"] = previous_pb_utils

    return module.TritonPythonModel()


def request(image, landmark=None):
    image_value = image
    if isinstance(image_value, np.ndarray) and image_value.ndim == 3:
        image_value = image_value[None, ...]
    landmark_value = np.array(
        REFERENCE if landmark is None else landmark,
        copy=True,
    )
    if landmark_value.ndim == 2 and landmark_value.shape == (5, 2):
        landmark_value = landmark_value[None, ...]
    return FakeRequest(
        {
            "RAW_IMAGE": FakeTensor("RAW_IMAGE", image_value),
            "LANDMARK": FakeTensor("LANDMARK", landmark_value),
        }
    )


class AdaFacePreprocessTests(unittest.TestCase):
    def setUp(self):
        self.model = load_preprocessor()

    def assert_success(
        self,
        response,
        expected_shape=(1, 3, 112, 112),
        expected_value=None,
    ):
        self.assertIsInstance(response, FakeResponse)
        self.assertFalse(response.has_error())
        output_tensors = response.output_tensors()
        self.assertEqual(len(output_tensors), 1)
        output_tensor = output_tensors[0]
        self.assertIsInstance(output_tensor, FakeTensor)
        self.assertEqual(output_tensor.name(), "input")
        actual = output_tensor.as_numpy()
        if expected_shape is not None:
            self.assertEqual(actual.shape, expected_shape)
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(actual.flags.c_contiguous)
        if expected_value is not None:
            np.testing.assert_allclose(
                actual,
                expected_value,
                rtol=0.0,
                atol=1e-5,
            )
        return actual

    def assert_error(self, response):
        self.assertIsInstance(response, FakeResponse)
        self.assertTrue(response.has_error())
        error = response.error()
        self.assertIsInstance(error, FakeTritonError)
        return error

    def test_finalize_message_is_stable(self):
        stdout = StringIO()

        with redirect_stdout(stdout):
            self.model.finalize()

        self.assertEqual(stdout.getvalue(), "adaface_preprocess finalized.\n")

    def test_input_validation_error_is_stable_and_not_logged(self):
        image = np.zeros((112, 112, 3), dtype=np.float32)
        stderr = StringIO()

        with redirect_stderr(stderr), self.assertNoLogs(
            "adaface_preprocess", level="ERROR"
        ):
            response = self.model.execute([request(image)])[0]

        error = self.assert_error(response)
        self.assertEqual(
            str(error),
            "INVALID_INPUT: batch index 0: RAW_IMAGE must have dtype uint8",
        )
        self.assertEqual(stderr.getvalue(), "")

    def test_internal_errors_are_redacted_logged_and_request_isolated(self):
        image = np.zeros((112, 112, 3), dtype=np.uint8)
        aligned = np.zeros((112, 112, 3), dtype=np.uint8)
        internal_path = r"C:\private\opencv\imgwarp.cpp:123"
        runtime_detail = "runtime boom at /srv/private/model.py:77"
        model_globals = self.model.execute.__globals__
        align_mock = mock.Mock(
            side_effect=[
                cv2.error(f"{internal_path}: assertion failed"),
                RuntimeError(runtime_detail),
                aligned,
            ]
        )

        with mock.patch.dict(
            model_globals, {"align_face": align_mock}
        ), self.assertLogs("adaface_preprocess", level="ERROR") as captured:
            responses = self.model.execute(
                [request(image), request(image), request(image)]
            )

        self.assertEqual(len(responses), 3)
        incident_ids = []
        for response in responses[:2]:
            error = self.assert_error(response)
            message = str(error)
            match = re.fullmatch(
                r"INTERNAL_ERROR: adaface_preprocess failed; "
                r"incident=([0-9a-f]{12})",
                message,
            )
            self.assertIsNotNone(match)
            incident_ids.append(match.group(1))
            for secret in (
                "opencv",
                "imgwarp",
                "assertion failed",
                "runtime boom",
                "/srv/private",
            ):
                self.assertNotIn(secret, message.lower())

        self.assertEqual(len(set(incident_ids)), 2)
        self.assertEqual(len(captured.records), 2)
        for request_index, (record, incident) in enumerate(
            zip(captured.records, incident_ids)
        ):
            self.assertIn(
                f"request_index={request_index}", record.getMessage()
            )
            self.assertIn(f"incident={incident}", record.getMessage())
            self.assertIsNotNone(record.exc_info)

        server_log = "\n".join(captured.output)
        self.assertIn(internal_path, server_log)
        self.assertIn(runtime_detail, server_log)
        self.assertNotIn("38.2946", server_log)
        self.assert_success(responses[2], expected_value=-1.0)

    def test_reports_each_missing_required_input(self):
        image = np.zeros((112, 112, 3), dtype=np.uint8)
        cases = [
            (
                "RAW_IMAGE",
                {"LANDMARK": FakeTensor("LANDMARK", REFERENCE.copy())},
            ),
            (
                "LANDMARK",
                {"RAW_IMAGE": FakeTensor("RAW_IMAGE", image)},
            ),
        ]

        for missing_name, inputs in cases:
            with self.subTest(missing_name=missing_name):
                response = self.model.execute([FakeRequest(inputs)])[0]

                error = self.assert_error(response)
                self.assertIn("missing", str(error).lower())
                self.assertIn(missing_name, str(error))

    def test_constant_bgr_is_normalized_in_bgr_order_without_rgb_swap(self):
        image = np.empty((112, 112, 3), dtype=np.uint8)
        image[:] = [10, 80, 240]

        responses = self.model.execute([request(image)])

        self.assertEqual(len(responses), 1)
        actual = self.assert_success(responses[0])
        expected_pixel = (
            np.array([10, 80, 240], dtype=np.float32) - 127.5
        ) / 127.5
        expected = np.broadcast_to(expected_pixel[:, None, None], actual.shape)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)

    def test_batch_four_matches_four_batch_one_references(self):
        images = np.stack(
            [
                np.full((112, 112, 3), color, dtype=np.uint8)
                for color in (
                    [10, 80, 240],
                    [220, 30, 90],
                    [40, 160, 70],
                    [180, 120, 20],
                )
            ],
            axis=0,
        )
        landmarks = np.stack(
            [
                REFERENCE + np.array(offset, dtype=np.float32)
                for offset in ((0, 0), (1, -1), (-2, 1), (2, 2))
            ],
            axis=0,
        )

        response = self.model.execute(
            [
                FakeRequest(
                    {
                        "RAW_IMAGE": FakeTensor("RAW_IMAGE", images),
                        "LANDMARK": FakeTensor("LANDMARK", landmarks),
                    }
                )
            ]
        )[0]
        actual = self.assert_success(
            response, expected_shape=(4, 3, 112, 112)
        )

        references = []
        for image, landmark in zip(images, landmarks):
            references.append(
                self.assert_success(
                    self.model.execute([request(image, landmark)])[0]
                )[0]
            )
        np.testing.assert_allclose(
            actual, np.stack(references, axis=0), rtol=0.0, atol=0.0
        )

    def test_bad_second_sample_is_indexed_and_next_request_isolated(self):
        image = np.zeros((112, 112, 3), dtype=np.uint8)
        images = np.stack([image, image], axis=0)
        landmarks = np.stack([REFERENCE, REFERENCE], axis=0)
        landmarks[1, 0, 0] = np.nan

        responses = self.model.execute(
            [
                FakeRequest(
                    {
                        "RAW_IMAGE": FakeTensor("RAW_IMAGE", images),
                        "LANDMARK": FakeTensor("LANDMARK", landmarks),
                    }
                ),
                request(image),
            ]
        )

        self.assertEqual(len(responses), 2)
        error = self.assert_error(responses[0])
        self.assertIn("batch index 1", str(error))
        self.assertIn("finite", str(error).lower())
        self.assert_success(responses[1])

    def test_rejects_empty_wrong_rank_and_mismatched_batches(self):
        image = np.zeros((112, 112, 3), dtype=np.uint8)
        cases = {
            "empty": (
                np.empty((0, 112, 112, 3), dtype=np.uint8),
                np.empty((0, 5, 2), dtype=np.float32),
                "non-empty",
            ),
            "image rank": (image, REFERENCE[None, ...], "[B,H,W,3]"),
            "landmark rank": (image[None, ...], REFERENCE, "[B,5,2]"),
            "batch mismatch": (
                np.stack([image, image]),
                REFERENCE[None, ...],
                "batch size",
            ),
        }
        for case_name, (images, landmarks, expected) in cases.items():
            with self.subTest(case=case_name):
                response = self.model.execute(
                    [
                        FakeRequest(
                            {
                                "RAW_IMAGE": FakeTensor("RAW_IMAGE", images),
                                "LANDMARK": FakeTensor(
                                    "LANDMARK", landmarks
                                ),
                            }
                        )
                    ]
                )[0]
                self.assertIn(expected, str(self.assert_error(response)))

    def test_rejects_float32_image(self):
        image = np.zeros((112, 112, 3), dtype=np.float32)

        responses = self.model.execute([request(image)])

        self.assertEqual(len(responses), 1)
        error = self.assert_error(responses[0])
        self.assertIn("uint8", str(error))

    def test_rejects_invalid_image_structure(self):
        cases = [
            ("not_ndarray", [1, 2, 3], "numpy array"),
            ("wrong_rank", np.zeros((112, 112), dtype=np.uint8), "[H,W,3]"),
            (
                "wrong_channel_count",
                np.zeros((112, 112, 1), dtype=np.uint8),
                "[H,W,3]",
            ),
            (
                "zero_width",
                np.zeros((112, 0, 3), dtype=np.uint8),
                "positive",
            ),
        ]

        for case_name, image, expected_text in cases:
            with self.subTest(case_name=case_name):
                response = self.model.execute([request(image)])[0]

                error = self.assert_error(response)
                self.assertIn(expected_text, str(error))

    def test_rejects_oversized_images_before_alignment(self):
        pixel = np.zeros(3, dtype=np.uint8)
        cases = [
            (
                "max_side",
                np.lib.stride_tricks.as_strided(
                    pixel,
                    shape=(8193, 1, 3),
                    strides=(0, 0, pixel.strides[0]),
                    writeable=False,
                ),
                "max side",
            ),
            (
                "max_pixels",
                np.lib.stride_tricks.as_strided(
                    pixel,
                    shape=(4097, 2048, 3),
                    strides=(0, 0, pixel.strides[0]),
                    writeable=False,
                ),
                "max pixels",
            ),
        ]
        model_globals = self.model.execute.__globals__

        for case_name, image, expected_text in cases:
            with self.subTest(case_name=case_name):
                blocked_align = mock.Mock(
                    side_effect=AssertionError(
                        "align_face must not run for an oversized image"
                    )
                )
                with mock.patch.dict(
                    model_globals, {"align_face": blocked_align}
                ):
                    response = self.model.execute([request(image)])[0]

                error = self.assert_error(response)
                self.assertIn(expected_text, str(error).lower())
                blocked_align.assert_not_called()

    def test_accepts_non_contiguous_image_within_resource_limits(self):
        backing = np.empty((112, 224, 3), dtype=np.uint8)
        backing[:] = [10, 80, 240]
        image = backing[:, ::2, :]
        self.assertFalse(image.flags.c_contiguous)

        response = self.model.execute([request(image)])[0]

        actual = self.assert_success(response)
        expected_pixel = (
            np.array([10, 80, 240], dtype=np.float32) - 127.5
        ) / 127.5
        expected = np.broadcast_to(expected_pixel[:, None, None], actual.shape)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)

    def test_rejects_flat_landmark_without_implicit_reshape(self):
        image = np.zeros((112, 112, 3), dtype=np.uint8)

        responses = self.model.execute(
            [request(image, REFERENCE.reshape(10))]
        )

        self.assertEqual(len(responses), 1)
        error = self.assert_error(responses[0])
        self.assertIn("[5,2]", str(error))

    def test_rejects_landmark_dtypes_other_than_float32(self):
        image = np.zeros((112, 112, 3), dtype=np.uint8)

        for dtype in (np.float64, np.int32):
            with self.subTest(dtype=dtype):
                landmark = REFERENCE.astype(dtype)
                response = self.model.execute(
                    [request(image, landmark)]
                )[0]

                error = self.assert_error(response)
                self.assertIn("float32", str(error))

    def test_rejects_non_finite_landmark(self):
        image = np.zeros((112, 112, 3), dtype=np.uint8)
        landmark = REFERENCE.copy()
        landmark[0, 0] = np.nan

        responses = self.model.execute([request(image, landmark)])

        self.assertEqual(len(responses), 1)
        error = self.assert_error(responses[0])
        self.assertIn("finite", str(error).lower())

    def test_rejects_implausible_landmarks_before_estimation(self):
        image = np.zeros((112, 112, 3), dtype=np.uint8)
        near_coincident = np.array(
            [
                [56.0, 56.0],
                [56.1, 56.0],
                [56.0, 56.1],
                [55.9, 56.0],
                [56.0, 55.9],
            ],
            dtype=np.float32,
        )
        extreme = REFERENCE.copy()
        extreme[0, 0] = np.float32(1e20)
        cases = [
            ("tiny", REFERENCE * np.float32(1e-20), "spread"),
            ("normalized", REFERENCE / np.float32(112.0), "spread"),
            ("near_coincident", near_coincident, "spread"),
            ("extreme", extreme, "margin"),
        ]
        transform_module = self.model.execute.__globals__["trans"]

        for case_name, landmark, expected_text in cases:
            with self.subTest(case_name=case_name):
                blocked_transform = mock.Mock(
                    side_effect=AssertionError(
                        "SimilarityTransform must not run for invalid landmarks"
                    )
                )
                with mock.patch.object(
                    transform_module,
                    "SimilarityTransform",
                    blocked_transform,
                ):
                    response = self.model.execute(
                        [request(image, landmark)]
                    )[0]

                error = self.assert_error(response)
                self.assertIn(expected_text, str(error).lower())
                blocked_transform.assert_not_called()

    def test_rejects_extreme_similarity_scale_before_warp(self):
        class FixedTransform:
            def __init__(self, scale):
                self.params = np.array(
                    [
                        [scale, 0.0, 0.0],
                        [0.0, scale, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )

            def estimate(self, source, destination):
                return True

        image = np.zeros((112, 112, 3), dtype=np.uint8)
        model_globals = self.model.execute.__globals__
        transform_module = model_globals["trans"]
        cv2_module = model_globals["cv2"]

        for scale in (1e-20, 1e20):
            with self.subTest(scale=scale):
                blocked_warp = mock.Mock(
                    side_effect=AssertionError(
                        "warpAffine must not run for an extreme scale"
                    )
                )
                with mock.patch.object(
                    transform_module,
                    "SimilarityTransform",
                    return_value=FixedTransform(scale),
                ), mock.patch.object(
                    cv2_module, "warpAffine", blocked_warp
                ):
                    response = self.model.execute([request(image)])[0]

                error = self.assert_error(response)
                self.assertIn("scale", str(error).lower())
                blocked_warp.assert_not_called()

    def test_accepts_plausible_landmarks_for_large_image(self):
        pixel = np.zeros(3, dtype=np.uint8)
        image = np.lib.stride_tricks.as_strided(
            pixel,
            shape=(2048, 2048, 3),
            strides=(0, 0, pixel.strides[0]),
            writeable=False,
        )
        landmark = REFERENCE * np.float32(16.0)
        aligned = np.zeros((112, 112, 3), dtype=np.uint8)
        cv2_module = self.model.execute.__globals__["cv2"]

        with mock.patch.object(
            cv2_module, "warpAffine", return_value=aligned
        ):
            response = self.model.execute([request(image, landmark)])[0]

        self.assert_success(response, expected_value=-1.0)

    def test_rejects_non_finite_alignment_matrix(self):
        class NonFiniteTransform:
            params = np.full((3, 3), np.nan, dtype=np.float64)

            def estimate(self, source, destination):
                return True

        image = np.zeros((112, 112, 3), dtype=np.uint8)
        transform_module = self.model.execute.__globals__["trans"]
        with mock.patch.object(
            transform_module,
            "SimilarityTransform",
            return_value=NonFiniteTransform(),
        ):
            response = self.model.execute([request(image)])[0]

        error = self.assert_error(response)
        self.assertIn("transform", str(error).lower())
        self.assertIn("finite", str(error).lower())

    def test_rejects_degenerate_landmarks_with_transform_error(self):
        image = np.zeros((112, 112, 3), dtype=np.uint8)
        degenerate_landmark = np.zeros((5, 2), dtype=np.float32)

        response = self.model.execute(
            [request(image, degenerate_landmark)]
        )[0]

        error = self.assert_error(response)
        self.assertIn("transform", str(error).lower())

    def test_requests_copy_landmarks_into_independent_buffers(self):
        image = np.zeros((112, 112, 3), dtype=np.uint8)
        explicit_landmark = REFERENCE.copy()

        first_request = request(image)
        second_request = request(image)
        explicit_request = request(image, explicit_landmark)

        first_landmark = _get_input_tensor_by_name(
            first_request, "LANDMARK"
        ).as_numpy()
        second_landmark = _get_input_tensor_by_name(
            second_request, "LANDMARK"
        ).as_numpy()
        copied_explicit_landmark = _get_input_tensor_by_name(
            explicit_request, "LANDMARK"
        ).as_numpy()
        self.assertFalse(np.shares_memory(first_landmark, REFERENCE))
        self.assertFalse(np.shares_memory(second_landmark, REFERENCE))
        self.assertFalse(np.shares_memory(first_landmark, second_landmark))
        self.assertFalse(
            np.shares_memory(copied_explicit_landmark, explicit_landmark)
        )
        first_landmark[0, 0, 0] = -1.0
        np.testing.assert_array_equal(second_landmark[0], REFERENCE)

    def test_bad_request_does_not_stop_later_good_request(self):
        bad_image = np.zeros((0, 112, 3), dtype=np.uint8)
        good_image = np.full((112, 112, 3), 127, dtype=np.uint8)

        responses = self.model.execute(
            [request(bad_image), request(good_image)]
        )

        self.assertEqual(len(responses), 2)
        self.assert_error(responses[0])
        expected_value = np.full(
            (1, 3, 112, 112),
            (127.0 - 127.5) / 127.5,
            dtype=np.float32,
        )
        self.assert_success(responses[1], expected_value=expected_value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
