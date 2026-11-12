import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    ROOT
    / "model_repository"
    / "face_keypoint_2d106_pre"
    / "1"
    / "model.py"
)


class FakeTensor:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def as_numpy(self):
        return self.value


class FakeInferenceResponse:
    def __init__(self, output_tensors=None, error=None):
        self.output_tensors = output_tensors or []
        self.error = error


class FakeTritonError(RuntimeError):
    pass


def load_preprocessor():
    pb_utils = types.ModuleType("triton_python_backend_utils")
    pb_utils.Tensor = FakeTensor
    pb_utils.InferenceResponse = FakeInferenceResponse
    pb_utils.TritonError = FakeTritonError
    pb_utils.get_input_tensor_by_name = lambda request, name: request[name]

    skimage = types.ModuleType("skimage")
    skimage.transform = types.ModuleType("skimage.transform")

    replacements = {
        "triton_python_backend_utils": pb_utils,
        "skimage": skimage,
        "skimage.transform": skimage.transform,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    try:
        spec = importlib.util.spec_from_file_location(
            "face_keypoint_2d106_preprocess", MODEL_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module

    def fake_transform_face(
        data, center, output_size, scale, rotation=0
    ):
        crop = np.empty((output_size, output_size, 3), dtype=np.uint8)
        crop[:] = data[0, 0]
        matrix = np.array(
            [
                [scale, 0.0, center[0]],
                [0.0, scale, center[1]],
            ],
            dtype=np.float64,
        )
        return crop, matrix

    module.transform_face = fake_transform_face
    model = module.TritonPythonModel()
    model.initialize({})
    return model


class FaceKeypoint2d106PreprocessTests(unittest.TestCase):
    def test_bgr_input_is_swapped_once_to_rgb_model_tensor(self):
        model = load_preprocessor()
        image_bgr = np.empty((192, 192, 3), dtype=np.uint8)
        image_bgr[:] = [10, 20, 30]
        request = {
            "RAW_IMAGE": FakeTensor("RAW_IMAGE", image_bgr[None, ...]),
            "BBOX": FakeTensor(
                "BBOX", np.array([[0, 0, 192, 192]], dtype=np.float32)
            ),
        }

        response = model.execute([request])[0]

        self.assertIsNone(response.error)
        actual = response.output_tensors[0].as_numpy()[0, :, 96, 96]
        np.testing.assert_array_equal(actual, [30.0, 20.0, 10.0])

    def test_batch_four_matches_four_batch_one_references(self):
        model = load_preprocessor()
        images = np.stack(
            [
                np.full((192, 192, 3), [10, 20, 30], dtype=np.uint8),
                np.full((192, 192, 3), [40, 50, 60], dtype=np.uint8),
                np.full((192, 192, 3), [70, 80, 90], dtype=np.uint8),
                np.full((192, 192, 3), [100, 110, 120], dtype=np.uint8),
            ]
        )
        boxes = np.array(
            [
                [0, 0, 192, 192],
                [16, 12, 160, 180],
                [24, 30, 180, 150],
                [32, 18, 152, 174],
            ],
            dtype=np.float32,
        )
        original_images = images.copy()
        original_boxes = boxes.copy()

        response = model.execute(
            [
                {
                    "RAW_IMAGE": FakeTensor("RAW_IMAGE", images),
                    "BBOX": FakeTensor("BBOX", boxes),
                }
            ]
        )[0]

        self.assertIsNone(response.error)
        actual_images = response.output_tensors[0].as_numpy()
        actual_matrices = response.output_tensors[1].as_numpy()
        references = []
        reference_matrices = []
        for image, box in zip(images, boxes):
            single = model.execute(
                [
                    {
                        "RAW_IMAGE": FakeTensor(
                            "RAW_IMAGE", image[None, ...]
                        ),
                        "BBOX": FakeTensor("BBOX", box[None, ...]),
                    }
                ]
            )[0]
            self.assertIsNone(single.error)
            references.append(single.output_tensors[0].as_numpy()[0])
            reference_matrices.append(single.output_tensors[1].as_numpy()[0])

        self.assertEqual(actual_images.shape, (4, 3, 192, 192))
        self.assertEqual(actual_matrices.shape, (4, 2, 3))
        self.assertEqual(actual_images.dtype, np.float32)
        self.assertEqual(actual_matrices.dtype, np.float64)
        self.assertTrue(np.isfinite(actual_images).all())
        self.assertTrue(np.isfinite(actual_matrices).all())
        self.assertTrue(actual_images.flags.c_contiguous)
        self.assertTrue(actual_matrices.flags.c_contiguous)
        expected_matrices = np.array(
            [
                [[2.0 / 3.0, 0.0, 96.0], [0.0, 2.0 / 3.0, 96.0]],
                [
                    [192.0 / (168.0 * 1.5), 0.0, 88.0],
                    [0.0, 192.0 / (168.0 * 1.5), 96.0],
                ],
                [
                    [192.0 / (156.0 * 1.5), 0.0, 102.0],
                    [0.0, 192.0 / (156.0 * 1.5), 90.0],
                ],
                [
                    [192.0 / (156.0 * 1.5), 0.0, 92.0],
                    [0.0, 192.0 / (156.0 * 1.5), 96.0],
                ],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(
            actual_matrices, expected_matrices, rtol=0.0, atol=1e-7
        )
        self.assertFalse(
            np.array_equal(actual_matrices[0], actual_matrices[1])
        )
        self.assertFalse(np.array_equal(actual_images[0], actual_images[1]))
        np.testing.assert_array_equal(actual_images, np.stack(references))
        np.testing.assert_array_equal(
            actual_matrices, np.stack(reference_matrices)
        )
        np.testing.assert_array_equal(images, original_images)
        np.testing.assert_array_equal(boxes, original_boxes)

    def test_invalid_second_box_is_indexed_and_next_request_isolated(self):
        model = load_preprocessor()
        image = np.zeros((192, 192, 3), dtype=np.uint8)
        bad_request = {
            "RAW_IMAGE": FakeTensor(
                "RAW_IMAGE", np.stack([image, image])
            ),
            "BBOX": FakeTensor(
                "BBOX",
                np.array(
                    [[0, 0, 192, 192], [10, 10, 10, 80]],
                    dtype=np.float32,
                ),
            ),
        }
        good_request = {
            "RAW_IMAGE": FakeTensor("RAW_IMAGE", image[None, ...]),
            "BBOX": FakeTensor(
                "BBOX", np.array([[0, 0, 192, 192]], dtype=np.float32)
            ),
        }

        responses = model.execute([bad_request, good_request])

        self.assertEqual(len(responses), 2)
        self.assertIsNotNone(responses[0].error)
        self.assertIn("batch index 1", str(responses[0].error))
        self.assertIsNone(responses[1].error)

    def test_bbox_batch_requires_float32(self):
        image = np.zeros((192, 192, 3), dtype=np.uint8)
        response = load_preprocessor().execute(
            [
                {
                    "RAW_IMAGE": FakeTensor(
                        "RAW_IMAGE", image[None, ...]
                    ),
                    "BBOX": FakeTensor(
                        "BBOX",
                        np.array(
                            [[0, 0, 192, 192]], dtype=np.float64
                        ),
                    ),
                }
            ]
        )[0]

        self.assertIsNotNone(response.error)
        self.assertIn("BBOX", str(response.error))
        self.assertIn("float32", str(response.error))

    def test_second_box_rejects_unsafe_geometry(self):
        image = np.zeros((192, 192, 3), dtype=np.uint8)
        valid = np.array([0, 0, 192, 192], dtype=np.float32)
        cases = (
            (
                np.array([10, 10, 10.5, 80], dtype=np.float32),
                "at least 1",
            ),
            (
                np.array([-100, 10, -20, 80], dtype=np.float32),
                "margin",
            ),
        )
        for invalid, expected in cases:
            with self.subTest(expected=expected):
                response = load_preprocessor().execute(
                    [
                        {
                            "RAW_IMAGE": FakeTensor(
                                "RAW_IMAGE", np.stack([image, image])
                            ),
                            "BBOX": FakeTensor(
                                "BBOX", np.stack([valid, invalid])
                            ),
                        }
                    ]
                )[0]

                self.assertIsNotNone(response.error)
                message = str(response.error)
                self.assertIn("batch index 1", message)
                self.assertIn(expected, message)

    def test_oversized_image_is_rejected_before_transform(self):
        model = load_preprocessor()
        model_globals = model.execute.__globals__
        preprocess = model_globals["preprocess"]
        error_type = model_globals["InputValidationError"]
        pixel = np.zeros((1, 1, 3), dtype=np.uint8)
        bbox = np.array([0, 0, 1, 1], dtype=np.float32)
        cases = (
            ((8193, 1, 3), "max side"),
            ((4097, 2049, 3), "max pixels"),
        )
        for shape, expected in cases:
            with self.subTest(shape=shape):
                image = np.broadcast_to(pixel, shape)
                transform = mock.Mock(
                    side_effect=AssertionError("transform must not be called")
                )
                with mock.patch.dict(
                    model_globals, {"transform_face": transform}
                ):
                    with self.assertRaisesRegex(error_type, expected):
                        preprocess(
                            image,
                            bbox,
                            input_size=192,
                            input_mean=0.0,
                            input_std=1.0,
                        )
                transform.assert_not_called()

    def test_rejects_wrong_rank_empty_and_mismatched_batches(self):
        image = np.zeros((192, 192, 3), dtype=np.uint8)
        box = np.array([0, 0, 192, 192], dtype=np.float32)
        cases = (
            (image, box[None, ...], "[B,H,W,3]"),
            (image[None, ...], box, "[B,4]"),
            (
                np.empty((0, 192, 192, 3), dtype=np.uint8),
                np.empty((0, 4), dtype=np.float32),
                "non-empty",
            ),
            (np.stack([image, image]), box[None, ...], "batch size"),
        )
        for images, boxes, expected in cases:
            with self.subTest(expected=expected):
                response = load_preprocessor().execute(
                    [
                        {
                            "RAW_IMAGE": FakeTensor("RAW_IMAGE", images),
                            "BBOX": FakeTensor("BBOX", boxes),
                        }
                    ]
                )[0]
                self.assertIsNotNone(response.error)
                self.assertIn(expected, str(response.error))


if __name__ == "__main__":
    unittest.main(verbosity=2)
