import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT / "model_repository"


def scaled_face_landmarks(height, width):
    reference = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    return reference * np.array(
        [width / 112.0, height / 112.0], dtype=np.float32
    )


def load_python_model(model_name):
    fake_pb_utils = types.ModuleType("triton_python_backend_utils")

    class FakeTensor:
        def __init__(self, name, value):
            self.name = name
            self.value = value

        def as_numpy(self):
            return self.value

    class FakeInferenceResponse:
        def __init__(self, output_tensors):
            self.output_tensors = output_tensors

    fake_pb_utils.Tensor = FakeTensor
    fake_pb_utils.InferenceResponse = FakeInferenceResponse
    fake_pb_utils.get_input_tensor_by_name = lambda request, name: request[name]

    module_path = REPOSITORY / model_name / "1" / "model.py"
    spec = importlib.util.spec_from_file_location(
        f"{model_name}_color_test", module_path
    )
    module = importlib.util.module_from_spec(spec)

    previous_pb_utils = sys.modules.get("triton_python_backend_utils")
    sys.modules["triton_python_backend_utils"] = fake_pb_utils
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_pb_utils is None:
            sys.modules.pop("triton_python_backend_utils", None)
        else:
            sys.modules["triton_python_backend_utils"] = previous_pb_utils

    return module, fake_pb_utils


class ClipImageColorOrderTests(unittest.TestCase):
    def test_bgr_decode_is_converted_to_rgb_before_normalization(self):
        module, _ = load_python_model("clip_image_preprocess")
        model = module.TritonPythonModel()
        model.target_size = (224, 224)
        model.mean = np.array(
            [0.48145466, 0.4578275, 0.40821073], dtype=np.float32
        )
        model.std = np.array(
            [0.26862954, 0.26130258, 0.27577711], dtype=np.float32
        )
        model.output_dtype = np.float32

        bgr = np.full((8, 12, 3), [10, 80, 240], dtype=np.uint8)
        encoded_ok, encoded = cv2.imencode(".png", bgr)
        self.assertTrue(encoded_ok)

        actual = model._preprocess(encoded.tobytes())

        rgb_pixel = np.array([240, 80, 10], dtype=np.float32) / 255.0
        expected_pixel = (rgb_pixel - model.mean) / model.std
        expected = np.broadcast_to(
            expected_pixel.reshape(3, 1, 1), (3, 224, 224)
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


class FaceQaColorOrderTests(unittest.TestCase):
    def test_rank_one_alignment_matches_modelscope_umeyama_branch(self):
        module, _ = load_python_model("face_qa_pre")
        landmarks = np.array(
            [[50, 20], [40, 20], [30, 20], [20, 20], [10, 20]],
            dtype=np.float32,
        )
        expected = np.array(
            [
                [-0.32888092, 1.21879704, 41.51664658],
                [-1.21879704, -0.32888092, 115.04231071],
            ],
            dtype=np.float64,
        )

        actual = module._alignment_matrix(landmarks)

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)

    def test_five_landmarks_align_face_before_normalization(self):
        module, fake_pb_utils = load_python_model("face_qa_pre")
        y_coords, x_coords = np.indices((180, 160), dtype=np.uint8)
        bgr = np.stack(
            (x_coords, y_coords, np.full_like(x_coords, 200)), axis=-1
        )
        reference = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        landmarks = reference + np.array([20.0, 30.0], dtype=np.float32)
        request = {
            "RAW_FACE_IMAGE": fake_pb_utils.Tensor(
                "RAW_FACE_IMAGE", bgr[None, ...]
            ),
            "LANDMARK": fake_pb_utils.Tensor(
                "LANDMARK", landmarks[None, ...]
            ),
        }

        response = module.TritonPythonModel().execute([request])[0]
        actual = response.output_tensors[0].as_numpy()

        expected_rgb = np.array([200.0, 90.0, 70.0], dtype=np.float32)
        expected = (expected_rgb / 255.0 - 0.5) / 0.5
        self.assertEqual(actual.shape, (1, 3, 112, 112))
        np.testing.assert_allclose(
            actual[0, :, 60, 50], expected, rtol=0.0, atol=2.0 / 255.0
        )

    def test_bgr_input_is_converted_to_rgb_before_normalization(self):
        module, fake_pb_utils = load_python_model("face_qa_pre")
        bgr = np.full((112, 112, 3), [10, 80, 240], dtype=np.uint8)
        original = bgr.copy()
        request = {
            "RAW_FACE_IMAGE": fake_pb_utils.Tensor(
                "RAW_FACE_IMAGE", bgr[None, ...]
            ),
            "LANDMARK": fake_pb_utils.Tensor(
                "LANDMARK", scaled_face_landmarks(112, 112)[None, ...]
            ),
        }

        response = module.TritonPythonModel().execute([request])[0]
        actual = response.output_tensors[0].as_numpy()

        rgb_pixel = np.array([240, 80, 10], dtype=np.float32)
        expected_pixel = (rgb_pixel / 255.0 - 0.5) / 0.5
        expected = np.broadcast_to(
            expected_pixel.reshape(1, 3, 1, 1), (1, 3, 112, 112)
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue(actual.flags.c_contiguous)
        np.testing.assert_array_equal(bgr, original)

    def test_oversized_image_is_rejected_before_opencv(self):
        module, _ = load_python_model("face_qa_pre")
        pixel = np.zeros((1, 1, 3), dtype=np.uint8)
        cases = (
            ((8193, 1, 3), "max side"),
            ((4097, 2049, 3), "max pixels"),
        )
        for shape, expected in cases:
            with self.subTest(shape=shape):
                image = np.broadcast_to(pixel, shape)
                with mock.patch.object(
                    module.cv2,
                    "cvtColor",
                    side_effect=AssertionError("OpenCV must not be called"),
                ) as cvt_color:
                    with self.assertRaisesRegex(
                        module.InputValidationError, expected
                    ):
                        module.preprocess(
                            image,
                            scaled_face_landmarks(shape[0], shape[1]),
                        )
                cvt_color.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
