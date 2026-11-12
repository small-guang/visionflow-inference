"""Offline tests for the manual AdaFace Triton HTTP smoke client."""

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

try:
    from test import test_adaface_image as client
except ImportError:
    client = None


class _FakeInferInput:
    instances = []

    def __init__(self, name, shape, datatype):
        self.name = name
        self.shape = tuple(shape)
        self.datatype = datatype
        self.data = None
        type(self).instances.append(self)

    def set_data_from_numpy(self, value):
        self.data = value


class _FakeRequestedOutput:
    instances = []

    def __init__(self, name):
        self.name = name
        type(self).instances.append(self)


class _FakeResult:
    def __init__(self, embedding):
        self.embedding = embedding
        self.requested_names = []

    def as_numpy(self, name):
        self.requested_names.append(name)
        return self.embedding


class _FakeInferenceServerClient:
    instances = []
    ready = True
    embedding = None
    infer_error = None
    close_error = None

    def __init__(self, url):
        self.url = url
        self.ready_names = []
        self.infer_calls = []
        self.close_calls = 0
        self.closed = False
        self.result = _FakeResult(type(self).embedding)
        type(self).instances.append(self)

    def is_model_ready(self, model_name):
        self.ready_names.append(model_name)
        return type(self).ready

    def infer(self, model_name, inputs, outputs):
        self.infer_calls.append(
            {
                "model_name": model_name,
                "inputs": inputs,
                "outputs": outputs,
            }
        )
        if type(self).infer_error is not None:
            raise type(self).infer_error
        return self.result

    def close(self):
        self.close_calls += 1
        self.closed = True
        if type(self).close_error is not None:
            raise type(self).close_error


class _FakeHttpClient:
    InferInput = _FakeInferInput
    InferRequestedOutput = _FakeRequestedOutput
    InferenceServerClient = _FakeInferenceServerClient


class AdaFaceImageModuleTests(unittest.TestCase):
    def test_smoke_client_module_exists(self):
        self.assertIsNotNone(client, "missing test.test_adaface_image smoke client")


@unittest.skipIf(client is None, "AdaFace smoke client is not implemented")
class AdaFaceImageClientTests(unittest.TestCase):
    def setUp(self):
        _FakeInferInput.instances.clear()
        _FakeRequestedOutput.instances.clear()
        _FakeInferenceServerClient.instances.clear()
        _FakeInferenceServerClient.ready = True
        _FakeInferenceServerClient.infer_error = None
        _FakeInferenceServerClient.close_error = None
        embedding = np.zeros((1, 512), dtype=np.float32)
        embedding[0, 0] = 1.0
        _FakeInferenceServerClient.embedding = embedding

    def write_landmarks(self, directory, value):
        path = Path(directory) / "landmarks.json"
        if isinstance(value, str):
            path.write_text(value, encoding="utf-8")
        else:
            path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_parse_args_requires_paths_and_defaults_to_local_http(self):
        args = client.parse_args(
            ["--image", "face.png", "--landmarks", "face.json"]
        )

        self.assertEqual(args.image, "face.png")
        self.assertEqual(args.landmarks, "face.json")
        self.assertEqual(args.url, "localhost:8000")

    def test_parse_args_requires_each_input_path_independently(self):
        cases = {
            "missing image": (["--landmarks", "face.json"], "--image"),
            "missing landmarks": (["--image", "face.png"], "--landmarks"),
        }

        for case_name, (arguments, missing_option) in cases.items():
            with self.subTest(case=case_name):
                stderr = StringIO()
                with mock.patch("sys.stderr", stderr):
                    with self.assertRaises(SystemExit) as captured:
                        client.parse_args(arguments)

                self.assertEqual(captured.exception.code, 2)
                self.assertIn(missing_option, stderr.getvalue())

    def test_help_documents_landmark_order_and_url_format(self):
        stdout = StringIO()

        with redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as captured:
                client.parse_args(["--help"])

        self.assertEqual(captured.exception.code, 0)
        normalized_help = " ".join(stdout.getvalue().split())
        landmark_order = (
            "left eye, right eye, nose tip, left mouth corner, "
            "right mouth corner"
        )
        self.assertIn(landmark_order, normalized_help)
        self.assertIn("host:port[/base]", normalized_help)
        self.assertIn("without a URL scheme", normalized_help)
        normalized_docstring = " ".join((client.__doc__ or "").split())
        self.assertIn(landmark_order, normalized_docstring)
        self.assertIn("host:port[/base]", normalized_docstring)
        self.assertIn("without a URL scheme", normalized_docstring)

    def test_main_prints_summary_without_the_embedding_vector(self):
        image = np.zeros((8, 9, 3), dtype=np.uint8)
        landmarks = np.zeros((5, 2), dtype=np.float32)
        embedding = np.zeros(512, dtype=np.float32)
        embedding[:2] = [0.6, 0.8]
        stdout = StringIO()

        with mock.patch.object(
            client,
            "load_inputs",
            return_value=(image, landmarks),
        ), mock.patch.object(
            client,
            "run_inference",
            return_value=embedding,
        ), redirect_stdout(stdout):
            exit_code = client.main(
                ["--image", "face.png", "--landmarks", "face.json"]
            )

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn(f"model: {client.MODEL_NAME}\n", output)
        self.assertIn("image_shape: (8, 9, 3)\n", output)
        self.assertIn("embedding_shape: (512,)\n", output)
        self.assertIn("embedding_dtype: float32\n", output)
        self.assertIn("embedding_norm: 1.000000\n", output)
        self.assertNotIn("\nembedding: ", output)
        self.assertNotIn("[0.6, 0.8", output)

    def test_load_inputs_reads_color_image_without_swapping_bgr(self):
        bgr = np.empty((3, 4, 3), dtype=np.uint8)
        bgr[:] = [10, 80, 240]
        landmark_values = [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ]

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "face.png"
            landmarks_path = self.write_landmarks(directory, landmark_values)
            with mock.patch.object(cv2, "imread", return_value=bgr) as read_mock:
                image, landmarks = client.load_inputs(image_path, landmarks_path)

        read_mock.assert_called_once_with(str(image_path), cv2.IMREAD_COLOR)
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(image.shape, (3, 4, 3))
        self.assertTrue(image.flags.c_contiguous)
        np.testing.assert_array_equal(image[0, 0], [10, 80, 240])
        self.assertEqual(landmarks.dtype, np.float32)
        self.assertEqual(landmarks.shape, (5, 2))
        self.assertTrue(landmarks.flags.c_contiguous)
        np.testing.assert_allclose(landmarks, landmark_values, rtol=0.0, atol=1e-5)

    def test_load_inputs_reports_an_undecodable_image(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "missing.png"
            landmarks_path = self.write_landmarks(
                directory,
                [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]],
            )
            with mock.patch.object(cv2, "imread", return_value=None):
                with self.assertRaisesRegex(
                    FileNotFoundError,
                    r"unable to decode.*missing\.png",
                ):
                    client.load_inputs(image_path, landmarks_path)

    def test_load_inputs_rejects_non_strict_or_non_fp32_landmarks(self):
        cases = {
            "wrong shape": [[0, 0]] * 4,
            "boolean coordinate": [[True, 0]] + [[0, 0]] * 4,
            "non-finite JSON constant": "[[NaN, 0], [0, 0], [0, 0], [0, 0], [0, 0]]",
            "outside FP32 range": [[1e100, 0]] + [[0, 0]] * 4,
        }
        image = np.zeros((8, 8, 3), dtype=np.uint8)

        for case_name, value in cases.items():
            with self.subTest(case=case_name), tempfile.TemporaryDirectory() as directory:
                image_path = Path(directory) / "face.png"
                landmarks_path = self.write_landmarks(directory, value)
                with mock.patch.object(cv2, "imread", return_value=image):
                    with self.assertRaisesRegex(
                        ValueError,
                        r"LANDMARK.*finite FP32.*\[5,2\]",
                    ):
                        client.load_inputs(image_path, landmarks_path)

    def test_run_inference_checks_readiness_and_wires_exact_tensor_contract(self):
        image = np.empty((8, 9, 3), dtype=np.uint8)
        image[:] = [10, 80, 240]
        landmarks = np.arange(10, dtype=np.float32).reshape(5, 2)

        embedding = client.run_inference(
            "triton.example:8000",
            image,
            landmarks,
            httpclient_module=_FakeHttpClient,
        )

        np.testing.assert_array_equal(
            embedding,
            _FakeInferenceServerClient.embedding[0],
        )
        self.assertEqual(len(_FakeInferenceServerClient.instances), 1)
        fake_client = _FakeInferenceServerClient.instances[0]
        self.assertTrue(fake_client.closed)
        self.assertEqual(fake_client.close_calls, 1)
        self.assertEqual(fake_client.url, "triton.example:8000")
        self.assertEqual(fake_client.ready_names, [client.MODEL_NAME])
        self.assertEqual(len(fake_client.infer_calls), 1)
        infer_call = fake_client.infer_calls[0]
        self.assertEqual(infer_call["model_name"], "adaface_recognition_pipeline")
        self.assertEqual(len(infer_call["inputs"]), 2)
        raw_image, landmark = infer_call["inputs"]
        self.assertEqual(
            (raw_image.name, raw_image.shape, raw_image.datatype),
            ("RAW_IMAGE", (1, *image.shape), "UINT8"),
        )
        self.assertEqual(
            (landmark.name, landmark.shape, landmark.datatype),
            ("LANDMARK", (1, 5, 2), "FP32"),
        )
        self.assertEqual(raw_image.data.shape, (1, *image.shape))
        self.assertEqual(landmark.data.shape, (1, 5, 2))
        np.testing.assert_array_equal(raw_image.data[0], image)
        np.testing.assert_array_equal(landmark.data[0], landmarks)
        self.assertEqual(len(infer_call["outputs"]), 1)
        self.assertEqual(infer_call["outputs"][0].name, "EMBEDDING")
        self.assertEqual(fake_client.result.requested_names, ["EMBEDDING"])

    def test_run_inference_rejects_invalid_images_before_http(self):
        landmarks = np.zeros((5, 2), dtype=np.float32)
        cases = {
            "wrong dtype": (
                np.zeros((8, 8, 3), dtype=np.float32),
                "UINT8",
            ),
            "two dimensions": (
                np.zeros((8, 8), dtype=np.uint8),
                r"\[H,W,3\]",
            ),
            "four channels": (
                np.zeros((8, 8, 4), dtype=np.uint8),
                r"\[H,W,3\]",
            ),
        }

        for case_name, (image, error_text) in cases.items():
            with self.subTest(case=case_name):
                with self.assertRaisesRegex(ValueError, error_text):
                    client.run_inference(
                        "localhost:8000",
                        image,
                        landmarks,
                        httpclient_module=_FakeHttpClient,
                    )

        self.assertEqual(_FakeInferenceServerClient.instances, [])

    def test_run_inference_rejects_invalid_landmarks_before_http(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        cases = {
            "wrong dtype": np.zeros((5, 2), dtype=np.float64),
            "wrong shape": np.zeros((10,), dtype=np.float32),
            "not finite": np.full((5, 2), np.nan, dtype=np.float32),
        }

        for case_name, landmarks in cases.items():
            with self.subTest(case=case_name):
                with self.assertRaisesRegex(
                    ValueError,
                    r"LANDMARK.*finite FP32.*\[5,2\]",
                ):
                    client.run_inference(
                        "localhost:8000",
                        image,
                        landmarks,
                        httpclient_module=_FakeHttpClient,
                    )

        self.assertEqual(_FakeInferenceServerClient.instances, [])

    def test_run_inference_makes_non_contiguous_inputs_contiguous(self):
        image_backing = np.empty((8, 18, 3), dtype=np.uint8)
        image_backing[:] = [10, 80, 240]
        image = image_backing[:, ::2, :]
        landmark_backing = np.arange(20, dtype=np.float32).reshape(5, 4)
        landmarks = landmark_backing[:, ::2]
        expected_image = image.copy()
        expected_landmarks = landmarks.copy()
        self.assertFalse(image.flags.c_contiguous)
        self.assertFalse(landmarks.flags.c_contiguous)

        client.run_inference(
            "localhost:8000",
            image,
            landmarks,
            httpclient_module=_FakeHttpClient,
        )

        fake_client = _FakeInferenceServerClient.instances[0]
        infer_call = fake_client.infer_calls[0]
        raw_image, landmark = infer_call["inputs"]
        self.assertTrue(raw_image.data.flags.c_contiguous)
        self.assertTrue(landmark.data.flags.c_contiguous)
        np.testing.assert_array_equal(raw_image.data[0], expected_image)
        np.testing.assert_array_equal(landmark.data[0], expected_landmarks)

    def test_run_inference_refuses_an_unready_model_before_infer(self):
        _FakeInferenceServerClient.ready = False
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        landmarks = np.zeros((5, 2), dtype=np.float32)

        with self.assertRaisesRegex(
            RuntimeError,
            r"not ready.*adaface_recognition_pipeline",
        ):
            client.run_inference(
                "localhost:8000",
                image,
                landmarks,
                httpclient_module=_FakeHttpClient,
            )

        fake_client = _FakeInferenceServerClient.instances[0]
        self.assertEqual(fake_client.infer_calls, [])
        self.assertTrue(fake_client.closed)
        self.assertEqual(fake_client.close_calls, 1)

    def test_run_inference_closes_the_client_after_infer_raises(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        landmarks = np.zeros((5, 2), dtype=np.float32)
        infer_error = RuntimeError("inference transport failed")
        _FakeInferenceServerClient.infer_error = infer_error

        with self.assertRaises(RuntimeError) as captured:
            client.run_inference(
                "localhost:8000",
                image,
                landmarks,
                httpclient_module=_FakeHttpClient,
            )

        self.assertIs(captured.exception, infer_error)
        fake_client = _FakeInferenceServerClient.instances[0]
        self.assertTrue(fake_client.closed)
        self.assertEqual(fake_client.close_calls, 1)

    def test_run_inference_propagates_close_failure_after_success(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        landmarks = np.zeros((5, 2), dtype=np.float32)
        close_error = RuntimeError("client close failed")
        _FakeInferenceServerClient.close_error = close_error

        with self.assertRaises(RuntimeError) as captured:
            client.run_inference(
                "localhost:8000",
                image,
                landmarks,
                httpclient_module=_FakeHttpClient,
            )

        self.assertIs(captured.exception, close_error)
        fake_client = _FakeInferenceServerClient.instances[0]
        self.assertTrue(fake_client.closed)
        self.assertEqual(fake_client.close_calls, 1)

    def test_run_inference_preserves_infer_error_when_close_also_fails(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        landmarks = np.zeros((5, 2), dtype=np.float32)
        infer_error = RuntimeError("inference transport failed")
        close_error = RuntimeError("client close failed")
        _FakeInferenceServerClient.infer_error = infer_error
        _FakeInferenceServerClient.close_error = close_error

        with self.assertRaises(RuntimeError) as captured:
            client.run_inference(
                "localhost:8000",
                image,
                landmarks,
                httpclient_module=_FakeHttpClient,
            )

        self.assertIs(captured.exception, infer_error)
        self.assertTrue(
            any(
                "close" in note.lower()
                for note in getattr(infer_error, "__notes__", ())
            )
        )
        fake_client = _FakeInferenceServerClient.instances[0]
        self.assertTrue(fake_client.closed)
        self.assertEqual(fake_client.close_calls, 1)

    def test_validate_embedding_rejects_bad_server_outputs(self):
        cases = {
            "missing": (None, "does not contain"),
            "unbatched": (np.zeros(512, dtype=np.float32), "shape"),
            "wrong batch": (np.zeros((2, 512), dtype=np.float32), "shape"),
            "wrong width": (np.zeros((1, 511), dtype=np.float32), "shape"),
            "wrong dtype": (np.zeros((1, 512), dtype=np.float64), "FP32"),
            "not finite": (
                np.full((1, 512), np.nan, dtype=np.float32),
                "finite",
            ),
            "not normalized": (
                np.zeros((1, 512), dtype=np.float32),
                "L2-normalized",
            ),
        }

        for case_name, (embedding, message) in cases.items():
            with self.subTest(case=case_name):
                with self.assertRaisesRegex((RuntimeError, AssertionError), message):
                    client.validate_embedding(embedding)

    def test_run_inference_validates_each_bad_server_response(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        landmarks = np.zeros((5, 2), dtype=np.float32)
        cases = {
            "missing": (None, RuntimeError, "does not contain"),
            "wrong shape": (
                np.zeros(512, dtype=np.float32),
                AssertionError,
                "shape",
            ),
            "wrong dtype": (
                np.zeros((1, 512), dtype=np.float64),
                AssertionError,
                "FP32",
            ),
            "not finite": (
                np.full((1, 512), np.nan, dtype=np.float32),
                AssertionError,
                "finite",
            ),
            "not normalized": (
                np.zeros((1, 512), dtype=np.float32),
                AssertionError,
                "L2-normalized",
            ),
        }

        for case_name, (embedding, error_type, message) in cases.items():
            with self.subTest(case=case_name):
                _FakeInferenceServerClient.embedding = embedding

                with self.assertRaisesRegex(error_type, message):
                    client.run_inference(
                        "localhost:8000",
                        image,
                        landmarks,
                        httpclient_module=_FakeHttpClient,
                    )

                fake_client = _FakeInferenceServerClient.instances[-1]
                self.assertEqual(len(fake_client.infer_calls), 1)
                self.assertTrue(fake_client.closed)
                self.assertEqual(fake_client.close_calls, 1)

    def test_missing_tritonclient_dependency_has_an_actionable_error(self):
        missing = ModuleNotFoundError("No module named 'tritonclient'")
        missing.name = "tritonclient"

        with mock.patch.object(
            client.importlib,
            "import_module",
            side_effect=missing,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                r"tritonclient\[http\].*install",
            ):
                client.load_httpclient()


if __name__ == "__main__":
    unittest.main(verbosity=2)
