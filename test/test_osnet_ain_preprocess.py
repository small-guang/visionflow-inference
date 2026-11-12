import importlib.util
import json
import sys
import types
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    ROOT
    / "model_repository"
    / "osnet_ain_x1_0_image_preprocess"
    / "1"
    / "model.py"
)


class FakeTensor:
    def __init__(self, value, name=""):
        self.value = value
        self.name = name

    def as_numpy(self):
        return self.value


class FakeInferenceResponse:
    def __init__(self, output_tensors=None, error=None):
        self.output_tensors = output_tensors or []
        self.error = error


class FakeTritonError(RuntimeError):
    pass


def fake_pb_utils():
    module = types.ModuleType("triton_python_backend_utils")
    module.Tensor = lambda name, value: FakeTensor(value, name)
    module.InferenceResponse = FakeInferenceResponse
    module.TritonError = FakeTritonError
    module.get_input_tensor_by_name = lambda request, name: request.get(name)
    module.get_output_config_by_name = lambda config, name: next(
        output for output in config["output"] if output["name"] == name
    )
    module.triton_string_to_numpy = lambda data_type: np.float32
    return module


def load_preprocessor():
    pb_utils = fake_pb_utils()
    spec = importlib.util.spec_from_file_location("osnet_ain_preprocess", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get("triton_python_backend_utils")
    sys.modules["triton_python_backend_utils"] = pb_utils
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("triton_python_backend_utils", None)
        else:
            sys.modules["triton_python_backend_utils"] = previous
    model = module.TritonPythonModel()
    model.initialize(
        {
            "model_config": json.dumps(
                {"output": [{"name": "image", "data_type": "TYPE_FP32"}]}
            )
        }
    )
    return model


def make_gradient_png():
    y, x = np.mgrid[:9, :7]
    image = np.stack(
        [(x * 31) % 256, (y * 27) % 256, ((x + y) * 19) % 256], axis=2
    ).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8)


def make_solid_png(width, height, mode="RGB"):
    buffer = BytesIO()
    Image.new(mode, (width, height)).save(buffer, format="PNG")
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8)


def convert_png_to_jpeg(encoded_png):
    with Image.open(BytesIO(encoded_png.tobytes())) as image:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=95)
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8)


def reference_torchreid_preprocess(encoded):
    with Image.open(BytesIO(encoded.tobytes())) as decoded:
        image = decoded.convert("RGB")
    image = image.resize((128, 256), Image.Resampling.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return np.ascontiguousarray(np.transpose((pixels - mean) / std, (2, 0, 1)))


class OsnetAinPreprocessTests(unittest.TestCase):
    def setUp(self):
        self.model = load_preprocessor()

    def test_batch_four_matches_pillow_reference(self):
        encoded = make_gradient_png()
        batch = np.repeat(encoded[None, :], 4, axis=0)
        response = self.model.execute([{"IMAGE_BYTES": FakeTensor(batch)}])[0]
        actual = response.output_tensors[0].as_numpy()
        expected = reference_torchreid_preprocess(encoded)
        self.assertEqual(actual.shape, (4, 3, 256, 128))
        np.testing.assert_allclose(actual, np.repeat(expected[None], 4, axis=0))

    def test_one_bad_request_does_not_abort_following_request(self):
        valid = make_gradient_png()[None, :]
        responses = self.model.execute(
            [
                {"IMAGE_BYTES": FakeTensor(np.zeros((1, 0), dtype=np.uint8))},
                {"IMAGE_BYTES": FakeTensor(valid)},
            ]
        )
        self.assertIsNotNone(responses[0].error)
        self.assertIsNone(responses[1].error)

    def test_accepts_batch_larger_than_four_offline(self):
        encoded = make_gradient_png()
        batch = np.repeat(encoded[None, :], 5, axis=0)
        response = self.model.execute([{"IMAGE_BYTES": FakeTensor(batch)}])[0]
        self.assertIsNone(response.error)
        self.assertEqual(
            response.output_tensors[0].as_numpy().shape,
            (5, 3, 256, 128),
        )

    def test_accepts_jpeg_batch_one(self):
        jpeg = convert_png_to_jpeg(make_gradient_png())
        response = self.model.execute(
            [{"IMAGE_BYTES": FakeTensor(jpeg[None, :])}]
        )[0]
        self.assertIsNone(response.error)
        self.assertEqual(
            response.output_tensors[0].as_numpy().shape,
            (1, 3, 256, 128),
        )

    def test_mixed_png_jpeg_batch_accepts_zero_tail_padding(self):
        png = make_gradient_png()
        jpeg = convert_png_to_jpeg(png)
        row_size = max(png.size, jpeg.size)
        batch = np.zeros((2, row_size), dtype=np.uint8)
        batch[0, : png.size] = png
        batch[1, : jpeg.size] = jpeg
        response = self.model.execute([{"IMAGE_BYTES": FakeTensor(batch)}])[0]
        self.assertIsNone(response.error)
        actual = response.output_tensors[0].as_numpy()
        expected = np.stack(
            [
                reference_torchreid_preprocess(png),
                reference_torchreid_preprocess(jpeg),
            ],
            axis=0,
        )
        np.testing.assert_allclose(actual, expected)

    def test_rejects_corrupt_encoded_bytes(self):
        corrupt = np.full((1, 32), 255, dtype=np.uint8)
        response = self.model.execute([{"IMAGE_BYTES": FakeTensor(corrupt)}])[0]
        self.assertIn("unable to decode", str(response.error))

    def test_rejects_wrong_input_rank(self):
        encoded = make_gradient_png()
        response = self.model.execute([{"IMAGE_BYTES": FakeTensor(encoded)}])[0]
        self.assertIn("must have shape", str(response.error))

    def test_rejects_encoded_image_larger_than_sixteen_mib(self):
        encoded = make_gradient_png()
        padded = np.pad(
            encoded,
            (0, 16 * 1024 * 1024 + 1 - encoded.size),
            mode="constant",
        )[None, :]
        response = self.model.execute([{"IMAGE_BYTES": FakeTensor(padded)}])[0]
        self.assertIn("encoded byte limit", str(response.error))

    def test_rejects_image_dimension_larger_than_8192(self):
        encoded = make_solid_png(8193, 1)[None, :]
        response = self.model.execute([{"IMAGE_BYTES": FakeTensor(encoded)}])[0]
        self.assertIn("dimensions exceed limit", str(response.error))

    def test_rejects_image_larger_than_8388608_pixels(self):
        encoded = make_solid_png(4096, 2049, mode="1")[None, :]
        response = self.model.execute([{"IMAGE_BYTES": FakeTensor(encoded)}])[0]
        self.assertIn("pixel count exceeds limit", str(response.error))


if __name__ == "__main__":
    unittest.main(verbosity=2)
