import asyncio
import importlib.util
import inspect
import json
import logging
import re as stdlib_re
import sys
import types
import unittest
from io import BytesIO
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODEL_REPOSITORY = ROOT / "model_repository"
EXPLICIT_BATCH_SIZE = 4

PYTHON_EXECUTABLE_MODELS = {
    "adaface_preprocess",
    "clip_image_preprocess",
    "clip_text_preprocess",
    "face_keypoint_2d106_post",
    "face_keypoint_2d106_pre",
    "face_qa_postprocess",
    "face_qa_pre",
    "face_rec_pre",
    "osnet_ain_x1_0_image_preprocess",
    "sbs_r50_ibn_image_preprocess",
    "sixdrepnet_postprocess",
    "sixdrepnet_preprocess",
    "TBPS_clip_image_preprocess",
    "TBPS_clip_text_preprocess",
    "yolo26m_pose_postprocess",
    "yolo26m_pose_preprocess",
    "yolo26s_seg_postprocess",
    "yolo26s_seg_preprocess",
    "yolov8n_face_postprocess",
    "yolov8n_face_preprocess",
}


class FakeTensor:
    def __init__(self, name, value=None):
        if value is None:
            value = name
            name = ""
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


class FakeTritonModelException(RuntimeError):
    pass


def make_fake_pb_utils():
    module = types.ModuleType("triton_python_backend_utils")
    module.Tensor = FakeTensor
    module.InferenceResponse = FakeInferenceResponse
    module.TritonError = FakeTritonError
    module.TritonModelException = FakeTritonModelException
    module.get_input_tensor_by_name = lambda request, name: request.get(name)
    module.get_output_config_by_name = lambda config, name: next(
        (
            output
            for output in config.get("output", [])
            if output.get("name") == name
        ),
        None,
    )
    module.triton_string_to_numpy = lambda data_type: {
        "TYPE_FP32": np.float32,
        "TYPE_INT64": np.int64,
    }[data_type]
    return module


def model_config_json_from_repository(model_name):
    """Derive the initialize payload from the real repository config."""
    from test.test_batch_model_repository import (
        _scalar,
        _tensor_contracts,
        parse_config,
    )

    parsed = parse_config(MODEL_REPOSITORY / model_name / "config.pbtxt")

    def tensors(field_name):
        return [
            {
                "name": name,
                "data_type": data_type,
                "dims": list(dims),
            }
            for name, data_type, dims in _tensor_contracts(parsed, field_name)
        ]

    return json.dumps(
        {
            "name": _scalar(parsed, "name"),
            "max_batch_size": _scalar(parsed, "max_batch_size"),
            "input": tensors("input"),
            "output": tensors("output"),
        }
    )


class _FakeRegexPattern:
    def findall(self, text):
        return stdlib_re.findall(r"[A-Za-z]+|[0-9]|[^\sA-Za-z0-9]+", text)


def _make_fake_regex():
    module = types.ModuleType("regex")
    module.IGNORECASE = stdlib_re.IGNORECASE
    module.compile = lambda pattern, flags=0: _FakeRegexPattern()
    module.sub = stdlib_re.sub
    module.findall = lambda pattern, text: pattern.findall(text)
    return module


def load_python_model(model_name):
    model_dir = MODEL_REPOSITORY / model_name / "1"
    module_path = model_dir / "model.py"
    module_name = f"{model_name}_batch_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)

    replacements = {
        "triton_python_backend_utils": make_fake_pb_utils(),
    }
    if model_name == "face_keypoint_2d106_pre":
        fake_skimage = types.ModuleType("skimage")
        fake_transform = types.ModuleType("skimage.transform")
        fake_skimage.transform = fake_transform
        replacements["skimage"] = fake_skimage
        replacements["skimage.transform"] = fake_transform
    if model_name == "TBPS_clip_text_preprocess":
        fake_ftfy = types.ModuleType("ftfy")
        fake_ftfy.fix_text = lambda text: text
        replacements["ftfy"] = fake_ftfy
        replacements["regex"] = _make_fake_regex()

    previous_modules = {name: sys.modules.get(name) for name in replacements}
    previous_path = list(sys.path)
    sys.modules.update(replacements)
    sys.path.insert(0, str(model_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous_path
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    model = module.TritonPythonModel()
    output_names = {
        "clip_image_preprocess": ["image"],
        "clip_text_preprocess": ["INPUT_IDS"],
        "TBPS_clip_image_preprocess": ["image"],
        "TBPS_clip_text_preprocess": [],
        "sbs_r50_ibn_image_preprocess": ["image"],
        "osnet_ain_x1_0_image_preprocess": ["image"],
        "face_rec_pre": ["processed_input"],
        "face_qa_pre": ["QA_PREPROCESSED_IMAGE"],
        "face_qa_postprocess": ["quality_score"],
        "face_keypoint_2d106_pre": [
            "PREPROCESSED_IMAGE",
            "AFFINE_MATRIX",
        ],
        "face_keypoint_2d106_post": ["LANDMARKS_106"],
        "sixdrepnet_preprocess": ["PREPROCESSED_IMAGE"],
        "sixdrepnet_postprocess": ["HEAD_POSE"],
        "yolov8n_face_preprocess": [
            "PREPROCESSED_IMAGE",
            "LETTERBOX_META",
        ],
        "yolo26m_pose_preprocess": [
            "PREPROCESSED_IMAGE",
            "LETTERBOX_META",
        ],
        "yolo26s_seg_preprocess": [
            "PREPROCESSED_IMAGE",
            "LETTERBOX_META",
        ],
        "yolov8n_face_postprocess": [
            "DETECTIONS",
            "NUM_DETECTIONS",
        ],
        "yolo26m_pose_postprocess": [
            "DETECTIONS",
            "KEYPOINTS",
            "NUM_DETECTIONS",
        ],
    }[model_name]
    if model_name == "clip_text_preprocess":
        model.initialize(
            {"model_config": model_config_json_from_repository(model_name)}
        )
    elif hasattr(model, "initialize"):
        output_config = [
            {
                "name": name,
                "data_type": "TYPE_INT64" if name != "image" else "TYPE_FP32",
            }
            for name in output_names
        ]
        model.initialize({"model_config": json.dumps({"output": output_config})})
    if hasattr(module, "logger"):
        module.logger.disabled = True
    if model_name == "face_keypoint_2d106_pre":
        module.transform_face = lambda data, *args, **kwargs: (
            data.copy(),
            np.eye(2, 3, dtype=np.float64),
        )
    return model


def run_execute(model, requests):
    result = model.execute(requests)
    if inspect.isawaitable(result):
        return asyncio.run(result)
    return result


def response_output(response, name):
    return next(
        tensor.as_numpy()
        for tensor in response.output_tensors
        if tensor.name == name
    )


def make_encoded_images():
    y, x = np.mgrid[:11, :7]
    rgb = np.stack(
        [(x * 31) % 256, (y * 19) % 256, ((x + y) * 23) % 256], axis=2
    ).astype(np.uint8)

    png_buffer = BytesIO()
    Image.fromarray(rgb, mode="RGB").save(png_buffer, format="PNG")
    png = np.frombuffer(png_buffer.getvalue(), dtype=np.uint8)

    jpeg_buffer = BytesIO()
    Image.fromarray(rgb[::-1], mode="RGB").save(
        jpeg_buffer, format="JPEG", quality=95
    )
    jpeg = np.frombuffer(jpeg_buffer.getvalue(), dtype=np.uint8)
    return png, jpeg


def make_encoded_image(format_name, size=(7, 11), **save_options):
    width, height = size
    y, x = np.mgrid[:height, :width]
    rgb = np.stack(
        [(x * 31) % 256, (y * 19) % 256, ((x + y) * 23) % 256], axis=2
    ).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(rgb, mode="RGB").save(
        buffer, format=format_name, **save_options
    )
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8)


def pad_encoded_rows(*rows):
    row_size = max(row.size for row in rows)
    batch = np.zeros((len(rows), row_size), dtype=np.uint8)
    for index, row in enumerate(rows):
        batch[index, : row.size] = row
    return batch


def encoded_image_batch(model_name, *rows):
    if model_name == "osnet_ain_x1_0_image_preprocess":
        return pad_encoded_rows(*rows)
    return np.array([[row.tobytes()] for row in rows], dtype=object)


ENCODED_IMAGE_MODELS = (
    "TBPS_clip_image_preprocess",
    "sbs_r50_ibn_image_preprocess",
    "osnet_ain_x1_0_image_preprocess",
)


class PythonExecutableInventoryTests(unittest.TestCase):
    def test_backend_python_default_configs_match_batch_four_inventory(self):
        from test.test_batch_model_repository import parse_config

        actual = set()
        for config_path in MODEL_REPOSITORY.glob("*/config.pbtxt"):
            parsed = parse_config(config_path)
            if parsed.get("backend") == ["python"]:
                self.assertNotIn("default_model_filename", parsed)
                self.assertTrue(
                    (config_path.parent / "1" / "model.py").is_file(),
                    config_path.parent.name,
                )
                actual.add(config_path.parent.name)

        self.assertEqual(actual, PYTHON_EXECUTABLE_MODELS)


class SixdrepnetPythonRegistrationTests(unittest.TestCase):
    def test_sixdrepnet_python_models_load_with_their_registered_outputs(self):
        for model_name in ("sixdrepnet_preprocess", "sixdrepnet_postprocess"):
            with self.subTest(model=model_name):
                self.assertIsNotNone(load_python_model(model_name))


FACE_REFERENCE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class FaceRecognitionPreprocessBatchTests(unittest.TestCase):
    def test_batch_four_matches_four_batch_one_references(self):
        images = np.stack(
            [
                np.full((112, 112, 3), value, dtype=np.uint8)
                for value in (32, 80, 144, 192)
            ]
        )
        landmarks = np.repeat(
            FACE_REFERENCE[None, ...], EXPLICIT_BATCH_SIZE, axis=0
        )
        model = load_python_model("face_rec_pre")

        response = run_execute(
            model,
            [
                {
                    "raw_image": FakeTensor("raw_image", images),
                    "landmark": FakeTensor("landmark", landmarks),
                }
            ],
        )[0]
        self.assertIsNone(response.error)
        actual = response_output(response, "processed_input")

        references = []
        for image, landmark in zip(images, landmarks):
            single = run_execute(
                model,
                [
                    {
                        "raw_image": FakeTensor(
                            "raw_image", image[None, ...]
                        ),
                        "landmark": FakeTensor(
                            "landmark", landmark[None, ...]
                        ),
                    }
                ],
            )[0]
            self.assertIsNone(single.error)
            references.append(response_output(single, "processed_input")[0])

        self.assertEqual(
            actual.shape, (EXPLICIT_BATCH_SIZE, 3, 112, 112)
        )
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(actual.flags.c_contiguous)
        self.assertTrue(np.isfinite(actual).all())
        self.assertFalse(np.allclose(actual[0], actual[1]))
        np.testing.assert_allclose(
            actual, np.stack(references), rtol=0.0, atol=0.0
        )

    def test_bad_second_landmark_is_indexed_and_requests_are_isolated(self):
        image = np.zeros((112, 112, 3), dtype=np.uint8)
        images = np.stack([image, image])
        landmarks = np.stack(
            [FACE_REFERENCE, np.zeros((5, 2), dtype=np.float32)]
        )
        model = load_python_model("face_rec_pre")

        responses = run_execute(
            model,
            [
                {
                    "raw_image": FakeTensor("raw_image", images),
                    "landmark": FakeTensor("landmark", landmarks),
                },
                {
                    "raw_image": FakeTensor(
                        "raw_image", image[None, ...]
                    ),
                    "landmark": FakeTensor(
                        "landmark", FACE_REFERENCE[None, ...]
                    ),
                },
            ],
        )

        self.assertEqual(len(responses), 2)
        self.assertIsNotNone(responses[0].error)
        self.assertIn("batch index 1", str(responses[0].error))
        self.assertIsNone(responses[1].error)

    def test_internal_failure_on_second_sample_is_redacted_and_isolated(self):
        image = np.zeros((112, 112, 3), dtype=np.uint8)
        images = np.stack([image, image])
        landmarks = np.stack([FACE_REFERENCE, FACE_REFERENCE])
        model = load_python_model("face_rec_pre")
        model_globals = model.execute.__globals__
        logger = model_globals["logger"]
        logger.disabled = False
        original_norm_crop = model_globals["norm_crop"]
        call_count = 0
        secret = r"SECRET at C:\private\imgwarp.cpp:77"

        def fail_second(*args, **kwargs):
            nonlocal call_count
            current = call_count
            call_count += 1
            if current == 1:
                raise RuntimeError(secret)
            return original_norm_crop(*args, **kwargs)

        with mock.patch.dict(
            model_globals, {"norm_crop": fail_second}
        ), self.assertLogs("face_rec_pre", level="ERROR") as captured:
            responses = run_execute(
                model,
                [
                    {
                        "raw_image": FakeTensor("raw_image", images),
                        "landmark": FakeTensor("landmark", landmarks),
                    },
                    {
                        "raw_image": FakeTensor(
                            "raw_image", image[None, ...]
                        ),
                        "landmark": FakeTensor(
                            "landmark", FACE_REFERENCE[None, ...]
                        ),
                    },
                ],
            )

        self.assertEqual(len(responses), 2)
        self.assertIsNotNone(responses[0].error)
        message = str(responses[0].error)
        incident_match = stdlib_re.fullmatch(
            r"INTERNAL_ERROR: face_rec_pre failed; incident=([0-9a-f]{12})",
            message,
        )
        self.assertIsNotNone(incident_match)
        self.assertNotIn("SECRET", message)
        self.assertNotIn("private", message)
        self.assertIsNone(responses[1].error)
        self.assertEqual(len(captured.records), 1)
        record = captured.records[0]
        self.assertIn("request_index=0", record.getMessage())
        self.assertIn("batch_index=1", record.getMessage())
        self.assertIn(
            f"incident={incident_match.group(1)}", record.getMessage()
        )
        self.assertIsNotNone(record.exc_info)
        self.assertIn(secret, "\n".join(captured.output))

    def test_second_sample_rejects_unsafe_landmark_geometry(self):
        image = np.zeros((112, 112, 3), dtype=np.uint8)
        images = np.stack([image, image])
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
        cases = (
            (FACE_REFERENCE * np.float32(1e6), "margin"),
            (near_coincident, "spread"),
        )
        for unsafe, expected in cases:
            with self.subTest(expected=expected):
                landmarks = np.stack([FACE_REFERENCE, unsafe])
                response = run_execute(
                    load_python_model("face_rec_pre"),
                    [
                        {
                            "raw_image": FakeTensor(
                                "raw_image", images
                            ),
                            "landmark": FakeTensor(
                                "landmark", landmarks
                            ),
                        }
                    ],
                )[0]

                self.assertIsNotNone(response.error)
                message = str(response.error)
                self.assertTrue(message.startswith("INVALID_INPUT:"))
                self.assertIn("batch index 1", message)
                self.assertIn(expected, message.lower())

    def test_second_sample_rejects_unsafe_alignment_transform(self):
        class FixedTransform:
            def __init__(self, matrix):
                self.params = matrix

            def estimate(self, source, destination):
                return True

        image = np.zeros((112, 112, 3), dtype=np.uint8)
        images = np.stack([image, image])
        landmarks = np.stack([FACE_REFERENCE, FACE_REFERENCE])
        identity = np.eye(3, dtype=np.float64)
        cases = (
            (np.full((3, 3), np.nan, dtype=np.float64), "finite"),
            (
                np.array(
                    [
                        [1e20, 0.0, 0.0],
                        [0.0, 1e20, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                ),
                "scale",
            ),
        )
        for unsafe_matrix, expected in cases:
            with self.subTest(expected=expected):
                model = load_python_model("face_rec_pre")
                transform_module = model.execute.__globals__["trans"]
                with mock.patch.object(
                    transform_module,
                    "SimilarityTransform",
                    side_effect=[
                        FixedTransform(identity),
                        FixedTransform(unsafe_matrix),
                    ],
                ):
                    response = run_execute(
                        model,
                        [
                            {
                                "raw_image": FakeTensor(
                                    "raw_image", images
                                ),
                                "landmark": FakeTensor(
                                    "landmark", landmarks
                                ),
                            }
                        ],
                    )[0]

                self.assertIsNotNone(response.error)
                message = str(response.error)
                self.assertTrue(message.startswith("INVALID_INPUT:"))
                self.assertIn("batch index 1", message)
                self.assertIn(expected, message.lower())

    def test_rejects_empty_wrong_rank_and_mismatched_batches(self):
        image = np.zeros((112, 112, 3), dtype=np.uint8)
        cases = (
            (
                np.empty((0, 112, 112, 3), dtype=np.uint8),
                np.empty((0, 5, 2), dtype=np.float32),
                "non-empty",
            ),
            (image, FACE_REFERENCE[None, ...], "[B,H,W,3]"),
            (image[None, ...], FACE_REFERENCE, "[B,5,2]"),
            (
                np.stack([image, image]),
                FACE_REFERENCE[None, ...],
                "batch size",
            ),
        )
        for images, landmarks, expected in cases:
            with self.subTest(expected=expected):
                response = run_execute(
                    load_python_model("face_rec_pre"),
                    [
                        {
                            "raw_image": FakeTensor(
                                "raw_image", images
                            ),
                            "landmark": FakeTensor(
                                "landmark", landmarks
                            ),
                        }
                    ],
                )[0]
                self.assertIsNotNone(response.error)
                self.assertIn(expected, str(response.error))


class ClipStringImageBatchTests(unittest.TestCase):
    def test_batch_four_string_values_match_batch_one_references(self):
        first, second = make_encoded_images()
        values = np.asarray(
            [
                [first.tobytes()],
                [second.tobytes()],
                [first.tobytes()],
                [second.tobytes()],
            ],
            dtype=object,
        )
        model = load_python_model("clip_image_preprocess")
        responses = run_execute(
            model, [{"IMAGE_BYTES": FakeTensor("IMAGE_BYTES", values)}]
        )
        self.assertEqual(len(responses), 1)
        self.assertIsNone(responses[0].error)
        actual = response_output(responses[0], "image")

        references = []
        for value in values:
            response = run_execute(
                model,
                [
                    {
                        "IMAGE_BYTES": FakeTensor(
                            "IMAGE_BYTES", value[None, :]
                        )
                    }
                ],
            )[0]
            self.assertIsNone(response.error)
            references.append(response_output(response, "image")[0])
        self.assertEqual(
            actual.shape, (EXPLICIT_BATCH_SIZE, 3, 224, 224)
        )
        np.testing.assert_allclose(
            actual, np.stack(references, axis=0), rtol=0.0, atol=0.0
        )

    def test_bad_second_string_sample_is_indexed_and_requests_are_isolated(self):
        valid, _ = make_encoded_images()
        invalid = b"SECRET_ENCODED_PAYLOAD"
        bad_values = np.asarray(
            [[valid.tobytes()], [invalid]], dtype=object
        )
        good_values = np.asarray([[valid.tobytes()]], dtype=object)
        responses = run_execute(
            load_python_model("clip_image_preprocess"),
            [
                {"IMAGE_BYTES": FakeTensor("IMAGE_BYTES", bad_values)},
                {"IMAGE_BYTES": FakeTensor("IMAGE_BYTES", good_values)},
            ],
        )
        self.assertEqual(len(responses), 2)
        self.assertIsNotNone(responses[0].error)
        self.assertIn("batch index 1", str(responses[0].error))
        self.assertNotIn("SECRET_ENCODED_PAYLOAD", str(responses[0].error))
        self.assertIsNone(responses[1].error)

    def test_string_input_must_have_shape_batch_by_one(self):
        valid, _ = make_encoded_images()
        invalid_values = (
            np.asarray([valid.tobytes()], dtype=object),
            np.asarray(
                [[valid.tobytes(), valid.tobytes()]], dtype=object
            ),
            np.empty((0, 1), dtype=object),
        )
        for values in invalid_values:
            with self.subTest(shape=values.shape):
                response = run_execute(
                    load_python_model("clip_image_preprocess"),
                    [{"IMAGE_BYTES": FakeTensor("IMAGE_BYTES", values)}],
                )[0]
                self.assertIsNotNone(response.error)


class EncodedImageBatchTests(unittest.TestCase):
    def test_tbps_sample_helper_accepts_encoded_bytes_and_returns_chw(self):
        encoded, _ = make_encoded_images()
        model = load_python_model("TBPS_clip_image_preprocess")
        actual = model._preprocess_image(encoded.tobytes())
        self.assertEqual(actual.shape, (3, 224, 224))

    def test_batch_four_matches_four_batch_one_references(self):
        first, second = make_encoded_images()
        unpadded_rows = (first, second, first, second)
        for model_name in ENCODED_IMAGE_MODELS:
            with self.subTest(model=model_name):
                model = load_python_model(model_name)
                batch = encoded_image_batch(model_name, *unpadded_rows)
                responses = run_execute(
                    model, [{"IMAGE_BYTES": FakeTensor("IMAGE_BYTES", batch)}]
                )
                self.assertEqual(len(responses), 1)
                self.assertIsNone(responses[0].error)
                actual = response_output(responses[0], "image")

                references = []
                for row in unpadded_rows:
                    response = run_execute(
                        model,
                        [
                            {
                                "IMAGE_BYTES": FakeTensor(
                                    "IMAGE_BYTES", encoded_image_batch(model_name, row)
                                )
                            }
                        ],
                    )[0]
                    self.assertIsNone(response.error)
                    references.append(response_output(response, "image")[0])
                expected = np.stack(references, axis=0)

                self.assertEqual(actual.shape[0], EXPLICIT_BATCH_SIZE)
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

    def test_bad_sample_is_indexed_and_does_not_abort_next_request(self):
        valid, _ = make_encoded_images()
        width = max(valid.size, 32)
        bad_batch = np.zeros((2, width), dtype=np.uint8)
        bad_batch[0, : valid.size] = valid
        bad_batch[1, :32] = 255
        good_batch = pad_encoded_rows(valid)

        for model_name in ENCODED_IMAGE_MODELS:
            with self.subTest(model=model_name):
                model = load_python_model(model_name)
                responses = run_execute(
                    model,
                    [
                        {"IMAGE_BYTES": FakeTensor("IMAGE_BYTES", encoded_image_batch(model_name, *bad_batch))},
                        {"IMAGE_BYTES": FakeTensor("IMAGE_BYTES", encoded_image_batch(model_name, *good_batch))},
                    ],
                )
                self.assertEqual(len(responses), 2)
                self.assertIsNotNone(responses[0].error)
                self.assertIn("batch index 1", str(responses[0].error))
                self.assertIsNone(responses[1].error)
                self.assertEqual(response_output(responses[1], "image").shape[0], 1)

    def test_empty_batch_and_wrong_rank_are_request_errors(self):
        valid, _ = make_encoded_images()
        invalid_inputs = (
            np.empty((0, valid.size), dtype=np.uint8),
            valid,
        )
        for model_name in ENCODED_IMAGE_MODELS:
            for invalid in invalid_inputs:
                with self.subTest(model=model_name, shape=invalid.shape):
                    model = load_python_model(model_name)
                    response = run_execute(
                        model,
                        [
                            {
                                "IMAGE_BYTES": FakeTensor(
                                    "IMAGE_BYTES", invalid
                                )
                            }
                        ],
                    )[0]
                    self.assertIsNotNone(response.error)

    def test_decoder_formats_remain_supported_with_trailing_zero_bytes(self):
        common_formats = ("PNG", "JPEG", "WEBP", "BMP", "GIF", "TIFF", "PPM")
        encoded_images = {
            format_name: make_encoded_image(format_name)
            for format_name in common_formats
        }
        encoded_images["ICO"] = make_encoded_image("ICO", size=(32, 32))
        for model_name in ENCODED_IMAGE_MODELS:
            model = load_python_model(model_name)
            supported_formats = common_formats
            if model_name != "sbs_r50_ibn_image_preprocess":
                supported_formats += ("ICO",)
            for format_name in supported_formats:
                encoded = encoded_images[format_name]
                with self.subTest(model=model_name, format=format_name):
                    plain_response = run_execute(
                        model,
                        [
                            {
                                "IMAGE_BYTES": FakeTensor(
                                    "IMAGE_BYTES", encoded_image_batch(model_name, encoded)
                                )
                            }
                        ],
                    )[0]
                    self.assertIsNone(plain_response.error)
                    plain_output = response_output(plain_response, "image")

                    zero_padded = np.pad(encoded, (0, 9))[None, :]
                    padded_response = run_execute(
                        model,
                        [
                            {
                                "IMAGE_BYTES": FakeTensor(
                                    "IMAGE_BYTES", encoded_image_batch(model_name, *zero_padded)
                                )
                            }
                        ],
                    )[0]
                    self.assertIsNone(padded_response.error)
                    np.testing.assert_allclose(
                        response_output(padded_response, "image"),
                        plain_output,
                        rtol=0.0,
                        atol=0.0,
                    )

    def test_sbs_and_osnet_accept_batch_five_offline(self):
        valid, _ = make_encoded_images()
        batch = np.repeat(valid[None, :], 5, axis=0)
        for model_name in (
            "sbs_r50_ibn_image_preprocess",
            "osnet_ain_x1_0_image_preprocess",
        ):
            with self.subTest(model=model_name):
                response = run_execute(
                    load_python_model(model_name),
                    [{"IMAGE_BYTES": FakeTensor("IMAGE_BYTES", encoded_image_batch(model_name, *batch))}],
                )[0]
                self.assertIsNone(response.error)
                self.assertEqual(response_output(response, "image").shape[0], 5)


class TextBatchTests(unittest.TestCase):
    def test_clip_text_real_config_initializes_and_returns_only_input_ids(self):
        model_config = json.loads(
            model_config_json_from_repository("clip_text_preprocess")
        )
        self.assertEqual(
            [output["name"] for output in model_config["output"]],
            ["INPUT_IDS"],
        )
        self.assertIsNone(
            make_fake_pb_utils().get_output_config_by_name(
                model_config, "UNKNOWN_OUTPUT"
            )
        )

        model = load_python_model("clip_text_preprocess")
        values = np.asarray([[b"first person"], [b"second person"]], dtype=object)
        response = run_execute(
            model,
            [{"TEXT": FakeTensor("TEXT", values)}],
        )[0]

        self.assertIsNone(response.error)
        self.assertEqual(
            [tensor.name for tensor in response.output_tensors],
            ["INPUT_IDS"],
        )
        input_ids = response_output(response, "INPUT_IDS")
        self.assertEqual(input_ids.shape, (2, 52))
        self.assertEqual(input_ids.dtype, np.int64)
        self.assertTrue(input_ids.flags.c_contiguous)

    def test_batch_four_produces_one_response_with_one_output_row_per_text(self):
        inputs = {
            "clip_text_preprocess": (
                "TEXT",
                ("INPUT_IDS",),
                52,
            ),
            "TBPS_clip_text_preprocess": (
                "TEXT_IN",
                ("TEXT_TOKENS",),
                77,
            ),
        }
        values = np.asarray(
            [
                [b"first person"],
                [b"second person"],
                [b"third person"],
                [b"fourth person"],
            ],
            dtype=object,
        )
        for model_name, (input_name, output_names, width) in inputs.items():
            with self.subTest(model=model_name):
                model = load_python_model(model_name)
                responses = run_execute(
                    model,
                    [{input_name: FakeTensor(input_name, values)}],
                )
                self.assertEqual(len(responses), 1)
                self.assertIsNone(responses[0].error)
                for output_name in output_names:
                    actual = response_output(responses[0], output_name)
                    references = []
                    for row in values:
                        response = run_execute(
                            model,
                            [
                                {
                                    input_name: FakeTensor(
                                        input_name, row[None, :]
                                    )
                                }
                            ],
                        )[0]
                        self.assertIsNone(response.error)
                        references.append(response_output(response, output_name)[0])
                    self.assertEqual(
                        actual.shape, (EXPLICIT_BATCH_SIZE, width)
                    )
                    np.testing.assert_array_equal(
                        actual, np.stack(references, axis=0)
                    )

    def test_bad_text_sample_is_indexed_and_does_not_abort_next_request(self):
        inputs = {
            "clip_text_preprocess": "TEXT",
            "TBPS_clip_text_preprocess": "TEXT_IN",
        }
        bad_values = np.asarray([[b"valid"], [b"\xff"]], dtype=object)
        good_values = np.asarray([[b"still valid"]], dtype=object)
        for model_name, input_name in inputs.items():
            with self.subTest(model=model_name):
                responses = run_execute(
                    load_python_model(model_name),
                    [
                        {input_name: FakeTensor(input_name, bad_values)},
                        {input_name: FakeTensor(input_name, good_values)},
                    ],
                )
                self.assertEqual(len(responses), 2)
                self.assertIsNotNone(responses[0].error)
                self.assertIn("batch index 1", str(responses[0].error))
                self.assertIsNone(responses[1].error)

    def test_tokenizer_failure_on_second_sample_reports_its_batch_index(self):
        inputs = {
            "clip_text_preprocess": "TEXT",
            "TBPS_clip_text_preprocess": "TEXT_IN",
        }
        poison_text = "PRIVATE_POISON_TEXT_92a61"
        backend_secret = "tokenizer backend secret 6f703"
        bad_values = np.asarray(
            [[b"first person"], [poison_text.encode("utf-8")]], dtype=object
        )
        good_values = np.asarray([[b"first person"]], dtype=object)
        for model_name, input_name in inputs.items():
            with self.subTest(model=model_name):
                model = load_python_model(model_name)
                if model_name == "clip_text_preprocess":
                    original = model.process_texts

                    def fail_second(texts, context_length=52):
                        if texts == [poison_text]:
                            raise RuntimeError(backend_secret)
                        return original(texts, context_length=context_length)

                    model.process_texts = fail_second
                else:
                    globals_dict = model._execute_request.__globals__
                    original = globals_dict["tokenize"]

                    def fail_second(texts, context_length=77, return_length=False):
                        if texts == [poison_text]:
                            raise RuntimeError(backend_secret)
                        return original(
                            texts,
                            context_length=context_length,
                            return_length=return_length,
                        )

                    globals_dict["tokenize"] = fail_second

                try:
                    responses = run_execute(
                        model,
                        [
                            {input_name: FakeTensor(input_name, bad_values)},
                            {input_name: FakeTensor(input_name, good_values)},
                        ],
                    )
                finally:
                    if model_name == "TBPS_clip_text_preprocess":
                        globals_dict["tokenize"] = original
                self.assertEqual(len(responses), 2)
                self.assertIsNotNone(responses[0].error)
                error_message = str(responses[0].error)
                self.assertIn("unable to preprocess text", error_message)
                self.assertIn("batch index 1", error_message)
                self.assertNotIn(poison_text, error_message)
                self.assertNotIn(backend_secret, error_message)
                self.assertIsNone(responses[1].error)

    def test_text_input_must_have_shape_batch_by_one(self):
        inputs = {
            "clip_text_preprocess": "TEXT",
            "TBPS_clip_text_preprocess": "TEXT_IN",
        }
        invalid_values = (
            np.asarray([b"flat"], dtype=object),
            np.asarray([[b"first", b"second"]], dtype=object),
            np.empty((0, 1), dtype=object),
        )
        for model_name, input_name in inputs.items():
            for values in invalid_values:
                with self.subTest(model=model_name, shape=values.shape):
                    response = run_execute(
                        load_python_model(model_name),
                        [{input_name: FakeTensor(input_name, values)}],
                    )[0]
                    self.assertIsNotNone(response.error)

    def test_string_and_utf8_bytes_inputs_are_equivalent(self):
        inputs = {
            "clip_text_preprocess": (
                "TEXT",
                ("INPUT_IDS",),
            ),
            "TBPS_clip_text_preprocess": ("TEXT_IN", ("TEXT_TOKENS",)),
        }
        for model_name, (input_name, output_names) in inputs.items():
            with self.subTest(model=model_name):
                model = load_python_model(model_name)
                bytes_response = run_execute(
                    model,
                    [
                        {
                            input_name: FakeTensor(
                                input_name,
                                np.asarray([[b"same text"]], dtype=object),
                            )
                        }
                    ],
                )[0]
                string_response = run_execute(
                    model,
                    [
                        {
                            input_name: FakeTensor(
                                input_name,
                                np.asarray([["same text"]], dtype=object),
                            )
                        }
                    ],
                )[0]
                self.assertIsNone(bytes_response.error)
                self.assertIsNone(string_response.error)
                for output_name in output_names:
                    np.testing.assert_array_equal(
                        response_output(bytes_response, output_name),
                        response_output(string_response, output_name),
                    )


class FaceQaBatchTests(unittest.TestCase):
    def test_preprocess_batch_four_matches_four_batch_one_references(self):
        images = np.stack(
            [
                np.full((17, 23, 3), [10, 20, 30], dtype=np.uint8),
                np.full((17, 23, 3), [40, 50, 60], dtype=np.uint8),
                np.full((17, 23, 3), [70, 80, 90], dtype=np.uint8),
                np.full((17, 23, 3), [100, 110, 120], dtype=np.uint8),
            ]
        )
        original_images = images.copy()
        landmark = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        ) * np.array([23.0 / 112.0, 17.0 / 112.0], dtype=np.float32)
        landmarks = np.repeat(landmark[None, ...], EXPLICIT_BATCH_SIZE, axis=0)
        original_landmarks = landmarks.copy()
        model = load_python_model("face_qa_pre")

        response = run_execute(
            model,
            [
                {
                    "RAW_FACE_IMAGE": FakeTensor(
                        "RAW_FACE_IMAGE", images
                    ),
                    "LANDMARK": FakeTensor("LANDMARK", landmarks),
                }
            ],
        )[0]

        self.assertIsNone(response.error)
        actual = response_output(response, "QA_PREPROCESSED_IMAGE")
        references = []
        for image, sample_landmark in zip(images, landmarks):
            single = run_execute(
                model,
                [
                    {
                        "RAW_FACE_IMAGE": FakeTensor(
                            "RAW_FACE_IMAGE", image[None, ...]
                        ),
                        "LANDMARK": FakeTensor(
                            "LANDMARK", sample_landmark[None, ...]
                        ),
                    }
                ],
            )[0]
            self.assertIsNone(single.error)
            references.append(
                response_output(single, "QA_PREPROCESSED_IMAGE")[0]
            )

        self.assertEqual(
            actual.shape, (EXPLICIT_BATCH_SIZE, 3, 112, 112)
        )
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue(actual.flags.c_contiguous)
        np.testing.assert_allclose(
            actual, np.stack(references), rtol=0.0, atol=0.0
        )
        np.testing.assert_array_equal(images, original_images)
        np.testing.assert_array_equal(landmarks, original_landmarks)

    def test_postprocess_batch_four_matches_four_batch_one_references(self):
        quality = np.array(
            [
                [0.6, 0.5, 0.8, 0.1, 0.9, 0.0, 0.2, 0.7, 0.4, 1.0],
                [0.5, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.6, 0.7, 0.8],
                [0.9, 0.8, 0.7, 0.6, 0.5, 1.0, 0.9, 0.8, 0.7, 0.6],
                [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
            ],
            dtype=np.float32,
        )
        original_quality = quality.copy()
        model = load_python_model("face_qa_postprocess")
        response = run_execute(
            model,
            [{"quality": FakeTensor("quality", quality)}],
        )[0]

        self.assertIsNone(response.error)
        actual = response_output(response, "quality_score")
        np.testing.assert_allclose(
            actual,
            np.array([[5.2], [4.1], [7.5], [2.5]], dtype=np.float32),
            rtol=0.0,
            atol=1e-6,
        )
        references = []
        for row in quality:
            single = run_execute(
                model,
                [{"quality": FakeTensor("quality", row[None, ...])}],
            )[0]
            self.assertIsNone(single.error)
            references.append(response_output(single, "quality_score")[0])
        np.testing.assert_array_equal(actual, np.stack(references))
        self.assertEqual(actual.shape, (EXPLICIT_BATCH_SIZE, 1))
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue(actual.flags.c_contiguous)
        np.testing.assert_array_equal(quality, original_quality)

    def test_face_qa_postprocess_uses_modelscope_continuous_mean_score(self):
        quality = np.array(
            [[0.9, 0.8, 0.7, 0.6, 0.55, 0.51, 0.49, 0.4, 0.3, 0.2]],
            dtype=np.float32,
        )

        response = run_execute(
            load_python_model("face_qa_postprocess"),
            [{"quality": FakeTensor("quality", quality)}],
        )[0]

        self.assertIsNone(response.error)
        np.testing.assert_allclose(
            response_output(response, "quality_score"),
            np.array([[5.45]], dtype=np.float32),
            rtol=0.0,
            atol=1e-6,
        )

    def test_bad_request_does_not_abort_following_request(self):
        bad = np.zeros((10,), dtype=np.float32)
        good = np.ones((1, 10), dtype=np.float32)
        responses = run_execute(
            load_python_model("face_qa_postprocess"),
            [
                {"quality": FakeTensor("quality", bad)},
                {"quality": FakeTensor("quality", good)},
            ],
        )

        self.assertEqual(len(responses), 2)
        self.assertIsNotNone(responses[0].error)
        self.assertIsNone(responses[1].error)
        np.testing.assert_array_equal(
            response_output(responses[1], "quality_score"),
            np.array([[10.0]], dtype=np.float32),
        )

    def test_postprocess_quality_requires_float32(self):
        response = run_execute(
            load_python_model("face_qa_postprocess"),
            [
                {
                    "quality": FakeTensor(
                        "quality", np.ones((1, 10), dtype=np.float64)
                    )
                }
            ],
        )[0]

        self.assertIsNotNone(response.error)
        self.assertIn("quality", str(response.error))
        self.assertIn("float32", str(response.error))


class FaceKeypointPostprocessBatchTests(unittest.TestCase):
    @staticmethod
    def _raw_batch():
        first = np.linspace(-1.0, 1.0, 212, dtype=np.float32)
        second = np.linspace(1.0, -1.0, 212, dtype=np.float32)
        return np.stack([first, second])

    def test_batch_four_matches_four_batch_one_references(self):
        raw_output = np.stack(
            [
                np.linspace(-1.0, 1.0, 212, dtype=np.float32),
                np.linspace(1.0, -1.0, 212, dtype=np.float32),
                np.linspace(-0.5, 0.5, 212, dtype=np.float32),
                np.linspace(0.75, -0.25, 212, dtype=np.float32),
            ]
        )
        affine_matrices = np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[2.0, 0.0, 4.0], [0.0, 2.0, 8.0]],
                [[0.5, 0.0, -3.0], [0.0, 0.5, 6.0]],
                [[4.0, 0.0, 12.0], [0.0, 4.0, -4.0]],
            ],
            dtype=np.float64,
        )
        original_raw_output = raw_output.copy()
        original_affine_matrices = affine_matrices.copy()
        model = load_python_model("face_keypoint_2d106_post")

        response = run_execute(
            model,
            [
                {
                    "RAW_OUTPUT": FakeTensor("RAW_OUTPUT", raw_output),
                    "AFFINE_MATRIX": FakeTensor(
                        "AFFINE_MATRIX", affine_matrices
                    ),
                }
            ],
        )[0]

        self.assertIsNone(response.error)
        actual = response_output(response, "LANDMARKS_106")
        references = []
        for raw, matrix in zip(raw_output, affine_matrices):
            single = run_execute(
                model,
                [
                    {
                        "RAW_OUTPUT": FakeTensor(
                            "RAW_OUTPUT", raw[None, ...]
                        ),
                        "AFFINE_MATRIX": FakeTensor(
                            "AFFINE_MATRIX", matrix[None, ...]
                        ),
                    }
                ],
            )[0]
            self.assertIsNone(single.error)
            references.append(response_output(single, "LANDMARKS_106")[0])

        self.assertEqual(
            actual.shape, (EXPLICIT_BATCH_SIZE, 106, 2)
        )
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue(actual.flags.c_contiguous)
        cropped = (
            raw_output.reshape(EXPLICIT_BATCH_SIZE, 106, 2) + 1.0
        ) * 96.0
        expected = cropped.copy()
        expected[1, :, 0] = (cropped[1, :, 0] - 4.0) / 2.0
        expected[1, :, 1] = (cropped[1, :, 1] - 8.0) / 2.0
        expected[2, :, 0] = (cropped[2, :, 0] + 3.0) / 0.5
        expected[2, :, 1] = (cropped[2, :, 1] - 6.0) / 0.5
        expected[3, :, 0] = (cropped[3, :, 0] - 12.0) / 4.0
        expected[3, :, 1] = (cropped[3, :, 1] + 4.0) / 4.0
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)
        self.assertFalse(np.array_equal(actual[0], actual[1]))
        np.testing.assert_allclose(
            actual, np.stack(references), rtol=0.0, atol=0.0
        )
        np.testing.assert_array_equal(raw_output, original_raw_output)
        np.testing.assert_array_equal(
            affine_matrices, original_affine_matrices
        )

    def test_rejects_shape_batch_finite_and_transform_errors(self):
        raw = self._raw_batch()
        matrices = np.repeat(
            np.eye(2, 3, dtype=np.float64)[None, ...], 2, axis=0
        )
        nonfinite_raw = raw.copy()
        nonfinite_raw[1, 0] = np.nan
        nonfinite_matrix = matrices.copy()
        nonfinite_matrix[1, 0, 0] = np.inf
        singular = matrices.copy()
        singular[1] = 0.0
        cases = (
            (raw[0], matrices, "[B,212]"),
            (raw, matrices[0], "[B,2,3]"),
            (raw, matrices[:1], "batch size"),
            (nonfinite_raw, matrices, "batch index 1"),
            (raw, nonfinite_matrix, "batch index 1"),
            (raw, singular, "batch index 1"),
        )
        for raw_output, affine_matrices, expected in cases:
            with self.subTest(expected=expected):
                response = run_execute(
                    load_python_model("face_keypoint_2d106_post"),
                    [
                        {
                            "RAW_OUTPUT": FakeTensor(
                                "RAW_OUTPUT", raw_output
                            ),
                            "AFFINE_MATRIX": FakeTensor(
                                "AFFINE_MATRIX", affine_matrices
                            ),
                        }
                    ],
                )[0]
                self.assertIsNotNone(response.error)
                self.assertIn(expected, str(response.error))

    def test_inputs_require_the_configured_dtypes(self):
        raw = self._raw_batch()[:1]
        matrix = np.eye(2, 3, dtype=np.float64)[None, ...]
        cases = (
            (
                raw.astype(np.float64),
                matrix,
                "RAW_OUTPUT",
                "float32",
            ),
            (
                raw,
                matrix.astype(np.float32),
                "AFFINE_MATRIX",
                "float64",
            ),
        )
        for raw_output, affine_matrices, input_name, expected_dtype in cases:
            with self.subTest(input_name=input_name):
                response = run_execute(
                    load_python_model("face_keypoint_2d106_post"),
                    [
                        {
                            "RAW_OUTPUT": FakeTensor(
                                "RAW_OUTPUT", raw_output
                            ),
                            "AFFINE_MATRIX": FakeTensor(
                                "AFFINE_MATRIX", affine_matrices
                            ),
                        }
                    ],
                )[0]

                self.assertIsNotNone(response.error)
                self.assertIn(input_name, str(response.error))
                self.assertIn(expected_dtype, str(response.error))

    def test_rejects_unsafe_affine_singular_values(self):
        raw = self._raw_batch()
        safe = np.eye(2, 3, dtype=np.float64)
        cases = (
            (
                np.array(
                    [[1.0, 0.0, 0.0], [0.0, 1e-15, 0.0]],
                    dtype=np.float64,
                ),
                "scale",
            ),
            (
                np.array(
                    [[1e20, 0.0, 0.0], [0.0, 1e20, 0.0]],
                    dtype=np.float64,
                ),
                "scale",
            ),
            (
                np.array(
                    [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                    dtype=np.float64,
                ),
                "condition",
            ),
        )
        for unsafe, expected in cases:
            with self.subTest(expected=expected):
                matrices = np.stack([safe, unsafe])
                response = run_execute(
                    load_python_model("face_keypoint_2d106_post"),
                    [
                        {
                            "RAW_OUTPUT": FakeTensor("RAW_OUTPUT", raw),
                            "AFFINE_MATRIX": FakeTensor(
                                "AFFINE_MATRIX", matrices
                            ),
                        }
                    ],
                )[0]

                self.assertIsNotNone(response.error)
                message = str(response.error)
                self.assertIn("batch index 1", message)
                self.assertIn(expected, message)

    def test_bad_second_transform_does_not_abort_following_request(self):
        raw = self._raw_batch()
        matrices = np.repeat(
            np.eye(2, 3, dtype=np.float64)[None, ...], 2, axis=0
        )
        matrices[1] = 0.0
        good_matrix = np.eye(2, 3, dtype=np.float64)[None, ...]
        responses = run_execute(
            load_python_model("face_keypoint_2d106_post"),
            [
                {
                    "RAW_OUTPUT": FakeTensor("RAW_OUTPUT", raw),
                    "AFFINE_MATRIX": FakeTensor(
                        "AFFINE_MATRIX", matrices
                    ),
                },
                {
                    "RAW_OUTPUT": FakeTensor(
                        "RAW_OUTPUT", raw[:1]
                    ),
                    "AFFINE_MATRIX": FakeTensor(
                        "AFFINE_MATRIX", good_matrix
                    ),
                },
            ],
        )

        self.assertEqual(len(responses), 2)
        self.assertIsNotNone(responses[0].error)
        self.assertIn("batch index 1", str(responses[0].error))
        self.assertIsNone(responses[1].error)


class FaceQaPreInternalErrorTests(unittest.TestCase):
    SECRET = r"SECRET at C:\private\imgwarp.cpp:77"

    def _assert_redacted(self, response, model_name):
        self.assertIsNotNone(response.error)
        message = str(response.error)
        incident_match = stdlib_re.fullmatch(
            rf"INTERNAL_ERROR: {model_name} failed; incident=([0-9a-f]{{12}})",
            message,
        )
        self.assertIsNotNone(incident_match)
        self.assertNotIn("SECRET", message)
        self.assertNotIn("private", message)
        return incident_match.group(1)

    def test_face_qa_pre_redacts_second_sample_internal_failure(self):
        model_name = "face_qa_pre"
        model = load_python_model(model_name)
        model_globals = model.execute.__globals__
        original = model_globals["preprocess"]
        images = np.stack(
            [
                np.full((8, 8, 3), 10, dtype=np.uint8),
                np.full((8, 8, 3), 40, dtype=np.uint8),
            ]
        )
        landmark = np.array(
            [
                [2.7353, 3.6926],
                [5.2523, 3.6787],
                [4.0018, 5.1240],
                [2.9678, 6.5975],
                [5.0521, 6.5860],
            ],
            dtype=np.float32,
        )
        landmarks = np.repeat(landmark[None, ...], 2, axis=0)

        def fail_selected(image, sample_landmark):
            if int(image[0, 0, 0]) == 40:
                raise RuntimeError(self.SECRET)
            return original(image, sample_landmark)

        logger = logging.getLogger(model_name)
        logger.disabled = False
        with mock.patch.dict(
            model_globals, {"preprocess": fail_selected}
        ), self.assertLogs(model_name, level="ERROR") as captured:
            responses = run_execute(
                model,
                [
                    {
                        "RAW_FACE_IMAGE": FakeTensor(
                            "RAW_FACE_IMAGE", images
                        ),
                        "LANDMARK": FakeTensor("LANDMARK", landmarks),
                    },
                    {
                        "RAW_FACE_IMAGE": FakeTensor(
                            "RAW_FACE_IMAGE", images[:1]
                        ),
                        "LANDMARK": FakeTensor(
                            "LANDMARK", landmarks[:1]
                        ),
                    },
                ],
            )

        self.assertEqual(len(responses), 2)
        incident = self._assert_redacted(responses[0], model_name)
        self.assertIsNone(responses[1].error)
        self.assertEqual(len(captured.records), 1)
        record = captured.records[0]
        self.assertIn("request_index=0", record.getMessage())
        self.assertIn("batch_index=1", record.getMessage())
        self.assertIn(f"incident={incident}", record.getMessage())
        self.assertIsNotNone(record.exc_info)
        self.assertIn(self.SECRET, "\n".join(captured.output))


YOLO_PREPROCESSORS = (
    ("yolov8n_face_preprocess", 5),
    ("yolo26m_pose_preprocess", 7),
    ("yolo26s_seg_preprocess", 7),
)


class YoloPreprocessBatchTests(unittest.TestCase):
    @staticmethod
    def _images():
        y, x = np.mgrid[:19, :31]
        first = np.stack(
            [(x * 7) % 256, (y * 11) % 256, ((x + y) * 13) % 256],
            axis=2,
        ).astype(np.uint8)
        second = np.stack(
            [((x + 5) * 17) % 256, ((y + 3) * 19) % 256, (x * y) % 256],
            axis=2,
        ).astype(np.uint8)
        third = np.stack(
            [((x + 2) * 23) % 256, ((y + 7) * 29) % 256, (x + 2 * y) % 256],
            axis=2,
        ).astype(np.uint8)
        fourth = np.stack(
            [((x + 11) * 31) % 256, ((y + 13) * 37) % 256, (2 * x + y) % 256],
            axis=2,
        ).astype(np.uint8)
        return np.stack([first, second, third, fourth])

    def test_batch_four_matches_four_batch_one_references(self):
        for model_name, meta_width in YOLO_PREPROCESSORS:
            with self.subTest(model_name=model_name):
                images = self._images()
                original = images.copy()
                model = load_python_model(model_name)
                response = run_execute(
                    model,
                    [{"RAW_IMAGE": FakeTensor("RAW_IMAGE", images)}],
                )[0]

                self.assertIsNone(response.error)
                actual_image = response_output(
                    response, "PREPROCESSED_IMAGE"
                )
                actual_meta = response_output(response, "LETTERBOX_META")
                image_references = []
                meta_references = []
                for image in images:
                    single = run_execute(
                        model,
                        [
                            {
                                "RAW_IMAGE": FakeTensor(
                                    "RAW_IMAGE", image[None, ...]
                                )
                            }
                        ],
                    )[0]
                    self.assertIsNone(single.error)
                    image_references.append(
                        response_output(single, "PREPROCESSED_IMAGE")[0]
                    )
                    meta_references.append(
                        response_output(single, "LETTERBOX_META")[0]
                    )

                self.assertEqual(
                    actual_image.shape[0], EXPLICIT_BATCH_SIZE
                )
                self.assertEqual(actual_image.shape[1], 3)
                self.assertEqual(
                    actual_meta.shape, (EXPLICIT_BATCH_SIZE, meta_width)
                )
                self.assertEqual(actual_image.dtype, np.float32)
                self.assertEqual(actual_meta.dtype, np.float32)
                self.assertTrue(np.isfinite(actual_image).all())
                self.assertTrue(np.isfinite(actual_meta).all())
                self.assertTrue(actual_image.flags.c_contiguous)
                self.assertTrue(actual_meta.flags.c_contiguous)
                self.assertFalse(np.array_equal(actual_image[0], actual_image[1]))
                np.testing.assert_allclose(
                    actual_image,
                    np.stack(image_references),
                    rtol=0.0,
                    atol=0.0,
                )
                np.testing.assert_allclose(
                    actual_meta,
                    np.stack(meta_references),
                    rtol=0.0,
                    atol=0.0,
                )
                np.testing.assert_array_equal(images, original)

    def test_different_fixed_buckets_remain_isolated_requests(self):
        square = np.full((1, 640, 640, 3), 10, dtype=np.uint8)
        wide = np.full((1, 360, 640, 3), 20, dtype=np.uint8)
        for model_name, meta_width in YOLO_PREPROCESSORS:
            with self.subTest(model=model_name):
                model = load_python_model(model_name)
                responses = run_execute(
                    model,
                    [
                        {"RAW_IMAGE": FakeTensor("RAW_IMAGE", square)},
                        {"RAW_IMAGE": FakeTensor("RAW_IMAGE", wide)},
                    ],
                )

                self.assertEqual(len(responses), 2)
                self.assertTrue(
                    all(response.error is None for response in responses)
                )
                square_image = response_output(
                    responses[0], "PREPROCESSED_IMAGE"
                )
                wide_image = response_output(
                    responses[1], "PREPROCESSED_IMAGE"
                )
                self.assertEqual(square_image.shape, (1, 3, 640, 640))
                self.assertEqual(wide_image.shape, (1, 3, 384, 640))
                square_meta = response_output(
                    responses[0], "LETTERBOX_META"
                )
                wide_meta = response_output(
                    responses[1], "LETTERBOX_META"
                )
                self.assertEqual(square_meta.shape, (1, meta_width))
                self.assertEqual(wide_meta.shape, (1, meta_width))
                np.testing.assert_array_equal(
                    square_meta[0, :2], [640, 640]
                )
                np.testing.assert_array_equal(
                    wide_meta[0, :2], [360, 640]
                )

    def test_rejects_rank_dtype_empty_and_resource_inputs(self):
        images = self._images()
        oversized = np.lib.stride_tricks.as_strided(
            np.zeros((1, 1, 1, 3), dtype=np.uint8),
            shape=(1, 8193, 1, 3),
            strides=(0, 0, 0, 1),
        )
        cases = (
            (images[0], "[B,H,W,3]"),
            (images.transpose(0, 3, 1, 2), "[B,H,W,3]"),
            (images.astype(np.float32), "uint8"),
            (np.empty((0, 19, 31, 3), dtype=np.uint8), "non-empty"),
            (np.empty((1, 0, 31, 3), dtype=np.uint8), "batch index 0"),
            (oversized, "batch index 0"),
        )
        for model_name, _ in YOLO_PREPROCESSORS:
            for value, expected in cases:
                with self.subTest(model_name=model_name, expected=expected):
                    response = run_execute(
                        load_python_model(model_name),
                        [{"RAW_IMAGE": FakeTensor("RAW_IMAGE", value)}],
                    )[0]
                    self.assertIsNotNone(response.error)
                    self.assertIn(expected, str(response.error))

    def test_bad_request_does_not_abort_following_request(self):
        for model_name, _ in YOLO_PREPROCESSORS:
            with self.subTest(model_name=model_name):
                images = self._images()
                responses = run_execute(
                    load_python_model(model_name),
                    [
                        {
                            "RAW_IMAGE": FakeTensor(
                                "RAW_IMAGE", images.astype(np.float32)
                            )
                        },
                        {
                            "RAW_IMAGE": FakeTensor(
                                "RAW_IMAGE", images[:1]
                            )
                        },
                    ],
                )
                self.assertIsNotNone(responses[0].error)
                self.assertIsNone(responses[1].error)
                self.assertEqual(len(responses), 2)

    def test_independent_letterbox_oracle_and_rejects_extreme_aspect_ratio(self):
        images = np.stack(
            [
                np.full((101, 200, 3), [10, 20, 30], dtype=np.uint8),
                np.full((101, 200, 3), [40, 50, 60], dtype=np.uint8),
            ]
        )
        expected_common = np.array(
            [[101, 200, 3.2, 0, 30], [101, 200, 3.2, 0, 30]],
            dtype=np.float32,
        )
        for model_name, meta_width in YOLO_PREPROCESSORS:
            with self.subTest(model_name=model_name):
                model = load_python_model(model_name)
                response = run_execute(
                    model,
                    [{"RAW_IMAGE": FakeTensor("RAW_IMAGE", images)}],
                )[0]
                self.assertIsNone(response.error)
                actual = response_output(response, "PREPROCESSED_IMAGE")
                meta = response_output(response, "LETTERBOX_META")
                self.assertEqual(actual.shape, (2, 3, 384, 640))
                np.testing.assert_allclose(
                    meta[:, :5], expected_common, rtol=0.0, atol=1e-6
                )
                if meta_width == 7:
                    np.testing.assert_array_equal(
                        meta[:, 5:],
                        np.array([[384, 640], [384, 640]], dtype=np.float32),
                    )
                np.testing.assert_allclose(
                    actual[0, :, 0, 0],
                    np.full(3, 114 / 255, dtype=np.float32),
                    rtol=0.0,
                    atol=1e-7,
                )
                np.testing.assert_allclose(
                    actual[0, :, 30, 0],
                    np.array([30, 20, 10], dtype=np.float32) / 255,
                    rtol=0.0,
                    atol=1e-7,
                )

                narrow = np.full((2000, 1, 3), 20, dtype=np.uint8)
                cv2_module = model._letterbox.__globals__["cv2"]
                with mock.patch.object(
                    cv2_module, "resize"
                ) as resize_mock, mock.patch.object(
                    cv2_module, "copyMakeBorder"
                ) as border_mock:
                    with self.assertRaisesRegex(
                        ValueError, "aspect ratio|resize dimension"
                    ):
                        model._letterbox(narrow)
                resize_mock.assert_not_called()
                border_mock.assert_not_called()

                class HeterogeneousImageBatch(np.ndarray):
                    def __new__(cls, samples):
                        value = np.empty(
                            (len(samples), 1, 1, 3), dtype=np.uint8
                        ).view(cls)
                        value.samples = samples
                        return value

                    def __array_finalize__(self, source):
                        self.samples = getattr(source, "samples", ())

                    def __iter__(self):
                        return iter(self.samples)

                mixed = HeterogeneousImageBatch(
                    [
                        np.full((64, 64, 3), 10, dtype=np.uint8),
                        narrow,
                    ]
                )
                good = np.full((1, 64, 64, 3), 10, dtype=np.uint8)
                responses = run_execute(
                    model,
                    [
                        {"RAW_IMAGE": FakeTensor("RAW_IMAGE", mixed)},
                        {"RAW_IMAGE": FakeTensor("RAW_IMAGE", good)},
                    ],
                )
                self.assertEqual(len(responses), 2)
                self.assertIsNotNone(responses[0].error)
                message = str(responses[0].error)
                self.assertIn("INVALID_INPUT", message)
                self.assertIn("batch index 1", message)
                self.assertIn("resize dimension", message)
                self.assertNotIn("INTERNAL_ERROR", message)
                self.assertIsNone(responses[1].error)


class YoloPostprocessBatchTests(unittest.TestCase):
    @staticmethod
    def _face_inputs():
        raw = np.zeros((EXPLICIT_BATCH_SIZE, 5, 8400), dtype=np.float32)
        raw[0, :, 0] = [160, 142, 192, 160, 0.9]
        raw[1, :, 0] = [199.33334, 320, 293.33334, 426.6667, 0.8]
        raw[2, :, 0] = [224, 206, 128, 96, 0.7]
        raw[3, :, 0] = [256, 300, 160, 200, 0.6]
        meta = np.array(
            [
                [101, 200, 3.2, 0, 30],
                [120, 80, 640 / 120, 26, 0],
                [101, 200, 3.2, 0, 30],
                [120, 80, 640 / 120, 26, 0],
            ],
            dtype=np.float32,
        )
        return raw, meta

    @staticmethod
    def _pose_inputs():
        raw = np.zeros((EXPLICIT_BATCH_SIZE, 2, 57), dtype=np.float32)
        raw[0, 0, :6] = [64, 62, 256, 222, 0.9, 0]
        raw[1, 0, :6] = [52.666668, 106.66667, 346, 533.3334, 0.8, 1]
        raw[2, 0, :6] = [100, 96, 300, 256, 0.7, 2]
        raw[3, 0, :6] = [136, 140, 376, 500, 0.6, 3]
        raw[0, 0, 6:] = np.tile([96, 94, 0.7], 17)
        raw[1, 0, 6:] = np.tile([186, 213.33334, 0.8], 17)
        raw[2, 0, 6:] = np.tile([160, 136, 0.6], 17)
        raw[3, 0, 6:] = np.tile([236, 260, 0.5], 17)
        meta = np.array(
            [
                [101, 200, 3.2, 0, 30, 384, 640],
                [120, 80, 640 / 120, 26, 0, 640, 480],
                [101, 200, 3.2, 0, 30, 384, 640],
                [120, 80, 640 / 120, 26, 0, 640, 480],
            ],
            dtype=np.float32,
        )
        return raw, meta

    def _assert_batch_matches_b1_references(self, model_name, inputs, outputs):
        raw, meta = inputs()
        original_raw = raw.copy()
        original_meta = meta.copy()
        model = load_python_model(model_name)
        response = run_execute(
            model,
            [
                {
                    "RAW_OUTPUT": FakeTensor("RAW_OUTPUT", raw),
                    "LETTERBOX_META": FakeTensor("LETTERBOX_META", meta),
                }
            ],
        )[0]
        self.assertIsNone(response.error)

        per_row = []
        for index in range(EXPLICIT_BATCH_SIZE):
            single = run_execute(
                model,
                [
                    {
                        "RAW_OUTPUT": FakeTensor(
                            "RAW_OUTPUT", raw[index : index + 1]
                        ),
                        "LETTERBOX_META": FakeTensor(
                            "LETTERBOX_META", meta[index : index + 1]
                        ),
                    }
                ],
            )[0]
            self.assertIsNone(single.error)
            per_row.append(single)

        for output_name in outputs:
            actual = response_output(response, output_name)
            expected = np.concatenate(
                [response_output(single, output_name) for single in per_row],
                axis=0,
            )
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
            self.assertTrue(actual.flags.c_contiguous)
            if output_name == "NUM_DETECTIONS":
                self.assertEqual(actual.dtype, np.int32)
            else:
                self.assertEqual(actual.dtype, np.float32)
                self.assertTrue(np.isfinite(actual).all())
        detections = response_output(response, "DETECTIONS")
        counts = response_output(response, "NUM_DETECTIONS")
        self.assertEqual(
            detections.shape, (EXPLICIT_BATCH_SIZE, 100, 6)
        )
        self.assertEqual(counts.shape, (EXPLICIT_BATCH_SIZE, 1))
        self.assertFalse(np.array_equal(detections[0], detections[1]))
        np.testing.assert_array_equal(raw, original_raw)
        np.testing.assert_array_equal(meta, original_meta)

        np.testing.assert_allclose(
            detections[0, 0, :4], [20, 10, 80, 60], atol=1e-5
        )
        np.testing.assert_allclose(
            detections[1, 0, :4], [5, 20, 60, 100], atol=2e-5
        )
        swapped_response = run_execute(
            model,
            [
                {
                    "RAW_OUTPUT": FakeTensor("RAW_OUTPUT", raw),
                    "LETTERBOX_META": FakeTensor(
                        "LETTERBOX_META", meta[::-1].copy()
                    ),
                }
            ],
        )[0]
        self.assertIsNone(swapped_response.error)
        self.assertFalse(
            np.allclose(
                response_output(swapped_response, "DETECTIONS"), detections
            )
        )

    def test_face_batch_four_matches_four_batch_one_references(self):
        self._assert_batch_matches_b1_references(
            "yolov8n_face_postprocess",
            self._face_inputs,
            ["DETECTIONS", "NUM_DETECTIONS"],
        )

    def test_pose_batch_four_matches_four_batch_one_references(self):
        self._assert_batch_matches_b1_references(
            "yolo26m_pose_postprocess",
            self._pose_inputs,
            ["DETECTIONS", "KEYPOINTS", "NUM_DETECTIONS"],
        )
        response = run_execute(
            load_python_model("yolo26m_pose_postprocess"),
            [
                {
                    "RAW_OUTPUT": FakeTensor(
                        "RAW_OUTPUT", self._pose_inputs()[0]
                    ),
                    "LETTERBOX_META": FakeTensor(
                        "LETTERBOX_META", self._pose_inputs()[1]
                    ),
                }
            ],
        )[0]
        self.assertEqual(
            response_output(response, "KEYPOINTS").shape,
            (EXPLICIT_BATCH_SIZE, 100, 17, 3),
        )
        keypoints = response_output(response, "KEYPOINTS")
        np.testing.assert_allclose(
            keypoints[0, 0, 0], [30, 20, 0.7], atol=1e-5
        )
        np.testing.assert_allclose(
            keypoints[1, 0, 0], [30, 40, 0.8], atol=2e-5
        )

    def test_rejects_dtype_shape_batch_finite_and_unsafe_meta(self):
        model_cases = (
            (
                "yolov8n_face_postprocess",
                self._face_inputs,
                "[B,5,N]",
            ),
            (
                "yolo26m_pose_postprocess",
                self._pose_inputs,
                "[B,N,57]",
            ),
        )
        for model_name, inputs, raw_shape_message in model_cases:
            raw, meta = inputs()
            nonfinite_raw = raw.copy()
            nonfinite_raw[1].reshape(-1)[0] = np.nan
            nonfinite_meta = meta.copy()
            nonfinite_meta[1, 2] = np.inf
            unsafe_meta = meta.copy()
            unsafe_meta[1, 2] = 0.0
            inconsistent_padding = meta.copy()
            inconsistent_padding[1, 3] += 1
            tiny_original_dimension = meta.copy()
            tiny_original_dimension[1, 0] = np.float32(5e-6)
            extreme_aspect_meta = meta.copy()
            extreme_aspect_meta[1, :5] = [2000, 1, 0.32, 15, 0]
            if extreme_aspect_meta.shape[1] == 7:
                extreme_aspect_meta[1, 5:] = [640, 32]
            resource_meta = meta.copy()
            resource_meta[1, 0] = 8193
            oversized_shape = (
                (1, 5, 100_001)
                if model_name == "yolov8n_face_postprocess"
                else (1, 100_001, 57)
            )
            oversized_raw = np.lib.stride_tricks.as_strided(
                np.zeros((1, 1, 1), dtype=np.float32),
                shape=oversized_shape,
                strides=(0, 0, 0),
            )
            cases = (
                (raw.astype(np.float64), meta, "RAW_OUTPUT", "float32"),
                (raw, meta.astype(np.float64), "LETTERBOX_META", "float32"),
                (raw[0], meta, "RAW_OUTPUT", raw_shape_message),
                (raw, meta[0], "LETTERBOX_META", "[B,"),
                (raw, meta[:1], "batch size", None),
                (nonfinite_raw, meta, "batch index 1", "finite"),
                (raw, nonfinite_meta, "batch index 1", "finite"),
                (raw, unsafe_meta, "batch index 1", "ratio"),
                (
                    raw,
                    inconsistent_padding,
                    "batch index 1",
                    "left padding",
                ),
                (
                    raw,
                    tiny_original_dimension,
                    "batch index 1",
                    "at least 1",
                ),
                (
                    raw,
                    extreme_aspect_meta,
                    "batch index 1",
                    "resize dimension",
                ),
                (raw, resource_meta, "batch index 1", "max side"),
                (oversized_raw, meta[:1], "batch index 0", "max predictions"),
            )
            if model_name == "yolo26m_pose_postprocess":
                inconsistent_input = meta.copy()
                inconsistent_input[1, 6] += 32
                cases += (
                    (
                        raw,
                        inconsistent_input,
                        "batch index 1",
                        "input dimensions",
                    ),
                )
            for raw_value, meta_value, expected, extra in cases:
                with self.subTest(model_name=model_name, expected=expected):
                    response = run_execute(
                        load_python_model(model_name),
                        [
                            {
                                "RAW_OUTPUT": FakeTensor(
                                    "RAW_OUTPUT", raw_value
                                ),
                                "LETTERBOX_META": FakeTensor(
                                    "LETTERBOX_META", meta_value
                                ),
                            }
                        ],
                    )[0]
                    self.assertIsNotNone(response.error)
                    message = str(response.error)
                    self.assertIn(expected, message)
                    if extra:
                        self.assertIn(extra, message)

    def test_bad_request_does_not_abort_following_request(self):
        for model_name, inputs in (
            ("yolov8n_face_postprocess", self._face_inputs),
            ("yolo26m_pose_postprocess", self._pose_inputs),
        ):
            with self.subTest(model_name=model_name):
                raw, meta = inputs()
                responses = run_execute(
                    load_python_model(model_name),
                    [
                        {
                            "RAW_OUTPUT": FakeTensor(
                                "RAW_OUTPUT", raw.astype(np.float64)
                            ),
                            "LETTERBOX_META": FakeTensor(
                                "LETTERBOX_META", meta
                            ),
                        },
                        {
                            "RAW_OUTPUT": FakeTensor(
                                "RAW_OUTPUT", raw[:1]
                            ),
                            "LETTERBOX_META": FakeTensor(
                                "LETTERBOX_META", meta[:1]
                            ),
                        },
                    ],
                )
                self.assertIsNotNone(responses[0].error)
                self.assertIsNone(responses[1].error)
                self.assertEqual(len(responses), 2)

    def test_face_rejects_n_by_five_and_ambiguous_layouts(self):
        _, meta = self._face_inputs()
        for shape in ((2, 6, 5), (2, 5, 5), (2, 1, 5, 6)):
            with self.subTest(shape=shape):
                response = run_execute(
                    load_python_model("yolov8n_face_postprocess"),
                    [
                        {
                            "RAW_OUTPUT": FakeTensor(
                                "RAW_OUTPUT", np.zeros(shape, dtype=np.float32)
                            ),
                            "LETTERBOX_META": FakeTensor(
                                "LETTERBOX_META", meta
                            ),
                        }
                    ],
                )[0]
                self.assertIsNotNone(response.error)
                self.assertIn("[B,5,N]", str(response.error))


class YoloInternalErrorTests(unittest.TestCase):
    SECRET = r"SECRET at C:\private\yolo.cpp:91"

    def _assert_redacted(self, response, model_name):
        self.assertIsNotNone(response.error)
        message = str(response.error)
        incident_match = stdlib_re.fullmatch(
            rf"INTERNAL_ERROR: {model_name} failed; incident=([0-9a-f]{{12}})",
            message,
        )
        self.assertIsNotNone(incident_match)
        self.assertNotIn("SECRET", message)
        self.assertNotIn("private", message)
        return incident_match.group(1)

    def test_preprocess_redacts_second_sample_internal_failure(self):
        images = YoloPreprocessBatchTests._images()
        for model_name, _ in YOLO_PREPROCESSORS:
            with self.subTest(model_name=model_name):
                model = load_python_model(model_name)
                original = model._letterbox

                def fail_selected(image):
                    if int(image[0, 0, 0]) == int(images[1, 0, 0, 0]):
                        raise RuntimeError(self.SECRET)
                    return original(image)

                logger = logging.getLogger(model_name)
                logger.disabled = False
                with mock.patch.object(
                    model, "_letterbox", side_effect=fail_selected
                ), self.assertLogs(model_name, level="ERROR") as captured:
                    responses = run_execute(
                        model,
                        [
                            {
                                "RAW_IMAGE": FakeTensor(
                                    "RAW_IMAGE", images
                                )
                            },
                            {
                                "RAW_IMAGE": FakeTensor(
                                    "RAW_IMAGE", images[:1]
                                )
                            },
                        ],
                    )
                self.assertEqual(len(responses), 2)
                incident = self._assert_redacted(responses[0], model_name)
                self.assertIsNone(responses[1].error)
                record = captured.records[0]
                self.assertIn("request_index=0", record.getMessage())
                self.assertIn("batch_index=1", record.getMessage())
                self.assertIn(f"incident={incident}", record.getMessage())
                self.assertIsNotNone(record.exc_info)

    def test_postprocess_redacts_second_sample_internal_failure(self):
        for model_name, inputs in (
            ("yolov8n_face_postprocess", YoloPostprocessBatchTests._face_inputs),
            ("yolo26m_pose_postprocess", YoloPostprocessBatchTests._pose_inputs),
        ):
            with self.subTest(model_name=model_name):
                raw, meta = inputs()
                model = load_python_model(model_name)
                original = model._postprocess
                calls = 0

                def fail_second(raw_sample, meta_sample):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise RuntimeError(self.SECRET)
                    return original(raw_sample, meta_sample)

                logger = logging.getLogger(model_name)
                logger.disabled = False
                with mock.patch.object(
                    model, "_postprocess", side_effect=fail_second
                ), self.assertLogs(model_name, level="ERROR") as captured:
                    responses = run_execute(
                        model,
                        [
                            {
                                "RAW_OUTPUT": FakeTensor(
                                    "RAW_OUTPUT", raw
                                ),
                                "LETTERBOX_META": FakeTensor(
                                    "LETTERBOX_META", meta
                                ),
                            },
                            {
                                "RAW_OUTPUT": FakeTensor(
                                    "RAW_OUTPUT", raw[:1]
                                ),
                                "LETTERBOX_META": FakeTensor(
                                    "LETTERBOX_META", meta[:1]
                                ),
                            },
                        ],
                    )
                self.assertEqual(len(responses), 2)
                incident = self._assert_redacted(responses[0], model_name)
                self.assertIsNone(responses[1].error)
                record = captured.records[0]
                self.assertIn("request_index=0", record.getMessage())
                self.assertIn("batch_index=1", record.getMessage())
                self.assertIn(f"incident={incident}", record.getMessage())
                self.assertIsNotNone(record.exc_info)


class FaceAnalysisInternalErrorTests(unittest.TestCase):
    SECRET = FaceQaPreInternalErrorTests.SECRET
    _assert_redacted = FaceQaPreInternalErrorTests._assert_redacted

    def test_face_qa_post_redacts_request_internal_failure(self):
        model_name = "face_qa_postprocess"
        model = load_python_model(model_name)

        class ExplodingTensor:
            def as_numpy(inner_self):
                raise RuntimeError(self.SECRET)

        logger = logging.getLogger(model_name)
        logger.disabled = False
        with self.assertLogs(model_name, level="ERROR") as captured:
            responses = run_execute(
                model,
                [
                    {"quality": ExplodingTensor()},
                    {
                        "quality": FakeTensor(
                            "quality", np.ones((1, 10), dtype=np.float32)
                        )
                    },
                ],
            )

        self.assertEqual(len(responses), 2)
        incident = self._assert_redacted(responses[0], model_name)
        self.assertIsNone(responses[1].error)
        self.assertEqual(len(captured.records), 1)
        record = captured.records[0]
        self.assertIn("request_index=0", record.getMessage())
        self.assertNotIn("batch_index=", record.getMessage())
        self.assertIn(f"incident={incident}", record.getMessage())
        self.assertIsNotNone(record.exc_info)
        self.assertIn(self.SECRET, "\n".join(captured.output))

    def test_keypoint_pre_redacts_second_sample_internal_failure(self):
        model_name = "face_keypoint_2d106_pre"
        model = load_python_model(model_name)
        model_globals = model.execute.__globals__
        original = model_globals["preprocess"]
        images = np.stack(
            [
                np.full((192, 192, 3), 10, dtype=np.uint8),
                np.full((192, 192, 3), 40, dtype=np.uint8),
            ]
        )
        boxes = np.repeat(
            np.array([[0, 0, 192, 192]], dtype=np.float32), 2, axis=0
        )

        def fail_selected(image, *args, **kwargs):
            if int(image[0, 0, 0]) == 40:
                raise RuntimeError(self.SECRET)
            return original(image, *args, **kwargs)

        logger = logging.getLogger(model_name)
        logger.disabled = False
        with mock.patch.dict(
            model_globals, {"preprocess": fail_selected}
        ), self.assertLogs(model_name, level="ERROR") as captured:
            responses = run_execute(
                model,
                [
                    {
                        "RAW_IMAGE": FakeTensor("RAW_IMAGE", images),
                        "BBOX": FakeTensor("BBOX", boxes),
                    },
                    {
                        "RAW_IMAGE": FakeTensor(
                            "RAW_IMAGE", images[:1]
                        ),
                        "BBOX": FakeTensor("BBOX", boxes[:1]),
                    },
                ],
            )

        self.assertEqual(len(responses), 2)
        incident = self._assert_redacted(responses[0], model_name)
        self.assertIsNone(responses[1].error)
        self.assertEqual(len(captured.records), 1)
        record = captured.records[0]
        self.assertIn("request_index=0", record.getMessage())
        self.assertIn("batch_index=1", record.getMessage())
        self.assertIn(f"incident={incident}", record.getMessage())
        self.assertIsNotNone(record.exc_info)
        self.assertIn(self.SECRET, "\n".join(captured.output))

    def test_keypoint_post_redacts_second_sample_internal_failure(self):
        model_name = "face_keypoint_2d106_post"
        model = load_python_model(model_name)
        model_globals = model.execute.__globals__
        original = model_globals["postprocess"]
        raw = FaceKeypointPostprocessBatchTests._raw_batch()
        matrices = np.repeat(
            np.eye(2, 3, dtype=np.float64)[None, ...], 2, axis=0
        )

        def fail_selected(prediction, *args, **kwargs):
            if float(prediction[0]) == 1.0:
                raise RuntimeError(self.SECRET)
            return original(prediction, *args, **kwargs)

        logger = logging.getLogger(model_name)
        logger.disabled = False
        with mock.patch.dict(
            model_globals, {"postprocess": fail_selected}
        ), self.assertLogs(model_name, level="ERROR") as captured:
            responses = run_execute(
                model,
                [
                    {
                        "RAW_OUTPUT": FakeTensor("RAW_OUTPUT", raw),
                        "AFFINE_MATRIX": FakeTensor(
                            "AFFINE_MATRIX", matrices
                        ),
                    },
                    {
                        "RAW_OUTPUT": FakeTensor(
                            "RAW_OUTPUT", raw[:1]
                        ),
                        "AFFINE_MATRIX": FakeTensor(
                            "AFFINE_MATRIX", matrices[:1]
                        ),
                    },
                ],
            )

        self.assertEqual(len(responses), 2)
        incident = self._assert_redacted(responses[0], model_name)
        self.assertIsNone(responses[1].error)
        self.assertEqual(len(captured.records), 1)
        record = captured.records[0]
        self.assertIn("request_index=0", record.getMessage())
        self.assertIn("batch_index=1", record.getMessage())
        self.assertIn(f"incident={incident}", record.getMessage())
        self.assertIsNotNone(record.exc_info)
        self.assertIn(self.SECRET, "\n".join(captured.output))


if __name__ == "__main__":
    unittest.main(verbosity=2)
