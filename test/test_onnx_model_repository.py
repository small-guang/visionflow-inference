"""ONNX source contracts and the runtime backend inventory."""

import hashlib
import importlib.util
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT / "model_repository"
MODEL_CATALOG = ROOT / "tools" / "model_pipeline" / "catalog.json"
GIT_ATTRIBUTES = ROOT / ".gitattributes"

TARGETS = {
    "clip_image_trt": {
        "input": ("image", "TYPE_FP32", (3, 224, 224)),
        "output": ("unnorm_image_features", "TYPE_FP32", (512,)),
        "onnx_input": ("image", "FLOAT"),
        "onnx_output": ("unnorm_image_features", "FLOAT"),
    },
    "clip_text_trt": {
        "input": ("text", "TYPE_INT64", (52,)),
        "output": ("unnorm_text_features", "TYPE_FP32", (512,)),
        "onnx_input": ("text", "INT64"),
        "onnx_output": ("unnorm_text_features", "FLOAT"),
    },
    "TBPS_clip_image_trt": {
        "input": ("image", "TYPE_FP32", (3, 224, 224)),
        "output": ("image_feat", "TYPE_FP32", (512,)),
        "onnx_input": ("image", "FLOAT"),
        "onnx_output": ("image_feat", "FLOAT"),
    },
    "TBPS_clip_text_trt": {
        "input": ("text", "TYPE_INT64", (77,)),
        "output": ("text_feat", "TYPE_FP32", (512,)),
        "onnx_input": ("text", "INT64"),
        "onnx_output": ("text_feat", "FLOAT"),
    },
    "w600k_r50_trt": {
        "input": ("input.1", "TYPE_FP32", (3, 112, 112)),
        "output": ("683", "TYPE_FP32", (512,)),
        "onnx_input": ("input.1", "FLOAT"),
        "onnx_output": ("683", "FLOAT"),
    },
    "face_qa": {
        "input": ("input", "TYPE_FP32", (3, 112, 112)),
        "output": ("quality", "TYPE_FP32", (10,)),
        "onnx_input": ("input", "FLOAT"),
        "onnx_output": ("quality", "FLOAT"),
    },
}

ONNX_MODELS = {
    "adaface_ir50_webface4m_onnx",
    "clip_image_trt",
    "clip_text_trt",
    "face_keypoint_2d106",
    "face_qa",
    "osnet_ain_x1_0_onnx",
    "sbs_r50_ibn_onnx",
    "sixdrepnet_onnx",
    "TBPS_clip_image_trt",
    "TBPS_clip_text_trt",
    "w600k_r50_trt",
    "yolo26m_pose_onnx",
    "yolo26s_seg_onnx",
    "yolov8n_face_onnx",
}
TENSORRT_MODELS = {
    "yolov8n_face_onnx", "yolo26m_pose_onnx", "yolo26s_seg_onnx",
    "clip_image_trt", "clip_text_trt", "TBPS_clip_image_trt", "TBPS_clip_text_trt",
    "sbs_r50_ibn_onnx", "w600k_r50_trt", "face_qa", "face_keypoint_2d106",
}
PYTHON_MODELS = {
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
ENSEMBLE_MODELS = {
    "adaface_recognition_pipeline",
    "clip_ensemble_trt",
    "clip_text_ensemble",
    "face_keypoint_2d106_ensemble",
    "face_qa_ensemble",
    "face_recognition_pipeline",
    "osnet_ain_x1_0_image_ensemble",
    "sbs_r50_ibn_image_ensemble",
    "sixdrepnet_ensemble",
    "TBPS_clip_image_ensemble",
    "TBPS_clip_text_ensemble",
    "yolo26m_pose_ensemble",
    "yolo26s_seg_ensemble",
    "yolov8n_face_ensemble",
}

# Independent graph-boundary contracts. Triton config contracts intentionally live
# in test_batch_model_repository.py and are not imported here.
ONNX_SAMPLE_IO = {
    "adaface_ir50_webface4m_onnx": {
        "inputs": (("input", "FLOAT", (3, 112, 112)),),
        "outputs": (("embedding", "FLOAT", (512,)),),
    },
    "clip_image_trt": {
        "inputs": (("image", "FLOAT", (3, 224, 224)),),
        "outputs": (("unnorm_image_features", "FLOAT", (512,)),),
    },
    "clip_text_trt": {
        "inputs": (("text", "INT64", (52,)),),
        "outputs": (("unnorm_text_features", "FLOAT", (512,)),),
    },
    "face_keypoint_2d106": {
        "inputs": (("data", "FLOAT", (3, 192, 192)),),
        "outputs": (("fc1", "FLOAT", (212,)),),
    },
    "face_qa": {
        "inputs": (("input", "FLOAT", (3, 112, 112)),),
        "outputs": (("quality", "FLOAT", (10,)),),
    },
    "osnet_ain_x1_0_onnx": {
        "inputs": (("image", "FLOAT", (3, 256, 128)),),
        "outputs": (("embedding", "FLOAT", (512,)),),
    },
    "sbs_r50_ibn_onnx": {
        "inputs": (("image", "FLOAT", (3, 384, 128)),),
        "outputs": (("embedding", "FLOAT", (2048,)),),
    },
    "sixdrepnet_onnx": {
        "inputs": (("image", "FLOAT", (3, 224, 224)),),
        "outputs": (("rotation_matrix", "FLOAT", (3, 3)),),
    },
    "TBPS_clip_image_trt": {
        "inputs": (("image", "FLOAT", (3, 224, 224)),),
        "outputs": (("image_feat", "FLOAT", (512,)),),
    },
    "TBPS_clip_text_trt": {
        "inputs": (("text", "INT64", (77,)),),
        "outputs": (("text_feat", "FLOAT", (512,)),),
    },
    "w600k_r50_trt": {
        "inputs": (("input.1", "FLOAT", (3, 112, 112)),),
        "outputs": (("683", "FLOAT", (512,)),),
    },
    "yolo26m_pose_onnx": {
        "inputs": (("images", "FLOAT", (3, -1, -1)),),
        "outputs": (("output0", "FLOAT", (-1, 57)),),
    },
    "yolo26s_seg_onnx": {
        "inputs": (("images", "FLOAT", (3, -1, -1)),),
        "outputs": (
            ("output0", "FLOAT", (-1, 38)),
            ("output1", "FLOAT", (32, -1, -1)),
        ),
    },
    "yolov8n_face_onnx": {
        "inputs": (("images", "FLOAT", (3, -1, -1)),),
        "outputs": (("output0", "FLOAT", (5, -1)),),
    },
}

EXPECTED_ARTIFACT_TARGETS = {
    "clip_image_trt": {
        "deployed_onnx": "model_repository/clip_image_trt/1/model.onnx",
        "onnx_size": 345198918,
        "onnx_sha256": "10ca7abc0ef046a19ff2f670a8e3e278d02939c305c340b530ee23136e94cd5b",
    },
    "clip_text_trt": {
        "deployed_onnx": "model_repository/clip_text_trt/1/model.onnx",
        "onnx_size": 408501331,
        "onnx_sha256": "cfcc0be7a2a39577ef62cdcd0f1eebffa992975d549bf1bac9c706dfe3140c0c",
    },
    "TBPS_clip_image_trt": {
        "deployed_onnx": "model_repository/TBPS_clip_image_trt/1/model.onnx",
        "onnx_size": 345199066,
        "onnx_sha256": "ed5c71dde43ca0b749ba7a13690d8297d2cae19945d484789b88f855a202c14e",
    },
    "TBPS_clip_text_trt": {
        "deployed_onnx": "model_repository/TBPS_clip_text_trt/1/model.onnx",
        "onnx_size": 254389818,
        "onnx_sha256": "4ff52ecb9371c8df795b5d0ffb2211c8c993234a5b5549ac68b6495161b4bede",
    },
    "w600k_r50_trt": {
        "deployed_onnx": "model_repository/w600k_r50_trt/1/model.onnx",
        "onnx_size": 174383866,
        "onnx_sha256": "6afbf406aa229a439abbca7436cc42be254d4e3af6200d8b7ae4c1fec0c18f2f",
    },
    "face_qa": {
        "deployed_onnx": "model_repository/face_qa/1/model.onnx",
        "onnx_size": 406124,
        "onnx_sha256": "05a4ccd4f2a41da88447916c594cd55c53bc9ad5a2c1e19a1d65d23abbc5af29",
    },
}

SIXDREPNET_GENERATED_ARTIFACT = {
    "source_checkpoint": "tools/models/6DRepNet/6DRepNet_300W_LP_AFLW2000.pth",
    "checkpoint_size": 157315191,
    "checkpoint_sha256": (
        "9bdcd0e072df0d21e92288b40ea1d1b3cf7173c44728ab9c45c4d673f361e018"
    ),
    "deployed_onnx": "model_repository/sixdrepnet_onnx/1/model.onnx",
}
SIXDREPNET_RELEASE_CHECKPOINT_ENV = "SIXDREPNET_RELEASE_CHECKPOINT"

ONNX_RUNTIME_INPUT_SHAPES = {
    "adaface_ir50_webface4m_onnx": (3, 112, 112),
    "clip_image_trt": (3, 224, 224),
    "clip_text_trt": (52,),
    "face_keypoint_2d106": (3, 192, 192),
    "face_qa": (3, 112, 112),
    "osnet_ain_x1_0_onnx": (3, 256, 128),
    "sbs_r50_ibn_onnx": (3, 384, 128),
    "sixdrepnet_onnx": (3, 224, 224),
    "TBPS_clip_image_trt": (3, 224, 224),
    "TBPS_clip_text_trt": (77,),
    "w600k_r50_trt": (3, 112, 112),
    "yolo26m_pose_onnx": (3, 128, 128),
    "yolo26s_seg_onnx": (3, 128, 128),
    "yolov8n_face_onnx": (3, 64, 64),
}

ONNX_PLATFORM_FIELD = re.compile(
    r'^\s*platform\s*:\s*"onnxruntime_onnx"\s*$', re.MULTILINE
)
ONNX_FILENAME_FIELD = re.compile(
    r'^\s*default_model_filename\s*:\s*"model\.onnx"\s*$', re.MULTILINE
)
ONNX_BACKEND_FIELD = re.compile(
    r'^\s*backend\s*:\s*"onnxruntime"\s*$', re.MULTILINE
)
PYTHON_BACKEND_FIELD = re.compile(r'^\s*backend\s*:\s*"python"\s*$', re.MULTILINE)
ENSEMBLE_PLATFORM_FIELD = re.compile(
    r'^\s*platform\s*:\s*"ensemble"\s*$', re.MULTILINE
)
TENSORRT_PLATFORM_FIELD = re.compile(
    r'^\s*platform\s*:\s*"tensorrt_plan"\s*$', re.MULTILINE
)
TENSORRT_BACKEND_FIELD = re.compile(
    r'^\s*backend\s*:\s*"tensorrt"\s*$', re.MULTILINE
)


def strip_textproto_comments(config_text):
    """Remove line comments so inactive text cannot affect config assertions."""
    return re.sub(r"(?m)(?:#|//).*?$", "", config_text)


def read_config(config_path):
    return strip_textproto_comments(config_path.read_text(encoding="utf-8"))


def parse_first_tensor_block(config_text, block_name):
    """Return name, Triton data type, and dims from the first input/output block."""
    match = re.search(
        rf"\b{re.escape(block_name)}\s*(?:\[\s*)?\{{([^{{}}]*)\}}", config_text
    )
    if match is None:
        raise AssertionError(f"Missing {block_name} block")
    block = match.group(1)
    name = re.search(r'name:\s*"([^"]+)"', block)
    data_type = re.search(r"data_type:\s*(TYPE_[A-Z0-9_]+)", block)
    dims = re.search(r"dims:\s*\[([^\]]*)\]", block)
    if name is None or data_type is None or dims is None:
        raise AssertionError(f"Malformed {block_name} block: {block!r}")
    return (
        name.group(1),
        data_type.group(1),
        tuple(int(value.strip()) for value in dims.group(1).split(",") if value.strip()),
    )


def first_input(config_text):
    return parse_first_tensor_block(config_text, "input")


def first_output(config_text):
    return parse_first_tensor_block(config_text, "output")


def onnx_shape(value_info):
    return tuple(dimension.dim_value for dimension in value_info.type.tensor_type.shape.dim)


def onnx_declared_shape(value_info):
    dimensions = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(dimension.dim_value)
        elif dimension.HasField("dim_param") and dimension.dim_param:
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append(None)
    return tuple(dimensions)


def assert_dynamic_batch_sample_shape(actual_shape, expected_sample_shape):
    """Require an explicit symbolic batch plus the exact sample contract."""
    actual_shape = tuple(actual_shape)
    expected_sample_shape = tuple(expected_sample_shape)
    for dimension in expected_sample_shape:
        if not isinstance(dimension, int) or isinstance(dimension, bool):
            raise AssertionError(
                f"Invalid expected sample dimension {dimension!r}"
            )
        if dimension == 0 or dimension < -1:
            raise AssertionError(
                f"Invalid expected sample dimension {dimension!r}"
            )

    if len(actual_shape) != len(expected_sample_shape) + 1:
        raise AssertionError(
            f"ONNX rank mismatch: actual {actual_shape}, expected sample "
            f"{expected_sample_shape} with a leading batch"
        )
    if actual_shape[0] != "batch":
        raise AssertionError(
            "Leading batch dimension must be symbolic 'batch', "
            f"got {actual_shape[0]!r}"
        )
    actual_sample_shape = actual_shape[1:]

    for axis, (actual, expected) in enumerate(
        zip(actual_sample_shape, expected_sample_shape)
    ):
        if expected == -1:
            positive_static = (
                isinstance(actual, int)
                and not isinstance(actual, bool)
                and actual > 0
            )
            named_symbolic = isinstance(actual, str) and bool(actual.strip())
            if not (positive_static or named_symbolic):
                raise AssertionError(
                    f"Sample axis {axis} must be a positive static dimension "
                    f"or non-empty symbol, got {actual!r}"
                )
        elif actual != expected:
            raise AssertionError(
                f"Sample axis {axis} mismatch: expected {expected}, got {actual!r}"
            )


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_text_bytes(path):
    text = path.read_text(encoding="utf-8")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def load_face_qa_preprocessor():
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

    module_path = REPOSITORY / "face_qa_pre" / "1" / "model.py"
    spec = importlib.util.spec_from_file_location("face_qa_pre_model", module_path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get("triton_python_backend_utils")
    sys.modules["triton_python_backend_utils"] = fake_pb_utils
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("triton_python_backend_utils", None)
        else:
            sys.modules["triton_python_backend_utils"] = previous
    return module, fake_pb_utils


class OnnxSampleAxisCompatibilityHelperTests(unittest.TestCase):
    def test_accepts_only_explicit_symbolic_batch(self):
        expected = (3, 112, 112)
        assert_dynamic_batch_sample_shape(("batch", 3, 112, 112), expected)
        for actual in ((3, 112, 112), (1, 3, 112, 112)):
            with self.subTest(actual=actual), self.assertRaises(AssertionError):
                assert_dynamic_batch_sample_shape(actual, expected)

    def test_dynamic_sample_axes_accept_positive_static_or_named_symbolic_dims(self):
        expected = (3, -1, -1)
        for actual in (
            ("batch", 3, 640, 384),
            ("batch", 3, "height", "width"),
        ):
            with self.subTest(actual=actual):
                assert_dynamic_batch_sample_shape(actual, expected)

    def test_rejects_axis_order_feature_width_rank_and_non_one_static_batch(self):
        expected = (3, 112, 112)
        invalid = (
            ("batch", 112, 3, 112),
            ("batch", 3, 112, 111),
            ("batch", 3, 112),
            (2, 3, 112, 112),
        )
        for actual in invalid:
            with self.subTest(actual=actual):
                with self.assertRaises(AssertionError):
                    assert_dynamic_batch_sample_shape(actual, expected)

    def test_dynamic_sample_axes_reject_unset_empty_and_non_positive_dims(self):
        expected = (3, -1, -1)
        for bad_dimension in (None, "", 0, -1, -8):
            actual = ("batch", 3, bad_dimension, "width")
            with self.subTest(actual=actual):
                with self.assertRaises(AssertionError):
                    assert_dynamic_batch_sample_shape(actual, expected)


@unittest.skipUnless(importlib.util.find_spec("onnx"), "onnx is not installed")
class OnnxModelStructureTests(unittest.TestCase):
    def test_all_onnx_boundaries_match_independent_sample_contracts(self):
        import onnx

        self.assertEqual(set(ONNX_SAMPLE_IO), ONNX_MODELS)
        for model_name, contract in ONNX_SAMPLE_IO.items():
            with self.subTest(model=model_name):
                model = onnx.load(
                    REPOSITORY / model_name / "1" / "model.onnx",
                    load_external_data=False,
                )
                onnx.checker.check_model(model)
                for field_name, actual_values in (
                    ("inputs", model.graph.input),
                    ("outputs", model.graph.output),
                ):
                    expected_values = contract[field_name]
                    self.assertEqual(len(actual_values), len(expected_values))
                    for actual, expected in zip(actual_values, expected_values):
                        expected_name, expected_type, expected_sample_shape = expected
                        self.assertEqual(actual.name, expected_name)
                        self.assertEqual(
                            onnx.TensorProto.DataType.Name(
                                actual.type.tensor_type.elem_type
                            ),
                            expected_type,
                        )
                        assert_dynamic_batch_sample_shape(
                            onnx_declared_shape(actual),
                            expected_sample_shape,
                        )

    def test_every_graph_input_uses_exact_batch_symbol(self):
        import onnx

        for model_name in ONNX_MODELS:
            with self.subTest(model=model_name):
                model = onnx.load(
                    REPOSITORY / model_name / "1" / "model.onnx",
                    load_external_data=False,
                )
                for value_info in model.graph.input:
                    first = value_info.type.tensor_type.shape.dim[0]
                    self.assertEqual(first.dim_param, "batch")
                    self.assertFalse(first.HasField("dim_value"))

    def test_every_batch_correlated_output_uses_a_symbolic_first_axis(self):
        import onnx

        for model_name in ONNX_MODELS:
            with self.subTest(model=model_name):
                model = onnx.load(
                    REPOSITORY / model_name / "1" / "model.onnx",
                    load_external_data=False,
                )
                for value_info in model.graph.output:
                    first = value_info.type.tensor_type.shape.dim[0]
                    self.assertTrue(first.dim_param)
                    self.assertFalse(first.HasField("dim_value"))


@unittest.skipUnless(
    importlib.util.find_spec("onnxruntime"), "onnxruntime is not installed"
)
class OnnxRuntimeBatchParityTests(unittest.TestCase):
    @staticmethod
    def _make_batch(model_name, sample_shape, input_type, batch_size=4):
        if input_type == "tensor(int64)":
            batch = np.zeros((batch_size, *sample_shape), dtype=np.int64)
            for index in range(batch_size):
                if model_name == "TBPS_clip_text_trt":
                    batch[index, :3] = (49406, index + 1, 49407)
                else:
                    batch[index, :2] = (index + 1, index + 2)
            return batch
        if input_type != "tensor(float)":
            raise AssertionError(f"Unsupported runtime input type: {input_type}")
        return np.random.default_rng(20260901).standard_normal(
            (batch_size, *sample_shape), dtype=np.float32
        )

    def test_all_deployed_models_execute_true_batch_four_with_b1_parity(self):
        import onnxruntime as ort

        self.assertEqual(set(ONNX_RUNTIME_INPUT_SHAPES), ONNX_MODELS)
        for model_name, sample_shape in ONNX_RUNTIME_INPUT_SHAPES.items():
            with self.subTest(model=model_name):
                session = ort.InferenceSession(
                    str(REPOSITORY / model_name / "1" / "model.onnx"),
                    providers=["CPUExecutionProvider"],
                )
                self.assertEqual(len(session.get_inputs()), 1)
                runtime_input = session.get_inputs()[0]
                output_names = [value.name for value in session.get_outputs()]
                batch = self._make_batch(
                    model_name,
                    sample_shape,
                    runtime_input.type,
                    batch_size=4,
                )
                actual = session.run(output_names, {runtime_input.name: batch})
                per_row = [
                    session.run(
                        output_names, {runtime_input.name: row[None, ...]}
                    )
                    for row in batch
                ]
                for output_index, (output_name, batched) in enumerate(
                    zip(output_names, actual)
                ):
                    expected = np.concatenate(
                        [values[output_index] for values in per_row], axis=0
                    )
                    self.assertEqual(batched.shape, expected.shape, output_name)
                    self.assertEqual(batched.shape[0], 4, output_name)
                    self.assertTrue(np.isfinite(batched).all(), output_name)
                    np.testing.assert_allclose(
                        batched,
                        expected,
                        rtol=1e-4,
                        atol=1e-5,
                        err_msg=f"{model_name}:{output_name}",
                    )
                    for left, right in itertools.combinations(range(4), 2):
                        self.assertFalse(
                            np.allclose(
                                batched[left],
                                batched[right],
                                rtol=1e-5,
                                atol=1e-6,
                            ),
                            (
                                f"{model_name}:{output_name} copied rows "
                                f"{left}/{right}"
                            ),
                        )
                del session


@unittest.skipUnless(importlib.util.find_spec("onnx"), "onnx is not installed")
class OnnxTargetStructureTests(unittest.TestCase):
    def test_target_models_preserve_graph_io_names_and_types(self):
        import onnx

        for model_name, contract in TARGETS.items():
            with self.subTest(model=model_name):
                model = onnx.load(REPOSITORY / model_name / "1" / "model.onnx")
                onnx.checker.check_model(model)
                expected_input = contract["onnx_input"]
                expected_output = contract["onnx_output"]
                self.assertEqual(
                    len(model.graph.input), 1, "Model must have exactly one graph input"
                )
                self.assertEqual(
                    len(model.graph.output), 1, "Model must have exactly one graph output"
                )
                input_value = model.graph.input[0]
                output_value = model.graph.output[0]
                self.assertEqual(
                    (
                        input_value.name,
                        onnx.TensorProto.DataType.Name(input_value.type.tensor_type.elem_type),
                    ),
                    expected_input,
                )
                self.assertEqual(
                    (
                        output_value.name,
                        onnx.TensorProto.DataType.Name(output_value.type.tensor_type.elem_type),
                    ),
                    expected_output,
                )

    def test_target_configs_declare_exact_per_sample_shapes(self):
        for model_name, contract in TARGETS.items():
            with self.subTest(model=model_name):
                config = read_config(REPOSITORY / model_name / "config.pbtxt")
                self.assertEqual(first_input(config), contract["input"])
                self.assertEqual(first_output(config), contract["output"])


class OnnxModelRepositoryTests(unittest.TestCase):
    def test_release_metadata_is_pinned_to_lf_by_git_attributes(self):
        expected_lines = {
            ".gitattributes text eol=lf",
            "tools/model_pipeline/catalog.json text eol=lf",
        }
        attributes_text = GIT_ATTRIBUTES.read_bytes().decode("utf-8")
        self.assertEqual(set(attributes_text.splitlines()), expected_lines)
        self.assertNotIn("\r", attributes_text)

        checked_paths = (
            "tools/model_pipeline/catalog.json",
        )
        git = shutil.which("git")
        if git is None:
            self.skipTest("git is not installed")
        completed = subprocess.run(
            [git, "check-attr", "eol", "--", *checked_paths],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        expected_output = "".join(
            f"{path}: eol: lf\n" for path in checked_paths
        )
        self.assertEqual(completed.stdout, expected_output)

        for relative_path in checked_paths:
            with self.subTest(path=relative_path):
                self.assertNotIn(b"\r", (ROOT / relative_path).read_bytes())

    def test_catalog_authenticates_deployed_onnx_and_source_checkpoint(self):
        catalog = json.loads(MODEL_CATALOG.read_text(encoding="utf-8"))
        self.assertEqual(catalog["schema_version"], 1)
        self.assertEqual(set(catalog["models"]), ONNX_MODELS)
        for model_name, expected in EXPECTED_ARTIFACT_TARGETS.items():
            entry = catalog["models"][model_name]
            self.assertEqual(entry["reference_onnx"], expected["deployed_onnx"])
            self.assertEqual(entry["onnx_size"], expected["onnx_size"])
            self.assertEqual(entry["onnx_sha256"], expected["onnx_sha256"])

        for model_name, entry in catalog["models"].items():
            with self.subTest(model=model_name):
                self.assertEqual(
                    entry["reference_onnx"],
                    f"model_repository/{model_name}/1/model.onnx",
                )
                path = ROOT / entry["reference_onnx"]
                self.assertTrue(path.is_file(), f"Missing deployed ONNX: {path}")
                self.assertEqual(path.stat().st_size, entry["onnx_size"])
                self.assertEqual(sha256_file(path), entry["onnx_sha256"])

        generated = catalog["models"]["sixdrepnet_onnx"]
        source = catalog["sources"][generated["source"]]
        expected = SIXDREPNET_GENERATED_ARTIFACT
        self.assertEqual(generated["reference_onnx"], expected["deployed_onnx"])
        self.assertEqual(source["local_path"], expected["source_checkpoint"])
        self.assertEqual(source["size"], expected["checkpoint_size"])
        self.assertEqual(source["sha256"], expected["checkpoint_sha256"])
        self.assertNotEqual(
            (generated["onnx_size"], generated["onnx_sha256"]),
            (
                157321159,
                "56d532fb20e211b57a1cdb35aeee26a1928d3e5c493c4e6949da6f1341232265",
            ),
            "dynamic release must not reuse the old fixed-batch artifact",
        )

        configured_checkpoint = os.environ.get(SIXDREPNET_RELEASE_CHECKPOINT_ENV)
        if configured_checkpoint is not None:
            self.assertTrue(
                configured_checkpoint.strip(),
                f"{SIXDREPNET_RELEASE_CHECKPOINT_ENV} must not be empty when set",
            )
            checkpoint_path = Path(configured_checkpoint)
            self.assertTrue(checkpoint_path.is_file(), str(checkpoint_path))
            self.assertEqual(checkpoint_path.stat().st_size, source["size"])
            self.assertEqual(sha256_file(checkpoint_path), source["sha256"])

    @unittest.skipUnless(importlib.util.find_spec("onnx"), "onnx is not installed")
    def test_generated_sixdrepnet_onnx_has_symbolic_batch_and_static_sample_dims(self):
        import onnx

        model_path = ROOT / SIXDREPNET_GENERATED_ARTIFACT["deployed_onnx"]
        self.assertTrue(model_path.is_file(), f"Missing deployed ONNX: {model_path}")
        model = onnx.load(model_path, load_external_data=False)
        self.assertEqual(
            [(item.domain, int(item.version)) for item in model.opset_import],
            [("", 17)],
        )
        inferred = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(inferred)
        self.assertEqual(len(inferred.graph.input), 1)
        self.assertEqual(len(inferred.graph.output), 1)
        self.assertEqual(inferred.graph.input[0].name, "image")
        self.assertEqual(inferred.graph.output[0].name, "rotation_matrix")
        self.assertEqual(
            onnx_declared_shape(inferred.graph.input[0]), ("batch", 3, 224, 224)
        )
        self.assertEqual(
            onnx_declared_shape(inferred.graph.output[0]), ("batch", 3, 3)
        )

    def test_only_native_model_versions_contain_plan_artifacts(self):
        allowed = {
            f"{model}/1/{filename}"
            for model in TENSORRT_MODELS
            for filename in ("model.plan", "model_throughput.plan")
        }
        plan_artifacts = {
            path.relative_to(REPOSITORY).as_posix()
            for path in REPOSITORY.glob("*/[0-9]*/*.plan")
        }
        self.assertFalse(plan_artifacts - allowed, plan_artifacts - allowed)

        catalog_text = MODEL_CATALOG.read_text(encoding="utf-8")
        for plan_field in ("preserved_plan", "plan_sha256", "plan_size"):
            with self.subTest(field=plan_field):
                self.assertNotIn(plan_field, catalog_text)

    def test_target_configs_use_native_tensorrt_and_preserve_io_contracts(self):
        for model_name, contract in TARGETS.items():
            with self.subTest(model=model_name):
                config = read_config(REPOSITORY / model_name / "config.pbtxt")
                self.assertRegex(config, TENSORRT_BACKEND_FIELD)
                self.assertRegex(config, r'default_model_filename:\s*"model_throughput\.plan"')
                self.assertNotRegex(config, ONNX_PLATFORM_FIELD)
                self.assertNotRegex(config, ONNX_BACKEND_FIELD)
                self.assertRegex(
                    config,
                    re.compile(
                        rf'^\s*name\s*:\s*"{re.escape(model_name)}"\s*$',
                        re.MULTILINE,
                    ),
                )
                self.assertRegex(
                    config,
                    re.compile(r"^\s*max_batch_size\s*:\s*4\s*$", re.MULTILINE),
                )
                self.assertRegex(config, r"\bkind\s*:\s*KIND_GPU\b")
                self.assertEqual(first_input(config), contract["input"])
                self.assertEqual(first_output(config), contract["output"])

    def test_repository_model_classification_is_exact(self):
        actual = {"onnx": set(), "python": set(), "ensemble": set(), "tensorrt": set()}
        for config_path in REPOSITORY.glob("*/config.pbtxt"):
            config = read_config(config_path)
            model_name = config_path.parent.name
            if ONNX_PLATFORM_FIELD.search(config) or ONNX_BACKEND_FIELD.search(config):
                actual["onnx"].add(model_name)
            elif PYTHON_BACKEND_FIELD.search(config):
                actual["python"].add(model_name)
            elif ENSEMBLE_PLATFORM_FIELD.search(config):
                actual["ensemble"].add(model_name)
            elif TENSORRT_PLATFORM_FIELD.search(config) or TENSORRT_BACKEND_FIELD.search(config):
                actual["tensorrt"].add(model_name)
            else:
                self.fail(f"Unclassified model config: {config_path}")

        self.assertEqual(actual["onnx"], ONNX_MODELS - TENSORRT_MODELS)
        self.assertEqual(actual["tensorrt"], TENSORRT_MODELS)
        self.assertEqual(actual["python"], PYTHON_MODELS)
        self.assertEqual(actual["ensemble"], ENSEMBLE_MODELS)

    def test_every_ensemble_has_a_trackable_version_one_directory(self):
        for model_name in ENSEMBLE_MODELS:
            with self.subTest(ensemble=model_name):
                version_directory = REPOSITORY / model_name / "1"
                self.assertTrue(
                    version_directory.is_dir(),
                    f"Missing ensemble version directory: {version_directory}",
                )
                self.assertTrue(
                    (version_directory / ".gitkeep").is_file(),
                    f"Ensemble version directory is not trackable: {version_directory}",
                )

    def test_only_selected_model_configs_activate_tensorrt(self):
        for config_path in REPOSITORY.glob("*/config.pbtxt"):
            with self.subTest(config=config_path.parent.name):
                config = read_config(config_path)
                native = bool(TENSORRT_PLATFORM_FIELD.search(config) or TENSORRT_BACKEND_FIELD.search(config))
                self.assertEqual(native, config_path.parent.name in TENSORRT_MODELS)

    def test_target_onnx_deployment_artifacts_exist(self):
        for model_name in TARGETS:
            with self.subTest(model=model_name):
                self.assertTrue(
                    (REPOSITORY / model_name / "1" / "model.onnx").is_file(),
                    "Missing deployment artifact 1/model.onnx",
                )

@unittest.skipUnless(importlib.util.find_spec("onnx"), "onnx is not installed")
class OsnetAinPipelineContractTests(unittest.TestCase):
    def test_osnet_ain_contract_matches_onnx_and_ensemble(self):
        import onnx

        onnx_config = read_config(
            REPOSITORY / "osnet_ain_x1_0_onnx" / "config.pbtxt"
        )
        pre_config = read_config(
            REPOSITORY / "osnet_ain_x1_0_image_preprocess" / "config.pbtxt"
        )
        ensemble = read_config(
            REPOSITORY / "osnet_ain_x1_0_image_ensemble" / "config.pbtxt"
        )
        for config in (onnx_config, pre_config, ensemble):
            self.assertRegex(
                config,
                re.compile(
                    r"^\s*max_batch_size\s*:\s*4\s*$",
                    re.MULTILINE,
                ),
            )

        self.assertRegex(pre_config, PYTHON_BACKEND_FIELD)
        self.assertEqual(
            first_input(pre_config),
            ("IMAGE_BYTES", "TYPE_UINT8", (-1,)),
        )
        self.assertRegex(pre_config, r"\bkind\s*:\s*KIND_CPU\b")
        self.assertRegex(pre_config, r"\bcount\s*:\s*1\b")

        self.assertRegex(onnx_config, ONNX_PLATFORM_FIELD)
        self.assertRegex(onnx_config, ONNX_FILENAME_FIELD)
        self.assertRegex(onnx_config, r"\bkind\s*:\s*KIND_GPU\b")
        self.assertRegex(onnx_config, r"\bcount\s*:\s*1\b")
        self.assertRegex(onnx_config, r"\bgpus\s*:\s*\[\s*0\s*\]")
        self.assertEqual(
            first_input(onnx_config),
            ("image", "TYPE_FP32", (3, 256, 128)),
        )

        self.assertRegex(ensemble, ENSEMBLE_PLATFORM_FIELD)
        self.assertEqual(
            first_input(ensemble),
            ("IMAGE_BYTES", "TYPE_UINT8", (-1,)),
        )
        self.assertEqual(
            first_output(ensemble),
            ("embedding", "TYPE_FP32", (512,)),
        )
        self.assertEqual(first_output(pre_config), first_input(onnx_config))
        self.assertEqual(
            first_output(onnx_config),
            ("embedding", "TYPE_FP32", (512,)),
        )
        self.assertIn(
            'model_name: "osnet_ain_x1_0_image_preprocess"',
            ensemble,
        )
        self.assertIn('model_name: "osnet_ain_x1_0_onnx"', ensemble)
        for mapping in (
            'key: "IMAGE_BYTES"\n      value: "IMAGE_BYTES"',
            'key: "image"\n      value: "preprocessed_image"',
            'key: "embedding"\n      value: "embedding"',
        ):
            self.assertIn(mapping, ensemble)

        model_path = (
            REPOSITORY / "osnet_ain_x1_0_onnx" / "1" / "model.onnx"
        )
        model = onnx.load(
            model_path,
            load_external_data=False,
        )
        onnx.checker.check_model(model)
        self.assertEqual(len(model.graph.input), 1)
        self.assertEqual(len(model.graph.output), 1)
        self.assertEqual(model.graph.input[0].name, "image")
        self.assertEqual(model.graph.output[0].name, "embedding")
        self.assertEqual(
            model.graph.input[0].type.tensor_type.elem_type,
            onnx.TensorProto.FLOAT,
        )
        self.assertEqual(
            model.graph.output[0].type.tensor_type.elem_type,
            onnx.TensorProto.FLOAT,
        )
        self.assertEqual(onnx_shape(model.graph.input[0]), (0, 3, 256, 128))
        self.assertEqual(onnx_shape(model.graph.output[0]), (0, 512))
        identity = json.loads(MODEL_CATALOG.read_text(encoding="utf-8"))["models"][
            "osnet_ain_x1_0_onnx"
        ]
        self.assertEqual(sha256_file(model_path), identity["onnx_sha256"])
        self.assertEqual(model_path.stat().st_size, identity["onnx_size"])


class FaceQaPipelineContractTests(unittest.TestCase):
    def test_preprocessor_declared_output_matches_face_qa_input(self):
        pre_config = read_config(REPOSITORY / "face_qa_pre" / "config.pbtxt")
        inference_config = read_config(REPOSITORY / "face_qa" / "config.pbtxt")

        self.assertEqual(first_output(pre_config)[1:], first_input(inference_config)[1:])

    def test_preprocessor_runtime_output_matches_face_qa_input(self):
        import numpy as np

        module, fake_pb_utils = load_face_qa_preprocessor()
        raw_image = np.zeros((1, 32, 48, 3), dtype=np.uint8)
        landmark = np.array(
            [[[12, 12], [36, 12], [24, 18], [16, 25], [32, 25]]],
            dtype=np.float32,
        )
        request = {
            "RAW_FACE_IMAGE": fake_pb_utils.Tensor("RAW_FACE_IMAGE", raw_image),
            "LANDMARK": fake_pb_utils.Tensor("LANDMARK", landmark),
        }

        response = module.TritonPythonModel().execute([request])[0]
        output = response.output_tensors[0].as_numpy()

        inference_config = read_config(REPOSITORY / "face_qa" / "config.pbtxt")
        self.assertEqual(output.shape, (1, *first_input(inference_config)[2]))


@unittest.skipUnless(importlib.util.find_spec("onnx"), "onnx is not installed")
class AdaFacePipelineContractTests(unittest.TestCase):
    @staticmethod
    def _top_level_field_value_starts(config_text, field_name):
        starts = []
        brace_depth = 0
        bracket_depth = 0
        in_string = False
        escaped = False
        index = 0
        while index < len(config_text):
            character = config_text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                index += 1
                continue

            if character == '"':
                in_string = True
            elif character == "{":
                brace_depth += 1
            elif character == "}":
                brace_depth -= 1
            elif character == "[":
                bracket_depth += 1
            elif character == "]":
                bracket_depth -= 1
            elif (
                brace_depth == 0
                and bracket_depth == 0
                and (character.isalpha() or character == "_")
            ):
                end = index + 1
                while end < len(config_text) and (
                    config_text[end].isalnum() or config_text[end] == "_"
                ):
                    end += 1
                if config_text[index:end] == field_name:
                    starts.append(end)
                index = end
                continue
            index += 1
        return starts

    @classmethod
    def _top_level_quoted_field_values(cls, config_text, field_name):
        values = []
        for start in cls._top_level_field_value_starts(config_text, field_name):
            match = re.match(r'\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', config_text[start:])
            if match is None:
                raise AssertionError(f"Malformed top-level {field_name} field")
            values.append(match.group(1))
        return values

    @classmethod
    def _replace_top_level_quoted_field(
        cls,
        config_text,
        field_name,
        replacement,
    ):
        value_spans = []
        for start in cls._top_level_field_value_starts(config_text, field_name):
            match = re.match(
                r'\s*:\s*"(?P<value>[^"\\]*(?:\\.[^"\\]*)*)"',
                config_text[start:],
            )
            if match is not None:
                value_spans.append(
                    (
                        start + match.start("value"),
                        start + match.end("value"),
                    )
                )

        updated = config_text
        for value_start, value_end in reversed(value_spans):
            updated = updated[:value_start] + replacement + updated[value_end:]
        return updated, len(value_spans)

    @classmethod
    def _top_level_message_count(cls, config_text, field_name):
        count = 0
        for start in cls._top_level_field_value_starts(config_text, field_name):
            cursor = start
            while cursor < len(config_text) and config_text[cursor].isspace():
                cursor += 1
            if cursor < len(config_text) and config_text[cursor] == ":":
                cursor += 1
                while cursor < len(config_text) and config_text[cursor].isspace():
                    cursor += 1

            if cursor < len(config_text) and config_text[cursor] == "{":
                count += 1
                continue
            if cursor >= len(config_text) or config_text[cursor] != "[":
                raise AssertionError(f"Malformed top-level {field_name} block")

            bracket_depth = 1
            brace_depth = 0
            in_string = False
            escaped = False
            cursor += 1
            while cursor < len(config_text) and bracket_depth:
                character = config_text[cursor]
                if in_string:
                    if escaped:
                        escaped = False
                    elif character == "\\":
                        escaped = True
                    elif character == '"':
                        in_string = False
                elif character == '"':
                    in_string = True
                elif character == "{":
                    if bracket_depth == 1 and brace_depth == 0:
                        count += 1
                    brace_depth += 1
                elif character == "}":
                    brace_depth -= 1
                elif character == "[":
                    bracket_depth += 1
                elif character == "]":
                    bracket_depth -= 1
                cursor += 1
            if bracket_depth:
                raise AssertionError(f"Unterminated top-level {field_name} block")
        return count

    def assert_top_level_config_contract(
        self,
        config_text,
        expected_name,
        expected_input_count,
        expected_output_count,
    ):
        self.assertEqual(
            self._top_level_quoted_field_values(config_text, "name"),
            [expected_name],
        )
        self.assertEqual(
            self._top_level_message_count(config_text, "input"),
            expected_input_count,
        )
        self.assertEqual(
            self._top_level_message_count(config_text, "output"),
            expected_output_count,
        )

    def test_config_wiring_matches_adaface_pipeline_contract(self):
        onnx_config = read_config(
            REPOSITORY / "adaface_ir50_webface4m_onnx" / "config.pbtxt"
        )
        pre_config = read_config(REPOSITORY / "adaface_preprocess" / "config.pbtxt")
        ensemble = read_config(
            REPOSITORY / "adaface_recognition_pipeline" / "config.pbtxt"
        )

        self.assert_top_level_config_contract(
            onnx_config,
            "adaface_ir50_webface4m_onnx",
            expected_input_count=1,
            expected_output_count=1,
        )
        self.assert_top_level_config_contract(
            pre_config,
            "adaface_preprocess",
            expected_input_count=2,
            expected_output_count=1,
        )
        self.assert_top_level_config_contract(
            ensemble,
            "adaface_recognition_pipeline",
            expected_input_count=2,
            expected_output_count=1,
        )

        spaced_name_config, spacing_replacement_count = re.subn(
            r"(?m)^name\s*:",
            "name :",
            onnx_config,
            count=1,
        )
        self.assertEqual(spacing_replacement_count, 1)
        wrong_top_level_name, name_replacement_count = (
            self._replace_top_level_quoted_field(
                spaced_name_config,
                "name",
                "wrong_model",
            )
        )
        self.assertEqual(name_replacement_count, 1)
        self.assertEqual(
            wrong_top_level_name.count(
                'name: "adaface_ir50_webface4m_onnx"'
            ),
            1,
        )
        extra_input = (
            onnx_config
            + '\ninput [{ name: "extra" data_type: TYPE_FP32 dims: [1] }]\n'
        )
        extra_output = (
            onnx_config
            + '\noutput [{ name: "extra" data_type: TYPE_FP32 dims: [1] }]\n'
        )
        for corruption, corrupted_config in (
            ("wrong top-level name", wrong_top_level_name),
            ("extra input", extra_input),
            ("extra output", extra_output),
        ):
            with self.subTest(corruption=corruption):
                with self.assertRaises(AssertionError):
                    self.assert_top_level_config_contract(
                        corrupted_config,
                        "adaface_ir50_webface4m_onnx",
                        expected_input_count=1,
                        expected_output_count=1,
                    )

        for config in (onnx_config, pre_config, ensemble):
            self.assertRegex(
                config,
                re.compile(r"^\s*max_batch_size\s*:\s*4\s*$", re.MULTILINE),
            )

        self.assertRegex(pre_config, PYTHON_BACKEND_FIELD)
        self.assertEqual(
            first_input(pre_config),
            ("RAW_IMAGE", "TYPE_UINT8", (-1, -1, 3)),
        )
        self.assertRegex(
            pre_config,
            re.compile(
                r'\binput\s*\[\s*\{[^{}]*\}\s*,?\s*\{'
                r'(?=[^{}]*\bname\s*:\s*"LANDMARK")'
                r'(?=[^{}]*\bdata_type\s*:\s*TYPE_FP32)'
                r'(?=[^{}]*\bdims\s*:\s*\[\s*5\s*,\s*2\s*\])'
                r"[^{}]*\}\s*\]",
                re.DOTALL,
            ),
        )
        self.assertEqual(
            first_output(pre_config),
            ("input", "TYPE_FP32", (3, 112, 112)),
        )

        self.assertRegex(onnx_config, ONNX_PLATFORM_FIELD)
        self.assertRegex(onnx_config, ONNX_FILENAME_FIELD)
        self.assertEqual(first_output(pre_config), first_input(onnx_config))
        self.assertEqual(
            first_output(onnx_config),
            ("embedding", "TYPE_FP32", (512,)),
        )
        self.assertRegex(
            onnx_config,
            re.compile(
                r'\binstance_group\s*\{(?=[^{}]*\bname\s*:\s*'
                r'"adaface_ir50_webface4m_onnx")'
                r"(?=[^{}]*\bkind\s*:\s*KIND_GPU)"
                r"(?=[^{}]*\bcount\s*:\s*1)"
                r"(?=[^{}]*\bgpus\s*:\s*\[\s*0\s*\])[^{}]*\}",
                re.DOTALL,
            ),
        )

        self.assertRegex(ensemble, ENSEMBLE_PLATFORM_FIELD)
        self.assertEqual(
            first_input(ensemble),
            ("RAW_IMAGE", "TYPE_UINT8", (-1, -1, 3)),
        )
        self.assertRegex(
            ensemble,
            re.compile(
                r'\binput\s*\[\s*\{[^{}]*\}\s*,?\s*\{'
                r'(?=[^{}]*\bname\s*:\s*"LANDMARK")'
                r'(?=[^{}]*\bdata_type\s*:\s*TYPE_FP32)'
                r'(?=[^{}]*\bdims\s*:\s*\[\s*5\s*,\s*2\s*\])'
                r"[^{}]*\}\s*\]",
                re.DOTALL,
            ),
        )
        self.assertEqual(
            first_output(ensemble),
            ("EMBEDDING", "TYPE_FP32", (512,)),
        )
        self.assertEqual(
            re.findall(r'\bmodel_name\s*:\s*"([^"]+)"', ensemble),
            ["adaface_preprocess", "adaface_ir50_webface4m_onnx"],
        )
        self.assertEqual(
            re.findall(r"\bmodel_version\s*:\s*(-?\d+)", ensemble),
            ["-1", "-1"],
        )
        self.assertEqual(
            re.findall(
                r'\{\s*key\s*:\s*"([^"]+)"\s+value\s*:\s*"([^"]+)"\s*\}',
                ensemble,
            ),
            [
                ("RAW_IMAGE", "RAW_IMAGE"),
                ("LANDMARK", "LANDMARK"),
                ("input", "adaface_input"),
                ("input", "adaface_input"),
                ("embedding", "EMBEDDING"),
            ],
        )

    def test_onnx_artifact_graph_matches_adaface_contract(self):
        import onnx

        model = onnx.load(
            REPOSITORY
            / "adaface_ir50_webface4m_onnx"
            / "1"
            / "model.onnx",
            load_external_data=False,
        )
        onnx.checker.check_model(model)
        self.assertEqual(len(model.graph.input), 1)
        self.assertEqual(len(model.graph.output), 1)

        graph_input = model.graph.input[0]
        graph_output = model.graph.output[0]
        self.assertEqual(graph_input.name, "input")
        self.assertEqual(graph_output.name, "embedding")
        self.assertEqual(
            graph_input.type.tensor_type.elem_type,
            onnx.TensorProto.FLOAT,
        )
        self.assertEqual(
            graph_output.type.tensor_type.elem_type,
            onnx.TensorProto.FLOAT,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
