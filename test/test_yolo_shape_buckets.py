"""Fixed rectangle selection and coordinate recovery for the three YOLO chains."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import cv2  # Initialize native modules before temporarily patching sys.modules.
import numpy as np

from test.test_batch_python_models import (
    FakeTensor,
    make_fake_pb_utils,
    response_output,
)


REPOSITORY = Path(__file__).resolve().parents[1] / "model_repository"
YOLO_MODELS = ("yolov8n_face", "yolo26m_pose", "yolo26s_seg")
EXPECTED_SHAPES = {
    (192, 640), (384, 640), (480, 640), (640, 640),
    (640, 480), (640, 384), (640, 192),
}


def load_model(name):
    spec = importlib.util.spec_from_file_location(
        f"shape_buckets_{name}", REPOSITORY / name / "1" / "model.py"
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules, {"triton_python_backend_utils": make_fake_pb_utils()}
    ):
        spec.loader.exec_module(module)
    model = module.TritonPythonModel()
    model.initialize({"model_config": json.dumps({"max_batch_size": 4})})
    return module, model


class YoloShapeBucketTests(unittest.TestCase):
    def test_batch_four_uses_fixed_rectangles_without_reducing_content_scale(self):
        # Original H/W, expected model H/W, scale, left/top padding.
        cases = (
            ((180, 640), (192, 640), 1.0, (0, 6)),
            ((101, 200), (384, 640), 3.2, (0, 30)),
            ((480, 640), (480, 640), 1.0, (0, 0)),
            ((512, 640), (640, 640), 1.0, (0, 64)),
            ((120, 80), (640, 480), 640 / 120, (26, 0)),
            ((200, 101), (640, 384), 3.2, (30, 0)),
            ((640, 180), (640, 192), 1.0, (6, 0)),
        )
        for name in YOLO_MODELS:
            _, model = load_model(f"{name}_preprocess")
            for (height, width), target, ratio, (left, top) in cases:
                with self.subTest(model=name, image=(height, width)):
                    images = np.empty((4, height, width, 3), dtype=np.uint8)
                    for index in range(4):
                        images[index] = [10 + index, 20 + index, 30 + index]
                    response = model.execute([
                        {"RAW_IMAGE": FakeTensor("RAW_IMAGE", images)}
                    ])[0]
                    self.assertIsNone(response.error)
                    actual = response_output(response, "PREPROCESSED_IMAGE")
                    meta = response_output(response, "LETTERBOX_META")
                    self.assertEqual(actual.shape, (4, 3, *target))
                    self.assertEqual(actual.dtype, np.float32)
                    self.assertTrue(actual.flags.c_contiguous)
                    expected_meta = [height, width, ratio, left, top]
                    if name != "yolov8n_face":
                        expected_meta.extend(target)
                    np.testing.assert_allclose(
                        meta, np.tile(expected_meta, (4, 1)), rtol=1e-6, atol=1e-6
                    )
                    for index in range(4):
                        np.testing.assert_allclose(
                            actual[index, :, top, left],
                            np.array([30 + index, 20 + index, 10 + index]) / 255,
                            rtol=0, atol=1e-7,
                        )
                    resized = cv2.resize(
                        images[0], (round(width * ratio), round(height * ratio))
                    )[..., ::-1].transpose(2, 0, 1) / 255
                    np.testing.assert_allclose(
                        actual[0, :, top:top + resized.shape[1], left:left + resized.shape[2]],
                        resized, rtol=0, atol=1e-7,
                    )

    def test_bucket_boundaries_cover_landscape_and_portrait_with_seven_shapes(self):
        boundaries = (
            (191, 192), (192, 192), (193, 384),
            (383, 384), (384, 384), (385, 480),
            (479, 480), (480, 480), (481, 640), (640, 640),
        )
        for name in YOLO_MODELS:
            _, model = load_model(f"{name}_preprocess")
            observed_shapes = set()
            for original_short, target_short in boundaries:
                for portrait in (False, True):
                    original = (640, original_short) if portrait else (original_short, 640)
                    target = (640, target_short) if portrait else (target_short, 640)
                    with self.subTest(model=name, image=original):
                        image, _ = model._letterbox(np.zeros((*original, 3), np.uint8))
                        self.assertEqual(image.shape[:2], target)
                        observed_shapes.add(image.shape[:2])
            self.assertEqual(observed_shapes, EXPECTED_SHAPES)

    def test_postprocessors_recover_coordinates_after_additional_bucket_padding(self):
        cases = (
            ([101, 200, 3.2, 0, 30, 384, 640], [20, 10, 100, 60], [40, 30]),
            ([120, 80, 640 / 120, 26, 0, 640, 480], [5, 20, 60, 100], [30, 40]),
        )
        for name in YOLO_MODELS:
            _, model = load_model(f"{name}_postprocess")
            for values, original_box, original_keypoint in cases:
                with self.subTest(model=name, image=values[:2]):
                    meta = np.array(values, dtype=np.float32)
                    box = np.array(original_box, np.float32) * values[2]
                    box[[0, 2]] += values[3]
                    box[[1, 3]] += values[4]
                    if name == "yolov8n_face":
                        raw = np.zeros((5, 6), np.float32)
                        raw[:, 0] = [
                            (box[0] + box[2]) / 2, (box[1] + box[3]) / 2,
                            box[2] - box[0], box[3] - box[1], 0.9,
                        ]
                        detections, count = model._postprocess(raw, meta[:5])
                    elif name == "yolo26m_pose":
                        raw = np.zeros((1, 57), np.float32)
                        raw[0, :6] = [*box, 0.9, 0]
                        keypoint = np.array(original_keypoint) * values[2] + values[3:5]
                        raw[0, 6:] = np.tile([*keypoint, 0.8], 17)
                        detections, keypoints, count = model._postprocess(raw, meta)
                        np.testing.assert_allclose(
                            keypoints[0, :, :2], np.tile(original_keypoint, (17, 1)),
                            rtol=0, atol=1e-4,
                        )
                    else:
                        raw = np.zeros((1, 38), np.float32)
                        raw[0, :7] = [*box, 0.9, 0, 1]
                        proto = np.zeros((32, values[5] // 4, values[6] // 4), np.float32)
                        proto[0] = 10
                        detections, masks, count = model._postprocess(raw, proto, meta)
                        self.assertEqual(masks[0].shape, (160, 20))
                        binary_mask = np.unpackbits(
                            masks[0], axis=-1, count=160, bitorder="big"
                        )
                        np.testing.assert_array_equal(binary_mask, np.ones((160, 160), np.uint8))
                    self.assertEqual(int(count[0]), 1)
                    np.testing.assert_allclose(detections[0, :4], original_box, rtol=0, atol=1e-4)

    def test_postprocessors_reject_old_stride_only_padding(self):
        legacy_meta = np.array([101, 200, 3.2, 0, 14, 352, 640], np.float32)
        for name in YOLO_MODELS:
            with self.subTest(model=name):
                module, _ = load_model(f"{name}_postprocess")
                meta = legacy_meta[:5] if name == "yolov8n_face" else legacy_meta
                with self.assertRaisesRegex(ValueError, "padding|input dimensions"):
                    module._validate_meta(meta)


if __name__ == "__main__":
    unittest.main()
