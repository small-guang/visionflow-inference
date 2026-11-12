import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
META_32 = np.array([32, 32, 20, 0, 0, 640, 640], dtype=np.float32)
META_640 = np.array([640, 640, 1, 0, 0, 640, 640], dtype=np.float32)
FACE_META_32 = META_640[:5].copy()


def load_postprocess_module(module_name, relative_path):
    sys.modules.setdefault("triton_python_backend_utils", types.ModuleType("triton_python_backend_utils"))
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


POSE_MODULE = load_postprocess_module(
    "yolo26m_pose_postprocess_model",
    "model_repository/yolo26m_pose_postprocess/1/model.py",
)
FACE_MODULE = load_postprocess_module(
    "yolov8n_face_postprocess_model",
    "model_repository/yolov8n_face_postprocess/1/model.py",
)
SEG_MODULE = load_postprocess_module(
    "yolo26s_seg_postprocess_model",
    "model_repository/yolo26s_seg_postprocess/1/model.py",
)


def load_seg_client_module():
    tritonclient = types.ModuleType("tritonclient")
    http = types.ModuleType("tritonclient.http")
    tritonclient.http = http
    previous_client = sys.modules.get("tritonclient")
    previous_http = sys.modules.get("tritonclient.http")
    sys.modules["tritonclient"] = tritonclient
    sys.modules["tritonclient.http"] = http
    try:
        spec = importlib.util.spec_from_file_location(
            "yolo26s_seg_image_client", ROOT / "test/test_yolo26s_seg_image.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_client is None:
            sys.modules.pop("tritonclient", None)
        else:
            sys.modules["tritonclient"] = previous_client
        if previous_http is None:
            sys.modules.pop("tritonclient.http", None)
        else:
            sys.modules["tritonclient.http"] = previous_http


SEG_CLIENT_MODULE = load_seg_client_module()


class FakeTensor:
    def __init__(self, name, value):
        self.name = name
        self._value = value

    def as_numpy(self):
        return self._value


class FakeInferenceResponse:
    def __init__(self, output_tensors=None, error=None):
        self.output_tensors = output_tensors or []
        self.error = error


class FakeTritonError(Exception):
    pass


def install_fake_pb_utils(module):
    module.pb_utils.Tensor = FakeTensor
    module.pb_utils.InferenceResponse = FakeInferenceResponse
    module.pb_utils.TritonError = FakeTritonError
    module.pb_utils.get_input_tensor_by_name = lambda request, name: request.get(name)


def request(**inputs):
    return {name: FakeTensor(name, value) for name, value in inputs.items()}


def output_map(response):
    return {tensor.name: tensor.as_numpy() for tensor in response.output_tensors}


def independent_letterbox_meta(original_h, original_w):
    ratio = min(640 / original_h, 640 / original_w)
    resized_w = round(original_w * ratio)
    resized_h = round(original_h * ratio)
    target_w = min(size for size in (192, 384, 480, 640) if size >= resized_w)
    target_h = min(size for size in (192, 384, 480, 640) if size >= resized_h)
    half_pad_w = (target_w - resized_w) / 2
    half_pad_h = (target_h - resized_h) / 2
    left = round(half_pad_w - 0.1)
    right = round(half_pad_w + 0.1)
    top = round(half_pad_h - 0.1)
    bottom = round(half_pad_h + 0.1)
    return np.array(
        [
            original_h,
            original_w,
            ratio,
            left,
            top,
            resized_h + top + bottom,
            resized_w + left + right,
        ],
        dtype=np.float32,
    )


def initialized_model(module):
    model = module.TritonPythonModel()
    model.initialize({})
    return model


class PoseEndToEndPostprocessTest(unittest.TestCase):
    def setUp(self):
        self.model = initialized_model(POSE_MODULE)

    def test_highly_overlapping_instances_are_both_retained_and_aligned(self):
        raw = np.zeros((2, 57), dtype=np.float32)
        raw[0, :6] = [1, 1, 29, 29, 0.8, 0]
        raw[1, :6] = [2, 2, 30, 30, 0.9, 0]
        raw[0, 6:] = np.tile([5, 6, 0.7], 17)
        raw[1, 6:] = np.tile([20, 21, 0.95], 17)

        detections, keypoints, num_detections = self.model._postprocess(raw, META_640)

        self.assertEqual(int(num_detections[0]), 2)
        np.testing.assert_allclose(detections[:2, 4], [0.9, 0.8])
        np.testing.assert_allclose(detections[0, :4], [2, 2, 30, 30])
        np.testing.assert_allclose(keypoints[0, 0], [20, 21, 0.95])
        np.testing.assert_allclose(keypoints[1, 0], [5, 6, 0.7])

    def test_results_are_sorted_and_limited_to_top_100(self):
        count = 120
        scores = np.linspace(0.5, 0.99, count, dtype=np.float32)
        order = np.random.default_rng(7).permutation(count)
        raw = np.zeros((count, 57), dtype=np.float32)
        raw[:, :4] = [10, 10, 20, 20]
        raw[:, 4] = scores[order]
        raw[:, 5] = 0
        raw[:, 6:] = np.tile([15, 15, 0.9], 17)

        detections, _, num_detections = self.model._postprocess(raw, META_640)

        self.assertEqual(int(num_detections[0]), 100)
        np.testing.assert_allclose(detections[:100, 4], np.sort(scores)[::-1][:100])

    def test_confidence_invalid_boxes_and_empty_output(self):
        raw = np.zeros((2, 57), dtype=np.float32)
        raw[0, :6] = [1, 1, 10, 10, 0.49, 0]
        raw[1, :6] = [10, 10, 5, 5, 0.9, 0]

        detections, keypoints, num_detections = self.model._postprocess(raw, META_640)
        self.assertEqual(int(num_detections[0]), 0)
        self.assertFalse(np.any(detections))
        self.assertFalse(np.any(keypoints))

        detections, keypoints, num_detections = self.model._postprocess(
            np.zeros((0, 57), dtype=np.float32),
            META_640,
        )
        self.assertEqual(int(num_detections[0]), 0)
        self.assertEqual(detections.shape, (100, 6))
        self.assertEqual(keypoints.shape, (100, 17, 3))

    def test_non_end_to_end_layouts_are_rejected(self):
        for shape in ((2, 56), (56, 8400), (1, 2, 57)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "end-to-end layout"):
                    self.model._postprocess(np.zeros(shape, dtype=np.float32), META_640)

    def test_helper_preserves_inputs_and_rejects_nonfinite_values(self):
        raw = np.zeros((1, 57), dtype=np.float32)
        raw[0, :6] = [2, 3, 20, 21, 0.9, 0]
        raw[0, 6:] = np.tile([4, 5, 0.8], 17)
        meta = META_640.copy()
        original_raw = raw.copy()
        original_meta = meta.copy()

        self.model._postprocess(raw, meta)

        np.testing.assert_array_equal(raw, original_raw)
        np.testing.assert_array_equal(meta, original_meta)
        raw[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            self.model._postprocess(raw, meta)
        raw[0, 0] = 2
        meta[2] = 0
        with self.assertRaisesRegex(ValueError, "ratio"):
            self.model._postprocess(raw, meta)


class FacePostprocessMathTest(unittest.TestCase):
    def setUp(self):
        self.model = initialized_model(FACE_MODULE)

    def test_nms_scaling_and_helper_input_immutability(self):
        raw = np.zeros((5, 6), dtype=np.float32)
        raw[:, 0] = [16, 16, 10, 10, 0.9]
        raw[:, 1] = [16, 16, 10, 10, 0.8]
        meta = FACE_META_32.copy()
        original_raw = raw.copy()
        original_meta = meta.copy()

        detections, num_detections = self.model._postprocess(raw, meta)

        self.assertEqual(int(num_detections[0]), 1)
        np.testing.assert_allclose(
            detections[0], [11, 11, 21, 21, 0.9, 0]
        )
        np.testing.assert_array_equal(raw, original_raw)
        np.testing.assert_array_equal(meta, original_meta)

    def test_nonfinite_prediction_and_unsafe_meta_are_rejected(self):
        raw = np.zeros((5, 6), dtype=np.float32)
        raw[:, 0] = [16, 16, 10, 10, np.nan]
        with self.assertRaisesRegex(ValueError, "finite"):
            self.model._postprocess(raw, FACE_META_32)

        meta = FACE_META_32.copy()
        meta[2] = 0
        with self.assertRaisesRegex(ValueError, "ratio"):
            self.model._postprocess(np.nan_to_num(raw), meta)

    def test_n_by_five_and_ambiguous_five_by_five_are_rejected(self):
        for shape in ((6, 5), (5, 5), (1, 5, 6)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, r"\[5,N\]"):
                    self.model._postprocess(
                        np.zeros(shape, dtype=np.float32), FACE_META_32
                    )


class SegmentEndToEndPostprocessTest(unittest.TestCase):
    def setUp(self):
        self.model = initialized_model(SEG_MODULE)
        install_fake_pb_utils(SEG_MODULE)
        self.proto = np.zeros((32, 8, 8), dtype=np.float32)
        self.proto[0] = 10

    @staticmethod
    def prediction_batch(batch_size=1, prediction_count=2):
        return np.zeros((batch_size, prediction_count, 38), dtype=np.float32)

    def test_highly_overlapping_instances_are_both_retained_and_aligned(self):
        raw = self.prediction_batch()[0]
        raw[0, :6] = [20, 20, 580, 580, 0.8, 0]
        raw[1, :6] = [40, 40, 600, 600, 0.9, 0]
        raw[0, 6] = 1
        raw[1, 6] = -1

        detections, mask_crops, num_detections = self.model._postprocess(
            raw, self.proto, META_32
        )

        self.assertEqual(int(num_detections[0]), 2)
        np.testing.assert_allclose(detections[:2, 4], [0.9, 0.8])
        np.testing.assert_allclose(detections[0, :4], [2, 2, 30, 30])
        self.assertEqual(detections.shape, (2, 6))
        self.assertEqual(mask_crops.shape, (2, 160, 20))
        self.assertEqual(int(mask_crops[0].sum()), 0)
        np.testing.assert_array_equal(
            mask_crops[1], np.full((160, 20), 255, dtype=np.uint8)
        )

    def test_only_person_instances_get_mask_crops_and_keep_alignment(self):
        raw = self.prediction_batch(prediction_count=4)[0]
        raw[:, :6] = [
            [20, 20, 200, 200, 0.99, 56],
            [400, 400, 600, 600, 0.6, 0],
            [200, 200, 400, 400, 0.98, 41],
            [0, 0, 640, 640, 0.9, 0],
        ]
        raw[:, 6] = [5, 1, 6, -1]

        with mock.patch.object(
            self.model,
            "_render_mask_crops_into",
            wraps=self.model._render_mask_crops_into,
        ) as render_masks:
            detections, mask_crops, count = self.model._postprocess(
                raw, self.proto, META_32
            )

        self.assertEqual(int(count[0]), 2)
        np.testing.assert_allclose(
            detections,
            [[0, 0, 32, 32, 0.9, 0], [20, 20, 30, 30, 0.6, 0]],
        )
        self.assertEqual(mask_crops.shape, (2, 160, 20))
        self.assertFalse(np.any(mask_crops[0]))
        np.testing.assert_array_equal(
            mask_crops[1], np.full((160, 20), 255, dtype=np.uint8)
        )
        render_masks.assert_called_once()
        np.testing.assert_array_equal(
            render_masks.call_args.args[2], raw[[3, 1], 6:]
        )

    def test_non_person_instances_do_not_consume_top_100_slots(self):
        raw = self.prediction_batch(prediction_count=103)[0]
        raw[:, :4] = [40, 40, 600, 600]
        raw[:100, 4] = 0.99
        raw[:100, 5] = np.arange(100) % 79 + 1
        raw[100:, 4] = [0.5, 0.8, 0.49]
        raw[:, 6] = 1

        detections, mask_crops, count = self.model._postprocess(
            raw, self.proto, META_32
        )

        self.assertEqual(int(count[0]), 2)
        self.assertEqual(detections.shape, (2, 6))
        np.testing.assert_allclose(detections[:, 4], [0.8, 0.5])
        np.testing.assert_array_equal(detections[:, 5], [0, 0])
        np.testing.assert_array_equal(
            mask_crops, np.full((2, 160, 20), 255, dtype=np.uint8)
        )

    def test_non_person_only_output_skips_mask_rendering(self):
        raw = self.prediction_batch(prediction_count=79)[0]
        raw[:, :4] = [40, 40, 600, 600]
        raw[:, 4] = 0.99
        raw[:, 5] = np.arange(1, 80)
        raw[:, 6] = 1

        with mock.patch.object(
            self.model,
            "_render_mask_crops_into",
            wraps=self.model._render_mask_crops_into,
        ) as render_masks:
            detections, mask_crops, count = self.model._postprocess(
                raw, self.proto, META_32
            )

        self.assertEqual(int(count[0]), 0)
        self.assertEqual(detections.shape, (1, 6))
        self.assertEqual(mask_crops.shape, (1, 160, 20))
        self.assertFalse(np.any(detections))
        self.assertFalse(np.any(mask_crops))
        render_masks.assert_not_called()

    def test_results_are_sorted_and_limited_to_top_100(self):
        count = 120
        scores = np.linspace(0.5, 0.99, count, dtype=np.float32)
        order = np.random.default_rng(11).permutation(count)
        raw = self.prediction_batch(prediction_count=count)[0]
        raw[:, :4] = [40, 40, 600, 600]
        raw[:, 4] = scores[order]
        raw[:, 5] = 0
        raw[:, 6] = 1

        detections, mask_crops, num_detections = self.model._postprocess(
            raw, self.proto, META_32
        )

        self.assertEqual(int(num_detections[0]), 100)
        self.assertEqual(detections.shape, (100, 6))
        self.assertEqual(mask_crops.shape, (100, 160, 20))
        np.testing.assert_allclose(detections[:100, 4], np.sort(scores)[::-1][:100])

    def test_tied_scores_stably_keep_the_first_100_input_rows(self):
        count = 120
        raw = self.prediction_batch(prediction_count=count)[0]
        raw[:, :4] = [40, 40, 600, 600]
        raw[:, 0] += np.arange(count, dtype=np.float32)
        raw[:, 4] = 0.75

        detections, mask_crops, num_detections = self.model._postprocess(
            raw, self.proto, META_640
        )

        self.assertEqual(int(num_detections[0]), 100)
        self.assertEqual(detections.shape, (100, 6))
        self.assertEqual(mask_crops.shape, (100, 160, 20))
        np.testing.assert_array_equal(
            detections[:, :4], raw[:100, :4]
        )

    def test_confidence_invalid_boxes_and_empty_output(self):
        raw = self.prediction_batch()[0]
        raw[0, :6] = [20, 20, 200, 200, 0.49, 0]
        raw[1, :6] = [200, 200, 100, 100, 0.9, 0]

        detections, mask_crops, num_detections = self.model._postprocess(
            raw, self.proto, META_32
        )
        self.assertEqual(int(num_detections[0]), 0)
        self.assertEqual(detections.shape, (1, 6))
        self.assertFalse(np.any(detections))
        self.assertEqual(mask_crops.shape, (1, 160, 20))
        self.assertFalse(np.any(mask_crops))

        detections, mask_crops, num_detections = self.model._postprocess(
            np.zeros((0, 38), dtype=np.float32),
            self.proto,
            META_32,
        )
        self.assertEqual(int(num_detections[0]), 0)
        self.assertEqual(detections.shape, (1, 6))
        self.assertEqual(mask_crops.shape, (1, 160, 20))
        self.assertFalse(np.any(detections))
        self.assertFalse(np.any(mask_crops))

    def test_boxes_without_area_after_client_round_and_clip_are_filtered(self):
        cases = {
            "subpixel": [3.6, 1.0, 4.4, 2.0],
            "clipped_edge": [-0.4, 1.0, 0.4, 2.0],
            "ties_to_even": [3.5, 1.0, 4.5, 2.0],
        }
        for case_name, original_box in cases.items():
            with self.subTest(case=case_name):
                raw = self.prediction_batch(prediction_count=1)[0]
                raw[0, :4] = np.asarray(original_box, dtype=np.float32) * 20.0
                raw[0, 4:6] = [0.9, 0]
                raw[0, 6] = 1

                detections, mask_crops, count = self.model._postprocess(
                    raw, self.proto, META_32
                )

                self.assertEqual(int(count[0]), 0)
                self.assertEqual(detections.shape, (1, 6))
                self.assertEqual(mask_crops.shape, (1, 160, 20))
                self.assertFalse(np.any(detections))
                self.assertFalse(np.any(mask_crops))

    def test_same_width_box_with_rounded_pixel_area_keeps_float_coordinates(self):
        original_box = np.array([3.4, 1.0, 4.2, 2.0], dtype=np.float32)
        raw = self.prediction_batch(prediction_count=1)[0]
        raw[0, :4] = original_box * 20.0
        raw[0, 4:6] = [0.9, 0]
        raw[0, 6] = 1

        detections, mask_crops, count = self.model._postprocess(
            raw, self.proto, META_32
        )

        self.assertEqual(int(count[0]), 1)
        np.testing.assert_allclose(detections[0, :4], original_box, atol=1e-6)
        np.testing.assert_array_equal(
            mask_crops[0], np.full((160, 20), 255, dtype=np.uint8)
        )

    def test_non_end_to_end_layouts_are_rejected(self):
        for shape in ((2, 39), (116, 8400), (1, 2, 38)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "end-to-end layout"):
                    self.model._postprocess(np.zeros(shape, dtype=np.float32), self.proto, META_32)

    def test_hand_calculated_normalized_crop_oracle_and_input_immutability(self):
        raw = np.zeros((1, 38), dtype=np.float32)
        raw[0, :6] = [0, 0, 640, 640, 0.9, 0]
        raw[0, 6] = 1
        proto = self.proto.copy()
        meta = META_32.copy()
        original_raw, original_proto, original_meta = raw.copy(), proto.copy(), meta.copy()

        detections, mask_crops, count = self.model._postprocess(raw, proto, meta)

        np.testing.assert_allclose(detections[0], [0, 0, 32, 32, 0.9, 0])
        self.assertEqual(int(count[0]), 1)
        self.assertEqual(detections.shape, (1, 6))
        np.testing.assert_array_equal(
            mask_crops[0], np.full((160, 20), 255, dtype=np.uint8)
        )
        np.testing.assert_array_equal(raw, original_raw)
        np.testing.assert_array_equal(proto, original_proto)
        np.testing.assert_array_equal(meta, original_meta)

    def test_execute_b4_packs_to_dynamic_max_and_zero_pads_short_samples(self):
        model = SEG_MODULE.TritonPythonModel()
        model.initialize(
            {"model_config": json.dumps({"max_batch_size": 4})}
        )
        raw = self.prediction_batch(batch_size=4, prediction_count=4)
        raw[0, 0, :6] = [0, 0, 640, 640, 0.9, 0]
        raw[0, 0, 6] = 1
        raw[0, 1, :6] = [20, 20, 200, 200, 0.1, 0]
        raw[0, 2, :6] = [0, 0, 640, 640, 0.99, 56]
        raw[0, 3, :6] = [20, 20, 200, 200, 0.98, 41]
        raw[1, 0, :6] = [40, 40, 400, 400, 0.8, 0]
        raw[1, 0, 6] = 1
        raw[1, 1, :6] = [200, 200, 600, 600, 0.7, 0]
        raw[1, 1, 6] = -1
        raw[1, 2, :6] = [0, 0, 640, 640, 0.99, 62]
        raw[2, 0, :6] = [10, 10, 210, 210, 0.95, 0]
        raw[2, 0, 6] = 1
        raw[2, 1, :6] = [220, 20, 420, 220, 0.85, 0]
        raw[2, 1, 6] = -1
        raw[2, 2, :6] = [20, 240, 620, 620, 0.75, 0]
        raw[2, 2, 6] = 1
        raw[2, 3, :6] = [0, 0, 640, 640, 0.99, 58]
        raw[3, 0, :6] = [0, 0, 640, 640, 0.99, 39]
        proto = np.repeat(self.proto[None, ...], 4, axis=0)
        meta = np.repeat(META_32[None, ...], 4, axis=0)

        response = model.execute(
            [request(RAW_OUTPUT=raw, MASK_PROTO=proto, LETTERBOX_META=meta)]
        )[0]

        self.assertIsNone(response.error)
        outputs = output_map(response)
        self.assertEqual(set(outputs), {"DETECTIONS", "MASK_CROPS_PACKED", "NUM_DETECTIONS"})
        self.assertEqual(outputs["DETECTIONS"].shape, (4, 3, 6))
        self.assertEqual(outputs["MASK_CROPS_PACKED"].shape, (4, 3, 160, 20))
        self.assertEqual(outputs["NUM_DETECTIONS"].shape, (4, 1))
        np.testing.assert_array_equal(
            outputs["NUM_DETECTIONS"], [[1], [2], [3], [0]]
        )
        self.assertFalse(np.any(outputs["DETECTIONS"][:, :, 5]))
        np.testing.assert_array_equal(
            outputs["MASK_CROPS_PACKED"][0, 0], np.full((160, 20), 255, dtype=np.uint8)
        )
        self.assertFalse(np.any(outputs["DETECTIONS"][0, 1]))
        self.assertFalse(np.any(outputs["MASK_CROPS_PACKED"][0, 1]))
        np.testing.assert_array_equal(
            outputs["MASK_CROPS_PACKED"][1, 0], np.full((160, 20), 255, dtype=np.uint8)
        )
        self.assertFalse(np.any(outputs["MASK_CROPS_PACKED"][1, 1]))
        np.testing.assert_array_equal(
            outputs["MASK_CROPS_PACKED"][2, 0],
            np.full((160, 20), 255, dtype=np.uint8),
        )
        self.assertFalse(np.any(outputs["MASK_CROPS_PACKED"][2, 1]))
        np.testing.assert_array_equal(
            outputs["MASK_CROPS_PACKED"][2, 2],
            np.full((160, 20), 255, dtype=np.uint8),
        )
        for batch_index, count in enumerate((1, 2, 3, 0)):
            self.assertFalse(np.any(outputs["DETECTIONS"][batch_index, count:]))
            self.assertFalse(np.any(outputs["MASK_CROPS_PACKED"][batch_index, count:]))
        self.assertFalse(np.array_equal(outputs["DETECTIONS"][0], outputs["DETECTIONS"][1]))
        for value in outputs.values():
            self.assertTrue(value.flags.c_contiguous)
            self.assertTrue(np.isfinite(value).all())
        self.assertEqual(outputs["DETECTIONS"].dtype, np.float32)
        self.assertEqual(outputs["MASK_CROPS_PACKED"].dtype, np.uint8)
        self.assertEqual(outputs["NUM_DETECTIONS"].dtype, np.int32)

        for batch_index in range(4):
            with self.subTest(batch_index=batch_index):
                b1 = model.execute(
                    [
                        request(
                            RAW_OUTPUT=raw[batch_index : batch_index + 1],
                            MASK_PROTO=proto[batch_index : batch_index + 1],
                            LETTERBOX_META=meta[batch_index : batch_index + 1],
                        )
                    ]
                )[0]
                self.assertIsNone(b1.error)
                b1_outputs = output_map(b1)
                count = int(outputs["NUM_DETECTIONS"][batch_index, 0])
                self.assertEqual(b1_outputs["DETECTIONS"].shape[0], 1)
                self.assertEqual(b1_outputs["MASK_CROPS_PACKED"].shape[0], 1)
                np.testing.assert_array_equal(
                    b1_outputs["DETECTIONS"][0, :count],
                    outputs["DETECTIONS"][batch_index, :count],
                )
                np.testing.assert_array_equal(
                    b1_outputs["MASK_CROPS_PACKED"][0, :count],
                    outputs["MASK_CROPS_PACKED"][batch_index, :count],
                )
                np.testing.assert_array_equal(
                    b1_outputs["NUM_DETECTIONS"][0],
                    outputs["NUM_DETECTIONS"][batch_index],
                )

    def test_execute_entirely_empty_b2_uses_single_zero_padding_slot(self):
        raw = self.prediction_batch(batch_size=2, prediction_count=2)
        proto = np.stack([self.proto, self.proto])
        meta = np.stack([META_32, META_32])

        response = self.model.execute(
            [request(RAW_OUTPUT=raw, MASK_PROTO=proto, LETTERBOX_META=meta)]
        )[0]

        self.assertIsNone(response.error)
        outputs = output_map(response)
        self.assertEqual(outputs["DETECTIONS"].shape, (2, 1, 6))
        self.assertEqual(outputs["MASK_CROPS_PACKED"].shape, (2, 1, 160, 20))
        self.assertEqual(outputs["NUM_DETECTIONS"].shape, (2, 1))
        for value in outputs.values():
            self.assertFalse(np.any(value))

    def test_execute_rejects_batch_mismatch_and_different_original_sizes(self):
        raw = self.prediction_batch(batch_size=2, prediction_count=1)
        proto = np.stack([self.proto, self.proto])
        meta = np.stack([META_32, META_32])
        mismatch = self.model.execute(
            [request(RAW_OUTPUT=raw, MASK_PROTO=proto[:1], LETTERBOX_META=meta)]
        )[0]
        self.assertRegex(str(mismatch.error), "batch size must match")

        different_size = meta.copy()
        different_size[1] = np.array([32, 64, 10, 0, 32, 384, 640], dtype=np.float32)
        response = self.model.execute(
            [request(RAW_OUTPUT=raw, MASK_PROTO=proto, LETTERBOX_META=different_size)]
        )[0]
        self.assertRegex(str(response.error), "same original height and width")

    def test_execute_identifies_invalid_batch_index_and_isolates_requests(self):
        raw = self.prediction_batch(batch_size=2, prediction_count=1)
        raw[:, 0, :6] = [0, 0, 640, 640, 0.9, 0]
        raw[1, 0, 4] = np.nan
        proto = np.stack([self.proto, self.proto])
        meta = np.stack([META_32, META_32])
        bad = request(RAW_OUTPUT=raw, MASK_PROTO=proto, LETTERBOX_META=meta)
        good = request(
            RAW_OUTPUT=np.nan_to_num(raw[:1]),
            MASK_PROTO=proto[:1],
            LETTERBOX_META=meta[:1],
        )

        bad_response, good_response = self.model.execute([bad, good])

        self.assertRegex(str(bad_response.error), "INVALID_INPUT: batch index 1")
        self.assertNotIn("nan", str(bad_response.error).lower())
        self.assertIsNone(good_response.error)
        good_outputs = output_map(good_response)
        self.assertEqual(good_outputs["DETECTIONS"].shape, (1, 1, 6))
        self.assertEqual(good_outputs["MASK_CROPS_PACKED"].shape, (1, 1, 160, 20))
        self.assertFalse(np.any(good_outputs["MASK_CROPS_PACKED"]))

    def test_strict_dtypes_finite_values_resources_and_geometry(self):
        raw = self.prediction_batch()[0]
        proto = self.proto.copy()
        with self.assertRaisesRegex(ValueError, "RAW_OUTPUT must have dtype float32"):
            self.model._postprocess(raw.astype(np.float64), proto, META_32)
        with self.assertRaisesRegex(ValueError, "MASK_PROTO must have dtype float32"):
            self.model._postprocess(raw, proto.astype(np.float64), META_32)
        raw[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            self.model._postprocess(raw, proto, META_32)
        too_many = np.zeros((100001, 38), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "max predictions"):
            self.model._postprocess(too_many, proto, META_32)
        unsafe_meta = META_32.copy()
        unsafe_meta[2] = 0
        with self.assertRaisesRegex(ValueError, "ratio"):
            self.model._postprocess(np.nan_to_num(raw), proto, unsafe_meta)

    def test_initialize_parses_batch_limit_accepts_b4_and_rejects_above_b1(self):
        batch_four_model = SEG_MODULE.TritonPythonModel()
        batch_four_model.initialize(
            {"model_config": json.dumps({"max_batch_size": 4})}
        )
        self.assertEqual(batch_four_model.max_batch_size, 4)

        limited_model = SEG_MODULE.TritonPythonModel()
        limited_model.initialize(
            {"model_config": json.dumps({"max_batch_size": 1})}
        )
        self.assertEqual(limited_model.max_batch_size, 1)

        raw = self.prediction_batch(batch_size=4, prediction_count=1)
        proto = np.repeat(self.proto[None, ...], 4, axis=0)
        meta = np.repeat(META_32[None, ...], 4, axis=0)
        accepted = batch_four_model.execute(
            [request(RAW_OUTPUT=raw, MASK_PROTO=proto, LETTERBOX_META=meta)]
        )[0]
        self.assertIsNone(accepted.error)

        rejected = limited_model.execute(
            [request(RAW_OUTPUT=raw, MASK_PROTO=proto, LETTERBOX_META=meta)]
        )[0]
        self.assertRegex(
            str(rejected.error),
            "batch size 4 exceeds configured maximum 1",
        )

    def test_mask_crop_budget_is_enforced_before_render_or_crop_allocation(self):
        model = SEG_MODULE.TritonPythonModel()
        model.initialize({})
        raw = np.zeros((2, 1, 38), dtype=np.float32)
        raw[:, 0, :6] = [0, 0, 640, 640, 0.9, 0]
        proto = np.zeros((2, 32, 1, 1), dtype=np.float32)
        meta = np.stack([META_32, META_32])
        allocated_shapes = []

        def allocation_spy(shape, dtype):
            allocated_shapes.append(tuple(shape))
            return np.zeros(shape, dtype=dtype)

        with mock.patch.object(
            SEG_MODULE, "MAX_MASK_OUTPUT_VALUES", 51_199
        ), mock.patch.object(
            SEG_MODULE, "_allocate_zeros", side_effect=allocation_spy
        ), mock.patch.object(
            SEG_MODULE.cv2,
            "warpAffine",
            wraps=SEG_MODULE.cv2.warpAffine,
        ) as warp_affine:
            response = model.execute(
                [request(RAW_OUTPUT=raw, MASK_PROTO=proto, LETTERBOX_META=meta)]
            )[0]

        self.assertRegex(
            str(response.error), "MASK_CROPS_PACKED logical output values exceed budget 51199"
        )
        warp_affine.assert_not_called()
        self.assertFalse(
            any(shape[-2:] in ((160, 20), (160, 160)) for shape in allocated_shapes),
            allocated_shapes,
        )

    def test_raw_and_proto_total_value_budgets_are_enforced(self):
        model = SEG_MODULE.TritonPythonModel()
        model.initialize({})
        meta = META_32.reshape(1, 7)

        raw_over_budget = np.broadcast_to(
            np.zeros((1, 1, 518), dtype=np.float32), (1, 8100, 518)
        )
        proto_for_raw = np.zeros((1, 512, 1, 1), dtype=np.float32)
        raw_response = model.execute(
            [
                request(
                    RAW_OUTPUT=raw_over_budget,
                    MASK_PROTO=proto_for_raw,
                    LETTERBOX_META=meta,
                )
            ]
        )[0]
        self.assertRegex(
            str(raw_response.error), "RAW_OUTPUT total values exceed 4194304"
        )

        raw_for_proto = np.zeros((1, 1, 38), dtype=np.float32)
        proto_over_budget = np.broadcast_to(
            np.zeros((1, 1, 1, 1), dtype=np.float32),
            (1, 32, 512, 513),
        )
        proto_response = model.execute(
            [
                request(
                    RAW_OUTPUT=raw_for_proto,
                    MASK_PROTO=proto_over_budget,
                    LETTERBOX_META=meta,
                )
            ]
        )[0]
        self.assertRegex(
            str(proto_response.error), "MASK_PROTO total values exceed 8388608"
        )

    def test_last_invalid_sample_is_rejected_before_output_allocation(self):
        self.assertTrue(hasattr(SEG_MODULE, "_allocate_zeros"))
        raw = self.prediction_batch(batch_size=2, prediction_count=1)
        raw[1, 0, 4] = np.nan
        proto = np.stack([self.proto, self.proto])
        meta = np.stack([META_32, META_32])

        with mock.patch.object(
            SEG_MODULE, "_allocate_zeros", wraps=np.zeros
        ) as allocator:
            response = self.model.execute(
                [request(RAW_OUTPUT=raw, MASK_PROTO=proto, LETTERBOX_META=meta)]
            )[0]

        self.assertRegex(str(response.error), "batch index 1: RAW_OUTPUT values must be finite")
        allocator.assert_not_called()

    def test_proto_and_meta_nonfinite_errors_identify_batch_index(self):
        self.assertTrue(hasattr(SEG_MODULE, "_allocate_zeros"))
        raw = self.prediction_batch(batch_size=2, prediction_count=1)
        proto = np.stack([self.proto, self.proto])
        meta = np.stack([META_32, META_32])

        cases = []
        bad_proto = proto.copy()
        bad_proto[1, 0, 0, 0] = np.inf
        cases.append((raw, bad_proto, meta, "MASK_PROTO values must be finite"))
        bad_meta = meta.copy()
        bad_meta[1, 2] = np.nan
        cases.append((raw, proto, bad_meta, "LETTERBOX_META values must be finite"))
        for case_raw, case_proto, case_meta, message in cases:
            with self.subTest(message=message), mock.patch.object(
                SEG_MODULE, "_allocate_zeros", wraps=np.zeros
            ) as allocator:
                response = self.model.execute(
                    [
                        request(
                            RAW_OUTPUT=case_raw,
                            MASK_PROTO=case_proto,
                            LETTERBOX_META=case_meta,
                        )
                    ]
                )[0]
                self.assertRegex(str(response.error), f"batch index 1: {message}")
                allocator.assert_not_called()

    def test_canonical_geometry_matches_extreme_aspect_letterbox_oracle(self):
        meta = independent_letterbox_meta(7, 1792)
        geometry = SEG_MODULE._validate_meta(meta)
        independent_ratio = 640 / 1792

        self.assertEqual(geometry.original_h, 7)
        self.assertEqual(geometry.original_w, 1792)
        self.assertEqual(geometry.resized_h, round(7 * independent_ratio))
        self.assertEqual(geometry.resized_h, 2)
        self.assertEqual(geometry.resized_w, 640)
        self.assertEqual(geometry.top, 95)
        self.assertEqual(geometry.bottom, 95)
        self.assertEqual(geometry.input_h, 192)
        self.assertEqual(geometry.input_w, 640)

        proto = np.zeros((32, 48, 160), dtype=np.float32)
        proto[0] = 10
        raw = np.zeros((1, 38), dtype=np.float32)
        raw[0, :6] = [0, 0, 640, 192, 0.9, 0]
        raw[0, 6] = 1
        _, mask_crops, count = self.model._postprocess(raw, proto, meta)
        self.assertEqual(int(count[0]), 1)
        np.testing.assert_array_equal(
            mask_crops[0], np.full((160, 20), 255, dtype=np.uint8)
        )

    def test_zero_logits_and_negative_coefficients_produce_zero_crops(self):
        raw = np.zeros((1, 38), dtype=np.float32)
        raw[0, :6] = [0, 0, 640, 640, 0.9, 0]

        _, mask_crops, count = self.model._postprocess(raw, self.proto, META_32)

        self.assertEqual(int(count[0]), 1)
        self.assertFalse(np.any(mask_crops[0]))

        raw[0, 6] = -1
        _, mask_crops, count = self.model._postprocess(raw, self.proto, META_32)
        self.assertEqual(int(count[0]), 1)
        self.assertFalse(np.any(mask_crops[0]))

    def test_non_grid_bbox_is_normalized_directly_to_crop_coordinates(self):
        meta = independent_letterbox_meta(9, 17)
        ratio = 640 / 17
        original_box = np.array([2.3, 1.2, 7.7, 5.8], dtype=np.float32)
        letterbox_box = original_box.copy()
        letterbox_box[[0, 2]] = letterbox_box[[0, 2]] * ratio + meta[3]
        letterbox_box[[1, 3]] = letterbox_box[[1, 3]] * ratio + meta[4]
        raw = np.zeros((1, 38), dtype=np.float32)
        raw[0, :4] = letterbox_box
        raw[0, 4:6] = [0.9, 0]
        raw[0, 6] = 1
        proto = np.zeros((32, 11, 20), dtype=np.float32)
        proto[0] = 10

        detections, mask_crops, count = self.model._postprocess(raw, proto, meta)

        self.assertEqual(int(count[0]), 1)
        np.testing.assert_allclose(detections[0, :4], original_box, atol=1e-5)
        np.testing.assert_array_equal(
            mask_crops[0], np.full((160, 20), 255, dtype=np.uint8)
        )

    def test_non_grid_bbox_uses_bilinear_proto_sampling(self):
        meta = independent_letterbox_meta(9, 17)
        ratio = 640 / 17
        original_box = np.array([2.3, 1.2, 7.7, 5.8], dtype=np.float32)
        letterbox_box = original_box.copy()
        letterbox_box[[0, 2]] = letterbox_box[[0, 2]] * ratio + meta[3]
        letterbox_box[[1, 3]] = letterbox_box[[1, 3]] * ratio + meta[4]
        raw = np.zeros((1, 38), dtype=np.float32)
        raw[0, :4] = letterbox_box
        raw[0, 4:6] = [0.9, 0]
        raw[0, 6] = 1
        proto = np.zeros((32, 11, 20), dtype=np.float32)
        proto[0] = np.arange(20, dtype=np.float32)[None, :] - 5.5

        _, mask_crops, count = self.model._postprocess(raw, proto, meta)

        self.assertEqual(int(count[0]), 1)
        x1_proto = original_box[0] * 20 / 17
        x2_proto = original_box[2] * 20 / 17
        sample_x = (
            x1_proto
            + (np.arange(160, dtype=np.float32) + 0.5)
            * (x2_proto - x1_proto)
            / 160
            - 0.5
        )
        expected_columns = sample_x > 5.5
        expected = np.broadcast_to(expected_columns, (160, 160)).astype(np.uint8)
        np.testing.assert_array_equal(
            np.unpackbits(mask_crops[0], axis=-1, count=160, bitorder="big"), expected
        )

    def test_2d_proto_gradient_maps_vertical_padding_and_clipped_non_grid_box(self):
        meta = independent_letterbox_meta(9, 17)
        ratio = 640 / 17
        raw_original_box = np.array([2.3, -1.2, 17.4, 5.8], dtype=np.float32)
        clipped_original_box = np.array([2.3, 0.0, 17.0, 5.8], dtype=np.float32)
        letterbox_box = raw_original_box.copy()
        letterbox_box[[0, 2]] = letterbox_box[[0, 2]] * ratio + meta[3]
        letterbox_box[[1, 3]] = letterbox_box[[1, 3]] * ratio + meta[4]
        raw = np.zeros((1, 38), dtype=np.float32)
        raw[0, :4] = letterbox_box
        raw[0, 4:6] = [0.9, 0]
        raw[0, 6] = 1
        proto_h, proto_w = 44, 20
        proto = np.zeros((32, proto_h, proto_w), dtype=np.float32)
        proto_y, proto_x = np.indices((proto_h, proto_w), dtype=np.float32)
        threshold = 400.125
        proto[0] = proto_x + 32.0 * proto_y - threshold

        detections, mask_crops, count = self.model._postprocess(raw, proto, meta)

        self.assertEqual(int(count[0]), 1)
        np.testing.assert_allclose(
            detections[0, :4], clipped_original_box, atol=1e-5
        )
        geometry = SEG_MODULE._validate_meta(meta)
        self.assertGreater(geometry.top, 0)
        x1_proto, x2_proto = (
            clipped_original_box[[0, 2]] * geometry.ratio + geometry.left
        ) * (proto_w / geometry.input_w)
        y1_proto, y2_proto = (
            clipped_original_box[[1, 3]] * geometry.ratio + geometry.top
        ) * (proto_h / geometry.input_h)
        sample_x = (
            x1_proto
            + (np.arange(160, dtype=np.float32) + 0.5)
            * (x2_proto - x1_proto)
            / 160
            - 0.5
        )
        sample_y = (
            y1_proto
            + (np.arange(160, dtype=np.float32) + 0.5)
            * (y2_proto - y1_proto)
            / 160
            - 0.5
        )
        expected_logits = (
            sample_x[None, :] + 32.0 * sample_y[:, None] - threshold
        )
        unambiguous = np.abs(expected_logits) > 1.0
        expected = expected_logits > 0.0
        binary_mask = np.unpackbits(mask_crops[0], axis=-1, count=160, bitorder="big")
        np.testing.assert_array_equal(binary_mask[unambiguous], expected[unambiguous])
        self.assertTrue(np.any(binary_mask == 0))
        self.assertTrue(np.any(binary_mask == 1))

    def test_high_resolution_meta_never_allocates_original_size_masks(self):
        meta = independent_letterbox_meta(2560, 2560).reshape(1, 7)
        raw = np.zeros((1, 1, 38), dtype=np.float32)
        raw[0, 0, :6] = [0, 0, 640, 640, 0.9, 0]
        raw[0, 0, 6] = 1
        proto = self.proto.reshape(1, 32, 8, 8)
        allocated_shapes = []

        def allocation_spy(shape, dtype):
            allocated_shapes.append(tuple(shape))
            return np.zeros(shape, dtype=dtype)

        with mock.patch.object(SEG_MODULE, "_allocate_zeros", allocation_spy), mock.patch.object(
            SEG_MODULE.cv2,
            "resize",
            side_effect=AssertionError("full-image resize is forbidden"),
        ):
            response = self.model.execute(
                [request(RAW_OUTPUT=raw, MASK_PROTO=proto, LETTERBOX_META=meta)]
            )[0]

        self.assertIsNone(response.error)
        outputs = output_map(response)
        self.assertEqual(outputs["DETECTIONS"].shape, (1, 1, 6))
        self.assertEqual(outputs["MASK_CROPS_PACKED"].shape, (1, 1, 160, 20))
        np.testing.assert_array_equal(
            outputs["MASK_CROPS_PACKED"][0, 0],
            np.full((160, 20), 255, dtype=np.uint8),
        )
        self.assertNotIn((2560, 2560), allocated_shapes)
        self.assertNotIn((1, 1, 2560, 2560), allocated_shapes)

    def test_internal_failure_returns_one_redacted_response_per_request(self):
        raw = self.prediction_batch(batch_size=1, prediction_count=1)
        proto = self.proto.reshape(1, 32, 8, 8)
        meta = META_32.reshape(1, 7)
        good_request = request(
            RAW_OUTPUT=raw, MASK_PROTO=proto, LETTERBOX_META=meta
        )

        with mock.patch.object(
            self.model,
            "_process_sample",
            side_effect=RuntimeError("SECRET_RAW_VALUE_123"),
            create=True,
        ), mock.patch.object(SEG_MODULE.logger, "exception"):
            responses = self.model.execute([good_request, good_request])

        self.assertEqual(len(responses), 2)
        for response in responses:
            error = str(response.error)
            self.assertRegex(
                error,
                r"^INTERNAL_ERROR: yolo26s_seg_postprocess failed; incident=[0-9a-f]{12}$",
            )
            self.assertNotIn("SECRET_RAW_VALUE_123", error)

    def test_client_selects_dynamic_batch_row_and_ignores_padding(self):
        detections = np.zeros((2, 3, 6), dtype=np.float32)
        detections[1, 0, :6] = [1, 2, 3, 4, 0.8, 5]
        detections[1, 1:] = 987
        mask_crops = np.zeros((2, 3, 160, 20), dtype=np.uint8)
        mask_crops[1, 0] = 255
        mask_crops[1, 1:] = 7
        counts = np.array([[0], [1]], dtype=np.int32)

        selected_detections, selected_crops, count = SEG_CLIENT_MODULE.select_batch_outputs(
            detections, mask_crops, counts, batch_index=1
        )

        self.assertEqual(count, 1)
        self.assertEqual(selected_detections.shape, (1, 6))
        self.assertEqual(selected_crops.shape, (1, 160, 160))
        self.assertFalse(np.any(selected_detections == 987))
        self.assertFalse(np.any(selected_crops == 7))

    def test_client_select_batch_outputs_validates_dynamic_crop_contract(self):
        counts = np.array([[0]], dtype=np.int32)
        with self.assertRaisesRegex(ValueError, "between 1 and 100"):
            SEG_CLIENT_MODULE.select_batch_outputs(
                np.zeros((1, 0, 6), dtype=np.float32),
                np.zeros((1, 0, 160, 20), dtype=np.uint8),
                counts,
            )
        with self.assertRaisesRegex(ValueError, "between 1 and 100"):
            SEG_CLIENT_MODULE.select_batch_outputs(
                np.zeros((1, 101, 6), dtype=np.float32),
                np.zeros((1, 101, 160, 20), dtype=np.uint8),
                counts,
            )
        with self.assertRaisesRegex(ValueError, "batch and capacity"):
            SEG_CLIENT_MODULE.select_batch_outputs(
                np.zeros((1, 2, 6), dtype=np.float32),
                np.zeros((1, 1, 160, 20), dtype=np.uint8),
                counts,
            )
        with self.assertRaisesRegex(ValueError, "160,20"):
            SEG_CLIENT_MODULE.select_batch_outputs(
                np.zeros((1, 2, 6), dtype=np.float32),
                np.zeros((1, 2, 159, 160), dtype=np.uint8),
                counts,
            )
        with self.assertRaisesRegex(ValueError, "detection capacity"):
            SEG_CLIENT_MODULE.select_batch_outputs(
                np.zeros((1, 2, 6), dtype=np.float32),
                np.zeros((1, 2, 160, 20), dtype=np.uint8),
                np.array([[3]], dtype=np.int32),
            )

    def test_client_select_online_outputs_accepts_dynamic_capacity(self):
        detections = np.zeros((1, 3, 6), dtype=np.float32)
        detections[0, :2, 4] = [0.9, 0.8]
        detections[0, 2] = 987
        mask_crops = np.zeros((1, 3, 160, 20), dtype=np.uint8)
        mask_crops[0, :2] = 255
        mask_crops[0, 2] = 7

        selected_detections, selected_crops, count = SEG_CLIENT_MODULE.select_online_outputs(
            detections,
            mask_crops,
            np.array([[2]], dtype=np.int32),
            expect_masks=True,
            expected_image_shape=(480, 640),
        )

        self.assertEqual(count, 2)
        self.assertEqual(selected_detections.shape, (2, 6))
        self.assertEqual(selected_crops.shape, (2, 160, 160))
        self.assertFalse(np.any(selected_detections == 987))
        self.assertFalse(np.any(selected_crops == 7))

    def test_client_select_online_outputs_supports_boxes_only(self):
        selected_detections, selected_crops, count = SEG_CLIENT_MODULE.select_online_outputs(
            np.zeros((1, 2, 6), dtype=np.float32),
            None,
            np.array([[1]], dtype=np.int32),
            expect_masks=False,
            expected_image_shape=(4, 5),
        )

        self.assertEqual(count, 1)
        self.assertEqual(selected_detections.shape, (1, 6))
        self.assertIsNone(selected_crops)

    def test_client_restore_mask_crops_places_nearest_neighbor_mask_in_bbox(self):
        detections = np.array([[1.2, 0.6, 5.4, 4.6, 0.9, 0]], dtype=np.float32)
        mask_crops = np.zeros((1, 160, 160), dtype=np.uint8)
        mask_crops[0, :, 80:] = 1

        masks = SEG_CLIENT_MODULE.restore_mask_crops(
            detections, mask_crops, image_shape=(6, 8)
        )

        expected = np.zeros((1, 6, 8), dtype=np.uint8)
        expected[0, 1:5, 3:5] = 1
        self.assertEqual(masks.dtype, np.uint8)
        np.testing.assert_array_equal(masks, expected)
        self.assertFalse(np.any(masks[0, :1]))
        self.assertFalse(np.any(masks[0, 5:]))
        self.assertFalse(np.any(masks[0, :, :1]))
        self.assertFalse(np.any(masks[0, :, 5:]))

    def test_client_restore_mask_crops_validates_alignment_shape_and_area(self):
        detection = np.array([[1, 1, 2, 2, 0.9, 0]], dtype=np.float32)
        crop = np.ones((1, 160, 160), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "same N"):
            SEG_CLIENT_MODULE.restore_mask_crops(
                detection, np.ones((2, 160, 160), dtype=np.uint8), (4, 4)
            )
        with self.assertRaisesRegex(ValueError, "positive H and W"):
            SEG_CLIENT_MODULE.restore_mask_crops(detection, crop, (0, 4))
        with self.assertRaisesRegex(ValueError, "positive area"):
            SEG_CLIENT_MODULE.restore_mask_crops(
                np.array([[3.6, 1, 4.4, 2, 0.9, 0]], dtype=np.float32),
                crop,
                (4, 4),
            )

    def test_client_draw_predictions_ignores_padded_detection_and_mask_slots(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        detections = np.zeros((100, 6), dtype=np.float32)
        detections[0] = [1, 1, 6, 6, 0.9, 0]
        masks = np.zeros((100, 8, 8), dtype=np.uint8)
        masks[0, 2:5, 2:5] = 1
        masks[1:] = 1

        expected = SEG_CLIENT_MODULE.draw_predictions(
            image, detections[:1], masks[:1]
        )
        actual = SEG_CLIENT_MODULE.draw_predictions(
            image, detections, masks, num_detections=1
        )

        np.testing.assert_array_equal(actual, expected)

    def test_client_main_sends_batched_image_and_selects_row_prefix(self):
        image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        detections = np.zeros((1, 3, 6), dtype=np.float32)
        detections[0, 0] = [1, 1, 4, 3, 0.9, 2]
        detections[0, 1:] = 987
        mask_crops = np.zeros((1, 3, 160, 20), dtype=np.uint8)
        mask_crops[0, 0] = 255
        mask_crops[0, 1:] = 7
        counts = np.array([[1]], dtype=np.int32)
        captured = {}
        events = []
        expected_mask = np.zeros((1, 4, 5), dtype=np.uint8)
        expected_mask[0, 1:3, 1:4] = 1

        class FakeInferInput:
            def __init__(self, name, shape, data_type):
                captured["input_name"] = name
                captured["input_shape"] = tuple(shape)
                captured["input_dtype"] = data_type

            def set_data_from_numpy(self, value):
                captured["input_data"] = value.copy()

        class FakeResult:
            def as_numpy(self, name):
                return {
                    "DETECTIONS": detections,
                    "MASK_CROPS_PACKED": mask_crops,
                    "NUM_DETECTIONS": counts,
                }[name]

        class FakeClient:
            def __init__(self, url):
                captured["url"] = url

            def infer(self, model, inputs, outputs):
                captured["model"] = model
                captured["inputs"] = inputs
                captured["outputs"] = outputs
                return FakeResult()

        class FakeRequestedOutput:
            def __init__(self, name):
                self.name = name

        fake_httpclient = types.SimpleNamespace(
            InferenceServerClient=FakeClient,
            InferInput=FakeInferInput,
            InferRequestedOutput=FakeRequestedOutput,
        )

        def fake_draw(draw_image, draw_detections, draw_masks, **kwargs):
            captured["draw_image"] = draw_image.copy()
            captured["draw_detections"] = draw_detections.copy()
            captured["draw_masks"] = draw_masks.copy()
            captured["draw_count"] = kwargs["num_detections"]
            return draw_image.copy()

        def fake_restore(restore_detections, restore_crops, image_shape):
            events.append("restore")
            np.testing.assert_array_equal(restore_detections, detections[0, :1])
            np.testing.assert_array_equal(restore_crops, np.ones((1, 160, 160), dtype=np.uint8))
            self.assertEqual(tuple(image_shape), (4, 5))
            return expected_mask.copy()

        def fake_print(*args, **kwargs):
            text = " ".join(str(arg) for arg in args)
            if "raw_mask_crops_shape:" in text:
                events.append("raw_mask_crops_metadata")

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = str(Path(temporary_directory) / "result.jpg")
            argv = [
                "test_yolo26s_seg_image.py",
                "--image",
                "fake.jpg",
                "--save-result",
                output_path,
            ]
            with mock.patch.object(SEG_CLIENT_MODULE, "httpclient", fake_httpclient), mock.patch.object(
                SEG_CLIENT_MODULE.cv2, "imread", return_value=image.copy()
            ), mock.patch.object(
                SEG_CLIENT_MODULE.cv2, "imwrite", return_value=True
            ), mock.patch.object(
                SEG_CLIENT_MODULE, "draw_predictions", side_effect=fake_draw
            ), mock.patch.object(
                SEG_CLIENT_MODULE, "restore_mask_crops", side_effect=fake_restore
            ), mock.patch.object(sys, "argv", argv), mock.patch(
                "builtins.print", side_effect=fake_print
            ) as print_mock:
                SEG_CLIENT_MODULE.main()

        self.assertEqual(captured["input_name"], "RAW_IMAGE")
        self.assertEqual(captured["input_shape"], (1, 4, 5, 3))
        self.assertEqual(captured["input_dtype"], "UINT8")
        np.testing.assert_array_equal(captured["input_data"], image[None, ...])
        self.assertEqual(
            [output.name for output in captured["outputs"]],
            ["DETECTIONS", "NUM_DETECTIONS", "MASK_CROPS_PACKED"],
        )
        self.assertEqual(captured["draw_count"], 1)
        np.testing.assert_array_equal(captured["draw_detections"], detections[0, :1])
        np.testing.assert_array_equal(captured["draw_masks"], expected_mask)
        self.assertFalse(np.any(captured["draw_detections"] == 987))
        self.assertFalse(np.any(captured["draw_masks"] == 7))
        printed = "\n".join(" ".join(str(arg) for arg in call.args) for call in print_mock.call_args_list)
        self.assertIn("raw_mask_crops_shape: (1, 3, 160, 20)", printed)
        self.assertIn("raw_mask_crops_mb:", printed)
        self.assertLess(
            events.index("raw_mask_crops_metadata"), events.index("restore")
        )


if __name__ == "__main__":
    unittest.main()
