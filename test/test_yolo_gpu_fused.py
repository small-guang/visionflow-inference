"""CUDA fusion boundaries: stable ranking, clipping, empty rows and packed masks."""
import unittest
from unittest import mock

import numpy as np

from test.test_yolo_gpu_postprocess import (
    CUDA_AVAILABLE, META, cp, execute, load_model, outputs,
)


@unittest.skipUnless(CUDA_AVAILABLE, "requires CuPy and a CUDA GPU")
class FusedYoloPostprocessTests(unittest.TestCase):
    def compare(self, name, values, *, fused=True):
        expected = outputs(execute(load_model(name, False), values, False))
        model = load_model(name, True)
        for gpu_inputs in (False, True):
            with self.subTest(gpu_inputs=gpu_inputs):
                # Small YOLO end-to-end outputs must avoid the multi-kernel sort.
                context = mock.patch.object(cp, "lexsort", side_effect=AssertionError("unfused sort"))
                if not fused:
                    context = mock.patch.object(cp, "lexsort", wraps=cp.lexsort)
                with context:
                    actual = outputs(execute(model, values, gpu_inputs))
                for key, value in expected.items():
                    self.assertEqual(actual[key].shape, value.shape)
                    if np.issubdtype(value.dtype, np.floating):
                        np.testing.assert_allclose(actual[key], value, rtol=1e-6, atol=1e-5)
                    else:
                        np.testing.assert_array_equal(actual[key], value)

    def test_pose_candidate_boundaries_stable_ties_clipping_and_empty_rows(self):
        for count in (0, 1, 257, 300, 1024, 1025):
            with self.subTest(candidates=count):
                raw = np.zeros((4, count, 57), dtype=np.float32)
                raw[:3, :, :6] = [-1, 3, 641, 638, 0.75, 1.5]
                raw[:3, :, 0] += np.arange(count, dtype=np.float32) % 620
                raw[0, :, 4] = np.resize(np.array([0.75, 0.9, 0.6], np.float32), count)
                raw[:3, :, 6:] = np.tile([-2, 700, 0.625], 17)
                raw[1, ::3, 4] = np.nextafter(np.float32(0.5), np.float32(0))
                raw[1, 1::3, 4] = 0.5
                raw[2, ::2, 2] = -2
                self.compare("yolo26m_pose_postprocess", {
                    "RAW_OUTPUT": raw, "LETTERBOX_META": np.tile(META, (4, 1)),
                }, fused=count <= 1024)

    def test_seg_rank_boundaries_person_rounding_and_pixel_area(self):
        for count in (0, 1, 257, 300, 1024, 1025):
            with self.subTest(candidates=count):
                raw = np.zeros((4, count, 8), dtype=np.float32)
                raw[:3, :, :6] = [0, 0, 640, 640, 0.75, 0]
                raw[:3, :, 0] += np.arange(count, dtype=np.float32) % 620
                raw[0, :, 4] = np.resize(np.array([0.75, 0.9, 0.6], np.float32), count)
                raw[:3, :, 6] = 1
                raw[1, ::3, 5] = 0.5  # ties-to-even retains person class
                raw[1, 1::3, 5] = 0.50001
                raw[2, ::2, :4] = [3.5, 1, 4.5, 2]  # rounded zero width
                proto = np.zeros((4, 2, 9, 13), dtype=np.float32)
                yy, xx = np.indices((9, 13), dtype=np.float32)
                proto[:, 0] = xx + 8 * yy - 17.125
                self.compare("yolo26s_seg_postprocess", {
                    "RAW_OUTPUT": raw, "MASK_PROTO": proto,
                    "LETTERBOX_META": np.tile(META, (4, 1)),
                }, fused=count <= 1024)

    def test_face_fused_nms_keeps_numpy_ties_and_handles_strided_inputs(self):
        rng = np.random.default_rng(83)
        raw = np.zeros((4, 5, 600), dtype=np.float32)
        raw[:3, :2] = rng.uniform(-30, 680, (3, 2, 600))
        raw[:3, 2:4] = rng.uniform(-5, 100, (3, 2, 600))
        raw[:3, 4] = rng.choice([0.1, 0.5, 0.75, 0.9], (3, 600))
        raw = raw[:, :, ::2]
        self.compare("yolov8n_face_postprocess", {
            "RAW_OUTPUT": raw, "LETTERBOX_META": np.tile(META[:5], (4, 1)),
        })


if __name__ == "__main__":
    unittest.main()
