import re
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from test.sixdrepnet_backend_test_utils import (
    FakeResponse,
    FakeTensor,
    FakeTritonError,
    load_backend,
    make_request,
)


def rotation_matrix(pitch, yaw, roll):
    pitch, yaw, roll = np.deg2rad(
        np.asarray([pitch, yaw, roll], dtype=np.float64)
    )
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cr, sr = np.cos(roll), np.sin(roll)
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]],
        dtype=np.float64,
    )
    ry = np.array(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=np.float64,
    )
    rz = np.array(
        [[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return (rz @ ry @ rx).astype(np.float32)


class SixDRepNetPostprocessTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    MODEL_PATH = (
        ROOT
        / "model_repository"
        / "sixdrepnet_postprocess"
        / "1"
        / "model.py"
    )

    def setUp(self):
        self.module, self.model = load_backend(
            "sixdrepnet_postprocess", self.MODEL_PATH
        )
        self.request = lambda value: make_request(ROTATION_MATRIX=value)

    def assert_error(self, response, expected_message=None):
        self.assertIsInstance(response, FakeResponse)
        self.assertTrue(response.has_error())
        self.assertIsInstance(response.error(), FakeTritonError)
        if expected_message is not None:
            self.assertEqual(str(response.error()), expected_message)
        return response.error()

    def assert_success(self, response, batch_size=1):
        self.assertIsInstance(response, FakeResponse)
        self.assertFalse(response.has_error())
        tensors = response.output_tensors()
        self.assertEqual(len(tensors), 1)
        self.assertIsInstance(tensors[0], FakeTensor)
        self.assertEqual(tensors[0].name(), "HEAD_POSE")
        value = tensors[0].as_numpy()
        self.assertEqual(value.shape, (batch_size, 3))
        self.assertEqual(value.dtype, np.float32)
        self.assertTrue(value.flags.c_contiguous)
        return value

    def test_identity_decodes_to_zero_degrees(self):
        actual = self.module.decode_head_pose(np.eye(3, dtype=np.float32))

        np.testing.assert_allclose(
            actual, np.zeros(3, dtype=np.float32), rtol=0.0, atol=1e-6
        )

    def test_known_non_singular_rotations_decode_in_pitch_yaw_roll_order(self):
        cases = (
            (20.0, -30.0, 15.0),
            (-25.0, 40.0, -35.0),
            (10.0, 89.0, -45.0),
            (-15.0, -89.0, 70.0),
        )

        for expected in cases:
            with self.subTest(angles=expected):
                actual = self.module.decode_head_pose(rotation_matrix(*expected))
                np.testing.assert_allclose(
                    actual,
                    np.asarray(expected, dtype=np.float32),
                    rtol=0.0,
                    atol=2e-4,
                )

    def test_gimbal_lock_outputs_are_finite_and_reconstruct_input(self):
        cases = (
            (25.0, 90.0, -35.0),
            (-20.0, -90.0, 50.0),
        )

        for source_angles in cases:
            with self.subTest(angles=source_angles):
                source = rotation_matrix(*source_angles)
                decoded = self.module.decode_head_pose(source)

                self.assertEqual(decoded.shape, (3,))
                self.assertTrue(np.isfinite(decoded).all())
                reconstructed = rotation_matrix(*decoded)
                np.testing.assert_allclose(
                    reconstructed, source, rtol=0.0, atol=2e-6
                )

    def test_batch_of_four_equals_four_single_item_requests(self):
        angles = (
            (20.0, -30.0, 15.0),
            (-25.0, 40.0, -35.0),
            (10.0, 89.0, -45.0),
            (-15.0, -89.0, 70.0),
        )
        rotations = np.stack([rotation_matrix(*angle) for angle in angles])

        batched = self.assert_success(
            self.model.execute([self.request(rotations)])[0], batch_size=4
        )
        singles = np.concatenate(
            [
                self.assert_success(
                    self.model.execute(
                        [self.request(rotations[index:index + 1])]
                    )[0]
                )
                for index in range(4)
            ],
            axis=0,
        )
        np.testing.assert_allclose(batched, singles, rtol=0.0, atol=2e-6)

    def test_validation_errors_reject_non_batch_contract_and_bad_matrices(self):
        scaled = np.eye(3, dtype=np.float32)
        scaled[0, 0] = 1.01
        reflection = np.diag([1.0, 1.0, -1.0]).astype(np.float32)
        cases = (
            ("missing", make_request(), "missing required input ROTATION_MATRIX"),
            (
                "non_array",
                self.request([1, 2, 3]),
                "ROTATION_MATRIX must be a numpy array",
            ),
            (
                "unbatched",
                self.request(np.eye(3, dtype=np.float32)),
                "ROTATION_MATRIX batch must have shape [B,3,3]",
            ),
            (
                "wrong_trailing_shape",
                self.request(np.zeros((1, 3, 2), dtype=np.float32)),
                "ROTATION_MATRIX batch must have shape [B,3,3]",
            ),
            (
                "wrong_dtype",
                self.request(np.eye(3, dtype=np.float64)[None, ...]),
                "ROTATION_MATRIX must have dtype float32",
            ),
            (
                "empty_batch",
                self.request(np.zeros((0, 3, 3), dtype=np.float32)),
                "input batch must be non-empty",
            ),
            (
                "non_finite",
                self.request(np.full((1, 3, 3), np.nan, dtype=np.float32)),
                "batch index 0: ROTATION_MATRIX must contain only finite values",
            ),
            (
                "non_orthogonal",
                self.request(scaled[None, ...]),
                "batch index 0: ROTATION_MATRIX must be orthogonal",
            ),
            (
                "bad_determinant",
                self.request(reflection[None, ...]),
                "batch index 0: ROTATION_MATRIX determinant must be 1",
            ),
        )

        for name, request, detail in cases:
            with self.subTest(name=name):
                self.assert_error(
                    self.model.execute([request])[0],
                    f"INVALID_INPUT: sixdrepnet_postprocess: {detail}",
                )

    def test_bad_second_sample_rejects_whole_request_with_batch_index(self):
        rotations = np.stack(
            [np.eye(3, dtype=np.float32), np.zeros((3, 3), dtype=np.float32)]
        )
        good = rotation_matrix(20.0, -30.0, 15.0)[None, ...]

        responses = self.model.execute([self.request(rotations), self.request(good)])

        self.assertEqual(len(responses), 2)
        self.assert_error(
            responses[0],
            "INVALID_INPUT: sixdrepnet_postprocess: batch index 1: "
            "ROTATION_MATRIX must be orthogonal",
        )
        np.testing.assert_allclose(
            self.assert_success(responses[1])[0],
            np.array([20.0, -30.0, 15.0], dtype=np.float32),
            rtol=0.0,
            atol=2e-4,
        )

    def test_internal_error_is_logged_redacted_with_request_and_batch_indices(self):
        matrix = rotation_matrix(20.0, -30.0, 15.0)
        good_value = self.module.decode_head_pose(matrix)
        private_detail = r"runtime boom at C:\private\angles\decode.py:77"
        with mock.patch.object(
            self.module,
            "decode_head_pose",
            side_effect=[good_value, RuntimeError(private_detail), good_value],
        ), self.assertLogs("sixdrepnet_postprocess", level="ERROR") as captured:
            responses = self.model.execute(
                [
                    self.request(np.stack([matrix, matrix])),
                    self.request(matrix[None, ...]),
                ]
            )

        self.assertEqual(len(responses), 2)
        error = self.assert_error(responses[0])
        match = re.fullmatch(
            r"INTERNAL_ERROR: sixdrepnet_postprocess failed; incident=([0-9a-f]{12})",
            str(error),
        )
        self.assertIsNotNone(match)
        incident = match.group(1)
        self.assertNotIn("private", str(error).lower())
        self.assertNotIn("angles", str(error).lower())
        self.assert_success(responses[1])

        self.assertEqual(len(captured.records), 1)
        record = captured.records[0]
        self.assertIn("request_index=0", record.getMessage())
        self.assertIn("batch_index=1", record.getMessage())
        self.assertIn(f"incident={incident}", record.getMessage())
        self.assertIsNone(record.exc_info)
        self.assertNotIn(private_detail, captured.output[0])
        self.assertNotIn("private", captured.output[0].lower())
        self.assertNotIn("angles", captured.output[0].lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
