import hashlib
import os
import stat
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.onnx_export_validation as validation


def write_identity_model(path):
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])
    node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph([node], "identity", [input_info], [output_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, path)


class ValidateOnnxOutputTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = Path(self.temp_dir.name) / "identity.onnx"
        write_identity_model(self.model_path)
        self.input_value = np.array([[1.0, 2.0]], dtype=np.float32)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_accepts_matching_finite_output_and_returns_exact_array(self):
        from tools.onnx_export_validation import validate_onnx_output

        actual = validate_onnx_output(
            self.model_path,
            "input",
            "output",
            self.input_value,
            self.input_value,
        )

        np.testing.assert_array_equal(actual, self.input_value)

    def test_rejects_numerical_mismatch(self):
        from tools.onnx_export_validation import validate_onnx_output

        with self.assertRaisesRegex(RuntimeError, "numerical mismatch"):
            validate_onnx_output(
                self.model_path,
                "input",
                "output",
                self.input_value,
                self.input_value + 1,
            )

    def test_rejects_equal_values_with_different_dtype(self):
        from tools.onnx_export_validation import validate_onnx_output

        with self.assertRaisesRegex(RuntimeError, "dtype mismatch"):
            validate_onnx_output(
                self.model_path,
                "input",
                "output",
                self.input_value,
                self.input_value.astype(np.float64),
            )

    def test_rejects_nonfinite_expected_output(self):
        from tools.onnx_export_validation import validate_onnx_output

        with self.assertRaisesRegex(RuntimeError, "expected output contains"):
            validate_onnx_output(
                self.model_path,
                "input",
                "output",
                self.input_value,
                np.array([[np.nan, 2.0]], dtype=np.float32),
            )

    def test_rejects_output_shape_mismatch(self):
        from tools.onnx_export_validation import validate_onnx_output

        with self.assertRaisesRegex(RuntimeError, "output shape mismatch"):
            validate_onnx_output(
                self.model_path,
                "input",
                "output",
                self.input_value,
                np.array([1.0, 2.0], dtype=np.float32),
            )

    def test_rejects_wrong_output_name(self):
        from tools.onnx_export_validation import validate_onnx_output

        with self.assertRaisesRegex(RuntimeError, "output name"):
            validate_onnx_output(
                self.model_path,
                "input",
                "wrong",
                self.input_value,
                self.input_value,
            )

    def test_rejects_nonfinite_actual_output(self):
        from tools.onnx_export_validation import validate_onnx_output

        with self.assertRaisesRegex(RuntimeError, "actual output contains"):
            validate_onnx_output(
                self.model_path,
                "input",
                "output",
                np.array([[np.nan, 2.0]], dtype=np.float32),
                self.input_value,
            )

    def test_rejects_wrong_input_name(self):
        from tools.onnx_export_validation import validate_onnx_output

        with self.assertRaisesRegex(RuntimeError, "input name"):
            validate_onnx_output(
                self.model_path,
                "wrong",
                "output",
                self.input_value,
                self.input_value,
            )


class AtomicOnnxPublicationTests(unittest.TestCase):
    def test_publication_returns_the_validated_models_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.onnx"
            digest = validation.publish_onnx_atomically(
                target, lambda candidate: candidate.write_bytes(b"validated model")
            )
            self.assertEqual(digest, hashlib.sha256(b"validated model").hexdigest())
            self.assertEqual(target.read_bytes(), b"validated model")
            self.assertEqual(set(target.parent.iterdir()), {target})

    def test_hash_failure_preserves_existing_model_and_removes_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.onnx"
            target.write_bytes(b"previous model")
            with unittest.mock.patch.object(
                validation, "file_sha256", side_effect=OSError("forced hash failure")
            ), self.assertRaisesRegex(OSError, "forced hash failure"):
                validation.publish_onnx_atomically(
                    target, lambda candidate: candidate.write_bytes(b"new model")
                )
            self.assertEqual(target.read_bytes(), b"previous model")
            self.assertEqual(set(target.parent.iterdir()), {target})

    def test_failure_preserves_existing_target_and_removes_candidate(self):
        from tools.onnx_export_validation import publish_onnx_atomically

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            target_path = directory_path / "model.onnx"
            sentinel = b"previously validated model"
            target_path.write_bytes(sentinel)
            original_entries = set(directory_path.iterdir())

            def build_and_validate(candidate_path):
                candidate_path.write_bytes(b"invalid candidate")
                raise RuntimeError("forced validation failure")

            with self.assertRaisesRegex(RuntimeError, "forced validation failure"):
                publish_onnx_atomically(target_path, build_and_validate)

            self.assertEqual(target_path.read_bytes(), sentinel)
            self.assertEqual(set(directory_path.iterdir()), original_entries)

    def test_success_preserves_existing_publication_mode(self):
        from tools.onnx_export_validation import publish_onnx_atomically

        with tempfile.TemporaryDirectory() as directory:
            target_path = Path(directory) / "model.onnx"
            target_path.write_bytes(b"old")
            os.chmod(target_path, 0o640)
            expected_mode = stat.S_IMODE(target_path.stat().st_mode)

            publish_onnx_atomically(
                target_path, lambda candidate: candidate.write_bytes(b"validated")
            )

            self.assertEqual(target_path.read_bytes(), b"validated")
            self.assertEqual(stat.S_IMODE(target_path.stat().st_mode), expected_mode)

    def test_read_only_success_preserves_existing_publication_mode(self):
        from tools.onnx_export_validation import publish_onnx_atomically

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            target_path = directory_path / "model.onnx"
            target_path.write_bytes(b"old")
            os.chmod(target_path, 0o444)
            expected_mode = stat.S_IMODE(target_path.stat().st_mode)

            try:
                publish_onnx_atomically(
                    target_path,
                    lambda candidate: candidate.write_bytes(b"validated"),
                )

                self.assertEqual(target_path.read_bytes(), b"validated")
                self.assertEqual(
                    stat.S_IMODE(target_path.stat().st_mode),
                    expected_mode,
                )
                self.assertEqual(set(directory_path.iterdir()), {target_path})
            finally:
                for path in directory_path.iterdir():
                    os.chmod(path, 0o666)

    def test_mode_setting_failure_preserves_old_target_and_cleans_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            target_path = directory_path / "model.onnx"
            target_path.write_bytes(b"old")
            original_mode = stat.S_IMODE(target_path.stat().st_mode)
            real_chmod = os.chmod

            def fail_read_only_mode(path, mode):
                if stat.S_IMODE(mode) == 0o444:
                    raise OSError("forced mode-setting failure")
                return real_chmod(path, mode)

            with unittest.mock.patch.object(
                validation.os,
                "chmod",
                side_effect=fail_read_only_mode,
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "forced mode-setting failure",
                ):
                    validation.publish_file_atomically(
                        target_path,
                        lambda candidate: candidate.write_bytes(b"replacement"),
                        desired_mode=0o444,
                    )

            self.assertEqual(target_path.read_bytes(), b"old")
            self.assertEqual(
                stat.S_IMODE(target_path.stat().st_mode),
                original_mode,
            )
            self.assertEqual(set(directory_path.iterdir()), {target_path})

    def test_read_only_candidate_cleanup_does_not_mask_validation_error(self):
        from tools.onnx_export_validation import publish_onnx_atomically

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            target_path = directory_path / "model.onnx"
            target_path.write_bytes(b"old")

            def fail_with_read_only_candidate(candidate_path):
                candidate_path.write_bytes(b"invalid")
                os.chmod(candidate_path, 0o444)
                raise RuntimeError("forced validation failure")

            try:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "forced validation failure",
                ):
                    publish_onnx_atomically(
                        target_path,
                        fail_with_read_only_candidate,
                    )

                self.assertEqual(target_path.read_bytes(), b"old")
                self.assertEqual(set(directory_path.iterdir()), {target_path})
            finally:
                for path in directory_path.iterdir():
                    os.chmod(path, 0o666)

    def test_read_only_replace_failure_restores_mode_and_cleans_candidate(self):
        from tools.onnx_export_validation import publish_onnx_atomically

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            target_path = directory_path / "model.onnx"
            target_path.write_bytes(b"old")
            os.chmod(target_path, 0o444)
            original_mode = stat.S_IMODE(target_path.stat().st_mode)

            try:
                with unittest.mock.patch.object(
                    validation.os,
                    "replace",
                    side_effect=OSError("forced replace failure"),
                ):
                    with self.assertRaisesRegex(OSError, "forced replace failure"):
                        publish_onnx_atomically(
                            target_path,
                            lambda candidate: candidate.write_bytes(b"replacement"),
                        )

                self.assertEqual(target_path.read_bytes(), b"old")
                self.assertEqual(
                    stat.S_IMODE(target_path.stat().st_mode),
                    original_mode,
                )
                self.assertEqual(set(directory_path.iterdir()), {target_path})
            finally:
                for path in directory_path.iterdir():
                    os.chmod(path, 0o666)

    def test_cleanup_failure_is_not_allowed_to_mask_primary_error(self):
        from tools.onnx_export_validation import publish_onnx_atomically

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            target_path = directory_path / "model.onnx"
            target_path.write_bytes(b"old")
            candidates = []

            def fail_validation(candidate_path):
                candidates.append(candidate_path)
                candidate_path.write_bytes(b"invalid")
                raise RuntimeError("forced validation failure")

            path_type = type(target_path)
            real_unlink = path_type.unlink

            def fail_candidate_cleanup(path, *args, **kwargs):
                if path in candidates:
                    raise OSError("forced cleanup failure")
                return real_unlink(path, *args, **kwargs)

            with unittest.mock.patch.object(
                path_type,
                "unlink",
                new=fail_candidate_cleanup,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "forced validation failure",
                ) as raised:
                    publish_onnx_atomically(target_path, fail_validation)

            self.assertIn(
                "forced cleanup failure",
                "\n".join(getattr(raised.exception, "__notes__", ())),
            )
            self.assertEqual(target_path.read_bytes(), b"old")
            for candidate in candidates:
                candidate.unlink(missing_ok=True)
            self.assertEqual(set(directory_path.iterdir()), {target_path})

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits required")
    def test_new_publication_is_group_and_other_readable(self):
        from tools.onnx_export_validation import publish_onnx_atomically

        with tempfile.TemporaryDirectory() as directory:
            target_path = Path(directory) / "model.onnx"
            publish_onnx_atomically(
                target_path, lambda candidate: candidate.write_bytes(b"validated")
            )

            self.assertEqual(stat.S_IMODE(target_path.stat().st_mode) & 0o044, 0o044)


if __name__ == "__main__":
    unittest.main()
