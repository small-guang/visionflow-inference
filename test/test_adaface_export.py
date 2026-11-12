import contextlib
import hashlib
import io
import json
import os
import stat
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import export_adaface_ir50_webface4m_onnx as exporter
import tools.onnx_export_validation as validation


class TinyAdaFace(torch.nn.Module):
    def forward(self, image):
        raw = image.mean(dim=(2, 3))
        raw = torch.cat((raw, raw), dim=1)
        norm = torch.linalg.vector_norm(raw, dim=1, keepdim=True)
        return raw / norm, norm


class ExportMismatchAdaFace(TinyAdaFace):
    def forward(self, image):
        embedding, norm = super().forward(image)
        if torch.onnx.is_in_onnx_export():
            embedding = -embedding
        return embedding, norm


class RepeatFirstBatchAdaFace(TinyAdaFace):
    def forward(self, image):
        embedding, norm = super().forward(image[:1])
        return (
            embedding.expand(image.shape[0], 6),
            norm.expand(image.shape[0], 1),
        )


def write_checker_valid_model(path, domain):
    embedding = np.full((1, 6), 1 / np.sqrt(6), dtype=np.float32)
    input_info = onnx.helper.make_tensor_value_info(
        "input",
        onnx.TensorProto.FLOAT,
        ["batch", 3, 8, 8],
    )
    output_info = onnx.helper.make_tensor_value_info(
        "embedding",
        onnx.TensorProto.FLOAT,
        ["batch", 6],
    )
    nodes = [
        onnx.helper.make_node(
            "ReduceMean",
            ["input"],
            ["channel_means"],
            axes=[2, 3],
            keepdims=0,
        ),
        onnx.helper.make_node(
            "Concat", ["channel_means", "channel_means"], ["raw"], axis=1
        ),
        onnx.helper.make_node(
            "ReduceL2", ["raw"], ["norm"], axes=[1], keepdims=1
        ),
        onnx.helper.make_node("Div", ["raw", "norm"], ["embedding"]),
    ]
    initializers = []
    if domain:
        nodes = []
        initializers = [
            onnx.helper.make_tensor(
                "embedding",
                onnx.TensorProto.FLOAT,
                [1, 6],
                embedding.flatten().tolist(),
            )
        ]
        output_info = onnx.helper.make_tensor_value_info(
            "embedding", onnx.TensorProto.FLOAT, [1, 6]
        )
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes,
            "domain-test",
            [input_info],
            [output_info],
            initializers,
        ),
        opset_imports=[onnx.helper.make_opsetid(domain, exporter.OPSET_VERSION)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return embedding


def valid_identity(**overrides):
    identity = {
        "schema_version": 1,
        "model": "adaface_ir50_webface4m",
        "source_repository": exporter.OFFICIAL_REPOSITORY,
        "source_file_id": exporter.OFFICIAL_FILE_ID,
        "checkpoint_filename": exporter.CHECKPOINT_NAME,
        "checkpoint_size": 123,
        "checkpoint_sha256": "a" * 64,
    }
    identity.update(overrides)
    return identity


class AdaFaceSourceTrustTests(unittest.TestCase):
    def test_pinned_official_source_and_license_exist(self):
        vendor = ROOT / "tools" / "vendor" / "adaface"
        source_bytes = (vendor / "net.py").read_bytes()
        license_bytes = (vendor / "LICENSE").read_bytes()
        source = source_bytes.decode("utf-8")
        metadata = (vendor / "来源说明.md").read_text(encoding="utf-8")
        license_text = license_bytes.decode("utf-8")

        self.assertIn("def build_model", source)
        self.assertIn("def IR_50", source)
        self.assertEqual(
            hashlib.sha256(source_bytes).hexdigest(),
            "b4db4eb0174a385fd29e5f616391b50d443f455990c8b88dcab1f8021af8ba4c",
        )
        self.assertEqual(
            hashlib.sha256(license_bytes).hexdigest(),
            "95b6e493eb9dba27f2150304e790ae254bab18d1611f4d6e2ade28fa3a271583",
        )
        self.assertIn("d8114b3ca8c54cd81ef59ac34c19eda1c548ca17", metadata)
        self.assertIn("1BmDRrhPsHSbXcWZoYFPJg2KJn1sd3QpN", metadata)
        self.assertIn("BGR", metadata)
        self.assertIn("MIT License", license_text)


class AdaFaceIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.identity_path = Path(self.temporary_directory.name) / "identity.json"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_identity(self, identity):
        self.identity_path.write_text(json.dumps(identity), encoding="utf-8")

    def test_valid_identity_normalizes_uppercase_sha256(self):
        identity = valid_identity(checkpoint_sha256="A" * 64)
        self.write_identity(identity)

        loaded = exporter.load_identity(self.identity_path)

        self.assertEqual(loaded, valid_identity(checkpoint_sha256="a" * 64))

    def test_identity_fields_must_be_exact(self):
        cases = {
            "missing": {
                key: value
                for key, value in valid_identity().items()
                if key != "model"
            },
            "extra": valid_identity(unexpected="value"),
        }
        for name, identity in cases.items():
            with self.subTest(name=name):
                self.write_identity(identity)
                with self.assertRaisesRegex(RuntimeError, "Identity fields mismatch"):
                    exporter.load_identity(self.identity_path)

    def test_schema_version_must_be_integer_one(self):
        for value in (True, False, 0, 2, "1", None):
            with self.subTest(value=value):
                self.write_identity(valid_identity(schema_version=value))
                with self.assertRaisesRegex(RuntimeError, "schema_version must be 1"):
                    exporter.load_identity(self.identity_path)

    def test_checkpoint_size_must_be_a_positive_integer(self):
        for value in (True, False, 0, -1, "123", None):
            with self.subTest(value=value):
                self.write_identity(valid_identity(checkpoint_size=value))
                with self.assertRaisesRegex(RuntimeError, "positive integer"):
                    exporter.load_identity(self.identity_path)

    def test_pinned_identity_metadata_must_match_constants(self):
        cases = (
            ("model", "wrong-model", "Identity model"),
            ("source_repository", "https://example.invalid", "source_repository"),
            ("source_file_id", "wrong-file-id", "source_file_id"),
            ("checkpoint_filename", "wrong.ckpt", "checkpoint_filename"),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                self.write_identity(valid_identity(**{field: value}))
                with self.assertRaisesRegex(RuntimeError, message):
                    exporter.load_identity(self.identity_path)

    def test_checkpoint_sha256_must_be_64_hexadecimal_characters(self):
        for value in (123, "a" * 63, "a" * 65, "g" * 64):
            with self.subTest(value=value):
                self.write_identity(valid_identity(checkpoint_sha256=value))
                with self.assertRaisesRegex(RuntimeError, "64-character hexadecimal"):
                    exporter.load_identity(self.identity_path)

    def test_identity_must_be_a_json_object(self):
        self.write_identity([])

        with self.assertRaisesRegex(RuntimeError, "must contain an object"):
            exporter.load_identity(self.identity_path)

    def test_duplicate_json_key_is_rejected(self):
        serialized = json.dumps(valid_identity())
        duplicated = serialized.replace(
            '"schema_version": 1',
            '"schema_version": 1, "schema_version": 1',
            1,
        )
        self.identity_path.write_text(duplicated, encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "duplicate.*schema_version"):
            exporter.load_identity(self.identity_path)

    def test_invalid_utf8_is_wrapped_as_runtime_error(self):
        self.identity_path.write_bytes(b"\xff")

        with self.assertRaisesRegex(RuntimeError, "Unable to load identity file"):
            try:
                exporter.load_identity(self.identity_path)
            except UnicodeError as error:
                self.fail(f"UnicodeError escaped load_identity: {error}")


class AdaFaceRecordedIdentityTests(unittest.TestCase):
    def test_default_recorded_identity_is_exact(self):
        expected_identity = {
            "schema_version": 1,
            "model": "adaface_ir50_webface4m",
            "source_repository": "https://github.com/mk-minchul/AdaFace",
            "source_file_id": "1BmDRrhPsHSbXcWZoYFPJg2KJn1sd3QpN",
            "checkpoint_filename": "adaface_ir50_webface4m.ckpt",
            "checkpoint_size": 596517267,
            "checkpoint_sha256": (
                "52cca7c64808fea6f44f9b9aee2b0e091bf96c1ab4f6e31bedcdf5d77009b4f8"
            ),
        }

        self.assertTrue(
            exporter.DEFAULT_IDENTITY.is_file(),
            "recorded AdaFace identity is missing",
        )
        self.assertEqual(
            exporter.load_identity(exporter.DEFAULT_IDENTITY),
            expected_identity,
        )

    def test_local_official_checkpoint_matches_recorded_identity(self):
        checkpoint_path = exporter.DEFAULT_CHECKPOINT
        if not checkpoint_path.is_file():
            self.skipTest("official AdaFace checkpoint is not available locally")

        identity = exporter.load_identity(exporter.DEFAULT_IDENTITY)

        self.assertEqual(identity["checkpoint_size"], checkpoint_path.stat().st_size)
        self.assertEqual(
            identity["checkpoint_sha256"],
            exporter.file_sha256(checkpoint_path),
        )


class AdaFaceCheckpointTrustTests(unittest.TestCase):
    def load_payload(self, payload):
        checkpoint_bytes = b"trusted"
        checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "checkpoint.ckpt"
            checkpoint.write_bytes(checkpoint_bytes)
            with unittest.mock.patch.object(
                exporter.torch, "load", return_value=payload
            ) as torch_load:
                state_dict = exporter.load_checkpoint(
                    checkpoint,
                    expected_size=len(checkpoint_bytes),
                    expected_sha256=checkpoint_sha256,
                )
            checkpoint_path = str(checkpoint)
        return state_dict, torch_load, checkpoint_path

    def test_hash_mismatch_fails_before_torch_load(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "checkpoint.ckpt"
            checkpoint.write_bytes(b"untrusted")

            with unittest.mock.patch.object(exporter.torch, "load") as torch_load:
                with self.assertRaisesRegex(RuntimeError, "Checkpoint SHA256 mismatch"):
                    exporter.load_checkpoint(
                        checkpoint,
                        expected_size=checkpoint.stat().st_size,
                        expected_sha256="0" * 64,
                    )

            torch_load.assert_not_called()

    def test_size_mismatch_fails_before_hashing_or_torch_load(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "checkpoint.ckpt"
            checkpoint.write_bytes(b"official-shaped-test")

            with (
                unittest.mock.patch.object(exporter, "file_sha256") as file_sha256,
                unittest.mock.patch.object(exporter.torch, "load") as torch_load,
            ):
                with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                    exporter.load_checkpoint(
                        checkpoint,
                        expected_size=1,
                        expected_sha256="0" * 64,
                    )

            file_sha256.assert_not_called()
            torch_load.assert_not_called()

    def test_only_model_state_is_returned_without_the_head(self):
        payload = {
            "state_dict": {
                "model.layer.weight": torch.ones(1),
                "head.kernel": "ignored non-tensor head value",
            }
        }

        state_dict, torch_load, checkpoint_path = self.load_payload(payload)

        self.assertEqual(set(state_dict), {"layer.weight"})
        torch_load.assert_called_once_with(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

    def test_checkpoint_payload_must_be_a_mapping(self):
        with self.assertRaisesRegex(RuntimeError, "Checkpoint must contain a mapping"):
            self.load_payload([])

    def test_state_dict_must_exist_and_be_a_mapping(self):
        for name, payload in (
            ("missing", {}),
            ("not-mapping", {"state_dict": []}),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(RuntimeError, "state_dict.*mapping"):
                    self.load_payload(payload)

    def test_state_dict_must_contain_model_parameters(self):
        for name, state_dict in (
            ("empty", {}),
            ("head-only", {"head.kernel": torch.ones(1)}),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(RuntimeError, "no model parameters"):
                    self.load_payload({"state_dict": state_dict})

    def test_model_state_values_must_be_tensors(self):
        payload = {"state_dict": {"model.layer.weight": "not-a-tensor"}}

        with self.assertRaisesRegex(TypeError, r"model\.layer\.weight"):
            self.load_payload(payload)


class AdaFaceWrapperTests(unittest.TestCase):
    def test_batched_wrapper_returns_normalized_embeddings(self):
        wrapper = exporter.BatchedAdaFace(TinyAdaFace()).eval()
        image = torch.stack(
            (
                torch.ones((3, 8, 8), dtype=torch.float32),
                torch.full((3, 8, 8), 2.0, dtype=torch.float32),
            )
        )

        embedding = wrapper(image)

        self.assertEqual(embedding.shape, (2, 6))
        torch.testing.assert_close(
            torch.linalg.vector_norm(embedding, dim=1),
            torch.ones(2),
            rtol=0.0,
            atol=exporter.NORM_ATOL,
        )


class AdaFaceOnnxExportTests(unittest.TestCase):
    def test_graph_shape_rejects_symbolic_and_unset_dimensions(self):
        for name, shape in (
            ("symbolic", ["channels", 8]),
            ("unset", [None, 8]),
        ):
            with self.subTest(name=name):
                value_info = onnx.helper.make_tensor_value_info(
                    "input",
                    onnx.TensorProto.FLOAT,
                    shape,
                )

                with self.assertRaisesRegex(
                    RuntimeError,
                    "dimension 0.*dim_value",
                ):
                    exporter.graph_shape(value_info)

    def test_candidate_rejects_nondefault_opset_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate_path = Path(directory) / "candidate.onnx"
            actual = write_checker_valid_model(candidate_path, "ai.onnx")
            wrapper = unittest.mock.Mock(
                side_effect=lambda image: torch.from_numpy(
                    np.repeat(actual, image.shape[0], axis=0)
                )
            )

            with self.assertRaisesRegex(RuntimeError, "default ONNX domain"):
                exporter.validate_candidate(
                    candidate_path,
                    wrapper,
                    torch.zeros((1, 3, 8, 8), dtype=torch.float32),
                    6,
                )

            wrapper.assert_not_called()

    def test_candidate_accepts_checker_valid_default_domain_model(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate_path = Path(directory) / "candidate.onnx"
            write_checker_valid_model(candidate_path, "")
            wrapper = exporter.BatchedAdaFace(TinyAdaFace()).eval()

            exporter.validate_candidate(
                candidate_path,
                wrapper,
                torch.ones((1, 3, 8, 8), dtype=torch.float32),
                6,
            )

            with torch.no_grad():
                embeddings = wrapper(
                    exporter._make_distinct_b2(
                        torch.ones((1, 3, 8, 8), dtype=torch.float32)
                    )
                )
            self.assertFalse(torch.allclose(embeddings[0], embeddings[1]))

    def test_toy_export_has_dynamic_batch_contract_and_returns_matching_digest(self):
        wrapper = exporter.BatchedAdaFace(TinyAdaFace()).eval()
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "model.onnx"

            digest = exporter.export_release(
                wrapper,
                output_path,
                embedding_size=6,
                input_shape=(1, 3, 8, 8),
            )

            model = onnx.load(output_path)
            onnx.checker.check_model(model)
            self.assertEqual(len(model.opset_import), 1)
            self.assertEqual(model.opset_import[0].version, exporter.OPSET_VERSION)
            self.assertEqual(len(model.graph.input), 1)
            self.assertEqual(len(model.graph.output), 1)
            self.assertEqual(model.graph.input[0].name, "input")
            self.assertEqual(model.graph.output[0].name, "embedding")
            self.assertEqual(
                model.graph.input[0].type.tensor_type.elem_type,
                onnx.TensorProto.FLOAT,
            )
            self.assertEqual(
                model.graph.output[0].type.tensor_type.elem_type,
                onnx.TensorProto.FLOAT,
            )
            self.assertEqual(
                tuple(
                    (dimension.dim_value, dimension.dim_param)
                    for dimension in model.graph.input[0].type.tensor_type.shape.dim
                ),
                ((0, "batch"), (3, ""), (8, ""), (8, "")),
            )
            self.assertEqual(
                tuple(
                    (dimension.dim_value, dimension.dim_param)
                    for dimension in model.graph.output[0].type.tensor_type.shape.dim
                ),
                ((0, "batch"), (6, "")),
            )
            self.assertEqual(digest, exporter.file_sha256(output_path))
            self.assertEqual(set(output_path.parent.iterdir()), {output_path})

            generator = torch.Generator().manual_seed(20260807)
            b1_example = torch.randn(
                (1, 3, 8, 8),
                generator=generator,
                dtype=torch.float32,
            )
            session = ort.InferenceSession(
                str(output_path),
                providers=["CPUExecutionProvider"],
            )
            for batch in (
                b1_example,
                exporter._make_distinct_b2(b1_example),
            ):
                with self.subTest(batch_size=batch.shape[0]), torch.no_grad():
                    expected = torch.cat(
                        [wrapper(row) for row in batch.split(1)], dim=0
                    ).cpu().numpy()
                    if batch.shape[0] == 2:
                        self.assertFalse(
                            np.allclose(
                                expected[0],
                                expected[1],
                                rtol=exporter.PARITY_RTOL,
                                atol=exporter.PARITY_ATOL,
                            )
                        )
                    actual = session.run(
                        ["embedding"],
                        {"input": batch.cpu().numpy()},
                    )[0]
                    np.testing.assert_allclose(
                        actual,
                        expected,
                        rtol=exporter.PARITY_RTOL,
                        atol=exporter.PARITY_ATOL,
                    )
                    self.assertTrue(
                        np.allclose(
                            np.linalg.norm(actual, axis=1),
                            np.ones(actual.shape[0]),
                            rtol=0.0,
                            atol=exporter.NORM_ATOL,
                        )
                    )
            del session

    def test_parity_failure_preserves_existing_model(self):
        wrapper = exporter.BatchedAdaFace(TinyAdaFace()).eval()
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            output_path = directory_path / "model.onnx"
            old_model = b"previously validated model"
            old_digest = hashlib.sha256(old_model).hexdigest()
            output_path.write_bytes(old_model)
            original_entries = set(directory_path.iterdir())

            with unittest.mock.patch.object(
                exporter,
                "validate_candidate",
                side_effect=RuntimeError("forced parity failure"),
                create=True,
            ):
                with self.assertRaisesRegex(RuntimeError, "forced parity failure"):
                    exporter.export_release(
                        wrapper,
                        output_path,
                        embedding_size=6,
                        input_shape=(1, 3, 8, 8),
                    )

            self.assertEqual(output_path.read_bytes(), old_model)
            self.assertEqual(set(directory_path.iterdir()), original_entries)
            self.assertEqual(
                exporter.file_sha256(output_path),
                old_digest,
            )

    def test_write_official_identity_validates_checkpoint_and_publishes_atomically(self):
        checkpoint_bytes = b"official-download-test"
        checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
        payload = {
            "state_dict": {
                "model.layer.weight": torch.ones(1),
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            checkpoint_path = directory_path / exporter.CHECKPOINT_NAME
            identity_path = directory_path / "identity.json"
            checkpoint_path.write_bytes(checkpoint_bytes)

            with (
                unittest.mock.patch.object(
                    exporter.torch,
                    "load",
                    return_value=payload,
                ) as torch_load,
                unittest.mock.patch.object(exporter, "build_model") as build_model,
            ):
                identity = exporter.write_official_identity(
                    checkpoint_path,
                    identity_path,
                    confirmed_source_file_id=exporter.OFFICIAL_FILE_ID,
                )

            self.assertEqual(identity["checkpoint_size"], len(checkpoint_bytes))
            self.assertEqual(identity["checkpoint_sha256"], checkpoint_sha256)
            self.assertEqual(identity["source_file_id"], exporter.OFFICIAL_FILE_ID)
            self.assertEqual(
                identity["checkpoint_filename"],
                exporter.CHECKPOINT_NAME,
            )
            torch_load.assert_called_once_with(
                str(checkpoint_path),
                map_location="cpu",
                weights_only=True,
            )
            build_model.assert_called_once()
            self.assertEqual(
                set(directory_path.iterdir()),
                {checkpoint_path, identity_path},
            )
            self.assertEqual(exporter.load_identity(identity_path), identity)
            if os.name == "posix":
                self.assertEqual(
                    stat.S_IMODE(identity_path.stat().st_mode),
                    0o644,
                )

    def test_write_identity_requires_tofu_confirmation_before_file_access(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "missing.ckpt"
            identity_path = Path(directory) / "identity.json"

            with (
                unittest.mock.patch.object(exporter, "file_sha256") as file_sha256,
                unittest.mock.patch.object(exporter.torch, "load") as torch_load,
            ):
                with self.assertRaises(TypeError):
                    exporter.write_official_identity(
                        checkpoint_path,
                        identity_path,
                    )

            file_sha256.assert_not_called()
            torch_load.assert_not_called()

    def test_wrong_tofu_confirmation_is_rejected_before_file_access(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "missing.ckpt"
            identity_path = Path(directory) / "identity.json"

            with (
                unittest.mock.patch.object(exporter, "file_sha256") as file_sha256,
                unittest.mock.patch.object(exporter.torch, "load") as torch_load,
            ):
                with self.assertRaisesRegex(ValueError, "TOFU caller assertion"):
                    exporter.write_official_identity(
                        checkpoint_path,
                        identity_path,
                        confirmed_source_file_id="wrong-file-id",
                    )

            file_sha256.assert_not_called()
            torch_load.assert_not_called()

    def test_write_identity_rejects_resolved_checkpoint_identity_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            shared_path = Path(directory) / exporter.CHECKPOINT_NAME
            original = b"official-download-test"
            shared_path.write_bytes(original)

            with (
                unittest.mock.patch.object(exporter, "file_sha256") as file_sha256,
                unittest.mock.patch.object(exporter.torch, "load") as torch_load,
            ):
                with self.assertRaisesRegex(RuntimeError, "must be different"):
                    exporter.write_official_identity(
                        shared_path,
                        shared_path.parent / "." / shared_path.name,
                        confirmed_source_file_id=exporter.OFFICIAL_FILE_ID,
                    )

            self.assertEqual(shared_path.read_bytes(), original)
            file_sha256.assert_not_called()
            torch_load.assert_not_called()

    def test_existing_identity_is_a_trust_root_and_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            checkpoint_path = directory_path / exporter.CHECKPOINT_NAME
            identity_path = directory_path / "identity.json"
            checkpoint_path.write_bytes(b"official-download-test")
            original_identity = b"existing trust root"
            identity_path.write_bytes(original_identity)

            with (
                unittest.mock.patch.object(exporter.torch, "load") as torch_load,
                unittest.mock.patch.object(exporter, "build_model") as build_model,
            ):
                with self.assertRaisesRegex(FileExistsError, "trust root"):
                    exporter.write_official_identity(
                        checkpoint_path,
                        identity_path,
                        confirmed_source_file_id=exporter.OFFICIAL_FILE_ID,
                    )

            self.assertEqual(identity_path.read_bytes(), original_identity)
            torch_load.assert_not_called()
            build_model.assert_not_called()

    def test_force_overwrite_updates_read_only_identity_and_preserves_mode(self):
        checkpoint_bytes = b"official-download-test"
        payload = {"state_dict": {"model.layer.weight": torch.ones(1)}}
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            checkpoint_path = directory_path / exporter.CHECKPOINT_NAME
            identity_path = directory_path / "identity.json"
            checkpoint_path.write_bytes(checkpoint_bytes)
            identity_path.write_bytes(b"existing trust root")
            os.chmod(identity_path, 0o444)
            identity_mode = stat.S_IMODE(identity_path.stat().st_mode)

            try:
                with (
                    unittest.mock.patch.object(
                        exporter.torch,
                        "load",
                        return_value=payload,
                    ),
                    unittest.mock.patch.object(exporter, "build_model"),
                ):
                    identity = exporter.write_official_identity(
                        checkpoint_path,
                        identity_path,
                        confirmed_source_file_id=exporter.OFFICIAL_FILE_ID,
                        overwrite=True,
                    )

                self.assertEqual(
                    identity["checkpoint_sha256"],
                    hashlib.sha256(checkpoint_bytes).hexdigest(),
                )
                self.assertEqual(
                    stat.S_IMODE(identity_path.stat().st_mode),
                    identity_mode,
                )
                self.assertEqual(
                    set(directory_path.iterdir()),
                    {checkpoint_path, identity_path},
                )
            finally:
                os.chmod(identity_path, 0o666)

    def test_force_identity_replace_failure_preserves_trust_root_without_tmp(self):
        checkpoint_bytes = b"official-download-test"
        payload = {"state_dict": {"model.layer.weight": torch.ones(1)}}
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            checkpoint_path = directory_path / exporter.CHECKPOINT_NAME
            identity_path = directory_path / "identity.json"
            checkpoint_path.write_bytes(checkpoint_bytes)
            original_identity = b"existing trust root"
            identity_path.write_bytes(original_identity)
            os.chmod(identity_path, 0o444)
            identity_mode = stat.S_IMODE(identity_path.stat().st_mode)

            try:
                with (
                    unittest.mock.patch.object(
                        exporter.torch,
                        "load",
                        return_value=payload,
                    ),
                    unittest.mock.patch.object(exporter, "build_model"),
                    unittest.mock.patch.object(
                        validation.os,
                        "replace",
                        side_effect=OSError("forced identity replace failure"),
                    ),
                    unittest.mock.patch.object(
                        exporter,
                        "load_identity",
                    ) as load_identity,
                ):
                    with self.assertRaisesRegex(
                        OSError,
                        "forced identity replace failure",
                    ):
                        exporter.write_official_identity(
                            checkpoint_path,
                            identity_path,
                            confirmed_source_file_id=exporter.OFFICIAL_FILE_ID,
                            overwrite=True,
                        )

                load_identity.assert_not_called()
                self.assertEqual(identity_path.read_bytes(), original_identity)
                self.assertEqual(
                    stat.S_IMODE(identity_path.stat().st_mode),
                    identity_mode,
                )
                self.assertEqual(
                    set(directory_path.iterdir()),
                    {checkpoint_path, identity_path},
                )
            finally:
                os.chmod(identity_path, 0o666)

    def test_real_parity_mismatch_rejects_new_release_without_artifacts(self):
        wrapper = exporter.BatchedAdaFace(ExportMismatchAdaFace()).eval()
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            output_path = directory_path / "model.onnx"

            with self.assertRaisesRegex(RuntimeError, "numerical mismatch"):
                exporter.export_release(
                    wrapper,
                    output_path,
                    embedding_size=6,
                    input_shape=(1, 3, 8, 8),
                )

            self.assertEqual(list(directory_path.iterdir()), [])

    def test_candidate_rejects_copying_first_embedding_across_b2(self):
        good_wrapper = exporter.BatchedAdaFace(TinyAdaFace()).eval()
        copying_wrapper = exporter.BatchedAdaFace(
            RepeatFirstBatchAdaFace()
        ).eval()
        example = torch.arange(
            1,
            1 + 3 * 8 * 8,
            dtype=torch.float32,
        ).reshape(1, 3, 8, 8)
        with torch.no_grad():
            distinct_embeddings = good_wrapper(
                exporter._make_distinct_b2(example)
            )
        self.assertFalse(
            torch.allclose(
                distinct_embeddings[0],
                distinct_embeddings[1],
                rtol=exporter.PARITY_RTOL,
                atol=exporter.PARITY_ATOL,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            candidate_path = Path(directory) / "copy-first.onnx"
            with torch.no_grad():
                torch.onnx.export(
                    copying_wrapper,
                    example,
                    str(candidate_path),
                    input_names=["input"],
                    output_names=["embedding"],
                    opset_version=exporter.OPSET_VERSION,
                    do_constant_folding=True,
                    dynamic_axes={
                        "input": {0: "batch"},
                        "embedding": {0: "batch"},
                    },
                    dynamo=False,
                )
            candidate = onnx.load(candidate_path)
            candidate.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 6
            onnx.save(candidate, candidate_path)

            with self.assertRaisesRegex(RuntimeError, "numerical mismatch"):
                exporter.validate_candidate(
                    candidate_path,
                    good_wrapper,
                    example,
                    6,
                )

    def test_torch_output_validation_rejects_invalid_contracts(self):
        example = torch.zeros((2, 3, 8, 8), dtype=torch.float32)
        cases = (
            ("shape", torch.ones((2, 5), dtype=torch.float32), "shape"),
            ("dtype", torch.ones((2, 6), dtype=torch.float64), "dtype"),
            (
                "non-finite",
                torch.tensor(
                    [
                        [float("nan"), 0.0, 0.0, 0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    dtype=torch.float32,
                ),
                "non-finite",
            ),
            (
                "non-unit",
                torch.ones((2, 6), dtype=torch.float32),
                "L2-normalized",
            ),
        )
        for name, output, message in cases:
            with self.subTest(name=name):
                wrapper = unittest.mock.Mock(return_value=output)
                with self.assertRaisesRegex(RuntimeError, message):
                    exporter.validate_torch_output(wrapper, example, 6)


class AdaFaceCliTests(unittest.TestCase):
    def test_parse_args_exposes_defaults_and_record_flag(self):
        defaults = exporter.parse_args([])

        self.assertEqual(defaults.checkpoint, exporter.DEFAULT_CHECKPOINT)
        self.assertEqual(defaults.identity, exporter.DEFAULT_IDENTITY)
        self.assertEqual(defaults.output, exporter.DEFAULT_OUTPUT)
        self.assertFalse(defaults.record_official_identity)
        self.assertIsNone(defaults.confirm_official_file_id)
        self.assertFalse(defaults.force_record_official_identity)

        recorded = exporter.parse_args(
            [
                "--record-official-identity",
                "--confirm-official-file-id",
                exporter.OFFICIAL_FILE_ID,
            ]
        )

        self.assertTrue(recorded.record_official_identity)
        self.assertEqual(
            recorded.confirm_official_file_id,
            exporter.OFFICIAL_FILE_ID,
        )

    def test_record_only_options_are_rejected_outside_record_mode(self):
        invalid_arguments = (
            ["--force-record-official-identity"],
            ["--confirm-official-file-id", exporter.OFFICIAL_FILE_ID],
            ["--record-official-identity"],
            [
                "--record-official-identity",
                "--confirm-official-file-id",
                "wrong-file-id",
            ],
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(
                io.StringIO()
            ):
                with self.assertRaises(SystemExit):
                    exporter.parse_args(arguments)

    def test_record_identity_main_prints_json_and_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            checkpoint_path = directory_path / exporter.CHECKPOINT_NAME
            identity_path = directory_path / "identity.json"
            output_path = directory_path / "model.onnx"
            checkpoint_path.write_bytes(b"official-download-test")
            identity = valid_identity(
                checkpoint_size=checkpoint_path.stat().st_size,
                checkpoint_sha256=hashlib.sha256(
                    checkpoint_path.read_bytes()
                ).hexdigest(),
            )
            arguments = unittest.mock.Mock(
                checkpoint=checkpoint_path,
                identity=identity_path,
                output=output_path,
                record_official_identity=True,
                confirm_official_file_id=exporter.OFFICIAL_FILE_ID,
                force_record_official_identity=True,
            )

            with (
                unittest.mock.patch.object(
                    exporter,
                    "parse_args",
                    return_value=arguments,
                ),
                unittest.mock.patch.object(
                    exporter,
                    "write_official_identity",
                    return_value=identity,
                ) as write_identity,
                unittest.mock.patch.object(exporter, "load_identity") as load_identity,
                unittest.mock.patch.object(exporter, "export_release") as export_release,
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                exporter.main([])

            write_identity.assert_called_once_with(
                checkpoint_path.resolve(),
                identity_path.resolve(),
                confirmed_source_file_id=exporter.OFFICIAL_FILE_ID,
                overwrite=True,
            )
            load_identity.assert_not_called()
            export_release.assert_not_called()
            self.assertEqual(json.loads(stdout.getvalue()), identity)

    def test_record_main_rejects_checkpoint_identity_collision_before_write(self):
        with tempfile.TemporaryDirectory() as directory:
            shared_path = Path(directory) / exporter.CHECKPOINT_NAME
            original = b"official-download-test"
            shared_path.write_bytes(original)
            arguments = unittest.mock.Mock(
                checkpoint=shared_path,
                identity=shared_path.parent / "." / shared_path.name,
                output=Path(directory) / "model.onnx",
                record_official_identity=True,
                confirm_official_file_id=exporter.OFFICIAL_FILE_ID,
                force_record_official_identity=True,
            )

            with (
                unittest.mock.patch.object(
                    exporter,
                    "parse_args",
                    return_value=arguments,
                ),
                unittest.mock.patch.object(
                    exporter,
                    "write_official_identity",
                ) as write_identity,
            ):
                with self.assertRaisesRegex(RuntimeError, "must be different"):
                    exporter.main([])

            write_identity.assert_not_called()
            self.assertEqual(shared_path.read_bytes(), original)

    def test_normal_main_loads_trust_chain_exports_and_prints_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            checkpoint_path = directory_path / exporter.CHECKPOINT_NAME
            identity_path = directory_path / "identity.json"
            output_path = directory_path / "model.onnx"
            checkpoint_path.write_bytes(b"trusted")
            identity = valid_identity()
            state_dict = {"layer.weight": torch.ones(1)}
            wrapper = unittest.mock.Mock()
            onnx_digest = "b" * 64
            arguments = unittest.mock.Mock(
                checkpoint=checkpoint_path,
                identity=identity_path,
                output=output_path,
                record_official_identity=False,
                confirm_official_file_id=None,
                force_record_official_identity=False,
            )
            calls = []

            def load_identity(path):
                calls.append(("load_identity", path))
                return identity

            def load_checkpoint(path, expected_size, expected_sha256):
                calls.append(
                    (
                        "load_checkpoint",
                        path,
                        expected_size,
                        expected_sha256,
                    )
                )
                return state_dict

            def build_model(received_state_dict):
                calls.append(("build_model", received_state_dict))
                return wrapper

            def export_release(received_wrapper, path):
                calls.append(("export_release", received_wrapper, path))
                return onnx_digest

            with (
                unittest.mock.patch.object(
                    exporter,
                    "parse_args",
                    return_value=arguments,
                ),
                unittest.mock.patch.object(
                    exporter,
                    "load_identity",
                    side_effect=load_identity,
                ),
                unittest.mock.patch.object(
                    exporter,
                    "load_checkpoint",
                    side_effect=load_checkpoint,
                ),
                unittest.mock.patch.object(
                    exporter,
                    "build_model",
                    side_effect=build_model,
                ),
                unittest.mock.patch.object(
                    exporter,
                    "export_release",
                    side_effect=export_release,
                ),
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                exporter.main([])

            self.assertEqual(
                calls,
                [
                    ("load_identity", identity_path.resolve()),
                    (
                        "load_checkpoint",
                        checkpoint_path.resolve(),
                        identity["checkpoint_size"],
                        identity["checkpoint_sha256"],
                    ),
                    ("build_model", state_dict),
                    ("export_release", wrapper, output_path.resolve()),
                ],
            )
            output = stdout.getvalue()
            self.assertIn(f"Checkpoint SHA256: {identity['checkpoint_sha256']}", output)
            self.assertIn(f"ONNX SHA256: {onnx_digest}", output)
            self.assertIn(
                "Input: input [batch,3,112,112] FLOAT, aligned BGR", output
            )
            self.assertIn(
                "Output: embedding [batch,512] FLOAT, per-row L2-normalized",
                output,
            )
            self.assertIn(f"Exported: {output_path.resolve()}", output)

    def test_normal_main_rejects_all_input_output_path_collisions_before_load(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            cases = (
                (
                    "checkpoint-identity",
                    directory_path / "shared.ckpt",
                    directory_path / "shared.ckpt",
                    directory_path / "model.onnx",
                ),
                (
                    "checkpoint-output",
                    directory_path / "model.onnx",
                    directory_path / "identity.json",
                    directory_path / "model.onnx",
                ),
                (
                    "identity-output",
                    directory_path / "checkpoint.ckpt",
                    directory_path / "model.onnx",
                    directory_path / "model.onnx",
                ),
            )
            for name, checkpoint_path, identity_path, output_path in cases:
                with self.subTest(name=name):
                    checkpoint_path.write_bytes(b"checkpoint")
                    before = {
                        path: path.read_bytes()
                        for path in directory_path.iterdir()
                        if path.is_file()
                    }
                    arguments = unittest.mock.Mock(
                        checkpoint=checkpoint_path,
                        identity=identity_path,
                        output=output_path,
                        record_official_identity=False,
                        confirm_official_file_id=None,
                        force_record_official_identity=False,
                    )

                    with (
                        unittest.mock.patch.object(
                            exporter,
                            "parse_args",
                            return_value=arguments,
                        ),
                        unittest.mock.patch.object(
                            exporter,
                            "load_identity",
                        ) as load_identity,
                        unittest.mock.patch.object(
                            exporter,
                            "export_release",
                        ) as export_release,
                    ):
                        with self.assertRaisesRegex(RuntimeError, "must be different"):
                            exporter.main([])

                    load_identity.assert_not_called()
                    export_release.assert_not_called()
                    self.assertEqual(
                        {
                            path: path.read_bytes()
                            for path in directory_path.iterdir()
                            if path.is_file()
                        },
                        before,
                    )


class AdaFaceArchitectureTests(unittest.TestCase):
    def test_architecture_loader_rejects_a_missing_spec_or_loader(self):
        spec_without_loader = unittest.mock.Mock(loader=None)
        for name, invalid_spec in (
            ("missing-spec", None),
            ("missing-loader", spec_without_loader),
        ):
            with self.subTest(name=name):
                with unittest.mock.patch.object(
                    exporter.importlib.util,
                    "spec_from_file_location",
                    return_value=invalid_spec,
                ) as spec_from_file_location:
                    with self.assertRaisesRegex(
                        RuntimeError, "Unable to load architecture"
                    ):
                        exporter.load_architecture_module()

                spec_from_file_location.assert_called_once_with(
                    "_vendored_adaface_net",
                    exporter.VENDORED_ARCHITECTURE,
                )

    def test_build_model_uses_ir50_strict_state_loading_and_eval(self):
        state_dict = {}
        model = TinyAdaFace()
        model.eval = unittest.mock.Mock(wraps=model.eval)
        model.load_state_dict = unittest.mock.Mock(wraps=model.load_state_dict)
        factory = unittest.mock.Mock(return_value=model)
        architecture = unittest.mock.Mock(build_model=factory)

        with unittest.mock.patch.object(
            exporter,
            "load_architecture_module",
            return_value=architecture,
        ):
            wrapper = exporter.build_model(state_dict)

        factory.assert_called_once_with("ir_50")
        model.eval.assert_called_once_with()
        model.load_state_dict.assert_called_once_with(state_dict, strict=True)
        self.assertIs(wrapper.model, model)
        self.assertFalse(wrapper.training)
        self.assertFalse(model.training)

    def test_build_model_does_not_swallow_strict_load_errors(self):
        state_dict = {"unexpected": torch.ones(1)}
        model = TinyAdaFace()
        model.load_state_dict = unittest.mock.Mock(
            side_effect=RuntimeError("strict load failed")
        )
        factory = unittest.mock.Mock(return_value=model)
        architecture = unittest.mock.Mock(build_model=factory)

        with unittest.mock.patch.object(
            exporter,
            "load_architecture_module",
            return_value=architecture,
        ):
            with self.assertRaisesRegex(RuntimeError, "strict load failed"):
                exporter.build_model(state_dict)

        factory.assert_called_once_with("ir_50")
        model.load_state_dict.assert_called_once_with(state_dict, strict=True)

    def test_real_vendored_ir50_batched_forward_smoke(self):
        torch.manual_seed(0)
        architecture = exporter.load_architecture_module()
        wrapper = exporter.BatchedAdaFace(
            architecture.build_model("ir_50")
        ).eval()
        image = torch.randn((2, 3, 112, 112), dtype=torch.float32)

        with torch.no_grad():
            embedding = wrapper(image)

        self.assertEqual(embedding.shape, (2, exporter.EXPECTED_EMBEDDING_SIZE))
        self.assertTrue(torch.isfinite(embedding).all().item())
        torch.testing.assert_close(
            torch.linalg.vector_norm(embedding, dim=1),
            torch.ones(2),
            rtol=0.0,
            atol=exporter.NORM_ATOL,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
