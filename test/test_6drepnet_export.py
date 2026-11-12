import contextlib
import hashlib
import importlib
import io
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

from tools import export_6drepnet_onnx as exporter


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "tools" / "vendor" / "sixdrepnet"
CHECKPOINT = ROOT / "tools" / "models" / "6DRepNet" / exporter.CHECKPOINT_NAME
RELEASE_CHECKPOINT_ENV = "SIXDREPNET_RELEASE_CHECKPOINT"
UPSTREAM_COMMIT = "464b2ba55c3d9b3d3b707c6271cf329a983ded20"
REPVGG_SHA256 = "063606c836cfabd788af805cbd32796b46c90c7dee92c770542d72e7478d83ff"
SE_BLOCK_SHA256 = "9ecc3e47c0c0c351c703a3e95018f65dc90915ca9797c976ee800da22474fdad"
LICENSE_SHA256 = "af1baea20cf7e28b66400084e04ff8754a473b6f6488b4cd749b33d0fd83bfb2"


class TinyPoseModel(nn.Module):
    def forward(self, images):
        pooled = images.mean(dim=(2, 3))
        poses = torch.cat((pooled, pooled + 0.5), dim=1)
        return exporter.ortho6d_to_rotation_matrix(poses)


def _without_optional_terminal_newline(data):
    if data.endswith(b"\r\n"):
        return data[:-2]
    if data.endswith(b"\n"):
        return data[:-1]
    return data


class SixDRepNetSourceTrustTests(unittest.TestCase):
    def test_vendored_source_has_pinned_provenance_and_license(self):
        source = (VENDOR / "来源说明.md").read_text(encoding="utf-8")
        repvgg = (VENDOR / "repvgg.py").read_text(encoding="utf-8")
        se_block = (VENDOR / "se_block.py").read_text(encoding="utf-8")
        license_text = (VENDOR / "LICENSE").read_text(encoding="utf-8")

        self.assertIn(UPSTREAM_COMMIT, source)
        self.assertIn("sixdrepnet/backbone/repvgg.py", source)
        self.assertIn("sixdrepnet/backbone/se_block.py", source)
        self.assertIn("def create_RepVGG_B1g2", repvgg)
        self.assertIn("from .se_block import SEBlock", repvgg)
        self.assertIn("class SEBlock", se_block)
        self.assertIn("MIT License", license_text)
        self.assertIn("Copyright (c) 2022 Thorsten Hempel", license_text)

        for expected_sha256 in (REPVGG_SHA256, SE_BLOCK_SHA256, LICENSE_SHA256):
            self.assertIn(expected_sha256, source)

    def test_vendored_files_match_pinned_upstream_sha256(self):
        local_import = b"from .se_block import SEBlock"
        upstream_import = b"from backbone.se_block import SEBlock"
        repvgg = (VENDOR / "repvgg.py").read_bytes()

        self.assertEqual(repvgg.count(local_import), 1)
        self.assertNotIn(upstream_import, repvgg)
        upstream_equivalent_repvgg = repvgg.replace(local_import, upstream_import)
        upstream_equivalent_repvgg = _without_optional_terminal_newline(
            upstream_equivalent_repvgg
        )

        self.assertEqual(
            hashlib.sha256(upstream_equivalent_repvgg).hexdigest(),
            REPVGG_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(
                _without_optional_terminal_newline(
                    (VENDOR / "se_block.py").read_bytes()
                )
            ).hexdigest(),
            SE_BLOCK_SHA256,
        )
        self.assertEqual(
            hashlib.sha256((VENDOR / "LICENSE").read_bytes()).hexdigest(),
            LICENSE_SHA256,
        )

    def test_vendored_repvgg_is_importable(self):
        repvgg = importlib.import_module("tools.vendor.sixdrepnet.repvgg")

        self.assertTrue(callable(repvgg.create_RepVGG_B1g2))
        model = repvgg.create_RepVGG_B1g2(deploy=True)
        self.assertGreater(sum(parameter.numel() for parameter in model.parameters()), 0)


class SixDRepNetCheckpointTests(unittest.TestCase):
    def test_size_mismatch_prevents_deserialization(self):
        data = b"untrusted"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            path.write_bytes(data)

            with mock.patch.object(
                exporter.torch, "load", return_value={}
            ) as torch_load:
                with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                    exporter.load_checkpoint(
                        path,
                        len(data) + 1,
                        hashlib.sha256(data).hexdigest(),
                    )

            torch_load.assert_not_called()

    def test_hash_mismatch_prevents_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            path.write_bytes(b"untrusted")

            with mock.patch.object(
                exporter.torch, "load", return_value={}
            ) as torch_load:
                with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                    exporter.load_checkpoint(path, path.stat().st_size, "0" * 64)

            torch_load.assert_not_called()

    def test_verified_checkpoint_deserializes_from_private_snapshot(self):
        data = b"trusted checkpoint bytes"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            path.write_bytes(data)
            loaded_tensor = torch.zeros(1)
            source_handle = path.open("rb")
            observed = {}

            def deserialize_snapshot(snapshot, **kwargs):
                with open(path, "wb") as mutated_source:
                    mutated_source.write(b"x" * len(data))
                observed["snapshot"] = snapshot
                observed["data"] = snapshot.read()
                observed["kwargs"] = kwargs
                return {"weight": loaded_tensor}

            with (
                mock.patch.object(Path, "open", return_value=source_handle),
                mock.patch.object(
                    exporter.torch,
                    "load",
                    side_effect=deserialize_snapshot,
                ) as torch_load,
            ):
                state_dict = exporter.load_checkpoint(
                    path,
                    len(data),
                    hashlib.sha256(data).hexdigest(),
                )

        torch_load.assert_called_once()
        snapshot = observed["snapshot"]
        self.assertIsNot(snapshot, source_handle)
        self.assertNotEqual(snapshot.name, source_handle.name)
        self.assertEqual(observed["data"], data)
        self.assertTrue(source_handle.closed)
        self.assertTrue(snapshot.closed)
        self.assertEqual(
            observed["kwargs"],
            {"map_location": "cpu", "weights_only": True},
        )
        self.assertIs(state_dict["weight"], loaded_tensor)

    def test_checkpoint_payload_requires_string_tensor_mapping(self):
        data = b"trusted checkpoint bytes"
        invalid_payloads = (
            ([], "mapping"),
            ({1: torch.zeros(1)}, "keys"),
            ({"weight": object()}, "values"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            path.write_bytes(data)
            for payload, message in invalid_payloads:
                with self.subTest(message=message):
                    with mock.patch.object(
                        exporter.torch,
                        "load",
                        return_value=payload,
                    ):
                        with self.assertRaisesRegex(TypeError, message):
                            exporter.load_checkpoint(
                                path,
                                len(data),
                                hashlib.sha256(data).hexdigest(),
                            )

    def test_build_model_rejects_missing_and_unexpected_keys(self):
        class TinyStateModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                for index in range(58):
                    self.register_buffer(f"tensor_{index}", torch.zeros(1))

        state_dict = dict(TinyStateModel().state_dict())
        del state_dict["tensor_0"]
        state_dict["unexpected"] = torch.zeros(1)

        with mock.patch.object(exporter, "SixDRepNet", TinyStateModel):
            with self.assertRaisesRegex(
                RuntimeError,
                r"missing=\['tensor_0'\].*unexpected=\['unexpected'\]",
            ):
                exporter.build_model(state_dict)

    def test_build_model_rejects_tensor_shape_mismatches(self):
        class TinyStateModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                for index in range(58):
                    self.register_buffer(f"tensor_{index}", torch.zeros(1))

        state_dict = dict(TinyStateModel().state_dict())
        state_dict["tensor_0"] = torch.zeros(2)

        with mock.patch.object(exporter, "SixDRepNet", TinyStateModel):
            with self.assertRaisesRegex(RuntimeError, r"shape mismatch.*tensor_0"):
                exporter.build_model(state_dict)

    def test_real_checkpoint_builds_deploy_model_and_runs_forward(self):
        configured_checkpoint = os.environ.get(RELEASE_CHECKPOINT_ENV)
        if configured_checkpoint is None:
            checkpoint = CHECKPOINT
            if not checkpoint.is_file():
                self.skipTest(
                    f"real 6DRepNet checkpoint is absent; set {RELEASE_CHECKPOINT_ENV} "
                    "for release verification"
                )
        else:
            self.assertTrue(
                configured_checkpoint.strip(),
                f"{RELEASE_CHECKPOINT_ENV} must not be empty when set",
            )
            checkpoint = Path(configured_checkpoint)
            self.assertTrue(
                checkpoint.is_file(),
                f"{RELEASE_CHECKPOINT_ENV} is not a file: {checkpoint}",
            )

        self.assertEqual(checkpoint.stat().st_size, 157315191)
        self.assertEqual(
            exporter.file_sha256(checkpoint),
            "9bdcd0e072df0d21e92288b40ea1d1b3cf7173c44728ab9c45c4d673f361e018",
        )
        state_dict = exporter.load_checkpoint(
            checkpoint,
            exporter.CHECKPOINT_SIZE,
            exporter.CHECKPOINT_SHA256,
        )
        model = exporter.build_model(state_dict)

        self.assertEqual(len(state_dict), 58)
        self.assertEqual(tuple(state_dict["linear_reg.weight"].shape), (6, 2048))
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 224, 224))
        self.assertEqual(tuple(output.shape), (1, 3, 3))


class SixDRepNetRotationTests(unittest.TestCase):
    def test_known_pose_preserves_x_y_z_column_semantics(self):
        pose = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        expected = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                ]
            ]
        )

        torch.testing.assert_close(
            exporter.ortho6d_to_rotation_matrix(pose),
            expected,
        )

    def test_non_degenerate_poses_produce_proper_rotation_matrices(self):
        poses = torch.tensor(
            [
                [1.0, 2.0, 3.0, -2.0, 1.0, 0.5],
                [0.5, -1.0, 2.0, 2.0, 1.5, -0.25],
            ]
        )

        rotations = exporter.ortho6d_to_rotation_matrix(poses)
        identity = torch.eye(3).expand(2, -1, -1)

        torch.testing.assert_close(
            rotations.transpose(1, 2) @ rotations,
            identity,
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            torch.linalg.det(rotations),
            torch.ones(2),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_degenerate_and_large_poses_produce_proper_rotation_matrices(self):
        large = 3.0e38
        poses = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.5e-6],
                [large, -2.0e38, 1.0e38, -1.0e38, large, 2.0e38],
            ],
            dtype=torch.float32,
        )

        rotations = exporter.ortho6d_to_rotation_matrix(poses)

        self.assertTrue(torch.isfinite(rotations).all().item())
        torch.testing.assert_close(
            rotations[0, :, 0],
            torch.tensor([1.0, 0.0, 0.0]),
        )
        torch.testing.assert_close(
            rotations.transpose(1, 2) @ rotations,
            torch.eye(3).expand(4, -1, -1),
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            torch.linalg.det(rotations),
            torch.ones(4),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_collinear_threshold_has_deterministic_boundary_behavior(self):
        threshold = 1e-6
        self.assertEqual(getattr(exporter, "COLLINEAR_EPS", None), threshold)
        poses = torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0, 0.0, threshold * 0.5],
                [1.0, 0.0, 0.0, 1.0, 0.0, threshold * 2.0],
            ],
            dtype=torch.float32,
        )
        expected = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                ],
            ]
        )

        torch.testing.assert_close(
            exporter.ortho6d_to_rotation_matrix(poses),
            expected,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_pose_shape_must_be_n_by_six(self):
        for shape in ((6,), (2, 5), (2, 6, 1)):
            with self.subTest(shape=shape):
                try:
                    exporter.ortho6d_to_rotation_matrix(torch.zeros(shape))
                except ValueError as error:
                    self.assertRegex(str(error), r"shape \[N, 6\]")
                except Exception as error:
                    self.fail(
                        f"expected ValueError, got {type(error).__name__}: {error}"
                    )
                else:
                    self.fail("ValueError not raised")


class SixDRepNetOnnxTests(unittest.TestCase):
    def test_export_has_dynamic_batch_signature_and_matches_torch(self):
        model = TinyPoseModel().eval()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.onnx"

            digest = exporter.export_release(model, output)

            self.assertEqual(digest, exporter.file_sha256(output))
            self.assertEqual(set(output.parent.iterdir()), {output})

            onnx_model = onnx.load(output, load_external_data=False)
            self.assertEqual(
                [(item.domain, item.version) for item in onnx_model.opset_import],
                [("", exporter.OPSET_VERSION)],
            )
            graph = onnx_model.graph
            self.assertEqual(len(graph.input), 1)
            self.assertEqual(graph.input[0].name, "image")
            self.assertEqual(
                graph.input[0].type.tensor_type.elem_type,
                onnx.TensorProto.FLOAT,
            )
            self.assertEqual(exporter.graph_shape(graph.input[0]), ("batch", 3, 224, 224))
            self.assertEqual(len(graph.output), 1)
            self.assertEqual(graph.output[0].name, "rotation_matrix")
            self.assertEqual(
                graph.output[0].type.tensor_type.elem_type,
                onnx.TensorProto.FLOAT,
            )
            self.assertEqual(exporter.graph_shape(graph.output[0]), ("batch", 3, 3))

            session = exporter.ort.InferenceSession(
                str(output),
                providers=["CPUExecutionProvider"],
            )
            for batch_size in (1, 4):
                with self.subTest(batch_size=batch_size):
                    value = np.linspace(
                        -1.0,
                        1.0,
                        batch_size * 3 * 224 * 224,
                        dtype=np.float32,
                    ).reshape(batch_size, 3, 224, 224)
                    with torch.no_grad():
                        expected = model(torch.from_numpy(value)).numpy()
                    actual = session.run(["rotation_matrix"], {"image": value})[0]
                    self.assertEqual(actual.shape, (batch_size, 3, 3))
                    np.testing.assert_allclose(
                        actual, expected, rtol=1e-4, atol=1e-5
                    )

    def test_export_validation_failure_preserves_existing_release(self):
        model = TinyPoseModel().eval()
        old_model = b"old model"
        old_digest = hashlib.sha256(old_model).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.onnx"
            output.write_bytes(old_model)

            with mock.patch.object(
                exporter,
                "validate_candidate",
                side_effect=RuntimeError("forced validation failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "forced validation failure"):
                    exporter.export_release(model, output)

            self.assertEqual(output.read_bytes(), old_model)
            self.assertEqual(exporter.file_sha256(output), old_digest)
            self.assertEqual(
                sorted(path.name for path in output.parent.iterdir()),
                ["model.onnx"],
            )

    def test_export_rejects_training_mode_before_creating_candidate(self):
        model = nn.BatchNorm2d(3)
        running_mean = model.running_mean.clone()
        running_variance = model.running_var.clone()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.onnx"
            with mock.patch.object(exporter, "publish_onnx_atomically") as publish:
                with self.assertRaisesRegex(RuntimeError, "evaluation mode"):
                    exporter.export_release(model, output)

            publish.assert_not_called()
            self.assertFalse(output.exists())

        self.assertTrue(model.training)
        torch.testing.assert_close(model.running_mean, running_mean)
        torch.testing.assert_close(model.running_var, running_variance)

    def test_export_rejects_non_cpu_or_non_float32_model_state(self):
        invalid_models = []

        float64_parameter = nn.Linear(3, 3).double().eval()
        invalid_models.append((float64_parameter, "float32"))

        float64_buffer = TinyPoseModel().eval()
        float64_buffer.register_buffer(
            "invalid_buffer",
            torch.zeros(1, dtype=torch.float64),
        )
        invalid_models.append((float64_buffer, "float32"))

        meta_buffer = TinyPoseModel().eval()
        meta_buffer.register_buffer(
            "invalid_buffer",
            torch.empty(1, device="meta"),
        )
        invalid_models.append((meta_buffer, "CPU"))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.onnx"
            for model, message in invalid_models:
                with self.subTest(message=message, model=type(model).__name__):
                    with mock.patch.object(
                        exporter,
                        "publish_onnx_atomically",
                    ) as publish:
                        with self.assertRaisesRegex(RuntimeError, message):
                            exporter.export_release(model, output)
                    publish.assert_not_called()

    def test_validate_candidate_rejects_missing_wrong_or_extra_opsets(self):
        model = TinyPoseModel().eval()
        example = torch.zeros(1, 3, 224, 224)
        invalid_imports = (
            (),
            (("", exporter.OPSET_VERSION - 1),),
            (("", exporter.OPSET_VERSION), ("unexpected.domain", 1)),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.onnx"
            exporter.export_release(model, output)
            valid_model = onnx.load(output, load_external_data=False)

            for imports in invalid_imports:
                with self.subTest(imports=imports):
                    candidate = onnx.ModelProto()
                    candidate.CopyFrom(valid_model)
                    del candidate.opset_import[:]
                    for domain, version in imports:
                        item = candidate.opset_import.add()
                        item.domain = domain
                        item.version = version
                    onnx.save_model(candidate, output)

                    with self.assertRaisesRegex(RuntimeError, "ONNX opset imports"):
                        exporter.validate_candidate(output, model, (example,))

    def test_validate_candidate_finds_recursive_external_tensors(self):
        model = TinyPoseModel().eval()
        example = torch.zeros(1, 3, 224, 224)

        def external_tensor(name):
            tensor = onnx.TensorProto()
            tensor.name = name
            tensor.data_type = onnx.TensorProto.FLOAT
            tensor.dims.append(1)
            tensor.data_location = onnx.TensorProto.EXTERNAL
            entry = tensor.external_data.add()
            entry.key = "location"
            entry.value = f"{name}.bin"
            return tensor

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.onnx"
            exporter.export_release(model, output)
            valid_model = onnx.load(output, load_external_data=False)

            nested_candidate = onnx.ModelProto()
            nested_candidate.CopyFrom(valid_model)
            nested_graph = onnx.helper.make_graph(
                [],
                "nested",
                [],
                [],
                initializer=[external_tensor("nested_external")],
            )
            attribute = nested_candidate.graph.node[0].attribute.add()
            attribute.name = "nested_graph"
            attribute.type = onnx.AttributeProto.GRAPH
            attribute.g.CopyFrom(nested_graph)

            sparse_candidate = onnx.ModelProto()
            sparse_candidate.CopyFrom(valid_model)
            sparse = sparse_candidate.graph.sparse_initializer.add()
            sparse.values.CopyFrom(external_tensor("sparse_external"))
            sparse.indices.CopyFrom(
                onnx.helper.make_tensor(
                    "sparse_indices",
                    onnx.TensorProto.INT64,
                    [1],
                    [0],
                )
            )
            sparse.dims.append(1)

            for candidate, tensor_name in (
                (nested_candidate, "nested_external"),
                (sparse_candidate, "sparse_external"),
            ):
                with self.subTest(tensor_name=tensor_name):
                    onnx.save_model(candidate, output)
                    with self.assertRaisesRegex(
                        RuntimeError,
                        rf"self-contained.*{tensor_name}",
                    ):
                        exporter.validate_candidate(output, model, (example,))

    def test_graph_shape_accepts_batch_symbol_and_rejects_unset_or_non_positive_dimensions(self):
        invalid_shapes = (
            ([1, None, 3], "unset"),
            ([1, 0, 3], "positive"),
            ([1, -1, 3], "positive"),
        )
        valid = onnx.helper.make_tensor_value_info(
            "value", onnx.TensorProto.FLOAT, ["batch", 3, 224, 224]
        )
        self.assertEqual(exporter.graph_shape(valid), ("batch", 3, 224, 224))
        for shape, message in invalid_shapes:
            with self.subTest(shape=shape):
                value_info = onnx.helper.make_tensor_value_info(
                    "value",
                    onnx.TensorProto.FLOAT,
                    shape,
                )
                with self.assertRaisesRegex(RuntimeError, message):
                    exporter.graph_shape(value_info)

    def test_rotation_validation_rejects_empty_batches_and_reports_batch_index(self):
        invalid_rotations = (
            (np.empty((0, 3, 3), dtype=np.float32), "shape"),
            (np.stack((np.eye(3), np.zeros((3, 3)))).astype(np.float32), "batch index 1"),
        )
        for rotation, message in invalid_rotations:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    exporter.validate_rotation_matrices(rotation)

    def test_validate_candidate_rejects_symbolic_nonleading_dimensions_and_mismatched_batch_symbols(self):
        model = TinyPoseModel().eval()
        example = torch.zeros(1, 3, 224, 224)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.onnx"
            exporter.export_release(model, output)
            valid_model = onnx.load(output, load_external_data=False)

            symbolic_nonleading = onnx.ModelProto()
            symbolic_nonleading.CopyFrom(valid_model)
            symbolic_nonleading.graph.input[0].type.tensor_type.shape.dim[1].dim_param = "channels"
            symbolic_nonleading.graph.input[0].type.tensor_type.shape.dim[1].ClearField("dim_value")

            mismatched_batch = onnx.ModelProto()
            mismatched_batch.CopyFrom(valid_model)
            mismatched_batch.graph.output[0].type.tensor_type.shape.dim[0].dim_param = "other_batch"

            for candidate, message in (
                (symbolic_nonleading, "shape"),
                (mismatched_batch, "batch"),
            ):
                with self.subTest(message=message):
                    onnx.save_model(candidate, output)
                    with self.assertRaisesRegex(RuntimeError, message):
                        exporter.validate_candidate(output, model, (example,))

    def test_rotation_validation_rejects_wrong_shape_dtype_and_values(self):
        invalid_rotations = (
            (np.eye(3, dtype=np.float32), "shape"),
            (np.eye(3, dtype=np.float64)[None], "dtype"),
            (
                np.full((1, 3, 3), np.nan, dtype=np.float32),
                "finite",
            ),
            (np.zeros((1, 3, 3), dtype=np.float32), "orthogonality"),
            (
                np.diag([1.0, 1.0, -1.0]).astype(np.float32)[None],
                "determinant",
            ),
        )
        for rotation, message in invalid_rotations:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    exporter.validate_rotation_matrices(rotation)

    def test_cli_defaults(self):
        arguments = exporter.parse_args([])

        self.assertEqual(arguments.checkpoint, exporter.DEFAULT_CHECKPOINT)
        self.assertEqual(arguments.output, exporter.DEFAULT_OUTPUT)

    def test_main_prints_authenticated_checkpoint_identity_without_reopening_it(self):
        state_dict = {"weight": torch.zeros(1)}
        model = TinyPoseModel().eval()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pth"
            output = Path(directory) / "model.onnx"
            output.write_bytes(b"onnx")
            digest = exporter.file_sha256(output)

            stdout = io.StringIO()
            with (
                mock.patch.object(
                    exporter,
                    "load_checkpoint",
                    return_value=state_dict,
                ) as load_checkpoint,
                mock.patch.object(exporter, "build_model", return_value=model),
                mock.patch.object(
                    exporter,
                    "export_release",
                    return_value=digest,
                ),
                contextlib.redirect_stdout(stdout),
            ):
                result = exporter.main(
                    ["--checkpoint", str(checkpoint), "--output", str(output)]
                )

        self.assertEqual(result, 0)
        load_checkpoint.assert_called_once_with(
            checkpoint,
            exporter.CHECKPOINT_SIZE,
            exporter.CHECKPOINT_SHA256,
        )
        text = stdout.getvalue()
        self.assertFalse(checkpoint.exists())
        self.assertIn(f"Checkpoint size: {exporter.CHECKPOINT_SIZE}", text)
        self.assertIn(f"Checkpoint SHA256: {exporter.CHECKPOINT_SHA256}", text)
        self.assertIn(f"ONNX size: {len(b'onnx')}", text)
        self.assertIn(f"ONNX SHA256: {digest}", text)
        self.assertIn("Input: image FLOAT [batch,3,224,224]\n", text)
        self.assertIn("Output: rotation_matrix FLOAT [batch,3,3]\n", text)

    def _assert_cli_collision(self, checkpoint, output):
        with (
            mock.patch.object(exporter, "load_checkpoint") as load_checkpoint,
            mock.patch.object(exporter, "export_release") as export_release,
        ):
            with self.assertRaisesRegex(RuntimeError, "path collision"):
                exporter.main(
                    ["--checkpoint", str(checkpoint), "--output", str(output)]
                )
        load_checkpoint.assert_not_called()
        export_release.assert_not_called()

    def test_cli_rejects_direct_and_resolved_path_aliases_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "model.onnx"
            collisions = (
                (output, output),
                (root / "missing" / ".." / "model.onnx", output),
            )
            for checkpoint, candidate_output in collisions:
                with self.subTest(checkpoint=checkpoint):
                    self._assert_cli_collision(checkpoint, candidate_output)

    def test_cli_rejects_hardlink_path_aliases_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            checkpoint = root / "checkpoint.pth"
            output = root / "model.onnx"
            checkpoint.write_bytes(b"same inode")
            try:
                os.link(checkpoint, output)
            except OSError as error:
                self.skipTest(f"hardlinks are not supported: {error}")
            self._assert_cli_collision(checkpoint, output)



if __name__ == "__main__":
    unittest.main()
