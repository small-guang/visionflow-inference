import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.export_adaface_ir50_webface4m_onnx as adaface_exporter
import tools.export_chinese_clip_onnx as chinese_clip_exporter
import tools.export_osnet_ain_onnx as osnet_ain_exporter
import tools.export_tbps_clip_onnx as tbps_clip_exporter
import tools.dynamic_batch_onnx as dynamic_batch
from tools.model_pipeline.catalog import Catalog
from tools.model_pipeline.export_worker import export_graph
from tools.onnx_export_validation import publication_mode


class ExportMismatch(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export():
            return x + 1
        return x


class TinyClipBackbone(nn.Module):
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        flattened = image.flatten(1)
        return torch.stack((flattened.mean(1), flattened.square().mean(1)), dim=1)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        text = text.to(torch.float32)
        return torch.stack((text.mean(1), text.square().mean(1)), dim=1)


def make_distinct_batch(example: torch.Tensor) -> torch.Tensor:
    second = example.clone()
    second.reshape(1, -1)[0, -1] += 1
    return torch.cat((example, second), dim=0)


def write_identity_model(
    path: Path,
    input_name: str,
    output_name: str,
    shape: list[int],
    operator_type: str = "Identity",
    opset_version: int = 17,
) -> None:
    input_info = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, shape)
    output_info = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, shape)
    node = helper.make_node(operator_type, [input_name], [output_name])
    graph = helper.make_graph([node], "identity", [input_info], [output_info])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset_version)]
    )
    model.ir_version = 10
    onnx.save(model, path)


def write_face_qa_like_model(path: Path) -> None:
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, 112, 112]
    )
    output_info = helper.make_tensor_value_info("quality", TensorProto.FLOAT, [1, 10])
    output_shape = helper.make_tensor(
        "quality_shape", TensorProto.INT64, [2], [1, 10]
    )
    reduce_mean = helper.make_node(
        "ReduceMean",
        ["input"],
        ["mean"],
        axes=[1, 2, 3],
        keepdims=0,
    )
    unsqueeze = helper.make_node(
        "Unsqueeze", ["mean"], ["mean_column"], axes=[1]
    )
    expand = helper.make_node(
        "Expand", ["mean_column", "quality_shape"], ["quality"]
    )
    graph = helper.make_graph(
        [reduce_mean, unsqueeze, expand],
        "face_qa_like",
        [input_info],
        [output_info],
        initializer=[output_shape],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 6
    onnx.save(model, path)


def write_w600k_like_model(path: Path) -> None:
    input_info = helper.make_tensor_value_info(
        "input.1", TensorProto.FLOAT, [1, 3, 112, 112]
    )
    output_info = helper.make_tensor_value_info("683", TensorProto.FLOAT, [1, 512])
    batch_index = helper.make_tensor("batch_index", TensorProto.INT64, [], [0])
    width = helper.make_tensor("width", TensorProto.INT64, [1], [512])
    graph = helper.make_graph(
        [
            helper.make_node(
                "ReduceMean", ["input.1"], ["mean"], axes=[1, 2, 3], keepdims=0
            ),
            helper.make_node("Unsqueeze", ["mean"], ["mean_column"], axes=[1]),
            helper.make_node("Shape", ["input.1"], ["input_shape"]),
            helper.make_node("Gather", ["input_shape", "batch_index"], ["batch_size"]),
            helper.make_node("Unsqueeze", ["batch_size"], ["batch_size_vector"], axes=[0]),
            helper.make_node(
                "Concat", ["batch_size_vector", "width"], ["output_shape"], axis=0
            ),
            helper.make_node("Expand", ["mean_column", "output_shape"], ["683"]),
        ],
        "w600k_like",
        [input_info],
        [output_info],
        initializer=[batch_index, width],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 6
    onnx.save(model, path)


class ExporterValidationIntegrationTests(unittest.TestCase):
    def test_clip_export_examples_are_deterministic_without_global_rng_mutation(self):
        global_state = torch.random.get_rng_state().clone()
        for exporter in (chinese_clip_exporter, tbps_clip_exporter):
            with self.subTest(exporter=exporter.__name__):
                first = exporter._make_image_export_example()
                self.assertTrue(torch.equal(torch.random.get_rng_state(), global_state))
                second = exporter._make_image_export_example()
                self.assertTrue(torch.equal(first, second))
                self.assertTrue(torch.equal(torch.random.get_rng_state(), global_state))

    def test_adaface_exporter_uses_shared_atomic_publication(self):
        self.assertIs(
            adaface_exporter.publish_onnx_atomically,
            __import__(
                "tools.onnx_export_validation",
                fromlist=["publish_onnx_atomically"],
            ).publish_onnx_atomically,
        )

    def test_osnet_exporter_uses_shared_atomic_publication(self):
        self.assertIs(
            osnet_ain_exporter.publish_onnx_atomically,
            __import__(
                "tools.onnx_export_validation",
                fromlist=["publish_onnx_atomically"],
            ).publish_onnx_atomically,
        )

    def test_clip_exporters_reject_onnx_numerical_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            for exporter in (chinese_clip_exporter, tbps_clip_exporter):
                with self.subTest(exporter=exporter.__name__):
                    directory_path = Path(directory)
                    output_path = directory_path / f"{exporter.__name__}.onnx"
                    sentinel = b"previously validated model"
                    output_path.write_bytes(sentinel)
                    original_entries = set(directory_path.iterdir())
                    with self.assertRaisesRegex(RuntimeError, "numerical mismatch"):
                        exporter.export_one(
                            ExportMismatch().eval(),
                            torch.zeros((1, 2), dtype=torch.float32),
                            output_path,
                            "input",
                            "output",
                        )
                    self.assertEqual(output_path.read_bytes(), sentinel)
                    self.assertEqual(set(directory_path.iterdir()), original_entries)

    def test_clip_exporters_publish_batch_native_b2_models(self):
        backbone = TinyClipBackbone().eval()
        cases = (
            (
                chinese_clip_exporter,
                chinese_clip_exporter.ImageEncoder(backbone).eval(),
                torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2),
                "image",
                "unnorm_image_features",
            ),
            (
                chinese_clip_exporter,
                chinese_clip_exporter.TextEncoder(backbone).eval(),
                torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
                "text",
                "unnorm_text_features",
            ),
            (
                tbps_clip_exporter,
                tbps_clip_exporter.NormalizedImageEncoder(backbone).eval(),
                torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2) + 1,
                "image",
                "image_feat",
            ),
            (
                tbps_clip_exporter,
                tbps_clip_exporter.NormalizedTextEncoder(backbone).eval(),
                torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
                "text",
                "text_feat",
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            for index, case in enumerate(cases):
                exporter, wrapper, example, input_name, output_name = case
                with self.subTest(exporter=exporter.__name__, input=input_name):
                    output_path = Path(directory) / f"model-{index}.onnx"
                    with mock.patch.object(
                        exporter,
                        "validate_onnx_output",
                        wraps=exporter.validate_onnx_output,
                    ) as validate_onnx_output:
                        exporter.export_one(
                            wrapper,
                            example,
                            output_path,
                            input_name,
                            output_name,
                        )

                    self.assertEqual(
                        [
                            call.args[3].shape[0]
                            for call in validate_onnx_output.call_args_list
                        ],
                        [1, 2],
                    )

                    model = onnx.load(output_path)
                    self.assertEqual(
                        model.graph.input[0].type.tensor_type.shape.dim[0].dim_param,
                        "batch",
                    )
                    self.assertEqual(
                        model.graph.output[0].type.tensor_type.shape.dim[0].dim_param,
                        "batch",
                    )

                    session = ort.InferenceSession(
                        str(output_path), providers=["CPUExecutionProvider"]
                    )
                    for batch in (example, make_distinct_batch(example)):
                        with self.subTest(batch_size=batch.shape[0]), torch.no_grad():
                            expected = torch.cat(
                                [
                                    wrapper(row).detach().cpu()
                                    for row in batch.split(1)
                                ],
                                dim=0,
                            ).numpy()
                            if batch.shape[0] == 2:
                                self.assertFalse(
                                    np.allclose(expected[0], expected[1])
                                )
                            actual = session.run(
                                [output_name], {input_name: batch.cpu().numpy()}
                            )[0]
                            self.assertEqual(actual.shape, (batch.shape[0], 2))
                            np.testing.assert_allclose(
                                actual, expected, rtol=1e-5, atol=1e-6
                            )
                    del session

    def test_w600k_recipe_publishes_only_after_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_path = directory_path / "source.onnx"
            wrapped_path = directory_path / "wrapped.onnx"
            sentinel = b"previously validated model"
            write_w600k_like_model(source_path)
            wrapped_path.write_bytes(sentinel)
            original_entries = set(directory_path.iterdir())

            with mock.patch.object(
                dynamic_batch,
                "validate_dynamic_batch_model",
                side_effect=RuntimeError("forced validation failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "forced validation failure"):
                    export_graph(Catalog().models["w600k_r50_trt"], {}, source_path, wrapped_path)

            self.assertEqual(wrapped_path.read_bytes(), sentinel)
            self.assertEqual(set(directory_path.iterdir()), original_entries)

    def test_w600k_recipe_preserves_batched_contract_and_values(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.onnx"
            wrapped_path = Path(directory) / "wrapped.onnx"
            write_w600k_like_model(source_path)
            export_graph(Catalog().models["w600k_r50_trt"], {}, source_path, wrapped_path)

            wrapped = onnx.load(wrapped_path)
            onnx.checker.check_model(wrapped)
            self.assertEqual(wrapped.graph.input[0].name, "input.1")
            self.assertEqual(wrapped.graph.output[0].name, "683")
            self.assertEqual(wrapped.graph.input[0].type.tensor_type.shape.dim[0].dim_param, "batch")
            self.assertEqual(wrapped.graph.output[0].type.tensor_type.shape.dim[0].dim_param, "batch")
            self.assertEqual(len(wrapped.graph.node), 7)

            batch = np.random.default_rng(2026).standard_normal(
                (2, 3, 112, 112), dtype=np.float32
            )
            source_session = ort.InferenceSession(
                str(source_path), providers=["CPUExecutionProvider"]
            )
            wrapped_session = ort.InferenceSession(
                str(wrapped_path), providers=["CPUExecutionProvider"]
            )
            expected = np.stack(
                [source_session.run(["683"], {"input.1": row[None]})[0][0] for row in batch]
            )
            actual = wrapped_session.run(["683"], {"input.1": batch})[0]
            self.assertEqual(actual.shape, (2, 512))
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
            del source_session, wrapped_session

    def test_face_qa_recipe_preserves_batched_contract_and_values(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.onnx"
            wrapped_path = Path(directory) / "wrapped.onnx"
            write_face_qa_like_model(source_path)

            export_graph(Catalog().models["face_qa"], {}, source_path, wrapped_path)

            wrapped = onnx.load(wrapped_path)
            onnx.checker.check_model(wrapped)
            self.assertEqual(len(wrapped.graph.input), 1)
            self.assertEqual(len(wrapped.graph.output), 1)
            self.assertEqual(wrapped.graph.input[0].name, "input")
            self.assertEqual(wrapped.graph.output[0].name, "quality")
            self.assertEqual(
                tuple(
                    (dimension.dim_value, dimension.dim_param)
                    for dimension in wrapped.graph.input[0].type.tensor_type.shape.dim
                ),
                ((0, "batch"), (3, ""), (112, ""), (112, "")),
            )
            self.assertEqual(
                tuple(
                    (dimension.dim_value, dimension.dim_param)
                    for dimension in wrapped.graph.output[0].type.tensor_type.shape.dim
                ),
                ((0, "batch"), (10, "")),
            )
            self.assertEqual(len(wrapped.graph.node), 3)

            batch = np.random.default_rng(2026).standard_normal(
                (2, 3, 112, 112), dtype=np.float32
            )
            source_session = ort.InferenceSession(
                str(source_path), providers=["CPUExecutionProvider"]
            )
            wrapped_session = ort.InferenceSession(
                str(wrapped_path), providers=["CPUExecutionProvider"]
            )
            expected = np.stack(
                [source_session.run(["quality"], {"input": row[None]})[0][0] for row in batch]
            )
            actual = wrapped_session.run(["quality"], {"input": batch})[0]
            self.assertEqual(actual.shape, (2, 10))
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
            del source_session, wrapped_session

    def test_face_qa_recipe_publishes_only_after_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_path = directory_path / "source.onnx"
            wrapped_path = directory_path / "wrapped.onnx"
            sentinel = b"previously validated artifact"
            write_face_qa_like_model(source_path)
            wrapped_path.write_bytes(sentinel)
            original_entries = set(directory_path.iterdir())

            with mock.patch.object(
                dynamic_batch,
                "validate_dynamic_batch_model",
                side_effect=RuntimeError("forced validation failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "forced validation failure"):
                    export_graph(Catalog().models["face_qa"], {}, source_path, wrapped_path)

            self.assertEqual(wrapped_path.read_bytes(), sentinel)
            self.assertEqual(set(directory_path.iterdir()), original_entries)

    def test_face_qa_publication_mode_uses_default_or_existing_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            absent_path = directory_path / "absent.onnx"
            self.assertEqual(publication_mode(absent_path), 0o644)

            existing_path = directory_path / "existing.onnx"
            existing_path.write_bytes(b"existing")
            os.chmod(existing_path, 0o640)
            expected_mode = stat.S_IMODE(existing_path.stat().st_mode)
            self.assertEqual(
                publication_mode(existing_path), expected_mode
            )

    def test_face_qa_recipe_preserves_existing_publication_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_path = directory_path / "source.onnx"
            wrapped_path = directory_path / "wrapped.onnx"
            write_face_qa_like_model(source_path)
            wrapped_path.write_bytes(b"existing")
            os.chmod(wrapped_path, 0o640)
            expected_mode = stat.S_IMODE(wrapped_path.stat().st_mode)

            export_graph(Catalog().models["face_qa"], {}, source_path, wrapped_path)

            self.assertEqual(stat.S_IMODE(wrapped_path.stat().st_mode), expected_mode)

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits required")
    def test_face_qa_new_publication_is_group_and_other_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_path = directory_path / "source.onnx"
            wrapped_path = directory_path / "wrapped.onnx"
            write_face_qa_like_model(source_path)

            export_graph(Catalog().models["face_qa"], {}, source_path, wrapped_path)

            published_mode = stat.S_IMODE(wrapped_path.stat().st_mode)
            self.assertEqual(published_mode & 0o044, 0o044)


if __name__ == "__main__":
    unittest.main()
