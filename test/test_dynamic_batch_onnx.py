import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.dynamic_batch_onnx as dynamic_batch
from tools.model_pipeline.catalog import Catalog
from tools.model_pipeline.export_worker import export_graph


def write_batch_one_identity(path: Path, *, operator_type: str = "Identity") -> None:
    """Write a graph that accepts B2 at runtime despite B1 boundary metadata."""
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])
    node = helper.make_node(operator_type, ["input"], ["output"])
    graph = helper.make_graph([node], "batch_one_identity", [input_info], [output_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, path)


def write_batch_one_int64_output_model(path: Path) -> None:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])
    output_info = helper.make_tensor_value_info("output", TensorProto.INT64, [1, 2])
    graph = helper.make_graph(
        [helper.make_node("Cast", ["input"], ["output"], to=TensorProto.INT64)],
        "int64_output",
        [input_info],
        [output_info],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, path)


def write_identity_model(path: Path, input_name: str, output_name: str, shape: list[int]) -> None:
    input_info = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, shape)
    output_info = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, shape)
    node = helper.make_node("Identity", [input_name], [output_name])
    graph = helper.make_graph([node], "identity", [input_info], [output_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 6
    onnx.save(model, path)


def write_w600k_like_model(path: Path) -> None:
    """Write a lightweight [B, 3, 112, 112] -> [B, 512] graph."""
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


def immutable_graph_summary(model: onnx.ModelProto) -> tuple[tuple[bytes, ...], ...]:
    return (
        tuple(node.SerializeToString() for node in model.graph.node),
        tuple(initializer.SerializeToString() for initializer in model.graph.initializer),
        tuple(opset.SerializeToString() for opset in model.opset_import),
        tuple(function.SerializeToString() for function in model.functions),
    )


def write_face_qa_like_model(path: Path) -> None:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 112, 112])
    output_info = helper.make_tensor_value_info("quality", TensorProto.FLOAT, [1, 10])
    output_shape = helper.make_tensor("quality_shape", TensorProto.INT64, [2], [1, 10])
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["input"], ["mean"], axes=[1, 2, 3], keepdims=0),
            helper.make_node("Unsqueeze", ["mean"], ["mean_column"], axes=[1]),
            helper.make_node("Expand", ["mean_column", "quality_shape"], ["quality"]),
        ],
        "face_qa_like",
        [input_info],
        [output_info],
        initializer=[output_shape],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 6
    onnx.save(model, path)


def dim_values(value_info: onnx.ValueInfoProto) -> tuple[tuple[int, str], ...]:
    return tuple(
        (dimension.dim_value, dimension.dim_param)
        for dimension in value_info.type.tensor_type.shape.dim
    )


class DynamicBatchOnnxTests(unittest.TestCase):
    def test_keypoint_recipe_preserves_batch_values_and_tensor_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.onnx"
            target_path = Path(directory) / "candidate.onnx"
            graph = helper.make_graph(
                [helper.make_node("Flatten", ["data"], ["flat"], axis=1),
                 helper.make_node("Slice", ["flat", "start", "end", "axis"], ["fc1"])],
                "keypoint_like",
                [helper.make_tensor_value_info("data", TensorProto.FLOAT, [1, 3, 192, 192])],
                [helper.make_tensor_value_info("fc1", TensorProto.FLOAT, [1, 212])],
                initializer=[helper.make_tensor(name, TensorProto.INT64, [1], [value])
                             for name, value in (("start", 0), ("end", 212), ("axis", 1))],
            )
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
            model.ir_version = 6
            onnx.save(model, source_path)
            original = source_path.read_bytes()

            export_graph(Catalog().models["face_keypoint_2d106"], {}, source_path, target_path)

            candidate = onnx.load(target_path)
            self.assertEqual(dim_values(candidate.graph.input[0]), ((0, "batch"), (3, ""), (192, ""), (192, "")))
            self.assertEqual(dim_values(candidate.graph.output[0]), ((0, "batch"), (212, "")))
            batch = np.random.default_rng(2026).standard_normal((2, 3, 192, 192), dtype=np.float32)
            session = ort.InferenceSession(str(target_path), providers=["CPUExecutionProvider"])
            actual = session.run(["fc1"], {"data": batch})[0]
            np.testing.assert_array_equal(actual, batch.reshape(2, -1)[:, :212])
            self.assertEqual(source_path.read_bytes(), original)
            del session

    def test_w600k_recipe_publishes_batched_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_path = directory_path / "source.onnx"
            target_path = directory_path / "target.onnx"
            write_w600k_like_model(source_path)

            export_graph(Catalog().models["w600k_r50_trt"], {}, source_path, target_path)

            model = onnx.load(target_path)
            self.assertEqual(model.graph.input[0].type.tensor_type.shape.dim[0].dim_param, "batch")
            self.assertEqual(model.graph.output[0].type.tensor_type.shape.dim[0].dim_param, "batch")
            self.assertEqual(len(model.graph.node), 7)
            batch = np.random.default_rng(2026).standard_normal(
                (2, 3, 112, 112), dtype=np.float32
            )
            source_session = ort.InferenceSession(
                str(source_path), providers=["CPUExecutionProvider"]
            )
            session = ort.InferenceSession(str(target_path), providers=["CPUExecutionProvider"])
            expected = np.stack(
                [source_session.run(["683"], {"input.1": row[None]})[0][0] for row in batch]
            )
            actual = session.run(["683"], {"input.1": batch})[0]
            self.assertEqual(actual.shape, (2, 512))
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
            del source_session, session

    def test_face_qa_recipe_publishes_batched_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_path = directory_path / "source.onnx"
            target_path = directory_path / "target.onnx"
            write_face_qa_like_model(source_path)

            export_graph(Catalog().models["face_qa"], {}, source_path, target_path)

            model = onnx.load(target_path)
            self.assertEqual(model.graph.input[0].type.tensor_type.shape.dim[0].dim_param, "batch")
            self.assertEqual(model.graph.output[0].type.tensor_type.shape.dim[0].dim_param, "batch")
            self.assertEqual(len(model.graph.node), 3)
            batch = np.random.default_rng(2026).standard_normal(
                (2, 3, 112, 112), dtype=np.float32
            )
            source_session = ort.InferenceSession(
                str(source_path), providers=["CPUExecutionProvider"]
            )
            session = ort.InferenceSession(str(target_path), providers=["CPUExecutionProvider"])
            expected = np.stack(
                [source_session.run(["quality"], {"input": row[None]})[0][0] for row in batch]
            )
            actual = session.run(["quality"], {"input": batch})[0]
            self.assertEqual(actual.shape, (2, 10))
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
            del source_session, session

    def test_set_symbolic_batch_requires_a_ranked_tensor(self):
        value_info = helper.make_tensor_value_info("scalar", TensorProto.FLOAT, [])

        with self.assertRaisesRegex(ValueError, "rank"):
            dynamic_batch.set_symbolic_batch(value_info)

    def test_contract_rejects_symbolic_or_unknown_non_batch_dimension(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.onnx"
            write_batch_one_identity(source_path)

            for name, configure_dimension, expected_description in (
                (
                    "symbolic",
                    lambda dimension: setattr(dimension, "dim_param", "width"),
                    "width",
                ),
                (
                    "unknown",
                    lambda dimension: None,
                    "unknown",
                ),
            ):
                with self.subTest(name=name):
                    model = onnx.load(source_path)
                    dimension = model.graph.input[0].type.tensor_type.shape.dim[1]
                    dimension.ClearField("dim_value")
                    dimension.ClearField("dim_param")
                    configure_dimension(dimension)

                    with self.assertRaisesRegex(
                        ValueError, rf"dimension 1.*{expected_description}"
                    ):
                        dynamic_batch._validate_model_contract(
                            model,
                            input_name="input",
                            output_names=("output",),
                            sample_shape=(2,),
                            output_sample_shapes=((2,),),
                        )

    def test_contract_rejects_invalid_output_dtype_or_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.onnx"
            write_batch_one_identity(source_path)

            for name, configure_output, expected_description in (
                (
                    "dtype",
                    lambda output: setattr(
                        output.type.tensor_type, "elem_type", TensorProto.INT64
                    ),
                    "float",
                ),
                (
                    "shape",
                    lambda output: setattr(
                        output.type.tensor_type.shape.dim[1], "dim_value", 3
                    ),
                    "dimension 1.*3",
                ),
            ):
                with self.subTest(name=name):
                    model = onnx.load(source_path)
                    configure_output(model.graph.output[0])

                    with self.assertRaisesRegex(
                        ValueError, rf"output 'output'.*{expected_description}"
                    ):
                        dynamic_batch._validate_model_contract(
                            model,
                            input_name="input",
                            output_names=("output",),
                            sample_shape=(2,),
                            output_sample_shapes=((2,),),
                        )

    def test_publisher_requires_one_output_sample_shape_per_output(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_path = directory_path / "source.onnx"
            target_path = directory_path / "target.onnx"
            write_batch_one_identity(source_path)

            with self.assertRaisesRegex(TypeError, "output_sample_shapes"):
                dynamic_batch.publish_dynamic_batch_model(
                    source_path,
                    target_path,
                    input_name="input",
                    output_names=("output",),
                    sample_shape=(2,),
                )
            with self.assertRaisesRegex(ValueError, "output_sample_shapes.*length"):
                dynamic_batch.publish_dynamic_batch_model(
                    source_path,
                    target_path,
                    input_name="input",
                    output_names=("output",),
                    sample_shape=(2,),
                    output_sample_shapes=(),
                )

    def test_publisher_makes_boundary_batch_symbolic_and_validates_b2(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_path = directory_path / "source.onnx"
            target_path = directory_path / "target.onnx"
            write_batch_one_identity(source_path)
            source = onnx.load(source_path)
            source_summary = immutable_graph_summary(source)

            dynamic_batch.publish_dynamic_batch_model(
                source_path,
                target_path,
                input_name="input",
                output_names=("output",),
                sample_shape=(2,),
                output_sample_shapes=((2,),),
            )

            model = onnx.load(target_path)
            onnx.checker.check_model(model)
            self.assertEqual(immutable_graph_summary(model), source_summary)
            self.assertEqual(dim_values(model.graph.input[0])[0], (0, "batch"))
            self.assertEqual(dim_values(model.graph.output[0])[0], (0, "batch"))
            self.assertEqual(dim_values(model.graph.input[0])[1], (2, ""))
            self.assertEqual(dim_values(model.graph.output[0])[1], (2, ""))

            batch = np.random.default_rng(123).standard_normal((2, 2), dtype=np.float32)
            source_session = ort.InferenceSession(
                str(source_path), providers=["CPUExecutionProvider"]
            )
            target_session = ort.InferenceSession(
                str(target_path), providers=["CPUExecutionProvider"]
            )
            expected = np.stack(
                [source_session.run(["output"], {"input": row[None]})[0][0] for row in batch]
            )
            actual = target_session.run(["output"], {"input": batch})[0]
            self.assertEqual(actual.shape, (2, 2))
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
            del source_session, target_session

    def test_validation_reports_named_numerical_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_path = directory_path / "source.onnx"
            candidate_path = directory_path / "candidate.onnx"
            write_batch_one_identity(source_path)
            write_batch_one_identity(candidate_path, operator_type="Neg")
            candidate = onnx.load(candidate_path)
            dynamic_batch.set_symbolic_batch(candidate.graph.input[0])
            dynamic_batch.set_symbolic_batch(candidate.graph.output[0])
            onnx.save(candidate, candidate_path)

            with self.assertRaisesRegex(
                RuntimeError, r"output.*max_abs_error="
            ):
                dynamic_batch.validate_dynamic_batch_model(
                    source_path,
                    candidate_path,
                    input_name="input",
                    output_names=("output",),
                    sample_shape=(2,),
                    output_sample_shapes=((2,),),
                )

    def test_validation_rejects_non_float32_source_and_candidate_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_path = directory_path / "source.onnx"
            candidate_path = directory_path / "candidate.onnx"
            write_batch_one_int64_output_model(source_path)
            write_batch_one_int64_output_model(candidate_path)
            candidate = onnx.load(candidate_path)
            dynamic_batch.set_symbolic_batch(candidate.graph.input[0])
            dynamic_batch.set_symbolic_batch(candidate.graph.output[0])
            onnx.save(candidate, candidate_path)

            with self.assertRaisesRegex(
                RuntimeError, r"output output: expected FLOAT.*int64"
            ):
                dynamic_batch.validate_dynamic_batch_model(
                    source_path,
                    candidate_path,
                    input_name="input",
                    output_names=("output",),
                    sample_shape=(2,),
                    output_sample_shapes=((2,),),
                )

    def test_validation_rejects_candidate_runtime_output_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_path = directory_path / "source.onnx"
            candidate_path = directory_path / "candidate.onnx"
            write_batch_one_identity(source_path)
            input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])
            output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])
            graph = helper.make_graph(
                [helper.make_node("ReduceMean", ["input"], ["output"], axes=[1], keepdims=1)],
                "wrong_runtime_shape",
                [input_info],
                [output_info],
            )
            candidate = helper.make_model(
                graph, opset_imports=[helper.make_opsetid("", 17)]
            )
            candidate.ir_version = 10
            dynamic_batch.set_symbolic_batch(candidate.graph.input[0])
            dynamic_batch.set_symbolic_batch(candidate.graph.output[0])
            onnx.save(candidate, candidate_path)

            with self.assertRaisesRegex(
                RuntimeError, r"output output: candidate output shape mismatch.*\(2, 1\)"
            ):
                dynamic_batch.validate_dynamic_batch_model(
                    source_path,
                    candidate_path,
                    input_name="input",
                    output_names=("output",),
                    sample_shape=(2,),
                    output_sample_shapes=((2,),),
                )

    def test_validation_failure_leaves_existing_target_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_path = directory_path / "source.onnx"
            target_path = directory_path / "target.onnx"
            sentinel = b"previously validated target"
            write_batch_one_identity(source_path)
            target_path.write_bytes(sentinel)
            original_entries = set(directory_path.iterdir())

            with mock.patch.object(
                dynamic_batch,
                "validate_dynamic_batch_model",
                side_effect=RuntimeError("forced candidate failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "forced candidate failure"):
                    dynamic_batch.publish_dynamic_batch_model(
                        source_path,
                        target_path,
                        input_name="input",
                        output_names=("output",),
                        sample_shape=(2,),
                        output_sample_shapes=((2,),),
                    )

            self.assertEqual(target_path.read_bytes(), sentinel)
            self.assertEqual(set(directory_path.iterdir()), original_entries)


if __name__ == "__main__":
    unittest.main()
