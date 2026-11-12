"""Release-artifact contract for the AdaFace IR50 WebFace4M ONNX model."""

import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL = (
    ROOT
    / "model_repository"
    / "adaface_ir50_webface4m_onnx"
    / "1"
    / "model.onnx"
)
CATALOG = ROOT / "tools" / "model_pipeline" / "catalog.json"


def _sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dynamic_batch_sample_shape(value_info):
    dimensions = value_info.type.tensor_type.shape.dim
    if not dimensions:
        raise AssertionError(f"{value_info.name!r} must have a batch dimension")
    batch_dimension = dimensions[0]
    if (
        batch_dimension.dim_param != "batch"
        or batch_dimension.HasField("dim_value")
    ):
        raise AssertionError(
            f"{value_info.name!r} dimension 0 must be symbolic 'batch'"
        )
    if not all(dimension.HasField("dim_value") for dimension in dimensions[1:]):
        raise AssertionError(
            f"{value_info.name!r} sample axes must use static dimensions"
        )
    return tuple(dimension.dim_value for dimension in dimensions[1:])


def _iter_tensor_protos(message):
    if message.DESCRIPTOR.full_name == "onnx.TensorProto":
        yield message
        return

    for field, value in message.ListFields():
        if field.message_type is None:
            continue
        children = value if field.is_repeated else (value,)
        for child in children:
            yield from _iter_tensor_protos(child)


def _assert_self_contained_model(model):
    for tensor in _iter_tensor_protos(model):
        if tensor.data_location == tensor.EXTERNAL:
            raise AssertionError(
                f"tensor {tensor.name!r} uses an external data location"
            )
        if tensor.external_data:
            raise AssertionError(
                f"tensor {tensor.name!r} contains external data metadata"
            )


@unittest.skipUnless(importlib.util.find_spec("onnx"), "onnx is not installed")
class AdaFaceModelArtifactTests(unittest.TestCase):
    def require_real_model(self):
        self.assertTrue(
            MODEL.is_file(),
            f"missing real AdaFace release artifact: {MODEL}",
        )

    def test_graph_contract_and_catalog_identity(self):
        import onnx

        self.require_real_model()
        model = onnx.load(MODEL, load_external_data=False)
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
        self.assertEqual(
            _dynamic_batch_sample_shape(graph_input), (3, 112, 112)
        )
        self.assertEqual(_dynamic_batch_sample_shape(graph_output), (512,))
        self.assertEqual(
            [(opset.domain, opset.version) for opset in model.opset_import],
            [("", 17)],
        )

        identity = json.loads(CATALOG.read_text(encoding="utf-8"))["models"][
            "adaface_ir50_webface4m_onnx"
        ]
        self.assertEqual(_sha256_file(MODEL), identity["onnx_sha256"])
        self.assertEqual(MODEL.stat().st_size, identity["onnx_size"])

    def test_release_is_a_self_contained_single_file_onnx(self):
        import onnx

        self.require_real_model()
        self.assertEqual(
            sorted(path.name for path in MODEL.parent.iterdir()),
            [".gitkeep", "model.onnx"],
        )
        model = onnx.load(MODEL, load_external_data=False)

        _assert_self_contained_model(model)

    @unittest.skipUnless(
        importlib.util.find_spec("onnxruntime"),
        "onnxruntime is not installed",
    )
    def test_cpu_inference_is_finite_and_l2_normalized(self):
        import onnxruntime as ort

        self.require_real_model()
        batch = np.random.default_rng(20260807).uniform(
            -1.0,
            1.0,
            size=(2, 3, 112, 112),
        ).astype(np.float32)
        batch = np.ascontiguousarray(batch)
        session = ort.InferenceSession(
            str(MODEL),
            providers=["CPUExecutionProvider"],
        )

        outputs = session.run(["embedding"], {"input": batch})
        expected = np.concatenate(
            [
                session.run(["embedding"], {"input": row[None, ...]})[0]
                for row in batch
            ],
            axis=0,
        )

        self.assertEqual(len(outputs), 1)
        embedding = outputs[0]
        self.assertEqual(embedding.shape, (2, 512))
        self.assertEqual(embedding.dtype, np.float32)
        self.assertTrue(np.isfinite(embedding).all())
        np.testing.assert_allclose(embedding, expected, rtol=1e-4, atol=1e-5)
        self.assertFalse(
            np.allclose(embedding[0], embedding[1], rtol=1e-5, atol=1e-6)
        )
        np.testing.assert_allclose(
            np.linalg.norm(embedding, axis=1),
            np.ones(2, dtype=np.float32),
            rtol=0,
            atol=1e-5,
        )


class AdaFaceArtifactTestImplementationTests(unittest.TestCase):
    def test_model_digest_does_not_load_the_whole_artifact(self):
        source = Path(__file__).read_text(encoding="utf-8")
        whole_file_expression = "MODEL." + "read_bytes()"

        self.assertNotIn(whole_file_expression, source)

    @unittest.skipUnless(importlib.util.find_spec("onnx"), "onnx is not installed")
    def test_self_contained_check_rejects_nested_external_tensor(self):
        import onnx

        external_tensor = onnx.helper.make_tensor(
            "external_weight",
            onnx.TensorProto.FLOAT,
            [1],
            [0.0],
        )
        external_tensor.data_location = onnx.TensorProto.EXTERNAL
        location = external_tensor.external_data.add()
        location.key = "location"
        location.value = "weights.bin"
        nested_graph = onnx.helper.make_graph(
            [
                onnx.helper.make_node(
                    "Constant",
                    [],
                    ["nested_value"],
                    value=external_tensor,
                )
            ],
            "nested_graph",
            [],
            [],
        )
        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                [
                    onnx.helper.make_node(
                        "NestedGraphCarrier",
                        [],
                        [],
                        body=nested_graph,
                    )
                ],
                "outer_graph",
                [],
                [],
            )
        )
        with self.assertRaisesRegex(AssertionError, "external"):
            _assert_self_contained_model(model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
