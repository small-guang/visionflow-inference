"""Offline integrity and deployment checks; these do not claim GPU execution."""
import hashlib
import importlib
from pathlib import Path
import tempfile
import unittest


def implementation():
    try:
        return importlib.import_module("gpu_demo.prepare")
    except ModuleNotFoundError:
        return None


def fixture_manifest():
    return {
        "schema_version": 1,
        "source_repository": "https://github.com/opencv/opencv_zoo",
        "source_revision": "a" * 40,
        "models": {
            "yunet_gpu": {
                "source_path": "models/face_detection_yunet/model.onnx",
                "sha256": hashlib.sha256(b"model").hexdigest(),
                "size": 5,
                "weights_license": "MIT",
                "license_path": "models/face_detection_yunet/LICENSE",
                "license_sha256": hashlib.sha256(b"MIT License\n").hexdigest(),
                "license_size": 12,
                "license_scope_url": "https://github.com/opencv/opencv_zoo/blob/" + "a" * 40 + "/models/face_detection_yunet/README.md#license",
            }
        },
    }


def varint(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def field(number, value):
    if isinstance(value, int):
        return varint(number << 3) + varint(value)
    value = value.encode() if isinstance(value, str) else value
    return varint((number << 3) | 2) + varint(len(value)) + value


def tensor(name, shape):
    dimensions = b"".join(field(1, field(1, dim) if isinstance(dim, int) else field(2, dim)) for dim in shape)
    return field(1, name) + field(2, field(1, field(1, 1) + field(2, dimensions)))


def fixture_onnx():
    graph = field(11, tensor("images", ["batch", 3, "height", "width"]))
    graph += field(12, tensor("output0", ["batch", "anchors", 57]))
    metadata = field(1, "license") + field(2, "AGPL-3.0 License (https://ultralytics.com/license)")
    return field(1, 8) + field(7, graph) + field(14, metadata)


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.api = implementation()
        self.assertIsNotNone(self.api, "GPU preparation implementation is missing")

    def test_valid_source_model_and_license_are_accepted(self):
        self.api.validate_manifest(fixture_manifest())
        self.api.verify_blob(b"model", fixture_manifest()["models"]["yunet_gpu"]["sha256"], 5)

    def test_tampered_weight_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.api.verify_blob(b"other", fixture_manifest()["models"]["yunet_gpu"]["sha256"], 5)

    def test_truncated_weight_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "size"):
            self.api.verify_blob(b"mod", fixture_manifest()["models"]["yunet_gpu"]["sha256"], 5)

    def test_unknown_or_missing_weight_license_is_rejected(self):
        for license_value in (None, "", "unverified", "non-commercial"):
            with self.subTest(license_value=license_value):
                manifest = fixture_manifest()
                manifest["models"]["yunet_gpu"]["weights_license"] = license_value
                with self.assertRaisesRegex(ValueError, "license"):
                    self.api.validate_manifest(manifest)

    def test_manifest_must_have_frozen_upstream_revision(self):
        manifest = fixture_manifest()
        manifest["source_revision"] = "main"
        with self.assertRaisesRegex(ValueError, "revision"):
            self.api.validate_manifest(manifest)

    def test_unsafe_source_paths_and_model_names_are_rejected(self):
        for field, value in (("source_path", "../private/model.onnx"),
                             ("source_path", "C:/private/model.onnx"),
                             ("license_path", "models\\..\\LICENSE")):
            with self.subTest(field=field, value=value):
                manifest = fixture_manifest()
                manifest["models"]["yunet_gpu"][field] = value
                with self.assertRaises(ValueError):
                    self.api.validate_manifest(manifest)
        manifest = fixture_manifest()
        manifest["models"]["../escape"] = manifest["models"].pop("yunet_gpu")
        with self.assertRaises(ValueError):
            self.api.validate_manifest(manifest)

    def test_output_path_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ("../escape", "/absolute", "C:/escape", "..\\escape"):
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    self.api.checked_destination(root, relative)
            self.assertEqual(self.api.checked_destination(root, "yunet_gpu/1/model.onnx"),
                             root / "yunet_gpu" / "1" / "model.onnx")

    def test_license_bytes_must_match_frozen_license_digest(self):
        spec = fixture_manifest()["models"]["yunet_gpu"]
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.api.verify_blob(b"MIT Licensx\n", spec["license_sha256"], spec["license_size"])

    def test_config_binds_native_onnxruntime_to_gpu(self):
        signature = {"inputs": [{"name": "input", "dtype": "FP32", "shape": [1, 3, 320, 320]}],
                     "outputs": [{"name": "output", "dtype": "FP32", "shape": [1, 2100, 4]}]}
        config = self.api.render_config("yunet_gpu", signature)
        self.assertIn('backend: "onnxruntime"', config)
        self.assertIn('default_model_filename: "model.onnx"', config)
        self.assertIn("max_batch_size: 0", config)
        self.assertIn("kind: KIND_GPU", config)
        self.assertIn("gpus: [0]", config)
        self.assertNotIn("KIND_CPU", config)
        self.assertIn('name: "input"', config)
        self.assertIn("dims: [1, 3, 320, 320]", config)

    def test_config_keeps_dynamic_tensor_dimensions(self):
        signature = {"inputs": [{"name": "input", "dtype": "FP32", "shape": [1, 3, "height", 320]}],
                     "outputs": [{"name": "output", "dtype": "FP32", "shape": [1, 10]}]}
        config = self.api.render_config("yunet_gpu", signature)
        self.assertIn("dims: [1, 3, -1, 320]", config)

    def test_multiple_outputs_parse_with_real_triton_model_config_protobuf(self):
        from google.protobuf import text_format
        from tritonclient.grpc import model_config_pb2
        signature = {"inputs": [{"name": "input", "dtype": "FP32", "shape": [1, 3, 640, 640]}],
                     "outputs": [{"name": "cls_8", "dtype": "FP32", "shape": [1, 6400, 1]},
                                 {"name": "bbox_8", "dtype": "FP32", "shape": [1, 6400, 4]},
                                 {"name": "kps_8", "dtype": "FP32", "shape": [1, 6400, 10]}]}
        config = self.api.render_config("yunet_gpu", signature)
        try:
            parsed = text_format.Parse(config, model_config_pb2.ModelConfig())
        except text_format.ParseError as exc:
            self.fail("generated GPU config is invalid Triton protobuf: " + str(exc))
        self.assertEqual([tensor.name for tensor in parsed.output], ["cls_8", "bbox_8", "kps_8"])
        self.assertEqual(parsed.instance_group[0].kind, model_config_pb2.ModelInstanceGroup.KIND_GPU)

    def test_onnx_signature_and_embedded_license_are_read_from_graph(self):
        result = self.api.inspect_onnx(fixture_onnx())
        self.assertEqual(result["inputs"], [{"name": "images", "dtype": "FP32", "shape": ["batch", 3, "height", "width"]}])
        self.assertEqual(result["outputs"][0]["shape"], ["batch", "anchors", 57])
        self.assertIn("AGPL-3.0", result["metadata"]["license"])

    def test_truncated_onnx_cannot_generate_a_deployment(self):
        with self.assertRaisesRegex(ValueError, "protobuf|ONNX"):
            self.api.inspect_onnx(fixture_onnx()[:-1])

    def test_unknown_pose_onnx_is_rejected_by_read_only_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            native = source / "yolo26m_pose_onnx" / "1"
            native.mkdir(parents=True)
            (native / "model.onnx").write_bytes(b"unknown weight")
            with self.assertRaisesRegex(ValueError, "size|SHA256"):
                self.api.collect_pose(source)
            self.assertEqual((native / "model.onnx").read_bytes(), b"unknown weight")

    def test_non_gpu_native_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "GPU"):
            self.api.validate_native_config('backend: "tensorrt"\ninstance_group [{kind: KIND_CPU}]')

    def test_local_official_license_cannot_be_replaced_with_arbitrary_text(self):
        self.assertTrue(callable(getattr(self.api, "verify_agpl_license", None)), "frozen license verification is missing")
        with self.assertRaisesRegex(ValueError, "size|SHA256"):
            self.api.verify_agpl_license(b"unverified license")


if __name__ == "__main__":
    unittest.main()
