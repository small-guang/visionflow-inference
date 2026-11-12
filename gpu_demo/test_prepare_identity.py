"""Offline preparation checks; no CPU or GPU model inference is performed."""
import copy
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import onnx
from onnx import TensorProto, helper


def fixture_sources(cache):
    graph = helper.make_graph([helper.make_node("Identity", ["input"], ["output"])], "fixture",
                              [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 2, 2])],
                              [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 2, 2])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    data = model.SerializeToString()
    license_data = b"MIT License\n"
    (cache / "fixture.onnx").write_bytes(data)
    (cache / "fixture.LICENSE").write_bytes(license_data)
    revision = "a" * 40
    return {"schema_version": 1, "source_repository": "https://github.com/opencv/opencv_zoo",
            "source_revision": revision, "models": {"yunet_gpu": {
                "source_path": "models/fixture/model.onnx", "sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
                "cache_filename": "fixture.onnx", "weights_license": "MIT", "license_path": "models/fixture/LICENSE",
                "license_sha256": hashlib.sha256(license_data).hexdigest(), "license_size": len(license_data),
                "license_cache_filename": "fixture.LICENSE",
                "license_scope_url": "https://github.com/opencv/opencv_zoo/blob/" + revision + "/models/fixture/README.md#license"}}}


@contextmanager
def fixture_pose_source(root):
    """Small approved artifacts replace the large model only inside this test."""
    from gpu_demo import prepare
    project = root / "public-project"
    source = root / "readonly-source"
    native = source / prepare.NATIVE / "1"
    native.mkdir(parents=True)
    graph = helper.make_graph([helper.make_node("Identity", ["images"], ["output0"])], "pose-fixture",
                              [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 2, 2])],
                              [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 3, 2, 2])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.metadata_props.add(key="license", value="AGPL-3.0 License")
    data, engine = model.SerializeToString(), b"approved-test-engine"
    (native / "model.onnx").write_bytes(data)
    (native / "model_throughput.plan").write_bytes(engine)
    catalog = {"models": {prepare.NATIVE: {"source": "approved-source", "onnx_sha256": hashlib.sha256(data).hexdigest(),
                                         "onnx_size": len(data)}},
               "sources": {"approved-source": {"urls": ["https://example.com/approved-source"], "sha256": "b" * 64}}}
    catalog_path = project / "tools/model_pipeline/catalog.json"
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_text(json.dumps(catalog))
    for name in prepare.POSE_MODELS:
        path = project / "model_repository" / name
        (path / "1").mkdir(parents=True)
        config = ('name: "' + name + '"\nbackend: "tensorrt"\n'
                  'default_model_filename: "model_throughput.plan"\ninstance_group [{kind: KIND_GPU}]\n') if name == prepare.NATIVE else 'backend: "python"\n'
        (path / "config.pbtxt").write_text(config)
        if name != prepare.NATIVE:
            (path / "1/model.py").write_text("# Public fixture preprocessing source\n")
    with patch.multiple(prepare, PROJECT=project, PLAN_SHA256=hashlib.sha256(engine).hexdigest(), PLAN_SIZE=len(engine)):
        yield source


class IdentityPrepareTests(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module("gpu_demo.prepare_identity")
        except ModuleNotFoundError:
            self.api = None
        self.assertIsNotNone(self.api, "identity GPU preparation is missing")

    def test_verified_source_and_license_generate_native_gpu_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            sources = fixture_sources(cache)
            result = self.api.prepare_identity(sources, cache, root / "out", offline=True)
            model = root / "out/identity_model_repository/yunet_gpu"
            self.assertEqual((model / "1/model.onnx").read_bytes(), (cache / "fixture.onnx").read_bytes())
            config = (model / "config.pbtxt").read_text()
            self.assertIn('backend: "onnxruntime"', config)
            self.assertIn("KIND_GPU", config)
            self.assertIn("max_batch_size: 0", config)
            self.assertEqual(result["models"]["yunet_gpu"]["weights_license"], "MIT")
            self.assertFalse(result["gpu_verified"])

    def test_bad_weight_does_not_publish_a_partial_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            sources = fixture_sources(cache)
            (cache / "fixture.onnx").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "size|SHA256"):
                self.api.prepare_identity(sources, cache, root / "out", offline=True)
            self.assertFalse((root / "out/identity_model_repository").exists())

    def test_bad_license_does_not_publish_a_partial_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            sources = fixture_sources(cache)
            (cache / "fixture.LICENSE").write_bytes(b"MIT Licensx\n")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                self.api.prepare_identity(sources, cache, root / "out", offline=True)
            self.assertFalse((root / "out/identity_model_repository").exists())

    def test_unknown_license_and_unsafe_cache_name_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = fixture_sources(root)
            unknown = copy.deepcopy(sources)
            unknown["models"]["yunet_gpu"]["weights_license"] = "unverified"
            with self.assertRaisesRegex(ValueError, "license"):
                self.api.prepare_identity(unknown, root, root / "out", offline=True)
            sources["models"]["yunet_gpu"]["cache_filename"] = "../escape.onnx"
            with self.assertRaises(ValueError):
                self.api.prepare_identity(sources, root, root / "out", offline=True)

    def test_offline_missing_asset_does_not_use_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = fixture_sources(root)
            (root / "fixture.onnx").unlink()
            with self.assertRaises(FileNotFoundError):
                self.api.prepare_identity(sources, root, root / "out", offline=True)

    def test_direct_pose_source_does_not_require_pose_only_staging_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            sources = fixture_sources(cache)
            with fixture_pose_source(root) as source:
                try:
                    result = self.api.prepare_identity(sources, cache, root / "out", offline=True, source_repository=source)
                except TypeError as exc:
                    self.fail("direct source-repository preparation is missing: " + str(exc))
                self.assertIn("yolo26m_pose_onnx", result["models"])
                self.assertTrue((root / "out/identity_model_repository/yolo26m_pose_onnx/1/model_throughput.plan").is_file())
                self.assertFalse((root / "out/model_repository").exists())
                self.assertFalse((root / "out/model-manifest.json").exists())

    def test_direct_pose_source_still_rejects_unknown_weights_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            sources = fixture_sources(cache)
            with fixture_pose_source(root) as source:
                path = source / "yolo26m_pose_onnx/1/model.onnx"
                damaged = bytearray(path.read_bytes())
                damaged[-1] ^= 1
                path.write_bytes(damaged)
                try:
                    with self.assertRaisesRegex(ValueError, "SHA256"):
                        self.api.prepare_identity(sources, cache, root / "out", offline=True, source_repository=source)
                except TypeError as exc:
                    self.fail("direct source-repository validation is missing: " + str(exc))
                self.assertFalse((root / "out/identity_model_repository").exists())

    def test_final_repository_is_a_hash_checked_offline_source_without_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            sources = fixture_sources(cache)
            self.api.prepare_identity(sources, cache, root / "out", offline=True)
            absent_cache = root / "absent-cache"
            try:
                result = self.api.prepare_identity(sources, absent_cache, root / "out", offline=True)
            except FileNotFoundError as exc:
                self.fail("final model/license files should be reusable without a download cache: " + str(exc))
            self.assertIn("yunet_gpu", result["models"])
            self.assertFalse(absent_cache.exists())

    def test_final_output_cannot_overlap_the_read_only_pose_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = fixture_sources(root)
            with self.assertRaisesRegex(ValueError, "overlap"):
                self.api.prepare_identity(sources, root, root / "out", offline=True,
                                          source_repository=root / "out/identity_model_repository")

    def test_reused_final_weight_is_not_trusted_without_hash_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            sources = fixture_sources(cache)
            self.api.prepare_identity(sources, cache, root / "out", offline=True)
            model = root / "out/identity_model_repository/yunet_gpu/1/model.onnx"
            damaged = bytearray(model.read_bytes())
            damaged[-1] ^= 1
            model.write_bytes(damaged)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                self.api.prepare_identity(sources, root / "absent-cache", root / "out", offline=True)


if __name__ == "__main__":
    unittest.main()
