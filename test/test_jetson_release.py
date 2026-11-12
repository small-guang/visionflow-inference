"""Release artifact validation and standalone deployment behavior."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy" / "jetson"


class ReleaseInventoryTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((DEPLOY / "release.py").is_file(), "release helper is missing")
        spec = importlib.util.spec_from_file_location("jetson_release", DEPLOY / "release.py")
        self.release = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.release)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for relative, content in {
            "model_repository/native/config.pbtxt": 'backend: "tensorrt"\ndefault_model_filename: "model_throughput.plan"',
            "model_repository/native/1/model_throughput.plan": "engine",
            "model_repository/native/1/model.onnx": "source",
            "model_repository/ort/config.pbtxt": 'platform: "onnxruntime_onnx"',
            "model_repository/ort/1/model.onnx": "graph",
            "model_repository/pre/config.pbtxt": 'backend: "python"',
            "model_repository/pre/1/model.py": "# backend",
            "model_repository/pre/1/vocab.txt": "vocabulary",
            "model_repository/ensemble/config.pbtxt": 'platform: "ensemble"',
            "model_repository/ensemble/1/.gitkeep": "",
        }.items():
            self.write(relative, content)
        self.write("profiles/model_sets.json", json.dumps({
            "schema_version": 1, "model_sets": {
                "example": {"entry_point": "ensemble", "models": ["native", "ort", "pre", "ensemble"]}
            },
        }))
        self.write("deploy/jetson/release_models.json", json.dumps({
            "schema_version": 1, "aliases": ["example"],
        }))

    def write(self, relative, content):
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return target

    def inventory(self):
        return self.release.inventory(self.root)

    def test_manifest_hashes_models_and_resources_and_marks_missing_provenance(self):
        result = self.inventory()
        self.assertEqual(result["model_count"], 4)
        self.assertEqual(result["missing_provenance"], ["native/1/model_throughput.plan.json"])
        self.assertEqual(result["files"]["model_repository/pre/1/vocab.txt"],
                         hashlib.sha256(b"vocabulary").hexdigest())
        self.assertIn("profiles/model_sets.json", result["files"])

    def test_missing_active_engine_rejected_even_when_reference_onnx_exists(self):
        (self.root / "model_repository/native/1/model_throughput.plan").unlink()
        with self.assertRaisesRegex(ValueError, "model_throughput.plan"):
            self.inventory()

    def test_missing_python_backend_rejected(self):
        (self.root / "model_repository/pre/1/model.py").unlink()
        with self.assertRaisesRegex(ValueError, "model.py"):
            self.inventory()

    def test_unknown_alias_dependency_rejected(self):
        shutil.rmtree(self.root / "model_repository/ort")
        with self.assertRaisesRegex(ValueError, "ort"):
            self.inventory()

    def test_stale_provenance_rejected(self):
        self.write("model_repository/native/1/model_throughput.plan.json", json.dumps({
            "source_sha256": hashlib.sha256(b"source").hexdigest(),
            "plan_sha256": "0" * 64,
        }))
        with self.assertRaisesRegex(ValueError, "plan_sha256"):
            self.inventory()

    def test_valid_provenance_accepted(self):
        self.write("model_repository/native/1/model_throughput.plan.json", json.dumps({
            "source_sha256": hashlib.sha256(b"source").hexdigest(),
            "plan_sha256": hashlib.sha256(b"engine").hexdigest(),
            "build_command": ["trtexec", "--saveEngine=model_throughput.plan"],
            "precision": "FP16", "model": "native",
        }))
        self.assertEqual(self.inventory()["missing_provenance"], [])

    def test_logs_and_caches_are_excluded_but_tokenizer_data_is_preserved(self):
        self.write("model_repository/pre/1/__pycache__/model.pyc", "cache")
        self.write("model_repository/native/1/model.plan.build.log", "log")
        self.write("model_repository/pre/1/.secret", "secret")
        result = self.inventory()
        self.assertFalse(any("__pycache__" in key or key.endswith(".log") or ".secret" in key
                             for key in result["files"]))
        self.assertIn("model_repository/pre/1/vocab.txt", result["files"])

    def test_external_symlink_rejected(self):
        (self.root / "model_repository/pre/1/external").symlink_to("/etc/hostname")
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.inventory()

    def test_publishing_requires_verification_of_the_same_image_and_full_model_set(self):
        build = {"image_id": "sha256:abc", "model_count": 4,
                 "default_model_config": "orin_gpu", "default_runtime_model_count": 4}
        for report in (None, {"image_id": "sha256:def", "model_set": "all", "passed": True},
                       {"image_id": "sha256:abc", "model_set": "sbs", "passed": True},
                       {"image_id": "sha256:abc", "model_set": "all", "passed": False}):
            with self.subTest(report=report), self.assertRaises(ValueError):
                self.release.validate_verification(build, report)
        self.release.validate_verification(build, {
            "image_id": "sha256:abc", "model_set": "all", "passed": True,
            "model_config": "orin_gpu",
            "smoke": {"model_set": "all", "model_config": "orin_gpu", "ready_models": 4},
        })

    def test_snapshot_preserves_empty_ensemble_version_directories(self):
        context = self.root / "snapshot"
        self.assertTrue(hasattr(self.release, "snapshot"), "snapshot helper missing")
        self.release.snapshot(self.root, context)
        self.assertTrue((context / "model_repository/ensemble/1").is_dir())
        self.assertFalse((context / "model_repository/ensemble/1/.gitkeep").exists())
        self.release.inventory(context)

    def test_snapshot_excludes_unselected_services_and_rewrites_only_packaged_manifest(self):
        path = self.root / "profiles/model_sets.json"
        source = json.loads(path.read_text())
        source["model_sets"]["unused"] = {
            "entry_point": "unused_ensemble", "models": ["pre", "unused_ort", "unused_ensemble"],
        }
        path.write_text(json.dumps(source))
        original = path.read_bytes()
        self.write("model_repository/unused_ort/config.pbtxt", 'platform: "onnxruntime_onnx"')
        self.write("model_repository/unused_ort/1/model.onnx", "unused graph")
        self.write("model_repository/unused_ensemble/config.pbtxt", 'platform: "ensemble"')
        self.write("model_repository/unused_ensemble/1/.gitkeep", "")
        context = self.root / "snapshot"
        self.release.snapshot(self.root, context)
        self.assertEqual(path.read_bytes(), original)
        packaged = json.loads((context / "profiles/model_sets.json").read_text())
        self.assertEqual(list(packaged["model_sets"]), ["example"])
        self.assertTrue((context / "model_repository/pre/1/model.py").is_file(), "shared dependency must remain")
        self.assertFalse((context / "model_repository/unused_ort").exists())
        self.assertFalse((context / "model_repository/unused_ensemble").exists())
        self.assertTrue((self.root / "model_repository/unused_ort/1/model.onnx").is_file())
        manifest = self.release.start_triton.load_manifest(context / "profiles/model_sets.json")
        with self.assertRaisesRegex(ValueError, "unknown model-set alias"):
            self.release.start_triton.resolve_model_set("unused", manifest)

    def test_check_does_not_require_artifacts_of_excluded_services(self):
        source = json.loads((self.root / "profiles/model_sets.json").read_text())
        source["model_sets"]["unused"] = {"entry_point": "missing", "models": ["missing"]}
        self.write("profiles/model_sets.json", json.dumps(source))
        self.assertTrue(hasattr(self.release, "release_aliases"), "release selection loader missing")
        aliases = self.release.release_aliases(self.root)
        result = self.release.inventory(self.root, aliases=aliases)
        self.assertEqual(result["model_count"], 4)
        self.assertEqual(result["model_sets"], ["example"])

    def test_release_selection_rejects_empty_duplicate_all_and_unknown_aliases(self):
        self.assertTrue(hasattr(self.release, "release_aliases"), "release selection loader missing")
        for aliases in ([], ["example", "example"], ["all"], ["unknown"], "example", [1]):
            with self.subTest(aliases=aliases):
                self.write("deploy/jetson/release_models.json", json.dumps({"schema_version": 1, "aliases": aliases}))
                with self.assertRaises(ValueError):
                    self.release.release_aliases(self.root)

    def test_jetson_selection_excludes_three_services_and_their_entire_chains(self):
        self.assertTrue((DEPLOY / "release_models.json").is_file(), "Jetson release selection is missing")
        original = (ROOT / "profiles/model_sets.json").read_text()
        self.write("profiles/model_sets.json", original)
        self.write("deploy/jetson/release_models.json", (DEPLOY / "release_models.json").read_text())
        source = json.loads(original)["model_sets"]
        for definition in source.values():
            for model in definition["models"]:
                self.write(f"model_repository/{model}/config.pbtxt", 'platform: "ensemble"')
                self.write(f"model_repository/{model}/1/.gitkeep", "")
        context = self.root / "snapshot"
        self.release.snapshot(self.root, context)
        manifest = self.release.start_triton.load_manifest(context / "profiles/model_sets.json")
        excluded_aliases = {"adaface", "osnet", "sixdrepnet"}
        self.assertEqual(set(manifest.model_sets), set(source) - excluded_aliases)
        models = set(self.release.start_triton.resolve_model_set("all", manifest))
        self.assertEqual(len(models), 38)
        self.assertEqual({p.name for p in (context / "model_repository").iterdir()}, models)
        self.assertFalse(models & {m for a in excluded_aliases for m in source[a]["models"]})

    def test_runtime_context_excludes_native_reference_but_keeps_ort_onnx(self):
        context = self.root / "snapshot"
        self.release.snapshot(self.root, context)
        release = self.release.inventory(context)
        self.assertTrue(hasattr(self.release, "archive_reference_onnx"), "reference archival helper missing")
        self.release.archive_reference_onnx(context, self.root / "reference_onnx", release)
        reference = "model_repository/native/1/model.onnx"
        self.assertNotIn(reference, release["files"])
        self.assertEqual(release["reference_onnx"][reference], hashlib.sha256(b"source").hexdigest())
        self.assertFalse((context / reference).exists())
        self.assertEqual((self.root / "reference_onnx/native/1/model.onnx").read_text(), "source")
        self.assertTrue((context / "model_repository/ort/1/model.onnx").exists())
        self.assertTrue((context / "model_repository/native/1/model_throughput.plan").exists())
        self.assertTrue((self.root / reference).exists(), "development reference must be preserved")

    def test_ensemble_without_version_is_rejected(self):
        shutil.rmtree(self.root / "model_repository/ensemble/1")
        with self.assertRaisesRegex(ValueError, "numeric model version"):
            self.inventory()

    def test_actual_subset_smoke_cannot_qualify_a_full_release(self):
        with self.assertRaises(ValueError):
            self.release.validate_verification({"image_id": "sha256:abc", "model_count": 4}, {
                "image_id": "sha256:abc", "model_set": "all", "passed": True,
                "smoke": {"model_set": "example", "ready_models": 3},
            })

    def run_log_failure(self, failure):
        output = self.root / "release"
        (output / "deploy").mkdir(parents=True)
        (output / "context/profiles").mkdir(parents=True)
        shutil.copyfile(self.root / "profiles/model_sets.json", output / "context/profiles/model_sets.json")
        shutil.copyfile(DEPLOY / "compose.yaml", output / "deploy/compose.yaml")
        if failure == "path":
            (output / "verification.log").mkdir()
        (output / "verification.json").write_text('{"passed": true}')
        args = SimpleNamespace(output=output, model_set="all", timeout=10, model_config="")
        record = {"image_id": "sha256:abc", "image": "example:1", "model_count": 4,
                  "compose_sha256": self.release.sha256(output / "deploy/compose.yaml")}

        def commands(*command, **kwargs):
            if command[:2] == ("docker", "inspect"):
                return json.dumps([{"Image": "sha256:abc", "Mounts": []}])
            if command[:2] == ("docker", "exec"):
                return '{"model_set":"all","ready_models":4}'
            return "container"

        def subprocesses(command, **kwargs):
            if "logs" in command and failure != "path":
                raise failure
            return SimpleNamespace(returncode=0)

        with mock.patch.object(self.release, "load_build", return_value=record), \
                mock.patch.object(self.release, "run", side_effect=commands), \
                mock.patch.object(self.release.subprocess, "run", side_effect=subprocesses) as docker:
            with self.assertRaises((ValueError, OSError, SystemExit, KeyboardInterrupt)):
                self.release.verify(args)
        self.assertTrue(any("down" in call.args[0] for call in docker.call_args_list))
        self.assertFalse(json.loads((output / "verification.json").read_text())["passed"])

    def test_log_collection_failure_still_cleans_up_and_invalidates_previous_success(self):
        self.run_log_failure("path")

    def test_termination_during_logging_invalidates_success(self):
        self.run_log_failure(SystemExit(143))

    def test_interrupt_during_logging_invalidates_success(self):
        self.run_log_failure(KeyboardInterrupt())


@unittest.skipUnless(shutil.which("docker"), "Docker Compose CLI required")
class StandaloneComposeTests(unittest.TestCase):
    def render(self, image="example.invalid/triton:1.0.0-jetson", model_set="sbs"):
        self.assertTrue((DEPLOY / "compose.yaml").is_file(), "release Compose is missing")
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            shutil.copyfile(DEPLOY / "compose.yaml", folder / "compose.yaml")
            (folder / ".env").write_text(
                f"TRITON_IMAGE={image}\nTRITON_MODEL_SET={model_set}\nTRITON_HTTP_PORT=18000\n"
            )
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith(("TRITON_", "COMPOSE_"))}
            return subprocess.run(["docker", "compose", "config", "--format", "json"],
                                  cwd=folder, env=env, text=True, capture_output=True)

    def test_only_two_files_are_needed_and_models_are_not_mounted(self):
        result = self.render()
        self.assertEqual(result.returncode, 0, result.stderr)
        config = json.loads(result.stdout)
        service = config["services"]["triton-server"]
        self.assertNotIn("build", service)
        self.assertIsNone(service.get("command"))
        self.assertEqual(service["environment"]["TRITON_MODEL_SET"], "sbs")
        self.assertEqual(service["environment"]["TRITON_MODEL_CONFIG"], "orin_gpu")
        self.assertEqual(service["platform"], "linux/arm64")
        self.assertEqual(service["ports"][0]["published"], "18000")
        self.assertEqual([(v["type"], v["target"]) for v in service["volumes"]],
                         [("volume", "/var/log/triton")])

    def test_missing_image_fails_before_deployment(self):
        result = self.render(image="")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TRITON_IMAGE", result.stderr)

    def test_empty_model_set_is_not_silently_replaced_with_all(self):
        result = self.render(model_set="")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["services"]["triton-server"]
                         ["environment"]["TRITON_MODEL_SET"], "")


class ReleaseSmokeTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((DEPLOY / "smoke.py").is_file(), "smoke helper is missing")
        spec = importlib.util.spec_from_file_location("release_smoke", DEPLOY / "smoke.py")
        self.smoke = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.smoke)

    def test_image_and_landmark_probes_share_batch_and_coordinate_system(self):
        image = self.smoke.probe_input({"name": "RAW_IMAGE", "datatype": "UINT8", "shape": [-1, -1, -1, 3]}, 4, (384, 640))
        landmarks = self.smoke.probe_input({"name": "LANDMARK", "datatype": "FP32", "shape": [-1, 5, 2]}, 4, (384, 640))
        self.assertEqual(image.shape, (4, 384, 640, 3))
        self.assertEqual(landmarks.shape, (4, 5, 2))
        self.assertTrue((landmarks[:, :, 0] < 640).all())
        self.assertTrue((landmarks[:, :, 1] < 384).all())

    def test_binary_response_requires_finite_values_and_matching_schema(self):
        expected = [{"name": "embedding", "datatype": "FP32", "shape": [-1, 2]}]
        header = {"outputs": [{"name": "embedding", "datatype": "FP32", "shape": [1, 2],
                               "parameters": {"binary_data_size": 8}}]}
        good = np.array([1, 2], dtype=np.float32).tobytes()
        self.smoke.validate_outputs(header, good, expected, 1)
        with self.assertRaisesRegex(ValueError, "finite"):
            self.smoke.validate_outputs(header, np.array([1, np.nan], dtype=np.float32).tobytes(), expected, 1)
        with self.assertRaisesRegex(ValueError, "batch"):
            self.smoke.validate_outputs(header, good, expected, 4)
        with self.assertRaisesRegex(ValueError, "size"):
            self.smoke.validate_outputs(header, good[:4], expected, 1)


if __name__ == "__main__":
    unittest.main()
