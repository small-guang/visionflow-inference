"""Orin-only GPU startup and frozen-release resource contracts."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy" / "jetson"
YOLO = ("yolov8n_face", "yolo26m_pose", "yolo26s_seg")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class OrinStartupTests(unittest.TestCase):
    def setUp(self):
        self.startup = load_module("orin_startup_test", ROOT / "scripts/start_triton.py")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repository = self.root / "models"
        self.definitions = {}
        for alias in (*YOLO, "other"):
            entry = alias + "_ensemble"
            self.definitions[alias] = self.startup.ModelSetDefinition(
                entry, (alias + "_preprocess", alias + "_onnx", alias + "_postprocess", entry)
            )
            for model in self.definitions[alias].models:
                self.write(f"models/{model}/config.pbtxt", 'platform: "ensemble"')
            if alias in YOLO:
                self.write(f"models/{entry}/configs/orin_gpu.pbtxt", 'backend: "python"')
                self.write(f"models/{entry}/1/model.py", "# wrapper")
        self.manifest = self.startup.Manifest(1, self.definitions)

    def write(self, relative, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def command(self, selection="all", profile="orin_gpu"):
        return self.startup.build_triton_command(
            ("tritonserver", "--model-repository=/models"), selection,
            self.manifest, self.repository, model_config_name=profile,
        )

    def add_tbps(self):
        entry = "TBPS_clip_image_ensemble"
        self.definitions["tbps_clip_image"] = self.startup.ModelSetDefinition(
            entry, ("TBPS_clip_image_preprocess", "TBPS_clip_image_trt", entry))
        for model in self.definitions["tbps_clip_image"].models:
            self.write(f"models/{model}/config.pbtxt", 'platform: "ensemble"')
        self.write(f"models/{entry}/configs/orin_gpu.pbtxt", 'backend: "python"')
        self.write(f"models/{entry}/1/model.py", "# TBPS wrapper")
        self.write("models/TBPS_clip_image_preprocess/1/gpu.py", "# TBPS GPU preprocessing")

    def test_tbps_gpu_loads_only_public_wrapper_and_preserves_other_chains(self):
        self.add_tbps()
        loaded = [a for a in self.command("tbps_clip_image,other") if a.startswith("--load-model=")]
        self.assertEqual(loaded, ["--load-model=TBPS_clip_image_ensemble",
                                  *("--load-model=" + m for m in self.definitions["other"].models)])
        self.assertNotIn("--load-model=TBPS_clip_image_preprocess", self.command("all"))
        self.assertNotIn("--load-model=TBPS_clip_image_trt", self.command("all"))

    def test_tbps_cpu_selection_keeps_original_three_node_chain(self):
        self.add_tbps()
        loaded = [a for a in self.command("tbps_clip_image", profile="") if a.startswith("--load-model=")]
        self.assertEqual(loaded, ["--load-model=" + model for model in self.definitions["tbps_clip_image"].models])
        self.assertEqual(self.command("all", profile=""), ("tritonserver", "--model-repository=/models"))

    def test_tbps_gpu_requires_wrapper_config_and_preprocessing_helper(self):
        self.add_tbps()
        resources = ("TBPS_clip_image_ensemble/configs/orin_gpu.pbtxt",
                     "TBPS_clip_image_ensemble/1/model.py", "TBPS_clip_image_preprocess/1/gpu.py")
        for relative in resources:
            path = self.repository / relative
            original = path.read_bytes()
            path.unlink()
            try:
                with self.subTest(resource=relative), self.assertRaisesRegex(
                        self.startup.StartupConfigError, "Orin GPU resource"):
                    self.command("tbps_clip_image")
            finally:
                path.write_bytes(original)

    def test_tbps_gpu_preserves_native_resources_without_loading_native_instance(self):
        self.add_tbps()
        plan = self.write("models/TBPS_clip_image_trt/1/model_throughput.plan", "same native engine")
        original = plan.read_bytes()
        self.command("tbps_clip_image")
        self.assertEqual(plan.read_bytes(), original)
        (self.repository / "TBPS_clip_image_trt/config.pbtxt").unlink()
        with self.assertRaisesRegex(self.startup.StartupConfigError, "model config is missing"):
            self.command("tbps_clip_image")

    def test_image_and_face_fusion_only_changes_gpu_runtime_closure(self):
        for alias, entry in (("clip_image", "clip_ensemble_trt"), ("sbs", "sbs_r50_ibn_image_ensemble"),
                             ("face_keypoint_2d106", "face_keypoint_2d106_ensemble"),
                             ("face_qa", "face_qa_ensemble"),
                             ("face_recognition", "face_recognition_pipeline")):
            chain = (alias + "_preprocess", alias + "_native", entry)
            manifest = self.startup.Manifest(1, {alias: self.startup.ModelSetDefinition(entry, chain)})
            self.assertEqual(self.startup.resolve_runtime_models(alias, manifest, "orin_gpu"), (entry,))
            self.assertEqual(self.startup.resolve_runtime_models(alias, manifest, ""), chain)
            self.assertEqual(manifest.model_sets[alias].models, chain)

    def test_gpu_loads_three_public_wrappers_and_keeps_other_chains(self):
        command = self.command()
        self.assertIn("--model-config-name=orin_gpu", command)
        self.assertIn("--model-control-mode=explicit", command)
        loaded = [arg.split("=", 1)[1] for arg in command if arg.startswith("--load-model=")]
        self.assertEqual(loaded, [*(alias + "_ensemble" for alias in YOLO),
                                  *self.definitions["other"].models])

    def test_subset_has_no_cpu_yolo_instances(self):
        command = self.command("yolo26m_pose")
        self.assertEqual([a for a in command if a.startswith("--load-model=")],
                         ["--load-model=yolo26m_pose_ensemble"])

    def test_shared_dependency_needed_by_another_chain_is_preserved(self):
        original = self.definitions["other"]
        self.definitions["other"] = self.startup.ModelSetDefinition(
            original.entry_point, ("yolov8n_face_onnx", *original.models))
        self.assertIn("--load-model=yolov8n_face_onnx", self.command("yolov8n_face,other"))

    def test_cpu_default_command_stays_automatic(self):
        self.assertEqual(self.command(profile=""),
                         ("tritonserver", "--model-repository=/models"))

    def test_unknown_profile_is_rejected(self):
        for profile in ("gpu", "../orin_gpu", "orin_gpu "):
            with self.subTest(profile=profile), self.assertRaisesRegex(
                    self.startup.StartupConfigError, "TRITON_MODEL_CONFIG"):
                self.command(profile=profile)

    def test_gpu_requires_explicit_alternate_config_and_wrapper(self):
        for relative in ("configs/orin_gpu.pbtxt", "1/model.py"):
            path = self.repository / "yolo26m_pose_ensemble" / relative
            content = path.read_bytes()
            path.unlink()
            try:
                with self.subTest(relative=relative), self.assertRaisesRegex(
                        self.startup.StartupConfigError, "Orin GPU"):
                    self.command("yolo26m_pose")
            finally:
                path.write_bytes(content)

    def test_gpu_still_validates_full_resource_chain(self):
        (self.repository / "yolo26m_pose_onnx/config.pbtxt").unlink()
        with self.assertRaisesRegex(self.startup.StartupConfigError, "model config is missing"):
            self.command("yolo26m_pose")

    def test_environment_drives_real_launcher_dry_run(self):
        manifest = self.write("manifest.json", json.dumps({
            "schema_version": 1,
            "model_sets": {alias: {"entry_point": d.entry_point, "models": d.models}
                           for alias, d in self.definitions.items()},
        }))
        wrapper = self.write("log.py", "")
        stdout = io.StringIO()
        with mock.patch.dict(os.environ, {"TRITON_MODEL_SET": "yolov8n_face",
                                          "TRITON_MODEL_CONFIG": "orin_gpu"}, clear=True), \
                contextlib.redirect_stdout(stdout):
            result = self.startup.main([
                "--manifest", str(manifest), "--model-repository", str(self.repository),
                "--log-wrapper", str(wrapper), "--dry-run", "--", "tritonserver",
            ])
        self.assertEqual(result, 0)
        self.assertIn("--model-config-name=orin_gpu", json.loads(stdout.getvalue()))


class OrinReleaseResourcesTests(unittest.TestCase):
    def test_snapshot_and_inventory_include_shared_gpu_runtime(self):
        release = load_module("orin_release_test", DEPLOY / "release.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = {
                "model_repository/pre/config.pbtxt": 'backend: "python"',
                "model_repository/pre/1/model.py": "# backend",
                "model_repository/entry/config.pbtxt": 'platform: "ensemble"',
                "model_repository/entry/1/model.py": "# GPU wrapper",
                "model_repository/entry/configs/orin_gpu.pbtxt": 'backend: "python"',
                "scripts/yolo_orin_pipeline.py": "# shared runtime",
                "scripts/yolo_request_batching.py": "# batching runtime",
                "scripts/yolo_orin_metrics.py": "# metrics runtime",
                "scripts/tbps_orin_pipeline.py": "# TBPS runtime",
                "scripts/unrelated.py": "# excluded",
                "deploy/jetson/compose.gpu.yaml": 'services: {}',
                "deploy/jetson/release_models.json": json.dumps({"schema_version": 1, "aliases": ["example"]}),
                "profiles/model_sets.json": json.dumps({"schema_version": 1, "model_sets": {
                    "example": {"entry_point": "entry", "models": ["pre", "entry"]}}}),
            }
            for relative, content in inputs.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            context = root / "snapshot"
            release.snapshot(root, context)
            helper = context / "scripts/yolo_orin_pipeline.py"
            self.assertTrue(helper.is_file(), "snapshot omitted same-process GPU runtime")
            self.assertTrue((context / "deploy/jetson/compose.gpu.yaml").is_file())
            for relative in ("model_repository/entry/1/model.py",
                             "model_repository/entry/configs/orin_gpu.pbtxt",
                             "model_repository/pre/1/model.py"):
                self.assertEqual((context / relative).read_bytes(), (root / relative).read_bytes())
            self.assertFalse((context / "scripts/unrelated.py").exists())
            for name in ("yolo_request_batching.py", "yolo_orin_metrics.py"):
                self.assertEqual((context / "scripts" / name).read_bytes(), (root / "scripts" / name).read_bytes())
                self.assertEqual(release.inventory(context)["files"][f"scripts/{name}"],
                                 hashlib.sha256((root / "scripts" / name).read_bytes()).hexdigest())
            result = release.inventory(context)
            self.assertIn("model_repository/entry/1/model.py", result["files"])
            self.assertIn("model_repository/entry/configs/orin_gpu.pbtxt", result["files"])
            self.assertEqual(result["files"]["scripts/yolo_orin_pipeline.py"],
                             hashlib.sha256(helper.read_bytes()).hexdigest())
            self.assertEqual((context / "scripts/tbps_orin_pipeline.py").read_bytes(),
                             (root / "scripts/tbps_orin_pipeline.py").read_bytes())
            self.assertEqual(result["files"]["scripts/tbps_orin_pipeline.py"],
                             hashlib.sha256((root / "scripts/tbps_orin_pipeline.py").read_bytes()).hexdigest())

    def test_runtime_and_release_build_contexts_include_gpu_runtime(self):
        runtime = (ROOT / "Dockerfile.triton.jetson").read_text(encoding="utf-8")
        self.assertIn("COPY scripts/yolo_orin_pipeline.py /usr/local/bin/yolo_orin_pipeline.py", runtime)
        for name in ("yolo_request_batching.py", "yolo_orin_metrics.py"):
            self.assertIn(f"COPY scripts/{name} /usr/local/bin/{name}", runtime)
            self.assertIn(f"!scripts/{name}", (ROOT / "Dockerfile.triton.jetson.dockerignore").read_text())
            self.assertIn(f"COPY scripts/{name} /usr/local/bin/{name}", (DEPLOY / "Dockerfile").read_text())
            self.assertIn(f"!scripts/{name}", (DEPLOY / "Dockerfile.dockerignore").read_text())
        self.assertIn("!scripts/yolo_orin_pipeline.py", (ROOT / "Dockerfile.triton.jetson.dockerignore").read_text())
        self.assertIn("yolo_orin_pipeline.py", (DEPLOY / "Dockerfile").read_text())
        self.assertIn("!scripts/yolo_orin_pipeline.py", (DEPLOY / "Dockerfile.dockerignore").read_text())
        self.assertIn("COPY scripts/tbps_orin_pipeline.py /usr/local/bin/tbps_orin_pipeline.py", runtime)
        self.assertIn("!scripts/tbps_orin_pipeline.py", (ROOT / "Dockerfile.triton.jetson.dockerignore").read_text())
        self.assertIn("COPY scripts/tbps_orin_pipeline.py /usr/local/bin/tbps_orin_pipeline.py",
                      (DEPLOY / "Dockerfile").read_text())
        self.assertIn("!scripts/tbps_orin_pipeline.py", (DEPLOY / "Dockerfile.dockerignore").read_text())

    def test_tensorrt_python_bindings_reuse_platform_native_libraries(self):
        runtime = (ROOT / "Dockerfile.triton.jetson").read_text(encoding="utf-8")
        self.assertIn("ARG TENSORRT_PYTHON_PACKAGE=tensorrt-cu13==10.16.1.11", runtime)
        self.assertIn("ARG TENSORRT_BINDINGS_PACKAGE=tensorrt-cu13-bindings==10.16.1.11", runtime)
        self.assertIn("ARG TENSORRT_PYPI_INDEX=https://repo.huaweicloud.com/repository/pypi/simple", runtime)
        self.assertIn("--index-url ${TENSORRT_PYPI_INDEX}", runtime)
        self.assertIn("--no-deps", runtime)
        self.assertIn("import tensorrt as trt", runtime)
        self.assertNotIn("tensorrt-cu13-libs", runtime)

    def test_release_build_delivers_frozen_overlay_and_all_fused_runtime_files(self):
        release = load_module("orin_build_release_test", DEPLOY / "release.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = {
                "model_repository/native/config.pbtxt": 'backend: "tensorrt"\ndefault_model_filename: "model_throughput.plan"',
                "model_repository/native/1/model_throughput.plan": "engine",
                "model_repository/native/1/model.onnx": "reference",
                "model_repository/pre/config.pbtxt": 'backend: "python"',
                "model_repository/pre/1/model.py": "# helper",
                "model_repository/pre/1/gpu.py": "# CUDA helper",
                "model_repository/entry/config.pbtxt": 'platform: "ensemble"',
                "model_repository/entry/1/model.py": "# fused wrapper",
                "model_repository/entry/configs/orin_gpu.pbtxt": 'backend: "python"',
                "scripts/yolo_orin_pipeline.py": "# shared runtime",
                "scripts/yolo_request_batching.py": "# batching runtime",
                "scripts/yolo_orin_metrics.py": "# metrics runtime",
                "scripts/tbps_orin_pipeline.py": "# TBPS runtime",
                "deploy/jetson/release_models.json": json.dumps({"schema_version": 1, "aliases": ["example"]}),
                "profiles/model_sets.json": json.dumps({"schema_version": 1, "model_sets": {
                    "example": {"entry_point": "entry", "models": ["native", "pre", "entry"]}}}),
            }
            for relative, content in inputs.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            for name in ("Dockerfile", "Dockerfile.dockerignore", "smoke.py", "compose.yaml", "compose.gpu.yaml", ".env.example"):
                shutil.copyfile(DEPLOY / name, root / "deploy/jetson" / name)
            for name in ("start_triton.py", "run_triton_with_rotating_logs.py"):
                shutil.copyfile(ROOT / "scripts" / name, root / "scripts" / name)
            args = SimpleNamespace(output=root / "artifacts/release", image="example.invalid/triton:orin-test",
                                   runtime_image="example.invalid/runtime:orin-test")
            native_runtime = {"Architecture": "arm64", "Os": "linux", "Id": "sha256:runtime",
                              "RootFS": {"Layers": ["base"]}}
            candidate = {"Id": "sha256:candidate", "RootFS": {"Layers": ["base", "models"]}}
            with mock.patch.object(release, "ROOT", root), mock.patch.object(release, "DEPLOY", root / "deploy/jetson"), \
                    mock.patch.object(release.platform, "machine", return_value="aarch64"), \
                    mock.patch.object(release, "inspect_image", side_effect=[native_runtime, candidate]), \
                    mock.patch.object(release, "run", return_value="test"), \
                    mock.patch.object(release.subprocess, "run", return_value=SimpleNamespace(returncode=1)), \
                    contextlib.redirect_stdout(io.StringIO()):
                release.build(args)
            record = json.loads((args.output / "build.json").read_text())
            self.assertEqual(record["default_model_config"], "orin_gpu")
            self.assertEqual(record["default_runtime_model_count"], 3)
            self.assertEqual(record["compose_gpu_sha256"], release.sha256(args.output / "deploy/compose.gpu.yaml"))
            frozen = json.loads((args.output / "release.json").read_text())
            self.assertEqual(frozen["default_model_config"], record["default_model_config"])
            self.assertEqual(frozen["default_runtime_model_count"], record["default_runtime_model_count"])
            for relative in ("model_repository/entry/1/model.py", "model_repository/entry/configs/orin_gpu.pbtxt",
                             "model_repository/pre/1/gpu.py", "model_repository/native/1/model_throughput.plan",
                             "scripts/yolo_orin_pipeline.py", "scripts/tbps_orin_pipeline.py",
                             "scripts/yolo_request_batching.py", "scripts/yolo_orin_metrics.py"):
                self.assertEqual((args.output / "context" / relative).read_bytes(), (root / relative).read_bytes())
                self.assertIn(relative, frozen["files"])
            self.assertFalse((args.output / "context/model_repository/native/1/model.onnx").exists())


class OrinReleaseSmokeTests(unittest.TestCase):
    def test_tbps_gpu_smoke_checks_one_wrapper_and_probes_all_batches(self):
        startup = load_module("tbps_smoke_startup_test", ROOT / "scripts/start_triton.py")
        smoke = load_module("tbps_smoke_test", DEPLOY / "smoke.py")
        entry = "TBPS_clip_image_ensemble"
        manifest = startup.Manifest(1, {"tbps_clip_image": startup.ModelSetDefinition(
            entry, ("TBPS_clip_image_preprocess", "TBPS_clip_image_trt", entry))})
        metadata = {"name": entry, "inputs": [{"name": "IMAGE_BYTES", "datatype": "BYTES", "shape": [-1, 1]}],
                    "outputs": [{"name": "image_feat", "datatype": "FP32", "shape": [-1, 512]}]}
        paths = []

        def request(path):
            paths.append(path)
            return contextlib.nullcontext()

        def get_json(path):
            return ({"backend": "python", "instance_group": [{"kind": "KIND_GPU"}]}
                    if path.endswith("/config") else metadata)

        with mock.patch.dict(sys.modules, {"start_triton": startup}), \
                mock.patch.dict(os.environ, {"TRITON_MODEL_SET": "tbps_clip_image",
                                            "TRITON_MODEL_CONFIG": "orin_gpu"}, clear=True), \
                mock.patch.object(startup, "load_manifest", return_value=manifest), \
                mock.patch.object(smoke, "request", side_effect=request), \
                mock.patch.object(smoke, "get_json", side_effect=get_json), \
                mock.patch.object(smoke, "infer") as infer, \
                mock.patch.object(smoke.packages, "version", return_value="test"), \
                mock.patch.object(Path, "glob", return_value=[]):
            result = smoke.smoke()
        self.assertEqual(paths, ["/v2/health/ready", "/v2/models/TBPS_clip_image_ensemble/ready"])
        self.assertEqual(result["ready_models"], 1)
        self.assertEqual([call.args[2] for call in infer.call_args_list], [1, 2, 3, 4])
        payload = smoke.probe_input(metadata["inputs"][0], 4, (224, 224))
        self.assertEqual(payload.dtype, np.dtype(object))
        self.assertEqual(payload.shape, (4, 1))
        self.assertTrue(all(isinstance(value, bytes) and value.startswith(b"\x89PNG") for value in payload[:, 0]))

    def run_smoke(self, profile, config=None):
        startup = load_module("orin_smoke_startup_test", ROOT / "scripts/start_triton.py")
        smoke = load_module("orin_smoke_test", DEPLOY / "smoke.py")
        entry = "yolo26m_pose_ensemble"
        manifest = startup.Manifest(1, {"yolo26m_pose": startup.ModelSetDefinition(
            entry, ("yolo26m_pose_preprocess", "yolo26m_pose_onnx", "yolo26m_pose_postprocess", entry))})
        paths = []

        def request(path):
            paths.append(path)
            return contextlib.nullcontext()

        def get_json(path):
            if path.endswith("/config"):
                return config or {"backend": "python", "max_batch_size": 4,
                                  "dynamic_batching": {}, "instance_group": [{"kind": "KIND_GPU", "count": 2}]}
            return {"name": entry, "inputs": [], "outputs": []}

        with mock.patch.dict(sys.modules, {"start_triton": startup}), \
                mock.patch.dict(os.environ, {"TRITON_MODEL_SET": "yolo26m_pose",
                                            "TRITON_MODEL_CONFIG": profile}, clear=True), \
                mock.patch.object(startup, "load_manifest", return_value=manifest), \
                mock.patch.object(smoke, "request", side_effect=request), \
                mock.patch.object(smoke, "get_json", side_effect=get_json), \
                mock.patch.object(smoke, "infer") as infer, \
                mock.patch.object(smoke.packages, "version", return_value="test"), \
                mock.patch.object(Path, "glob", return_value=[]):
            result = smoke.smoke()
        return result, paths, infer

    def test_gpu_smoke_checks_only_loaded_wrappers_and_all_shape_batch_probes(self):
        result, paths, infer = self.run_smoke("orin_gpu")
        self.assertEqual(paths, ["/v2/health/ready", "/v2/models/yolo26m_pose_ensemble/ready"])
        self.assertEqual(result["ready_models"], 1)
        self.assertEqual(result["model_config"], "orin_gpu")
        self.assertEqual(infer.call_count, 28)

    def test_cpu_smoke_keeps_full_dependency_checks(self):
        result, paths, _ = self.run_smoke("")
        self.assertEqual(result["ready_models"], 4)
        self.assertIn("/v2/models/yolo26m_pose_onnx/ready", paths)

    def test_gpu_smoke_rejects_a_cpu_instance_or_fallback_ensemble(self):
        for config in ({"platform": "ensemble"},
                       {"backend": "python", "instance_group": [{"kind": "KIND_CPU"}]}):
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, "Orin GPU"):
                self.run_smoke("orin_gpu", config)

    def test_gpu_smoke_rejects_stale_or_waiting_cross_request_batch_configuration(self):
        baseline = {"backend": "python", "max_batch_size": 4, "dynamic_batching": {},
                    "instance_group": [{"kind": "KIND_GPU", "count": 2}]}
        for changes in ({"dynamic_batching": None}, {"max_batch_size": 8},
                        {"dynamic_batching": {"max_queue_delay_microseconds": 1000}},
                        {"dynamic_batching": {"preferred_batch_size": [4]}},
                        {"instance_group": [{"kind": "KIND_GPU", "count": 1}]}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "cross-request batching"):
                self.run_smoke("orin_gpu", {**baseline, **changes})


class OrinReleaseVerifyTests(unittest.TestCase):
    def setUp(self):
        self.release = load_module("orin_verify_release_test", DEPLOY / "release.py")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        (self.output / "deploy").mkdir()
        (self.output / "context/profiles").mkdir(parents=True)
        for name in ("compose.yaml", "compose.gpu.yaml"):
            shutil.copyfile(DEPLOY / name, self.output / "deploy" / name)
        manifest = {"schema_version": 1, "model_sets": {"yolo26m_pose": {
            "entry_point": "yolo26m_pose_ensemble", "models": ["yolo26m_pose_preprocess",
                "yolo26m_pose_onnx", "yolo26m_pose_postprocess", "yolo26m_pose_ensemble"]}}}
        (self.output / "context/profiles/model_sets.json").write_text(json.dumps(manifest))
        self.record = {
            "image_id": "sha256:example", "image": "example.invalid/triton:orin", "model_count": 4,
            "default_model_config": "orin_gpu", "default_runtime_model_count": 1,
            "compose_sha256": self.release.sha256(self.output / "deploy/compose.yaml"),
            "compose_gpu_sha256": self.release.sha256(self.output / "deploy/compose.gpu.yaml"),
        }
        self.calls = []

    def verify(self, profile="orin_gpu", smoke_profile=None, ready_models=None):
        args = SimpleNamespace(output=self.output, model_set="all", timeout=10, model_config=profile)

        def run(*command, **kwargs):
            self.calls.append((command, kwargs))
            if "up" in command:
                self.generated_env = Path(command[command.index("--env-file") + 1]).read_text()
            if command[:2] == ("docker", "inspect"):
                return json.dumps([{"Image": self.record["image_id"], "Mounts": []}])
            if command[:2] == ("docker", "exec"):
                return json.dumps({"model_set": "all", "model_config": profile if smoke_profile is None else smoke_profile,
                                   "ready_models": (1 if profile else 4) if ready_models is None else ready_models})
            return "container"

        with mock.patch.object(self.release, "load_build", return_value=self.record), \
                mock.patch.object(self.release, "run", side_effect=run), \
                mock.patch.object(self.release.subprocess, "run", return_value=SimpleNamespace(returncode=0)), \
                mock.patch.dict(os.environ, {"TRITON_MODEL_CONFIG": "orin_gpu"}), \
                contextlib.redirect_stdout(io.StringIO()):
            self.release.verify(args)

    def test_gpu_verification_uses_delivered_base_and_runtime_coverage(self):
        original_cpu_report = self.output / "verification.json"
        original_cpu_report.write_text('{"cpu": "preserved"}')
        self.verify("orin_gpu")
        report = json.loads((self.output / "verification-orin_gpu.json").read_text())
        self.assertTrue(report["passed"])
        self.assertEqual(report["model_config"], "orin_gpu")
        self.assertEqual(report["smoke"]["ready_models"], 1)
        self.assertEqual(json.loads(original_cpu_report.read_text()), {"cpu": "preserved"})
        up, options = next((c, k) for c, k in self.calls if "up" in c)
        self.assertFalse(any(Path(arg).name == "compose.gpu.yaml" for arg in up))
        self.assertIn("\nTRITON_MODEL_CONFIG=orin_gpu\n", self.generated_env)
        self.assertFalse(any(key.startswith("TRITON_") for key in options["env"]))

    def test_explicit_cpu_verification_ignores_inherited_gpu_environment(self):
        self.verify("")
        report = json.loads((self.output / "verification.json").read_text())
        self.assertTrue(report["passed"])
        self.assertEqual(report["model_config"], "")
        self.assertEqual(report["smoke"]["ready_models"], 4)
        up = next(c for c, _ in self.calls if "up" in c)
        self.assertFalse(any(Path(arg).name == "compose.gpu.yaml" for arg in up))
        self.assertIn("\nTRITON_MODEL_CONFIG=\n", self.generated_env)

    def test_gpu_verification_rejects_wrong_profile_or_cpu_closure_count(self):
        for profile, count in (("", 1), ("orin_gpu", 4)):
            with self.subTest(smoke_profile=profile, count=count), self.assertRaisesRegex(ValueError, "coverage"):
                self.verify("orin_gpu", smoke_profile=profile, ready_models=count)
        self.assertFalse(json.loads((self.output / "verification-orin_gpu.json").read_text())["passed"])

    def test_gpu_verification_rejects_unfrozen_overlay_before_starting(self):
        (self.output / "deploy/compose.gpu.yaml").write_text("services: {}")
        with self.assertRaisesRegex(ValueError, "GPU Compose"):
            self.verify("orin_gpu")
        self.assertFalse(self.calls)

    def test_legacy_candidate_without_default_gpu_contract_is_rejected(self):
        legacy_record = {"image_id": self.record["image_id"], "model_count": 4}
        for profile, smoke_profile in (("orin_gpu", "orin_gpu"), ("", "orin_gpu")):
            report = {"image_id": self.record["image_id"], "model_set": "all", "passed": True,
                      "model_config": profile, "smoke": {"model_set": "all", "ready_models": 4,
                                                           "model_config": smoke_profile}}
            with self.subTest(profile=profile, smoke_profile=smoke_profile), self.assertRaisesRegex(ValueError, "default"):
                self.release.validate_verification(legacy_record, report)

    def test_publish_requires_matching_default_gpu_profile_and_runtime_count(self):
        report = {"image_id": self.record["image_id"], "model_set": "all", "passed": True,
                  "model_config": "orin_gpu", "smoke": {"model_set": "all", "ready_models": 1,
                                                         "model_config": "orin_gpu"}}
        self.release.validate_verification(self.record, report)
        for profile, smoke_profile, count in (("", "", 4), ("orin_gpu", "", 1),
                                             ("", "orin_gpu", 1), ("orin_gpu", "orin_gpu", 4)):
            invalid = {**report, "model_config": profile,
                       "smoke": {**report["smoke"], "model_config": smoke_profile, "ready_models": count}}
            with self.subTest(profile=profile, smoke_profile=smoke_profile, count=count), self.assertRaises(ValueError):
                self.release.validate_verification(self.record, invalid)

    def test_cli_defaults_to_delivered_gpu_profile_and_accepts_explicit_cpu(self):
        for options, expected in (([], "orin_gpu"), (["--model-config", ""], "")):
            with self.subTest(options=options), \
                    mock.patch.object(sys, "argv", ["release.py", "verify", "--output", str(self.output), *options]), \
                    mock.patch.object(self.release, "verify") as verify:
                self.assertEqual(self.release.main(), 0)
            self.assertEqual(verify.call_args.args[0].model_config, expected)

    def test_cli_requires_explicit_gpu_profile_argument(self):
        with mock.patch.object(sys, "argv", ["release.py", "verify", "--output", str(self.output),
                                              "--model-config", "orin_gpu"]), \
                mock.patch.object(self.release, "verify") as verify:
            self.assertEqual(self.release.main(), 0)
        self.assertEqual(verify.call_args.args[0].model_config, "orin_gpu")

    def test_publication_rejects_modified_gpu_delivery_overlay(self):
        (self.output / "deploy/compose.gpu.yaml").write_text("services: {}")
        with mock.patch.object(self.release, "load_build", return_value=self.record), \
                mock.patch.object(self.release, "run") as run, \
                self.assertRaisesRegex(ValueError, "GPU Compose"):
            self.release.publish(SimpleNamespace(output=self.output))
        run.assert_not_called()


@unittest.skipUnless(shutil.which("docker"), "Docker Compose CLI required")
class OrinGpuComposeTests(unittest.TestCase):
    def render(self, base, overlay=None, **extra):
        if overlay:
            self.assertTrue(overlay.is_file(), "Orin GPU overlay missing")
        env = {k: v for k, v in os.environ.items() if not k.startswith(("TRITON_", "COMPOSE_"))}
        env.update(TRITON_IMAGE="example.invalid/triton:orin-test", TRITON_MODEL_SET="yolo26m_pose", **extra)
        command = ["docker", "compose", "--env-file", os.devnull, "-f", str(base)]
        if overlay:
            command += ["-f", str(overlay)]
        result = subprocess.run([*command, "config", "--format", "json"], cwd=ROOT, env=env,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)["services"]["triton-server"]

    def test_jetson_base_defaults_to_gpu_and_preserves_explicit_cpu_profile(self):
        for base in (ROOT / "docker-compose.jetson.yaml", DEPLOY / "compose.yaml"):
            with self.subTest(base=base):
                self.assertEqual(self.render(base)["environment"]["TRITON_MODEL_CONFIG"], "orin_gpu")
                self.assertEqual(self.render(base, TRITON_MODEL_CONFIG="")["environment"]["TRITON_MODEL_CONFIG"], "")

    def test_development_overlay_uses_existing_command_and_model_selector(self):
        service = self.render(ROOT / "docker-compose.jetson.yaml", ROOT / "docker-compose.jetson.gpu.yaml")
        self.assertEqual(service["environment"]["TRITON_MODEL_CONFIG"], "orin_gpu")
        self.assertEqual(service["environment"]["TRITON_MODEL_SET"], "yolo26m_pose")
        self.assertEqual(service["runtime"], "nvidia")
        self.assertIn("start_triton.py", " ".join(service["command"]))

    def test_release_overlay_needs_no_source_mount_or_command_override(self):
        service = self.render(DEPLOY / "compose.yaml", DEPLOY / "compose.gpu.yaml")
        self.assertEqual(service["environment"]["TRITON_MODEL_CONFIG"], "orin_gpu")
        self.assertEqual(service["environment"]["TRITON_MODEL_SET"], "yolo26m_pose")
        self.assertIsNone(service.get("command"))
        self.assertEqual([(v["type"], v["target"]) for v in service["volumes"]],
                         [("volume", "/var/log/triton")])


if __name__ == "__main__":
    unittest.main()
