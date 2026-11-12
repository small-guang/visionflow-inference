import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test.test_batch_model_repository import (
    _repeated_messages,
    _scalar,
    parse_config,
)

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT / "model_repository"
MANIFEST_PATH = ROOT / "profiles" / "model_sets.json"

EXPECTED_ALIASES = {
    "adaface",
    "clip_image",
    "clip_text",
    "face_keypoint_2d106",
    "face_qa",
    "face_recognition",
    "osnet",
    "sbs",
    "sixdrepnet",
    "tbps_clip_image",
    "tbps_clip_text",
    "yolo26m_pose",
    "yolo26s_seg",
    "yolov8n_face",
}


def direct_dependencies(config):
    scheduling = _repeated_messages(config, "ensemble_scheduling")
    if len(scheduling) != 1:
        raise AssertionError("expected exactly one ensemble_scheduling block")
    return tuple(
        _scalar(step, "model_name")
        for step in _repeated_messages(scheduling[0], "step")
    )


def dependency_closure(configs, model_name):
    ordered = []

    def visit(current):
        config = configs[current]
        if config.get("platform") == ["ensemble"]:
            for child in direct_dependencies(config):
                visit(child)
        if current not in ordered:
            ordered.append(current)

    visit(model_name)
    return tuple(ordered)


class ModelSetManifestContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.configs = {
            path.parent.name: parse_config(path)
            for path in sorted(REPOSITORY.glob("*/config.pbtxt"))
        }

    def test_schema_and_aliases_are_exact(self):
        self.assertEqual(set(self.manifest), {"schema_version", "model_sets"})
        self.assertEqual(self.manifest["schema_version"], 1)
        self.assertEqual(set(self.manifest["model_sets"]), EXPECTED_ALIASES)

    def test_every_alias_is_the_actual_component_first_closure(self):
        covered = set()
        for alias, definition in self.manifest["model_sets"].items():
            with self.subTest(alias=alias):
                self.assertEqual(set(definition), {"entry_point", "models"})
                models = tuple(definition["models"])
                self.assertEqual(len(models), len(set(models)))
                entry_point = definition["entry_point"]
                self.assertEqual(models[-1], entry_point)
                self.assertEqual(models, dependency_closure(self.configs, entry_point))
                covered.update(models)
        self.assertEqual(covered, set(self.configs))


def load_startup_module():
    path = ROOT / "scripts" / "start_triton.py"
    module_name = "start_triton_test"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class StartupLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.startup = load_startup_module()
        cls.manifest = cls.startup.load_manifest(MANIFEST_PATH)
        cls.temp_directory = tempfile.TemporaryDirectory()
        cls.repository = Path(cls.temp_directory.name) / "models"
        cls.repository.mkdir()
        all_models = cls.startup.resolve_model_set("all", cls.manifest)
        for model in all_models:
            model_dir = cls.repository / model
            model_dir.mkdir(parents=True)
            (model_dir / "config.pbtxt").write_text(
                f'name: "{model}"\nmax_batch_size: 4\n', encoding="utf-8"
            )

    @classmethod
    def tearDownClass(cls):
        cls.temp_directory.cleanup()

    def test_command_matrix(self):
        base = ("tritonserver", "--model-repository=/models", "--log-info=true")
        cases = (
            ("all", ()),
            ("face_qa", (
                "--model-control-mode=explicit",
                "--load-model=face_qa_pre", "--load-model=face_qa",
                "--load-model=face_qa_postprocess", "--load-model=face_qa_ensemble",
            )),
        )
        for model_set, injected in cases:
            with self.subTest(model_set=model_set):
                actual = self.startup.build_triton_command(
                    base, model_set, self.manifest, self.repository
                )
                self.assertEqual(actual, (base[0], *injected, *base[1:]))

    def test_multiple_aliases_preserve_manifest_order_and_deduplicate(self):
        models = self.startup.resolve_model_set("face_qa, yolov8n_face", self.manifest)
        self.assertEqual(
            models,
            (
                "face_qa_pre", "face_qa", "face_qa_postprocess",
                "face_qa_ensemble", "yolov8n_face_preprocess",
                "yolov8n_face_onnx", "yolov8n_face_postprocess",
                "yolov8n_face_ensemble",
            ),
        )

    def test_rejects_invalid_selector_values(self):
        invalid_sets = (
            "", "face_qa,", ",face_qa", "face_qa,,osnet", "face_qa,face_qa",
            "unknown", "all,face_qa",
        )
        for value in invalid_sets:
            with self.subTest(value=value), self.assertRaises(
                self.startup.StartupConfigError
            ):
                self.startup.resolve_model_set(value, self.manifest)

    def test_rejects_conflicting_base_arguments(self):
        conflicts = (
            ("tritonserver", "--model-control-mode=explicit"),
            ("tritonserver", "--model-control-mode", "explicit"),
            ("tritonserver", "--load-model=face_qa"),
            ("tritonserver", "--load-model", "face_qa"),
            ("tritonserver", "--model-config-name=throughput"),
            ("tritonserver", "--model-config-name", "throughput"),
        )
        for command in conflicts:
            with self.subTest(command=command), self.assertRaisesRegex(
                self.startup.StartupConfigError, "conflicting Triton argument"
            ):
                self.startup.build_triton_command(
                    command, "all", self.manifest, self.repository
                )

    def _copy_repository(self):
        temporary = tempfile.TemporaryDirectory()
        copied_repository = Path(temporary.name) / "models"
        shutil.copytree(self.repository, copied_repository)
        return temporary, copied_repository

    def test_rejects_missing_selected_repository_paths(self):
        model = "face_qa_pre"
        cases = (
            ("directory", Path(model), "model directory is missing"),
            ("config", Path(model) / "config.pbtxt", "model config is missing"),
        )
        for description, relative_path, message in cases:
            with self.subTest(description=description):
                temporary, repository = self._copy_repository()
                try:
                    target = repository / relative_path
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                    with self.assertRaises(self.startup.StartupConfigError) as raised:
                        self.startup.build_triton_command(
                            ("tritonserver",), "face_qa",
                            self.manifest, repository,
                        )
                    self.assertEqual(str(raised.exception), f"{message}: {target}")
                finally:
                    temporary.cleanup()

    def test_load_manifest_rejects_malformed_data(self):
        valid = {
            "schema_version": 1,
            "model_sets": {
                "face_qa": {
                    "entry_point": "face_qa_ensemble",
                    "models": ["face_qa", "face_qa_ensemble"],
                }
            },
        }
        cases = (
            ("invalid json", "{", None),
            ("schema version", None, {**valid, "schema_version": 2}),
            (
                "reserved alias", None,
                {"schema_version": 1, "model_sets": {"all": valid["model_sets"]["face_qa"]}},
            ),
            (
                "duplicate model", None,
                {"schema_version": 1, "model_sets": {"face_qa": {
                    "entry_point": "face_qa_ensemble",
                    "models": ["face_qa", "face_qa", "face_qa_ensemble"],
                }}},
            ),
            (
                "absent entry point", None,
                {"schema_version": 1, "model_sets": {"face_qa": {
                    "entry_point": "missing", "models": ["face_qa"],
                }}},
            ),
            (
                "unsafe alias", None,
                {"schema_version": 1, "model_sets": {"face qa": valid["model_sets"]["face_qa"]}},
            ),
            (
                "unsafe model name", None,
                {"schema_version": 1, "model_sets": {"face_qa": {
                    "entry_point": "face_qa_ensemble",
                    "models": ["../face_qa", "face_qa_ensemble"],
                }}},
            ),
            (
                "empty model name", None,
                {"schema_version": 1, "model_sets": {"face_qa": {
                    "entry_point": "face_qa_ensemble",
                    "models": ["", "face_qa_ensemble"],
                }}},
            ),
            ("non-object root", None, []),
            (
                "non-object definition", None,
                {"schema_version": 1, "model_sets": {"face_qa": []}},
            ),
            ("non-object model sets", None, {"schema_version": 1, "model_sets": []}),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            for description, text, data in cases:
                with self.subTest(description=description):
                    path.write_text(
                        text if text is not None else json.dumps(data), encoding="utf-8"
                    )
                    with self.assertRaises(self.startup.StartupConfigError):
                        self.startup.load_manifest(path)

    def test_load_manifest_rejects_dot_path_component_names(self):
        cases = (
            ("alias", ".", {".": {
                "entry_point": "face_qa_ensemble",
                "models": ["face_qa", "face_qa_ensemble"],
            }}),
            ("alias", "..", {"..": {
                "entry_point": "face_qa_ensemble",
                "models": ["face_qa", "face_qa_ensemble"],
            }}),
            ("entry point", ".", {"face_qa": {
                "entry_point": ".", "models": ["face_qa", "."],
            }}),
            ("entry point", "..", {"face_qa": {
                "entry_point": "..", "models": ["face_qa", ".."],
            }}),
            ("model", ".", {"face_qa": {
                "entry_point": "face_qa_ensemble",
                "models": [".", "face_qa_ensemble"],
            }}),
            ("model", "..", {"face_qa": {
                "entry_point": "face_qa_ensemble",
                "models": ["..", "face_qa_ensemble"],
            }}),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            for field, name, model_sets in cases:
                with self.subTest(field=field, name=name):
                    path.write_text(json.dumps({
                        "schema_version": 1,
                        "model_sets": model_sets,
                    }), encoding="utf-8")
                    with self.assertRaises(self.startup.StartupConfigError):
                        self.startup.load_manifest(path)

    def test_main_dry_run_ignores_legacy_runtime_selector(self):
        wrapper = Path(self.temp_directory.name) / "rotating_log.py"
        wrapper.write_text("", encoding="utf-8")
        stdout = io.StringIO()
        argv = (
            "--manifest", str(MANIFEST_PATH), "--model-repository", str(self.repository),
            "--log-wrapper", str(wrapper), "--dry-run", "--", "tritonserver",
            "--model-repository=/models",
        )
        environment = {"TRITON_MODEL_SET": "face_qa", "TRITON_RUNTIME_PROFILE": "throughput"}
        with mock.patch.dict(os.environ, environment, clear=True), \
             mock.patch.object(self.startup.os, "execv") as execv, \
             contextlib.redirect_stdout(stdout):
            self.assertEqual(self.startup.main(argv), 0)
        self.assertEqual(
            json.loads(stdout.getvalue()),
            [
                "tritonserver", "--model-control-mode=explicit",
                "--load-model=face_qa_pre", "--load-model=face_qa",
                "--load-model=face_qa_postprocess", "--load-model=face_qa_ensemble",
                "--model-repository=/models",
            ],
        )
        execv.assert_not_called()

    def test_main_exec_handoff_uses_log_wrapper(self):
        wrapper = Path(self.temp_directory.name) / "rotating_log.py"
        wrapper.write_text("", encoding="utf-8")
        argv = (
            "--manifest", str(MANIFEST_PATH), "--model-repository", str(self.repository),
            "--log-wrapper", str(wrapper), "--", "tritonserver", "--log-info=true",
        )
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(self.startup.os, "execv") as execv:
            with self.assertRaisesRegex(AssertionError, "returned unexpectedly"):
                self.startup.main(argv)
        execv.assert_called_once_with(
            sys.executable,
            [sys.executable, str(wrapper), "--", "tritonserver", "--log-info=true"],
        )

    def test_main_rejects_explicit_empty_selector_environment(self):
        wrapper = Path(self.temp_directory.name) / "rotating_log.py"
        wrapper.write_text("", encoding="utf-8")
        argv = (
            "--manifest", str(MANIFEST_PATH), "--model-repository", str(self.repository),
            "--log-wrapper", str(wrapper), "--", "tritonserver",
        )
        cases = (
            (
                {"TRITON_MODEL_SET": "", "TRITON_RUNTIME_PROFILE": "default"},
                "TRITON_MODEL_SET contains an empty alias",
            ),
        )
        for environment, expected_error in cases:
            stderr = io.StringIO()
            with self.subTest(environment=environment), \
                 mock.patch.dict(os.environ, environment, clear=True), \
                 mock.patch.object(self.startup.os, "execv") as execv, \
                 contextlib.redirect_stderr(stderr):
                self.assertEqual(self.startup.main(argv), 2)
                self.assertIn(expected_error, stderr.getvalue())
                execv.assert_not_called()


class Amd64DeploymentContractTests(unittest.TestCase):
    def test_dockerfile_copies_model_set_launcher_and_manifest(self):
        dockerfile = (ROOT / "Dockerfile.triton").read_text(encoding="utf-8")

        for expected in (
            "COPY scripts/start_triton.py /usr/local/bin/start_triton.py",
            "COPY profiles/model_sets.json /opt/triton/model_sets.json",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, dockerfile)

    def test_compose_exposes_model_set_default_and_uses_launcher(self):
        compose = (ROOT / "docker-compose.yaml").read_text(encoding="utf-8")

        for expected in (
            "TRITON_MODEL_SET",
            "${TRITON_MODEL_SET-all}",
            "start_triton.py",
            "--manifest /opt/triton/model_sets.json",
            "--model-repository /models",
            "--log-wrapper /usr/local/bin/run_triton_with_rotating_logs.py",
            "--model-repository=/models",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, compose)
        for forbidden in (
            "${TRITON_MODEL_SET:-all}",
            "TRITON_RUNTIME_PROFILE",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, compose)

    def test_compose_delegates_model_control_flags_to_launcher(self):
        compose = (ROOT / "docker-compose.yaml").read_text(encoding="utf-8")

        for forbidden in (
            "--model-control-mode",
            "--load-model",
            "--model-config-name",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, compose)
