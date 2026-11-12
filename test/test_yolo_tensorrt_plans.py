"""Native plan routing and transactional offline builds."""
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from test.test_batch_model_repository import parse_config, _scalar, _repeated_messages

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("yolov8n_face_onnx", "yolo26m_pose_onnx", "yolo26s_seg_onnx")
SHAPES = ((192, 640), (384, 640), (480, 640), (640, 640), (640, 480), (640, 384), (640, 192))


class YoloNativePlanTests(unittest.TestCase):
    def builder(self):
        path = ROOT / "tools/build_yolo_tensorrt_plans.py"
        self.assertTrue(path.is_file(), "native plan builder is missing")
        spec = importlib.util.spec_from_file_location("plan_builder", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_runtime_routes_to_batch_plans_with_original_tensor_contracts(self):
        for name in MODELS:
            with self.subTest(model=name):
                config = parse_config(ROOT / 'model_repository' / name / 'config.pbtxt')
                self.assertEqual(_scalar(config, 'backend'), 'tensorrt')
                self.assertEqual(_scalar(config, 'default_model_filename'), 'model_throughput.plan')
                self.assertEqual(_scalar(config, 'max_batch_size'), 4)
                self.assertNotIn('optimization', config)
                groups = _repeated_messages(config, 'instance_group')
                self.assertEqual(len(groups), 1)
                self.assertEqual(_scalar(groups[0], 'count'), 1)
                for value in _repeated_messages(config, 'input') + _repeated_messages(config, 'output'):
                    self.assertEqual(_scalar(value, 'data_type'), 'TYPE_FP32')



    def test_builder_accepts_seven_buckets_and_the_full_batch_range(self):
        builder = self.builder()
        command = builder.build_command('trtexec', Path('source.onnx'), Path('target.plan'))
        self.assertIn('--fp16', command)
        self.assertIn('--memPoolSize=workspace:1024', command)
        self.assertIn('--skipInference', command)
        self.assertIn('--onnx=source.onnx', command)
        self.assertIn('--saveEngine=target.plan', command)
        values = {}
        for key in ('min', 'opt', 'max'):
            field = next(a for a in command if a.startswith(f'--{key}Shapes='))
            values[key] = tuple(map(int, field.split('images:')[1].split('x')))
        self.assertEqual(values['min'], (1, 3, 192, 192))
        self.assertEqual(values['opt'], (1, 3, 384, 640))
        self.assertEqual(values['max'], (4, 3, 640, 640))
        for batch in range(1, 5):
            for h, w in SHAPES:
                for minimum, actual, maximum in zip(values['min'], (batch, 3, h, w), values['max']):
                    self.assertLessEqual(minimum, actual)
                    self.assertLessEqual(actual, maximum)

    def test_cli_builds_each_selected_model_once(self):
        builder = self.builder()
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            for model in MODELS:
                version = repo / model / '1'
                version.mkdir(parents=True)
                (version / 'model.onnx').write_bytes(b'onnx source')
            cases = (
                ([], MODELS, 'trtexec', {}),
                (['--models', MODELS[1], '--trtexec', 'custom trtexec'], (MODELS[1],), 'custom trtexec', {}),
                (['--models', MODELS[0], '--precision', 'fp32'], (MODELS[0],), 'trtexec', {'precision': 'fp32'}),
            )
            for options, models, trtexec, precision in cases:
                argv = ['build_yolo_tensorrt_plans.py', '--model-repository', str(repo), *options]
                with self.subTest(options=options), mock.patch.object(sys, 'argv', argv), \
                     mock.patch.object(builder, 'build_plan') as build:
                    self.assertEqual(builder.main(), 0)
                    self.assertEqual(build.call_args_list, [
                        mock.call(repo, model, trtexec, **precision) for model in models
                    ])

    def test_legacy_helpers_reject_non_yolo_models_before_compilation(self):
        builder = self.builder()
        non_yolo_models = (
            'clip_image_trt', 'clip_text_trt', 'TBPS_clip_image_trt', 'TBPS_clip_text_trt',
            'sbs_r50_ibn_onnx', 'w600k_r50_trt', 'face_qa', 'face_keypoint_2d106',
        )
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            for model in non_yolo_models:
                version = repo / model / '1'
                version.mkdir(parents=True)
                (version / 'model.onnx').write_bytes(b'onnx source')

            def compile_engine(command, **kwargs):
                # Even a regression must never launch the real compiler.
                target = Path(next(a.split('=', 1)[1] for a in command if a.startswith('--saveEngine=')))
                target.write_bytes(b'compiled engine')
                return subprocess.CompletedProcess(command, 0)

            with mock.patch.object(builder.subprocess, 'run', side_effect=compile_engine) as compiler:
                for model in (*non_yolo_models, 'unknown'):
                    for helper, extra in (('model_paths', ()), ('build_plan', ('trtexec',))):
                        with self.subTest(model=model, helper=helper):
                            with self.assertRaises(ValueError):
                                getattr(builder, helper)(repo, model, *extra)
                compiler.assert_not_called()
            self.assertEqual(list(repo.rglob('*.plan*')), [])

    def test_legacy_main_dry_run_previews_precision_without_building(self):
        builder = self.builder()
        with tempfile.TemporaryDirectory(prefix='yolo preview ') as directory:
            repo = Path(directory)
            version = repo / MODELS[0] / '1'
            version.mkdir(parents=True)
            (version / 'model.onnx').write_bytes(b'onnx source')
            argv = ['build_yolo_tensorrt_plans.py', '--model-repository', str(repo),
                    '--models', MODELS[0], '--precision', 'fp32', '--dry-run']
            output = io.StringIO()
            with mock.patch.object(sys, 'argv', argv), redirect_stdout(output), \
                 mock.patch.object(builder, 'build_plan') as legacy_build, \
                 mock.patch.object(builder._builder, 'build_plan') as generic_build:
                self.assertEqual(builder.main(), 0)
                legacy_build.assert_not_called()
                generic_build.assert_not_called()
            command = output.getvalue()
            self.assertIn('--noTF32', command)
            self.assertNotIn('--fp16', command)
            self.assertIn('--maxShapes=images:4x3x640x640', command)
            self.assertIn(str(version / 'model.onnx'), command)
            self.assertIn(str(version / 'model_throughput.plan'), command)
            self.assertEqual(list(version.iterdir()), [version / 'model.onnx'])

    def test_legacy_main_preflights_all_sources_before_building(self):
        builder = self.builder()
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            version = repo / MODELS[0] / '1'
            version.mkdir(parents=True)
            (version / 'model.onnx').write_bytes(b'onnx source')
            argv = ['build_yolo_tensorrt_plans.py', '--model-repository', str(repo)]
            with mock.patch.object(sys, 'argv', argv), \
                 mock.patch.object(builder, 'build_plan') as legacy_build, \
                 mock.patch.object(builder._builder, 'build_plan') as generic_build:
                with self.assertRaises(FileNotFoundError):
                    builder.main()
                legacy_build.assert_not_called()
                generic_build.assert_not_called()

    def test_failed_or_empty_build_keeps_the_previously_deployed_plan(self):
        builder = self.builder()
        for code in (1, 0):
            with self.subTest(exit_code=code), tempfile.TemporaryDirectory() as directory:
                repo = Path(directory)
                version = repo / MODELS[0] / '1'
                version.mkdir(parents=True)
                (version / 'model.onnx').write_bytes(b'onnx source')
                (version / 'model_throughput.plan').write_bytes(b'previous valid engine')
                with mock.patch.object(builder.subprocess, 'run', return_value=subprocess.CompletedProcess([], code)):
                    with self.assertRaises(RuntimeError):
                        builder.build_plan(repo, MODELS[0], 'trtexec')
                self.assertEqual((version / 'model_throughput.plan').read_bytes(), b'previous valid engine')
                self.assertEqual(list(version.glob('.trt-build-*')), [])

    def test_successful_build_publishes_engine_and_hashes_its_source(self):
        builder = self.builder()
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            version = repo / MODELS[0] / '1'
            version.mkdir(parents=True)
            (version / 'model.onnx').write_bytes(b'onnx source')
            def compile_engine(command, **kwargs):
                target = Path(next(a.split('=', 1)[1] for a in command if a.startswith('--saveEngine=')))
                target.write_bytes(b'compiled engine')
                return subprocess.CompletedProcess(command, 0)
            with mock.patch.object(builder.subprocess, 'run', side_effect=compile_engine):
                result = builder.build_plan(repo, MODELS[0], 'trtexec')
            target = version / 'model_throughput.plan'
            self.assertEqual(target.read_bytes(), b'compiled engine')
            provenance = json.loads(target.with_suffix('.plan.json').read_text())
            self.assertEqual(provenance['source_sha256'], hashlib.sha256(b'onnx source').hexdigest())
            self.assertEqual(provenance['plan_sha256'], hashlib.sha256(b'compiled engine').hexdigest())
            self.assertEqual(provenance['profile'], 'throughput')
            self.assertEqual(Path(result), target)

    def test_failed_rebuild_preserves_the_deployed_provenance_and_log(self):
        builder = self.builder()
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            version = repo / MODELS[0] / '1'
            version.mkdir(parents=True)
            (version / 'model.onnx').write_bytes(b'onnx source')
            previous = {'model_throughput.plan': b'old engine', 'model_throughput.plan.json': b'old provenance', 'model_throughput.plan.build.log': b'old successful log'}
            for name, data in previous.items():
                (version / name).write_bytes(data)
            def fail_build(command, **kwargs):
                kwargs['stdout'].write('new build failed')
                return subprocess.CompletedProcess(command, 1)
            with mock.patch.object(builder.subprocess, 'run', side_effect=fail_build):
                with self.assertRaises(RuntimeError):
                    builder.build_plan(repo, MODELS[0], 'trtexec')
            for name, data in previous.items():
                self.assertEqual((version / name).read_bytes(), data, name)
            attempts = list(version.glob('model_throughput.plan.build-*.log'))
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0].read_text(), 'new build failed')

    def test_metadata_publication_failure_restores_the_previous_artifact_set(self):
        builder = self.builder()
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            version = repo / MODELS[0] / '1'
            version.mkdir(parents=True)
            (version / 'model.onnx').write_bytes(b'onnx source')
            previous = {'model_throughput.plan': b'old engine', 'model_throughput.plan.json': b'old provenance', 'model_throughput.plan.build.log': b'old successful log'}
            for name, data in previous.items():
                (version / name).write_bytes(data)
            def compile_engine(command, **kwargs):
                Path(next(a.split('=', 1)[1] for a in command if a.startswith('--saveEngine='))).write_bytes(b'new engine')
                kwargs['stdout'].write('new successful build')
                return subprocess.CompletedProcess(command, 0)
            replace = builder.os.replace
            failed = False
            def fail_metadata_once(source, target):
                nonlocal failed
                if Path(target) == version / 'model_throughput.plan.json' and not failed:
                    failed = True
                    raise OSError('metadata temporarily locked')
                return replace(source, target)
            with mock.patch.object(builder.subprocess, 'run', side_effect=compile_engine), mock.patch.object(builder.os, 'replace', side_effect=fail_metadata_once):
                with self.assertRaises(OSError):
                    builder.build_plan(repo, MODELS[0], 'trtexec')
            self.assertTrue(failed)
            for name, data in previous.items():
                self.assertEqual((version / name).read_bytes(), data, name)
            self.assertEqual(list(version.glob('.trt-build-*')), [])


if __name__ == '__main__':
    unittest.main()
