"""Fixed-shape models selected for native TensorRT deployment."""
import hashlib
import importlib.util
import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from test.test_batch_model_repository import parse_config, _scalar, _repeated_messages

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    'clip_image_trt': ('image', (3, 224, 224), 'TYPE_FP32'),
    'clip_text_trt': ('text', (52,), 'TYPE_INT64'),
    'TBPS_clip_image_trt': ('image', (3, 224, 224), 'TYPE_FP32'),
    'TBPS_clip_text_trt': ('text', (77,), 'TYPE_INT64'),
    'sbs_r50_ibn_onnx': ('image', (3, 384, 128), 'TYPE_FP32'),
    'w600k_r50_trt': ('input.1', (3, 112, 112), 'TYPE_FP32'),
    'face_qa': ('input', (3, 112, 112), 'TYPE_FP32'),
    'face_keypoint_2d106': ('data', (3, 192, 192), 'TYPE_FP32'),
}


def builder():
    path = ROOT / 'tools/build_tensorrt_plans.py'
    assert path.is_file(), 'generic plan builder is missing'
    spec = importlib.util.spec_from_file_location('generic_plan_builder', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CandidateTensorRTTests(unittest.TestCase):
    def test_optional_models_can_build_candidates_without_changing_legacy_default_set(self):
        for model, name, shape in (
            ('adaface_ir50_webface4m_onnx', 'input', '3x112x112'),
            ('osnet_ain_x1_0_onnx', 'image', '3x256x128'),
            ('sixdrepnet_onnx', 'image', '3x224x224'),
        ):
            self.assertNotIn(model, self.module.MODELS)
            command = self.module.engine_command('trtexec', self.source, self.destination,
                                                  model=model, max_batch=8)
            self.assertIn(f'--maxShapes={name}:8x{shape}', command)

    def setUp(self):
        self.module = builder()
        temporary = tempfile.TemporaryDirectory(prefix='trt candidate with spaces ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'exports' / 'source graph.onnx'
        self.source.parent.mkdir()
        self.source.write_bytes(b'original ONNX graph')
        self.destination = self.root / 'candidate engines' / 'custom batch 8.plan'
        compiler = mock.patch.object(self.module.subprocess, 'run', side_effect=self.compile_engine)
        compiler.start()
        self.addCleanup(compiler.stop)
        self.output = io.StringIO()
        # Keep routine build logs out of the test runner output.
        self.redirect = redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def compile_engine(self, command, **kwargs):
        candidate = Path(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--saveEngine=')))
        self.assertNotEqual(candidate, self.destination)
        candidate.write_bytes(b'compiled candidate engine')
        kwargs['stdout'].write('successful candidate compilation\n')
        return subprocess.CompletedProcess(command, 0)

    def build(self, **options):
        settings = {'model': 'clip_text_trt', 'max_batch': 8, **options}
        return self.module.build_engine(self.source, self.destination, **settings)

    def artifacts(self):
        return (self.destination,
                self.destination.with_name(self.destination.name + '.json'),
                self.destination.with_name(self.destination.name + '.build.log'))

    def previous_artifacts(self):
        self.destination.parent.mkdir(parents=True)
        previous = dict(zip(self.artifacts(), (b'previous engine', b'previous provenance', b'previous log')))
        for path, data in previous.items():
            path.write_bytes(data)
        return previous

    def assert_preserved(self, previous):
        for path, data in previous.items():
            self.assertEqual(path.read_bytes(), data, str(path))
        self.assertEqual(list(self.destination.parent.glob('.trt-build-*')), [])

    def test_b8_fixed_inputs_keep_names_dimensions_and_default_precisions(self):
        for model, (name, shape, _) in INPUTS.items():
            with self.subTest(model=model):
                command = self.module.engine_command('compiler with spaces', self.source, self.destination,
                                                     model=model, max_batch=8)
                suffix = 'x'.join(map(str, shape))
                self.assertEqual(command[0], 'compiler with spaces')
                self.assertIn(f'--onnx={self.source}', command)
                self.assertIn(f'--saveEngine={self.destination}', command)
                self.assertIn(f'--minShapes={name}:1x{suffix}', command)
                self.assertIn(f'--optShapes={name}:1x{suffix}', command)
                self.assertIn(f'--maxShapes={name}:8x{suffix}', command)
                self.assertIn('--memPoolSize=workspace:1024', command)
                self.assertIn('--skipInference', command)
                fp32 = model in ('face_qa', 'face_keypoint_2d106')
                self.assertEqual('--noTF32' in command, fp32)
                self.assertEqual('--fp16' in command, not fp32)
                self.assertFalse(any(arg.startswith(('--inputIOFormats=', '--outputIOFormats=')) for arg in command))
        self.assertFalse(self.destination.parent.exists())
        self.assertEqual(self.source.read_bytes(), b'original ONNX graph')

    def test_custom_optimum_batch_changes_only_batch_for_fixed_inputs(self):
        command = self.module.engine_command('trtexec', self.source, self.destination,
                                             model='TBPS_clip_text_trt', max_batch=8,
                                             opt_batch=3, workspace_mib=2048, precision='fp32')
        self.assertIn('--minShapes=text:1x77', command)
        self.assertIn('--optShapes=text:3x77', command)
        self.assertIn('--maxShapes=text:8x77', command)
        self.assertIn('--memPoolSize=workspace:2048', command)
        self.assertIn('--noTF32', command)
        self.assertNotIn('--fp16', command)

    def test_yolo_b8_keeps_the_existing_spatial_range(self):
        for model in ('yolov8n_face_onnx', 'yolo26m_pose_onnx', 'yolo26s_seg_onnx'):
            with self.subTest(model=model):
                command = self.module.engine_command('trtexec', self.source, self.destination,
                                                     model=model, max_batch=8, opt_batch=4)
                self.assertIn('--minShapes=images:1x3x192x192', command)
                self.assertIn('--optShapes=images:4x3x384x640', command)
                self.assertIn('--maxShapes=images:8x3x640x640', command)

    def test_invalid_options_are_rejected_without_creating_candidate_artifacts(self):
        invalid = [
            {field: value}
            for field in ('max_batch', 'opt_batch', 'workspace_mib')
            for value in (0, -1, True, False, 1.0, 1.5, '8', None)
        ]
        invalid += [{'opt_batch': 9}, {'model': 'unknown'}, {'precision': 'int8'}, {'precision': 'FP16'}]
        for options in invalid:
            settings = {'model': 'clip_text_trt', 'max_batch': 8, **options}
            for api in ('engine_command', 'build_engine'):
                with self.subTest(api=api, options=options), self.assertRaises(ValueError):
                    if api == 'engine_command':
                        self.module.engine_command('trtexec', self.source, self.destination, **settings)
                    else:
                        self.module.build_engine(self.source, self.destination, **settings)
        self.module.subprocess.run.assert_not_called()
        self.assertFalse(self.destination.parent.exists())

    def test_source_must_be_a_file_distinct_from_destination(self):
        alias = self.source.with_name('same graph hardlink.onnx')
        alias.hardlink_to(self.source)
        cases = ((self.root / 'missing.onnx', self.destination, FileNotFoundError),
                 (self.source.parent, self.destination, FileNotFoundError),
                 (self.source, self.source, ValueError),
                 (self.source, alias, ValueError),
                 (self.source, self.source.parent / '..' / 'exports' / self.source.name, ValueError))
        for source, destination, error in cases:
            for api in ('engine_command', 'build_engine'):
                with self.subTest(api=api, source=source, destination=destination), self.assertRaises(error):
                    options = {'model': 'clip_text_trt', 'max_batch': 8}
                    if api == 'engine_command':
                        self.module.engine_command('trtexec', source, destination, **options)
                    else:
                        self.module.build_engine(source, destination, **options)
        self.module.subprocess.run.assert_not_called()
        self.assertFalse(self.destination.parent.exists())
        self.assertEqual(self.source.read_bytes(), b'original ONNX graph')

    def test_b8_build_publishes_explicit_destination_and_complete_provenance(self):
        # An independent candidate must not change adjacent established plans.
        established = self.source.parent / 'model.plan'
        established.write_bytes(b'established engine')
        target = self.build(opt_batch=3, workspace_mib=2048, precision='fp32', trtexec='custom compiler')
        self.assertEqual(target, self.destination)
        plan, metadata_path, log_path = self.artifacts()
        self.assertEqual(plan.read_bytes(), b'compiled candidate engine')
        self.assertEqual(log_path.read_text(), 'successful candidate compilation\n')
        metadata = json.loads(metadata_path.read_text())
        self.assertEqual(metadata['model'], 'clip_text_trt')
        self.assertEqual(metadata['profile'], 'batch8')
        self.assertEqual(metadata['precision'], 'FP32')
        self.assertEqual(metadata['max_batch'], 8)
        self.assertEqual(metadata['opt_batch'], 3)
        self.assertEqual(metadata['workspace_mib'], 2048)
        self.assertEqual(metadata['source_sha256'], hashlib.sha256(b'original ONNX graph').hexdigest())
        self.assertEqual(metadata['plan_sha256'], hashlib.sha256(b'compiled candidate engine').hexdigest())
        self.assertEqual(metadata['build_log'], log_path.name)
        expected = ['custom compiler', f'--onnx={self.source}', f'--saveEngine={self.destination}',
                    '--noTF32', '--minShapes=text:1x52', '--optShapes=text:3x52',
                    '--maxShapes=text:8x52', '--memPoolSize=workspace:2048', '--skipInference']
        self.assertEqual(metadata['build_command'], expected)
        executed = self.module.subprocess.run.call_args.args[0]
        self.assertEqual([arg for arg in executed if not arg.startswith('--saveEngine=')],
                         [arg for arg in expected if not arg.startswith('--saveEngine=')])
        self.assertEqual(self.source.read_bytes(), b'original ONNX graph')
        self.assertEqual(established.read_bytes(), b'established engine')
        self.assertEqual(set(self.source.parent.iterdir()), {self.source, established})
        self.assertEqual(list(self.destination.parent.glob('.trt-build-*')), [])

    def test_build_defaults_and_optimum_equal_to_maximum_are_recorded(self):
        for model, precision in (('face_qa', 'FP32'), ('face_keypoint_2d106', 'FP32'), ('clip_text_trt', 'FP16')):
            with self.subTest(model=model):
                self.build(model=model, opt_batch=8)
                metadata = json.loads(self.artifacts()[1].read_text())
                self.assertEqual(metadata['precision'], precision)
                self.assertEqual(metadata['opt_batch'], 8)
                self.assertEqual(metadata['workspace_mib'], 1024)
                self.assertEqual(metadata['build_command'][0], 'trtexec')
        self.build(model='face_qa', precision='fp16')
        metadata = json.loads(self.artifacts()[1].read_text())
        self.assertEqual(metadata['precision'], 'FP16')
        self.assertEqual(metadata['opt_batch'], 1)
        self.assertIn('--fp16', metadata['build_command'])
        self.assertNotIn('--noTF32', metadata['build_command'])

    def test_explicit_engine_filename_keeps_its_sidecars_distinct(self):
        for filename in ('candidate batch 8.engine', 'candidate batch 8', 'candidate batch 8.json'):
            with self.subTest(filename=filename):
                self.destination = self.root / 'other candidates' / filename
                self.build()
                plan, metadata, log = self.artifacts()
                self.assertEqual(plan.read_bytes(), b'compiled candidate engine')
                self.assertEqual(json.loads(metadata.read_text())['profile'], 'batch8')
                self.assertEqual(log.read_text(), 'successful candidate compilation\n')

    def test_relative_paths_publish_resolved_destination_and_provenance(self):
        working = self.root / 'working directory'
        working.mkdir()
        previous_directory = Path.cwd()
        try:
            self.module.os.chdir(working)
            target = self.module.build_engine(
                Path('../exports/source graph.onnx'),
                Path('../candidate engines/../candidate engines/custom batch 8.plan'),
                model='clip_text_trt', max_batch=8,
            )
        finally:
            self.module.os.chdir(previous_directory)
        self.assertEqual(target, self.destination)
        self.assertEqual(target.read_bytes(), b'compiled candidate engine')
        command = json.loads(self.artifacts()[1].read_text())['build_command']
        self.assertIn(f'--onnx={self.source}', command)
        self.assertIn(f'--saveEngine={self.destination}', command)

    def test_source_cannot_be_overwritten_by_a_candidate_sidecar(self):
        self.destination.parent.mkdir()
        for artifact in self.artifacts()[1:]:
            with self.subTest(artifact=artifact):
                self.source = artifact
                self.source.write_bytes(b'ONNX under an explicit filename')
                with self.assertRaises(ValueError):
                    self.build()
                self.assertEqual(self.source.read_bytes(), b'ONNX under an explicit filename')
        self.module.subprocess.run.assert_not_called()
        self.assertFalse(self.destination.exists())

    def test_failed_empty_or_interrupted_compilation_preserves_previous_artifacts(self):
        previous = self.previous_artifacts()
        for failure in ('nonzero', 'missing', 'empty', 'launch', 'source_changed'):
            def fail_compile(command, **kwargs):
                candidate = Path(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--saveEngine=')))
                kwargs['stdout'].write(failure)
                if failure == 'launch':
                    raise FileNotFoundError('compiler unavailable')
                if failure != 'missing':
                    candidate.write_bytes(b'' if failure == 'empty' else b'new engine')
                if failure == 'source_changed':
                    self.source.write_bytes(b'modified ONNX graph')
                return subprocess.CompletedProcess(command, 1 if failure == 'nonzero' else 0)
            with self.subTest(failure=failure):
                self.source.write_bytes(b'original ONNX graph')
                with mock.patch.object(self.module.subprocess, 'run', side_effect=fail_compile):
                    with self.assertRaises(FileNotFoundError if failure == 'launch' else RuntimeError):
                        self.build()
                self.assert_preserved(previous)
        attempts = list(self.destination.parent.glob(self.destination.name + '.build-*.log'))
        self.assertEqual(len(attempts), 5)
        self.assertEqual({path.read_text() for path in attempts},
                         {'nonzero', 'missing', 'empty', 'launch', 'source_changed'})

    def test_preparation_failures_preserve_previous_artifacts(self):
        previous = self.previous_artifacts()
        copyfile = self.module.shutil.copyfile
        for stage in ('metadata', 'successful_log', 'backup'):
            def fail_copy(source, destination):
                if ((stage == 'successful_log' and Path(destination).name.endswith('.build.log'))
                        or (stage == 'backup' and Path(destination).name.endswith('.previous'))):
                    raise OSError('cannot prepare artifact')
                return copyfile(source, destination)
            with self.subTest(stage=stage):
                failure = (mock.patch.object(self.module.json, 'dumps', side_effect=OSError('metadata unavailable'))
                           if stage == 'metadata' else
                           mock.patch.object(self.module.shutil, 'copyfile', side_effect=fail_copy))
                with failure, self.assertRaises(OSError):
                    self.build()
                self.assert_preserved(previous)
        self.assertEqual(len(list(self.destination.parent.glob(self.destination.name + '.build-*.log'))), 3)

    def test_each_publication_failure_rolls_back_existing_or_absent_artifacts(self):
        for preexisting in ('absent', 'complete', 'missing_log'):
            for failed_index in range(3):
                with self.subTest(preexisting=preexisting, failed_index=failed_index):
                    self.destination = self.root / f'publish {preexisting} {failed_index}' / 'batch 8.plan'
                    previous = self.previous_artifacts() if preexisting != 'absent' else {}
                    if preexisting == 'missing_log':
                        log = self.artifacts()[2]
                        log.unlink()
                        del previous[log]
                    failed = False
                    replace = self.module.os.replace
                    def fail_once(source, destination):
                        nonlocal failed
                        if Path(destination) == self.artifacts()[failed_index] and not failed:
                            failed = True
                            raise OSError('publication unavailable')
                        return replace(source, destination)
                    with mock.patch.object(self.module.os, 'replace', side_effect=fail_once):
                        with self.assertRaises(OSError):
                            self.build()
                    self.assertTrue(failed)
                    self.assert_preserved(previous)
                    for path in self.artifacts():
                        if path not in previous:
                            self.assertFalse(path.exists(), str(path))


class FixedShapeTensorRTTests(unittest.TestCase):
    def test_deployment_build_publishes_only_batch_engine_and_preserves_old_b1(self):
        module = builder()
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            version = repo / 'clip_text_trt' / '1'
            version.mkdir(parents=True)
            (version / 'model.onnx').write_bytes(b'original ONNX')
            old_b1 = version / 'model.plan'
            old_b1.write_bytes(b'frozen B1 engine')
            def compile_engine(command, **kwargs):
                Path(next(a.split('=', 1)[1] for a in command if a.startswith('--saveEngine='))).write_bytes(b'batch engine')
                return subprocess.CompletedProcess(command, 0)
            with mock.patch.object(module.subprocess, 'run', side_effect=compile_engine):
                target = module.build_plan(repo, 'clip_text_trt', 'trtexec')
            self.assertEqual(target, version / 'model_throughput.plan')
            self.assertEqual(target.read_bytes(), b'batch engine')
            self.assertEqual(old_b1.read_bytes(), b'frozen B1 engine')
            metadata = json.loads(target.with_suffix('.plan.json').read_text())
            self.assertEqual(metadata['profile'], 'throughput')
            self.assertEqual(metadata['source_sha256'], hashlib.sha256(b'original ONNX').hexdigest())
            self.assertEqual(metadata['plan_sha256'], hashlib.sha256(b'batch engine').hexdigest())
            self.assertIn('--optShapes=text:1x52', metadata['build_command'])
            self.assertIn('--maxShapes=text:4x52', metadata['build_command'])

    def test_fixed_dimensions_and_tensor_names_keep_b1_to_b4_range(self):
        module = builder()
        for model, (name, shape, _) in INPUTS.items():
            with self.subTest(model=model):
                command = module.build_command('trtexec', Path('source with spaces.onnx'), Path('output.plan'), model=model)
                suffix = 'x'.join(map(str, shape))
                self.assertIn(f'--minShapes={name}:1x{suffix}', command)
                self.assertIn(f'--optShapes={name}:1x{suffix}', command)
                self.assertIn(f'--maxShapes={name}:4x{suffix}', command)
                self.assertIn('--onnx=source with spaces.onnx', command)
                self.assertIn('--noTF32' if model in ('face_qa', 'face_keypoint_2d106') else '--fp16', command)
                self.assertIn('--skipInference', command)

    def test_default_cli_previews_one_batch_engine_per_model(self):
        module = builder()
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            for model in module.MODELS:
                version = repo / model / '1'
                version.mkdir(parents=True)
                (version / 'model.onnx').write_bytes(b'source')
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(module.main(['--model-repository', str(repo), '--dry-run']), 0)
            commands = output.getvalue().splitlines()
            self.assertEqual(len(commands), 11)
            for command in commands:
                self.assertIn('model_throughput.plan', command)
            self.assertEqual(list(repo.rglob('*.plan*')), [])

    def test_precision_fallback_is_explicit_and_recorded_in_provenance(self):
        module = builder()
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            version = repo / 'clip_text_trt' / '1'
            version.mkdir(parents=True)
            (version / 'model.onnx').write_bytes(b'original ONNX')
            def compile_engine(command, **kwargs):
                self.assertNotIn('--fp16', command)
                self.assertIn('--noTF32', command)
                Path(next(a.split('=', 1)[1] for a in command if a.startswith('--saveEngine='))).write_bytes(b'full precision engine')
                return subprocess.CompletedProcess(command, 0)
            with mock.patch.object(module.subprocess, 'run', side_effect=compile_engine):
                target = module.build_plan(repo, 'clip_text_trt', 'trtexec', precision='fp32')
            metadata = json.loads(target.with_suffix('.plan.json').read_text())
            self.assertEqual(metadata['precision'], 'FP32')
            self.assertIn('--noTF32', metadata['build_command'])

    def test_invalid_build_selection_is_rejected_before_compilation(self):
        module = builder()
        for model, precision in (('unknown', 'fp16'), ('face_qa', 'int8')):
            with self.subTest(model=model, precision=precision):
                with self.assertRaises(ValueError):
                    module.build_command('trtexec', Path('s.onnx'), Path('o.plan'), model=model, precision=precision)

    def test_quality_model_default_records_full_precision_and_allows_explicit_override(self):
        module = builder()
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            version = repo / 'face_qa' / '1'
            version.mkdir(parents=True)
            (version / 'model.onnx').write_bytes(b'original quality graph')
            def compile_engine(command, **kwargs):
                self.assertIn('--noTF32', command)
                self.assertNotIn('--fp16', command)
                Path(next(a.split('=', 1)[1] for a in command if a.startswith('--saveEngine='))).write_bytes(b'quality engine')
                return subprocess.CompletedProcess(command, 0)
            with mock.patch.object(module.subprocess, 'run', side_effect=compile_engine):
                target = module.build_plan(repo, 'face_qa', 'trtexec')
            self.assertEqual(json.loads(target.with_suffix('.plan.json').read_text())['precision'], 'FP32')
        command = module.build_command('trtexec', Path('s.onnx'), Path('o.plan'), model='face_qa', precision='fp16')
        self.assertIn('--fp16', command)
        self.assertNotIn('--noTF32', command)


    def test_repository_routes_fixed_models_to_batch_plans_with_original_tensor_types(self):
        for model, (name, _, dtype) in INPUTS.items():
            with self.subTest(model=model):
                cfg = parse_config(ROOT / 'model_repository' / model / 'config.pbtxt')
                self.assertEqual(_scalar(cfg, 'backend'), 'tensorrt')
                self.assertEqual(_scalar(cfg, 'default_model_filename'), 'model_throughput.plan')
                self.assertEqual(_scalar(cfg, 'max_batch_size'), 4)
                inputs = _repeated_messages(cfg, 'input')
                self.assertEqual(_scalar(inputs[0], 'name'), name)
                self.assertEqual(_scalar(inputs[0], 'data_type'), dtype)
                self.assertTrue(all(_scalar(v, 'data_type') == 'TYPE_FP32' for v in _repeated_messages(cfg, 'output')))


if __name__ == '__main__':
    unittest.main()
