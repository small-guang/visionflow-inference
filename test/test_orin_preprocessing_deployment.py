"""Keep GPU preprocessing enabled and packaged through the Orin release path."""
import hashlib
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
GPU_HELPERS = (
    'image_orin_preprocess.py', 'face_orin_preprocess.py', 'orin_trt_runtime.py',
    'orin_pipeline_metrics.py', 'image_decode.py', 'image_orin_pipeline.py', 'face_orin_pipeline.py',
)


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class OrinPreprocessingDeploymentTests(unittest.TestCase):
    def test_startup_requires_explicit_preprocessor_profile(self):
        startup = load('pre_startup_test', 'scripts/start_triton.py')
        for name in startup.ORIN_GPU_PREPROCESSORS:
            with self.subTest(model=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model = root / name
                (model / '1').mkdir(parents=True)
                (model / 'config.pbtxt').write_text('backend: "python"')
                (model / '1/model.py').write_text('# model')
                manifest = startup.Manifest(1, {'test': startup.ModelSetDefinition(name, (name,))})
                args = (('tritonserver',), 'test', manifest, root)
                with self.assertRaisesRegex(startup.StartupConfigError, 'preprocessing resource'):
                    startup.build_triton_command(*args, model_config_name='orin_gpu')
                (model / 'configs').mkdir()
                (model / 'configs/orin_gpu.pbtxt').write_text('backend: "python"')
                self.assertIn('--model-config-name=orin_gpu',
                              startup.build_triton_command(*args, model_config_name='orin_gpu'))
                (model / 'configs/orin_gpu.pbtxt').unlink()
                self.assertNotIn('--model-config-name=orin_gpu', startup.build_triton_command(*args))

    def test_release_freezes_both_gpu_helpers_and_hashes_their_bytes(self):
        release = load('pre_release_test', 'deploy/jetson/release.py')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {
                'model_repository/pre/config.pbtxt': 'backend: "python"',
                'model_repository/pre/1/model.py': '# backend',
                'profiles/model_sets.json': json.dumps({'schema_version': 1, 'model_sets': {
                    'test': {'entry_point': 'pre', 'models': ['pre']}}}),
                'deploy/jetson/release_models.json': json.dumps({'schema_version': 1, 'aliases': ['test']}),
            }
            for name in GPU_HELPERS:
                files['scripts/' + name] = '# ' + name
            files['scripts/unrelated.py'] = '# excluded'
            for relative, content in files.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            context = root / 'snapshot'
            release.snapshot(root, context)
            recorded = release.inventory(context)['files']
            for name in GPU_HELPERS:
                relative = 'scripts/' + name
                self.assertEqual((context / relative).read_bytes(), (root / relative).read_bytes())
                self.assertEqual(recorded[relative], hashlib.sha256((root / relative).read_bytes()).hexdigest())
            self.assertFalse((context / 'scripts/unrelated.py').exists())

    def test_docker_contexts_deliver_both_helpers(self):
        for dockerfile in ('Dockerfile.triton.jetson', 'deploy/jetson/Dockerfile'):
            for name in GPU_HELPERS:
                self.assertIn(f'COPY scripts/{name} /usr/local/bin/{name}', (ROOT / dockerfile).read_text())
                self.assertIn('!scripts/' + name, (ROOT / (dockerfile + '.dockerignore')).read_text())

    def test_smoke_rejects_kind_gpu_without_actual_cuda_backend(self):
        smoke = load('pre_smoke_test', 'deploy/jetson/smoke.py')
        good = {'backend': 'python', 'max_batch_size': 4,
                'parameters': {'preprocess_backend': {'string_value': 'orin_gpu'}},
                'instance_group': [{'kind': 'KIND_GPU', 'count': 1}]}
        smoke.validate_gpu_preprocessing('face_rec_pre', good)
        for change in ({'parameters': {}}, {'parameters': {'preprocess_backend': {'string_value': 'cpu'}}},
                       {'instance_group': [{'kind': 'KIND_CPU'}]}, {'instance_group': []},
                       {'max_batch_size': 8}, {'backend': 'onnxruntime'}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'not enabled'):
                smoke.validate_gpu_preprocessing('face_rec_pre', {**good, **change})

    def test_fused_startup_requires_wrapper_and_profile_but_preserves_plan_resource(self):
        startup = load('fused_startup_test', 'scripts/start_triton.py')
        for name in startup.ORIN_FUSED_PREPROCESSING_ENTRY_POINTS:
            with self.subTest(model=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for model in (name, 'native'):
                    (root / model / '1').mkdir(parents=True)
                    (root / model / 'config.pbtxt').write_text('backend: "python"')
                plan = root / 'native/1/model_throughput.plan'
                plan.write_bytes(b'original plan')
                wrapper = root / name / '1/model.py'
                profile = root / name / 'configs/orin_gpu.pbtxt'
                wrapper.write_text('# wrapper')
                profile.parent.mkdir()
                profile.write_text('backend: "python"')
                manifest = startup.Manifest(1, {'test': startup.ModelSetDefinition(name, ('native', name))})
                command = startup.build_triton_command(('tritonserver',), 'test', manifest, root,
                                                      model_config_name='orin_gpu')
                self.assertEqual([arg for arg in command if arg.startswith('--load-model=')],
                                 ['--load-model=' + name])
                self.assertEqual(plan.read_bytes(), b'original plan')
                for required in (wrapper, profile, root / 'native/config.pbtxt'):
                    content = required.read_bytes()
                    required.unlink()
                    with self.assertRaises(startup.StartupConfigError):
                        startup.build_triton_command(('tritonserver',), 'test', manifest, root,
                                                     model_config_name='orin_gpu')
                    required.write_bytes(content)

    def test_smoke_rejects_nonfused_backend_or_missing_explicit_gpu_parameter(self):
        smoke = load('fused_validation_test', 'deploy/jetson/smoke.py')
        good = {'backend': 'python', 'max_batch_size': 4,
                'parameters': {'pipeline_backend': {'string_value': 'orin_gpu'}},
                'instance_group': [{'kind': 'KIND_GPU', 'count': 1}]}
        smoke.validate_fused_pipeline('face_recognition_pipeline', good)
        for change in ({'parameters': {}}, {'parameters': {'pipeline_backend': {'string_value': 'cpu'}}},
                       {'instance_group': [{'kind': 'KIND_CPU'}]}, {'instance_group': []},
                       {'max_batch_size': 8}, {'backend': 'tensorrt'},
                       {'backend': '', 'platform': 'ensemble'}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'not enabled'):
                smoke.validate_fused_pipeline('face_recognition_pipeline', {**good, **change})

    def test_fused_smoke_checks_only_selected_public_entries_and_all_batch_sizes(self):
        startup = load('fused_smoke_startup_test', 'scripts/start_triton.py')
        smoke = load('fused_smoke_test', 'deploy/jetson/smoke.py')
        definitions = {name: startup.ModelSetDefinition(name, (name + '_native', name))
                       for name in sorted(startup.ORIN_FUSED_PREPROCESSING_ENTRY_POINTS)}
        manifest = startup.Manifest(1, definitions)
        paths = []
        def request(path):
            paths.append(path)
            return contextlib.nullcontext()
        def get_json(path):
            if path.endswith('/config'):
                return {'backend': 'python', 'max_batch_size': 4,
                        'parameters': {'pipeline_backend': {'string_value': 'orin_gpu'}},
                        'instance_group': [{'kind': 'KIND_GPU', 'count': 1}]}
            return {'inputs': [], 'outputs': []}
        with mock.patch.dict(sys.modules, {'start_triton': startup}), \
                mock.patch.dict(os.environ, {'TRITON_MODEL_SET': 'all', 'TRITON_MODEL_CONFIG': 'orin_gpu'}, clear=True), \
                mock.patch.object(startup, 'load_manifest', return_value=manifest), \
                mock.patch.object(smoke, 'request', side_effect=request), \
                mock.patch.object(smoke, 'get_json', side_effect=get_json), \
                mock.patch.object(smoke, 'infer') as infer, \
                mock.patch.object(smoke.packages, 'version', return_value='test'), \
                mock.patch.object(Path, 'glob', return_value=[]):
            result = smoke.smoke()
        self.assertEqual(paths, ['/v2/health/ready', *[f'/v2/models/{name}/ready' for name in definitions]])
        self.assertEqual(result['ready_models'], 5)
        self.assertEqual([(call.args[0], call.args[2]) for call in infer.call_args_list],
                         [(name, batch) for name in definitions for batch in (1, 2, 3, 4)])


if __name__ == '__main__':
    unittest.main()
