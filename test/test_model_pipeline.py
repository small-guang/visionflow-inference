"""Model lifecycle safeguards use temporary data, never deployed model files."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


def digest(data):
    return hashlib.sha256(data).hexdigest()


class CatalogAndAcquisitionTests(unittest.TestCase):
    def test_current_catalog_preserves_model_closure_and_yolo_buckets(self):
        from tools.model_pipeline.catalog import Catalog
        catalog = Catalog()
        aliases = json.loads((catalog.root/'profiles/model_sets.json').read_text(encoding='utf-8'))['model_sets']
        self.assertEqual(len(catalog.models), 14)
        self.assertEqual(len(catalog.sources), 12)
        self.assertEqual(len(catalog.select(['native'])), 11)
        for alias, closure in aliases.items():
            name, = catalog.select([alias])
            self.assertIn(name, closure['models'])
        expected = [[192,640],[384,640],[480,640],[640,192],[640,384],[640,480],[640,640]]
        for model in catalog.models.values():
            if model['recipe'] == 'yolo':
                self.assertEqual(model['spatial_buckets'], expected)
        self.assertEqual(catalog.models['sbs_r50_ibn_onnx']['export']['num_classes'], 1041)

    def test_default_write_boundary_is_the_repository_containing_tools(self):
        from tools.model_pipeline.catalog import PROJECT_ROOT, relative_name
        self.assertEqual(PROJECT_ROOT, Path(__file__).resolve().parents[1])
        with self.assertRaises(ValueError):
            relative_name('.')

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = b'original model bytes'
        self.catalog_path = self.root/'catalog.json'
        self.catalog_path.write_text(json.dumps({'schema_version': 1, 'sources': {'sample': {
            'filename': 'source.pt', 'local_path': 'tools/models/source.pt',
            'sha256': digest(self.data), 'size': len(self.data), 'urls': []}},
            'models': {'sample_model': {'source': 'sample', 'aliases': ['sample_alias'],
                'recipe': 'sample', 'native_plan': True, 'reference_onnx': 'model_repository/sample_model/1/model.onnx',
                'onnx_sha256': digest(b'onnx'), 'onnx_size': 4}}, 'environments': {}}))

    def catalog(self):
        from tools.model_pipeline.catalog import Catalog
        return Catalog(self.catalog_path, project_root=self.root)

    def test_alias_selection_is_unambiguous_and_rejects_unknowns(self):
        catalog = self.catalog()
        self.assertEqual(catalog.select(['sample_alias', 'sample_model']), ['sample_model'])
        self.assertEqual(catalog.select(['all']), ['sample_model'])
        with self.assertRaises(ValueError):
            catalog.select(['missing'])

    def test_outputs_cannot_overlap_deployment_or_source_directories(self):
        catalog = self.catalog()
        for target in (self.root, self.root/'model_repository/new/model.plan',
                       self.root/'tools/models/new.pt',
                       self.root/'model_repository/../model_repository/new', self.root/'.env'):
            with self.subTest(path=target), self.assertRaises(ValueError):
                catalog.output_path(target)
        self.assertEqual(catalog.output_path(self.root/'artifacts/new'), self.root/'artifacts/new')

    def test_alternate_project_root_cannot_disable_the_tools_live_write_boundary(self):
        from tools.model_pipeline.catalog import PROJECT_ROOT
        with self.assertRaises(ValueError):
            self.catalog().output_path(PROJECT_ROOT/'model_repository/candidate/1/model.plan')

    def test_corrupt_local_source_is_not_loaded_or_cached(self):
        from tools.model_pipeline.acquire import acquire_source
        source = self.root/'bad.pt'
        source.write_bytes(b'corrupt')
        catalog = self.catalog()
        with self.assertRaises(ValueError):
            acquire_source(catalog, 'sample', self.root/'cache', local_file=source, offline=True)
        self.assertFalse(catalog.cached_source(self.root/'cache', 'sample').exists())

    def test_offline_acquisition_copies_verified_source_and_preserves_original(self):
        from tools.model_pipeline.acquire import acquire_source
        source = self.root/'tools/models/source.pt'
        source.parent.mkdir(parents=True)
        source.write_bytes(self.data)
        with mock.patch('urllib.request.urlopen', side_effect=AssertionError('network forbidden')):
            result = acquire_source(self.catalog(), 'sample', self.root/'cache', offline=True)
        self.assertEqual(result.read_bytes(), self.data)
        self.assertEqual(source.read_bytes(), self.data)
        self.assertNotEqual(result, source)

    def test_wrong_download_digest_does_not_replace_a_good_cached_file(self):
        from tools.model_pipeline.acquire import copy_verified
        target = self.root/'target'
        target.write_bytes(self.data)
        bad = self.root/'bad'
        bad.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            copy_verified(bad, target, digest(self.data), len(self.data))
        self.assertEqual(target.read_bytes(), self.data)

    def test_drive_confirmation_requires_google_download_host_and_matching_file_id(self):
        from tools.model_pipeline.acquire import drive_confirmation_url
        page = '<form action="https://drive.usercontent.google.com/download"><input name="id" value="abc"><input name="confirm" value="t"></form>'
        result = drive_confirmation_url(page, 'https://drive.google.com/uc?id=abc&export=download')
        self.assertIn('id=abc', result)
        with self.assertRaises(ValueError):
            drive_confirmation_url(page.replace('google.com', 'example.org'), 'https://drive.google.com/uc?id=abc')
        with self.assertRaises(ValueError):
            drive_confirmation_url(page, 'https://drive.google.com/uc?id=different')

    def test_catalog_rejects_source_path_escape_and_alias_collisions(self):
        from tools.model_pipeline.catalog import Catalog
        raw = json.loads(self.catalog_path.read_text())
        raw['sources']['sample']['filename'] = '../escape.pt'
        self.catalog_path.write_text(json.dumps(raw))
        with self.assertRaises(ValueError):
            Catalog(self.catalog_path, project_root=self.root)
        raw['sources']['sample']['filename'] = 'source.pt'
        raw['models']['another'] = dict(raw['models']['sample_model'])
        self.catalog_path.write_text(json.dumps(raw))
        with self.assertRaises(ValueError):
            Catalog(self.catalog_path, project_root=self.root)

    def test_export_refuses_existing_destination_before_starting_worker(self):
        from tools.model_pipeline.export import export_candidate
        target = self.root/'candidates/sample_model'
        target.mkdir(parents=True)
        with mock.patch('subprocess.run') as run, self.assertRaises(FileExistsError):
            export_candidate(self.catalog(), 'sample_model', self.root/'cache', self.root/'candidates')
        run.assert_not_called()

    def test_failed_export_does_not_publish_candidate_or_touch_source(self):
        from tools.model_pipeline.export import export_candidate
        source = self.root/'tools/models/source.pt'
        source.parent.mkdir(parents=True)
        source.write_bytes(self.data)
        with mock.patch('subprocess.run', return_value=mock.Mock(returncode=1)):
            with self.assertRaises(RuntimeError):
                export_candidate(self.catalog(), 'sample_model', self.root/'cache', self.root/'candidates')
        self.assertFalse((self.root/'candidates/sample_model').exists())
        self.assertEqual(source.read_bytes(), self.data)

    def test_candidate_record_must_match_model_source_and_onnx_before_publication(self):
        from tools.model_pipeline.export import export_candidate
        source = self.root/'tools/models/source.pt'
        source.parent.mkdir(parents=True)
        source.write_bytes(self.data)
        candidate_bytes = b'new candidate model'
        valid = {'schema_version': 1, 'model': 'sample_model',
                 'source_sha256': digest(self.data), 'onnx_sha256': digest(candidate_bytes)}
        cases = [
            ('wrong hash', {**valid, 'onnx_sha256': '0'*64}),
            ('wrong model', {**valid, 'model': 'another_model'}),
            ('wrong source', {**valid, 'source_sha256': '0'*64}),
            ('missing hash', {key: value for key, value in valid.items() if key != 'onnx_sha256'}),
            ('wrong schema', {**valid, 'schema_version': 2}),
            ('invalid JSON', '{'),
        ]
        for index, (name, record) in enumerate(cases):
            output = self.root/f'candidate-{index}'
            def worker(command, **kwargs):
                destination = Path(command[command.index('--output')+1])
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(candidate_bytes)
                destination.with_suffix('.rebuild.json').write_text(
                    record if isinstance(record, str) else json.dumps(record), encoding='utf-8')
                return mock.Mock(returncode=0)
            with self.subTest(case=name), mock.patch('subprocess.run', side_effect=worker):
                with self.assertRaises((ValueError, RuntimeError)):
                    export_candidate(self.catalog(), 'sample_model', self.root/'cache', output)
            self.assertFalse((output/'sample_model').exists())
            self.assertEqual(source.read_bytes(), self.data)

    def test_verified_candidate_publishes_model_and_record_together(self):
        from tools.model_pipeline.export import export_candidate
        source = self.root/'tools/models/source.pt'
        source.parent.mkdir(parents=True)
        source.write_bytes(self.data)
        candidate_bytes = b'new candidate model'
        record = {'schema_version': 1, 'model': 'sample_model',
                  'source_sha256': digest(self.data), 'onnx_sha256': digest(candidate_bytes)}
        def worker(command, **kwargs):
            destination = Path(command[command.index('--output')+1])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(candidate_bytes)
            destination.with_suffix('.rebuild.json').write_text(json.dumps(record), encoding='utf-8')
            return mock.Mock(returncode=0)
        with mock.patch('subprocess.run', side_effect=worker):
            result = export_candidate(self.catalog(), 'sample_model', self.root/'cache', self.root/'candidates')
        self.assertEqual(result.read_bytes(), candidate_bytes)
        self.assertEqual(json.loads(result.with_suffix('.rebuild.json').read_text()), record)
        self.assertEqual({path.name for path in result.parent.iterdir()}, {'model.onnx', 'model.rebuild.json'})
        self.assertEqual(source.read_bytes(), self.data)

    def test_export_creates_private_yolo_config_directory_before_import(self):
        from tools.model_pipeline.export import export_candidate
        source = self.root/'tools/models/source.pt'
        source.parent.mkdir(parents=True)
        source.write_bytes(self.data)
        def fail_worker(command, **kwargs):
            config = Path(kwargs['env']['YOLO_CONFIG_DIR'])
            self.assertTrue(config.is_dir())
            self.assertTrue(config.is_relative_to(self.root/'candidates'))
            return mock.Mock(returncode=1)
        with mock.patch('subprocess.run', side_effect=fail_worker), self.assertRaises(RuntimeError):
            export_candidate(self.catalog(), 'sample_model', self.root/'cache', self.root/'candidates')

    def test_reference_path_is_resolved_in_the_callers_directory_before_worker_launch(self):
        from tools.model_pipeline.export import export_candidate
        source = self.root/'tools/models/source.pt'
        source.parent.mkdir(parents=True)
        source.write_bytes(self.data)
        caller_relative = Path('caller-reference-root')
        expected = (caller_relative/'sample_model/1/model.onnx').resolve()
        def fail_worker(command, **kwargs):
            self.assertEqual(Path(command[command.index('--reference')+1]), expected)
            self.assertEqual(kwargs['cwd'], self.root)
            return mock.Mock(returncode=1)
        with mock.patch('subprocess.run', side_effect=fail_worker), self.assertRaises(RuntimeError):
            export_candidate(self.catalog(), 'sample_model', self.root/'cache', self.root/'candidates',
                             reference_root=caller_relative)

    def test_source_only_export_does_not_require_a_reference_file(self):
        from tools.model_pipeline.export import export_candidate
        source = self.root/'tools/models/source.pt'
        source.parent.mkdir(parents=True)
        source.write_bytes(self.data)
        def fail_worker(command, **kwargs):
            self.assertIn('--no-reference', command)
            self.assertNotIn('--reference', command)
            return mock.Mock(returncode=1)
        with mock.patch('subprocess.run', side_effect=fail_worker), self.assertRaises(RuntimeError):
            export_candidate(self.catalog(), 'sample_model', self.root/'cache', self.root/'candidates',
                             no_reference=True)

    def test_no_reference_cannot_silently_override_an_explicit_reference_root(self):
        from tools.model_pipeline.export import export_candidate
        with mock.patch('subprocess.run') as run, self.assertRaises(ValueError):
            export_candidate(self.catalog(), 'sample_model', self.root/'cache', self.root/'candidates',
                             no_reference=True, reference_root=self.root/'reference')
        run.assert_not_called()


class CandidateParityTests(unittest.TestCase):
    def test_source_only_checks_contract_and_does_not_claim_numerical_parity(self):
        import onnx
        from onnx import TensorProto, helper
        from tools.model_pipeline.export_worker import compare_reference
        spec = {'input_name': 'image', 'input_shape': [2], 'input_dtype': 'FP32',
                'output_names': ['embedding'], 'output_shapes': [[2]], 'recipe': 'sample'}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'candidate.onnx'
            def graph(wrong_shape=False):
                inputs = helper.make_tensor_value_info('image', TensorProto.FLOAT, ['batch', 2])
                outputs = helper.make_tensor_value_info('embedding', TensorProto.FLOAT, ['batch', 4 if wrong_shape else 2])
                node = helper.make_node('Concat', ['image', 'image'], ['embedding'], axis=1) if wrong_shape else helper.make_node('Identity', ['image'], ['embedding'])
                model = helper.make_model(helper.make_graph([node], 'candidate', [inputs], [outputs]),
                                          opset_imports=[helper.make_opsetid('', 17)])
                model.ir_version = 9
                onnx.save(model, path)
            graph()
            rows = compare_reference(spec, path, None, [1, 8])
            self.assertEqual([r['input_shape'][0] for r in rows], [1, 8])
            self.assertTrue(all(r['max_abs_error'] is None and not r['reference_compared'] for r in rows))
            graph(wrong_shape=True)
            with self.assertRaises(ValueError):
                compare_reference(spec, path, None, [8])

    def test_candidate_parity_exercises_batch8_and_rejects_numerical_regression(self):
        import onnx
        from onnx import TensorProto, helper
        from tools.model_pipeline.export_worker import compare_reference
        spec = {'input_name': 'image', 'input_shape': [2], 'input_dtype': 'FP32',
                'output_names': ['embedding'], 'recipe': 'sample'}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def graph(path, op):
                value_in = helper.make_tensor_value_info('image', TensorProto.FLOAT, ['batch', 2])
                value_out = helper.make_tensor_value_info('embedding', TensorProto.FLOAT, ['batch', 2])
                model = helper.make_model(helper.make_graph([helper.make_node(op, ['image'], ['embedding'])],
                      'parity', [value_in], [value_out]), opset_imports=[helper.make_opsetid('', 17)])
                model.ir_version = 9
                onnx.save(model, path)
            graph(root/'reference.onnx', 'Identity')
            graph(root/'candidate.onnx', 'Identity')
            rows = compare_reference(spec, root/'candidate.onnx', root/'reference.onnx', [1, 8])
            self.assertEqual([r['input_shape'][0] for r in rows], [1, 8])
            graph(root/'candidate.onnx', 'Neg')
            with self.assertRaises(AssertionError):
                compare_reference(spec, root/'candidate.onnx', root/'reference.onnx', [8])


if __name__ == '__main__':
    unittest.main()
