"""Private CPU export worker; all destinations are checked against the live tree."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import shutil
import tempfile

from .catalog import Catalog, sha256, verify_file, write_json


def export_graph(spec, source, checkpoint, output):
    recipe = spec['recipe']
    if recipe == 'dynamic_onnx':
        from tools.dynamic_batch_onnx import publish_dynamic_batch_model
        publish_dynamic_batch_model(checkpoint, output, input_name=spec['input_name'],
                                    output_names=tuple(spec['output_names']),
                                    sample_shape=tuple(spec['input_shape']),
                                    output_sample_shapes=tuple(tuple(s) for s in spec['output_shapes']))
        return
    import torch
    torch.set_num_threads(4)
    if recipe.startswith('chinese_clip_'):
        from cn_clip.clip import load_from_name, tokenize
        from tools import export_chinese_clip_onnx as impl
        model, _ = load_from_name(str(checkpoint), device='cpu', vision_model_name='ViT-B-16',
                                  text_model_name='RoBERTa-wwm-ext-base-chinese', input_resolution=224)
        model = model.float().eval()
        if recipe.endswith('_image'):
            wrapper, example = impl.ImageEncoder(model).eval(), impl._make_image_export_example()
        else:
            wrapper, example = impl.TextEncoder(model).eval(), tokenize(['测试'], context_length=52).long()
        impl.export_one(wrapper, example, output, spec['input_name'], spec['output_names'][0])
    elif recipe.startswith('openai_clip_'):
        import clip
        from tools import export_tbps_clip_onnx as impl
        model, _ = clip.load(str(checkpoint), device='cpu', jit=False)
        model = model.float().eval()
        if recipe.endswith('_image'):
            wrapper, example = impl.NormalizedImageEncoder(model).eval(), impl._make_image_export_example()
        else:
            wrapper, example = impl.NormalizedTextEncoder(model).eval(), torch.zeros((1, 77), dtype=torch.long)
            example[0, :2] = torch.tensor([49406, 49407])
        impl.export_one(wrapper, example, output, spec['input_name'], spec['output_names'][0])
    elif recipe in ('adaface', 'osnet', 'sixdrepnet'):
        if recipe == 'adaface':
            from tools import export_adaface_ir50_webface4m_onnx as impl
            state = impl.load_checkpoint(checkpoint, source['size'], source['sha256'])
        elif recipe == 'osnet':
            from tools import export_osnet_ain_onnx as impl
            state = impl.load_checkpoint(checkpoint, source['sha256'])
        else:
            from tools import export_6drepnet_onnx as impl
            state = impl.load_checkpoint(checkpoint, source['size'], source['sha256'])
        impl.export_release(impl.build_model(state), output)
    elif recipe == 'sbs':
        from tools import export_sbs_r50_ibn_onnx as impl
        classes = spec['export']['num_classes']
        state = impl.remap_legacy_head_keys(impl.load_checkpoint(checkpoint, classes))
        model = impl.build_sbs_model(classes)
        model.load_state_dict(state, strict=True)
        wrapper = impl.NormalizedSBS(model).eval()
        torch.manual_seed(2026)
        example = torch.randn(1, 3, 384, 128)
        impl.validate_torch_output(wrapper, example)
        with torch.no_grad():
            torch.onnx.export(wrapper, example, str(output), input_names=['image'], output_names=['embedding'],
                              dynamic_axes={'image': {0: 'batch'}, 'embedding': {0: 'batch'}},
                              opset_version=17, do_constant_folding=True)
        impl.validate_onnx(output, wrapper, (example, torch.randn(2, 3, 384, 128)))
    elif recipe == 'yolo':
        from ultralytics import YOLO
        expected = spec['export']['ultralytics']
        actual = importlib.metadata.version('ultralytics')
        if actual != expected:
            raise RuntimeError(f'YOLO rebuild requires ultralytics=={expected}; got {actual}')
        # Ultralytics writes alongside its checkpoint. Never export beside a source/cache file.
        with tempfile.TemporaryDirectory(prefix='.yolo-', dir=output.parent) as directory:
            private = Path(directory)/checkpoint.name
            shutil.copyfile(checkpoint, private)
            generated = Path(YOLO(str(private)).export(format='onnx', device='cpu',
                        imgsz=spec['export']['imgsz'], dynamic=True, opset=spec['export']['opset'],
                        simplify=spec['export']['simplify'], batch=1))
            shutil.copyfile(generated, output)
    else:
        raise ValueError(f'Unknown export recipe: {recipe}')


def compare_reference(spec, candidate, reference, batches):
    import numpy as np
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(str(candidate), full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    paths = (reference, candidate) if reference is not None else (candidate,)
    sessions = [ort.InferenceSession(str(p), sess_options=options, providers=['CPUExecutionProvider'])
                for p in paths]
    expected_type = {'FP32': 'tensor(float)', 'INT64': 'tensor(int64)'}[spec['input_dtype']]
    for session in sessions:
        inputs = session.get_inputs()
        if len(inputs) != 1 or inputs[0].name != spec['input_name'] or isinstance(inputs[0].shape[0], int):
            raise ValueError('Unexpected input contract or fixed batch')
        if [o.name for o in session.get_outputs()] != spec['output_names']:
            raise ValueError('Unexpected output names')
        if inputs[0].type != expected_type or len(inputs[0].shape) != len(spec['input_shape'])+1:
            raise ValueError('Unexpected input dtype or rank')
    if reference is None and len(spec.get('output_shapes', [])) != len(spec['output_names']):
        raise ValueError('Source-only validation requires declared output shapes')
    rows = []
    for batch in batches:
        if type(batch) is not int or batch < 1:
            raise ValueError('Check batches must be positive integers')
        rng = np.random.default_rng(20260910+batch)
        shape = (batch, *spec['input_shape'])
        if spec['input_dtype'] == 'INT64':
            feed = rng.integers(1, 100, shape, dtype=np.int64)
            if spec['recipe'] == 'openai_clip_text':
                feed[:, 0], feed[:, -1] = 49406, 49407
        elif spec.get('input_range') == '0_255':
            feed = rng.uniform(0, 255, shape).astype(np.float32)
        else:
            feed = rng.normal(0, .5, shape).astype(np.float32)
        outputs = [session.run(None, {spec['input_name']: feed}) for session in sessions]
        for index, (name, actual) in enumerate(zip(spec['output_names'], outputs[-1])):
            if actual.dtype != np.float32 or not np.isfinite(actual).all() or actual.shape[0] != batch:
                raise ValueError(f'Invalid rebuilt output: {name}')
            if spec.get('output_shapes'):
                declared = spec['output_shapes'][index]
                if (len(actual.shape) != len(declared)+1
                        or any(type(want) is int and got != want for got, want in zip(actual.shape[1:], declared))):
                    raise ValueError(f'Unexpected output shape: {name}: {actual.shape}')
            error = None
            if reference is not None:
                expected = outputs[0][index]
                if actual.shape != expected.shape or actual.dtype != expected.dtype:
                    raise ValueError(f'Output contract changed: {name}')
                np.testing.assert_allclose(actual, expected, rtol=spec.get('parity_rtol', 1e-4),
                                           atol=spec.get('parity_atol', 1e-4), err_msg=name)
                error = float(np.max(np.abs(actual.astype(np.float64)-expected)))
            rows.append({'input_shape': list(shape), 'output': name, 'output_shape': list(actual.shape),
                         'max_abs_error': error, 'reference_compared': reference is not None})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('catalog', 'project-root', 'model', 'checkpoint', 'output'):
        parser.add_argument('--'+name, required=True)
    reference_options = parser.add_mutually_exclusive_group(required=True)
    reference_options.add_argument('--reference')
    reference_options.add_argument('--no-reference', action='store_true')
    parser.add_argument('--check-batches', nargs='+', type=int, default=[1, 2])
    args = parser.parse_args(argv)
    catalog = Catalog(args.catalog, project_root=args.project_root)
    spec = catalog.models[args.model]
    source = catalog.sources[spec['source']]
    output = catalog.output_path(args.output)
    checkpoint = verify_file(Path(args.checkpoint), source['sha256'], source['size'])
    reference = None if args.no_reference else verify_file(Path(args.reference), spec['onnx_sha256'], spec['onnx_size'])
    if output.exists() or output == checkpoint.resolve():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    export_graph(spec, source, checkpoint, output)
    rows = compare_reference(spec, output, reference, args.check_batches)
    packages = {}
    for package in ('torch', 'torchvision', 'numpy', 'onnx', 'onnxruntime', 'ultralytics', 'cn-clip', 'openai-clip', 'fastreid'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    digest = sha256(output)
    write_json(output.with_suffix('.rebuild.json'), {
        'schema_version': 1, 'model': args.model, 'source_sha256': source['sha256'],
        'onnx_sha256': digest, 'reference_sha256': spec['onnx_sha256'],
        'byte_identical_to_reference': digest == spec['onnx_sha256'],
        'reference_validation': 'not_requested' if args.no_reference else 'compared',
        'recipe': spec['recipe'], 'parameters': spec.get('export', {}),
        'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages,
        'catalog_sha256': sha256(catalog.path), 'verified_utc': datetime.now(timezone.utc).isoformat(),
        'validation': rows})
    print(f'VERIFIED {args.model}: byte_identical={digest == spec["onnx_sha256"]}; '
          f'reference_compared={reference is not None}; {len(rows)} output checks', flush=True)


if __name__ == '__main__':
    main()
