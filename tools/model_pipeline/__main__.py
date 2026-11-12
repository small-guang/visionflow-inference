"""Download pinned sources, export ONNX, and build independent TensorRT plans."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .catalog import Catalog, DEFAULT_CATALOG, PROJECT_ROOT, sha256, verify_file, write_json


def assignments(values):
    result = {}
    for value in values:
        name, separator, path = value.partition('=')
        if not separator or not name or not path or name in result:
            raise ValueError(f'Expected a unique SOURCE_ID=PATH: {value}')
        result[name] = Path(path).resolve()
    return result


def audit(catalog, models, cache):
    sources, artifacts = {}, {}
    for sid in catalog.source_ids(models):
        spec = catalog.sources[sid]
        path = catalog.cached_source(cache, sid)
        if not path.exists() and spec.get('local_path'):
            path = catalog.root/spec['local_path']
        try:
            verify_file(path, spec['sha256'], spec['size'])
            sources[sid] = {'ok': True, 'path': str(path), 'sha256': spec['sha256']}
        except (OSError, ValueError) as error:
            sources[sid] = {'ok': False, 'error': str(error)}
    for model in models:
        spec = catalog.models[model]
        try:
            verify_file(catalog.reference(model), spec['onnx_sha256'], spec['onnx_size'])
            artifacts[model] = {'ok': True, 'onnx_sha256': spec['onnx_sha256']}
        except (OSError, ValueError) as error:
            artifacts[model] = {'ok': False, 'error': str(error)}
    return {'catalog_sha256': sha256(catalog.path), 'sources': sources, 'onnx': artifacts,
            'ok': all(r['ok'] for group in (sources, artifacts) for r in group.values())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=DEFAULT_CATALOG)
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('list', 'audit', 'acquire', 'onnx', 'plan'):
        command = commands.add_parser(name)
        command.add_argument('--models', nargs='+', default=['native' if name == 'plan' else 'all'])
        if name in ('audit', 'acquire', 'onnx'):
            command.add_argument('--cache', type=Path, default=Path('artifacts/sources'))
        if name in ('acquire', 'onnx'):
            command.add_argument('--source-file', action='append', default=[], metavar='SOURCE_ID=PATH')
            command.add_argument('--offline', action='store_true')
        if name in ('onnx', 'plan'):
            command.add_argument('--output', type=Path, required=True)
        if name == 'audit':
            command.add_argument('--output', type=Path)
        if name == 'onnx':
            command.add_argument('--python', default=sys.executable)
            command.add_argument('--sbs-python', help='Dedicated Python 3.10 SBS runtime')
            command.add_argument('--python-path', type=Path, help='Private modern dependency overlay')
            command.add_argument('--sbs-python-path', type=Path)
            references = command.add_mutually_exclusive_group()
            references.add_argument('--reference-root', type=Path)
            references.add_argument('--no-reference', action='store_true',
                                    help='Export from source only; validate ONNX contracts without a reference file')
            command.add_argument('--check-batches', type=int, nargs='+', default=[1, 2])
        if name == 'plan':
            command.add_argument('--onnx-root', type=Path, required=True)
            command.add_argument('--max-batch', type=int, required=True)
            command.add_argument('--opt-batch', type=int, default=1)
            command.add_argument('--workspace-mib', type=int, default=1024)
            command.add_argument('--precision', choices=['fp16', 'fp32'])
            command.add_argument('--trtexec', default='trtexec')
            command.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    catalog = Catalog(args.catalog, project_root=args.project_root)
    models = catalog.select(args.models)
    if args.command == 'list':
        for name in models:
            spec = catalog.models[name]
            print(f'{name}: {spec["source"]} -> {spec["recipe"]} -> ONNX; current_native_plan={spec["native_plan"]}')
    elif args.command == 'audit':
        result = audit(catalog, models, args.cache)
        if args.output:
            output = catalog.output_path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            write_json(output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result['ok'] else 1
    elif args.command in ('acquire', 'onnx'):
        from .acquire import acquire_source
        overrides = assignments(args.source_file)
        if set(overrides)-set(catalog.sources):
            raise ValueError('Unknown source-file override')
        for sid in catalog.source_ids(models):
            result = acquire_source(catalog, sid, args.cache, local_file=overrides.get(sid), offline=args.offline)
            print(f'VERIFIED SOURCE {sid}: {result}', flush=True)
        if args.command == 'onnx':
            from .export import export_candidate
            # Require the old SBS runtime explicitly; never guess an incompatible modern environment.
            if any(catalog.models[m]['environment'] == 'sbs' for m in models) and not args.sbs_python:
                raise ValueError('SBS requires --sbs-python pointing to its dedicated Python 3.10 runtime')
            for model in models:
                sbs = catalog.models[model]['environment'] == 'sbs'
                result = export_candidate(catalog, model, args.cache, args.output,
                         python=args.sbs_python if sbs else args.python,
                         python_path=args.sbs_python_path if sbs else args.python_path,
                         offline=True, reference_root=args.reference_root, no_reference=args.no_reference,
                         check_batches=args.check_batches)
                print(f'VERIFIED ONNX {result}', flush=True)
    elif args.command == 'plan':
        from tools.build_tensorrt_plans import build_engine, engine_command
        output = catalog.output_path(args.output)
        jobs = []
        for model in models:
            source = args.onnx_root.resolve()/model/'1/model.onnx'
            target = catalog.output_path(output/model/'1'/f'model_b{args.max_batch}.plan')
            if target.exists():
                raise FileExistsError(f'Choose a new candidate directory: {target}')
            settings = dict(model=model, max_batch=args.max_batch, opt_batch=args.opt_batch,
                            workspace_mib=args.workspace_mib, precision=args.precision)
            command = engine_command(args.trtexec, source, target, **settings)
            jobs.append((source, target, settings, command))
        for source, target, settings, command in jobs:
            if args.dry_run:
                print(json.dumps(command))
            else:
                print(build_engine(source, target, trtexec=args.trtexec, **settings))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
