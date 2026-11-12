"""Export in a child runtime and publish only a verified, independent candidate."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

from .acquire import acquire_source


def export_candidate(catalog, model, cache, output, *, python=sys.executable,
                     python_path=None, local_file=None, offline=False,
                     reference_root=None, no_reference=False, check_batches=(1, 2)):
    if no_reference and reference_root is not None:
        raise ValueError('--no-reference and --reference-root are mutually exclusive')
    output = catalog.output_path(output)
    target = catalog.output_path(output/model)
    if target.exists():
        raise FileExistsError(f'Choose a new candidate directory: {target}')
    checkpoint = acquire_source(catalog, catalog.models[model]['source'], cache,
                                local_file=local_file, offline=offline)
    output.mkdir(parents=True, exist_ok=True)
    log = output/f'{model}.{uuid.uuid4().hex}.export.log'
    with tempfile.TemporaryDirectory(prefix='.export-', dir=output) as temporary:
        staged = Path(temporary)/model
        destination = staged/'1/model.onnx'
        command = [str(python), '-m', 'tools.model_pipeline.export_worker',
                   '--catalog', str(catalog.path), '--project-root', str(catalog.root),
                   '--model', model, '--checkpoint', str(checkpoint), '--output', str(destination),
                   '--check-batches', *map(str, check_batches)]
        command += (['--no-reference'] if no_reference else
                    ['--reference', str(catalog.reference(model, reference_root).resolve())])
        environment = os.environ.copy()
        environment['PYTHONPATH'] = os.pathsep.join(
            [str(Path(python_path).resolve())] if python_path else [])
        environment['PYTHONPATH'] += os.pathsep+str(catalog.root)
        environment['CUDA_VISIBLE_DEVICES'] = ''
        environment['PYTHONUTF8'] = '1'
        environment['OMP_NUM_THREADS'] = '4'
        environment['MKL_NUM_THREADS'] = '4'
        yolo_settings = Path(temporary)/'yolo-settings'
        # Ultralytics tests parent writability and falls back to CWD if absent.
        yolo_settings.mkdir()
        environment['YOLO_CONFIG_DIR'] = str(yolo_settings)
        with log.open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=catalog.root, env=environment,
                                    stdout=stream, stderr=subprocess.STDOUT, check=False)
        if result.returncode or not destination.is_file() or not destination.with_suffix('.rebuild.json').is_file():
            raise RuntimeError(f'ONNX export/verification failed; see {log}')
        catalog.verify_candidate(model, destination)
        staged.rename(target)
    return target/'1/model.onnx'
