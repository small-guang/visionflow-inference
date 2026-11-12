"""Versioned model identities and a write boundary around the live repository."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT
DEFAULT_CATALOG = Path(__file__).with_name('catalog.json')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def relative_name(value: str) -> str:
    """Accept portable relative source and model names on Windows and Linux."""
    if not isinstance(value, str) or not value or value in ('.', '..') or '\\' in value or ':' in value:
        raise ValueError(f'Unsafe relative path: {value!r}')
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(part in ('.', '..') for part in path.parts):
        raise ValueError(f'Unsafe relative path: {value!r}')
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}
    if any(part.endswith((' ', '.')) or part.split('.')[0].upper() in reserved for part in path.parts):
        raise ValueError(f'Nonportable relative path: {value!r}')
    return value


def verify_file(path: Path, expected_sha256: str, expected_size: int | None = None) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ValueError(f'Size mismatch: {path}')
    if sha256(path) != expected_sha256:
        raise ValueError(f'SHA256 mismatch: {path}')
    return path


def write_json(path: Path, value) -> None:
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8', newline='\n')


class Catalog:
    def __init__(self, path=DEFAULT_CATALOG, *, project_root=PROJECT_ROOT):
        self.root = Path(project_root).resolve()
        self.path = Path(path).resolve()
        self.data = json.loads(self.path.read_text(encoding='utf-8'))
        if self.data.get('schema_version') != 1:
            raise ValueError('Unsupported model catalog schema')
        self.sources = self.data['sources']
        self.models = self.data['models']
        self.aliases = {}
        for source_id, source in self.sources.items():
            self._identifier(source_id)
            relative_name(source['filename'])
            if '/' in source['filename']:
                raise ValueError('Source filename must be a basename')
            if source.get('local_path'):
                relative_name(source['local_path'])
            self._identity(source['sha256'], source['size'])
            for url in source.get('urls', []):
                if not url.startswith('https://'):
                    raise ValueError('Source URLs must use HTTPS')
            if source.get('archive'):
                archive = source['archive']
                self._identity(archive['sha256'], archive['size'])
                relative_name(archive['member'])
                relative_name(archive['filename'])
                if archive.get('local_path'):
                    relative_name(archive['local_path'])
                if '/' in archive['filename'] or any(not u.startswith('https://') for u in archive['urls']):
                    raise ValueError('Invalid source archive')
        for name, model in self.models.items():
            self._identifier(name)
            if model['source'] not in self.sources:
                raise ValueError(f'Unknown source for {name}')
            relative_name(model['reference_onnx'])
            self._identity(model['onnx_sha256'], model['onnx_size'])
            for alias in (name, *model.get('aliases', [])):
                if alias in ('all', 'native') or alias in self.aliases:
                    raise ValueError(f'Duplicate/reserved model alias: {alias}')
                self.aliases[alias] = name

    def output_path(self, path):
        path = Path(path).resolve()
        protected = []
        # An alternate root must not disable the calling tool's own boundary.
        for root in {self.root, PROJECT_ROOT}:
            protected += [root/p for p in ('model_repository', 'tools/models', '.git', '.env')]
            protected += list(root.glob('docker-compose*.y*ml'))
        for reserved in protected:
            reserved = reserved.resolve()
            if path == reserved or path.is_relative_to(reserved) or reserved.is_relative_to(path):
                raise ValueError(f'Output overlaps the current deployment/source tree: {path}')
        return path

    @staticmethod
    def _identifier(name):
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError(f'Invalid catalog identifier: {name!r}')

    @staticmethod
    def _identity(digest, size):
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('Expected a lowercase SHA256')
        if type(size) is not int or size <= 0:
            raise ValueError('Artifact size must be a positive integer')

    def select(self, requested):
        result = []
        for value in requested:
            if value == 'all':
                names = self.models
            elif value == 'native':
                names = [name for name, spec in self.models.items() if spec['native_plan']]
            elif value in self.aliases:
                names = [self.aliases[value]]
            else:
                raise ValueError(f'Unknown model/alias: {value}')
            for name in names:
                if name not in result:
                    result.append(name)
        if not result:
            raise ValueError('No models selected')
        return result

    def source_ids(self, models):
        return list(dict.fromkeys(self.models[name]['source'] for name in models))

    def cached_source(self, cache, source_id):
        source = self.sources[source_id]
        return Path(cache)/source_id/source['sha256']/source['filename']

    def reference(self, model, reference_root=None):
        if reference_root is None:
            return self.root/self.models[model]['reference_onnx']
        return Path(reference_root)/model/'1/model.onnx'

    def verify_candidate(self, model, path):
        """Check a candidate's rebuild record before publishing its directory."""
        path = Path(path)
        record = json.loads(path.with_suffix('.rebuild.json').read_text(encoding='utf-8'))
        if not isinstance(record, dict) or type(record.get('schema_version')) is not int or record['schema_version'] != 1:
            raise ValueError('Invalid candidate rebuild record schema')
        source = self.sources[self.models[model]['source']]
        if record.get('model') != model or record.get('source_sha256') != source['sha256']:
            raise ValueError('Candidate rebuild record has a different model/source identity')
        digest = record.get('onnx_sha256')
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('Candidate rebuild record has no valid ONNX SHA256')
        verify_file(path, digest)
        return record
