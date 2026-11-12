"""Fetch exact source bytes. Download failures never publish a partial artifact."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import os
from pathlib import Path
import tempfile
import urllib.request
from urllib.parse import parse_qs, urlencode, urlparse
import zipfile

from .catalog import sha256, verify_file, write_json


def drive_confirmation_url(page, original_url):
    """Follow only the public Drive large-file form for the configured file ID."""
    class Form(HTMLParser):
        def __init__(self):
            super().__init__()
            self.action, self.values, self.inside = None, {}, False

        def handle_starttag(self, tag, attributes):
            attrs = dict(attributes)
            if tag == 'form':
                self.action, self.inside = attrs.get('action'), True
            if tag == 'input' and self.inside and attrs.get('name'):
                self.values[attrs['name']] = attrs.get('value', '')

        def handle_endtag(self, tag):
            if tag == 'form':
                self.inside = False
    form = Form()
    form.feed(page)
    parsed = urlparse(form.action or '')
    file_id = parse_qs(urlparse(original_url).query).get('id')
    if (parsed.scheme != 'https' or parsed.netloc != 'drive.usercontent.google.com'
            or parsed.path != '/download' or not file_id or form.values.get('id') != file_id[0]):
        raise ValueError('Unexpected Google Drive download confirmation form')
    return form.action+'?'+urlencode(form.values)


def _copy_stream(stream, target, expected_sha256, expected_size):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.acquire-', dir=target.parent) as temporary:
        candidate = Path(temporary)/target.name
        digest, count = hashlib.sha256(), 0
        with candidate.open('wb') as output:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                count += len(chunk)
                if count > expected_size:
                    raise ValueError(f'Download exceeds expected size: {target.name}')
                digest.update(chunk)
                output.write(chunk)
        if count != expected_size or digest.hexdigest() != expected_sha256:
            raise ValueError(f'Download/source identity mismatch: {target.name}')
        os.replace(candidate, target)
    return target


def copy_verified(source, target, expected_sha256, expected_size):
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve():
        raise ValueError('Source and target must differ')
    with source.open('rb') as stream:
        return _copy_stream(stream, target, expected_sha256, expected_size)


def _download(urls, target, digest, size):
    errors = []
    for url in urls:
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'Triton-model-pipeline/1.0'})
            with urllib.request.urlopen(request, timeout=45) as stream:
                if urlparse(url).hostname == 'drive.google.com' and 'text/html' in stream.headers.get('Content-Type', ''):
                    page = stream.read(1024*1024).decode('utf-8')
                    confirmed = drive_confirmation_url(page, url)
                    with urllib.request.urlopen(confirmed, timeout=45) as download:
                        _copy_stream(download, target, digest, size)
                else:
                    _copy_stream(stream, target, digest, size)
            return url
        except Exception as exc:
            errors.append(f'{url}: {type(exc).__name__}: {exc}')
    raise RuntimeError('No matching download succeeded. '+('; '.join(errors) or 'No verified download URL is configured.'))


def acquire_source(catalog, source_id, cache, *, local_file=None, offline=False):
    source = catalog.sources[source_id]
    cache = catalog.output_path(cache)
    target = catalog.output_path(catalog.cached_source(cache, source_id))
    if target.exists():
        return verify_file(target, source['sha256'], source['size'])
    original = Path(local_file).resolve() if local_file is not None else (
        catalog.root/source['local_path'] if source.get('local_path') else None)
    if original is not None and original.is_file():
        copy_verified(original, target, source['sha256'], source['size'])
        origin = {'method': 'verified_local_copy', 'path': str(original)}
    elif local_file is not None:
        raise FileNotFoundError(original)
    elif source.get('archive'):
        archive = source['archive']
        archive_path = catalog.output_path(cache/'archives'/archive['sha256']/archive['filename'])
        if archive_path.exists():
            verify_file(archive_path, archive['sha256'], archive['size'])
        else:
            local_archive = catalog.root/archive['local_path'] if archive.get('local_path') else None
            if local_archive and local_archive.is_file():
                copy_verified(local_archive, archive_path, archive['sha256'], archive['size'])
            elif offline:
                raise FileNotFoundError(f'No verified local source/archive for {source_id}')
            else:
                _download(archive['urls'], archive_path, archive['sha256'], archive['size'])
        with zipfile.ZipFile(archive_path) as source_zip:
            member = source_zip.getinfo(archive['member'])
            if member.is_dir() or member.file_size != source['size']:
                raise ValueError('Unexpected source archive member')
            with source_zip.open(member) as stream:
                _copy_stream(stream, target, source['sha256'], source['size'])
        origin = {'method': 'verified_archive_member', 'archive_sha256': archive['sha256'], 'member': archive['member']}
    elif offline:
        raise FileNotFoundError(f'No verified local source for {source_id}; supply --source-file {source_id}=PATH or run acquire without --offline')
    else:
        url = _download(source.get('urls', []), target, source['sha256'], source['size'])
        origin = {'method': 'verified_download', 'url': url}
    write_json(target.with_name(target.name+'.acquisition.json'), {
        'source_id': source_id, 'sha256': sha256(target), 'size': target.stat().st_size,
        'acquired_utc': datetime.now(timezone.utc).isoformat(), **origin})
    return target
