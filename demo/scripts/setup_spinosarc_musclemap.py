#!/usr/bin/env python3
"""Prepare the pinned official MuscleMap abdomen model for the local demo."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from _demo_paths import REPO_ROOT, runtime_root
import tarfile
import time
import urllib.request


REVISION = 'd11df779a4e146e7e913b89cb202fc06a6e2ef6a'
RECORD_ID = 19631081
VERSION = '0.0'
SOURCE_URL = f'https://codeload.github.com/MuscleMap/MuscleMap/tar.gz/{REVISION}'
REQUIRED_FILES = {'LICENSE', 'contrast_agnostic_abdomen_model.pth',
                  'contrast_agnostic_abdomen_model.json'}


def digest(path: Path, algorithm: str = 'sha256') -> str:
    result = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def fetch(url: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.part')
    last_update = 0.0

    def progress(blocks, block_size, total):
        nonlocal last_update
        now = time.monotonic()
        if now - last_update >= 5:
            received = blocks * block_size
            total_label = f' / {total / 1024**2:.1f} MiB' if total > 0 else ''
            print(f'{destination.name}: {received / 1024**2:.1f} MiB{total_label}', flush=True)
            last_update = now

    try:
        urllib.request.urlretrieve(url, temporary, reporthook=progress)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def check_manifest(directory: Path):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('record_id') != RECORD_ID or manifest.get('version') != VERSION:
        raise ValueError('The local MuscleMap manifest uses another model record/version.')
    if manifest.get('source_revision') != REVISION:
        raise ValueError('The local MuscleMap source revision differs from the pinned revision.')
    required_seen = set()
    for entry in manifest['files']:
        path = (directory / entry['path']).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError('MuscleMap manifest path escapes its directory.')
        if not path.is_file() or path.stat().st_size != entry['size'] or digest(path) != entry['sha256']:
            raise ValueError(f'MuscleMap asset checksum/size mismatch: {path.name}')
        required_seen.add(path.name)
    if not REQUIRED_FILES.issubset(required_seen):
        raise ValueError('MuscleMap manifest is missing an official model/config/license asset.')
    upstream = json.loads((directory / 'source' / 'UPSTREAM.json').read_text())
    if upstream.get('revision') != REVISION or digest(directory / 'source.tar.gz') != upstream['archive_sha256']:
        raise ValueError('The local MuscleMap source archive does not match its provenance.')
    config = json.loads((directory / 'source/scripts/models/abdomen/v0.0/contrast_agnostic_abdomen_model.json').read_text())
    if len(config['labels']) != 8 or config['model']['out_channels'] != 9:
        raise ValueError('The official abdomen configuration does not have eight muscle classes.')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=runtime_root() / 'var/spinosarc/musclemap')
    parser.add_argument('--verify-only', action='store_true', help='Verify local assets without network access.')
    args = parser.parse_args()
    directory = args.data_dir.resolve()
    if args.verify_only:
        check_manifest(directory)
        print('Pinned official MuscleMap abdomen assets verified; no network access.')
        return
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / 'source'
    archive = directory / 'source.tar.gz'
    if not (source / 'UPSTREAM.json').exists():
        if not archive.exists():
            fetch(SOURCE_URL, archive)
        with tarfile.open(archive) as bundle:
            prefix = bundle.getnames()[0].split('/')[0]
            for member in bundle.getmembers():
                relative = Path(member.name).relative_to(prefix)
                if str(relative) == '.' or not (member.isfile() or member.isdir()):
                    continue
                member.name = str(relative)
                bundle.extract(member, source, filter='data')
        (source / 'UPSTREAM.json').write_text(json.dumps({
            'repository': 'https://github.com/MuscleMap/MuscleMap',
            'revision': REVISION, 'archive_url': SOURCE_URL, 'archive_sha256': digest(archive),
        }, indent=2) + '\n')
    upstream = json.loads((source / 'UPSTREAM.json').read_text())
    if upstream.get('revision') != REVISION:
        raise ValueError('Refusing to overwrite an existing MuscleMap source with another revision.')
    record_path = directory / 'zenodo-abdomen-record.json'
    if not record_path.exists():
        request = urllib.request.Request(f'https://zenodo.org/api/records/{RECORD_ID}',
                                         headers={'Accept': 'application/json'})
        with urllib.request.urlopen(request, timeout=30) as response:
            record = json.load(response)
        record_path.write_text(json.dumps(record, indent=2) + '\n')
    record = json.loads(record_path.read_text())
    if record['id'] != RECORD_ID or str(record['metadata']['version']) != VERSION:
        raise ValueError('Zenodo model record/version differs from the pinned official release.')
    if record['metadata']['license']['id'] != 'mit-license':
        raise ValueError('The model record no longer has the expected MIT license.')
    model_dir = source / 'scripts/models/abdomen/v0.0'
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'record_id': RECORD_ID, 'source': f'https://zenodo.org/records/{RECORD_ID}',
                'version': VERSION, 'license': record['metadata']['license'],
                'source_revision': REVISION, 'files': []}
    seen = set()
    for entry in record['files']:
        filename = entry['key']
        if filename not in REQUIRED_FILES:
            continue
        seen.add(filename)
        destination = model_dir / filename
        algorithm, expected = entry['checksum'].split(':', 1)
        valid = (destination.is_file() and destination.stat().st_size == entry['size']
                 and digest(destination, algorithm) == expected)
        if not valid:
            fetch(entry['links']['self'], destination)
        if destination.stat().st_size != entry['size'] or digest(destination, algorithm) != expected:
            destination.unlink(missing_ok=True)
            raise ValueError(f'Official Zenodo checksum/size mismatch: {filename}')
        manifest['files'].append({
            'path': str(destination.relative_to(directory)), 'url': entry['links']['self'],
            'size': destination.stat().st_size, 'zenodo_checksum': entry['checksum'],
            'sha256': digest(destination),
        })
    if seen != REQUIRED_FILES:
        raise ValueError('Zenodo record is missing expected official model/config/license files.')
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    check_manifest(directory)
    print(f'MuscleMap abdomen v{VERSION} ready: {model_dir}')
    print('Runtime packages are installed by scripts/setup_spinosarc.py.')


if __name__ == '__main__':
    main()
