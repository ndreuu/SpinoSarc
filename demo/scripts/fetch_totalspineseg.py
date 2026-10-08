#!/usr/bin/env python3
"""Explicit, bounded preparation of the two official TotalSpineSeg model assets.

Only the standard library is needed. No package is installed and no inference
is run. Cached ZIPs, extracted model files, and an adapter-compatible SHA256
manifest remain local. This download does not establish a separate weights
license: the upstream code license and that unresolved fact are recorded.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import tempfile
import threading
import urllib.request
import urllib.error
import zipfile
try:
    from ._demo_paths import runtime_root
except ImportError:
    from _demo_paths import runtime_root


PACKAGE_VERSION = "20260730"
WEIGHTS_RELEASE = "r20260730"
DATASETS = ("Dataset101_TotalSpineSeg_step1", "Dataset102_TotalSpineSeg_step2")
ASSET_URLS = {
    dataset: f"https://github.com/neuropoly/totalspineseg/releases/download/{WEIGHTS_RELEASE}/{dataset}_{WEIGHTS_RELEASE}.zip"
    for dataset in DATASETS
}
MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024
MAX_UNPACKED_BYTES = 2 * 1024 * 1024 * 1024
# Auditing the actual pinned r20260730 ZIP directories found checkpoint sizes
# 1,129,915,218 and 1,130,070,162 bytes. Their compressed ZIPs fit below 1 GiB,
# but individual decoded checkpoints need this modestly larger 1.5 GiB bound.
MAX_MEMBER_BYTES = 1536 * 1024 * 1024
MAX_ENTRIES = 5000
CHUNK_BYTES = 1024 * 1024
MAX_DIRECTORY_BYTES = 16 * 1024 * 1024
RANGE_CHUNK_BYTES = 8 * 1024 * 1024
RANGE_HTTP_TIMEOUT_SECONDS = 180
RANGE_RETRY_BACKOFF = (1, 2, 4)
CONTENT_RANGE = re.compile(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)\Z")


class PreparationError(ValueError):
    pass


class TruncatedRangeError(PreparationError):
    """A valid range response ended early; retry without trusting its bytes."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_asset(url: str, destination: Path, *, max_bytes: int = MAX_ARCHIVE_BYTES,
                   opener=None, progress=None, range_workers: int = 1) -> dict:
    """Atomic cache write; partial or oversized responses never become a ZIP."""
    destination = Path(destination)
    if range_workers not in {1, 2, 4, 8}:
        raise PreparationError("Use one, two, four, or eight range workers.")
    if destination.exists():
        if destination.is_symlink() or not destination.is_file() or destination.stat().st_size > max_bytes:
            raise PreparationError("Invalid cached model archive.")
        if not zipfile.is_zipfile(destination):
            raise PreparationError("Cached archive is not a ZIP; remove the invalid cache explicitly.")
        return {"url": url, "archive": str(destination.resolve()), "sha256": sha256_file(destination),
                "bytes": destination.stat().st_size, "cached": True}
    destination.parent.mkdir(parents=True, exist_ok=True)
    if range_workers > 1:
        return _download_range_asset(url, destination, max_bytes=max_bytes,
                                     opener=opener, progress=progress, workers=range_workers)
    request = urllib.request.Request(url, headers={"User-Agent": "lumbar-mri-research-demo/0.1", "Accept-Encoding": "identity"})
    opener = opener or urllib.request.urlopen
    temporary = None
    try:
        with opener(request, timeout=60) as response:
            if getattr(response, "status", 200) != 200:
                raise PreparationError("Model archive download requires HTTP 200.")
            if response.headers.get("Content-Encoding", "identity").lower() not in {"", "identity"}:
                raise PreparationError("Encoded archive responses are unsupported.")
            length = response.headers.get("Content-Length")
            try:
                expected = int(length) if length is not None else None
            except ValueError as exc:
                raise PreparationError("Invalid archive Content-Length.") from exc
            if expected is not None and not 0 < expected <= max_bytes:
                raise PreparationError("Model archive exceeds the download limit.")
            final_url = response.geturl()
            if not final_url.startswith("https://"):
                raise PreparationError("Model archive redirect must use HTTPS.")
            received = 0
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", suffix=".partial", delete=False) as stream:
                temporary = Path(stream.name)
                while True:
                    chunk = response.read(min(CHUNK_BYTES, max_bytes - received + 1))
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > max_bytes:
                        raise PreparationError("Model archive exceeds the download limit.")
                    stream.write(chunk)
                    if progress:
                        progress(received, expected)
                stream.flush()
                os.fsync(stream.fileno())
            if received == 0 or (expected is not None and received != expected):
                raise PreparationError("Model archive body length does not match its header.")
            if not zipfile.is_zipfile(temporary):
                raise PreparationError("Downloaded model archive is not a ZIP.")
            digest = sha256_file(temporary)
            os.replace(temporary, destination)
            temporary = None
        return {"url": url, "archive": str(destination.resolve()), "sha256": digest,
                "bytes": received, "cached": False, "final_url": final_url}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _range_headers(response, start: int, end: int, *, max_bytes: int,
                   total: int | None = None, validator: tuple[str, str] | None = None) -> tuple[int, str]:
    """Strict range semantics; HTTP 200 is rejected before reading any body."""
    if getattr(response, "status", None) != 206:
        raise PreparationError("Archive range server must return HTTP 206; refusing a full response.")
    if response.headers.get("Content-Encoding", "identity").lower() not in {"", "identity"}:
        raise PreparationError("Archive range responses must have identity encoding.")
    match = CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
    if not match:
        raise PreparationError("Missing or invalid archive Content-Range.")
    actual_start, actual_end, actual_total = map(int, match.groups())
    if (actual_start, actual_end) != (start, end) or not 0 < actual_total <= max_bytes or end >= actual_total:
        raise PreparationError("Archive Content-Range does not match the requested bounded range.")
    if total is not None and actual_total != total:
        raise PreparationError("Archive total size changed between range requests.")
    length = response.headers.get("Content-Length")
    if length is not None:
        try:
            length = int(length)
        except ValueError as exc:
            raise PreparationError("Invalid archive range Content-Length.") from exc
        if length != end - start + 1:
            raise PreparationError("Archive range Content-Length does not match the requested range.")
    if validator and response.headers.get(validator[0]) != validator[1]:
        raise PreparationError("Archive resource validator changed between range requests.")
    final_url = response.geturl()
    if not final_url.startswith("https://"):
        raise PreparationError("Model archive redirect must use HTTPS.")
    return actual_total, final_url


def _retry_range_network(operation, cancelled: threading.Event):
    """At most four attempts; protocol and filesystem errors are never retried."""
    for attempt in range(len(RANGE_RETRY_BACKOFF) + 1):
        if cancelled.is_set():
            raise PreparationError("Archive range download was cancelled after another range failed.")
        try:
            return operation()
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code <= 599
            exc.close()
            if not retryable or attempt == len(RANGE_RETRY_BACKOFF):
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionResetError, http.client.IncompleteRead, TruncatedRangeError):
            if attempt == len(RANGE_RETRY_BACKOFF):
                raise
        if cancelled.wait(RANGE_RETRY_BACKOFF[attempt]):
            raise PreparationError("Archive range download was cancelled after another range failed.")


def _atomic_cache_record(path: Path, value: dict) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".source-", suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _chunk_cache(destination: Path, identity: dict, ranges: list[tuple[int, int]]) -> tuple[Path, dict]:
    """Only generated flat filenames are read; every reused chunk is SHA-checked."""
    directory = destination.parent / f".{destination.name}.ranges"
    if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
        raise PreparationError("Invalid archive range cache directory.")
    directory.mkdir(exist_ok=True)
    chunk_pattern = re.compile(r"[0-9]{12}-[0-9]{12}\.chunk\Z")
    entries = list(directory.iterdir())
    if len(entries) > 2 * len(ranges) + 100:
        raise PreparationError("Archive range cache exceeds its entry budget.")
    for path in entries:
        generated = path.name == "source.json" or chunk_pattern.fullmatch(path.name) or (
            path.name.startswith((".part-", ".source-")) and path.name.endswith(".partial"))
        if not generated or path.is_symlink() or not path.is_file():
            raise PreparationError("Archive range cache contains unexpected or unsafe entries.")
        if path.name.endswith(".partial"):
            path.unlink()  # Interrupted chunks are never trusted or reused.
    source_path = directory / "source.json"
    existing = {}
    if source_path.exists() and source_path.stat().st_size <= 1024 * 1024:
        try:
            existing = json.loads(source_path.read_text(encoding="utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            pass
    if not isinstance(existing, dict) or existing.get("schema_version") != 1 or existing.get("source") != identity or not identity["validator"]:
        for path in directory.glob("*.chunk"):
            path.unlink()
        existing = {"schema_version": 1, "source": identity, "chunks": {}}
    recorded = existing.get("chunks", {})
    if not isinstance(recorded, dict):
        recorded = {}
    expected = {f"{start:012d}-{end:012d}.chunk": end - start + 1 for start, end in ranges}
    verified = {}
    for path in directory.glob("*.chunk"):
        value = recorded.get(path.name)
        expected_size = expected.get(path.name)
        if (expected_size is None or not isinstance(value, dict) or value.get("bytes") != expected_size
                or path.stat().st_size != expected_size or value.get("sha256") != sha256_file(path)):
            path.unlink()
        else:
            verified[path.name] = value
    record = {"schema_version": 1, "source": identity, "chunks": verified}
    _atomic_cache_record(source_path, record)
    return directory, record


def _download_range_asset(url: str, destination: Path, *, max_bytes: int,
                          opener=None, progress=None, workers: int) -> dict:
    """Persistent verified 8 MiB chunks survive a failed run; final ZIP is atomic."""
    opener = opener or urllib.request.urlopen
    lock, cancelled = threading.Lock(), threading.Event()
    request_count = 0

    def open_response(request):
        nonlocal request_count
        with lock:
            request_count += 1
        return opener(request, timeout=RANGE_HTTP_TIMEOUT_SECONDS)

    base_headers = {"User-Agent": "lumbar-mri-research-demo/0.1", "Accept-Encoding": "identity"}
    probe = urllib.request.Request(url, headers={**base_headers, "Range": "bytes=0-0"})

    def read_probe():
        with open_response(probe) as response:
            total, final_url = _range_headers(response, 0, 0, max_bytes=max_bytes)
            if len(response.read(2)) != 1:
                raise PreparationError("Archive range probe body length is not one byte.")
            etag, modified = response.headers.get("ETag"), response.headers.get("Last-Modified")
            validator = ("ETag", etag) if etag else (("Last-Modified", modified) if modified else None)
            return total, final_url, validator

    total, final_url, validator = _retry_range_network(read_probe, cancelled)
    ranges = [(start, min(start + RANGE_CHUNK_BYTES, total) - 1) for start in range(0, total, RANGE_CHUNK_BYTES)]
    identity = {"url": url, "total": total, "validator": list(validator) if validator else None,
                "chunk_bytes": RANGE_CHUNK_BYTES}
    cache_dir, record = _chunk_cache(destination, identity, ranges)
    resumed_chunks = len(record["chunks"])
    received = sum(value["bytes"] for value in record["chunks"].values())
    if progress:
        progress(received, total)

    def fetch_slot(bounds):
        nonlocal received
        start, end = bounds
        filename = f"{start:012d}-{end:012d}.chunk"
        expected = end - start + 1
        if filename in record["chunks"]:
            return expected
        headers = {**base_headers, "Range": f"bytes={start}-{end}"}
        if validator:
            key, value = validator
            if key == "ETag" and not value.startswith("W/"):
                headers["If-Match"] = value
            elif key == "Last-Modified":
                headers["If-Unmodified-Since"] = value
        request = urllib.request.Request(url, headers=headers)

        def one_attempt():
            partial = None
            try:
                with open_response(request) as response:
                    _range_headers(response, start, end, max_bytes=max_bytes, total=total, validator=validator)
                    count, digest = 0, hashlib.sha256()
                    with tempfile.NamedTemporaryFile(dir=cache_dir, prefix=".part-", suffix=".partial", delete=False) as stream:
                        partial = Path(stream.name)
                        while True:
                            if cancelled.is_set():
                                raise PreparationError("Archive range download was cancelled after another range failed.")
                            chunk = response.read(min(CHUNK_BYTES, expected - count + 1))
                            if not chunk:
                                break
                            count += len(chunk)
                            if count > expected:
                                raise PreparationError("Archive range body exceeded the requested length.")
                            stream.write(chunk)
                            digest.update(chunk)
                        if count != expected:
                            raise TruncatedRangeError("Archive range body was shorter than the requested length.")
                        stream.flush()
                        os.fsync(stream.fileno())
                os.replace(partial, cache_dir / filename)
                partial = None
                return {"bytes": count, "sha256": digest.hexdigest()}
            finally:
                if partial is not None:
                    partial.unlink(missing_ok=True)

        value = _retry_range_network(one_attempt, cancelled)
        with lock:
            record["chunks"][filename] = value
            _atomic_cache_record(cache_dir / "source.json", record)
            received += expected
            if progress:
                progress(received, total)
        return expected

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_slot, bounds) for bounds in ranges]
        try:
            counts = [future.result() for future in as_completed(futures)]
        except BaseException:
            cancelled.set()
            for future in futures:
                future.cancel()
            raise  # Verified cache chunks deliberately remain available.
    if sum(counts) != total:
        raise PreparationError("Archive chunk size does not match its range probe.")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
            for start, end in ranges:
                filename = f"{start:012d}-{end:012d}.chunk"
                count, chunk_digest = 0, hashlib.sha256()
                with (cache_dir / filename).open("rb") as chunk:
                    for block in iter(lambda: chunk.read(CHUNK_BYTES), b""):
                        count += len(block)
                        chunk_digest.update(block)
                        stream.write(block)
                if count != end - start + 1 or chunk_digest.hexdigest() != record["chunks"][filename]["sha256"]:
                    raise PreparationError("A cached archive chunk changed before assembly.")
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.stat().st_size != total or not zipfile.is_zipfile(temporary):
            raise PreparationError("Downloaded model archive is not a valid ZIP of the probed size.")
        _check_zip_directory_budget(temporary)
        digest = sha256_file(temporary)
        os.replace(temporary, destination)
        temporary = None
        shutil.rmtree(cache_dir)
        return {"url": url, "archive": str(destination.resolve()), "sha256": digest,
                "bytes": total, "cached": False, "final_url": final_url,
                "range_workers": workers, "range_requests": request_count, "resumed_chunks": resumed_chunks}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _archive_members(archive: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    infos = archive.infolist()
    if len(infos) > MAX_ENTRIES:
        raise PreparationError("Model archive has too many entries.")
    total = 0
    seen = set()
    members = []
    for info in infos:
        name = info.filename
        path = PurePosixPath(name)
        if not name or not path.parts or "\\" in name or "\x00" in name or path.is_absolute() or ".." in path.parts or ":" in path.parts[0]:
            raise PreparationError("Unsafe model archive path.")
        normalized = path.as_posix().rstrip("/")
        if normalized in seen:
            raise PreparationError("Duplicate model archive path.")
        seen.add(normalized)
        mode = info.external_attr >> 16
        kind = stat.S_IFMT(mode)
        if kind not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise PreparationError("Symlinks or special archive entries are unsupported.")
        if info.flag_bits & 1:
            raise PreparationError("Encrypted model archives are unsupported.")
        if info.file_size > MAX_MEMBER_BYTES:
            raise PreparationError("Model archive member exceeds the unpacked limit.")
        total += info.file_size
        if total > MAX_UNPACKED_BYTES:
            raise PreparationError("Model archive exceeds the unpacked limit.")
        if info.file_size > 1024 * 1024 and info.file_size > max(info.compress_size, 1) * 500:
            raise PreparationError("Suspicious model archive compression ratio.")
        members.append((info, path))
    return members


def _check_zip_directory_budget(path: Path) -> None:
    """Bound the directory before zipfile allocates one ZipInfo per entry."""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 65557))
        tail = stream.read(65557)
    start = tail.rfind(b"PK\x05\x06")
    if start < 0 or len(tail) - start < 22:
        raise PreparationError("Invalid model ZIP directory.")
    fields = struct.unpack("<4s4H2LH", tail[start:start + 22])
    if fields[1] != 0 or fields[2] != 0:
        raise PreparationError("Multidisk model archives are unsupported.")
    if fields[4] > MAX_ENTRIES or fields[5] > MAX_DIRECTORY_BYTES:
        raise PreparationError("Model ZIP directory exceeds its entries or metadata budget.")


def model_files(dataset_dir: Path) -> list[Path]:
    folds = sorted(dataset_dir.glob("*/fold_0"))
    if len(folds) != 1 or len(folds[0].parent.name.split("__")) != 3:
        raise PreparationError("Expected one valid fold_0 model per dataset.")
    fold = folds[0]
    checkpoint = fold / "checkpoint_best.pth"
    if not checkpoint.is_file():
        checkpoint = fold / "checkpoint_final.pth"
    paths = [checkpoint, fold.parent / "plans.json", fold.parent / "dataset.json"]
    if any(not path.is_file() or path.is_symlink() or path.stat().st_size == 0 for path in paths):
        raise PreparationError("Required checkpoint/plans/dataset files are missing.")
    return paths


def extract_asset(archive_path: Path, dataset: str, data_dir: Path) -> list[dict]:
    """CRC-checked bounded extraction, then atomic installation of one dataset."""
    if dataset not in DATASETS:
        raise PreparationError("Unknown model dataset.")
    if archive_path.is_symlink() or not archive_path.is_file() or archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise PreparationError("Invalid model archive file.")
    release_dir = Path(data_dir) / "nnUNet" / "results" / WEIGHTS_RELEASE
    release_dir.mkdir(parents=True, exist_ok=True)
    license_artifacts = []
    _check_zip_directory_budget(archive_path)
    with tempfile.TemporaryDirectory(dir=release_dir.parent, prefix=f".{dataset}-") as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(archive_path) as archive:
            # Validate every path and declared resource budget before writing.
            members = _archive_members(archive)
            for info, relative in members:
                destination = staging.joinpath(*relative.parts)
                if info.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                count = 0
                with archive.open(info) as source, destination.open("xb") as output:
                    while True:
                        chunk = source.read(min(CHUNK_BYTES, MAX_MEMBER_BYTES - count + 1))
                        if not chunk:
                            break
                        count += len(chunk)
                        if count > info.file_size or count > MAX_MEMBER_BYTES:
                            raise PreparationError("Model member exceeded its declared size.")
                        output.write(chunk)
                if count != info.file_size:
                    raise PreparationError("Model member size does not match the archive directory.")
                # zipfile verifies each member's CRC at EOF.
                if "license" in relative.name.lower() or "copying" in relative.name.lower():
                    license_artifacts.append({"dataset": dataset, "archive_entry": relative.as_posix(),
                                              "sha256": sha256_file(destination), "bytes": count})
        staged_dataset = staging / dataset
        staged_files = model_files(staged_dataset)
        destination = release_dir / dataset
        if destination.exists():
            existing = model_files(destination)
            expected = {path.relative_to(staged_dataset).as_posix(): sha256_file(path) for path in staged_files}
            actual = {path.relative_to(destination).as_posix(): sha256_file(path) for path in existing}
            if expected != actual:
                raise PreparationError("Existing model differs from the archive; choose an empty data directory.")
        else:
            os.rename(staged_dataset, destination)
    return license_artifacts


def verify_manifest(data_dir: Path) -> dict:
    path = Path(data_dir) / "manifest.json"
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 1024 * 1024:
        raise PreparationError("A valid existing model manifest is required.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("provider") != "totalspineseg" or manifest.get("package_version") != PACKAGE_VERSION or manifest.get("weights_release") != WEIGHTS_RELEASE:
        raise PreparationError("Model manifest does not match the pinned provider.")
    recorded = manifest.get("files")
    if not isinstance(recorded, dict):
        raise PreparationError("Model manifest has no file hashes.")
    root = Path(data_dir).resolve()
    required = []
    for dataset in DATASETS:
        required.extend(model_files(root / "nnUNet" / "results" / WEIGHTS_RELEASE / dataset))
    for model_path in required:
        relative = model_path.relative_to(root).as_posix()
        if recorded.get(relative) != sha256_file(model_path):
            raise PreparationError("Installed model does not match its manifest hash.")
    return manifest


def prepare_models(data_dir: Path, archives_dir: Path, *, workers: int = 2, range_workers: int = 1,
                   opener=None, progress=None) -> dict:
    """Explicit network operation. Idempotent when an existing manifest verifies."""
    if workers not in {1, 2}:
        raise PreparationError("Use one or two download workers.")
    if range_workers not in {1, 2, 4, 8} or workers * range_workers > 8:
        raise PreparationError("Range workers must be 1/2/4/8, with at most eight total download connections.")
    data_dir, archives_dir = Path(data_dir), Path(archives_dir)
    if (data_dir / "manifest.json").exists():
        return verify_manifest(data_dir)

    def fetch(dataset):
        archive = archives_dir / f"{dataset}_{WEIGHTS_RELEASE}.zip"
        callback = (lambda current, expected: progress(dataset, current, expected)) if progress else None
        result = download_asset(ASSET_URLS[dataset], archive, opener=opener, progress=callback, range_workers=range_workers)
        return dataset, archive, result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        downloads = list(executor.map(fetch, DATASETS))
    sources, artifacts = [], []
    for dataset, archive, source in downloads:
        artifacts.extend(extract_asset(archive, dataset, data_dir))
        sources.append({"dataset": dataset, "url": source["url"], "archive_filename": archive.name,
                        "archive_sha256": source["sha256"], "archive_bytes": source["bytes"]})
    root = data_dir.resolve()
    files = {}
    for dataset in DATASETS:
        for path in model_files(root / "nnUNet" / "results" / WEIGHTS_RELEASE / dataset):
            files[path.relative_to(root).as_posix()] = sha256_file(path)
    manifest = {"schema_version": 1, "provider": "totalspineseg", "package_version": PACKAGE_VERSION,
                "weights_release": WEIGHTS_RELEASE, "files": files, "source_archives": sources,
                "prepared_at": datetime.now(timezone.utc).isoformat(),
                "code_license": "LGPL-3.0", "code_license_url": "https://github.com/neuropoly/totalspineseg/blob/r20260730/LICENSE",
                "weights_license_status": "not_separately_verified", "license_artifacts": artifacts}
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=data_dir, prefix=".manifest-", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, data_dir / "manifest.json")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=runtime_root() / "var/totalspineseg")
    parser.add_argument("--archives-dir", type=Path, default=runtime_root() / "data/totalspineseg/archives")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--range-workers", type=int, choices=(1, 2, 4, 8), default=1,
                        help="Optional parallel HTTP ranges per archive; workers × range-workers must be <= 8. HTTP 206 support is required.")
    parser.add_argument("--verify-only", action="store_true", help="Verify installed model hashes without network access.")
    args = parser.parse_args(argv)
    last_report = {}

    def progress(dataset, received, expected):
        bucket = received // (32 * CHUNK_BYTES)
        if last_report.get(dataset) != bucket:
            last_report[dataset] = bucket
            amount = f"{received / CHUNK_BYTES:.1f} MiB"
            if expected:
                amount += f" / {expected / CHUNK_BYTES:.1f} MiB"
            print(f"{dataset}: {amount}", flush=True)

    try:
        if args.verify_only:
            manifest = verify_manifest(args.data_dir)
        else:
            print(f"Preparing official TotalSpineSeg {WEIGHTS_RELEASE} assets; no package installation or inference.", flush=True)
            manifest = prepare_models(args.data_dir, args.archives_dir, workers=args.workers,
                                      range_workers=args.range_workers, progress=progress)
    except (PreparationError, OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        parser.exit(1, f"Preparation failed: {exc}\n")
    print(json.dumps({"status": "verified", "weights_release": manifest["weights_release"],
                      "manifest": str((args.data_dir / "manifest.json").resolve()),
                      "model_files": len(manifest["files"]), "weights_license_status": manifest["weights_license_status"]}, indent=2))


if __name__ == "__main__":
    main()
