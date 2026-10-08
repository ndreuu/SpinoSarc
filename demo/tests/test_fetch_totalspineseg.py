"""Synthetic archives exercise bounded downloads/extraction; no network/model."""

import io
import json
from pathlib import Path
import re
import stat
import struct
import sys
import threading
from types import SimpleNamespace
import zipfile
import urllib.error

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from demo.scripts import fetch_totalspineseg as module


def model_archive(dataset, *, extra=None, checkpoint="checkpoint_best.pth"):
    memory = io.BytesIO()
    folder = f"{dataset}/Trainer__Plans__3d_fullres"
    with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(f"{folder}/fold_0/{checkpoint}", b"SYNTHETIC TRANSPORT TEST; NOT A MODEL")
        archive.writestr(f"{folder}/plans.json", "{}")
        archive.writestr(f"{folder}/dataset.json", "{}")
        if extra:
            for info, content in extra:
                archive.writestr(info, content)
    return memory.getvalue()


class Response(io.BytesIO):
    def __init__(self, content, headers=None, status=200, url="https://test.invalid/fixture.zip"):
        super().__init__(content)
        self.status = status
        self.headers = headers if headers is not None else {"Content-Length": str(len(content))}
        self.url = url
        self.read_count = 0

    def geturl(self):
        return self.url

    def read(self, *args):
        self.read_count += 1
        return super().read(*args)


def test_download_is_atomic_and_cache_reuse_avoids_network(tmp_path):
    content = model_archive(module.DATASETS[0])
    path = tmp_path / "cached.zip"
    result = module.download_asset(module.ASSET_URLS[module.DATASETS[0]], path, opener=lambda *args, **kwargs: Response(content))
    assert path.read_bytes() == content
    assert result["cached"] is False
    assert result["sha256"] == module.sha256_file(path)
    cached = module.download_asset(result["url"], path, opener=lambda *args, **kwargs: pytest.fail("Cache must avoid network"))
    assert cached["cached"] is True
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("headers,status", [
    ({"Content-Length": "1000"}, 200),
    ({"Content-Length": "invalid"}, 200),
    ({"Content-Encoding": "gzip"}, 200),
    ({}, 206),
])
def test_bad_headers_rejected_before_body_read(tmp_path, headers, status):
    response = Response(b"body", headers, status)
    with pytest.raises(module.PreparationError):
        module.download_asset("https://test.invalid/a", tmp_path / "cache.zip", max_bytes=100,
                              opener=lambda *args, **kwargs: response)
    assert response.read_count == 0
    assert not (tmp_path / "cache.zip").exists()


@pytest.mark.parametrize("content,headers", [
    (b"x" * 101, {}),
    (b"short", {"Content-Length": "50"}),
    (b"not a zip", {}),
])
def test_invalid_bodies_leave_no_completed_or_partial_file(tmp_path, content, headers):
    with pytest.raises(module.PreparationError):
        module.download_asset("https://test.invalid/a", tmp_path / "cache.zip", max_bytes=100,
                              opener=lambda *args, **kwargs: Response(content, headers))
    assert not (tmp_path / "cache.zip").exists()
    assert not list(tmp_path.glob("*.partial"))
    assert not list(tmp_path.glob(".*.partial"))


def test_insecure_redirect_rejected(tmp_path):
    response = Response(b"content", url="http://test.invalid/insecure")
    with pytest.raises(module.PreparationError, match="HTTPS"):
        module.download_asset("https://test.invalid/a", tmp_path / "cache.zip", opener=lambda *args, **kwargs: response)


@pytest.mark.parametrize("entry", ["../outside", "/absolute", "C:/drive", "dir\\outside", "."])
def test_archive_path_traversal_rejected_before_install(tmp_path, entry):
    archive = tmp_path / "unsafe.zip"
    archive.write_bytes(model_archive(module.DATASETS[0], extra=[(entry, b"bad")]))
    with pytest.raises(module.PreparationError, match="path"):
        module.extract_asset(archive, module.DATASETS[0], tmp_path / "models")
    assert not (tmp_path / "outside").exists()
    assert not (tmp_path / "models" / "nnUNet" / "results" / module.WEIGHTS_RELEASE / module.DATASETS[0]).exists()


def test_symlink_archive_rejected(tmp_path):
    info = zipfile.ZipInfo("unsafe-link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive = tmp_path / "symlink.zip"
    archive.write_bytes(model_archive(module.DATASETS[0], extra=[(info, b"/private/etc/passwd")]))
    with pytest.raises(module.PreparationError, match="Symlinks"):
        module.extract_asset(archive, module.DATASETS[0], tmp_path / "models")


def test_archive_resource_limits(tmp_path, monkeypatch):
    archive = tmp_path / "model.zip"
    archive.write_bytes(model_archive(module.DATASETS[0]))
    monkeypatch.setattr(module, "MAX_ENTRIES", 2)
    with pytest.raises(module.PreparationError, match="entries"):
        module.extract_asset(archive, module.DATASETS[0], tmp_path / "models")
    monkeypatch.setattr(module, "MAX_ENTRIES", 5000)
    monkeypatch.setattr(module, "MAX_UNPACKED_BYTES", 10)
    with pytest.raises(module.PreparationError, match="unpacked"):
        module.extract_asset(archive, module.DATASETS[0], tmp_path / "models")


@pytest.mark.parametrize("size", [1_129_915_218, 1_130_070_162])
def test_pinned_checkpoint_member_sizes_fit_without_allocating_pixels_or_weights(size):
    # Metadata-only fixture: exercise the pre-extraction budget for observed
    # official sizes without manufacturing a 1.1 GB file or running a model.
    info = zipfile.ZipInfo(f"{module.DATASETS[0]}/Trainer__Plans__3d_fullres/fold_0/checkpoint_best.pth")
    info.file_size = size
    info.compress_size = 1_050_590_309
    archive = SimpleNamespace(infolist=lambda: [info])
    members = module._archive_members(archive)
    assert len(members) == 1
    assert members[0][0].file_size > 1024 * 1024 * 1024
    assert module.MAX_ARCHIVE_BYTES == 1024 * 1024 * 1024
    assert module.MAX_UNPACKED_BYTES == 2 * 1024 * 1024 * 1024


def test_member_above_one_and_half_gib_rejected_before_extraction():
    info = zipfile.ZipInfo("oversized-checkpoint.pth")
    info.file_size = 1536 * 1024 * 1024 + 1
    info.compress_size = 1024 * 1024 * 1024
    archive = SimpleNamespace(infolist=lambda: [info])
    with pytest.raises(module.PreparationError, match="member exceeds"):
        module._archive_members(archive)


def test_total_unpacked_budget_still_rejects_two_large_members():
    infos = []
    for name in ("checkpoint_best.pth", "checkpoint_final.pth"):
        info = zipfile.ZipInfo(name)
        info.file_size = 1_130_070_162
        info.compress_size = 1_050_590_309
        infos.append(info)
    with pytest.raises(module.PreparationError, match="archive exceeds the unpacked"):
        module._archive_members(SimpleNamespace(infolist=lambda: infos))


def test_zip_crc_is_checked_and_model_not_installed(tmp_path):
    content = bytearray(model_archive(module.DATASETS[0]))
    central = content.index(b"PK\x01\x02")
    struct.pack_into("<L", content, central + 16, 0)
    archive = tmp_path / "corrupt.zip"
    archive.write_bytes(content)
    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        module.extract_asset(archive, module.DATASETS[0], tmp_path / "models")
    assert not (tmp_path / "models" / "nnUNet" / "results" / module.WEIGHTS_RELEASE / module.DATASETS[0]).exists()


def test_prepare_writes_adapter_compatible_manifest_and_verifies_it(tmp_path):
    def opener(request, **kwargs):
        dataset = next(dataset for dataset in module.DATASETS if dataset in request.full_url)
        return Response(model_archive(dataset, extra=[(f"{dataset}/LICENSE", "test license artifact; no claim about rights")]))

    data = tmp_path / "data"
    manifest = module.prepare_models(data, tmp_path / "archives", opener=opener)
    assert manifest["provider"] == "totalspineseg"
    assert manifest["package_version"] == "20260730"
    assert manifest["weights_release"] == "r20260730"
    assert len(manifest["files"]) == 6
    assert len(manifest["source_archives"]) == 2
    assert len(manifest["license_artifacts"]) == 2
    assert manifest["code_license"] == "LGPL-3.0"
    assert manifest["weights_license_status"] == "not_separately_verified"
    assert module.verify_manifest(data) == manifest
    assert module.prepare_models(data, tmp_path / "archives", opener=lambda *args, **kwargs: pytest.fail("Verified installation must avoid network")) == manifest
    checkpoint = next(data.rglob("checkpoint_best.pth"))
    checkpoint.write_bytes(b"changed")
    with pytest.raises(module.PreparationError, match="manifest hash"):
        module.verify_manifest(data)


def test_checkpoint_final_fallback_and_nonoverwriting_install(tmp_path):
    archive = tmp_path / "model.zip"
    archive.write_bytes(model_archive(module.DATASETS[0], checkpoint="checkpoint_final.pth"))
    data = tmp_path / "data"
    module.extract_asset(archive, module.DATASETS[0], data)
    module.extract_asset(archive, module.DATASETS[0], data)
    checkpoint = next(data.rglob("checkpoint_final.pth"))
    checkpoint.write_bytes(b"different existing model")
    with pytest.raises(module.PreparationError, match="Existing model differs"):
        module.extract_asset(archive, module.DATASETS[0], data)
    assert checkpoint.read_bytes() == b"different existing model"


def test_verify_only_uses_no_network(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    module.prepare_models(data, tmp_path / "archives", opener=lambda request, **kwargs: Response(model_archive(next(dataset for dataset in module.DATASETS if dataset in request.full_url))))
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("Verify-only must avoid network"))
    module.main(["--data-dir", str(data), "--verify-only"])
    assert json.loads(capsys.readouterr().out)["status"] == "verified"


def range_transport(content, *, change_headers=None, change_body=None, status=206):
    calls = []

    def opener(request, **kwargs):
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range", ""))
        assert match is not None
        start, end = map(int, match.groups())
        calls.append((start, end, request))
        headers = {"Content-Range": f"bytes {start}-{end}/{len(content)}",
                   "Content-Length": str(end - start + 1), "ETag": '"fixture-source"'}
        body = content[start:end + 1]
        if change_headers:
            change_headers(start, end, headers)
        if change_body:
            body = change_body(start, end, body)
        return Response(body, headers, status=status)

    return opener, calls


def test_parallel_range_download_assembles_disjoint_slots_atomically(tmp_path, monkeypatch):
    content = model_archive(module.DATASETS[0])
    monkeypatch.setattr(module, "RANGE_CHUNK_BYTES", 64)
    opener, calls = range_transport(content)
    progress = []
    destination = tmp_path / "range.zip"
    result = module.download_asset("https://test.invalid/model", destination, opener=opener,
                                   range_workers=4, progress=lambda current, total: progress.append((current, total)))
    assert destination.read_bytes() == content
    assert result["bytes"] == len(content)
    assert result["range_workers"] == 4
    assert result["range_requests"] == len(calls)
    assert calls[0][0:2] == (0, 0)
    intervals = sorted((start, end) for start, end, _ in calls[1:])
    assert intervals[0][0] == 0 and intervals[-1][1] == len(content) - 1
    assert all(right_start == left_end + 1 for (_, left_end), (right_start, _) in zip(intervals, intervals[1:]))
    assert progress[-1] == (len(content), len(content))
    assert all(a[0] <= b[0] for a, b in zip(progress, progress[1:]))
    assert all(request.get_header("If-match") == '"fixture-source"' for _, _, request in calls[1:])
    assert not list(tmp_path.glob(".*.partial"))


def test_range_http_200_rejected_before_any_body_read(tmp_path):
    response = Response(b"full archive must not be read", status=200)
    with pytest.raises(module.PreparationError, match="HTTP 206"):
        module.download_asset("https://test.invalid/model", tmp_path / "cache.zip", range_workers=4,
                              opener=lambda *args, **kwargs: response)
    assert response.read_count == 0
    assert not list(tmp_path.glob(".*.partial"))


@pytest.mark.parametrize("change", [
    lambda start, end, h: h.update({"Content-Range": f"bytes {start + 1}-{end}/100"}),
    lambda start, end, h: h.update({"Content-Range": "missing"}),
    lambda start, end, h: h.update({"Content-Encoding": "gzip"}),
    lambda start, end, h: h.update({"Content-Length": "wrong"}),
    lambda start, end, h: h.update({"Content-Length": "2"}),
])
def test_invalid_range_probe_headers_rejected(tmp_path, change):
    opener, _ = range_transport(model_archive(module.DATASETS[0]), change_headers=change)
    with pytest.raises(module.PreparationError):
        module.download_asset("https://test.invalid/model", tmp_path / "cache.zip", range_workers=2, opener=opener)
    assert not list(tmp_path.glob(".*.partial"))


@pytest.mark.parametrize("kind", ["changed_total", "changed_etag", "truncated", "extra"])
def test_failed_range_slot_removes_partial_and_never_creates_cache(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(module, "RANGE_CHUNK_BYTES", 64)

    def headers(start, end, value):
        if end > 0:
            if kind == "changed_total":
                value["Content-Range"] = f"bytes {start}-{end}/9999"
            elif kind == "changed_etag":
                value["ETag"] = '"other-resource"'

    def body(start, end, value):
        if end > 0:
            if kind == "truncated":
                return value[:-1]
            if kind == "extra":
                return value + b"x"
        return value

    opener, _ = range_transport(model_archive(module.DATASETS[0]), change_headers=headers, change_body=body)
    with pytest.raises(module.PreparationError):
        module.download_asset("https://test.invalid/model", tmp_path / "cache.zip", range_workers=4, opener=opener)
    assert not (tmp_path / "cache.zip").exists()
    assert not list(tmp_path.glob(".*.partial"))


def test_range_probe_enforces_archive_total_limit(tmp_path):
    opener, _ = range_transport(model_archive(module.DATASETS[0]))
    with pytest.raises(module.PreparationError, match="bounded range"):
        module.download_asset("https://test.invalid/model", tmp_path / "cache.zip", max_bytes=100,
                              range_workers=4, opener=opener)


def test_range_global_connection_limit_and_manifest_integration(tmp_path, monkeypatch):
    with pytest.raises(module.PreparationError, match="eight total"):
        module.prepare_models(tmp_path / "models", tmp_path / "archives", workers=2, range_workers=8,
                              opener=lambda *args, **kwargs: pytest.fail("Invalid configuration must not access network"))
    monkeypatch.setattr(module, "RANGE_CHUNK_BYTES", 128)

    def opener(request, **kwargs):
        dataset = next(dataset for dataset in module.DATASETS if dataset in request.full_url)
        transport, _ = range_transport(model_archive(dataset))
        return transport(request, **kwargs)

    result = module.prepare_models(tmp_path / "models", tmp_path / "archives", workers=2, range_workers=4, opener=opener)
    assert len(result["files"]) == 6
    assert module.verify_manifest(tmp_path / "models") == result


def test_range_probe_requires_https_and_exact_one_byte_body(tmp_path):
    for body, url in [(b"two", "https://test.invalid/model"), (b"x", "http://test.invalid/model")]:
        response = Response(body, {"Content-Range": "bytes 0-0/100", "Content-Length": "1"}, status=206, url=url)
        with pytest.raises(module.PreparationError):
            module.download_asset("https://test.invalid/model", tmp_path / "cache.zip", range_workers=2,
                                  opener=lambda *args, **kwargs: response)


def test_truncated_range_is_retried_without_caching_partial_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "RANGE_RETRY_BACKOFF", (0, 0, 0))
    content = model_archive(module.DATASETS[0])
    attempts = 0

    def body(start, end, value):
        nonlocal attempts
        if end > 0:
            attempts += 1
            if attempts == 1:
                return value[:-1]
        return value

    opener, _ = range_transport(content, change_body=body)
    path = tmp_path / "cache.zip"
    module.download_asset("https://test.invalid/model", path, range_workers=2, opener=opener)
    assert attempts == 2
    assert path.read_bytes() == content
    assert not list(tmp_path.glob(".*.partial"))


def create_failed_resumable_download(tmp_path, monkeypatch):
    """Complete some chunks, then fail one chunk with four transient errors."""
    monkeypatch.setattr(module, "RANGE_CHUNK_BYTES", 64)
    monkeypatch.setattr(module, "RANGE_RETRY_BACKOFF", (0, 0, 0))
    content = model_archive(module.DATASETS[0])
    transport, calls = range_transport(content)
    completed = threading.Event()
    failure_calls = []
    last_start = (len(content) - 1) // 64 * 64

    def opener(request, **kwargs):
        start = int(request.get_header("Range").split("=")[1].split("-")[0])
        if start == last_start:
            assert completed.wait(1), "A prior chunk must finish before the synthetic failure."
            failure_calls.append(start)
            raise TimeoutError("synthetic transient network timeout")
        return transport(request, **kwargs)

    destination = tmp_path / "resumable.zip"
    with pytest.raises(TimeoutError):
        module.download_asset("https://test.invalid/model", destination, range_workers=2, opener=opener,
                              progress=lambda current, total: completed.set() if current else None)
    assert len(failure_calls) == 4
    cache = tmp_path / ".resumable.zip.ranges"
    record = json.loads((cache / "source.json").read_text())
    assert 0 < len(record["chunks"]) < (len(content) + 63) // 64
    assert not list(cache.glob("*.partial"))
    assert not destination.exists()
    for filename, value in record["chunks"].items():
        assert module.sha256_file(cache / filename) == value["sha256"]
    return content, destination, cache, record


def test_completed_chunks_resume_after_network_failure(tmp_path, monkeypatch):
    content, destination, cache, record = create_failed_resumable_download(tmp_path, monkeypatch)
    opener, calls = range_transport(content)
    result = module.download_asset("https://test.invalid/model", destination, range_workers=2, opener=opener)
    assert destination.read_bytes() == content
    assert result["resumed_chunks"] == len(record["chunks"])
    requested = {(start, end) for start, end, _ in calls[1:]}
    for filename in record["chunks"]:
        start, end = map(int, filename.removesuffix(".chunk").split("-"))
        assert (start, end) not in requested
    assert calls[0][0:2] == (0, 0), "Every resumed run must revalidate the resource."
    assert not cache.exists(), "Complete archive replaces the temporary chunk cache."


def test_changed_resource_validator_invalidates_completed_chunk_cache(tmp_path, monkeypatch):
    content, destination, cache, record = create_failed_resumable_download(tmp_path, monkeypatch)
    opener, calls = range_transport(content, change_headers=lambda start, end, h: h.update(ETag='"new-resource"'))
    result = module.download_asset("https://test.invalid/model", destination, range_workers=2, opener=opener)
    assert result["resumed_chunks"] == 0
    requested = {(start, end) for start, end, _ in calls[1:]}
    assert all(tuple(map(int, name.removesuffix(".chunk").split("-"))) in requested for name in record["chunks"])


def test_corrupted_completed_chunk_is_replaced_on_resume(tmp_path, monkeypatch):
    content, destination, cache, record = create_failed_resumable_download(tmp_path, monkeypatch)
    filename = next(iter(record["chunks"]))
    path = cache / filename
    value = path.read_bytes()
    path.write_bytes(bytes([value[0] ^ 1]) + value[1:])
    opener, calls = range_transport(content)
    result = module.download_asset("https://test.invalid/model", destination, range_workers=2, opener=opener)
    assert result["resumed_chunks"] == len(record["chunks"]) - 1
    wanted = tuple(map(int, filename.removesuffix(".chunk").split("-")))
    assert wanted in [(start, end) for start, end, _ in calls[1:]]
    assert destination.read_bytes() == content


@pytest.mark.parametrize("kind", ["timeout", "urlerror", "429", "503"])
def test_only_transient_network_failures_receive_four_bounded_attempts(monkeypatch, kind):
    monkeypatch.setattr(module, "RANGE_RETRY_BACKOFF", (0, 0, 0))
    calls = []

    def operation():
        calls.append(1)
        if len(calls) == 4:
            return "completed"
        if kind == "timeout":
            raise TimeoutError("transient")
        if kind == "urlerror":
            raise urllib.error.URLError("transient")
        raise urllib.error.HTTPError("https://test.invalid/model", int(kind), "transient", {}, io.BytesIO())

    assert module._retry_range_network(operation, threading.Event()) == "completed"
    assert len(calls) == 4


@pytest.mark.parametrize("kind", ["protocol", "404", "filesystem"])
def test_protocol_and_permanent_errors_are_not_retried(monkeypatch, kind):
    monkeypatch.setattr(module, "RANGE_RETRY_BACKOFF", (0, 0, 0))
    calls = []

    def operation():
        calls.append(1)
        if kind == "protocol":
            raise module.PreparationError("HTTP 200 range response")
        if kind == "404":
            raise urllib.error.HTTPError("https://test.invalid/model", 404, "permanent", {}, io.BytesIO())
        raise OSError("local disk is full")

    with pytest.raises((module.PreparationError, OSError)):
        module._retry_range_network(operation, threading.Event())
    assert len(calls) == 1


def test_unsafe_range_cache_entries_cannot_reference_external_files(tmp_path, monkeypatch):
    content = model_archive(module.DATASETS[0])
    cache = tmp_path / ".cache.zip.ranges"
    cache.mkdir()
    external = tmp_path / "external.json"
    external.write_text("do not read or modify")
    (cache / "source.json").symlink_to(external)
    opener, _ = range_transport(content)
    with pytest.raises(module.PreparationError, match="unsafe entries"):
        module.download_asset("https://test.invalid/model", tmp_path / "cache.zip", range_workers=2, opener=opener)
    assert external.read_text() == "do not read or modify"


def test_range_chunk_requests_use_longer_http_timeout(tmp_path, monkeypatch):
    transport, _ = range_transport(model_archive(module.DATASETS[0]))
    timeouts = []

    def opener(request, **kwargs):
        timeouts.append(kwargs["timeout"])
        return transport(request, **kwargs)

    module.download_asset("https://test.invalid/model", tmp_path / "cache.zip", range_workers=2, opener=opener)
    assert timeouts and set(timeouts) == {180}
