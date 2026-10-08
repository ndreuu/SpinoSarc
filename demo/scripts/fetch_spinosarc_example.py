#!/usr/bin/env python3
"""Fetch one published lumbar MRI case by byte ranges, without downloading 6 GB.

Source: Sudirman et al., Mendeley Data V2, DOI 10.17632/k57fr854j2.2.
The source DICOM bytes are retained unchanged. CC BY 4.0 attribution is saved.
No API credentials, third-party mirror, synthetic axial data, or model download.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import urllib.request
import zipfile
import zlib
from pathlib import Path
from _demo_paths import REPO_ROOT, runtime_root

ROOT = runtime_root()
SOURCE = "https://data.mendeley.com/datasets/k57fr854j2/2"
ARCHIVE_URL = (
    "https://data.mendeley.com/public-files/datasets/k57fr854j2/"
    "files/eab74360-db27-4ec5-ade3-7b1b8d88e2db/file_downloaded"
)
ARCHIVE_SIZE = 6_270_535_053
ARCHIVE_SHA256 = "ac6495c243c3e95a820b16bc751063e75f7d90e1d3ef22170f925a2448eadc1a"
ATTRIBUTION = (
    "Sudirman, Sud; Al Kafri, Ala; Natalia, Friska; Meidia, Hira; Afriliana, Nunik; "
    "Al-Rashdan, Wasfi; Bashtawi, Mohammad; Al-Jumaily, Mohammed (2019), "
    "Lumbar Spine MRI Dataset, Mendeley Data, V2, doi:10.17632/k57fr854j2.2"
)
SERIES = ("T2_TSE_SAG_384_0002", "T2_TSE_TRA_384_0004", "T1_TSE_TRA_0005")


def fetch_range(url: str, start: int, end: int) -> tuple[bytes, str]:
    request = urllib.request.Request(
        url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "SpinoSarc-open-demo/1"}
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        expected = f"bytes {start}-{end}/{ARCHIVE_SIZE}"
        if response.status != 206 or response.headers.get("Content-Range") != expected:
            raise RuntimeError("Server did not return the exact archive byte range requested.")
        payload = response.read(end - start + 2)
        resolved_url = response.url
    if len(payload) != end - start + 1:
        raise RuntimeError("Archive byte range is incomplete.")
    return payload, resolved_url


def archive_index() -> tuple[list[zipfile.ZipInfo], str]:
    tail_start = ARCHIVE_SIZE - 131_072
    tail, resolved_url = fetch_range(ARCHIVE_URL, tail_start, ARCHIVE_SIZE - 1)
    pos = tail.rfind(b"PK\x06\x06")
    if pos < 0:
        raise RuntimeError("Pinned public archive does not contain the expected ZIP64 directory.")
    header = struct.unpack_from("<4sQ2H2L4Q", tail, pos)
    disk, directory_disk, disk_count, count, size, offset = header[4:]
    if disk or directory_disk or disk_count != count or count > 65_535 or size > 20_000_000:
        raise RuntimeError("Unexpected archive directory layout.")
    directory, _ = fetch_range(resolved_url, offset, offset + size - 1)
    # The directory already stores each original local-header offset. With its
    # synthetic directory offset set to zero, ZipFile preserves those offsets.
    eocd = struct.pack("<4s4H2LH", b"PK\x05\x06", 0, 0, count, count, size, 0, 0)
    with zipfile.ZipFile(io.BytesIO(directory + eocd)) as archive:
        entries = archive.infolist()
    if len(entries) != count:
        raise RuntimeError("Archive directory entry count changed.")
    return entries, resolved_url


def make_manifest(base: Path, provenance: dict) -> dict:
    patient_dir = base / "sudirman-0001"
    return {
        "schema_version": 1,
        "name": "Open lumbar MRI example — Sudirman patient 0001",
        "source": SOURCE,
        "doi": "10.17632/k57fr854j2.2",
        "license": "CC BY 4.0",
        "attribution": ATTRIBUTION,
        "demo_paths": [str((patient_dir / folder).resolve()) for folder in SERIES[:2]],
        "alternative_axial_path": str((patient_dir / SERIES[2]).resolve()),
        "provenance_path": str((patient_dir / "provenance.json").resolve()),
        "native_axial": True,
        "synthetic_axial": False,
        "diagnostic_validation_status": "not_validated",
        "series_alignment_status": "needs_review",
        "alignment_note": (
            "Same StudyInstanceUID; FrameOfReferenceUID differs between series. "
            "Native published DICOM patient coordinates are retained. No registration "
            "was performed. Visually verify corresponding anatomy before interpreting "
            "cross-series measurements."
        ),
        "expected_axial_coverage": "Three lower lumbar disc blocks; numbering requires confirmation.",
        "source_frames": {"sagittal_t2": 15, "axial_t2": 12, "axial_t1": 12},
        "files_verified": len(provenance["files"]),
    }


def verify_existing(base: Path) -> dict | None:
    patient_dir = base / "sudirman-0001"
    provenance_file = patient_dir / "provenance.json"
    if not provenance_file.is_file():
        return None
    provenance = json.loads(provenance_file.read_text())
    if provenance.get("source") != SOURCE or len(provenance.get("files", [])) != 39:
        return None
    for item in provenance["files"]:
        parts = item["filename"].split("/")
        path = patient_dir.joinpath(*parts[3:])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            return None
    return provenance


def download(base: Path) -> dict:
    entries, resolved_url = archive_index()
    prefix = "01_MRI_Data/0001/"
    selected = [
        item for item in entries
        if item.filename.startswith(prefix) and not item.is_dir()
        and item.filename.split("/")[-2] in SERIES
    ]
    if len(selected) != 39:
        raise RuntimeError("Pinned example does not have the expected 39 native DICOM frames.")
    start = min(item.header_offset for item in selected)
    end = max(item.header_offset + 30 + len(item.filename.encode()) + item.compress_size for item in selected) + 511
    payload, _ = fetch_range(resolved_url, start, end)
    patient_dir = base / "sudirman-0001"
    files = []
    for item in selected:
        pos = item.header_offset - start
        header = struct.unpack_from("<4s5H3L2H", payload, pos)
        if header[0] != b"PK\x03\x04" or header[2] & 1:
            raise RuntimeError("Invalid or encrypted ZIP member.")
        local_name = payload[pos + 30:pos + 30 + header[-2]].decode("utf-8")
        if local_name != item.filename:
            raise RuntimeError("ZIP local filename differs from the directory.")
        compressed_start = pos + 30 + header[-2] + header[-1]
        compressed = payload[compressed_start:compressed_start + item.compress_size]
        if item.compress_type == zipfile.ZIP_DEFLATED:
            raw = zlib.decompress(compressed, -15)
        elif item.compress_type == zipfile.ZIP_STORED:
            raw = compressed
        else:
            raise RuntimeError("Unsupported ZIP compression.")
        if len(raw) != item.file_size or zlib.crc32(raw) & 0xFFFFFFFF != item.CRC:
            raise RuntimeError(f"ZIP size/CRC verification failed: {item.filename}")
        relative = Path(*item.filename.split("/")[3:])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Unsafe ZIP member path.")
        path = patient_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        files.append({
            "filename": item.filename,
            "header_offset": item.header_offset,
            "compress_size": item.compress_size,
            "file_size": item.file_size,
            "CRC": item.CRC,
            "local_path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "zip_crc_verified": True,
        })
    provenance = {
        "source": SOURCE, "doi": "10.17632/k57fr854j2.2", "license": "CC BY 4.0",
        "attribution": ATTRIBUTION, "official_archive_url": ARCHIVE_URL,
        "official_archive_size": ARCHIVE_SIZE, "official_archive_sha256": ARCHIVE_SHA256,
        "whole_archive_sha256_verified": False,
        "download_method": "HTTP byte ranges; ZIP CRC verified for every extracted source frame.",
        "patient": "0001", "files": files,
    }
    (patient_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "spinosarc-example")
    parser.add_argument("--verify-only", action="store_true", help="Verify saved source frames without network.")
    args = parser.parse_args()
    base = args.data_dir.resolve()
    provenance = verify_existing(base)
    if provenance is None:
        if args.verify_only:
            raise SystemExit("Published example is missing or its source frame checksums do not match.")
        print("Fetching one CC BY 4.0 published case from official archive byte ranges.", flush=True)
        provenance = download(base)
    else:
        print("Verified all 39 published source DICOM frames; no download needed.", flush=True)
    base.mkdir(parents=True, exist_ok=True)
    manifest = make_manifest(base, provenance)
    (base / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Ready:", base / "manifest.json", flush=True)


if __name__ == "__main__":
    main()
