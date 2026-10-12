"""Fetch original PALM initialization assets, without altering model code.

Run in conda pllm from the project root:
  python -m downstream.benchmark.ab_antigen.acquire_palm --output /abs/assets

Only the Heavy/Light RoFormer members are fetched from the versioned Zenodo ZIP.
ZIP CRCs and extracted SHA256 are checked; the full archive MD5 is NOT verified.
ESM2 is downloaded at a resolved, recorded Hugging Face revision.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import struct
import zipfile
import zlib

import requests
from huggingface_hub import HfApi, snapshot_download

URL = "https://zenodo.org/api/records/7794583/files/Model.zip/content"
SIZE = 2275427238


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_range(start, end):
    response = requests.get(
        URL, headers={"Range": f"bytes={start}-{end}"}, timeout=(30, 180)
    )
    response.raise_for_status()
    assert response.status_code == 206, "Server must honor byte range"
    assert response.headers["Content-Range"] == f"bytes {start}-{end}/{SIZE}"
    assert len(response.content) == end - start + 1
    return response.content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "acquisition.json"
    if manifest_path.exists():
        report = json.loads(manifest_path.read_text())
        for item in report["files"].values():
            assert sha256(Path(item["path"])) == item["sha256"]
        print("Existing complete asset manifest verified", flush=True)
        return
    tail_start = SIZE - 131072
    archive = zipfile.ZipFile(io.BytesIO(fetch_range(tail_start, SIZE - 1)))
    files = {}
    for member in archive.infolist():
        if member.is_dir() or not any(
            member.filename.startswith("Model_Zenodo/" + name + "/")
            for name in ("Heavy_roformer", "Light_roformer")
        ):
            continue
        dest = out / member.filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        offset = member.header_offset + tail_start
        header = fetch_range(offset, offset + 29)
        assert header[:4] == b"PK\x03\x04"
        name_size, extra_size = struct.unpack_from("<HH", header, 26)
        body_start = offset + 30 + name_size + extra_size
        print("Downloading", member.filename, member.compress_size, flush=True)
        compressed = fetch_range(body_start, body_start + member.compress_size - 1)
        assert member.compress_type in (zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED)
        data = zlib.decompress(compressed, -15) if member.compress_type == zipfile.ZIP_DEFLATED else compressed
        assert len(data) == member.file_size
        assert zlib.crc32(data) & 0xFFFFFFFF == member.CRC
        temp = dest.with_name(dest.name + ".part")
        temp.write_bytes(data)
        temp.replace(dest)
        files[member.filename] = {
            "path": str(dest), "sha256": sha256(dest), "size": len(data),
            "zip_crc32": member.CRC, "source": URL, "zip_member": member.filename,
        }
    model_id = "facebook/esm2_t30_150M_UR50D"
    revision = HfApi().model_info(model_id).sha
    esm_dir = out / "esm2_t30_150M_UR50D"
    snapshot_download(
        model_id, revision=revision, local_dir=str(esm_dir),
        allow_patterns=["config.json", "pytorch_model.bin", "tokenizer_config.json",
                        "special_tokens_map.json", "vocab.txt"], max_workers=4,
    )
    for file in sorted(esm_dir.iterdir()):
        if file.is_file():
            files["esm2/" + file.name] = {
                "path": str(file), "sha256": sha256(file), "size": file.stat().st_size,
                "source": model_id, "revision": revision,
            }
    report = {
        "acquired_at": datetime.now(timezone.utc).isoformat(), "files": files,
        "zenodo_record": "7794583", "archive_url": URL,
        "archive_md5_published": "8c6e74800811a44547036ee6419c4de2",
        "full_archive_md5_verified": False, "member_crc32_verified": True,
        "esm2_revision": revision, "model_loaded": False,
    }
    manifest_path.write_text(json.dumps(report, indent=2) + "\n")
    print("Saved", manifest_path, flush=True)


if __name__ == "__main__":
    main()
