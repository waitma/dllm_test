#!/usr/bin/env python3
"""Download and verify the external AB/TCR sources used by immune_receptor_v2.

Examples::

    conda run -n pllm python \
      /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/download_immune_receptor_v2_sources.py \
      --sources iedb,catnap,sabdab2
    conda run -n pllm python \
      /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/download_immune_receptor_v2_sources.py \
      --sources iedb,catnap --dry-run

Downloads are staged as ``*.partial``.  An existing artifact is replaced only
after size, checksum (when published), and archive validation pass.  A replaced
artifact is retained as ``*.invalid.<sha256-prefix>`` for provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen


PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
SABDAB2_RECORD = "20083995"
SABDAB2_MD5 = "0dbb4cc499e9eb77f14008b232f2c38c"
SABDAB2_BYTES = 876_381_859
IEDB_EXPORT_API = "https://www.iedb.org/export_data_v3.php"
CATNAP_DOWNLOAD_PAGE = (
    "https://www.hiv.lanl.gov/components/sequence/HIV/neutralization/"
    "download_db.comp"
)


@dataclass(frozen=True)
class Artifact:
    source: str
    url: str
    output: Path
    expected_bytes: int | None = None
    checksum_kind: str | None = None
    checksum: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        default="iedb,catnap,sabdab2",
        help="Comma-separated subset of iedb, catnap, sabdab2.",
    )
    parser.add_argument(
        "--catnap-date",
        default=None,
        help="Pin CATNAP YYYY-MM-DD release; default is latest listed release.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data/immune_receptor_v2/registries/external_download_manifest.json"
        ),
    )
    return parser.parse_args()


def _fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "immune-receptor-v2/1.0"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def _digest(path: Path, kind: str = "sha256") -> str:
    digest = hashlib.new(kind)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iedb_artifacts() -> list[Artifact]:
    payload = json.loads(_fetch(IEDB_EXPORT_API))
    wanted = {"tcell_full_v3.zip", "tcr_full_v3.zip"}
    artifacts: list[Artifact] = []
    for rows in payload["data"].values():
        for row in rows:
            for entry in row:
                title = entry.get("title")
                if title not in wanted:
                    continue
                artifacts.append(
                    Artifact(
                        source="iedb",
                        url=urljoin(IEDB_EXPORT_API, entry["url"]),
                        output=PROJECT_ROOT / "data/tcr" / title,
                    )
                )
    found = {artifact.output.name for artifact in artifacts}
    if found != wanted:
        raise RuntimeError(f"IEDB export API is missing: {sorted(wanted - found)}")
    return sorted(artifacts, key=lambda artifact: artifact.output.name)


def _catnap_artifacts(release_date: str | None) -> list[Artifact]:
    html = _fetch(CATNAP_DOWNLOAD_PAGE).decode("iso-8859-1", "replace")
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.I)
    names = (
        "assay",
        "abs",
        "heavy_seqs_aa",
        "light_seqs_aa",
        "viruses",
        "virseqs_aa",
    )
    candidates: dict[tuple[str, str], tuple[str, str]] = {}
    pattern = re.compile(
        r"/(assay|abs|heavy_seqs_aa|light_seqs_aa|viruses|virseqs_aa)_"
        r"(\d{4}-\d{2}-\d{2})\.(txt|fasta)$"
    )
    for href in hrefs:
        match = pattern.search(href)
        if match:
            stem, date, extension = match.groups()
            candidates[(stem, date)] = (href, extension)
    dates = sorted({date for _, date in candidates})
    if not dates:
        raise RuntimeError("No dated CATNAP download links found")
    selected_date = release_date or dates[-1]
    missing = [name for name in names if (name, selected_date) not in candidates]
    if missing:
        raise RuntimeError(
            f"CATNAP {selected_date} is missing required files: {missing}"
        )
    output_root = PROJECT_ROOT / "data/antibody_raw/catnap" / selected_date
    return [
        Artifact(
            source="catnap",
            url=urljoin(CATNAP_DOWNLOAD_PAGE, candidates[(name, selected_date)][0]),
            output=output_root
            / f"{name}_{selected_date}.{candidates[(name, selected_date)][1]}",
        )
        for name in names
    ]


def _sabdab2_artifacts() -> list[Artifact]:
    api_url = f"https://zenodo.org/api/records/{SABDAB2_RECORD}"
    payload = json.loads(_fetch(api_url))
    files = {item["key"]: item for item in payload.get("files", [])}
    item = files.get("splits.tar.gz")
    if item is None:
        raise RuntimeError("Pinned SAbDab2 Zenodo record lacks splits.tar.gz")
    checksum_kind, checksum = item["checksum"].split(":", 1)
    if item.get("size") != SABDAB2_BYTES or checksum != SABDAB2_MD5:
        raise RuntimeError(
            "Pinned SAbDab2 metadata changed: "
            f"size={item.get('size')} checksum={item.get('checksum')}"
        )
    return [
        Artifact(
            source="sabdab2",
            url=item["links"]["self"],
            output=PROJECT_ROOT / "data/sabdab2_ml/raw/splits.tar.gz",
            expected_bytes=SABDAB2_BYTES,
            checksum_kind=checksum_kind,
            checksum=checksum,
        )
    ]


def _validate(artifact: Artifact, path: Path) -> None:
    if artifact.expected_bytes is not None and path.stat().st_size != artifact.expected_bytes:
        raise RuntimeError(
            f"Size mismatch for {path}: {path.stat().st_size} != "
            f"{artifact.expected_bytes}"
        )
    if artifact.checksum_kind and artifact.checksum:
        observed = _digest(path, artifact.checksum_kind)
        if observed != artifact.checksum:
            raise RuntimeError(
                f"{artifact.checksum_kind} mismatch for {path}: {observed} != "
                f"{artifact.checksum}"
            )
    if path.name.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            if bad_member:
                raise RuntimeError(f"Corrupt ZIP member in {path}: {bad_member}")
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                if member.isfile():
                    extracted = archive.extractfile(member)
                    if extracted is not None:
                        for _ in iter(lambda: extracted.read(8 * 1024 * 1024), b""):
                            pass


def _download(artifact: Artifact, *, dry_run: bool) -> dict:
    output = artifact.output.resolve()
    partial = output.with_name(output.name + ".partial")
    result = {
        "source": artifact.source,
        "url": artifact.url,
        "output": str(output),
    }
    if dry_run:
        return {**result, "status": "dry_run"}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        try:
            _validate(artifact, output)
            Path(str(output) + ".aria2").unlink(missing_ok=True)
            return {
                **result,
                "status": "already_verified",
                "bytes": output.stat().st_size,
                "sha256": _digest(output),
            }
        except (OSError, RuntimeError, tarfile.TarError, zipfile.BadZipFile):
            pass
    downloader = shutil.which("aria2c")
    if downloader:
        command = [
            downloader,
            "--allow-overwrite=true",
            "--auto-file-renaming=false",
            "--check-integrity=true",
            "--continue=true",
            "--file-allocation=none",
            "--max-connection-per-server=4",
            "--split=4",
            "--retry-wait=5",
            "--max-tries=10",
            f"--dir={partial.parent}",
            f"--out={partial.name}",
            artifact.url,
        ]
    else:
        command = [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "10",
            "--retry-delay",
            "5",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            artifact.url,
        ]
    subprocess.run(command, check=True)
    _validate(artifact, partial)
    backup = None
    if output.exists():
        backup = output.with_name(f"{output.name}.invalid.{_digest(output)[:12]}")
        if not backup.exists():
            os.replace(output, backup)
    os.replace(partial, output)
    Path(str(partial) + ".aria2").unlink(missing_ok=True)
    Path(str(output) + ".aria2").unlink(missing_ok=True)
    return {
        **result,
        "status": "downloaded_verified",
        "bytes": output.stat().st_size,
        "sha256": _digest(output),
        "replaced_artifact": str(backup) if backup else None,
    }


def main() -> None:
    args = parse_args()
    selected = {value.strip().lower() for value in args.sources.split(",") if value.strip()}
    unknown = selected - {"iedb", "catnap", "sabdab2"}
    if unknown:
        raise ValueError(f"Unknown sources: {sorted(unknown)}")
    artifacts: list[Artifact] = []
    if "iedb" in selected:
        artifacts.extend(_iedb_artifacts())
    if "catnap" in selected:
        artifacts.extend(_catnap_artifacts(args.catnap_date))
    if "sabdab2" in selected:
        artifacts.extend(_sabdab2_artifacts())
    reports = []
    for artifact in artifacts:
        print(f"[{artifact.source}] {artifact.url} -> {artifact.output}", flush=True)
        report = _download(artifact, dry_run=args.dry_run)
        reports.append(report)
        print(json.dumps(report, sort_keys=True), flush=True)
    manifest = {
        "schema_version": "immune_receptor_external_download.v1",
        "artifacts": reports,
    }
    if not args.dry_run:
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_suffix(report_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, report_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
