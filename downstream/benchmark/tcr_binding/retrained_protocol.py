"""Official Nature Methods 2025 retrained-checkpoint protocol.

Verify and inspect the downloaded artifacts with::

    source /project_c20250601/251105016/miniforge3/etc/profile.d/conda.sh
    conda activate /project_c20250601/251105016/conda/envs/pllm
    /project_c20250601/251105016/conda/envs/protenix_abtcr/bin/python \
      /project_c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_retrained_baseline.py \
      --verify-only

The protocol never retrains a model.  It pairs each official fold checkpoint
with the exact test member in ``retrain.zip``.  TCR-H is the one published
implementation that also needs its fold's official train CSV at inference time
to reconstruct the correlation-filtered descriptor columns; those labels are
used only for deterministic feature selection, never for fitting parameters.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import zlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pandas as pd


BENCH = Path(__file__).resolve().parents[1]
REPO = BENCH.parents[1]
RAW_ROOT = (
    REPO / "data" / "ppi_task_raw" / "raw" / "nat_methods_tcr_benchmark"
)
RETRAIN_DATA_ZIP = RAW_ROOT / "retrain.zip"
RETRAIN_MODELS_ZIP = RAW_ROOT / "Retraining_model.zip"
MODEL_CACHE = RAW_ROOT / "Retraining_model"
INTEGRITY_CACHE = RAW_ROOT / "retrained_artifact_integrity.json"
OFFICIAL_REPO = BENCH / "baselines" / "TCREpitopeBenchmark"
WRAPPER_DIR = BENCH / "baselines" / "wrappers"

# One runner process evaluates three datasets per fold.  Keep verified large
# nested archives/checkpoints in memory by stable stat tuple so multi-GB RF
# artifacts are not re-read for CRC and SHA on every dataset.
_VERIFIED_FILE_CACHE: dict[Path, tuple[int, int, str]] = {}

T1_CITATION = "doi:10.1038/s41592-025-02910-0"
FIGSHARE_DOI = "https://doi.org/10.6084/m9.figshare.27020455"
SUPPLEMENTARY_TABLE_7 = (
    "https://static-content.springer.com/esm/"
    "art%3A10.1038%2Fs41592-025-02910-0/"
    "MediaObjects/41592_2025_2910_MOESM10_ESM.xlsx"
)

EXPECTED_ARCHIVES = {
    RETRAIN_DATA_ZIP: {
        "size": 20_005_661,
        "md5": "5cf77befd7a07e0cb359540f050fb7b4",
    },
    RETRAIN_MODELS_ZIP: {
        "size": 13_912_594_261,
        "md5": "aa6ea6c175738675692848d036c2244c",
    },
}


@dataclass(frozen=True)
class RetrainedModelSpec:
    """One runnable official retrained baseline implementation."""

    tag: str
    display_name: str
    artifact_dir: str
    checkpoint_template: str
    official_code_dir: Path
    env_python: Path
    infer_script: Path
    infer_kind: str
    auxiliary_artifacts: tuple[Path, ...] = ()
    drop_remainder_batch_size: int | None = None


MODEL_SPECS = {
    "atmtcr": RetrainedModelSpec(
        tag="atmtcr",
        display_name="ATM-TCR",
        artifact_dir="ATM_TCR",
        checkpoint_template="{fold}_1_1.pth",
        official_code_dir=OFFICIAL_REPO / "ATM_TCR",
        env_python=Path(
            "/project_c20250601/251105016/conda/envs/"
            "tcrbench_atmtcr/bin/python"
        ),
        infer_script=WRAPPER_DIR / "atmtcr_official_infer.py",
        infer_kind="atmtcr",
    ),
    "nettcr": RetrainedModelSpec(
        tag="nettcr",
        display_name="NetTCR",
        artifact_dir="NetTCR",
        checkpoint_template="{fold}_1_1model_weights.h5",
        official_code_dir=OFFICIAL_REPO / "NetTCR",
        env_python=Path(
            "/project_c20250601/251105016/conda/envs/"
            "tcrbench_nettcr/bin/python"
        ),
        infer_script=WRAPPER_DIR / "nettcr_official_infer.py",
        infer_kind="nettcr",
    ),
    "epitcr": RetrainedModelSpec(
        tag="epitcr",
        display_name="epiTCR",
        artifact_dir="epiTCR",
        checkpoint_template="{fold}_1_1.pickle",
        official_code_dir=OFFICIAL_REPO / "epiTCR" / "src",
        env_python=Path(
            "/project_c20250601/251105016/conda/envs/"
            "tcrbench_epitcr/bin/python"
        ),
        infer_script=WRAPPER_DIR / "epitcr_official_infer.py",
        infer_kind="epitcr",
    ),
    "teim": RetrainedModelSpec(
        tag="teim",
        display_name="TEIM",
        artifact_dir="TEIM",
        checkpoint_template="{fold}_1_1.pth",
        official_code_dir=OFFICIAL_REPO / "TEIM",
        env_python=Path(
            "/project_c20250601/251105016/conda/envs/"
            "protenix_abtcr/bin/python"
        ),
        infer_script=WRAPPER_DIR / "teim_retrained_official_infer.py",
        infer_kind="teim_retrained",
    ),
    "tcrh": RetrainedModelSpec(
        tag="tcrh",
        display_name="TCR-H",
        artifact_dir="TCR_H",
        checkpoint_template="{fold}_1_1.pkl",
        official_code_dir=OFFICIAL_REPO / "TCR-H",
        env_python=Path(
            "/project_c20250601/251105016/conda/envs/"
            "tcrbench_tcrh/bin/python"
        ),
        infer_script=WRAPPER_DIR / "tcrh_retrained_official_infer.py",
        infer_kind="tcrh_retrained",
    ),
    "teinet": RetrainedModelSpec(
        tag="teinet",
        display_name="TEINet",
        artifact_dir="TEINet",
        checkpoint_template="{fold}_1_1.pth",
        official_code_dir=OFFICIAL_REPO / "TEINet(TEINet-large_TEINet-small)",
        env_python=Path(
            "/project_c20250601/251105016/conda/envs/"
            "tcrbench_torch20/bin/python"
        ),
        infer_script=WRAPPER_DIR / "teinet_official_infer.py",
        infer_kind="teinet",
    ),
    "ergo_ae": RetrainedModelSpec(
        tag="ergo_ae",
        display_name="ERGO-AE",
        artifact_dir="ERGO_AE",
        checkpoint_template="{fold}_1_1.pth",
        official_code_dir=(
            OFFICIAL_REPO
            / "ERGO(ERGO-AE-vdj_ERGO-AE-mc_ERGO-lstm-vdj_ERGO-lstm-mc)"
        ),
        env_python=Path(
            "/project_c20250601/251105016/conda/envs/"
            "tcrbench_torch20/bin/python"
        ),
        infer_script=WRAPPER_DIR / "ergo_official_infer.py",
        infer_kind="ergo_ae_retrained",
        auxiliary_artifacts=(
            OFFICIAL_REPO
            / "ERGO(ERGO-AE-vdj_ERGO-AE-mc_ERGO-lstm-vdj_ERGO-lstm-mc)"
            / "TCR_Autoencoder"
            / "tcr_ae_dim_100.pt",
        ),
        drop_remainder_batch_size=50,
    ),
    "ergo_lstm": RetrainedModelSpec(
        tag="ergo_lstm",
        display_name="ERGO-lstm",
        artifact_dir="ERGO_lstm",
        checkpoint_template="{fold}_1_1.pth",
        official_code_dir=(
            OFFICIAL_REPO
            / "ERGO(ERGO-AE-vdj_ERGO-AE-mc_ERGO-lstm-vdj_ERGO-lstm-mc)"
        ),
        env_python=Path(
            "/project_c20250601/251105016/conda/envs/"
            "tcrbench_torch20/bin/python"
        ),
        infer_script=WRAPPER_DIR / "ergo_official_infer.py",
        infer_kind="ergo_lstm_retrained",
    ),
}

# Supplementary Table 7-1/7-2/7-3, AS rows, CDR3beta-only.  Values are the
# paper's five-fold mean and sample standard deviation, rounded to four digits.
PAPER_EXPECTED_AS = {
    "atmtcr": {
        "seen_test": {"auroc": 0.7728, "auroc_std": 0.0013,
                      "auprc": 0.7800, "auprc_std": 0.0049},
        "seen_independent": {"auroc": 0.6028, "auroc_std": 0.0077,
                             "auprc": 0.5838, "auprc_std": 0.0080},
        "unseen_independent": {"auroc": 0.5192, "auroc_std": 0.0108,
                               "auprc": 0.5136, "auprc_std": 0.0125},
    },
    "nettcr": {
        "seen_test": {"auroc": 0.7372, "auroc_std": 0.0053,
                      "auprc": 0.7482, "auprc_std": 0.0054},
        "seen_independent": {"auroc": 0.5626, "auroc_std": 0.0106,
                             "auprc": 0.5563, "auprc_std": 0.0100},
        "unseen_independent": {"auroc": 0.5203, "auroc_std": 0.0156,
                               "auprc": 0.5156, "auprc_std": 0.0125},
    },
    "epitcr": {
        "seen_test": {"auroc": 0.8142, "auroc_std": 0.0013,
                      "auprc": 0.8345, "auprc_std": 0.0014},
        "seen_independent": {"auroc": 0.6556, "auroc_std": 0.0027,
                             "auprc": 0.6976, "auprc_std": 0.0014},
        "unseen_independent": {"auroc": 0.5075, "auroc_std": 0.0086,
                               "auprc": 0.5106, "auprc_std": 0.0060},
    },
    "teim": {
        "seen_test": {"auroc": 0.7766, "auroc_std": 0.0028,
                      "auprc": 0.7932, "auprc_std": 0.0032},
        "seen_independent": {"auroc": 0.6090, "auroc_std": 0.0071,
                             "auprc": 0.6224, "auprc_std": 0.0088},
        "unseen_independent": {"auroc": 0.5326, "auroc_std": 0.0052,
                               "auprc": 0.5218, "auprc_std": 0.0062},
    },
    "tcrh": {
        "seen_test": {"auroc": 0.6865, "auroc_std": 0.0043,
                      "auprc": 0.6904, "auprc_std": 0.0056},
        "seen_independent": {"auroc": 0.5593, "auroc_std": 0.0059,
                             "auprc": 0.5562, "auprc_std": 0.0071},
        "unseen_independent": {"auroc": 0.5269, "auroc_std": 0.0122,
                               "auprc": 0.5267, "auprc_std": 0.0098},
    },
    "teinet": {
        "seen_test": {"auroc": 0.5770, "auroc_std": 0.0695,
                      "auprc": 0.5887, "auprc_std": 0.0801},
        "seen_independent": {"auroc": 0.5111, "auroc_std": 0.0191,
                             "auprc": 0.5164, "auprc_std": 0.0183},
        "unseen_independent": {"auroc": 0.5025, "auroc_std": 0.0097,
                               "auprc": 0.5057, "auprc_std": 0.0066},
    },
    "ergo_ae": {
        "seen_test": {"auroc": 0.7388, "auroc_std": 0.0033,
                      "auprc": 0.7519, "auprc_std": 0.0035},
        "seen_independent": {"auroc": 0.5636, "auroc_std": 0.0102,
                             "auprc": 0.5711, "auprc_std": 0.0086},
        "unseen_independent": {"auroc": 0.5099, "auroc_std": 0.0088,
                               "auprc": 0.5106, "auprc_std": 0.0024},
    },
    "ergo_lstm": {
        "seen_test": {"auroc": 0.5022, "auroc_std": 0.0027,
                      "auprc": 0.5012, "auprc_std": 0.0005},
        "seen_independent": {"auroc": 0.4987, "auroc_std": 0.0038,
                             "auprc": 0.4981, "auprc_std": 0.0033},
        "unseen_independent": {"auroc": 0.5009, "auroc_std": 0.0011,
                               "auprc": 0.5049, "auprc_std": 0.0049},
    },
}


def _digest_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    return _digest_file(path, "sha256")


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 digest of in-memory artifact bytes."""

    return hashlib.sha256(payload).hexdigest()


def _crc32_file(path: Path) -> int:
    crc = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            crc = zlib.crc32(chunk, crc)
    return crc & 0xFFFFFFFF


def verify_official_archives() -> dict:
    """Strictly MD5-check both Figshare archives, caching stable stat tuples."""

    cached = {}
    if INTEGRITY_CACHE.is_file():
        try:
            cached = json.loads(INTEGRITY_CACHE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cached = {}
    entries = {}
    for path, expected in EXPECTED_ARCHIVES.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing official retrained artifact: {path}")
        stat = path.stat()
        if stat.st_size != expected["size"]:
            raise RuntimeError(
                f"wrong size for {path}: {stat.st_size} != {expected['size']}"
            )
        prior = cached.get("artifacts", {}).get(str(path), {})
        stable = (
            prior.get("size") == stat.st_size
            and prior.get("mtime_ns") == stat.st_mtime_ns
            and prior.get("md5") == expected["md5"]
        )
        actual_md5 = expected["md5"] if stable else _digest_file(path, "md5")
        if actual_md5 != expected["md5"]:
            raise RuntimeError(
                f"wrong MD5 for {path}: {actual_md5} != {expected['md5']}"
            )
        entries[str(path)] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "md5": actual_md5,
            "verified_from_cache": stable,
        }
    result = {
        "source": FIGSHARE_DOI,
        "artifacts": entries,
    }
    INTEGRITY_CACHE.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _member_info(archive: ZipFile, member: str) -> ZipInfo:
    try:
        return archive.getinfo(member)
    except KeyError as exc:
        raise RuntimeError(f"official archive member is missing: {member}") from exc


def _matches_zip_member(path: Path, info: ZipInfo) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == info.file_size
        and _crc32_file(path) == info.CRC
    )


def _extract_member(archive: ZipFile, info: ZipInfo, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    digest = hashlib.sha256()
    try:
        with archive.open(info) as source, partial.open("wb") as target:
            while True:
                chunk = source.read(8 * 1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                digest.update(chunk)
        partial.replace(output)
    except NotImplementedError:
        # Some published nested archives use Deflate64 (method 9), unsupported
        # by Python's zipfile but supported by Info-ZIP.  Invoke it without a
        # shell, then subject the extracted bytes to the same size/CRC/SHA
        # checks as every other member.
        partial.unlink(missing_ok=True)
        archive_path = Path(str(archive.filename))
        extract_root = output.parent
        for _ in Path(info.filename).parts[:-1]:
            extract_root = extract_root.parent
        subprocess.run(
            ["unzip", "-o", str(archive_path), info.filename, "-d", str(extract_root)],
            check=True,
        )
        if not _matches_zip_member(output, info):
            raise RuntimeError(f"unzip fallback produced invalid member: {output}")
        digest = hashlib.sha256()
        with output.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    stat = output.stat()
    _VERIFIED_FILE_CACHE[output] = (
        stat.st_size,
        stat.st_mtime_ns,
        digest.hexdigest(),
    )


def _verified_sha256(path: Path, info: ZipInfo) -> str:
    cached = _VERIFIED_FILE_CACHE.get(path)
    if path.is_file():
        stat = path.stat()
        if cached is not None and cached[:2] == (stat.st_size, stat.st_mtime_ns):
            return cached[2]
    if not _matches_zip_member(path, info):
        raise RuntimeError(f"cached extraction does not match official member: {path}")
    stat = path.stat()
    digest = sha256_file(path)
    _VERIFIED_FILE_CACHE[path] = (stat.st_size, stat.st_mtime_ns, digest)
    return digest


def ensure_checkpoint(
    spec: RetrainedModelSpec,
    *,
    neg_source: str,
    fold: int,
) -> tuple[Path, dict]:
    """Extract and CRC-check one nested official fold checkpoint on demand."""

    nested_member = f"Retraining_model/{spec.artifact_dir}/{neg_source}.zip"
    nested_path = MODEL_CACHE / spec.artifact_dir / f"{neg_source}.zip"
    with ZipFile(RETRAIN_MODELS_ZIP) as outer:
        nested_info = _member_info(outer, nested_member)
        if not nested_path.is_file() or nested_path.stat().st_size != nested_info.file_size:
            _extract_member(outer, nested_info, nested_path)
        nested_sha256 = _verified_sha256(nested_path, nested_info)

    checkpoint_name = spec.checkpoint_template.format(fold=fold)
    checkpoint_member = f"{neg_source}/{checkpoint_name}"
    checkpoint_path = MODEL_CACHE / spec.artifact_dir / neg_source / checkpoint_name
    with ZipFile(nested_path) as nested:
        checkpoint_info = _member_info(nested, checkpoint_member)
        if (
            not checkpoint_path.is_file()
            or checkpoint_path.stat().st_size != checkpoint_info.file_size
        ):
            _extract_member(nested, checkpoint_info, checkpoint_path)
        checkpoint_sha256 = _verified_sha256(checkpoint_path, checkpoint_info)

    metadata = {
        "outer_archive": str(RETRAIN_MODELS_ZIP),
        "outer_archive_md5": EXPECTED_ARCHIVES[RETRAIN_MODELS_ZIP]["md5"],
        "nested_archive_member": nested_member,
        "nested_archive_path": str(nested_path),
        "nested_archive_sha256": nested_sha256,
        "checkpoint_member": checkpoint_member,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "fold": fold,
        "neg_source": neg_source,
        "auxiliary_artifacts": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in spec.auxiliary_artifacts
        ],
    }
    return checkpoint_path, metadata


def _track_fragment(track: str, neg_source: str) -> str:
    if track not in {"cdr3b", "cdr3ab"}:
        raise ValueError(f"unsupported input track: {track}")
    if track == "cdr3ab" and neg_source != "AS":
        raise ValueError("official others track is available with AS negatives only")
    return "only" if track == "cdr3b" else "others"


def resolve_test_member(*, eval_set: str, neg_source: str, fold: int,
                        track: str = "cdr3b") -> str:
    """Resolve an exact beta-only/others test member without mojibake names."""

    category = _track_fragment(track, neg_source)

    if eval_set == "seen_test":
        fragment = f"_{category}_seen/"
        filename = f"{fold}_1_1test.csv"
    elif eval_set == "seen_independent":
        fragment = f"_{category}_seen/"
        filename = "1_1_1independent_test.csv"
    elif eval_set == "unseen_independent":
        fragment = f"_{category}_unseen/"
        filename = "1_1_1independent_test.csv"
    else:
        raise ValueError(f"unsupported eval_set: {eval_set}")
    suffix = f"/{neg_source}/{filename}"
    with ZipFile(RETRAIN_DATA_ZIP) as archive:
        candidates = [
            name for name in archive.namelist()
            if fragment in name and name.endswith(suffix)
        ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one retrain.zip member for {eval_set}/{neg_source}/fold{fold}, "
            f"found {candidates}"
        )
    return candidates[0]


def load_official_test(
    *,
    eval_set: str,
    neg_source: str,
    fold: int,
    track: str = "cdr3b",
) -> tuple[pd.DataFrame, bytes, dict]:
    """Load exact official test bytes and return their provenance."""

    member = resolve_test_member(
        eval_set=eval_set,
        neg_source=neg_source,
        fold=fold,
        track=track,
    )
    with ZipFile(RETRAIN_DATA_ZIP) as archive:
        payload = archive.read(member)
    frame = pd.read_csv(BytesIO(payload)).fillna("")
    required = {"Epitope", "CDR3B", "Affinity"} | ({"CDR3A"} if track == "cdr3ab" else set())
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"official test member lacks columns {sorted(missing)}: {member}")
    metadata = {
        "data_track": track,
        "data_archive": str(RETRAIN_DATA_ZIP),
        "data_archive_md5": EXPECTED_ARCHIVES[RETRAIN_DATA_ZIP]["md5"],
        "test_member": member,
        "test_member_sha256": sha256_bytes(payload),
        "n_rows": int(len(frame)),
        "n_positive": int(frame["Affinity"].astype(int).sum()),
        "n_epitopes": int(frame["Epitope"].astype(str).nunique()),
    }
    return frame, payload, metadata


def resolve_train_member(*, neg_source: str, fold: int, track: str = "cdr3b") -> str:
    """Resolve the official seen fold-train member for the selected track."""

    fragment = f"_{_track_fragment(track, neg_source)}_seen/"
    suffix = f"/{neg_source}/{fold}_1_1train.csv"
    with ZipFile(RETRAIN_DATA_ZIP) as archive:
        candidates = [
            name for name in archive.namelist()
            if fragment in name and name.endswith(suffix)
        ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one retrain.zip train member for {neg_source}/fold{fold}, "
            f"found {candidates}"
        )
    return candidates[0]


def load_official_train(
    *,
    neg_source: str,
    fold: int,
    track: str = "cdr3b",
) -> tuple[pd.DataFrame, bytes, dict]:
    """Load exact official fold-train bytes for inference-time preprocessing."""

    member = resolve_train_member(neg_source=neg_source, fold=fold, track=track)
    with ZipFile(RETRAIN_DATA_ZIP) as archive:
        payload = archive.read(member)
    frame = pd.read_csv(BytesIO(payload)).fillna("")
    required = {"Epitope", "CDR3B", "Affinity"} | ({"CDR3A"} if track == "cdr3ab" else set())
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"official train member lacks columns {sorted(missing)}: {member}")
    metadata = {
        "data_track": track,
        "train_member": member,
        "train_member_sha256": sha256_bytes(payload),
        "n_train_rows": int(len(frame)),
        "n_train_positive": int(frame["Affinity"].astype(int).sum()),
        "n_train_epitopes": int(frame["Epitope"].astype(str).nunique()),
    }
    return frame, payload, metadata


__all__ = [
    "FIGSHARE_DOI",
    "INTEGRITY_CACHE",
    "MODEL_SPECS",
    "PAPER_EXPECTED_AS",
    "RETRAIN_DATA_ZIP",
    "RETRAIN_MODELS_ZIP",
    "SUPPLEMENTARY_TABLE_7",
    "T1_CITATION",
    "ensure_checkpoint",
    "load_official_train",
    "load_official_test",
    "resolve_train_member",
    "resolve_test_member",
    "verify_official_archives",
]
