#!/usr/bin/env python3
"""Build a unified interaction-task CSV from downloaded raw datasets."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import pandas as pd

try:
    import lmdb
except ImportError:  # pragma: no cover - only used when LMDB sources are present.
    lmdb = None


DEFAULT_ROOT = Path(
    "/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw"
)

FIELDNAMES = [
    "record_id",
    "source_id",
    "task_family",
    "dataset_name",
    "split",
    "raw_file",
    "row_index",
    "label",
    "entity_a_id",
    "entity_b_id",
    "sequence_a",
    "sequence_b",
    "sequence_context",
    "tcr",
    "tcr_full",
    "epitope",
    "hla_type",
    "hla_sequence",
    "antibody_heavy",
    "antibody_light",
    "antigen",
    "mutation",
    "value",
    "value_name",
    "measurement_type",
    "pdb_id",
    "chain_info",
    "raw_record_json",
]

INCLUDE_RAW_RECORD_JSON = False

SOURCES = {
    "stringdb_mint": {
        "task_family": "ppi_pretraining",
        "path": "raw/stringdb_mint",
        "source_url": "https://stringdb-downloads.org/download/",
        "notes": "STRING-DB v12.0 physical links/sequences are very large; raw download may remain partial.",
    },
    "figshare_gold_standard": {
        "task_family": "ppi_binary",
        "path": "raw/figshare_gold_standard",
        "source_url": "https://figshare.com/articles/dataset/PPI_prediction_from_sequence_gold_standard_dataset/21591618",
        "notes": "Gold-standard PPI pairs plus SwissProt FASTA.",
    },
    "saprot_humanppi": {
        "task_family": "ppi_binary",
        "path": "raw/saprot_humanppi",
        "source_url": "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/ppidata/human_ppi.zip",
        "notes": "HumanPPI split used by SaProt/MINT.",
    },
    "peer_yeastppi": {
        "task_family": "ppi_binary",
        "path": "raw/peer_yeastppi",
        "source_url": "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/ppidata/yeast_ppi.zip",
        "notes": "YeastPPI split from PEER.",
    },
    "skempi": {
        "task_family": "ppi_mutation_affinity",
        "path": "raw/skempi",
        "source_url": "https://life.bsc.es/pid/skempi2",
        "notes": "SKEMPI v2 CSV and structure tarball.",
    },
    "pdbbind": {
        "task_family": "protein_ligand_binding",
        "path": "raw/pdbbind",
        "source_url": "https://www.pdbbind-plus.org.cn/",
        "notes": "Site API requires login/subscription for dataset download.",
    },
    "swing_mutint": {
        "task_family": "mutational_ppi",
        "path": "raw/swing_mutint",
        "source_url": "https://github.com/jishnu-lab/SWING/tree/main/Data/MutInt_Model",
        "notes": "Mutation interaction perturbation model data.",
    },
    "flab": {
        "task_family": "antibody_binding",
        "path": "raw/flab",
        "source_url": "https://github.com/Graylab/FLAb/tree/main/data",
        "notes": "FLAb antibody binding data under data/binding.",
    },
    "sarscov2_binding_biorxiv": {
        "task_family": "antibody_sarscov2_binding",
        "path": "raw/sarscov2_binding_biorxiv",
        "source_url": "https://www.biorxiv.org/content/10.1101/2020.04.03.024885v1.supplementary-material",
        "notes": "Direct anonymous download returned HTTP 403 in this environment.",
    },
    "tdc_tcr_epitope": {
        "task_family": "tcr_epitope_binding",
        "path": "raw/tdc_tcr_epitope",
        "source_url": "https://tdcommons.ai/",
        "notes": "TDC TCREpitopeBinding/weber exported to CSV.",
    },
    "piste_tcr_epitope_hla": {
        "task_family": "tcr_epitope_hla",
        "path": "raw/piste_tcr_epitope_hla",
        "source_url": "https://github.com/Armilius/PISTE/tree/main/data",
        "notes": "TCR-epitope-HLA splits and references.",
    },
    "teim_interface": {
        "task_family": "tcr_epitope_interface",
        "path": "raw/teim_interface",
        "source_url": "https://github.com/pengxingang/TEIM",
        "notes": "TCR-epitope binding and interface/contact-map data.",
    },
    "oncoppi": {
        "task_family": "oncogenic_ppi",
        "path": "raw/oncoppi",
        "source_url": "https://github.com/ChengF-Lab/oncoPPIs",
        "notes": "Experimentally validated oncoPPI spreadsheets.",
    },
    "covabdab_neutralization": {
        "task_family": "antibody_neutralization",
        "path": "raw/covabdab_neutralization",
        "source_url": "https://opig.stats.ox.ac.uk/webapps/covabdab/",
        "notes": "CoV-AbDab SARS-CoV-2 antibody/neutralization table and annotations.",
    },
}

PARTIAL_SUFFIXES = (".aria2", ".tmp", ".part")
SKIP_SUFFIXES = {
    ".pkl",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".gz",
    ".tgz",
    ".tar",
    ".fa",
    ".fasta",
    ".bib",
    ".html",
    ".js",
}


def clean_value(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value)
    if text == "nan":
        return ""
    return text


def normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")


def normalized_row(row: Mapping[str, object]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, value in row.items():
        out[normalize_key(str(key))] = clean_value(value)
    return out


def first(row: Mapping[str, str], candidates: Iterable[str]) -> Tuple[str, str]:
    for key in candidates:
        norm = normalize_key(key)
        value = row.get(norm, "")
        if value:
            return value, norm
    return "", ""


def compact_json(row: Mapping[str, object]) -> str:
    if not INCLUDE_RAW_RECORD_JSON:
        return ""
    cleaned = {str(k): clean_value(v) for k, v in row.items()}
    return json.dumps(cleaned, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def infer_split(path: Path, dataset_name: str) -> str:
    text = "/".join(path.parts).lower() + "/" + dataset_name.lower()
    for split in ("train", "training", "valid", "validation", "val", "test", "dbpepneo"):
        if re.search(rf"(^|[/_.-]){split}([/_.-]|$)", text):
            return "train" if split == "training" else "valid" if split in {"validation", "val"} else split
    match = re.search(r"intra[0-9]+", text)
    if match:
        return match.group(0)
    return ""


def infer_measurement(path: Path, columns: Iterable[str]) -> str:
    text = f"{path.name} {' '.join(columns)}".lower()
    for token in ("kd", "k_d", "ic50", "ec50", "spr", "binary", "affinity", "neutralization", "y2h"):
        if token in text:
            return token.replace("k_d", "kd")
    return ""


def base_record(
    source_id: str,
    task_family: str,
    dataset_name: str,
    raw_file: str,
    row_index: int,
    split: str,
    raw_row: Mapping[str, object],
) -> Dict[str, str]:
    row = normalized_row(raw_row)
    label, _ = first(
        row,
        [
            "label",
            "Label",
            "Y",
            "target",
            "class",
            "Y2H_score",
            "Growth_score",
            "interaction",
            "bind",
            "binding",
        ],
    )
    entity_a_id, _ = first(
        row,
        [
            "protein1",
            "protein_1",
            "protein a",
            "protein_a",
            "Protein 1",
            "P1",
            "UniProt_ID_a",
            "Target_UPID",
            "target_id",
            "target",
            "id_a",
            "uid_a",
        ],
    )
    entity_b_id, _ = first(
        row,
        [
            "protein2",
            "protein_2",
            "protein b",
            "protein_b",
            "Protein 2",
            "P2",
            "UniProt_ID_b",
            "Interactor_UPID",
            "interactor_id",
            "interactor",
            "id_b",
            "uid_b",
        ],
    )
    sequence_a, _ = first(
        row,
        [
            "Target_Seq",
            "target_seq",
            "target_sequence",
            "sequence_a",
            "seq_a",
            "seq1",
            "primary_1",
            "protein1_sequence",
            "heavy_sequence",
            "vh_sequence",
            "heavy",
            "vh",
            "Mutated_Seq (unless WT)",
        ],
    )
    sequence_b, _ = first(
        row,
        [
            "Interactor_Seq",
            "interactor_seq",
            "interactor_sequence",
            "sequence_b",
            "seq_b",
            "seq2",
            "primary_2",
            "protein2_sequence",
            "antigen_sequence",
            "light_sequence",
            "vl_sequence",
            "light",
            "vl",
        ],
    )
    sequence_context, _ = first(row, ["sequence", "seq", "aa_seq", "peptide_sequence"])
    tcr, _ = first(row, ["tcr", "CDR3", "cdr3", "cdr3b", "cdr3_beta"])
    tcr_full, _ = first(row, ["tcr_full", "full_tcr", "tcr_sequence"])
    epitope, _ = first(row, ["epitope_aa", "epitope", "MT_pep", "peptide", "pep"])
    hla_type, _ = first(row, ["HLA_type", "hla_type", "hla", "mhc"])
    hla_sequence, _ = first(row, ["HLA_sequence", "hla_sequence"])
    antibody_heavy, _ = first(row, ["heavy", "heavy_sequence", "vh", "vh_sequence", "h_seq", "hchain"])
    antibody_light, _ = first(row, ["light", "light_sequence", "vl", "vl_sequence", "l_seq", "lchain"])
    antigen, _ = first(row, ["antigen", "antigen_name", "antigen_sequence", "target_antigen"])
    mutation, _ = first(
        row,
        ["Mutation", "Mutation(s)_PDB", "Mutation(s)_cleaned", "mutation", "mutations", "mut"],
    )
    pdb_id, _ = first(row, ["#Pdb", "pdb", "pdb_id", "structure", "complex"])
    chain_info, _ = first(row, ["pdb_chains", "chains", "chain", "chain_info"])

    value = ""
    value_name = ""
    for candidate in [
        "Affinity_mut_parsed",
        "Affinity_wt_parsed",
        "Kd",
        "KD",
        "kd",
        "IC50",
        "EC50",
        "Y2H_score",
        "Growth_score",
        "value",
        "score",
    ]:
        value, value_name = first(row, [candidate])
        if value:
            break

    return {
        "record_id": "",
        "source_id": source_id,
        "task_family": task_family,
        "dataset_name": dataset_name,
        "split": split,
        "raw_file": raw_file,
        "row_index": str(row_index),
        "label": label,
        "entity_a_id": entity_a_id,
        "entity_b_id": entity_b_id,
        "sequence_a": sequence_a,
        "sequence_b": sequence_b,
        "sequence_context": sequence_context,
        "tcr": tcr,
        "tcr_full": tcr_full,
        "epitope": epitope,
        "hla_type": hla_type,
        "hla_sequence": hla_sequence,
        "antibody_heavy": antibody_heavy,
        "antibody_light": antibody_light,
        "antigen": antigen,
        "mutation": mutation,
        "value": value,
        "value_name": value_name,
        "measurement_type": "",
        "pdb_id": pdb_id,
        "chain_info": chain_info,
        "raw_record_json": compact_json(raw_row),
    }


def parse_fasta_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    seqs: Dict[str, str] = {}
    current_id: Optional[str] = None
    chunks: List[str] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id and chunks:
                    seqs[current_id] = "".join(chunks)
                header = line[1:].split()[0]
                parts = header.split("|")
                current_id = parts[1] if len(parts) >= 2 else parts[0]
                chunks = []
            else:
                chunks.append(line)
        if current_id and chunks:
            seqs[current_id] = "".join(chunks)
    return seqs


def iter_figshare(root: Path) -> Iterator[Dict[str, str]]:
    source_id = "figshare_gold_standard"
    task_family = SOURCES[source_id]["task_family"]
    seqs = parse_fasta_map(root / "raw/figshare_gold_standard/human_swissprot_oneliner.fasta")
    fig_root = root / SOURCES[source_id]["path"]
    for path in sorted(fig_root.glob("Intra*_*.txt")):
        if any(path.name.endswith(suffix) for suffix in PARTIAL_SUFFIXES):
            continue
        label = "1" if "_pos_" in path.name else "0" if "_neg_" in path.name else ""
        split = infer_split(path, path.name)
        with path.open() as handle:
            for row_index, line in enumerate(handle):
                fields = line.strip().split()
                if len(fields) < 2:
                    continue
                a_id, b_id = fields[0], fields[1]
                raw_row = {"protein_a": a_id, "protein_b": b_id, "label": label}
                record = base_record(
                    source_id,
                    task_family,
                    path.stem,
                    str(path),
                    row_index,
                    split,
                    raw_row,
                )
                record["entity_a_id"] = a_id
                record["entity_b_id"] = b_id
                record["sequence_a"] = seqs.get(a_id, "")
                record["sequence_b"] = seqs.get(b_id, "")
                record["measurement_type"] = "binary"
                yield record


def dataframe_records(
    source_id: str,
    task_family: str,
    dataset_name: str,
    raw_file: str,
    frame: pd.DataFrame,
    row_offset: int = 0,
) -> Iterator[Dict[str, str]]:
    split = infer_split(Path(raw_file), dataset_name)
    measurement = infer_measurement(Path(raw_file), frame.columns)
    for local_index, raw_row in enumerate(frame.to_dict(orient="records")):
        if not any(clean_value(value) for value in raw_row.values()):
            continue
        record = base_record(
            source_id,
            task_family,
            dataset_name,
            raw_file,
            row_offset + local_index,
            split,
            raw_row,
        )
        record["measurement_type"] = measurement
        yield record


def iter_delimited_path(
    source_id: str,
    path: Path,
    dataset_name: Optional[str] = None,
    sep: Optional[str] = None,
    task_family: Optional[str] = None,
) -> Iterator[Dict[str, str]]:
    task_family = task_family or SOURCES[source_id]["task_family"]
    dataset_name = dataset_name or path.stem
    if sep is None:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        if source_id == "skempi":
            sep = ";"
    try:
        reader = pd.read_csv(
            path,
            sep=sep,
            chunksize=50000,
            low_memory=False,
            on_bad_lines="skip",
        )
        row_offset = 0
        for chunk in reader:
            for record in dataframe_records(
                source_id,
                task_family,
                dataset_name,
                str(path),
                chunk,
                row_offset,
            ):
                yield record
            row_offset += len(chunk)
    except Exception as exc:
        print(f"skip unreadable delimited file: {path}: {exc}")


def iter_zip_csv(
    source_id: str,
    zip_path: Path,
    include_all: bool = True,
) -> Iterator[Dict[str, str]]:
    task_family = SOURCES[source_id]["task_family"]
    if not zipfile.is_zipfile(zip_path):
        print(f"skip incomplete/non-zip file: {zip_path}")
        return
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for member in sorted(zf.namelist()):
                lower = member.lower()
                if member.endswith("/") or not lower.endswith((".csv", ".tsv", ".txt")):
                    continue
                if not include_all and not lower.endswith((".csv", ".tsv")):
                    continue
                sep = "\t" if lower.endswith(".tsv") else ","
                dataset_name = f"{zip_path.stem}:{member}"
                with zf.open(member) as handle:
                    try:
                        reader = pd.read_csv(
                            handle,
                            sep=sep,
                            chunksize=50000,
                            low_memory=False,
                            on_bad_lines="skip",
                        )
                        row_offset = 0
                        for chunk in reader:
                            for record in dataframe_records(
                                source_id,
                                task_family,
                                dataset_name,
                                f"{zip_path}!{member}",
                                chunk,
                                row_offset,
                            ):
                                yield record
                            row_offset += len(chunk)
                    except Exception as exc:
                        print(f"skip unreadable zip member: {zip_path}!{member}: {exc}")
    except zipfile.BadZipFile:
        print(f"skip bad zip file: {zip_path}")


def iter_xlsx(source_id: str, path: Path) -> Iterator[Dict[str, str]]:
    task_family = SOURCES[source_id]["task_family"]
    try:
        xl = pd.ExcelFile(path)
    except Exception as exc:
        print(f"skip unreadable xlsx file: {path}: {exc}")
        return
    for sheet in xl.sheet_names:
        try:
            frame = pd.read_excel(path, sheet_name=sheet)
        except Exception as exc:
            print(f"skip unreadable xlsx sheet: {path}:{sheet}: {exc}")
            continue
        frame = frame.dropna(how="all")
        if frame.empty:
            continue
        dataset_name = f"{path.stem}:{sheet}"
        for record in dataframe_records(
            source_id,
            task_family,
            dataset_name,
            f"{path}#{sheet}",
            frame,
        ):
            yield record


def iter_lmdb_dir(source_id: str, path: Path) -> Iterator[Dict[str, str]]:
    if lmdb is None:
        print(f"skip LMDB source because python-lmdb is unavailable: {path}")
        return
    task_family = SOURCES[source_id]["task_family"]
    dataset_name = path.name
    split = infer_split(path, dataset_name)
    try:
        env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, max_readers=1)
    except Exception as exc:
        print(f"skip unreadable lmdb dir: {path}: {exc}")
        return
    try:
        with env.begin() as txn:
            cursor = txn.cursor()
            for row_index, (key, value) in enumerate(cursor):
                try:
                    raw_row = pickle.loads(value)
                except Exception as exc:
                    print(f"skip unreadable lmdb record: {path}:{key!r}: {exc}")
                    continue
                if not isinstance(raw_row, dict):
                    raw_row = {"value": raw_row}
                record = base_record(
                    source_id,
                    task_family,
                    dataset_name,
                    str(path),
                    row_index,
                    split,
                    raw_row,
                )
                record["measurement_type"] = "binary"
                yield record
    finally:
        env.close()


def iter_generic_source(source_id: str, root: Path) -> Iterator[Dict[str, str]]:
    raw_root = root / SOURCES[source_id]["path"]
    if not raw_root.exists():
        return
    if source_id == "figshare_gold_standard":
        yield from iter_figshare(root)
        return

    include_paths: List[Path] = []
    if source_id == "swing_mutint":
        include_paths = sorted(raw_root.glob("**/Mutation_perturbation_model.csv"))
    elif source_id == "flab":
        include_paths = sorted((raw_root / "FLAb/data/binding").glob("**/*"))
    elif source_id == "piste_tcr_epitope_hla":
        patterns = [
            "**/train_data.csv",
            "**/val_data.csv",
            "**/test_data.csv",
            "**/dbpepneo_data.csv",
            "**/pos_training_data.csv",
            "**/pos_test_*.csv",
            "**/structure_test.csv",
            "**/example.csv",
        ]
        for pattern in patterns:
            include_paths.extend(raw_root.glob(pattern))
        include_paths = sorted(set(include_paths))
    elif source_id == "teim_interface":
        include_paths = sorted((raw_root / "TEIM/data/binding_data").glob("*.tsv"))
        include_paths.append(raw_root / "TEIM/data/stcrdab_pdb.csv")
    elif source_id == "skempi":
        include_paths = [raw_root / "skempi_v2.csv"]
    elif source_id == "tdc_tcr_epitope":
        include_paths = sorted(raw_root.glob("*.csv"))
    elif source_id == "oncoppi":
        include_paths = sorted(raw_root.glob("**/*.xlsx"))
    elif source_id == "covabdab_neutralization":
        include_paths = sorted(raw_root.glob("*.csv"))
    elif source_id in {"saprot_humanppi", "peer_yeastppi"}:
        include_paths = sorted(raw_root.glob("**/*.lmdb"))
        if not include_paths:
            include_paths = sorted(raw_root.glob("*.zip"))
    else:
        include_paths = []

    for path in include_paths:
        if not path.exists():
            continue
        if path.suffix.lower() == ".lmdb":
            yield from iter_lmdb_dir(source_id, path)
            continue
        if path.is_dir():
            continue
        if any(path.name.endswith(suffix) for suffix in PARTIAL_SUFFIXES):
            continue
        lower = path.name.lower()
        if lower.endswith(".csv.zip") or lower.endswith(".zip"):
            yield from iter_zip_csv(source_id, path)
        elif lower.endswith(".tsv"):
            if source_id == "teim_interface" and path.name == "positive_epi_dist.tsv":
                continue
            yield from iter_delimited_path(source_id, path, sep="\t")
        elif lower.endswith(".csv"):
            yield from iter_delimited_path(source_id, path)
        elif lower.endswith(".xlsx"):
            yield from iter_xlsx(source_id, path)


def build_manifest(root: Path, output_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for source_id, cfg in SOURCES.items():
        path = root / cfg["path"]
        files = [p for p in path.rglob("*") if p.is_file()] if path.exists() else []
        partials = [p for p in files if any(p.name.endswith(suffix) for suffix in PARTIAL_SUFFIXES)]
        completed = [
            p
            for p in files
            if not any(p.name.endswith(suffix) for suffix in PARTIAL_SUFFIXES)
            and not (p.parent / f"{p.name}.aria2").exists()
            and p.suffix.lower() not in {".tmp"}
        ]
        total_bytes = sum(p.stat().st_size for p in completed if p.exists())
        if source_id == "pdbbind":
            status = "blocked"
        elif source_id == "sarscov2_binding_biorxiv":
            status = "blocked"
        elif partials and completed:
            status = "partial"
        elif partials:
            status = "downloading_or_partial"
        elif completed:
            status = "downloaded"
        else:
            status = "missing"
        rows.append(
            {
                "source_id": source_id,
                "task_family": str(cfg["task_family"]),
                "raw_path": str(path),
                "status": status,
                "completed_file_count": str(len(completed)),
                "partial_file_count": str(len(partials)),
                "completed_bytes": str(total_bytes),
                "source_url": str(cfg["source_url"]),
                "notes": str(cfg["notes"]),
            }
        )
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_records(root: Path, output_path: Path) -> Counter:
    counts: Counter = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        record_id = 0
        for source_id in SOURCES:
            for record in iter_generic_source(source_id, root):
                record_id += 1
                record["record_id"] = str(record_id)
                writer.writerow({key: record.get(key, "") for key in FIELDNAMES})
                counts[source_id] += 1
    return counts


def write_summary(counts: Counter, manifest_rows: List[Dict[str, str]], output_path: Path) -> None:
    manifest_by_source = {row["source_id"]: row for row in manifest_rows}
    rows = []
    for source_id in SOURCES:
        manifest = manifest_by_source[source_id]
        rows.append(
            {
                "source_id": source_id,
                "task_family": manifest["task_family"],
                "manifest_status": manifest["status"],
                "record_count": str(counts.get(source_id, 0)),
                "raw_path": manifest["raw_path"],
                "notes": manifest["notes"],
            }
        )
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    global INCLUDE_RAW_RECORD_JSON

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument(
        "--include-raw-record-json",
        action="store_true",
        help="Copy every original input row into raw_record_json. This can make the output very large.",
    )
    args = parser.parse_args()
    INCLUDE_RAW_RECORD_JSON = args.include_raw_record_json

    root = args.root
    processed = args.processed_dir or root / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    manifest_path = processed / "interaction_sources_manifest.csv"
    records_path = processed / "interaction_records_unified.csv"
    summary_path = processed / "interaction_records_summary.csv"

    manifest_rows = build_manifest(root, manifest_path)
    counts = build_records(root, records_path)
    write_summary(counts, manifest_rows, summary_path)

    print(f"wrote {manifest_path}")
    print(f"wrote {records_path}")
    print(f"wrote {summary_path}")
    for source_id, count in counts.most_common():
        print(f"{source_id}: {count}")


if __name__ == "__main__":
    main()
