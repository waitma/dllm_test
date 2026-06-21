"""Rebuild a SAbDab murine humanization candidate split for Ophiuchus-Ab eval.

Example:
    python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/rebuild_igcraft_humanization_split.py \
        --summary-tsv /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/antibody/sabdab_summary.tsv \
        --output-dir /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ophiuchus_ab/igcraft_rebuild \
        --download-pdbs
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from abnumber import Chain


@dataclass(frozen=True)
class FastaRecord:
    header: str
    sequence: str
    label_chain_ids: tuple[str, ...]
    auth_chain_ids: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild a non-redundant paired murine SAbDab humanization split."
    )
    parser.add_argument("--summary-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-date", type=str, default="2024-02-01")
    parser.add_argument("--end-date", type=str, default="2025-03-25")
    parser.add_argument("--target-count", type=int, default=27)
    parser.add_argument("--known-pdbs", type=str, default="8tfh,8txu,8tvh")
    parser.add_argument(
        "--selection-mode",
        choices=["earliest", "known_augmented"],
        default="known_augmented",
        help="known_augmented keeps known paper-example PDBs, then fills by date/resolution.",
    )
    parser.add_argument("--download-pdbs", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def fetch_url_text(url: str, timeout: float) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def cached_text(cache_path: Path, url: str, timeout: float) -> str:
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_text()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    text = fetch_url_text(url, timeout=timeout)
    cache_path.write_text(text)
    return text


def parse_chain_tokens(header: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    parts = header.split("|")
    chain_text = ""
    for part in parts:
        if part.startswith("Chain "):
            chain_text = part.removeprefix("Chain ")
            break
        if part.startswith("Chains "):
            chain_text = part.removeprefix("Chains ")
            break
    label_ids: list[str] = []
    auth_ids: list[str] = []
    for token in chain_text.split(","):
        token = token.strip()
        if not token:
            continue
        match = re.match(r"(?P<label>[^\[\]\s,]+)(?:\[auth\s+(?P<auth>[^\]]+)\])?", token)
        if not match:
            continue
        label_id = match.group("label").strip()
        auth_id = (match.group("auth") or "").strip()
        if label_id:
            label_ids.append(label_id)
        if auth_id:
            auth_ids.append(auth_id)
    return tuple(label_ids), tuple(auth_ids)


def parse_fasta(text: str) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    header: str | None = None
    seq_parts: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                label_ids, auth_ids = parse_chain_tokens(header)
                records.append(FastaRecord(header, "".join(seq_parts), label_ids, auth_ids))
            header = line[1:]
            seq_parts = []
        else:
            seq_parts.append(line)
    if header is not None:
        label_ids, auth_ids = parse_chain_tokens(header)
        records.append(FastaRecord(header, "".join(seq_parts), label_ids, auth_ids))
    return records


def fasta_maps(records: list[FastaRecord]) -> tuple[dict[str, FastaRecord], dict[str, FastaRecord]]:
    label_map: dict[str, FastaRecord] = {}
    auth_map: dict[str, FastaRecord] = {}
    for record in records:
        for chain_id in record.label_chain_ids:
            label_map.setdefault(chain_id, record)
        for chain_id in record.auth_chain_ids:
            auth_map.setdefault(chain_id, record)
    return label_map, auth_map


def resolve_record(
    chain_id: str,
    label_map: dict[str, FastaRecord],
    auth_map: dict[str, FastaRecord],
) -> FastaRecord:
    # SAbDab chain IDs usually refer to author chains. Prefer auth IDs because
    # RCSB label chain IDs can collide with unrelated antigen chains.
    if chain_id in auth_map:
        return auth_map[chain_id]
    if chain_id in label_map:
        return label_map[chain_id]
    raise KeyError(chain_id)


def parse_resolution(value) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return math.inf
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.inf


def canonical_species(value) -> str:
    return str(value).strip().lower()


def is_nonempty(value) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    value_str = str(value).strip()
    return bool(value_str) and value_str.lower() not in {"nan", "na", "none"}


def candidate_rows(summary: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    df = summary.copy()
    df["pdb"] = df["pdb"].astype(str).str.lower()
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    mask = (
        df["date_parsed"].between(start, end, inclusive="both")
        & df["Hchain"].map(is_nonempty)
        & df["Lchain"].map(is_nonempty)
        & df["heavy_species"].map(canonical_species).eq("mus musculus")
        & df["light_species"].map(canonical_species).eq("mus musculus")
    )
    if "scfv" in df.columns:
        mask &= ~df["scfv"].astype(str).str.lower().isin({"true", "1", "yes"})
    return df[mask].copy()


def build_candidates(rows: pd.DataFrame, fasta_cache: Path, timeout: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    parsed_rows: list[dict] = []
    failed_rows: list[dict] = []
    for _, row in rows.iterrows():
        pdb = str(row["pdb"]).lower()
        try:
            fasta_text = cached_text(
                fasta_cache / f"{pdb}.fasta",
                f"https://www.rcsb.org/fasta/entry/{pdb}/display",
                timeout=timeout,
            )
            label_map, auth_map = fasta_maps(parse_fasta(fasta_text))
            heavy_record = resolve_record(str(row["Hchain"]).strip(), label_map, auth_map)
            light_record = resolve_record(str(row["Lchain"]).strip(), label_map, auth_map)
            heavy_chain = Chain(heavy_record.sequence, scheme="imgt", assign_germline=False)
            light_chain = Chain(light_record.sequence, scheme="imgt", assign_germline=False)
            if heavy_chain.chain_type != "H":
                raise ValueError(f"Hchain parsed as {heavy_chain.chain_type}")
            if light_chain.chain_type not in {"K", "L"}:
                raise ValueError(f"Lchain parsed as {light_chain.chain_type}")
            parsed = row.to_dict()
            parsed.update(
                {
                    "heavy_var": heavy_chain.seq,
                    "light_var": light_chain.seq,
                    "light_chain_type": light_chain.chain_type,
                    "pair_key": f"{heavy_chain.seq}|{light_chain.seq}",
                    "resolution_num": parse_resolution(row.get("resolution")),
                    "antigen_bound": is_nonempty(row.get("antigen_chain")),
                    "heavy_fasta_header": heavy_record.header,
                    "light_fasta_header": light_record.header,
                }
            )
            parsed_rows.append(parsed)
        except Exception as exc:  # noqa: BLE001 - preserve audit details for benchmark rebuilding.
            failed = row.to_dict()
            failed.update({"error_type": type(exc).__name__, "error": str(exc)})
            failed_rows.append(failed)
    parsed_df = pd.DataFrame(parsed_rows)
    failed_df = pd.DataFrame(failed_rows)
    return parsed_df, failed_df


def dedupe_candidates(parsed_df: pd.DataFrame) -> pd.DataFrame:
    if parsed_df.empty:
        return parsed_df
    sorted_df = parsed_df.sort_values(
        by=["date_parsed", "resolution_num", "pdb", "Hchain", "Lchain"],
        ascending=[True, True, True, True, True],
    ).copy()
    return sorted_df.drop_duplicates(subset=["pair_key"], keep="first").reset_index(drop=True)


def select_candidates(
    deduped_df: pd.DataFrame,
    target_count: int,
    known_pdbs: set[str],
    selection_mode: str,
) -> pd.DataFrame:
    if selection_mode == "earliest" or not known_pdbs:
        selected = deduped_df.head(target_count).copy()
    else:
        known = deduped_df[deduped_df["pdb"].isin(known_pdbs)].copy()
        selected_keys = set(known["pair_key"].tolist())
        filler = deduped_df[~deduped_df["pair_key"].isin(selected_keys)].head(max(0, target_count - len(known)))
        selected = pd.concat([known, filler], ignore_index=True)
        selected = selected.sort_values(
            by=["date_parsed", "resolution_num", "pdb", "Hchain", "Lchain"],
            ascending=[True, True, True, True, True],
        ).head(target_count)
    selected = selected.copy().reset_index(drop=True)
    selected["structure_id"] = selected.apply(
        lambda row: f"{row['pdb']}_{row['Hchain']}_{row['Lchain']}", axis=1
    )
    return selected


def download_selected_pdbs(selected_df: pd.DataFrame, output_dir: Path, timeout: float) -> Path:
    pdb_cache = output_dir / "pdb_cache"
    pair_pdb_dir = output_dir / "selected_pdb"
    pdb_cache.mkdir(parents=True, exist_ok=True)
    pair_pdb_dir.mkdir(parents=True, exist_ok=True)
    for _, row in selected_df.iterrows():
        pdb = str(row["pdb"]).lower()
        base_pdb = pdb_cache / f"{pdb}.pdb"
        if not base_pdb.exists() or base_pdb.stat().st_size == 0:
            pdb_text = fetch_url_text(f"https://files.rcsb.org/download/{pdb.upper()}.pdb", timeout=timeout)
            base_pdb.write_text(pdb_text)
        pair_pdb = pair_pdb_dir / f"{row['structure_id']}.pdb"
        if not pair_pdb.exists() or pair_pdb.stat().st_size == 0:
            shutil.copyfile(base_pdb, pair_pdb)
    return pair_pdb_dir


def write_chain_pairs(selected_df: pd.DataFrame, output_dir: Path) -> Path:
    chain_pairs = output_dir / "selected_chain_pairs.csv"
    selected_df[["structure_id", "Hchain", "Lchain"]].assign(
        chain_pair=lambda df: df["Hchain"].astype(str) + "-" + df["Lchain"].astype(str)
    )[["structure_id", "chain_pair"]].to_csv(chain_pairs, index=False, header=False)
    return chain_pairs


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    known_pdbs = {pdb.strip().lower() for pdb in args.known_pdbs.split(",") if pdb.strip()}

    summary = pd.read_csv(args.summary_tsv, sep="\t")
    raw_candidates = candidate_rows(summary, args.start_date, args.end_date)
    raw_candidates.to_csv(output_dir / "raw_mouse_rows.tsv", sep="\t", index=False)

    parsed_df, failed_df = build_candidates(raw_candidates, output_dir / "fasta_cache", args.timeout)
    parsed_df.to_csv(output_dir / "parsed_mouse_pairs.tsv", sep="\t", index=False)
    failed_df.to_csv(output_dir / "failed_mouse_rows.tsv", sep="\t", index=False)

    deduped_df = dedupe_candidates(parsed_df)
    deduped_df.to_csv(output_dir / "deduped_mouse_pairs.tsv", sep="\t", index=False)

    selected_df = select_candidates(deduped_df, args.target_count, known_pdbs, args.selection_mode)
    selected_df.to_csv(output_dir / "selected_mouse_pairs.tsv", sep="\t", index=False)
    chain_pairs_path = write_chain_pairs(selected_df, output_dir)

    selected_pdb_dir = None
    if args.download_pdbs:
        selected_pdb_dir = download_selected_pdbs(selected_df, output_dir, args.timeout)

    known_presence = {
        pdb: {
            "raw_rows": int(raw_candidates["pdb"].eq(pdb).sum()),
            "parsed_pairs": int(parsed_df["pdb"].eq(pdb).sum()) if not parsed_df.empty else 0,
            "deduped_pairs": int(deduped_df["pdb"].eq(pdb).sum()) if not deduped_df.empty else 0,
            "selected_pairs": int(selected_df["pdb"].eq(pdb).sum()) if not selected_df.empty else 0,
        }
        for pdb in sorted(known_pdbs)
    }
    summary_json = {
        "summary_tsv": str(args.summary_tsv),
        "date_range": [args.start_date, args.end_date],
        "target_count": args.target_count,
        "selection_mode": args.selection_mode,
        "raw_rows": int(len(raw_candidates)),
        "parsed_rows": int(len(parsed_df)),
        "failed_rows": int(len(failed_df)),
        "deduped_pairs": int(len(deduped_df)),
        "selected_pairs": int(len(selected_df)),
        "selected_bound_pairs": int(selected_df["antigen_bound"].sum()) if not selected_df.empty else 0,
        "selected_unbound_pairs": int((~selected_df["antigen_bound"]).sum()) if not selected_df.empty else 0,
        "known_presence": known_presence,
        "selected_pdb_dir": str(selected_pdb_dir) if selected_pdb_dir is not None else None,
        "chain_pairs_csv": str(chain_pairs_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_json, indent=2, sort_keys=True))
    print(json.dumps(summary_json, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
