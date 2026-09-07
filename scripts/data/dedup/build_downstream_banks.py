#!/usr/bin/env python3
"""Build downstream-test *banks* for train<->test deduplication.

Why: the integrated training corpus (OAS/OTS/nanobody/PPI/MINT/tcr_piste/...)
must not contain rows that leak into any downstream benchmark test set. Leakage
inflates every headline metric. This script reads every relevant downstream
*test* file once and writes compact, per-biological-type "banks" that the
matcher (``dedup_train_vs_downstream.py``) consumes.

Bank types (written under ``data/dedup/banks/``):
  * ``ab_cdrh3.txt``       -- unique CDR-H3 loops (antibody + nanobody) for
                              near-exact / cluster matching.
  * ``ab_heavy.fasta``     -- antibody heavy + nanobody VHH full variable seqs.
  * ``ab_light.fasta``     -- antibody light full variable seqs.
  * ``tcr_cdr3b.txt``      -- unique CDR3-beta clonotype loops.
  * ``tcr_cdr3a.txt``      -- unique CDR3-alpha loops.
  * ``antigen.fasta``      -- antigen/target chains seen in binder tests.
  * ``ppi_proteins.fasta`` -- every protein chain in PPI test sets.
  * ``manifest.json``      -- provenance: which files fed which bank + counts.

All sequences are normalized with the *same* function the shard builders use
(``records.normalize_sequence``: whitespace-strip, upper-case, J->L) so exact
string matches are consistent between banks and training rows.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def _load_normalizer():
    path = PROJECT_ROOT / "dllm/pipelines/qwen3_vl_arch/data/records.py"
    name = "_dedup_records"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass processing needs the module registered
    spec.loader.exec_module(mod)
    return mod.normalize_sequence, mod.is_valid_protein_sequence


normalize_sequence, is_valid_protein_sequence = _load_normalizer()

DATA = PROJECT_ROOT / "data"
DOWN = PROJECT_ROOT / "downstream"
BANK_DIR = DATA / "dedup" / "banks"


def _norm(value: Any) -> str:
    return normalize_sequence(value)


def _read_csv(path: Path) -> Iterable[dict[str, str]]:
    if not path.is_file():
        return
    with path.open(newline="") as handle:
        yield from csv.DictReader(handle)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


class Banks:
    def __init__(self) -> None:
        self.ab_cdrh3: set[str] = set()
        self.ab_heavy: set[str] = set()
        self.ab_light: set[str] = set()
        self.tcr_cdr3b: set[str] = set()
        self.tcr_cdr3a: set[str] = set()
        self.antigen: set[str] = set()
        self.ppi_proteins: set[str] = set()
        self.provenance: dict[str, dict[str, int]] = {}

    def add_cdr3(self, bank: set[str], value: Any, *, min_len: int = 3) -> int:
        seq = _norm(value)
        if len(seq) >= min_len and is_valid_protein_sequence(seq):
            bank.add(seq)
            return 1
        return 0

    def add_full(self, bank: set[str], value: Any, *, min_len: int = 20) -> int:
        seq = _norm(value)
        if len(seq) >= min_len and is_valid_protein_sequence(seq):
            bank.add(seq)
            return 1
        return 0

    def note(self, source: str, counts: dict[str, int]) -> None:
        self.provenance[source] = counts


def build_antibody(banks: Banks) -> None:
    # OAS paired holdout (light-chain pairing benchmark)
    p = DATA / "downstream/comp_chain/test_data_oas_holdout.csv"
    c = {"ab_cdrh3": 0, "ab_heavy": 0, "ab_light": 0}
    for row in _read_csv(p):
        c["ab_cdrh3"] += banks.add_cdr3(banks.ab_cdrh3, row.get("h_cdr3"))
        c["ab_heavy"] += banks.add_full(banks.ab_heavy, row.get("h_sequence") or row.get("cleaned_h_sequence"))
        c["ab_light"] += banks.add_full(banks.ab_light, row.get("l_sequence") or row.get("cleaned_l_sequence"))
    banks.note(str(p), c)

    # SAbDab CDR-infilling (heavy loops H1/H2/H3, 10 folds each)
    for loop in ("cdrh1", "cdrh2", "cdrh3"):
        c = {"ab_cdrh3": 0, "ab_heavy": 0, "ab_light": 0}
        base = DATA / f"downstream/cdr_infilling/sabdab/{loop}"
        for fold in range(10):
            for row in _read_jsonl(base / f"fold_{fold}" / "test.json"):
                if loop == "cdrh3":
                    c["ab_cdrh3"] += banks.add_cdr3(banks.ab_cdrh3, row.get("cdrh3_seq"))
                c["ab_heavy"] += banks.add_full(banks.ab_heavy, row.get("heavy_chain_seq"))
                c["ab_light"] += banks.add_full(banks.ab_light, row.get("light_chain_seq"))
        banks.note(f"sabdab_cdr_infill/{loop}", c)

    # FLAb attribute regression (paired heavy/light; no CDR3 column)
    flab = DATA / "downstream/flab/flab_raw"
    c = {"ab_heavy": 0, "ab_light": 0}
    for fp in sorted(flab.glob("*.csv")):
        for row in _read_csv(fp):
            c["ab_heavy"] += banks.add_full(banks.ab_heavy, row.get("heavy"))
            c["ab_light"] += banks.add_full(banks.ab_light, row.get("light"))
    banks.note(str(flab), c)


def build_nanobody(banks: Banks) -> None:
    root = DATA / "nanobody_raw/nbbench/hf_data"
    seq_cols = ("seq", "VHH_sequence", "seq_nanobody")
    for task_dir in sorted(p for p in root.glob("*") if p.is_dir()):
        test = task_dir / "test.csv"
        c = {"ab_heavy": 0, "ab_cdrh3": 0, "antigen": 0}
        for row in _read_csv(test):
            vhh = ""
            for col in seq_cols:
                if row.get(col):
                    vhh = str(row[col]).replace("-", "")  # CDRInfilling masks with '-'
                    break
            # CDRInfilling stores the native full seq in `label`
            if not vhh and task_dir.name == "CDRInfilling":
                vhh = str(row.get("label", ""))
            c["ab_heavy"] += banks.add_full(banks.ab_heavy, vhh)
            c["ab_cdrh3"] += banks.add_cdr3(banks.ab_cdrh3, row.get("CDR3"))
            for ag_col in ("Ag_sequence", "seq_antigen"):
                if row.get(ag_col):
                    c["antigen"] += banks.add_full(banks.antigen, row[ag_col])
        banks.note(f"nbbench/{task_dir.name}", c)


def build_tcr(banks: Banks) -> None:
    # OTS holdout (paired alpha/beta; beta chosen by anarci_type)
    p = DATA / "ots_paired_clean/final/holdout.csv"
    c = {"tcr_cdr3b": 0, "tcr_cdr3a": 0}
    for row in _read_csv(p):
        for idx in ("1", "2"):
            atype = str(row.get(f"chain{idx}_anarci_type", "")).strip().upper()
            cdr3 = row.get(f"chain{idx}_cdr3")
            if atype == "B":
                c["tcr_cdr3b"] += banks.add_cdr3(banks.tcr_cdr3b, cdr3)
            elif atype == "A":
                c["tcr_cdr3a"] += banks.add_cdr3(banks.tcr_cdr3a, cdr3)
    banks.note(str(p), c)

    # T4 unconditional CDR3b holdout text
    p = DOWN / "benchmark/data/tcr_generation/holdout_cdr3b.txt"
    c = {"tcr_cdr3b": 0}
    if p.is_file():
        for line in p.read_text().splitlines():
            c["tcr_cdr3b"] += banks.add_cdr3(banks.tcr_cdr3b, line)
    banks.note(str(p), c)

    # NM2025 binding, clustering, representation csvs (columns cdr3a/cdr3b)
    csv_globs = [
        DOWN / "benchmark/data/tcr_binding_nm2025",
        DOWN / "benchmark/data/tcr_clustering",
        DOWN / "benchmark/data/tcr_clustering_embed",
        DOWN / "benchmark/data/tcr_representation",
        DOWN / "benchmark/data/tcr_representation_paper6",
        DOWN / "benchmark/data/tcr_binding",
    ]
    for base in csv_globs:
        if not base.exists():
            continue
        c = {"tcr_cdr3b": 0, "tcr_cdr3a": 0}
        for fp in base.rglob("*.csv"):
            # Only test/eval splits define leakage; never fold a benchmark's own
            # TRAIN split into the bank (that would over-remove our training TCRs).
            rel = str(fp.relative_to(base)).lower()
            if "train" in rel or "background" in fp.name.lower():
                continue
            for row in _read_csv(fp):
                c["tcr_cdr3b"] += banks.add_cdr3(banks.tcr_cdr3b, row.get("cdr3b") or row.get("CDR3b"))
                c["tcr_cdr3a"] += banks.add_cdr3(banks.tcr_cdr3a, row.get("cdr3a") or row.get("CDR3a"))
        banks.note(str(base), c)

    # Public TCR-beta specificity benchmark (Track-A): references.csv stores the
    # protected CDR3-beta in a `cdr3b_reference` column (full CASS...F form). The
    # generic csv loop above does NOT catch it (wrong column name), so ingest it
    # explicitly -- these 1306 references are a hard-requirement protected set.
    p = DOWN / "benchmark/data/tcr_beta_public_benchmark/references.csv"
    c = {"tcr_cdr3b": 0}
    for row in _read_csv(p):
        c["tcr_cdr3b"] += banks.add_cdr3(
            banks.tcr_cdr3b, row.get("cdr3b_reference") or row.get("cdr3b") or row.get("CDR3b")
        )
    banks.note(str(p), c)


def build_ppi(banks: Banks) -> None:
    # IRBench STRING 90/90
    p = DOWN / "benchmark/data/ppi/test.csv"
    c = {"ppi_proteins": 0}
    for row in _read_csv(p):
        c["ppi_proteins"] += banks.add_full(banks.ppi_proteins, row.get("seqA"))
        c["ppi_proteins"] += banks.add_full(banks.ppi_proteins, row.get("seqB"))
    banks.note(str(p), c)

    # MINT general-PPI tasks
    mint_specs = [
        (DATA / "downstream/mint/human-ppi/processed_data_test.csv", ("sequence_1", "sequence_2")),
        (DATA / "downstream/mint/yeast-ppi/processed_data_test.csv", ("sequence_1", "sequence_2")),
        (DATA / "downstream/mint/ppi/Intra2_seqs.csv", ("seq1", "seq2")),
        (DATA / "downstream/mint/SKEMPI_v2/processed_data.csv", ("seq1", "seq2", "seq1_mut", "seq2_mut")),
        (DATA / "downstream/mint/mutational-ppi/processed_data.csv", ("seq1", "seq2")),
        (DATA / "downstream/mint/pdb-bind/processed_data.csv", ("seq",)),
    ]
    for path, cols in mint_specs:
        c = {"ppi_proteins": 0}
        for row in _read_csv(path):
            for col in cols:
                if row.get(col):
                    c["ppi_proteins"] += banks.add_full(banks.ppi_proteins, row[col])
        banks.note(str(path), c)


def _write_set(path: Path, values: set[str]) -> None:
    path.write_text("\n".join(sorted(values)) + ("\n" if values else ""))


def _write_fasta(path: Path, values: set[str]) -> None:
    with path.open("w") as handle:
        for i, seq in enumerate(sorted(values)):
            handle.write(f">t{i}\n{seq}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=BANK_DIR)
    parser.add_argument(
        "--domains",
        default="antibody,nanobody,tcr,ppi",
        help="Comma list of domains to build.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    banks = Banks()
    domains = {d.strip() for d in args.domains.split(",") if d.strip()}
    if "antibody" in domains:
        build_antibody(banks)
    if "nanobody" in domains:
        build_nanobody(banks)
    if "tcr" in domains:
        build_tcr(banks)
    if "ppi" in domains:
        build_ppi(banks)

    _write_set(args.out_dir / "ab_cdrh3.txt", banks.ab_cdrh3)
    _write_set(args.out_dir / "tcr_cdr3b.txt", banks.tcr_cdr3b)
    _write_set(args.out_dir / "tcr_cdr3a.txt", banks.tcr_cdr3a)
    _write_fasta(args.out_dir / "ab_heavy.fasta", banks.ab_heavy)
    _write_fasta(args.out_dir / "ab_light.fasta", banks.ab_light)
    _write_fasta(args.out_dir / "antigen.fasta", banks.antigen)
    _write_fasta(args.out_dir / "ppi_proteins.fasta", banks.ppi_proteins)

    summary = {
        "counts": {
            "ab_cdrh3": len(banks.ab_cdrh3),
            "ab_heavy": len(banks.ab_heavy),
            "ab_light": len(banks.ab_light),
            "tcr_cdr3b": len(banks.tcr_cdr3b),
            "tcr_cdr3a": len(banks.tcr_cdr3a),
            "antigen": len(banks.antigen),
            "ppi_proteins": len(banks.ppi_proteins),
        },
        "provenance": banks.provenance,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()
