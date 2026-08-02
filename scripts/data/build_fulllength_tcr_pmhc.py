#!/usr/bin/env python
"""Rebuild full-length five-entity TCR-pMHC records from VDJdb + McPAS.

Why (ARCH_AUDIT gap 2 / FUTURE_EXPERIMENTS D1): the current TCR sources are
CDR3 fragments with no full-length variable domains and no MHC+B2M, so the
grammar's intended layout

    <prots> MHC . B2M <protd> <binding> <prots> <pep> PEP <protd> <binding>
    <prots> <tcr> ALPHA . BETA <protd>

is never realized. This script reconstructs it:

1. Parse VDJdb (`data/tcr/vdjdb_full.txt`) and McPAS (`data/tcr/McPAS-TCR.csv`),
   human, MHC class I, keeping V/J genes + CDR3 (alpha & beta), peptide, MHC-A
   allele, and pairing key (VDJdb ``complex.id`` / McPAS row).
2. Stitch full-length alpha/beta variable+constant domains with Stitchr/thimble
   (batch TSV, TRA/TRB mode, human IMGT reference from ``stitchrdl -s human``).
3. Map the MHC-A allele to a full-length MHC-I heavy chain from IMGT/HLA
   (`data/tcr_pmhc_fulllength/imgt_hla/hla_prot.fasta`) and pair it with human
   B2M (mature, UniProt P61769).
4. Emit ``bioseq.v1`` JSONL records ``[mhc, b2m, peptide, tcr_alpha, tcr_beta]``
   with roles ``[mhc, mhc, peptide, tcr_alpha, tcr_beta]`` (MHC+B2M share role
   ``mhc`` so they render as one ``<prots> MHC . B2M <protd>`` block) and
   ``labels={"relation": "binding"}`` (VDJdb/McPAS are curated binders).

Run (env protenix_abtcr, needs Stitchr HUMAN ref + hla_prot.fasta already present):

    PATH=/vepfs-mlp2/.../protenix_abtcr/bin:$PATH \
    python scripts/data/build_fulllength_tcr_pmhc.py --source both --split-frac 0.005

Outputs under ``data/tcr_pmhc_fulllength/``:
    thimble_in_<src>.tsv, thimble_out_<src>.tsv (stitched aa),
    records_{train,valid,holdout}.jsonl, build_stats.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
OUT = ROOT / "data" / "tcr_pmhc_fulllength"
VDJDB = ROOT / "data" / "tcr" / "vdjdb_full.txt"
MCPAS = ROOT / "data" / "tcr" / "McPAS-TCR.csv"
HLA_FASTA = OUT / "imgt_hla" / "hla_prot.fasta"
# Downstream TCR CDR3-beta bank (test/eval side) built by
# scripts/data/dedup/build_downstream_banks.py. Records whose CDR3-beta appears
# here are dropped so this new source cannot leak into T1/T3/T4.
CDR3B_BANK = ROOT / "data" / "dedup" / "banks" / "tcr_cdr3b.txt"
# Mature human B2M (P61769 minus its 20-aa signal peptide).
B2M_MATURE = (
    "IQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEK"
    "DEYACRVNHVTLSQPKIVKWDRDM"
)
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

_AA = set("ACDEFGHIKLMNPQRSTVWY")
_GENE_RE = re.compile(r"^TR[ABGD][VJDC][0-9]")


def _clean_cdr3(seq: str) -> str:
    seq = (seq or "").strip().upper()
    return seq if seq and all(c in _AA for c in seq) else ""


def _gene(name: str) -> str:
    """Keep an IMGT-looking gene token (drop allele suffix; thimble defaults *01)."""
    name = (name or "").strip().split("*")[0].split(",")[0].strip()
    return name if _GENE_RE.match(name) else ""


# --------------------------------------------------------------------------- #
# IMGT/HLA class-I heavy-chain resolution
# --------------------------------------------------------------------------- #
def load_hla(fasta: Path) -> dict[str, str]:
    """allele-name -> protein seq, e.g. 'A*02:01:01:01' -> 'GSHSMRYF...'."""
    out: dict[str, str] = {}
    name, buf = None, []
    with fasta.open() as fh:
        for line in fh:
            if line.startswith(">"):
                if name and buf:
                    out[name] = "".join(buf)
                # header: '>HLA:HLA00001 A*01:01:01:01 365 bp'
                parts = line[1:].split()
                name = parts[1] if len(parts) > 1 else parts[0]
                buf = []
            else:
                buf.append(line.strip())
    if name and buf:
        out[name] = "".join(buf)
    return out


class HLAResolver:
    """Resolve a queried MHC-A allele to a full-length class-I heavy chain."""

    def __init__(self, hla: dict[str, str]) -> None:
        # Only class-I genes carry B2M; skip DR/DQ/DP (class II).
        self.by_allele = {k: v for k, v in hla.items() if k.split("*")[0] in {"A", "B", "C", "E", "F", "G"}}
        self.cache: dict[str, str] = {}

    @staticmethod
    def _norm(query: str) -> str:
        q = (query or "").strip()
        q = q.replace("HLA-", "").replace("HLA", "").strip()
        return q

    def resolve(self, query: str) -> str:
        q = self._norm(query)
        if not q or "*" not in q:
            return ""
        if q in self.cache:
            return self.cache[q]
        # exact, then 2-field prefix, then gene-level representative
        cands = [a for a in self.by_allele if a == q]
        if not cands:
            fields = q.split(":")
            two = ":".join(fields[:2]) if len(fields) >= 2 else q
            cands = sorted(a for a in self.by_allele if a.startswith(two + ":") or a == two)
        if not cands:
            gene = q.split("*")[0] + "*" + q.split("*")[1].split(":")[0]
            cands = sorted(a for a in self.by_allele if a.startswith(gene + ":") or a.startswith(gene + "*"))
        seq = self.by_allele[cands[0]] if cands else ""
        self.cache[q] = seq
        return seq


# --------------------------------------------------------------------------- #
# Source parsing -> intermediate rows
# --------------------------------------------------------------------------- #
def parse_vdjdb(limit: int | None) -> list[dict]:
    rows: list[dict] = []
    with VDJDB.open(newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for r in reader:
            if r.get("species") != "HomoSapiens":
                continue
            if r.get("mhc.class") != "MHCI":
                continue
            rows.append({
                "src": "vdjdb",
                "pair_id": r.get("complex.id") or "",
                "trav": _gene(r.get("v.alpha", "")), "traj": _gene(r.get("j.alpha", "")),
                "cdr3a": _clean_cdr3(r.get("cdr3.alpha", "")),
                "trbv": _gene(r.get("v.beta", "")), "trbj": _gene(r.get("j.beta", "")),
                "cdr3b": _clean_cdr3(r.get("cdr3.beta", "")),
                "peptide": _clean_cdr3(r.get("antigen.epitope", "")),
                "mhc_a": r.get("mhc.a", ""),
            })
            if limit and len(rows) >= limit:
                break
    return rows


def parse_mcpas(limit: int | None) -> list[dict]:
    rows: list[dict] = []
    with MCPAS.open(newline="", encoding="latin-1") as fh:
        reader = csv.DictReader(fh)
        for i, r in enumerate(reader):
            if (r.get("Species") or "").strip() != "Human":
                continue
            mhc = (r.get("MHC") or "").strip()
            if not (mhc.startswith(("HLA-A", "HLA-B", "HLA-C", "A*", "B*", "C*"))):
                continue  # class-I only (heuristic)
            rows.append({
                "src": "mcpas",
                "pair_id": f"mcpas{i}",
                "trav": _gene(r.get("TRAV", "")), "traj": _gene(r.get("TRAJ", "")),
                "cdr3a": _clean_cdr3(r.get("CDR3.alpha.aa", "")),
                "trbv": _gene(r.get("TRBV", "")), "trbj": _gene(r.get("TRBJ", "")),
                "cdr3b": _clean_cdr3(r.get("CDR3.beta.aa", "")),
                "peptide": _clean_cdr3(r.get("Epitope.peptide", "")),
                "mhc_a": mhc,
            })
            if limit and len(rows) >= limit:
                break
    return rows


# --------------------------------------------------------------------------- #
# thimble batch stitching
# --------------------------------------------------------------------------- #
THIMBLE_IN_COLS = [
    "TCR_name", "TRAV", "TRAJ", "TRA_CDR3", "TRBV", "TRBJ", "TRB_CDR3",
    "TRAC", "TRBC", "TRA_leader", "TRB_leader", "Linker", "Link_order",
    "TRA_5_prime_seq", "TRA_3_prime_seq", "TRB_5_prime_seq", "TRB_3_prime_seq",
]


def write_thimble_input(rows: list[dict], path: Path) -> list[dict]:
    """Write the thimble TSV; return rows that are stitchable (paired V/J/CDR3)."""
    stitchable: list[dict] = []
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(THIMBLE_IN_COLS)
        for idx, r in enumerate(rows):
            has_a = r["trav"] and r["traj"] and r["cdr3a"]
            has_b = r["trbv"] and r["trbj"] and r["cdr3b"]
            if not (has_a and has_b):
                continue  # need paired alpha+beta for the five-entity layout
            name = f"{r['src']}_{idx}"
            r["tcr_name"] = name
            w.writerow([
                name,
                r["trav"] if has_a else "", r["traj"] if has_a else "", r["cdr3a"] if has_a else "",
                r["trbv"] if has_b else "", r["trbj"] if has_b else "", r["cdr3b"] if has_b else "",
                "", "", "", "", "", "", "", "", "", "",
            ])
            stitchable.append(r)
    return stitchable


def run_thimble(in_tsv: Path, out_tsv: Path) -> None:
    cmd = ["thimble", "-in", str(in_tsv), "-o", str(out_tsv), "-s", "HUMAN", "-r", "TRA/TRB"]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def read_thimble_output(path: Path) -> dict[str, tuple[str, str]]:
    """TCR_name -> (TRA_aa, TRB_aa), keeping only clean full-length aa seqs."""
    out: dict[str, tuple[str, str]] = {}
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            a = (r.get("TRA_aa") or "").strip().upper()
            b = (r.get("TRB_aa") or "").strip().upper()
            # thimble marks failures with warnings + empty/partial aa
            if a and b and all(c in _AA for c in a) and all(c in _AA for c in b):
                out[r["TCR_name"]] = (a, b)
    return out


# --------------------------------------------------------------------------- #
def _load_cdr3b_bank() -> set[str]:
    """Downstream test CDR3-beta bank, normalized like the shard builders (J->L)."""
    if not CDR3B_BANK.is_file():
        print(f"WARNING: CDR3b bank {CDR3B_BANK} missing; skipping downstream dedup", flush=True)
        return set()
    bank: set[str] = set()
    for line in CDR3B_BANK.read_text().splitlines():
        s = line.strip().upper().replace("J", "L")
        if s:
            bank.add(s)
    return bank


def build_records(rows: list[dict], stitched: dict[str, tuple[str, str]],
                  resolver: HLAResolver) -> tuple[list[dict], int]:
    bank = _load_cdr3b_bank()
    leaked = 0
    recs: list[dict] = []
    for r in rows:
        name = r.get("tcr_name")
        if name not in stitched:
            continue
        if not r["peptide"]:
            continue
        # Downstream dedup: drop if the CDR3-beta clonotype key is in any TCR
        # test/eval set (exact single-linkage, per DEDUP_COVERAGE.md).
        if bank and r["cdr3b"].upper().replace("J", "L") in bank:
            leaked += 1
            continue
        mhc = resolver.resolve(r["mhc_a"])
        if not mhc:
            continue
        tra, trb = stitched[name]
        recs.append({
            "schema_version": "bioseq.v1",
            "task_type": "tcr_pmhc",
            "source": "tcr_pmhc_fulllength",
            "chains": [mhc, B2M_MATURE, r["peptide"], tra, trb],
            "chain_roles": ["mhc", "mhc", "peptide", "tcr_alpha", "tcr_beta"],
            "targets": [3, 4],
            "labels": {"relation": "binding"},
            "metadata": {"origin": r["src"], "mhc_a": r["mhc_a"], "pair_id": r["pair_id"]},
        })
    return recs, leaked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["vdjdb", "mcpas", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None, help="cap rows per source (validation)")
    ap.add_argument("--split-frac", type=float, default=0.005, help="valid/holdout fraction each")
    ap.add_argument("--seed", type=int, default=137)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("loading IMGT/HLA ...", flush=True)
    resolver = HLAResolver(load_hla(HLA_FASTA))

    parsed: list[dict] = []
    if args.source in ("vdjdb", "both"):
        parsed += parse_vdjdb(args.limit)
    if args.source in ("mcpas", "both"):
        parsed += parse_mcpas(args.limit)
    print(f"parsed rows (human MHCI): {len(parsed)}", flush=True)

    in_tsv = OUT / "thimble_in.tsv"
    out_tsv = OUT / "thimble_out.tsv"
    stitchable = write_thimble_input(parsed, in_tsv)
    print(f"stitchable (paired V/J/CDR3): {len(stitchable)}", flush=True)
    run_thimble(in_tsv, out_tsv)
    stitched = read_thimble_output(out_tsv)
    print(f"stitched OK: {len(stitched)}", flush=True)

    recs, leaked = build_records(stitchable, stitched, resolver)
    print(f"dropped by downstream CDR3b dedup: {leaked}", flush=True)
    print(f"final five-entity records: {len(recs)}", flush=True)

    # Dedup identical records (same 5 chains) and split by pair_id groups.
    seen: set[tuple] = set()
    uniq: list[dict] = []
    for rec in recs:
        key = tuple(rec["chains"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(rec)
    rng = random.Random(args.seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_eval = max(1, int(n * args.split_frac)) if n > 20 else 0
    holdout, valid, train = uniq[:n_eval], uniq[n_eval:2 * n_eval], uniq[2 * n_eval:]

    for split, data in (("train", train), ("valid", valid), ("holdout", holdout)):
        p = OUT / f"records_{split}.jsonl"
        with p.open("w") as fh:
            for rec in data:
                fh.write(json.dumps(rec) + "\n")
        print(f"wrote {p} ({len(data)} rows)", flush=True)

    stats = {
        "parsed_human_mhci": len(parsed),
        "stitchable_paired": len(stitchable),
        "stitched_ok": len(stitched),
        "downstream_cdr3b_leaked_dropped": leaked,
        "final_records": len(recs),
        "unique_records": n,
        "splits": {"train": len(train), "valid": len(valid), "holdout": len(holdout)},
        "b2m_len": len(B2M_MATURE),
    }
    (OUT / "build_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
