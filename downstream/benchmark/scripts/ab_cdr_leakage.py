#!/usr/bin/env python
"""AB CDR infilling -- measured train/test overlap for the two infilling test sets.

Why this exists
---------------
All four immune-fusion training runs passed ``--asd_antibody_benchmark_blocklist ""``,
which the then-current loader turned into a silent no-op, so antibody-side
decontamination was disabled for the whole run. The §0.5 audit therefore withdrew
the "clean tie" claim: absence of a visible anomaly does not establish clean data.
This script supplies the missing measurement.

What is measured
----------------
Test antibodies:
  * ``sabdab_kong``  -- 3,127 complexes, the split the Ophiuchus-Ab paper cites
    for Table 2 (union of the ten ``cdrh3/fold_*/test.json`` files).
  * ``sab23h2``      -- 60 antibodies, ``SAb-23-H2-Ab/fasta.files.native/*.fasta``
    (chain ``H``), the set behind Table 1.

Antibody-side training corpora actually consumed by the immune-fusion mix
(``--dataset_args oas+ots+asd_antibody+trait+tcr_native``; ``ots`` and ``trait``
are TCR sources and are therefore not antibody leakage candidates):
  * ``oas``          -- ``cleaned_merged_data_step_clustered_{train,valid}_oas_label.csv``,
                        heavy chain from ``cleaned_h_sequence``.
  * ``asd_antibody`` -- ``asd/step6_final/antibody/{train,valid}.csv``, ``heavy_fv``.

Criteria. The two exact criteria deliberately reuse this repo's own blocklist key
builders (``asd_antibody_benchmark_keys`` / ``oas_benchmark_keys``), so the result
answers precisely "what would the disabled blocklist have removed?":
  * ``exact_heavy``   -- verbatim heavy variable-domain match (the ``H:`` key).
  * ``exact_h3_core`` -- anchor-free CDR-H3 core match (the ``h3:`` key). Note the
    core is derived by the ``C[A-Z]{2,32}?[FW]G[A-Z]G`` motif and takes the
    *first* junction, so an antibody carrying an internal [FW]G-X-G inside H3
    yields a prefix of the true H3. That is a property of the blocklist itself and
    is reported as-is; for ``sabdab_kong`` the authoritative annotated
    ``cdrh3_seq`` is additionally reported as ``exact_h3_annotated``.
  * ``lev1_heavy``    -- heavy chain within Levenshtein distance 1 of a training
    heavy chain. Found by prefix/suffix blocking: if ``lev(a,b) <= 1`` then a's
    first half or its last half survives the edit unchanged, so indexing training
    heavies by both halves and verifying candidates is exact, not approximate.

Why exact/Lev<=1 on the heavy chain undercounts
-----------------------------------------------
The two corpora truncate the variable domain differently from the test files, so
the same antibody often appears with shifted termini. Example: Kong ``6myy``
stores ``VQLVQSGAEVKKPGASVKVSC...`` (118 aa) while ASD stores
``QVQLVQSGAEVKKPGASVKVSC...`` (121 aa) -- one leading residue plus one internal
substitution, i.e. Levenshtein 2, so both ``exact_heavy`` and ``lev1_heavy``
miss it even though it is the same antibody. A boundary-insensitive criterion is
therefore required:

  * ``identity_max``  -- highest whole-chain sequence identity (rapidfuzz
    normalized Levenshtein similarity) against any training heavy chain, reported
    as a distribution plus counts at 1.00 / >=0.99 / >=0.98 / >=0.95 / >=0.90.
  * ``identity_max_h3_mismatch`` -- the same maximum, but restricted to training
    antibodies whose CDR-H3 core *differs* from the test antibody's. Antibody
    heavy chains share germline frameworks, so unrelated antibodies routinely
    reach ~0.90 identity; this control measures that background so the
    ``identity_max`` thresholds can be read against it rather than in a vacuum.

Output:
    outputs/ab_cdr_leakage_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
ROOT = BENCH.parents[1]          # .../dllm_test
sys.path.insert(0, str(ROOT))

from dllm.pipelines.immune_llada.data.records import normalize_sequence  # noqa: E402
from dllm.pipelines.immune_llada.data.sources import (  # noqa: E402
    _cdr3_core_key,
    _h3_core_from_chain,
)

KONG_DIR = ROOT / "data" / "downstream" / "cdr_infilling" / "sabdab_kong" / "cdrh3"
SAB23H2_DIR = (ROOT / "data" / "downstream" / "cdr_infilling" / "sab23h2" /
               "SAb-23-H2-Ab" / "fasta.files.native")
OAS_DIR = ROOT / "data" / "oas_previous_clean" / "splits"
ASD_DIR = ROOT / "downstream" / "asd" / "step6_final" / "antibody"
OUT = BENCH / "outputs" / "ab_cdr_leakage_report.json"

CORPORA = {
    "oas": [(OAS_DIR / f"cleaned_merged_data_step_clustered_{s}_oas_label.csv",
             "cleaned_h_sequence") for s in ("train", "valid")],
    "asd_antibody": [(ASD_DIR / f"{s}.csv", "heavy_fv") for s in ("train", "valid")],
}


# ---------------------------------------------------------------------------
# Test sets
# ---------------------------------------------------------------------------

def load_kong() -> list[dict]:
    """Union of the ten Kong-fold CDR-H3 test files, deduped by PDB id."""
    seen: dict[str, dict] = {}
    for fold in sorted(KONG_DIR.glob("fold_*")):
        f = fold / "test.json"
        if not f.exists():
            continue
        text = f.read_text()
        recs = (json.loads(text) if text.lstrip().startswith("[")
                else [json.loads(l) for l in text.splitlines() if l.strip()])
        for r in recs:
            heavy = normalize_sequence(r.get("heavy_chain_seq"))
            if not heavy:
                continue
            seen.setdefault(str(r.get("pdb")), {
                "id": str(r.get("pdb")),
                "heavy": heavy,
                "h3_annotated": _cdr3_core_key(r.get("cdrh3_seq")),
                "fold": fold.name,
            })
    return list(seen.values())


def load_sab23h2() -> list[dict]:
    """Chain ``H`` of each native fasta in the SAb-23-H2 antibody test set.

    The directory was unpacked from a macOS tarball and carries one AppleDouble
    resource fork (``._<name>.fasta``) per real file; those are binary and must
    be skipped or the glob doubles to 120 entries.
    """
    out = []
    for f in sorted(SAB23H2_DIR.glob("*.fasta")):
        if f.name.startswith("._"):
            continue
        chains, cur = {}, None
        for line in f.read_text().splitlines():
            if line.startswith(">"):
                cur = line[1:].strip()
                chains[cur] = []
            elif cur:
                chains[cur].append(line.strip())
        heavy = normalize_sequence("".join(chains.get("H", [])))
        if heavy:
            out.append({"id": f.stem, "heavy": heavy, "h3_annotated": ""})
    return out


# ---------------------------------------------------------------------------
# Lev<=1 blocking index
# ---------------------------------------------------------------------------

def _halves(s: str) -> tuple[str, str]:
    k = len(s) // 2
    return s[:k], s[len(s) - k:]


class Lev1Index:
    """Exact Lev<=1 membership test over a streamed set of reference strings.

    Pigeonhole: if ``lev(a, b) <= 1`` then either a's first ``k`` characters or
    its last ``k`` characters (``k = len//2``) appear unchanged and in the same
    position-from-that-end in ``b``. Indexing every reference by both halves and
    verifying candidates with a real distance computation is therefore complete.
    Only queries are indexed, so memory is O(#test), independent of corpus size.
    """

    def __init__(self, queries: list[str]):
        from rapidfuzz.distance import Levenshtein
        self._lev = Levenshtein.distance
        self.queries = queries
        self.by_prefix: dict[str, list[int]] = defaultdict(list)
        self.by_suffix: dict[str, list[int]] = defaultdict(list)
        self.lengths = sorted({len(q) for q in queries})
        for i, q in enumerate(queries):
            p, s = _halves(q)
            self.by_prefix[p].append(i)
            self.by_suffix[s].append(i)
        # Queries may differ in length from refs by at most 1, so a ref must be
        # probed with every query half-length in {L-1, L, L+1}.
        self.half_lengths = sorted({l // 2 for l in self.lengths} |
                                   {(l + 1) // 2 for l in self.lengths} |
                                   {(l - 1) // 2 for l in self.lengths if l > 1})

    def hits(self, ref: str) -> set[int]:
        """Indices of queries within Levenshtein distance 1 of ``ref``."""
        cand: set[int] = set()
        n = len(ref)
        for k in self.half_lengths:
            if k <= 0 or k > n:
                continue
            cand.update(self.by_prefix.get(ref[:k], ()))
            cand.update(self.by_suffix.get(ref[n - k:], ()))
        return {i for i in cand
                if abs(len(self.queries[i]) - n) <= 1
                and self._lev(self.queries[i], ref, score_cutoff=1) <= 1}


# ---------------------------------------------------------------------------
# Corpus scan
# ---------------------------------------------------------------------------

IDENTITY_THRESHOLDS = (1.0, 0.99, 0.98, 0.95, 0.90)


def identity_scan(tests: dict[str, list[dict]], refs: list[str], ref_cores: list[str],
                  chunk: int = 100_000) -> dict:
    """Max whole-chain identity per test antibody, overall and vs H3-mismatched refs.

    The H3-mismatched maximum is the germline-framework background: it is the best
    a training antibody that is demonstrably *not* the same clone can score.
    """
    import numpy as np
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz.process import cdist

    out = {}
    for t, ts in tests.items():
        queries = [q["heavy"] for q in ts]
        q_cores = [_h3_core_from_chain(q["heavy"]) for q in ts]
        best = np.zeros(len(queries), dtype=np.float32)
        best_mm = np.zeros(len(queries), dtype=np.float32)
        for start in range(0, len(refs), chunk):
            sub = refs[start:start + chunk]
            sub_cores = ref_cores[start:start + chunk]
            D = cdist(queries, sub, scorer=Levenshtein.normalized_similarity,
                      score_cutoff=0.80, workers=-1)
            np.maximum(best, D.max(axis=1), out=best)
            # Mask out refs sharing this query's H3 core, then take the max again.
            same = np.zeros(D.shape, dtype=bool)
            core_pos: dict[str, list[int]] = defaultdict(list)
            for j, c in enumerate(sub_cores):
                if c:
                    core_pos[c].append(j)
            for i, qc in enumerate(q_cores):
                if qc and qc in core_pos:
                    same[i, core_pos[qc]] = True
            Dm = np.where(same, 0.0, D)
            np.maximum(best_mm, Dm.max(axis=1), out=best_mm)
        out[t] = {
            "identity_max": {
                "counts": {f">={thr:.2f}": int((best >= thr - 1e-9).sum())
                           for thr in IDENTITY_THRESHOLDS},
                "fracs": {f">={thr:.2f}": round(float((best >= thr - 1e-9).mean()), 4)
                          for thr in IDENTITY_THRESHOLDS},
                "percentiles": {f"p{p}": round(float(np.percentile(best, p)), 4)
                                for p in (5, 25, 50, 75, 95)},
            },
            "identity_max_h3_mismatch": {
                "counts": {f">={thr:.2f}": int((best_mm >= thr - 1e-9).sum())
                           for thr in IDENTITY_THRESHOLDS},
                "fracs": {f">={thr:.2f}": round(float((best_mm >= thr - 1e-9).mean()), 4)
                          for thr in IDENTITY_THRESHOLDS},
                "percentiles": {f"p{p}": round(float(np.percentile(best_mm, p)), 4)
                                for p in (5, 25, 50, 75, 95)},
            },
        }
    return out


def scan(name: str, files, tests: dict[str, list[dict]], report_every: int) -> dict:
    """One streaming pass over a corpus, scoring every criterion for every test set."""
    # heavy -> *all* test indices carrying it. The Kong set has 3,127 entries but
    # only 2,134 distinct heavy chains (the same antibody appears against several
    # antigens), so a heavy->single-index dict would silently drop 993 of them.
    heavy_exact: dict[str, dict[str, set[int]]] = {}
    for t, ts in tests.items():
        m: dict[str, set[int]] = defaultdict(set)
        for i, q in enumerate(ts):
            m[q["heavy"]].add(i)
        heavy_exact[t] = m
    h3_exact = {t: defaultdict(set) for t in tests}
    h3_annot = {t: defaultdict(set) for t in tests}
    for t, ts in tests.items():
        for i, q in enumerate(ts):
            core = _h3_core_from_chain(q["heavy"])
            if core:
                h3_exact[t][core].add(i)
            if q["h3_annotated"]:
                h3_annot[t][q["h3_annotated"]].add(i)
    lev = {t: Lev1Index([q["heavy"] for q in ts]) for t, ts in tests.items()}

    hit = {t: {k: set() for k in ("exact_heavy", "exact_h3_core",
                                  "exact_h3_annotated", "lev1_heavy")}
           for t in tests}
    uniq_heavy: dict[str, str] = {}   # heavy -> its H3 core, deduped for the identity pass
    n_rows = 0
    for path, col in files:
        if not path.exists():
            print(f"  [skip] {path} missing")
            continue
        print(f"  scanning {path.name} (column {col!r}) ...", flush=True)
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                n_rows += 1
                heavy = normalize_sequence(row.get(col))
                if not heavy:
                    continue
                if heavy in uniq_heavy:
                    continue
                core = _h3_core_from_chain(heavy)
                uniq_heavy[heavy] = core
                for t in tests:
                    hit[t]["exact_heavy"] |= heavy_exact[t].get(heavy, set())
                    if core:
                        hit[t]["exact_h3_core"] |= h3_exact[t].get(core, set())
                        hit[t]["exact_h3_annotated"] |= h3_annot[t].get(core, set())
                    hit[t]["lev1_heavy"] |= lev[t].hits(heavy)
                if report_every and n_rows % report_every == 0:
                    tot = {t: len(hit[t]["lev1_heavy"]) for t in tests}
                    print(f"    {n_rows:,} rows ({len(uniq_heavy):,} unique heavies), "
                          f"lev1 hits so far {tot}", flush=True)

    print(f"  {n_rows:,} rows -> {len(uniq_heavy):,} unique heavy chains; "
          f"running identity pass ...", flush=True)
    refs = list(uniq_heavy)
    ident = identity_scan(tests, refs, [uniq_heavy[r] for r in refs])

    res = {"n_corpus_rows": n_rows, "n_unique_heavy": len(uniq_heavy), "test_sets": {}}
    for t, ts in tests.items():
        n = len(ts)
        res["test_sets"][t] = {
            "n_test": n,
            **{k: {"n": len(v), "frac": round(len(v) / max(1, n), 4),
                   "ids": sorted(ts[i]["id"] for i in v)[:50]}
               for k, v in hit[t].items()},
            **ident[t],
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="+", default=list(CORPORA),
                    choices=list(CORPORA))
    ap.add_argument("--report-every", type=int, default=500_000)
    args = ap.parse_args()

    tests = {"sabdab_kong": load_kong(), "sab23h2": load_sab23h2()}
    for t, ts in tests.items():
        annot = sum(1 for q in ts if q["h3_annotated"])
        print(f"test set {t}: {len(ts)} antibodies, "
              f"{len({q['heavy'] for q in ts})} distinct heavy chains "
              f"(annotated H3 available for {annot})")

    report = {
        "test_sets": {t: {"n": len(ts),
                          "n_distinct_heavy": len({q["heavy"] for q in ts}),
                          "n_with_annotated_h3": sum(1 for q in ts if q["h3_annotated"])}
                      for t, ts in tests.items()},
        "criteria": {
            "exact_heavy": "verbatim heavy variable domain (blocklist 'H:' key)",
            "exact_h3_core": "anchor-free CDR-H3 core via [FW]G-X-G motif "
                             "(blocklist 'h3:' key); takes the first junction, so "
                             "an internal motif yields a prefix of the true H3",
            "exact_h3_annotated": "annotated cdrh3_seq from the test file, core-keyed "
                                  "(sabdab_kong only; matched against the corpus "
                                  "motif-derived core)",
            "lev1_heavy": "heavy chain within Levenshtein distance 1 of a training "
                          "heavy chain (exact test via prefix/suffix blocking)",
        },
        "blocklist_status": "all four immune-fusion runs passed "
                            "--asd_antibody_benchmark_blocklist \"\" -> antibody "
                            "decontamination disabled; this report measures what it "
                            "would have removed",
        "corpora": {},
    }
    for name in args.corpora:
        print(f"\n=== corpus {name} ===")
        report["corpora"][name] = scan(name, CORPORA[name], tests, args.report_every)
        for t, r in report["corpora"][name]["test_sets"].items():
            print(f"  {t} (n={r['n_test']}):")
            for k in ("exact_heavy", "lev1_heavy", "exact_h3_core",
                      "exact_h3_annotated"):
                print(f"    {k:22s} {r[k]['n']:5d}  {r[k]['frac']*100:6.2f}%")
            im, mm = r["identity_max"], r["identity_max_h3_mismatch"]
            print(f"    {'identity>=0.99':22s} {im['counts']['>=0.99']:5d}  "
                  f"{im['fracs']['>=0.99']*100:6.2f}%   "
                  f"(H3-mismatch control {mm['fracs']['>=0.99']*100:.2f}%)")
            print(f"    {'identity>=0.95':22s} {im['counts']['>=0.95']:5d}  "
                  f"{im['fracs']['>=0.95']*100:6.2f}%   "
                  f"(H3-mismatch control {mm['fracs']['>=0.95']*100:.2f}%)")
            print(f"    max-identity median {im['percentiles']['p50']:.4f} "
                  f"vs H3-mismatch control {mm['percentiles']['p50']:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
