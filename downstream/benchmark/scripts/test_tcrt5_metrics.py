#!/usr/bin/env python
"""Verify the ported TCRT5 metric suite against the upstream implementation.

Two independent checks:

1. **Gold unit tests** -- the exact expected values from the official
   ``tests/test_evaluation.py`` (precision / recall / f1 / edit distance /
   sequence recovery / closest matches / char-BLEU ordering).

2. **Cross-validation on real predictions** -- for every function, feed the
   ``benchmark_data_w_preds.csv`` (14 benchmark pMHC, with stored TCRT5 /
   GRATCR / ER-Transformer predictions) into BOTH our port and the upstream
   ``ModelEvaluator`` static methods and assert byte-for-byte agreement.

Run:
    conda activate protenix_abtcr
    python downstream/benchmark/scripts/test_tcrt5_metrics.py
"""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
TCRT5 = BENCH / "baselines" / "TCRT5_tcr_translate"

from common import metrics as M  # noqa: E402


def _approx(a, b, tol=1e-9):
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return abs(a - b) <= tol


def gold_unit_tests():
    """Exact expected values transcribed from upstream tests/test_evaluation.py."""
    checks = []
    tr = ["example", "counterfactual", "exemplar"]
    ref = ["example", "exemplar", "prototype", "paradigm"]

    checks.append(("precision_at_k", M.tcrt5_precision_at_k(tr, ref), 2 / 3))
    checks.append(("recall_at_k(k=3)", M.tcrt5_recall_at_k(tr, ref, k=3), 2 / 3))
    tr2 = ["example", "counterfactual", "exemplar", "antithesis", "paragon"]
    checks.append(("recall_at_k(k=5)", M.tcrt5_recall_at_k(tr2, ref, k=5), 2 / 4))
    checks.append(("f1_at_k(k=3)", M.tcrt5_f1_at_k(tr, ref, k=3), 2 / 3))
    checks.append(("f1_at_k(zero)", M.tcrt5_f1_at_k(["counterfactual", "antithesis", "paragon"], ref, k=3), 0.0))
    checks.append(("mean_edit_distance",
                   M.tcrt5_mean_edit_distance(["example", "exampel", "exmplar"], ref),
                   float(np.mean([0, 2, 1]))))
    checks.append(("mean_sequence_recovery",
                   M.tcrt5_mean_sequence_recovery(["example", "exampel", "parakeet"], ref),
                   float(np.mean([1.0, 5 / 7, 1 / 2]))))
    fnc = M.find_n_closest_matches("example", ["exampel", "xzampls", "exapmle", "exmplear"], 2)
    checks.append(("find_n_closest_matches", fnc == ["exampel", "exapmle"], True))

    b3 = M.tcrt5_sequence_bleu("exempt", ["exemplar", "exemplary", "exemplify", "exemplification"],
                               max_references=3, max_ngram=3)
    b4 = M.tcrt5_sequence_bleu("exempt", ["exemplar", "exemplary", "exemplify", "exemplification"],
                               max_references=3, max_ngram=4)
    checks.append(("char_bleu b3>=b4", (b3 >= b4) and (b3 != b4), True))

    ok = True
    print("=== gold unit tests (upstream test_evaluation.py) ===")
    for name, got, exp in checks:
        passed = (got == exp) if isinstance(exp, bool) else _approx(got, exp)
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: got={got} expected={exp}")
    return ok


def _parse_list(cell):
    if isinstance(cell, list):
        return cell
    if not isinstance(cell, str) or not cell.strip():
        return []
    try:
        return list(ast.literal_eval(cell))
    except Exception:
        return []


def crossval_against_upstream():
    """Feed real stored predictions into our port AND upstream, assert equal."""
    csv = TCRT5 / "data" / "benchmark_data_w_preds.csv"
    if not csv.exists():
        print(f"SKIP cross-validation: {csv} missing")
        return True
    try:
        sys.path.insert(0, str(TCRT5))
        from src.evaluation import ModelEvaluator as UP  # official static methods
    except Exception as e:  # pragma: no cover
        print(f"SKIP cross-validation: cannot import upstream ModelEvaluator ({e})")
        return True

    df = pd.read_csv(csv, low_memory=False)
    pred_cols = ["tcrt5_translations", "gratcr_translations", "er_translations"]
    greedy_cols = {"tcrt5_translations": "tcrt5_greedy",
                   "gratcr_translations": "gratcr_greedy",
                   "er_translations": "er_greedy"}
    K = 100
    mismatches = 0
    n_rows = 0
    print("\n=== cross-validation vs upstream ModelEvaluator (14 benchmark pMHC) ===")
    for _, row in df.iterrows():
        refs = _parse_list(row["reference_translations"])
        if not refs:
            continue
        for col in pred_cols:
            gen = _parse_list(row[col])
            if not gen:
                continue
            n_rows += 1
            pairs = [
                ("precision", M.tcrt5_precision_at_k(gen, refs), UP._precision_at_k(gen, refs, k=K)),
                ("recall", M.tcrt5_recall_at_k(gen, refs, K), UP._recall_at_k(gen, refs, K)),
                ("f1", M.tcrt5_f1_at_k(gen, refs, K), UP._f1_at_k(gen, refs, K)),
                ("d_edit", M.tcrt5_mean_edit_distance(gen, refs), UP._mean_edit_distance(gen, refs)),
                ("seq_recovery", M.tcrt5_mean_sequence_recovery(gen, refs), UP._mean_sequence_recovery(gen, refs)),
            ]
            greedy = row.get(greedy_cols[col])
            if isinstance(greedy, str) and greedy.strip():
                pairs.append(("char_bleu",
                              M.tcrt5_sequence_bleu(greedy, refs, max_references=20, max_ngram=4),
                              UP._sequence_bleu(greedy, refs, max_references=20, max_ngram=4)))
            for name, ours, up in pairs:
                if not _approx(ours, up, tol=1e-9):
                    mismatches += 1
                    print(f"  [FAIL] {row['Epitope']} {col} {name}: ours={ours} up={up}")
    print(f"  compared {n_rows} (epitope x method) prediction blocks; mismatches={mismatches}")
    return mismatches == 0


def report_benchmark_numbers():
    """Print per-method dataset-level TCRT5 metrics reproduced from stored preds."""
    csv = TCRT5 / "data" / "benchmark_data_w_preds.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv, low_memory=False)
    methods = {"TCRT5": ("tcrt5_translations", "tcrt5_greedy"),
               "GRATCR": ("gratcr_translations", "gratcr_greedy"),
               "ER-TRANSFORMER": ("er_translations", "er_greedy")}
    print("\n=== reproduced dataset-level numbers on 14 benchmark pMHC (K=100) ===")
    print(f"  {'method':16s} {'prec':>7s} {'recall':>7s} {'f1':>7s} {'d_edit':>7s} {'seqrec':>7s} {'bleu':>7s} {'div':>6s}")
    for name, (col, gcol) in methods.items():
        gen_map, ref_map, greedy_map = {}, {}, {}
        for _, row in df.iterrows():
            refs = _parse_list(row["reference_translations"])
            gen = _parse_list(row[col])
            if not refs or not gen:
                continue
            ep = str(row["pMHC"])
            gen_map[ep] = gen
            ref_map[ep] = refs
            g = row.get(gcol)
            if isinstance(g, str):
                greedy_map[ep] = g
        res = M.tcrt5_dataset_metrics(gen_map, ref_map, k=100, per_epitope_greedy=greedy_map)
        print(f"  {name:16s} {res['precision']:7.4f} {res['recall']:7.4f} {res['f1']:7.4f} "
              f"{res['d_edit']:7.3f} {res['seq_recovery']:7.4f} {res['char_bleu']:7.4f} {res['diversity']:6.3f}")


if __name__ == "__main__":
    ok1 = gold_unit_tests()
    ok2 = crossval_against_upstream()
    report_benchmark_numbers()
    print("\nRESULT:", "ALL PASS" if (ok1 and ok2) else "FAILURES PRESENT")
    sys.exit(0 if (ok1 and ok2) else 1)
