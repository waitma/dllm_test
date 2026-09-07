"""Composition realism of the T4 Setting-B designs.

``d_edit`` is a nearest-reference edit distance, so it does not penalise a
degenerate residue composition -- a poly-glycine design set can score well.
This script reports the amino-acid composition JSD against the real binders
alongside glycine content, so a d_edit improvement can be checked for whether
it came from better sequences or just from a shorter path to the nearest
reference.
"""

from __future__ import annotations

import collections
import json
import pathlib
import re
import sys

import numpy as np

BENCH = pathlib.Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark")
B = BENCH / "outputs" / "tcr_generation_bench" / "setting_B"
GB = BENCH / "data" / "tcr_generation_bench"
AA = "ACDEFGHIKLMNPQRSTVWY"
RESERVED = "RVRAYTYSK_HLA-A*03:01"


def core(s: str) -> str:
    s = str(s).upper()
    if s.startswith("C"):
        s = s[1:]
    if s and s[-1] in "FW":
        s = s[:-1]
    return s


def comp(seqs) -> np.ndarray:
    j = "".join(core(s) for s in seqs if core(s))
    n = len(j) or 1
    return np.array([j.count(a) / n for a in AA])


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    p, q = p + 1e-12, q + 1e-12
    m = (p + q) / 2
    kl = lambda a, b: float(np.sum(a * np.log2(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def row(name: str, seqs, ref: np.ndarray) -> None:
    c = [core(s) for s in seqs if core(s)]
    if not c:
        return
    j = "".join(c)
    gggg = np.mean([bool(re.search(r"G{4,}", x)) for x in c]) * 100
    print(f"  {name:34s} n={len(c):4d} compJSD={jsd(comp(seqs), ref):7.4f} "
          f"G%={j.count('G') / len(j) * 100:5.1f} GGGG={gggg:5.1f}% "
          f"len={np.mean([len(x) for x in c]):5.2f} uniq={len(set(c)) / len(c) * 100:5.1f}%")


def main() -> None:
    ev = json.loads((GB / "eval_conditional.json").read_text())
    unseen = set(json.loads((GB / "bioseq_unseen_pmhc.json").read_text()))
    cs = sorted(p for p, e in ev.items()
                if e["category"] == "benchmark14" and p in unseen and p != RESERVED)
    refs = [r for p in cs for r in ev[p]["ref_binders"]]
    ref = comp(refs)

    print("Composition realism vs the real binders (compJSD 0 = identical)\n")
    row("REFERENCES (real binders)", refs, ref)
    for t in ("olga", "tcrdiff", "tcrdesign"):
        f = B / t / "designs.jsonl"
        if f.exists():
            row(f"{t} baseline",
                [x for l in f.read_text().splitlines() if l.strip()
                 for x in json.loads(l)["sequences"]], ref)

    for label, path in (("esmc189000", "/tmp/vs_s2_designs_esmc.json"),
                        ("v3_121000", "/tmp/vs_s2_designs_v3.json")):
        p = pathlib.Path(path)
        if not p.exists():
            continue
        print()
        for k, v in json.loads(p.read_text()).items():
            row(f"{label} {k}", v["seqs"], ref)


if __name__ == "__main__":
    main()
