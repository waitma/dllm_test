"""Which TCR corpus actually supports learning the germline J motif?

esmc189000 (2026-07) learned the J region; v3 (2026-09) did not, and the two
were trained on entirely different TCR sources. Two candidate explanations:

  (A) convention -- esmc saw full ``C..F`` junctions, v3 saw anchor-free cores.
  (B) corpus content -- redundancy and J-region entropy differ sharply.

(A) is already weak: v3 scores the same whether it is fed cores or anchored
input (28.05% vs 27.58% J AAR), so its deficit is not an input-format artifact.

This tests (B) without training anything large. A cheap backoff n-gram is fit
on each corpus and evaluated on the SAME held-out target: the Setting-B
reference binders of the 6 common-unseen pMHC, which carry 0.0% (core|epitope)
leakage against every corpus (verified by check_esmc_leakage.py, exact and
Lev<=1). Corpora are subsampled to a common row budget so the comparison is
about content rather than volume.

If the proxy fit on PISTE predicts the J region better than the proxy fit on
v3's sources, then the corpus -- not the convention and not the decoder -- is
what changed between the two model generations.
"""

from __future__ import annotations

import collections
import json
import math
import pathlib
import random

import numpy as np
import pandas as pd

ROOT = pathlib.Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
GB = ROOT / "downstream" / "benchmark" / "data" / "tcr_generation_bench"
PISTE = ROOT / "data/ppi_task_raw/raw/piste_tcr_epitope_hla/PISTE/data"
WIN = 5
RESERVED = "RVRAYTYSK_HLA-A*03:01"


def core(s) -> str:
    s = "".join(str(s or "").split()).upper()
    if len(s) >= 5:
        if s[0] == "C":
            s = s[1:]
        if s and s[-1] in "FW":
            s = s[:-1]
    return s


# --------------------------------------------------------------------------- #
# backoff n-gram over the J window
# --------------------------------------------------------------------------- #

class JModel:
    """P(residue | offset-in-window, preceding 2 residues) with backoff.

    Deliberately simple and identical across corpora: any difference in test
    accuracy is a property of the training distribution, not of model capacity.
    """

    def __init__(self) -> None:
        self.c2: dict = collections.defaultdict(collections.Counter)
        self.c1: dict = collections.defaultdict(collections.Counter)
        self.c0: dict = collections.defaultdict(collections.Counter)
        self.g: collections.Counter = collections.Counter()

    def fit(self, cores) -> "JModel":
        for s in cores:
            if len(s) < WIN + 3:
                continue
            w = s[-WIN:]
            pre = s[:-WIN]
            for i, ch in enumerate(w):
                ctx = (pre + w[:i])
                p2, p1 = ctx[-2:], ctx[-1:]
                self.c2[(i, p2)][ch] += 1
                self.c1[(i, p1)][ch] += 1
                self.c0[i][ch] += 1
                self.g[ch] += 1
        return self

    def predict(self, ctx: str, i: int) -> str:
        for table, key in ((self.c2, (i, ctx[-2:])), (self.c1, (i, ctx[-1:])), (self.c0, i)):
            d = table.get(key)
            if d and sum(d.values()) >= 5:
                return d.most_common(1)[0][0]
        return self.g.most_common(1)[0][0] if self.g else "G"

    def score(self, targets) -> dict:
        hit = tot = exact = n = 0
        for s in targets:
            if len(s) < WIN + 3:
                continue
            w, pre = s[-WIN:], s[:-WIN]
            pred = []
            for i in range(WIN):
                # teacher-forced context, mirroring how the real probe measured AAR
                pred.append(self.predict(pre + w[:i], i))
            p = "".join(pred)
            hit += sum(a == b for a, b in zip(w, p))
            tot += WIN
            exact += p == w
            n += 1
        return {"aar": hit / max(1, tot), "exact": exact / max(1, n), "n": n}


# --------------------------------------------------------------------------- #

def load_targets():
    ev = json.loads((GB / "eval_conditional.json").read_text())
    unseen = set(json.loads((GB / "bioseq_unseen_pmhc.json").read_text()))
    common = sorted(p for p, e in ev.items()
                    if e["category"] == "benchmark14" and p in unseen and p != RESERVED)
    return [core(r) for p in common for r in ev[p]["ref_binders"] if core(r)], len(common)


def piste_cores():
    frames = []
    for f in sorted(PISTE.glob("**/*.csv")):
        try:
            d = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        if "CDR3" in d.columns:
            frames.append(d[["CDR3"]])
    d = pd.concat(frames, ignore_index=True)
    return [core(x) for x in d["CDR3"].dropna()]


def csv_cores(path, col="cdr3b"):
    d = pd.read_csv(path, low_memory=False, usecols=[col])
    return [core(x) for x in d[col].dropna()]


def repertoire_cores():
    f = sorted((ROOT / "data/tcr_repertoire").glob("**/train*.csv"))[:1]
    if not f:
        return []
    d = pd.read_csv(f[0], low_memory=False)
    col = [c for c in d.columns if c.lower() in ("cdr3b", "cdr3", "junction_aa")][0]
    return [core(x) for x in d[col].dropna()]


def entropy(cores) -> float:
    t = collections.Counter(s[-WIN:] for s in cores if len(s) >= WIN)
    n = sum(t.values()) or 1
    return -sum((v / n) * math.log2(v / n) for v in t.values())


def main() -> None:
    targets, n_pmhc = load_targets()
    print(f"共同测试目标: bioseq_unseen_common {n_pmhc} 个 pMHC 的 {len(targets)} 条参考序列")
    print("（对所有语料的 (core|epitope) 泄漏均为 0.0%，精确与 Lev<=1 都验证过）\n")

    corpora = {
        "tcr_piste      (esmc189000)": piste_cores(),
        "tcr_papers_v2  (v3)": csv_cores(ROOT / "data/tcr_papers_v2/dataset/train.csv"),
        "tcr_native     (v3)": csv_cores(ROOT / "data/tcr_native/dataset/train.csv"),
        "tcr_repertoire (v3, 最大源)": repertoire_cores(),
    }

    BUDGET = 140_000  # rows, so volume is held constant across corpora
    rng = random.Random(0)
    print(f"{'语料':30s} {'行数':>9s} {'唯一':>8s} {'重复':>6s} {'J熵':>6s} | "
          f"{'全量AAR':>8s} {'全量exact':>9s} | {f'{BUDGET//1000}k子样AAR':>12s} {'exact':>7s}")
    for name, cores in corpora.items():
        cores = [c for c in cores if len(c) >= WIN + 3]
        if not cores:
            continue
        full = JModel().fit(cores).score(targets)
        sub = cores if len(cores) <= BUDGET else rng.sample(cores, BUDGET)
        s = JModel().fit(sub).score(targets)
        uniq = len(set(cores))
        print(f"  {name:28s} {len(cores):9d} {uniq:8d} {len(cores)/uniq:5.2f}x "
              f"{entropy(cores):6.2f} | {full['aar']:8.3f} {full['exact']:9.3f} | "
              f"{s['aar']:12.3f} {s['exact']:7.3f}")

    # v3 actually trained on the union of its three sources; score that too.
    union = []
    for k, v in corpora.items():
        if "(v3" in k:
            union += [c for c in v if len(c) >= WIN + 3]
    if union:
        u = JModel().fit(union).score(targets)
        us = JModel().fit(random.Random(1).sample(union, min(BUDGET, len(union)))).score(targets)
        print(f"  {'v3 三源并集':28s} {len(union):9d} {len(set(union)):8d} "
              f"{len(union)/len(set(union)):5.2f}x {entropy(union):6.2f} | "
              f"{u['aar']:8.3f} {u['exact']:9.3f} | {us['aar']:12.3f} {us['exact']:7.3f}")

    print("\n对照: 真实模型 teacher-forced 的 J 区 AAR")
    print("  esmc189000 (anchored, 自己的约定)  0.567")
    print("  v3_121000  (core, 自己的约定)      0.281   ← 低于位置频率表 0.360")


if __name__ == "__main__":
    main()
