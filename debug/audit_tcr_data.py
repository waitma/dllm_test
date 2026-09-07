"""Integrity audit of the TCR training corpora (hypothesis: our data processing
is broken).

Composition was already cleared -- glycine content matches the eval references
(11.4% vs 11.8%) and 6+ glycine runs are 0.00% in training vs 31.7% in v3's
mode. So this looks for *structural* damage instead:

  A illegal characters / length outliers
  B over-stripped anchors. ``_cdr3_core_key`` removes a leading C and trailing
    F/W; ``tcr_native_exclusion_key`` notes the unified column already holds the
    core, so a second strip would eat a real residue. A core that genuinely
    starts with C or ends in F/W is exactly the case that would be corrupted.
  C cdr3b vs beta_fv agreement. tcr_native carries both, so the core must appear
    verbatim inside the full-length chain -- the strongest available check that
    the extraction is right.
  D duplicate/degenerate rows and epitope-side damage
"""

from __future__ import annotations

import collections
import pathlib
import re

import numpy as np
import pandas as pd

ROOT = pathlib.Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
AA = set("ACDEFGHIKLMNPQRSTVWY")
SOURCES = {
    "tcr_papers_v2": ROOT / "data/tcr_papers_v2/dataset/train.csv",
    "tcr_native": ROOT / "data/tcr_native/dataset/train.csv",
}


def hdr(t: str) -> None:
    print("\n" + "=" * 76)
    print(t)
    print("=" * 76)


def audit_a(name: str, cb: pd.Series) -> None:
    s = cb.astype(str)
    bad_chars = s[~s.map(lambda x: set(x) <= AA)]
    lens = s.str.len()
    print(f"  {name:16s} n={len(s):7d}  非法字符行={len(bad_chars):5d}  "
          f"len min/med/max = {lens.min()}/{int(lens.median())}/{lens.max()}  "
          f"len<5={int((lens < 5).sum()):4d}  len>25={int((lens > 25).sum()):4d}")
    if len(bad_chars):
        ex = bad_chars.head(4).tolist()
        chars = collections.Counter(c for x in bad_chars for c in x if c not in AA)
        print(f"       非法样例: {ex}")
        print(f"       非法字符统计: {dict(chars.most_common(8))}")


def audit_b(name: str, cb: pd.Series) -> None:
    """Signature of a second anchor strip: a real J motif would be truncated.

    Cores are stored anchor-free, so ~0% should start with C. But the tail is
    the tell: a correctly-stripped core often ends in F (e.g. YNEQF, TGELF),
    whereas over-stripping would have removed that F and left the preceding
    residue, depleting F-terminal cores relative to the reference set.
    """
    s = cb.astype(str)
    s = s[s.str.len() >= 6]
    startC = float((s.str[0] == "C").mean())
    endF = float(s.str[-1].isin(["F"]).mean())
    endW = float(s.str[-1].isin(["W"]).mean())
    endY = float(s.str[-1].isin(["Y"]).mean())
    print(f"  {name:16s} 以C开头={startC*100:5.2f}%  以F结尾={endF*100:5.2f}%  "
          f"以W结尾={endW*100:5.2f}%  以Y结尾={endY*100:5.2f}%")


def audit_c() -> None:
    """cdr3b must appear verbatim inside beta_fv where both are present."""
    df = pd.read_csv(SOURCES["tcr_native"], low_memory=False,
                     usecols=["cdr3b", "beta_fv", "sequence_scope"])
    both = df[df["cdr3b"].notna() & df["beta_fv"].notna()]
    if both.empty:
        print("  tcr_native: 没有同时含 cdr3b 与 beta_fv 的行")
        return
    cb = both["cdr3b"].astype(str).str.upper()
    fv = both["beta_fv"].astype(str).str.upper()
    inside = [c in f for c, f in zip(cb, fv)]
    # if the core were over-stripped, re-attaching the anchors should restore it
    anchored_inside = [("C" + c + "F") in f or ("C" + c + "W") in f
                       for c, f in zip(cb, fv)]
    print(f"  tcr_native 同时有 cdr3b+beta_fv 的行: {len(both)}")
    print(f"    cdr3b 原样出现在 beta_fv 中      : {np.mean(inside)*100:6.2f}%")
    print(f"    补回 C..F/W 后出现在 beta_fv 中  : {np.mean(anchored_inside)*100:6.2f}%")
    miss = [(c, f) for c, f, ok in zip(cb, fv, inside) if not ok][:3]
    for c, f in miss:
        print(f"    ✗ core={c}  fv={f[:60]}...")


def audit_d(name: str, df: pd.DataFrame) -> None:
    cb = df["cdr3b"].dropna().astype(str)
    ep = df["epitope_seq"].dropna().astype(str) if "epitope_seq" in df else pd.Series(dtype=str)
    dup = len(cb) - cb.nunique()
    homo = cb[cb.map(lambda x: len(set(x)) <= 2)]
    ep_bad = ep[~ep.map(lambda x: set(x.upper()) <= AA)] if len(ep) else pd.Series(dtype=str)
    print(f"  {name:16s} cdr3b 重复行={dup:7d} ({dup/max(1,len(cb))*100:5.1f}%)  "
          f"低复杂度(<=2种残基)={len(homo):4d}  epitope 非法={len(ep_bad):5d}")
    if len(ep):
        print(f"       epitope: 唯一={ep.nunique():5d}  长度 med={int(ep.str.len().median())}  "
              f"min={ep.str.len().min()}  max={ep.str.len().max()}")


def main() -> None:
    hdr("A. 非法字符与长度异常")
    frames = {}
    for name, path in SOURCES.items():
        cols = ["cdr3b", "epitope_seq"]
        df = pd.read_csv(path, low_memory=False,
                         usecols=lambda c: c in cols + ["beta_fv", "sequence_scope"])
        frames[name] = df
        audit_a(name, df["cdr3b"].dropna())

    hdr("B. 锚点是否被二次剥离（末端残基分布）")
    print("  参考：正确剥离的 core 常以 F 结尾（YNEQF/TGELF）或 Y 结尾（TDTQY/SYEQY）")
    for name, df in frames.items():
        audit_b(name, df["cdr3b"].dropna())
    # the eval references, stripped once, are the ground truth for this shape
    import json
    ev = json.loads((ROOT / "downstream/benchmark/data/tcr_generation_bench"
                     / "eval_conditional.json").read_text())
    refs = pd.Series([r[1:-1] for e in ev.values() for r in e["ref_binders"]])
    audit_b("EVAL参考(剥一次)", refs)

    hdr("C. cdr3b 与 beta_fv 的一致性（最强的提取正确性检查）")
    audit_c()

    hdr("D. 重复与退化行")
    for name, path in SOURCES.items():
        df = pd.read_csv(path, low_memory=False,
                         usecols=lambda c: c in ["cdr3b", "epitope_seq"])
        audit_d(name, df)


if __name__ == "__main__":
    main()
