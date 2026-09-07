"""Was esmc189000's training data decontaminated against the T4 eval set?

esmc189000 (``grammar_v2_esmc300m_integrated_llada_7l``) was trained 2026-07-17
on ``sources=oas,ots,nanobody,tcr_piste,tcr_pmhc_fulllength,ppi,neutralization``.
The T4 Setting-B blocklists (``t4_refbinder_blocklist.txt`` and the T2/T3 eval
blocklist) were only built in late August, so that run predates them by ~6
weeks. If PISTE carries the Setting-B reference binders, then esmc's 4.655
d_edit and its 56.7% J-region AAR are partly memorisation rather than
generalisation, and the "old architecture was better at TCR" reading collapses.

Reports, against the Setting-B references:
  * exact CDR3b-core overlap
  * (core | epitope) pair overlap -- the key that the project's own
    decontamination uses
  * edit-distance <= 1 near-duplicate overlap
and does the same for v3's corpora as the decontaminated control.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
GB = ROOT / "downstream" / "benchmark" / "data" / "tcr_generation_bench"
PISTE_DIR = ROOT / "data" / "ppi_task_raw" / "raw" / "piste_tcr_epitope_hla" / "PISTE" / "data"
RESERVED = "RVRAYTYSK_HLA-A*03:01"


def core(s) -> str:
    s = "".join(str(s or "").split()).upper()
    if len(s) >= 5:
        if s[0] == "C":
            s = s[1:]
        if s and s[-1] in "FW":
            s = s[:-1]
    return s


def load_eval():
    ev = json.loads((GB / "eval_conditional.json").read_text())
    unseen = set(json.loads((GB / "bioseq_unseen_pmhc.json").read_text()))
    common = sorted(p for p, e in ev.items()
                    if e["category"] == "benchmark14" and p in unseen and p != RESERVED)
    return ev, common


def piste_rows() -> pd.DataFrame:
    frames = []
    for f in sorted(PISTE_DIR.glob("**/*.csv")):
        try:
            d = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        if "CDR3" in d.columns:
            keep = ["CDR3"] + [c for c in ("MT_pep", "Label") if c in d.columns]
            d = d[keep].copy()
            d["__src"] = str(f.relative_to(PISTE_DIR))
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fulllength_rows() -> pd.DataFrame:
    q = ROOT / "data" / "tcr_pmhc_fulllength"
    frames = []
    for f in sorted(q.glob("**/*.csv")):
        try:
            d = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        cd = [c for c in d.columns if "cdr3" in c.lower() and "b" in c.lower()] or \
             [c for c in d.columns if "cdr3" in c.lower()]
        ep = [c for c in d.columns if any(t in c.lower() for t in ("epitope", "peptide", "pep"))]
        if not cd:
            continue
        out = pd.DataFrame({"CDR3": d[cd[0]]})
        out["MT_pep"] = d[ep[0]] if ep else None
        out["__src"] = str(f.relative_to(q))
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def report(name: str, df: pd.DataFrame, ev, pmhcs, label: str) -> None:
    if df.empty:
        print(f"  {name:34s} (no rows found)")
        return
    tr_core = set()
    tr_pair = set()
    for c, e in zip(df["CDR3"], df.get("MT_pep", pd.Series([None] * len(df)))):
        k = core(c)
        if not k:
            continue
        tr_core.add(k)
        if e is not None and not pd.isna(e):
            tr_pair.add(f"{k}|{str(e).strip().upper()}")

    refs_core, refs_pair = set(), set()
    for p in pmhcs:
        ep = ev[p]["epitope"].strip().upper()
        for r in ev[p]["ref_binders"]:
            k = core(r)
            if k:
                refs_core.add(k)
                refs_pair.add(f"{k}|{ep}")

    hit_core = refs_core & tr_core
    hit_pair = refs_pair & tr_pair
    print(f"  {name:34s} train_uniq={len(tr_core):7d}  "
          f"core命中 {len(hit_core):5d}/{len(refs_core):5d} = {len(hit_core)/max(1,len(refs_core))*100:5.1f}%   "
          f"(core|epitope)命中 {len(hit_pair):5d}/{len(refs_pair):5d} = "
          f"{len(hit_pair)/max(1,len(refs_pair))*100:5.1f}%")
    if hit_pair:
        ex = sorted(hit_pair)[:4]
        print(f"       泄漏样例: {ex}")


AA = "ACDEFGHIKLMNPQRSTVWY"


def lev1_variants(s: str):
    """All strings within edit distance 1 of ``s`` (sub / del / ins)."""
    yield s
    for i in range(len(s)):
        yield s[:i] + s[i + 1:]
        for a in AA:
            if a != s[i]:
                yield s[:i] + a + s[i + 1:]
    for i in range(len(s) + 1):
        for a in AA:
            yield s[:i] + a + s[i:]


def near_dup_report(name: str, df: pd.DataFrame, ev, pmhcs) -> None:
    """Edit-distance<=1 overlap, the sensitivity the project's own T2/T3
    blocklist uses. Exact-match cleanliness does not imply this is clean."""
    if df.empty:
        print(f"  {name:34s} (no rows)")
        return
    tr_core, tr_pair = set(), set()
    for c, e in zip(df["CDR3"], df.get("MT_pep", pd.Series([None] * len(df)))):
        k = core(c)
        if not k:
            continue
        tr_core.add(k)
        if e is not None and not pd.isna(e):
            tr_pair.add(f"{k}|{str(e).strip().upper()}")

    n_core = n_pair = tot = 0
    for p in pmhcs:
        ep = ev[p]["epitope"].strip().upper()
        for r in ev[p]["ref_binders"]:
            k = core(r)
            if not k:
                continue
            tot += 1
            hit_c = hit_p = False
            for v in lev1_variants(k):
                if not hit_c and v in tr_core:
                    hit_c = True
                if not hit_p and f"{v}|{ep}" in tr_pair:
                    hit_p = True
                if hit_c and hit_p:
                    break
            n_core += hit_c
            n_pair += hit_p
    print(f"  {name:34s} Lev<=1 core {n_core:4d}/{tot:4d} = {n_core/max(1,tot)*100:5.1f}%   "
          f"Lev<=1 (core|epitope) {n_pair:4d}/{tot:4d} = {n_pair/max(1,tot)*100:5.1f}%")


def main() -> None:
    ev, common = load_eval()
    all_p = sorted(ev)
    print("T4 Setting-B 参考答案 vs 各训练语料的重叠\n")
    print(f"评测切片: bioseq_unseen_common {len(common)} 个 pMHC / 全部 {len(all_p)} 个\n")

    piste = piste_rows()
    full = fulllength_rows()
    print("=== esmc189000 的 TCR 源（2026-07 训练，早于 8 月底的 T4 去污名单）===")
    for scope, pm in (("common-unseen 6", common), ("全部 34", all_p)):
        print(f"  -- 对照 {scope} pMHC")
        report("tcr_piste (PISTE 全部 split)", piste, ev, pm, scope)
        report("tcr_pmhc_fulllength", full, ev, pm, scope)

    print("\n=== v3 的 TCR 源（已过 8 月底去污）作为对照 ===")
    for name, path, cdr, epi in (
        ("tcr_papers_v2", ROOT / "data/tcr_papers_v2/dataset/train.csv", "cdr3b", "epitope_seq"),
        ("tcr_native", ROOT / "data/tcr_native/dataset/train.csv", "cdr3b", "epitope_seq"),
    ):
        d = pd.read_csv(path, low_memory=False, usecols=[cdr, epi])
        d = d.rename(columns={cdr: "CDR3", epi: "MT_pep"})
        for scope, pm in (("common-unseen 6", common), ("全部 34", all_p)):
            report(f"{name} [{scope}]", d, ev, pm, scope)

    print("\n=== 近似重复（Lev<=1），只看 common-unseen 6 pMHC ===")
    print("    精确匹配为 0 不代表这一项也为 0，这是更敏感的口径")
    near_dup_report("tcr_piste  (esmc189000)", piste, ev, common)
    for name, path in (("tcr_papers_v2 (v3)", ROOT / "data/tcr_papers_v2/dataset/train.csv"),
                       ("tcr_native    (v3)", ROOT / "data/tcr_native/dataset/train.csv")):
        d = pd.read_csv(path, low_memory=False, usecols=["cdr3b", "epitope_seq"])
        d = d.rename(columns={"cdr3b": "CDR3", "epitope_seq": "MT_pep"})
        near_dup_report(name, d, ev, common)


if __name__ == "__main__":
    main()
