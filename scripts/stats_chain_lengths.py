#!/usr/bin/env python3
"""统计抗体 (heavy/light) 与 TCR (alpha/beta) 各链的长度分布。

支持三种数据格式：
1. CSV (OAS / OTS 上游 split)
2. prepared JSONL: {"chains": [seq, ...], "chain_roles": ["antibody_heavy", ...]}
3. canonical JSONL: {"entities": [{"role": "antibody_heavy", "sequence": ..., "sequence_scope": ...}]}

按数据源分别输出统计，并给出全局汇总和定长建议。
用法: python scripts/stats_chain_lengths.py [--full-prepared]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

ROLE_ALIASES = {
    "antibody_heavy": "heavy",
    "heavy": "heavy",
    "antibody_light": "light",
    "light": "light",
    "tcr_alpha": "alpha",
    "alpha": "alpha",
    "tcr_beta": "beta",
    "beta": "beta",
}

PERCENTILES = (50, 90, 95, 99, 99.5, 99.9, 99.99)


def norm_role(role: str) -> str | None:
    return ROLE_ALIASES.get(str(role).lower())


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


class Collector:
    """lengths[source][role] -> list[int]"""

    def __init__(self) -> None:
        self.lengths: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

    def add(self, source: str, role: str, length: int) -> None:
        self.lengths[source][role].append(length)

    def read_oas_csv(self, path: Path, source: str) -> None:
        with path.open("r", encoding="utf-8") as f:
            for row in tqdm(csv.DictReader(f), desc=source):
                h = row.get("cleaned_h_sequence") or row.get("h_sequence") or ""
                l = row.get("cleaned_l_sequence") or row.get("l_sequence") or ""
                if h and h != "nan":
                    self.add(source, "heavy", len(h))
                if l and l != "nan":
                    self.add(source, "light", len(l))

    def read_ots_csv(self, path: Path, source: str) -> None:
        with path.open("r", encoding="utf-8") as f:
            for row in tqdm(csv.DictReader(f), desc=source):
                for seq_key, type_key in (
                    ("cleaned_chain1_seq", "chain1_type"),
                    ("cleaned_chain2_seq", "chain2_type"),
                ):
                    seq = row.get(seq_key) or ""
                    role = norm_role(row.get(type_key) or "")
                    if seq and seq != "nan" and role:
                        self.add(source, role, len(seq))

    def read_prepared_jsonl(self, path: Path, source: str) -> None:
        for rec in tqdm(iter_jsonl(path), desc=source):
            chains = rec.get("chains") or []
            roles = rec.get("chain_roles") or []
            for seq, role in zip(chains, roles):
                r = norm_role(role)
                if r and seq:
                    self.add(source, r, len(seq))

    def read_canonical_jsonl(self, path: Path, source: str) -> None:
        for rec in tqdm(iter_jsonl(path), desc=source):
            for ent in rec.get("entities") or []:
                r = norm_role(ent.get("role") or "")
                seq = ent.get("sequence") or ""
                if not (r and seq):
                    continue
                scope = ent.get("sequence_scope") or "unknown"
                self.add(f"{source}[{scope}]", r, len(seq))


def describe(arr: list[int]) -> dict:
    a = np.asarray(arr)
    out = {
        "count": int(a.size),
        "min": int(a.min()),
        "max": int(a.max()),
        "mean": round(float(a.mean()), 1),
    }
    for p in PERCENTILES:
        out[f"p{p}"] = float(np.percentile(a, p))
    return out


def fmt_row(label: str, s: dict) -> str:
    return (
        f"{label:<44} n={s['count']:>10,}  min={s['min']:>4}  max={s['max']:>4}  "
        f"mean={s['mean']:>6}  p50={s['p50']:>5.0f}  p99={s['p99']:>5.0f}  "
        f"p99.9={s['p99.9']:>5.0f}  p99.99={s['p99.99']:>5.0f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-prepared", action="store_true", help="扫描 prepared 目录全部 shard（默认每源 3 个）")
    ap.add_argument("--out", default="scripts/chain_length_stats.json")
    args = ap.parse_args()

    root = Path.cwd()
    c = Collector()

    # 1) 上游 CSV split（OAS / OTS，全量）
    oas = root / "data/oas_previous_clean/splits/cleaned_merged_data_step_clustered_train_oas_label.csv"
    if oas.exists():
        c.read_oas_csv(oas, "oas_csv_train")
    ots = root / "data/ots_paired_clean/final/train.csv"
    if ots.exists():
        c.read_ots_csv(ots, "ots_csv_train")

    # 2) canonical v2（含 sequence_scope）
    for sub in ("antibody", "tcr"):
        d = root / "data/immune_receptor_v2/canonical" / sub
        if d.exists():
            for p in sorted(d.glob("*.jsonl")):
                if p.name.endswith("_union.jsonl"):
                    continue
                c.read_canonical_jsonl(p, f"canon/{p.stem}")

    # 3) prepared immune_v5（当前 runtime 输入）
    prep = root / "data/prepared/immune_v5_receptor_completion/train"
    if prep.exists():
        groups: dict[str, list[Path]] = defaultdict(list)
        for p in sorted(prep.glob("*.jsonl")):
            groups[p.name.rsplit("-", 1)[0]].append(p)
        for prefix, files in groups.items():
            chosen = files if args.full_prepared else files[:3]
            for p in chosen:
                c.read_prepared_jsonl(p, f"prep/{prefix}")

    # ---- 报告 --------------------------------------------------------
    report: dict[str, dict[str, dict]] = {}
    print("\n" + "=" * 150)
    print("按数据源 / 链类型统计（单位：氨基酸残基数）")
    print("=" * 150)
    for source in sorted(c.lengths):
        for role in ("heavy", "light", "alpha", "beta"):
            arr = c.lengths[source].get(role)
            if not arr:
                continue
            s = describe(arr)
            report.setdefault(source, {})[role] = s
            print(fmt_row(f"{source} / {role}", s))

    # 全局按链类型（只汇总 variable_domain 级别的全长链）
    merged: dict[str, list[int]] = defaultdict(list)
    for source, roles in c.lengths.items():
        if "[" in source and "variable_domain" not in source and "full" not in source:
            continue
        for role, arr in roles.items():
            merged[role].extend(arr)

    print("\n" + "=" * 150)
    print("全局汇总（全长可变区链；排除 canonical 中 CDR3-only 等非全长 scope）")
    print("=" * 150)
    global_stats = {}
    for role in ("heavy", "light", "alpha", "beta"):
        if merged.get(role):
            global_stats[role] = describe(merged[role])
            print(fmt_row(f"ALL / {role}", global_stats[role]))

    print("\n" + "=" * 150)
    print("定长建议（按覆盖比例，向上取整到 8 的倍数；不含 <cls>/<eos>）")
    print("=" * 150)
    rec = {}
    for role, s in global_stats.items():
        line = {f"cover_{p}": int(np.ceil(s[f"p{p}"] / 8) * 8) for p in (99, 99.9, 99.99)}
        line["cover_100"] = int(s["max"])
        rec[role] = line
        print(
            f"{role:<8} 99%→{line['cover_99']:>4}   99.9%→{line['cover_99.9']:>4}   "
            f"99.99%→{line['cover_99.99']:>4}   100%→{line['cover_100']:>4}"
        )

    if global_stats:
        ab_max = max(global_stats[r]["max"] for r in ("heavy", "light") if r in global_stats)
        tcr_max = max(global_stats[r]["max"] for r in ("alpha", "beta") if r in global_stats)
        ab_999 = max(global_stats[r]["p99.9"] for r in ("heavy", "light") if r in global_stats)
        tcr_999 = max(global_stats[r]["p99.9"] for r in ("alpha", "beta") if r in global_stats)
        print()
        print(f"抗体 (H/L) : max={ab_max}, p99.9={ab_999:.0f}")
        print(f"TCR (α/β)  : max={tcr_max}, p99.9={tcr_999:.0f}")
        print(f"统一单链定长 (覆盖 99.9%, +2 token, 8对齐): {int(np.ceil((max(ab_999, tcr_999) + 2) / 8) * 8)}")
        print(f"统一单链定长 (覆盖 100%,  +2 token, 8对齐): {int(np.ceil((max(ab_max, tcr_max) + 2) / 8) * 8)}")

    # 保存结果
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    result = {"by_source": report, "global": global_stats, "recommendations": rec}
    with out_path.open("w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {out_path}")


if __name__ == "__main__":
    main()
