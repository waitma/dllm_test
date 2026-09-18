#!/usr/bin/env python3
"""
重新统计链长度，只保留可变区数据（variable_domain scope）。
排除 full_chain / cdr3 等非可变区全长的 scope。
"""

from __future__ import annotations

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
        """只保留 variable_domain scope 的序列"""
        for rec in tqdm(iter_jsonl(path), desc=source):
            for ent in rec.get("entities") or []:
                r = norm_role(ent.get("role") or "")
                seq = ent.get("sequence") or ""
                scope = ent.get("sequence_scope") or "unknown"
                
                # 只保留 variable_domain
                if not (r and seq and scope == "variable_domain"):
                    continue
                    
                self.add(f"{source}[variable_domain]", r, len(seq))


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
    root = Path.cwd()
    c = Collector()

    # 1) 上游 CSV split（OAS / OTS，全量）
    print("Reading upstream CSV splits...")
    oas = root / "data/oas_previous_clean/splits/cleaned_merged_data_step_clustered_train_oas_label.csv"
    if oas.exists():
        c.read_oas_csv(oas, "oas_csv_train")
    ots = root / "data/ots_paired_clean/final/train.csv"
    if ots.exists():
        c.read_ots_csv(ots, "ots_csv_train")

    # 2) canonical v2（只保留 variable_domain scope）
    print("\nReading canonical variable_domain data...")
    for sub in ("antibody", "tcr"):
        d = root / "data/immune_receptor_v2/canonical" / sub
        if d.exists():
            for p in sorted(d.glob("*.jsonl")):
                if p.name.endswith("_union.jsonl"):
                    continue
                c.read_canonical_jsonl(p, f"canon/{p.stem}")

    # 3) prepared immune_v5（采样前 5 个文件）
    print("\nReading prepared data (sampling 5 files per source)...")
    prep_root = root / "data/prepared/immune_v5_receptor_completion/train"
    if prep_root.exists():
        for prefix in ("oas", "ots", "asd_antibody", "tcr_native", "tcr_papers", "tcr_repertoire", "trait"):
            files = sorted(prep_root.glob(f"{prefix}-*.jsonl"))[:5]
            for p in files:
                c.read_prepared_jsonl(p, f"prep/{prefix}")

    # 计算全局统计
    print("\nComputing global statistics...")
    global_lengths = defaultdict(list)
    for src, roles in c.lengths.items():
        for role, lens in roles.items():
            global_lengths[role].extend(lens)

    # 输出
    print("\n" + "=" * 100)
    print("按数据源 / 链类型统计（仅 variable_domain，单位：氨基酸残基数）")
    print("=" * 100)
    
    for src in sorted(c.lengths.keys()):
        for role in sorted(c.lengths[src].keys()):
            stat = describe(c.lengths[src][role])
            print(fmt_row(f"{src} / {role}", stat))

    print("\n" + "=" * 100)
    print("全局汇总（仅 variable_domain）")
    print("=" * 100)
    
    for role in ["heavy", "light", "alpha", "beta"]:
        if role in global_lengths:
            stat = describe(global_lengths[role])
            print(fmt_row(f"ALL / {role}", stat))

    # 推荐定长
    print("\n" + "=" * 100)
    print("定长建议（向上取整到 8 的倍数，包含 +2 用于特殊 token）")
    print("=" * 100)
    
    recommendations = {}
    for role in ["heavy", "light", "alpha", "beta"]:
        if role not in global_lengths:
            continue
        stat = describe(global_lengths[role])
        recommendations[role] = {
            "cover_99": int(np.ceil((stat["p99"] + 2) / 8) * 8),
            "cover_99.9": int(np.ceil((stat["p99.9"] + 2) / 8) * 8),
            "cover_99.99": int(np.ceil((stat["p99.99"] + 2) / 8) * 8),
            "cover_100": int(np.ceil((stat["max"] + 2) / 8) * 8),
        }
        print(f"{role:<8} 99%→ {recommendations[role]['cover_99']:<4}  "
              f"99.9%→ {recommendations[role]['cover_99.9']:<4}  "
              f"99.99%→ {recommendations[role]['cover_99.99']:<4}  "
              f"100%→ {recommendations[role]['cover_100']}")

    # 统一配置建议
    all_max = max(global_lengths[r] for r in ["heavy", "light", "alpha", "beta"] if r in global_lengths)
    all_p999 = max(np.percentile(global_lengths[r], 99.9) for r in ["heavy", "light", "alpha", "beta"] if r in global_lengths)
    
    unified_p999 = int(np.ceil((all_p999 + 2) / 8) * 8)
    unified_max = int(np.ceil((all_max + 2) / 8) * 8)
    
    print(f"\n抗体 (H/L) : max={max(global_lengths['heavy'])}, p99.9={np.percentile(global_lengths['heavy'], 99.9):.0f}")
    print(f"TCR (α/β)  : max={max(global_lengths['beta'])}, p99.9={np.percentile(global_lengths['beta'], 99.9):.0f}")
    print(f"统一单链定长 (覆盖 99.9%, +2 token, 8对齐): {unified_p999}")
    print(f"统一单链定长 (覆盖 100%,  +2 token, 8对齐): {unified_max}")

    # 保存 JSON
    out_path = root / "scripts/chain_length_stats_variable_domain_only.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "by_source": {src: {role: describe(lens) for role, lens in roles.items()}
                         for src, roles in c.lengths.items()},
            "global": {role: describe(lens) for role, lens in global_lengths.items()},
            "recommendations": recommendations,
        }, f, indent=2)
    
    print(f"\n结果已保存到: {out_path}")


if __name__ == "__main__":
    main()
