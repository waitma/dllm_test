#!/usr/bin/env python3
"""
Inspect the longest chains from the statistics to verify data quality.
"""
import json
import csv
import sys
from pathlib import Path
from collections import defaultdict

csv.field_size_limit(sys.maxsize)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"


def read_oas_csv(csv_path, top_n=5):
    """Extract top N longest heavy/light chains from OAS CSV."""
    chains = {"heavy": [], "light": []}
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Match stats_chain_lengths.py field names
            h_seq = row.get("cleaned_h_sequence") or row.get("h_sequence") or ""
            l_seq = row.get("cleaned_l_sequence") or row.get("l_sequence") or ""
            
            if h_seq and h_seq != "nan":
                chains["heavy"].append((len(h_seq), h_seq, row.get("sequence_id", "unknown")))
            if l_seq and l_seq != "nan":
                chains["light"].append((len(l_seq), l_seq, row.get("sequence_id", "unknown")))
    
    # Sort by length descending and take top N
    for role in chains:
        chains[role].sort(key=lambda x: x[0], reverse=True)
        chains[role] = chains[role][:top_n]
    
    return chains


def read_ots_csv(csv_path, top_n=5):
    """Extract top N longest alpha/beta chains from OTS CSV."""
    chains = {"alpha": [], "beta": []}
    
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
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Match stats_chain_lengths.py: cleaned_chain1_seq, cleaned_chain2_seq
            for seq_key, type_key in (
                ("cleaned_chain1_seq", "chain1_type"),
                ("cleaned_chain2_seq", "chain2_type"),
            ):
                seq = row.get(seq_key) or ""
                role_raw = row.get(type_key) or ""
                role = ROLE_ALIASES.get(str(role_raw).lower())
                
                if seq and seq != "nan" and role in ("alpha", "beta"):
                    chains[role].append((len(seq), seq, row.get("complex.id", "unknown")))
    
    for role in chains:
        chains[role].sort(key=lambda x: x[0], reverse=True)
        chains[role] = chains[role][:top_n]
    
    return chains


def read_canonical_jsonl(jsonl_path, top_n=5):
    """Extract top N longest chains from canonical JSONL files."""
    chains = defaultdict(list)
    
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
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            
            # Match stats_chain_lengths.py: entities with role, sequence, sequence_scope
            for ent in record.get("entities") or []:
                role_raw = ent.get("role") or ""
                role = ROLE_ALIASES.get(str(role_raw).lower())
                seq = ent.get("sequence") or ""
                scope = ent.get("sequence_scope") or "unknown"
                
                if role and seq:
                    rec_id = record.get("biological_key", record.get("id", "unknown"))
                    chains[role].append((len(seq), seq, rec_id, scope))
    
    # Sort and take top N
    result = {}
    for role in chains:
        chains[role].sort(key=lambda x: x[0], reverse=True)
        result[role] = chains[role][:top_n]
    
    return result


def main():
    print("=" * 100)
    print("Inspecting Longest Chains from Training Data")
    print("=" * 100)
    
    # 1. OAS CSV - Heavy/Light
    print("\n### OAS CSV (Antibody Heavy/Light)")
    oas_path = DATA_ROOT / "oas_previous_clean/splits/cleaned_merged_data_step_clustered_train_oas_label.csv"
    if oas_path.exists():
        oas_chains = read_oas_csv(oas_path, top_n=3)
        for role, chain_list in oas_chains.items():
            print(f"\n{role.upper()} - Top 3 longest:")
            for length, seq, seq_id in chain_list:
                print(f"  Length: {length}, ID: {seq_id}")
                print(f"  Seq: {seq[:80]}..." if len(seq) > 80 else f"  Seq: {seq}")
    else:
        print(f"  File not found: {oas_path}")
    
    # 2. OTS CSV - Alpha/Beta
    print("\n\n### OTS CSV (TCR Alpha/Beta)")
    ots_path = DATA_ROOT / "ots_paired_clean/final/train.csv"
    if ots_path.exists():
        ots_chains = read_ots_csv(ots_path, top_n=3)
        for role, chain_list in ots_chains.items():
            print(f"\n{role.upper()} - Top 3 longest:")
            for length, seq, seq_id in chain_list:
                print(f"  Length: {length}, ID: {seq_id}")
                print(f"  Seq: {seq[:80]}..." if len(seq) > 80 else f"  Seq: {seq}")
    else:
        print(f"  File not found: {ots_path}")
    
    # 3. Canonical JSONL - check full_chain scope files
    print("\n\n### Canonical JSONL - Full Chain Scope (Suspected Long Chains)")
    
    canonical_files = [
        "immune_receptor_v2/canonical/antibody/abrank.jsonl",
        "immune_receptor_v2/canonical/antibody/catnap.jsonl",
        "immune_receptor_v2/canonical/tcr/fullchain_derived.jsonl",
    ]
    
    for rel_path in canonical_files:
        jsonl_path = DATA_ROOT / rel_path
        if not jsonl_path.exists():
            print(f"\n{rel_path}: NOT FOUND")
            continue
        
        print(f"\n{rel_path}:")
        chains = read_canonical_jsonl(jsonl_path, top_n=3)
        for role, chain_list in chains.items():
            if chain_list:
                print(f"  {role.upper()} - Top 3 longest:")
                for length, seq, rec_id, scope in chain_list:
                    print(f"    Length: {length}, Scope: {scope}, ID: {rec_id}")
                    print(f"    Seq: {seq[:100]}..." if len(seq) > 100 else f"    Seq: {seq}")
                    print()


if __name__ == "__main__":
    main()
