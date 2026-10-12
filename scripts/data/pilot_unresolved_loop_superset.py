"""Read-only CPU pilot of a conservative observed-string loop screen.

Activate pllm; run: python scripts/data/pilot_unresolved_loop_superset.py
Uses existing read-only inventory.sqlite and isolated edlib; samples unique H3
keys by deterministic SHA256 modulus, never exports or removes training records.
The screening edit bound is a proved SUPerset, not a new identity threshold.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/immune_data_readiness_20260930/unresolved_coverage"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT / "isolated_packages"))
from scripts.data.audit_v7_observation_roles import read, sha_file, sha_text, write


def edit_superset_bound(query_length):
    """80% identity and >=80% query coverage imply this whole-query edit bound.

For m aligned query residues, identity >=4/5 gives E <= m/4. Restoring
the n-m unaligned terminal query residues costs <=n-m deletions. Thus total
distance <=n-3m/4 <=n-3*ceil(4n/5)/4. Integer floor yields the safe bound.
No benchmark target-coverage requirement is imposed on its unsegmented input.
"""
    if query_length < 1:
        raise ValueError("An empty loop is not an auditable sequence")
    minimum_covered = (4 * query_length + 4) // 5
    return (4 * query_length - 3 * minimum_covered) // 4


def scan_loop(query, observations, align):
    # B means D/N and Z means E/Q. X is any canonical residue. These conservative
    # matching equalities preserve original input characters; they are not edits.
    equalities = [("B", "D"), ("B", "N"), ("Z", "E"), ("Z", "Q")]
    equalities += [("X", aa) for aa in "ACDEFGHIKLMNPQRSTVWY"]
    k = edit_superset_bound(len(query))
    for digest, sequence in observations:
        result = align(query, sequence, mode="HW", task="distance", k=k,
                       additionalEqualities=equalities)
        if result["editDistance"] >= 0:
            return digest, result["editDistance"], k
    return None


def main():
    import edlib
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modulus", type=int, default=467)
    p.add_argument("--all-keys", action="store_true")
    args = p.parse_args()
    if args.modulus < 1:
        raise ValueError("modulus must be positive")
    roles = ROOT / "output/immune_data_readiness_20260930/role_resolution/report.json"
    report = read(roles)
    coverage = ROOT / "data/dedup/immune_v7_20260930/benchmark_h3_coverage.json"
    seqs = {sha_text(s): s for s in read(coverage)["unresolved"]}
    # All 131 non-light observations: also protect the 48 disputed empty-source
    # cases, the known fragment, and two partial-native references.
    cases = [c for c in report["cases"] if not c["role"].startswith("light_")]
    observations = [(c["sequence_sha256"], seqs[c["sequence_sha256"]]) for c in cases]
    database = ROOT / "data/dedup/immune_v7_20260930/inventory.sqlite"
    started = time.monotonic()
    db_sha = sha_file(database)
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    samples, total = [], 0
    for (sequence,) in connection.execute("SELECT sequence FROM sequences WHERE role='h3'"):
        total += 1
        digest = hashlib.sha256(sequence.encode()).digest()
        if args.all_keys or int.from_bytes(digest[:8], "big") % args.modulus == 0:
            samples.append((digest.hex(), sequence))
    connection.close()
    read_seconds = time.monotonic() - started
    started = time.monotonic()
    hits, length_counts, matched_lengths = [], Counter(), Counter()
    exact_count = 0
    for digest, query in samples:
        length_counts[len(query)] += 1
        exact = any(query in sequence for _, sequence in observations)
        exact_count += exact
        match = scan_loop(query, observations, edlib.align)
        if match:
            target, distance, bound = match
            matched_lengths[len(query)] += 1
            hits.append({"training_h3_sha256": digest, "length": len(query),
                         "first_observation_sha256": target,
                         "superset_edit_distance": distance, "safe_edit_bound": bound,
                         "has_exact_substring_hit": exact})
    scan_seconds = time.monotonic() - started
    record = {"status": "pilot_superset_only_no_data_mutation", "created_at": datetime.now(timezone.utc).isoformat(),
              "input_hashes": {str(roles): sha_file(roles), str(coverage): sha_file(coverage), str(database): db_sha},
              "script_sha256": sha_file(__file__), "edlib_module_sha256": sha_file(edlib.__file__),
              "observations": len(observations), "inventory_h3_keys": total,
              "sampling": "all" if args.all_keys else f"first64bit_SHA256_mod_{args.modulus}_equals_0",
              "sampled_h3_keys": len(samples), "exact_substring_keys": exact_count,
              "superset_candidate_keys": len(hits), "length_counts": dict(length_counts),
              "matched_length_counts": dict(matched_lengths), "read_seconds": read_seconds,
              "scan_seconds": scan_seconds,
              "linear_projected_full_scan_seconds": scan_seconds * total / len(samples) if samples else None,
              "rows": hits, "limits": ["H3-key sample, not source-row removal count or complete leakage audit.",
                       "The superset intentionally includes false positives; scores do not mean 80% full-string identity.",
                       "Only actual observed residues can be protected. Unknown unobserved native loops cannot be certified.",
                       "Training rows without reliable loop keys need separate fail-closed handling.",
                       "Existing full-chain, loop, pair, antigen, and expanded TCR rules remain mandatory."]}
    filename = "superset_full_key_scan.json" if args.all_keys else "superset_pilot.json"
    write(OUT / filename, record)
    print(json.dumps({k: record[k] for k in ("observations", "inventory_h3_keys", "sampled_h3_keys",
               "exact_substring_keys", "superset_candidate_keys", "read_seconds", "scan_seconds",
               "linear_projected_full_scan_seconds")}))


if __name__ == "__main__":
    main()
