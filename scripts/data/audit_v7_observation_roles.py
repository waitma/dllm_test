"""Read-only role/observed-region audit; never edits datasets or issues certificates.

Activate pllm first. Number with the existing ANARCI environment (no download):
/vepfs-mlp2/c20250601/251105016/conda/envs/flow/bin/python \
  scripts/data/audit_v7_observation_roles.py --refresh-numbering
Subsequent runs can use pllm: python scripts/data/audit_v7_observation_roles.py
CPU logic tests: pytest -q scripts/tests/data/test_v7_observation_roles.py
Outputs contain sequence hashes, lengths and positions, not full sequences.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "data/dedup/immune_v7_20260930"
DEFAULT_OUT = ROOT / "output/immune_data_readiness_20260930/role_resolution"
PROVENANCE = ROOT / "output/immune_data_readiness_20260930/h3_source_provenance.json"
REGIONS = {"fr1": (1, 26), "cdr1": (27, 38), "fr2": (39, 55),
           "cdr2": (56, 65), "fr3": (66, 104), "cdr3": (105, 117),
           "fr4": (118, 128)}
AUTHOR_ROLES = {"2": "vnar", "3": "light_lambda", "4": "light_kappa"}


def sha_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def normalize(value):
    # Same whitespace/J normalization as data.records; NBbench bank removes gaps.
    return "".join(str(value or "").split()).upper().replace("J", "L")


def occurrences(haystack, needle):
    if not needle:
        return []
    return [i for i in range(len(haystack) - len(needle) + 1)
            if haystack.startswith(needle, i)]


def summarize_domain(sequence, domain, detail):
    """Validate the returned slice before assigning any raw residue indices."""
    numbering, start, end = domain
    non_gap = [(int(pos), str(ins).strip(), aa)
               for (pos, ins), aa in numbering if aa != "-"]
    numbered_sequence = "".join(aa for _, _, aa in non_gap)
    # Installed schemes.py sets end_index = si, despite an exclusive-end docstring.
    # Resolve this from exact equality, never from an assumed convention.
    modes = [("inclusive", end + 1), ("exclusive", end)]
    matches = [(mode, stop) for mode, stop in modes
               if numbered_sequence and sequence[start:stop] == numbered_sequence]
    valid = len(matches) == 1
    stop = matches[0][1] if valid else None
    positions = [{"imgt": p, "insertion": ins,
                  "raw_index": start + i if valid else None}
                 for i, (p, ins, _) in enumerate(non_gap)]
    regions = {}
    for name, (lo, hi) in REGIONS.items():
        selected = [(i, aa) for i, (pos, _, aa) in enumerate(non_gap) if lo <= pos <= hi]
        regions[name] = {"length": len(selected),
                         "sha256": sha_text("".join(aa for _, aa in selected)) if selected else None,
                         "raw_indices": [start + i for i, _ in selected] if valid else []}
    return {"chain_type": detail.get("chain_type"), "reported_start": start,
            "reported_end": end, "end_convention_verified": matches[0][0] if valid else None,
            "exact_input_mapping": valid, "raw_start": start if valid else None,
            "raw_stop_exclusive": stop, "unmapped_prefix_length": start if valid else None,
            "unmapped_suffix_length": len(sequence) - stop if valid else None,
            "non_gap_count": len(non_gap), "imgt_min": min((p for p, _, _ in non_gap), default=None),
            "imgt_max": max((p for p, _, _ in non_gap), default=None),
            "positions": positions, "regions": regions,
            "hmm_bitscore": detail.get("bitscore"), "hmm_evalue": detail.get("evalue")}


def number_cases(sequences, out, coverage_sha, ncpu, filename="numbering.json"):
    import anarci as package
    from anarci import anarci
    started = time.monotonic()
    items = sorted(sequences.items())
    numbered, details, _ = anarci(items, scheme="imgt", output=False, ncpu=ncpu)
    rows = []
    for (digest, sequence), domains, info in zip(items, numbered, details):
        rows.append({"sequence_sha256": digest, "length": len(sequence),
                     "domains": [summarize_domain(sequence, domain, detail)
                                 for domain, detail in zip(domains or [], info or [])]})
    package_dir = Path(package.__file__).resolve().parent
    dependencies = {str(p): sha_file(p) for p in sorted(package_dir.rglob("*"))
                    if p.is_file() and p.suffix in {".py", ".hmm", ".h3f", ".h3i", ".h3m", ".h3p"}}
    result = {"schema": "v7_observation_numbering_v1", "coverage_sha256": coverage_sha,
              "created_at": datetime.now(timezone.utc).isoformat(), "python": sys.executable,
              "scheme": "imgt", "ncpu": ncpu, "default_threshold_unchanged": True,
              "package_inputs": dependencies, "elapsed_seconds": time.monotonic() - started,
              "cases": rows}
    write(out / filename, result)
    print(json.dumps({"numbered": len(rows), "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    return result


def load_sources(provenance, expected_inputs):
    wanted = defaultdict(set)
    for case in provenance["cases"]:
        for occurrence in case["occurrences"]:
            wanted[occurrence["path"]].add(occurrence["row_zero_based"])
    sources, hashes = {}, {}
    for filename, indices in sorted(wanted.items()):
        digest = sha_file(filename)
        if expected_inputs.get(filename) != digest:
            raise ValueError("Source differs from frozen registry: " + filename)
        hashes[filename] = digest
        with Path(filename).open(newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                if index in indices:
                    sources[(filename, index)] = row
    return sources, hashes


def source_cdr_evidence(row, native):
    """Report source labels and exact locations, with no claim that labels are correct."""
    result = {}
    for cdr in ("CDR1", "CDR2", "CDR3"):
        field = cdr if cdr in row else cdr + "_nogaps"
        if field not in row:
            continue
        core = normalize(row[field]).replace("-", "")
        result[cdr.lower()] = {"field": field, "length": len(core),
                               "sha256": sha_text(core) if core else None,
                               "native_offsets": occurrences(native, core)}
    return result


def gap_evidence(masked, native, row):
    """Validate the actual source gap intervals against its paired native reference."""
    masked, native = normalize(masked), normalize(native)
    consistent = len(masked) == len(native) and all(a == "-" or a == b
                                                       for a, b in zip(masked, native))
    runs = [(m.start(), m.end()) for m in re.finditer("-+", masked)]
    result = {"same_length": len(masked) == len(native),
              "visible_residues_match_native": consistent, "gap_intervals": runs,
              "native_sha256": sha_text(native), "native_length": len(native)}
    labels = {}
    for name in ("CDR1", "CDR2", "CDR3"):
        value = normalize(row.get(name, ""))
        matches = [(start, stop) for start, stop in runs if native[start:stop] == value]
        labels[name.lower()] = {"length": len(value), "matching_gap_intervals": matches}
    result["source_loop_gap_matches"] = labels
    result["all_three_loops_have_distinct_unique_gap_matches"] = consistent and all(
        len(v["matching_gap_intervals"]) == 1 for v in labels.values()) and len(
            {tuple(v["matching_gap_intervals"][0]) for v in labels.values()
             if len(v["matching_gap_intervals"]) == 1}) == 3
    return result


def role_from_labels(labels, domains):
    roles = {AUTHOR_ROLES[v] for v in labels if v in AUTHOR_ROLES}
    if len(roles) != 1:
        return "pending_role_conflict" if roles else "heavy_source_label"
    role = next(iter(roles))
    expected = {"light_lambda": "L", "light_kappa": "K"}.get(role)
    if expected and any(d["chain_type"] != expected for d in domains):
        return "pending_role_numbering_conflict"
    return role


def observed_scope(domains, source_empty_cdr3=False):
    """Only adjudicate covered residues. Never infer an unsequenced native H3."""
    if len(domains) != 1:
        return "pending_no_numbered_domain" if not domains else "pending_multiple_domains"
    d = domains[0]
    if not d["exact_input_mapping"]:
        return "pending_numbering_input_mapping"
    if d["regions"]["cdr3"]["length"]:
        return "observed_numbered_cdr3"
    if d["unmapped_suffix_length"] == 0 and d["imgt_max"] < 105:
        return "observed_fragment_ends_before_h3"
    if d["unmapped_prefix_length"] == 0 and d["imgt_min"] > 117:
        return "observed_fragment_starts_after_h3"
    before = [p["raw_index"] for p in d["positions"] if p["imgt"] <= 104]
    after = [p["raw_index"] for p in d["positions"] if p["imgt"] >= 118]
    if (source_empty_cdr3 and before and after and max(before) + 1 == min(after)
            and d["unmapped_prefix_length"] == d["unmapped_suffix_length"] == 0):
        return "observed_empty_h3_source_and_numbering_agree"
    return "pending_h3_numbering_or_unmapped_residues"


def parse_embl(text):
    """Read an existing official EMBL snapshot, without deriving/translating sequences."""
    feature_text = "\n".join(line[5:] for line in text.splitlines() if line.startswith("FT   "))
    translations = re.findall(r'/translation="([^"]+)"', feature_text, flags=re.DOTALL)
    if len(translations) != 1:
        raise ValueError("Expected one explicitly supplied CDS translation")
    sequence = "".join(translations[0].split())
    description = " ".join(line[5:] for line in text.splitlines() if line.startswith("DE   "))
    feature_types = re.findall(r"^FT   ([A-Za-z_]+)\s", text, flags=re.MULTILINE)
    return sequence, {"description": description, "feature_types": feature_types,
                      "translation_sha256": sha_text(sequence), "translation_length": len(sequence),
                      "germline_source_explicit": '/germline' in feature_text}


def short_reference_evidence(native, masked, row, reference, ref_domains):
    """Align only exact supplied reference strings; never invent or extend residues."""
    native, reference = normalize(native), normalize(reference)
    result = {"native_sha256": sha_text(native), "reference_sha256": sha_text(reference),
              "native_is_exact_reference_prefix": reference.startswith(native),
              "reference_is_exact_native_prefix": native.startswith(reference),
              "native_length": len(native), "reference_length": len(reference)}
    gap = gap_evidence(masked, native, row)
    result["source_cdr3_mask_intervals"] = gap["source_loop_gap_matches"]["cdr3"]["matching_gap_intervals"]
    if reference.startswith(native) and len(ref_domains) == 1 and ref_domains[0]["exact_input_mapping"]:
        d = ref_domains[0]
        loop_indices = d["regions"]["cdr3"]["raw_indices"]
        covered = [i for i in loop_indices if i < len(native)]
        result.update(reference_cdr3_length=len(loop_indices), observed_native_cdr3_indices=covered,
                      observed_native_cdr3_length=len(covered),
                      native_covers_complete_reference_cdr3=bool(loop_indices) and len(covered) == len(loop_indices),
                      source_cdr3_matches_observed_reference_residues=(
                          "".join(native[i] for i in covered) == normalize(row["CDR3"])))
    if native.startswith(reference) and len(native) > len(reference):
        result.update(native_extension_length=len(native) - len(reference),
                      native_extension_sha256=sha_text(native[len(reference):]),
                      extension_matches_source_cdr3=native[len(reference):] == normalize(row["CDR3"]))
    return result


def adjudicate_short_reference(evidence):
    if (evidence.get("native_is_exact_reference_prefix")
            and evidence.get("observed_native_cdr3_length", 0) > 0
            and evidence.get("source_cdr3_matches_observed_reference_residues")
            and not evidence.get("native_covers_complete_reference_cdr3")):
        return "observed_partial_h3_exact_native_reference"
    if evidence.get("native_extension_length"):
        return "pending_source_construct_exceeds_native_reference"
    return "pending_source_short_cdr3_region_unverified"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--refresh-numbering", action="store_true")
    parser.add_argument("--ncpu", type=int, default=4)
    args = parser.parse_args()
    coverage_path = WORK / "benchmark_h3_coverage.json"
    coverage, provenance = read(coverage_path), read(PROVENANCE)
    coverage_sha = sha_file(coverage_path)
    if provenance["coverage_sha256"] != coverage_sha:
        raise ValueError("Source provenance coverage mismatch")
    registry = read(WORK / "registry.json")
    if sha_file(WORK / "registry.json") != coverage["registry_sha256"]:
        raise ValueError("Frozen registry changed")
    sequences = {sha_text(s): s for s in coverage["unresolved"]}
    if set(sequences) != {c["sequence_sha256"] for c in provenance["cases"]}:
        raise ValueError("Provenance does not cover every unresolved sequence exactly")
    sources, source_hashes = load_sources(provenance, registry["inputs"])
    numbering = (number_cases(sequences, args.output, coverage_sha, args.ncpu)
                 if args.refresh_numbering else read(args.output / "numbering.json"))
    if numbering["coverage_sha256"] != coverage_sha:
        raise ValueError("Numbering coverage mismatch")
    numbered = {r["sequence_sha256"]: r for r in numbering["cases"]}
    if set(numbered) != set(sequences):
        raise ValueError("Numbering cases mismatch")
    reference_dir = args.output / "source_references"
    reference_manifest = read(reference_dir / "fetch_manifest.json")
    reference_inputs, reference_sequences, reference_metadata = {}, {}, {}
    for entry in reference_manifest:
        if "path" not in entry:
            continue
        if sha_file(entry["path"]) != entry["sha256"]:
            raise ValueError("Downloaded reference changed: " + entry["path"])
        reference_inputs[entry["path"]] = entry["sha256"]
        path = Path(entry["path"])
        if path.suffix == ".embl":
            seq, metadata = parse_embl(path.read_text())
            reference_sequences[path.stem], reference_metadata[path.stem] = seq, metadata
    reference_binding = sha_text(json.dumps(reference_inputs, sort_keys=True))
    reference_numbering = (number_cases({sha_text(s): s for s in reference_sequences.values()},
                                       args.output, reference_binding, args.ncpu, "reference_numbering.json")
                           if args.refresh_numbering else read(args.output / "reference_numbering.json"))
    if reference_numbering["coverage_sha256"] != reference_binding:
        raise ValueError("Reference numbering input binding changed")
    reference_domains = {r["sequence_sha256"]: r["domains"] for r in reference_numbering["cases"]}
    cases = []
    for case in provenance["cases"]:
        digest = case["sequence_sha256"]
        sequence, domains = sequences[digest], numbered[digest]["domains"]
        if numbered[digest]["length"] != len(sequence):
            raise ValueError("Numbering length mismatch")
        evidence, labels, source_empty = [], [], False
        for occurrence in case["occurrences"]:
            filename, index, field = (occurrence[k] for k in ("path", "row_zero_based", "field"))
            row = sources[(filename, index)]
            if normalize(row[field]).replace("-", "") != sequence:
                raise ValueError("Source row/field no longer reproduces case hash")
            task = Path(filename).parent.name
            item = {"path": filename, "row_zero_based": index, "field": field, "task": task}
            if task == "nanobody_type":
                labels.append(row["label"])
                item["author_type_label"] = row["label"]
            native = normalize(row["label"] if task == "CDRInfilling" else row[field]).replace("-", "")
            item["source_cdr_annotations"] = source_cdr_evidence(row, native)
            if task == "polyreaction" and "CDR3_nogaps" in row:
                source_empty |= not normalize(row["CDR3_nogaps"])
            if task == "CDRInfilling":
                item["gap_native_evidence"] = gap_evidence(row["seq"], row["label"], row)
                accession = row.get("Comma-separated db-specific IDS")
                if accession in reference_sequences:
                    ref_seq = reference_sequences[accession]
                    item["original_reference"] = reference_metadata[accession]
                    item["short_reference_evidence"] = short_reference_evidence(
                        row["label"], row["seq"], row, ref_seq, reference_domains[sha_text(ref_seq)])
            if task in {"CDRInfilling", "VRClassification"}:
                item["source_reference_id"] = row.get("Comma-separated db-specific IDS")
            for key in ("subject_species", "subject_name"):
                if key in row:
                    item[key] = row[key]
            evidence.append(item)
        role = role_from_labels(labels, domains)
        scope = observed_scope(domains, source_empty)
        if role in {"light_lambda", "light_kappa"}:
            resolution = "author_light_role_numbered_l3"
        elif role == "vnar":
            resolution = "author_vnar_role_loop_annotation_pending"
        elif any(e["task"] in {"CDRInfilling", "VRClassification"} for e in evidence):
            short = [e["short_reference_evidence"] for e in evidence if "short_reference_evidence" in e]
            resolution = (adjudicate_short_reference(short[0]) if len(short) == 1
                          else "pending_source_short_cdr3_region_unverified")
        else:
            resolution = scope
        known_loops = []
        for domain_index, d in enumerate(domains):
            if not d["exact_input_mapping"]:
                continue
            for name in ("cdr1", "cdr2", "cdr3"):
                region = d["regions"][name]
                if region["length"]:
                    known_loops.append({"kind": name, "chain_type": d["chain_type"],
                                        "domain_index": domain_index, **region,
                                        "interpretation": "numbered_observed_residues"})
        is_pending = resolution.startswith("pending_") or role == "vnar"
        cases.append({"sequence_sha256": digest, "length": len(sequence), "role": role,
                      "resolution": resolution, "observed_scope": scope,
                      "observation_adjudication": "pending" if is_pending else "supported",
                      "heavy_h3_applicable": role not in {"light_lambda", "light_kappa", "vnar"},
                      "known_observed_loops": known_loops,
                      "prior_classification": case["prior_diagnostic"]["classification"],
                      "domain_count": len(domains), "source_empty_cdr3": source_empty,
                      "protection_required": ["entire_observed_input", "role_compatible_near_neighbors",
                                               "every_verified_observed_loop"],
                      "sources": evidence})
    source_readme = ROOT / "data/nanobody_raw/nbbench/hf_data/nanobody_type/README.md"
    report = {"schema": "v7_observation_roles_v1", "status": "evidence_only_not_certificate",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "input_hashes": {str(coverage_path): coverage_sha, str(PROVENANCE): sha_file(PROVENANCE),
                               str(WORK / "registry.json"): sha_file(WORK / "registry.json"),
                               str(source_readme): sha_file(source_readme), **source_hashes,
                               str(reference_dir / "fetch_manifest.json"): sha_file(reference_dir / "fetch_manifest.json"),
                               **reference_inputs},
              "numbering_sha256": sha_file(args.output / "numbering.json"),
              "reference_numbering_sha256": sha_file(args.output / "reference_numbering.json"),
              "script_sha256": sha_file(__file__), "case_count": len(cases),
              "role_counts": dict(Counter(c["role"] for c in cases)),
              "resolution_counts": dict(Counter(c["resolution"] for c in cases)),
              "observed_scope_counts": dict(Counter(c["observed_scope"] for c in cases)),
              "observation_adjudication_counts": dict(Counter(c["observation_adjudication"] for c in cases)),
              "cases": cases,
              "limits": ["Observed-input scope only; native biological completeness is not inferred.",
                         "Role N/A never exempts full-input or known-loop protection.",
                         "No prepared data, benchmark, gate, or certificate was edited."]}
    write(args.output / "report.json", report)
    write(args.output / "protection_rules_proposal.json", {
        "schema": "role_observed_protection_proposal_v1", "status": "proposed_not_applied",
        "report_sha256": sha_file(args.output / "report.json"),
        "scope": "actual_benchmark_input_and_target_observations",
        "existing_rules_preserved": True,
        "rules": [
            {"id": "observed_input_exact", "applies_to": "all_264",
             "test": "Normalized exact whole-input keys against all antibody heavy/light/vhh source keys; preserve pair checks."},
            {"id": "whole_chain_near", "applies_to": "all_role_compatible_full_chains",
             "minimum_identity": 0.95, "minimum_query_coverage": 0.8,
             "minimum_target_coverage": 0.8,
             "test": "Extend existing heavy rule to light and VNAR; also compare legacy mis-typed heavy/light pools. Do not drop old heavy-bank entries until the union audit passes."},
            {"id": "observed_fragment_additional_near", "applies_to": "proven_partial_or_unresolved_completeness",
             "minimum_identity": 0.95, "minimum_query_coverage": 0.8,
             "minimum_target_coverage": None,
             "test": "Additional local alignment screen of the complete observed fragment against compatible receptor chains. Removing target-coverage requirement adds protection; it does not replace existing filters. Record query/target coordinates and both coverages."},
            {"id": "verified_complete_loop", "applies_to": "verified_observed_complete_L3_H3_or_VNAR_loop",
             "minimum_identity": 0.8, "minimum_query_coverage": 0.8,
             "minimum_target_coverage": 0.8,
             "test": "Extend the existing H3 loop screen to role-correct loops. Store annotation source, chain, indices and SHA; never manufacture a loop for unknown numbering."},
            {"id": "partial_two_residue_annotation", "applies_to": "AJ237302_two_cases",
             "test": "Protect entire observed native/input fragments; retain the 2-residue partial-loop metadata as incomplete. Do not equate it to a complete H3 or globally blacklist an uncontextualized 2-mer. Complete underlying-reference H3 can only be an explicitly added, source-bound protection scope."},
            {"id": "missing_not_empty", "applies_to": "80_pending_cases",
             "test": "Keep annotation/completeness pending until applicable source evidence or independent numbering resolves it; no certificate claim of complete-loop novelty. Mandatory full-observed-input protection remains active."},
            {"id": "source_confirmed_empty_observation", "applies_to": "48_polyreaction_plus_1_pre_H3_fragment",
             "test": "Empty/unobserved H3 is not an empty sequence bank key and not a fabricated loop. Preserve entire observed inputs and all nonempty, verified observed loops; do not infer native biological absence."}
        ],
        "certificate_preconditions": ["Implement role-aware union protection in a new version, never edit current prepared data.",
                                       "Independently rescan final manifest under unchanged old rules plus expanded role/fragment rules.",
                                       "Resolve or explicitly scope remaining unsupported claims; this proposal itself clears no gate."]})
    print(json.dumps({k: report[k] for k in ("case_count", "role_counts", "resolution_counts", "observed_scope_counts")}))


if __name__ == "__main__":
    main()
