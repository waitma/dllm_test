"""Build audited Fv boundaries: PYTHONPATH=. <flow-env>/bin/python scripts/downstream/prepare_ab_fv_input.py.

Numbers only six unique reference chains. All m396 variants must preserve the
discarded WT flanks; no mutations may be hidden by the Fv protocol.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
from anarci import anarci

from downstream.grammar.ab_fv_input import MANIFEST, ROOT, adapt_pairs


def main():
    spec_path = ROOT / "data/downstream/specificity/hd-0_flu-1_cov-2.csv"
    wt_path = ROOT / "data/downstream/in_silico/rcsb_pdb_2G75.fasta"
    data_path = ROOT / "data/downstream/in_silico/Desautels_insilico_data.csv"
    spec = pd.read_csv(spec_path)
    items, roles = [], {}
    for row in spec.itertuples():
        for role, cap in (("h", 166), ("l", 134)):
            seq = getattr(row, role + "_sequence")
            if len(seq) > cap:
                key = hashlib.sha256(seq.encode()).hexdigest()
                items.append((key, seq)); roles[key] = role
    lines = wt_path.read_text().splitlines()
    for header, seq in zip(lines[::2], lines[1::2]):
        role = "l" if "Light Chain" in header else "h"
        items.append(("wt_" + role, seq)); roles["wt_" + role] = role
    numbered, details, _ = anarci(items, scheme="imgt", ncpu=2)
    manifest = {"version": 1, "method": "ANARCI_IMGT", "overflow": {}, "m396": {},
                "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (spec_path, wt_path, data_path)}}
    for (key, seq), domains, info in zip(items, numbered, details):
        if not domains or len(domains) != 1:
            raise ValueError(f"Expected exactly one variable domain: {key}")
        numbering, start, end = domains[0]
        fv = "".join(aa for _, aa in numbering if aa != "-")
        role = roles[key]
        if info[0]["chain_type"] not in ({"H"} if role == "h" else {"K", "L"}):
            raise ValueError(f"Wrong chain type: {key}")
        if seq[start:end + 1] != fv or not (100 <= len(fv) <= (166 if role == "h" else 134)):
            raise ValueError(f"Invalid Fv boundary: {key}")
        rule = {"role": role, "sequence": seq, "start": start, "end": end + 1,
                "chain_type": info[0]["chain_type"], "fv_length": len(fv)}
        if key.startswith("wt_"): manifest["m396"][role] = rule
        else: manifest["overflow"][key] = rule
    # Validate before publishing the manifest.
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        candidate = Path(temp) / "manifest.json"
        candidate.write_text(json.dumps(manifest))
        variants = pd.read_csv(data_path).Antibody_Sequence.tolist()
        _, audit = adapt_pairs("m396", [(s[:245], s[245:]) for s in variants], manifest_path=candidate)
        manifest["audit"] = audit
        manifest["audit"].pop("manifest_sha256")
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["audit"], indent=2))


if __name__ == "__main__":
    main()
