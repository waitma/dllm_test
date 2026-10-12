"""Recover benchmark H3 loops by IMGT numbering, never by guessed anchors.

Run in the existing ANARCI environment:
/vepfs-mlp2/c20250601/251105016/conda/envs/flow/bin/python scripts/data/recover_v7_h3.py --work /absolute/v7/audit --threads 8
The frozen banks remain immutable; a hash-bound derived annotation is written.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent/"tcr_native"))
from anarci_fv import segment_batch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work",type=Path,required=True)
    p.add_argument("--threads",type=int,default=8)
    p.add_argument("--batch-size",type=int,default=128)
    args=p.parse_args()
    raw=(args.work/"banks.json").read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    banks=json.loads(raw)
    pattern=re.compile(r"C[A-Z]{2,32}?[FW]G[A-Z]G")
    missing=[s for s in banks["heavy"] if not any(5<=len(m.group(0)[:-3])<=35 for m in pattern.finditer(s))]
    output=args.work/"benchmark_h3_annotations.json"
    report=json.loads(output.read_text()) if output.exists() else dict(banks_sha256=digest,status="running",annotations={},unresolved=[])
    if report["banks_sha256"]!=digest:
        raise ValueError("Frozen banks changed")
    done=set(report["annotations"])|set(report["unresolved"])
    todo=[s for s in missing if s not in done]
    for offset in range(0,len(todo),args.batch_size):
        sequences=todo[offset:offset+args.batch_size]
        items=[(str(i),seq) for i,seq in enumerate(sequences)]
        annotated=segment_batch(items,allowed_chain_types=("H",),ncpu=args.threads)
        for i,sequence in enumerate(sequences):
            result=annotated.get(str(i))
            if result and result.get("chain_type")=="H" and result.get("cdr3"):
                report["annotations"][sequence]=result["cdr3"]
            else:
                report["unresolved"].append(sequence)
        report["status"]="complete" if offset+args.batch_size>=len(todo) else "running"
        temporary=output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report,separators=(",",":"))+"\n")
        os.replace(temporary,output)
        print(json.dumps({"numbered":len(report["annotations"]),"unresolved":len(report["unresolved"]),"total":len(missing)}),flush=True)
    if not todo and not output.exists():
        report["status"]="complete"
        output.write_text(json.dumps(report)+"\n")


if __name__=="__main__":
    main()
