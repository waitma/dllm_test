"""Create a small, provenance-bound teacher pilot from native OAS training pairs.

In pllm: python -B -m scripts.research.prepare_stage1_pilot --source ABS_JSONL
  --bert ABS_CHECKPOINT --output ABS_JSON --records 2
This is an engineering sample, not a train/validation split or quality approval.
"""
import argparse
import hashlib
import json
from pathlib import Path

from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
from dllm.pipelines.immune_llada.fixed_chain.structure_mlm import file_sha256
from scripts.research.stage1_data import make_teacher_request


def run(args):
    source, bert, output = map(lambda x: Path(x).resolve(), (args.source,args.bert,args.output))
    if output.exists():
        raise FileExistsError(output)
    config = FixedChainConfig(**json.loads((bert / "immune_config.json").read_text()))
    requests, origins = [], []
    with source.open("rb") as stream:
        while len(requests) < args.records:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                raise ValueError("Not enough complete native OAS training pairs")
            row = json.loads(line)
            if (row.get("source") != "oas_paired" or row.get("split") != "train"
                    or row.get("chain_roles") != ["antibody_heavy","antibody_light"]):
                continue
            request = make_teacher_request(row,config)
            if len(request["chains"]) != 2:
                continue
            evidence = f"Native oas_paired H/L in training shard {source}, byte {offset}"
            request = make_teacher_request(row,config,inter_evidence=[dict(chains=[0,1],evidence=evidence)])
            requests.append(request)
            origins.append(dict(offset=offset,line_sha256=hashlib.sha256(line).hexdigest(),
                                request_sha256=request["request_sha256"]))
    result = dict(schema="immune.structure_teacher.pilot.v1", purpose="engineering_smoke_only",
                  source=str(source),source_sha256=file_sha256(source),origins=origins,
                  parent=str(bert),parent_sha256=file_sha256(bert / "model.safetensors"),requests=requests)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(output=str(output),records=len(requests),
                         chain_lengths=[[len(c["sequence"]) for c in r["chains"]] for r in requests])))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True)
    parser.add_argument("--bert",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--records",type=int,default=2)
    args=parser.parse_args()
    if args.records < 1:
        parser.error("records must be positive")
    run(args)
