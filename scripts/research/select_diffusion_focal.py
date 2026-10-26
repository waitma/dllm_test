"""Apply the predeclared EOS focal pilot rule using internal validation only.

Activate pllm; python -B -m scripts.research.select_diffusion_focal --run ABS.
Verifies matched parent/code/data/schedule/corruption streams before selection.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import torch

from scripts.research.bert_mlm_data import sha, write_json


def main(run):
    packages=json.loads((run/"packages.json").read_text())
    folders=[Path(packages[k]["output"]) for k in ("pilot_ce","pilot_focal")]
    provenance=[json.loads((folder/"run_provenance.json").read_text()) for folder in folders]
    finished=[json.loads((folder/"COMPLETE.json").read_text()) for folder in folders]
    if any(item["status"]!="success" or item["step"]!=200 or item["seen_records"]!=102400 for item in finished):
        raise ValueError("Both pilots must complete their matched 200-step budgets")
    for key in ("model","training","data_index","data_index_sha256","parent_checkpoint","parent_model_sha256","world_size"):
        if provenance[0][key]!=provenance[1][key]:
            raise ValueError("Pilot provenance differs: "+key)
    recipe=[dict(p["recipe"]) for p in provenance]
    if [item.pop("eos_gamma") for item in recipe]!=[0.,1.] or recipe[0]!=recipe[1]:
        raise ValueError("Pilots differ beyond EOS focal gamma")
    code=[{str(Path(p).relative_to(prov["source_root"])):h for p,h in prov["sources"].items()}
          for prov in provenance]
    if code[0]!=code[1]:
        raise ValueError("Paired pilots used different source code")
    for folder in folders:
        checkpoint=folder/"checkpoint-200"
        receipt=json.loads((checkpoint/"checkpoint_receipt.json").read_text())
        if receipt["status"]!="complete" or len(set(receipt["rank_parameter_sha256"]))!=1:
            raise ValueError("Incomplete/divergent pilot checkpoint")
        for name,expected in receipt["files"].items():
            if sha(checkpoint/name)!=expected:
                raise ValueError("Pilot checkpoint artifact changed")
    for rank in range(8):
        rng=[torch.load(folder/f"checkpoint-200/rng_rank{rank}.pt",weights_only=False,map_location="cpu")
             for folder in folders]
        if not torch.equal(rng[0]["corruption"],rng[1]["corruption"]):
            raise ValueError("Pilot corruption generator streams differ")
    initial=[json.loads((folder/"validation_step0.json").read_text())["summary"] for folder in folders]
    for key,value in initial[0].items():
        if value is not None and not math.isclose(value,initial[1][key],rel_tol=1e-6,abs_tol=1e-8):
            raise ValueError("Pilots did not start at matching validation: "+key)
    validation=[json.loads((folder/"validation_step200.json").read_text()) for folder in folders]
    for mode in validation[0]["modes"]:
        for source in validation[0]["modes"][mode]["sources"]:
            counts=[item["modes"][mode]["sources"][source]["counts"] for item in validation]
            for key in ("tokens","eos_tokens","relation_tokens","aa_tokens","keep_tokens","records","chains"):
                if counts[0][key]!=counts[1][key]:
                    raise ValueError("Validation populations differ")
    ce,focal=[item["summary"] for item in validation]
    decisions=dict(eos_ce_improves=focal["eos_loss"]<ce["eos_loss"],
        first_eos_not_worse=focal["first_eos_accuracy"]>=ce["first_eos_accuracy"],
        early_eos_not_worse=focal["early_eos_rate"]<=ce["early_eos_rate"],
        missing_eos_not_worse=focal["missing_eos_rate"]<=ce["missing_eos_rate"],
        source_macro_ce_within_2_percent=focal["loss"]<=ce["loss"]*1.02)
    result=dict(status="complete",created_utc=datetime.now(timezone.utc).isoformat(),
        parent_model_sha256=provenance[0]["parent_model_sha256"],steps_per_pilot=200,
        selected_eos_gamma=1. if all(decisions.values()) else 0.,rule_checks=decisions,
        matched_parent_data_code_schedule=True,all_rank_corruption_streams_equal=True,
        ce=ce,focal=focal,scope="short internal development screen; no downstream selection or established long-term gain",
        evidence={str(folder/"validation_step200.json"):sha(folder/"validation_step200.json") for folder in folders},
        selector_sha256=sha(__file__))
    write_json(run/"focal_selection.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    main(parser.parse_args().run)
