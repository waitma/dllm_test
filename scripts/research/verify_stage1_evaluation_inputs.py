"""Check every actual evaluation input against its residue sequence.

Activate pllm, then run this absolute script with --root ABS_EVALUATION_ROOT
and --arm mlm or pair_kd, in separate processes. No model forward is executed.
All native retrieval candidates and experimental structure inputs are checked
through the original checkpoint-bound collator; source weights stay unchanged.
"""
import argparse
import json
from pathlib import Path
import time

import torch
from evaluate_stage1_partner_retrieval import load,model_and_collator,assemble
from evaluate_stage1_frozen_gdp import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--arm',choices=['mlm','pair_kd'],required=True);args=p.parse_args();root=args.root
    assert root.is_absolute()
    out=root/(args.arm+'_input_verification.json')
    if out.exists():raise FileExistsError(out)
    plan,records,items=load(root/'native_partner')
    model,collator=model_and_collator(plan,args.arm,'cpu',4)
    vocab=[model.tokenizer.token(i) for i in range(model.tokenizer.vocab_size)]
    allowed=set('ACDEFGHIKLMNPQRSTVWY');counts={'native_candidates':0,'structure_inputs':0,'residues_checked':0}
    start=time.monotonic()
    for offset in range(0,len(items),64):
        chunk=items[offset:offset+64];features,target,regions,residue=assemble(records,chunk,collator,model,'cpu')
        assert set(features)=={'input_ids','attention_mask','role_ids','scope_ids','chain_ids','position_ids','target_slot_mask'}
        for j,item in enumerate(chunk):
            ids=features['chain_ids'][j][residue[j]].unique(sorted=True);assert len(ids)==2
            for chain,role in zip(ids,['h','l']):
                mask=residue[j]&(features['chain_ids'][j]==chain)
                decoded=''.join(vocab[i] for i in features['input_ids'][j,mask].tolist())
                assert set(decoded)<=allowed and decoded==records[item[role+'_row']][role]
                counts['residues_checked']+=len(decoded)
            decoded=''.join(vocab[i] for i in features['input_ids'][j,target[j]].tolist())
            assert decoded==records[item['target_row']][item['target']]
            assert (features['position_ids'][j,target[j]]>=1).all()
            masks=[target[j]&(((target[j].long().cumsum(0)-1)%8)==g) for g in range(8)]
            assert torch.equal(torch.stack(masks).sum(0),target[j].long())
        counts['native_candidates']+=len(chunk)
    structure=root/'experimental_structure'
    geometry=json.loads((structure/'geometry_plan.json').read_text())
    assert geometry['arms'][args.arm]['model_sha256']==plan['arms'][args.arm]['model_sha256']
    assert sha(structure/'structures.json')==geometry['files']['structures.json']
    from dllm.pipelines.immune_llada.data.records import BioSeqRecord,BioSeqChain
    records=json.loads((structure/'structures.json').read_text())
    for offset in range(0,len(records),64):
        chunk=records[offset:offset+64]
        seqs=[BioSeqRecord([BioSeqChain(r['h'],'antibody_heavy'),BioSeqChain(r['l'],'antibody_light')],
                          task_type='antibody',source='stage1_experimental_structure_eval') for r in chunk]
        batch=collator(seqs)
        assert not (batch['residue_mask'].bool()&batch['terminal_mask'].bool()).any()
        assert (batch['input_ids'][batch['relation_mask']]==model.tokenizer.mask_token_id).all()
        for j,r in enumerate(chunk):
            selected=batch['residue_mask'][j].bool()&batch['attention_mask'][j].bool()
            decoded=''.join(vocab[i] for i in batch['input_ids'][j,selected].tolist())
            assert decoded==r['h']+r['l'] and set(decoded)<=allowed
            counts['residues_checked']+=len(decoded)
        counts['structure_inputs']+=len(chunk)
    write(out,dict(status='passed',arm=args.arm,**counts,script_sha256=sha(__file__),
        native_plan_sha256=sha(root/'native_partner/plan.json'),geometry_plan_sha256=sha(structure/'geometry_plan.json'),
        model_sha256=plan['arms'][args.arm]['model_sha256'],all_original_sequences_match=True,
        selected_target_sequences_match=True,mask_partition_covers_every_target_once=True,
        no_labels_in_forward_features=True,structure_pooling_excludes_special_tokens=True,
        parameter_updates=False,model_forward_executed=False,seconds=time.monotonic()-start))
    print(json.dumps(dict(status='passed',arm=args.arm,**counts)),flush=True)


if __name__=='__main__':main()
