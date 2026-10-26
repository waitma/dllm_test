"""Prepare unlabeled OTS CDR3-pair replay using the existing TCR input protocol.

Activate pllm/protenix_abtcr; from repository root run:
python -B scripts/research/prepare_tcr_preservation_replay.py --out ABS_NEW_DIRECTORY
CPU only. Use the existing pretraining pilot train/screen OTS rows, reconstruct
observed junctions from annotated FR3/CDR3/FR4, and render label-free runtime
TCR records. No downstream task rows, epitopes, or labels are used for selection.
This prepares a future preservation control; it does not start training.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    parent=ROOT/'output/fixed_chain_exploration_20261014/data'
    manifest=json.loads((parent/'manifest.json').read_text())
    groups,counts={},{}
    for split in ['train','screen']:
        path=parent/f'{split}.jsonl'
        assert sha(path)==manifest['files'][split]['sha256']
        unique={};count=Counter()
        with path.open() as stream:
            for line_index,line in enumerate(stream):
                row=json.loads(line)
                if row.get('source')!='ots_paired':
                    continue
                count['ots_rows']+=1
                assert len(row['chains'])==2 and set(row['chain_roles'])=={'tcr_beta','tcr_alpha'}
                pair={};reason=None
                for i,role in enumerate(row['chain_roles']):
                    regions=row.get('regions',{}).get(str(i),{})
                    order=['FR1','CDR1','FR2','CDR2','FR3','CDR3','FR4']
                    if not all(isinstance(regions.get(k),str) for k in order):
                        reason='missing_annotation';break
                    if ''.join(regions[k] for k in order)!=row['chains'][i]:
                        reason='annotation_not_full_chain';break
                    if not regions['CDR3'] or not regions['FR3'].endswith('C') or not regions['FR4'].startswith(('F','W')):
                        reason='noncanonical_anchor';break
                    if any(row.get('chain_metadata',{}).get(str(i),{}).get('synthetic_residue_mask',[])):
                        reason='synthetic_input';break
                    junction=regions['FR3'][-1]+regions['CDR3']+regions['FR4'][0]
                    assert junction in row['chains'][i]
                    pair[role]=junction
                if reason:
                    count[reason]+=1;continue
                key=(pair['tcr_beta'],pair['tcr_alpha'])
                if key in unique:
                    count['duplicate_pair']+=1;continue
                unique[key]=dict(beta=key[0],alpha=key[1],parent_line_index=line_index,
                    parent_row_sha256=hashlib.sha256(line.encode()).hexdigest(),source='ots_paired')
        groups[split]=unique;counts[split]=dict(count)
    screen_keys=set(groups['screen'])
    train_keys=[k for k in groups['train'] if k not in screen_keys]
    excluded=len(groups['train'])-len(train_keys)
    random.Random(20261018).shuffle(train_keys)
    assert len(train_keys)>=4096 and len(screen_keys)>=100
    selected=dict(train=[groups['train'][k] for k in train_keys[:4096]],
                  screen=list(groups['screen'].values()))

    # Loading on CPU pins the same immutable model source as the measured runs.
    import torch
    torch.set_num_threads(4)
    from downstream.fixed_chain_eval import load,FixedChainBackbone
    checkpoint=ROOT/'output/downstream_generation/fixed_chain_75m_step50000_20261013/checkpoint'
    model,identity=load(checkpoint,device='cpu')
    backbone=FixedChainBackbone(model,identity)
    from downstream.grammar.tcr_features import TCRRepresentationProtocol
    protocol=TCRRepresentationProtocol(input_mode='cdr3ab')
    from dllm.pipelines.immune_llada.data.records import BioSeqRecord
    files={}
    for split,origins in selected.items():
        records=[protocol.record(x['beta'],x['alpha']) for x in origins]
        for start in range(0,len(records),32):
            batch=protocol.collate(records[start:start+32],backbone.collator)
            expected=torch.tensor([len(x['beta'])+len(x['alpha']) for x in origins[start:start+32]])
            observed=batch['residue_mask'] & batch['attention_mask'] & ~batch['synthetic_residue_mask']
            torch.testing.assert_close(observed.sum(1),expected,rtol=0,atol=0)
            assert not batch['relation_target_mask'].any()
        path=out/f'{split}.jsonl'
        with path.open('w') as stream:
            for record in records:
                row=record.to_dict();row['split']='train' if split=='train' else 'valid'
                assert not row.get('labels') and row['chain_roles']==['tcr_beta','tcr_alpha']
                restored=BioSeqRecord.from_prepared_dict(row)
                assert restored.to_dict()==row
                stream.write(json.dumps(row,separators=(',',':'))+'\n')
        write(out/f'{split}_origins.json',origins)
        files[split]=dict(path=str(path),rows=len(records),sha256=sha(path),
                          origins_sha256=sha(out/f'{split}_origins.json'))
    report=dict(status='prepared_only_no_training',created_utc=datetime.now(timezone.utc).isoformat(),
                selection='OTS natural paired chains only; annotated canonical observed junctions; seed20261018 shuffle;4096 train and all eligible screen; no downstream data read',
                files=files,counts=counts,train_pairs_excluded_for_screen_overlap=excluded,
                train_screen_pair_disjoint=True,labels_or_epitopes_present=False,
                schema='existing prepared BioSeqRecord JSONL; runtime TCR completion and observed pooling unchanged',
                parent_model_identity=identity,input_protocol=dict(sha256=protocol.sha256,provenance=protocol.provenance),
                source_sha256={str(p):sha(p) for p in [Path(__file__),parent/'manifest.json',parent/'train.jsonl',parent/'screen.jsonl',ROOT/'downstream/grammar/tcr_features.py']})
    write(out/'manifest.json',report)
    write(out/'verification.json',dict(status='passed',annotation_junctions_reconstructed=True,
          no_downstream_inputs_used=True,observed_residue_counts_equal_junction_lengths=True,
          runtime_protocol_collation_passed=True,prepared_record_roundtrip_passed=True))
    print(json.dumps(dict(files=files,counts=counts,excluded=excluded),indent=2))


if __name__=='__main__':
    main()
