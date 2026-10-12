"""After activating pllm, run python -B /absolute/path/audit_v5_canvas.py.

Read-only sampled checks of the configured final corpus; save only a JSON report.
This is not a full-corpus audit or a training launch.
"""
from pathlib import Path
import hashlib
import itertools
import json
import torch
from dllm.pipelines.immune_llada.v3.layout import ImmuneCollator, RECEPTORS, ordered_chains
from dllm.pipelines.immune_llada.v3.model import ImmuneConfig, ImmuneModel
from dllm.pipelines.immune_llada.data.records import BioSeqRecord


def main():
    root=Path(__file__).resolve().parents[2]
    version=root/'configs/models/v7_v5_layout_chain_global_20261007'
    settings=json.loads((version/'research_config.json').read_text())
    data=Path(json.loads((version/'version.json').read_text())['data_dir'])
    config=ImmuneConfig(**settings['base'])
    c=ImmuneCollator.from_config(config)
    g=ImmuneCollator.from_config(config, mode='generation')
    report=dict(scope='first 100 rows of first shard per source/split; sample, not full corpus',
                sources={},checked=0,synthetic_positions=0,max_object_tokens=0,max_total_tokens=0,
                capacity=config.full_capacity,failures=[])
    for split in ('train','valid'):
        seen=set()
        for path in sorted((data/split).glob('*.jsonl')):
            source=path.name.rsplit('-',1)[0]
            if source in seen:
                continue
            seen.add(source)
            n=0
            with path.open() as handle:
                for number,line in enumerate(itertools.islice(handle,100)):
                    try:
                        row=json.loads(line)
                        b=c([row]); inference=g([row])
                        features=ImmuneModel.feature_args(b)
                        # Eligible residues and delimiters masked at t=1; synthetic
                        # and tail slots must already be masks without supervision.
                        features['input_ids']=features['input_ids'].masked_fill(b['eligible_mask'],c.tokenizer.mask_token_id)
                        # A separately supervised context relation is not a target-length feature.
                        for key in features:
                            if key=='input_ids':
                                positions=b['target_slot_mask']
                                assert torch.equal(features[key][positions],inference[key][positions]),key
                            else:
                                assert torch.equal(features[key],inference[key]),key
                        assert int(b['target_slot_mask'].sum())==config.full_capacity
                        assert b['attention_mask'][b['target_slot_mask']].all()
                        assert b['terminal_mask'].sum()==1
                        slots=b['target_slot_mask'][0].nonzero().flatten()
                        cursor=0;observed=0
                        receptors=[chain for chain in ordered_chains(BioSeqRecord.from_prepared_dict(row)) if chain.role in RECEPTORS]
                        for chain in receptors:
                            flags=torch.tensor(chain.metadata.get('synthetic_residue_mask',[0]*len(chain.sequence))).bool()
                            positions=slots[cursor:cursor+len(chain.sequence)]
                            assert b['input_ids'][0,positions[flags]].eq(c.tokenizer.mask_token_id).all()
                            assert b['labels'][0,positions[flags]].eq(-100).all()
                            assert b['eligible_mask'][0,positions[~flags]].all()
                            report['synthetic_positions']+=int(flags.sum())
                            observed+=int((~flags).sum())
                            cursor+=len(chain.sequence)+1
                        assert int(b['eligible_mask'][b['target_slot_mask']].sum())==observed+len(receptors)
                        report['max_object_tokens']=max(report['max_object_tokens'],cursor)
                        report['max_total_tokens']=max(report['max_total_tokens'],int(b['attention_mask'].sum()))
                        report['checked']+=1;n+=1
                    except Exception as error:
                        report['failures'].append(dict(path=str(path),row=number,error=repr(error)))
            report['sources'][split+'/'+source]=n
    files=['v3/layout.py','v3/v5_canvas.py','v3/v5_generation.py','v3/model.py',
           'research_refined/model.py','research/model.py','research_next/model.py']
    paths=[root/'dllm/pipelines/immune_llada'/p for p in files]
    paths += [Path(__file__),root/'downstream/v5_layout.py',root/'scripts/research/run_v7_formal.py',version/'research_config.json']
    report['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report.update(status='passed' if not report['failures'] else 'failed', training_launched=False,retained_test_checkpoints=False)
    dest=root/'output/v7_v5_layout_20261007/canvas_validation.json'
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))
    if report['failures']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
