"""Activate pllm; python -B FILE --out ABS to freeze the ablation protocol.

No model execution or hyperparameter selection. Refuse to replace a prior plan.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path('/project_c20250601/251105016/project/dllm_test')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);out=p.parse_args().out
    assert not (out/'plan.json').exists()
    data=json.loads((out/'data_plan.json').read_text());parent=Path(data['parent'])
    audit=json.loads((out/'data_independent_audit.json').read_text());assert audit['status']=='passed'
    assert audit['rows_sha256']==sha(out/'rows.json') and audit['data_plan_sha256']==sha(out/'data_plan.json')
    parent_plan=json.loads((parent/'plan.json').read_text())
    source=out/'source';source.mkdir(exist_ok=False);sources={}
    for relative in ('downstream/cdr_block_ablation.py','downstream/run_cdr_block_ablation.py',
                     'downstream/audit_cdr_block_ablation.py','downstream/freeze_cdr_block_ablation.py',
                     'downstream/prepare_cdr_block_ablation.py','tests/test_cdr_block_ablation.py'):
        original=ROOT/'scripts'/relative;destination=source/original.name
        shutil.copy2(original,destination);sources[str(destination)]=sha(destination)
    families=[]
    common=dict(groups=3,rule='interleaved',temperature=.5,strength=.5)
    for kind in ('soft','hard'):
        for feedback in ('frozen','sequential'):
            families.append(dict(name=kind+'_'+feedback,context=kind,feedback=feedback,**common,prior_weights=[.5,0.0]))
    families.append(dict(name='soft_frozen_strength1',context='soft',feedback='frozen',**{**common,'strength':1.},prior_weights=[.5]))
    families.append(dict(name='soft_frozen_T1',context='soft',feedback='frozen',**{**common,'temperature':1.},prior_weights=[.5]))
    full='soft_frozen_prior0.5'
    primary=[dict(name=name,positive=full,negative=negative) for name,negative in (
        ('soft_distribution','hard_frozen_prior0.5'),('independent_feedback','soft_sequential_prior0.5'),('single_pass_anchor','soft_frozen_prior0.0'))]
    secondary=[dict(name=name,positive=full,negative=negative) for name,negative in (
        ('mask_dilution','soft_frozen_strength1_prior0.5'),('temperature_sharpening','soft_frozen_T1_prior0.5'))]
    external={**parent_plan['source_sha256'],**parent_plan['external_source_sha256']}
    external[str(parent/'plan.json')]=sha(parent/'plan.json')
    plan=dict(schema='cdr.block.ablation.protocol.v1',created_utc=datetime.now(timezone.utc).isoformat(),parent=str(parent),
        marginal_parent=parent_plan['parent'],checkpoints=data['checkpoints'],batch_size=24,precision='fp32',families=families,
        full_policy=full,primary_contrasts=primary,secondary_contrasts=secondary,primary_mode='cdrh3',
        controls=['single_pass','original4','original8'],source_sha256=sources,external_source_sha256=external,
        rows_sha256=sha(out/'rows.json'),data_plan_sha256=sha(out/'data_plan.json'),data_audit_sha256=sha(out/'data_independent_audit.json'),
        counts_by_mode=audit['counts_by_mode'],bootstrap=dict(repeats=50000,seed=20261016,unit='connected_component',family_size=6),
        inferential_rule='Primary full-minus-ablated H3 contrasts: soft distribution, independent feedback, single-pass anchor, on both checkpoints (6 tests). '
            'Report paired component 95% and Bonferroni 99.1666667% percentile intervals. Adjusted lower bound >0 supports benefit at this fixed setting; '
            'upper bound <0 supports harm. A cross-checkpoint component claim requires both adjusted intervals with the same sign. Crossing zero is inconclusive, not equivalence.',
        secondary_rule='H1/H2, MASK dilution, temperature, full-vs-controls and factorial main/interactions are descriptive 95% intervals; no winner selection.',
        sequential_definition='Query groups 0,1,2 in order. Bank begins at p0. Only just-scored positions receive raw query probabilities; unscored positions keep p0. '
            'Every scored group remains MASK. Output anchor is applied after all queries and never enters the bank.',
        hard_definition='Onehot argmax of the same feedback bank, retaining the same strength=.5 MASK blend; isolates distributional feedback from dilution.',
        cost='Every candidate uses 1 prior + 3 group queries. Shared ablation execution uses 31 sample-forwards per row including original4/8. Gates separate.',
        sample_use='No previous confirm or development rows are evaluated. New test allocation is frozen before scoring; no best policy selection or tuning on these results.',
        limitation='OAS supplementary source; inherited exact/cluster training exclusions, not universal pretraining decontamination. Sequential contrast is fixed-order one-sweep feedback, not a universal claim about all iterative sampling.')
    (out/'plan.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(status='frozen',plan_sha256=sha(out/'plan.json'),policies=13,rows=audit['rows'])))


if __name__=='__main__':main()
