"""Activate pllm; python -B FILE --out ABS to freeze the grouped-soft protocol.

Requires completed data audit and sampler unit tests. Does not execute a model.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/project_c20250601/251105016/project/dllm_test')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True)
    out = p.parse_args().out
    assert not (out/'plan.json').exists(), 'Do not overwrite a frozen protocol'
    data = json.loads((out/'data_plan.json').read_text()); parent = Path(data['parent'])
    assert json.loads((out/'data_independent_audit.json').read_text())['status'] == 'passed_scoped_checks'
    source = out/'source'; source.mkdir(exist_ok=False); sources = {}
    for name in ('cdr_block_soft.py', 'run_cdr_block_context.py', 'audit_cdr_block_context.py', 'audit_cdr_block_data.py', 'freeze_cdr_block_context.py'):
        target = source/name; shutil.copy2(ROOT/'scripts/downstream'/name, target); sources[str(target)] = sha(target)
    test = ROOT/'scripts/tests/test_cdr_block_soft.py'
    shutil.copy2(test, source/test.name); sources[str(source/test.name)] = sha(source/test.name)
    legacy = parent/'confirm_cdr_new_inference_v1.py'
    if not legacy.exists():
        legacy = ROOT/'scripts/downstream/confirm_cdr_new_inference.py'
    # Copy the reused implementation so later working-tree edits cannot alter it.
    shutil.copy2(legacy, source/'legacy_soft.py'); sources[str(source/'legacy_soft.py')] = sha(source/'legacy_soft.py')
    families = [dict(name=f'g{g}_{rule}_t{t}_s{s}', groups=g, rule=rule, temperature=t, strength=s)
        for g in (2, 3) for rule in ('interleaved', 'contiguous') for t in (.5, 1.) for s in (.25, .5, 1.)]
    old = json.loads((parent/'plan.json').read_text())
    external = {**old['source_sha256'], **old['external_source_sha256']}
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), parent=str(parent), checkpoints=data['checkpoints'],
        batch_size=24, precision='fp32', primary_checkpoint='112000', families=families, prior_weights=[0., .5, .8],
        source_sha256=sources, external_source_sha256=external, legacy_soft_source=str(source/'legacy_soft.py'),
        rows_sha256=sha(out/'rows.json'), data_plan_sha256=sha(out/'data_plan.json'),
        development_rows_sha256=sha(parent/'rows.json'), data_audit_sha256=sha(out/'data_independent_audit.json'),
        development='Reuse the previous 318 modern-SAbDab H3 development rows only. This is repeated development selection, not fresh validation.',
        confirmation='Frozen selected configuration on all new OAS confirm rows: 2094 each H1/H2/H3, both frozen checkpoints. Reserve remains unscored.',
        selection='Best development mean sequence AAR strictly above single_pass; ties fewer logical NFE, then policy name. Freeze before confirmation. If none improve, do not open confirmation.',
        controls=['single_pass', 'original4', 'original8'], bootstrap=dict(repeats=10000, seed=20261015, unit='connected_component'),
        success='Require positive H3 paired component 95% interval versus single pass on BOTH checkpoints at <=4 logical NFE; disclose all H1/H2 changes and cost. H1/H2 regression over 0.25 pp blocks recommendation. No claim of guaranteed monotonic AAR with steps.',
        limitation='OAS supplementary distribution, not official benchmark. Exact heavy/H3 and supplied cluster exclusion within bound current training data, not general pretraining homology decontamination.')
    (out/'plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(dict(status='frozen', candidates=len(families)*3, plan_sha256=sha(out/'plan.json'))))


if __name__ == '__main__': main()
