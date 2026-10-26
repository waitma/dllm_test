"""Prepare LoRA3000 second-block representation evaluation without submitting.

Activate pllm/protenix_abtcr and run from repository root:
python -B scripts/downstream/prepare_post_training_layer_suite.py
Creates a distinct run and seven single-GPU YAMLs, and runs the real-weight CPU gate.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from downstream.post_training_layer_eval import sha


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def main():
    import yaml
    baseline=ROOT/'output/downstream_generation/post_training_lora_s42_r1_20261014'
    reference=ROOT/'output/downstream_generation/bert_full_token_75m_best48828_20261017'
    run=ROOT/'output/downstream_generation/post_training_lora_s42_layer2_20261017'
    run.mkdir(exist_ok=False)
    shutil.copytree(baseline/'checkpoint',run/'checkpoint')
    shutil.copy2(ROOT/'scripts/downstream/run_post_training_layer_suite.py',run/'run_suite.py')
    groups=[['gdp_a1','specificity','m396'], ['t1-cdr3b'],
            ['t1-others_cdr3b','t1-cdr3ab','t1-others_longab'], ['uptcr'],
            ['t2a','t3broad','t3deep'], ['ag-binding'], ['ag-spr']]
    tasks=[t for group in groups for t in group]
    old=json.loads((baseline/'plan.json').read_text())
    comparison={}
    for task in tasks:
        assert json.loads((baseline/task/'status.json').read_text())['status']=='success',task
        sources=list((baseline/task).rglob('metrics.json'))+list((baseline/task).rglob('summary.json'))
        location=baseline/task/'result_location.json'
        if location.exists():
            dest=Path(json.loads(location.read_text())['path'])
            sources+=list(dest.rglob('metrics.json'))+list(dest.rglob('fewshot.json'))+list(dest.rglob('summary.json'))
        comparison[task]={str(p):sha(p) for p in sources}
    previous=json.loads((reference/'plan.json').read_text())
    data={p:sha(p) for p in previous['data_sha256'] if 'abbibench' not in p}
    for path,digest in data.items():
        assert digest==previous['data_sha256'][path],path
    audit=json.loads((reference/'input_audit.json').read_text())
    assert audit['status']=='passed'
    for task in tasks:
        # The complete existing input audit checked tighter 166/134 residue caps.
        keys=['uptcr-'+s for s in ('majority','few-shot','zero-shot')] if task=='uptcr' else [task]
        for key in keys:
            assert audit['checks'][key]['over_capacity']==0,key
    source_paths={Path(p) for p in old['code_sha256'] if Path(p).is_relative_to(ROOT) and not Path(p).is_relative_to(ROOT/'output')}
    source_paths.update(ROOT/part for part in (
        'downstream/post_training_layer_eval.py','downstream/post_training_eval.py',
        'scripts/downstream/run_post_training_layer_suite.py',
        'scripts/downstream/prepare_post_training_layer_suite.py',
        'scripts/downstream/run_v5_suite_task.py','downstream/grammar/ab_probes.py',
        'downstream/grammar/tcr_features.py','downstream/benchmark/tcr_binding/query_protocol.py',
        'downstream/benchmark/tcr_binding/run_uptcr_ours.py'))
    source_paths.add(run/'run_suite.py')
    code={str(p):sha(p) for p in sorted(source_paths)}
    identity={**old['identity'],'layer_index_one_based':2,'readout':'block2_pre_final_norm'}
    write(run/'plan.json',dict(tasks=tasks,groups=groups,step=3000,
          model='post_training_lora_seed42_3000_block2',model_sha256=identity['model_sha256'],identity=identity,
          checkpoint=str(run/'checkpoint'),readout='block2_pre_final_norm',retain_checkpoint=True,
          selection='user-requested fixed layer2; same previously specified LoRA weights; no layer sweep',
          scope='13 frozen-representation task entries; existing splits, heads and budgets',
          baseline_run=str(baseline),baseline_result_sha256=comparison,
          code_sha256=code,data_sha256=data,
          input_audit_binding=dict(path=str(reference/'input_audit.json'),sha256=sha(reference/'input_audit.json'),
                                   rationale='same hashed task files; existing tighter capacity audit covers fixed187/154'),
          created_at_utc=datetime.now(timezone.utc).isoformat(),
          protocol_notes=['AB: global observed-residue mean at block2; no extra normalization.',
                          'TCR: existing completion/observed pooling and hidden relation unchanged.',
                          'Specificity head100 epochs; GDP original Ridge; m396 both existing split views.',
                          'CoV-UniBind: replace embedding only; final-layer PLL unchanged, compare probe separately.',
                          'GPU uses original BF16 autocast; compare GDP CPU FP32 as a precision check.',
                          'Original data and historical results retained; no data cleaning in this run.']))
    with (run/'prepare_check.log').open('x') as log:
        result=subprocess.run([sys.executable,'-B',str(run/'run_suite.py'),'--run',str(run),'--check','--device','cpu'],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('CPU layer gate failed; inspect '+str(run/'prepare_check.log'))
    configs=[]
    template=ROOT/'eval_jobs/bert_full_token_best48828_g0_20261017.yml'
    for i,group in enumerate(groups):
        config=yaml.safe_load(template.read_text())
        config['TaskName']=f'post-lora-s42-layer2-g{i}-20261017'
        config['Description']='Fixed LoRA3000 block2 representation: '+','.join(group)
        config['Tags']=['immune','post-training','layer2','representation']
        entry=config['Entrypoint'].replace(str(reference),str(run)).replace(reference.name,run.name)
        entry=entry.replace('TCR_EVAL_STEP=48828','TCR_EVAL_STEP=3000')
        entry=entry.replace(previous['model_sha256'],identity['model_sha256'])
        entry=entry.replace('--group 0',f'--group {i}').replace('group_0.log',f'group_{i}.log')
        config['Entrypoint']=entry
        path=ROOT/f'eval_jobs/post_training_lora_s42_layer2_g{i}_20261017.yml'
        with path.open('x') as f:yaml.safe_dump(config,f,sort_keys=False)
        configs.append(str(path))
    source=run/'source';source.mkdir()
    for path in source_paths:
        if path.is_relative_to(ROOT) and not path.is_relative_to(ROOT/'output'):
            dest=source/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    write(run/'preparation.json',dict(status='cpu_passed_not_submitted',configs=configs,tasks=tasks,
          baseline_run=str(baseline),created_at_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(run=str(run),configs=configs,status='cpu_passed_not_submitted'),indent=2))


if __name__=='__main__':main()
