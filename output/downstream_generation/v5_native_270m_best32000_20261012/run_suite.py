"""Run pinned 270M validation-best suite: python -B SCRIPT --run ABS --group 0.

Use --task NAME for a worker, --report for status, --check for CPU CDR gates.
Reuses the repaired task implementations and AirGen CDR evaluator. The result
block is keyed by run name, never overwriting the step45000 comparison.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
sys.path.insert(0,str(ROOT))


def write(path,data):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(path)


def report(run):
    plan=json.loads((run/'plan.json').read_text());selection=json.loads((run/'selection.json').read_text())
    counts=Counter();rows=[]
    from scripts.downstream.report_v5_suite import summarize
    for task in plan['tasks']:
        folder=run/task;status=folder/'status.json'
        state=json.loads(status.read_text()) if status.exists() else {'status':'pending'}
        counts[state['status']]+=1
        detail=state.get('error','待执行')
        if state['status']=='running':detail='运行中'
        if state['status']=='success':
            roots=[folder];loc=folder/'result_location.json'
            if loc.exists():roots.append(Path(json.loads(loc.read_text())['path']))
            values=[]
            for base in roots:
                for pattern in ('**/metrics.json','**/fewshot.json','**/summary.json','**/sequence_metrics.json'):
                    for path in sorted(base.glob(pattern)):
                        value=json.loads(path.read_text())
                        if task.startswith('t1-') and 'eval_sets' not in value:continue
                        if task.startswith('t4-') and 'generation_validity' in value:
                            v=value['generation_validity'];values.append('CDR3语法有效 '+str(sum(x['valid'] for x in v.values()))+'/'+str(sum(x['requested'] for x in v.values())))
                        values+=summarize(value)
            detail='；'.join(values[:9]) or '已完成，详见产物'
        detail=detail.replace('|','/').replace('\n',' ')[:750]
        rows.append(f"| {task} | {state['status']} | {detail} | [产物]({folder}) |")
    stamp=datetime.now(timezone.utc).isoformat()
    start=f'<!-- {run.name}:start -->';end=f'<!-- {run.name}:end -->'
    text='\n'.join([start,f"## 270M 验证最佳 checkpoint {selection['step']} 下游评测",'',
        f"固定选点：{selection['selected_at_utc']}；268.59M，无ESMC，V5数据；epoch {selection['epoch']:.5f}，eval_loss {selection['eval_loss']:.6f}。按训练验证集最低eval_loss选取，与已完成的75.6M三组权重独立对照；本次为训练中期选点。",'',
        '当前入口的Ophiuchus-Ab官方权重端到端复现尚未完成，不宣称论文复现已验收。同样37项；CDR复用AirGen对齐评测（Kong4步/SAb2步）；TCR采用单链禁止dot v2，CDR3提取/语法有效不代表全长β域或binding有效。原划分、探针、评分器保持；预训练重叠未完全排除。与45000对应修复协议比较，不能与其原无约束失败协议混用。',
        f"更新：{stamp}；状态：`{dict(counts)}`。选点及SHA：[selection]({run/'selection.json'})。",'',
        '| 任务 | 状态 | 结果／说明 | 证据 |','|---|---|---|---|',*rows,end])
    lock=ROOT/'output/downstream_generation/v5_native_best_20261008/results_update.lock'
    with lock.open('w') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        target=ROOT/'downstream/benchmark/RESULTS.md';old=target.read_text()
        if start in old:
            a=old.index(start);b=old.index(end,a)+len(end);old=old[:a]+text+old[b:]
        else:
            head,tail=old.split('\n',1);old=head+'\n\n'+text+'\n'+tail
        temp=target.with_suffix('.latest.tmp');temp.write_text(old);temp.replace(target)
        write(run/'summary.json',dict(updated_utc=stamp,counts=dict(counts)))
    if counts['success']==len(plan['tasks']):
        (run/'checkpoint/model.safetensors').unlink(missing_ok=True)
        write(run/'snapshot_released.json',dict(all_tasks_success=True,training_checkpoint_unchanged=True))


def aligned(run,folder):
    path=run/'airgen_evaluate.py'
    spec=importlib.util.spec_from_file_location('latest_aligned',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.OUT=folder;module.CKPT=run/'checkpoint';module.report=lambda:None
    return module


def task_worker(run,task):
    import downstream.v5_native_eval as native
    original=native.load
    cache={}
    def load(checkpoint,device='cuda'):
        key=(str(checkpoint),device)
        if key not in cache:
            cache[key]=original(checkpoint,device)
            source=cache[key][1]['source_root']
            if source in sys.path:sys.path.remove(source)
        return cache[key]
    native.load=load
    from scripts.downstream import report_v5_suite,run_v5_suite_task
    report_v5_suite.update=report
    if task.startswith('cdr-'):
        from scripts.downstream import run_v5_cdr_eval
        def cdr():
            module=aligned(run,run/task)
            sys.argv=[str(module.__file__),'--audit'];module.main()
            sys.argv=[str(module.__file__),'--dataset',task[4:]];module.main()
        run_v5_cdr_eval.main=cdr
    sys.argv=[str(run_v5_suite_task.__file__),'--run',str(run),'--task',task]
    run_v5_suite_task.main()
    # Remove only reproducible retrieval scratch after task exit/success.
    for p in (run/task).glob('novelty_tmp_*'):
        if p.is_dir() and not p.is_symlink():shutil.rmtree(p)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--group',type=int);p.add_argument('--task');p.add_argument('--report',action='store_true');p.add_argument('--check',action='store_true')
    a=p.parse_args();plan=json.loads((a.run/'plan.json').read_text())
    if a.report:report(a.run);return
    for path,expected in plan['code_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise RuntimeError('Evaluation code changed: '+path)
    if a.check:
        import torch
        torch.set_num_threads(4)
        folder=a.run/'cpu_gate';folder.mkdir(exist_ok=True);module=aligned(a.run,folder)
        sys.argv=[str(module.__file__),'--audit'];module.main()
        sys.argv=[str(module.__file__),'--check'];module.main();return
    if a.task:task_worker(a.run,a.task);return
    if a.group is None:p.error('Supply group/task/report/check')
    results=[]
    for task in plan['groups'][a.group]:
        if (a.run/'STOP').exists():break
        folder=a.run/task;folder.mkdir(exist_ok=True)
        if (folder/'status.json').exists():continue
        with (folder/'run.log').open('x') as log:
            result=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--run',str(a.run),'--task',task],stdout=log,stderr=subprocess.STDOUT)
        if not (folder/'status.json').exists():write(folder/'status.json',dict(status='failed',error='Worker exited before task status',returncode=result.returncode))
        results.append(dict(task=task,returncode=result.returncode));write(a.run/f'group_{a.group}.json',dict(results=results));report(a.run)
    if any(x['returncode'] for x in results):raise SystemExit(1)


if __name__=='__main__':main()
