"""Iteration study: python -B SCRIPT --run ABS --dataset kong|sab23.

Use --check for CPU context/gold-invariance checks across every iteration value.
Fixed checkpoint, data, masks, argmax, precision and batch64; only max_iter
changes. This is a test-set sensitivity study, not model/protocol selection.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path('/project_c20250601/251105016/project/dllm_test')
sys.path.insert(0,str(ROOT))


def write(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(path)


def report(run):
    plan=json.loads((run/'plan.json').read_text());rows=[];counts={}
    for iteration in plan['iterations']:
        values=[];states=[];cost=[]
        for dataset,modes in [('kong',['cdrh1','cdrh2','cdrh3']),('sab23',['cdrh1','cdrh2','cdrh3','cdrl1','cdrl2','cdrl3'])]:
            folder=run/f'iter{iteration}'/dataset;p=folder/'status.json'
            state=json.loads(p.read_text()) if p.exists() else {'status':'pending'}
            states.append(dataset+': '+state['status']);counts[state['status']]=counts.get(state['status'],0)+1
            metrics=folder/'metrics.json'
            if state['status']=='success' and metrics.exists():
                data=json.loads(metrics.read_text())['modes'];values += [f"{data[m]['airgen_aar_percent']:.2f}" for m in modes]
            else:values += ['—']*len(modes)
            if (folder/'timing.json').exists():cost.append(dataset+' '+str(round(json.loads((folder/'timing.json').read_text())['seconds'],1))+'s')
        rows.append('| '+str(iteration)+' | '+' | '.join(values)+' | '+'; '.join(states)+' | '+'; '.join(cost)+' |')
    start=f'<!-- {run.name}:start -->';end=f'<!-- {run.name}:end -->'
    block='\n'.join([start,'## CDR infilling iteration 对照：checkpoint 269000','',
        '固定1/2/4/8/16/32步，全部Kong十折和SAb23H2六CDR；表内为AAR (%)。复用AirGen解码，原始位置/X掩码，argmax、temperature1、CFG0、batch64；Kong bf16、SAb fp32。只改变max_iter；默认协议仍为Kong4步/SAb2步。',
        '这是测试集上的敏感性分析，不能挑最高值替换默认主成绩后宣称无偏提升。耗时为预加载模型后的整数据集墙钟时间（含评分与写盘），不作为严格硬件性能基准。',
        f'选点：[selection]({run/"selection.json"})；逐样本预测与折分数：[产物]({run})。','',
        '| Iteration | Kong H1 | H2 | H3 | SAb H1 | H2 | H3 | L1 | L2 | L3 | 状态 | 耗时 |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|',*rows,end])
    with (ROOT/'output/downstream_generation/v5_native_best_20261008/results_update.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);target=ROOT/'downstream/benchmark/EVALUATION_PROCESS.md';text=target.read_text()
        if start in text:
            a=text.index(start);b=text.index(end,a)+len(end);text=text[:a]+block+text[b:]
        else:head,tail=text.split('\n',1);text=head+'\n\n'+block+'\n'+tail
        temp=target.with_suffix('.iterations.tmp');temp.write_text(text);temp.replace(target)
        write(run/'summary.json',dict(updated_utc=datetime.now(timezone.utc).isoformat(),counts=counts))
    if counts.get('success')==2*len(plan['iterations']):
        (run/'checkpoint/model.safetensors').unlink(missing_ok=True)
        write(run/'snapshot_released.json',dict(all_settings_success=True,training_weights_unchanged=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--dataset',choices=['kong','sab23']);p.add_argument('--check',action='store_true');p.add_argument('--report',action='store_true');a=p.parse_args()
    plan=json.loads((a.run/'plan.json').read_text())
    if a.report:report(a.run);return
    for path,digest in plan['code_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    import torch
    torch.set_num_threads(4)
    import downstream.v5_native_eval as native
    model,identity=native.load(a.run/'checkpoint','cpu' if a.check else 'cuda')
    if identity['source_root'] in sys.path:sys.path.remove(identity['source_root'])
    native.load=lambda *args,**kwargs:(model,identity)
    spec=importlib.util.spec_from_file_location('iteration_evaluator',a.run/'evaluate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.CKPT=a.run/'checkpoint';module.report=lambda:report(a.run)
    if not a.check and not a.dataset:p.error('--dataset required for evaluation')
    for iteration in plan['iterations']:
        folder=a.run/(('check_iter' if a.check else 'iter')+str(iteration));folder.mkdir(exist_ok=True)
        module.OUT=folder;module.STEPS={d:iteration for d in module.MODES}
        # Dataset workers share the same audit bytes. Distinct filenames avoid
        # concurrent writes to the evaluator's atomic temporary file.
        audit=folder/'protocol_audit.json'
        if not audit.exists():
            with (a.run/'audit.lock').open('w') as handle:
                fcntl.flock(handle,fcntl.LOCK_EX)
                if not audit.exists():sys.argv=[str(module.__file__),'--audit'];module.main()
        if a.check:
            sys.argv=[str(module.__file__),'--check'];module.main();continue
        if (folder/a.dataset/'status.json').exists():raise FileExistsError(folder/a.dataset)
        torch.cuda.synchronize();begin=time.monotonic()
        try:
            sys.argv=[str(module.__file__),'--dataset',a.dataset];module.main()
            torch.cuda.synchronize()
            write(folder/a.dataset/'timing.json',dict(seconds=time.monotonic()-begin,gpu=torch.cuda.get_device_name(),iteration=iteration,model_load_excluded=True))
        finally:report(a.run)


if __name__=='__main__':main()
