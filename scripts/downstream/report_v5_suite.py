"""Update only the native V5 execution block in EVALUATION_PROCESS.md: python -B SCRIPT --run ABS.

Uses completed metric artifacts; never turns a running/failed task into a score.
"""
import argparse
import fcntl
import json
import sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/project_c20250601/251105016/project/dllm_test')
sys.path.insert(0,str(ROOT))
START='<!-- native-v5-best-20261008:start -->';END='<!-- native-v5-best-20261008:end -->'


def summarize(value,prefix=''):
    out=[]
    if isinstance(value,dict):
        for k,v in value.items():
            key=prefix+'.'+k if prefix else k
            if isinstance(v,(int,float)) and not isinstance(v,bool) and any(term in k.lower() for term in ('auroc','auprc','spearman','accuracy','aar','purity','nmi','f1','pearson')):out.append(f'{key}={v:.4f}')
            elif isinstance(v,dict) and k not in ('identity','feature_manifest','provenance','per_pmhc','per_target','per_fold','per_assay','folds'):out+=summarize(v,key)
    return out


def update(run):
    plan=json.loads((run/'plan.json').read_text());selection=json.loads((run/'selection.json').read_text())
    rows=[];counts={}
    for task in plan['tasks']:
        folder=run/task;path=folder/'status.json'
        state=json.loads(path.read_text()) if path.exists() else {'status':'pending'}
        status=state['status'];counts[status]=counts.get(status,0)+1
        roots=[folder]
        loc=folder/'result_location.json'
        if loc.exists():roots.append(Path(json.loads(loc.read_text())['path']))
        artifacts=[]
        for root in roots:
            for pattern in ('**/metrics.json','**/fewshot.json','**/summary.json','**/sequence_metrics.json','**/summary.csv'):
                artifacts+=list(root.glob(pattern))
        metrics=[]
        if status=='success':
            for p in artifacts:
                if p.suffix=='.json':
                    try:metrics+=summarize(json.loads(p.read_text()))
                    except (ValueError,TypeError):pass
        detail='；'.join(metrics[:6]) if metrics else ('已完成，详见产物' if status=='success' else state.get('error','等待执行/产物'))
        if task.startswith('ag-chimera') and status=='success' and (folder/'metrics.json').exists():
            result=json.loads((folder/'metrics.json').read_text())
            if result.get('protocol')=='chimera_epitope_group_h3_sequence_v1':
                summary=result['summary']
                detail=(f"epitope_group H3 / {result['numbering_scheme']} / K={result['candidates']} / N={result['n_complexes']}；"
                        f"AAR={summary['best_aar']['mean']:.4f}；CAAR={summary['best_caar']['mean']:.4f}；"
                        '固定Fv骨架，纯序列条件；结构/epitope指标未测')
        if task.startswith('t4-') and status=='success' and (folder/'metrics.json').exists():
            result=json.loads((folder/'metrics.json').read_text())
            validity=result.get('generation_validity',{})
            if validity:
                valid=sum(v['valid'] for v in validity.values());requested=sum(v['requested'] for v in validity.values())
                detail=f'单链禁止dot v2；CDR3提取/语法有效 {valid}/{requested}（非完整β链或binding验证）；'+detail
        detail=detail.replace('|','/').replace('\n',' ')[:550]
        if task == 'ag-abbibench:g6_LC_benchmarking_data':
            continue
        rows.append(f"| {task} | {status} | {detail} | [目录]({folder}) |")
    content='\n'.join([START,'## 当前 V5 数据新模型：best checkpoint 45,000 全下游评测','',f"更新：{datetime.now(timezone.utc).isoformat()}。固定选点：训练验证 loss **1.992721**，step **45,000**，epoch **3.02074**；75.6M、无ESMC、V5对象302布局。独立评测，不停止当前训练。",'',
        '评测范围为现行 Ours AB/TCR 全任务，包含 UpTCR 与 AB–AG 三任务四子集；不重跑外部baseline，不纳入 VHH/Humanization/PPI/NbBench/FLAb。旧表保留为历史对照。',
        '输入适配沿用原划分、标签、探针和评分器；表征排除synthetic位置，CDR补全使用已知区间，生成由模型预测终止符、不给真实目标长度。TCR修复v2在采样时禁止单链任务的dot，原失败输出保留于attempt_history；CDR3提取成功不代表完整β链有效或实验证实binding。新生成协议与旧固定长度结果分开解释。V5预训练重叠未完全排除，不宣称无泄漏。','',
        f"状态计数：`{json.dumps(counts,ensure_ascii=False)}`。未完成项没有成绩；失败项保留具体原因。选择与完整计划：[selection.json]({run/'selection.json'})、[plan.json]({run/'plan.json'})。",'',
        '| 任务 | 状态 | 当前指标／说明 | 证据 |','|---|---|---|---|',*rows,END])
    target=ROOT/'downstream/benchmark/EVALUATION_PROCESS.md'
    with (run/'results_update.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        text=target.read_text()
        if START in text:
            a=text.index(START);b=text.index(END,a)+len(END);text=text[:a]+content+text[b:]
        else:
            first,rest=text.split('\n',1);text=first+'\n\n'+content+'\n'+rest
        tmp=target.with_suffix('.md.native.tmp');tmp.write_text(text);tmp.replace(target)
    from downstream.benchmark.ab_antigen.abbibench_scope import summarize_run
    (run/'summary.json').write_text(json.dumps({'counts':counts,'abbibench_benchmark16':summarize_run(run),'updated_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
    if counts.get('success')==len(plan['tasks']):
        # This is only the evaluation hardlink; never unlink the training checkpoint.
        snapshot=run/'checkpoint'/'model.safetensors'
        if snapshot.exists():snapshot.unlink()
        (run/'snapshot_released.json').write_text(json.dumps({'all_tasks_success':True,'training_checkpoint_unchanged':True})+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();update(a.run)
