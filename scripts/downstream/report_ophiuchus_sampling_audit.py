"""Verify and publish completed local sampling audit: activate pllm; python -B FILE."""
from pathlib import Path
import collections,fcntl,json,hashlib
import numpy as np
ROOT=Path('/project_c20250601/251105016/project/dllm_test')
RUN=ROOT/'output/downstream_generation/ophiuchus_sampling_audit_20261012'

def main():
    summary=json.loads((RUN/'summary.json').read_text());assert summary['status']=='complete'
    assert len(summary['results'])==11
    checked=[]
    for d in summary['results']:
        key=f"{d['dataset']}_{d['strategy']}_iter{d['iteration']}_{d['precision']}";folder=RUN/key
        by=collections.defaultdict(lambda:collections.defaultdict(list));seen=set()
        for line in (folder/'predictions.jsonl').open():
            x=json.loads(line);ident=(x['mode'],x['fold'],x['row']);assert ident not in seen;seen.add(ident)
            assert len(x['prediction_ids'])==len(x['reference_ids'])==x['n_masked']
            correct=sum(a==b for a,b in zip(x['prediction_ids'],x['reference_ids']));assert correct==x['n_correct']
            val=float(np.float32(correct)/np.float32(x['n_masked']));assert abs(val-x['aar'])<1e-7
            by[x['mode']][x['fold']].append(val)
        for mode,folds in by.items():
            assert len(folds)==(10 if d['dataset']=='kong' else 1)
            assert sum(map(len,folds.values()))==(3127 if d['dataset']=='kong' else 60)
            means=[np.mean(v)*100 for v in folds.values()]
            score=float(np.mean(means)) if d['dataset']=='kong' else float(np.round(means[0]/100,4)*100)
            assert abs(score-d['modes'][mode]['aar_percent'])<1e-5
        checked.append(dict(setting=key,records=len(seen),passed=True,predictions_sha256=hashlib.sha256((folder/'predictions.jsonl').read_bytes()).hexdigest()))
    assert sum(x['records'] for x in checked)==49065
    (RUN/'independent_score_audit.json').write_text(json.dumps(checked,indent=2)+'\n')
    start='<!-- ophiuchus-sampling-audit-20261012:start -->';end='<!-- ophiuchus-sampling-audit-20261012:end -->'
    lines=[start,'## Ophiuchus-Ab 官方权重：本地 CDR 采样复现核查（2026-10-12）','',
        '本地 A100 全量实测。官方 checkpoint 严格加载，无 missing/unexpected；直接执行 AirGen 原始 decoder/collator 方法，使用现有兼容主干；11个对应主干源码文件与AirGen逐字一致。原始解码与历史本地解码在两数据集小样本iter2输出一致。',
        '固定 batch64、temperature1、CFG0、seed42；Kong每CDR 3127条/10折宏平均，SAb每CDR60条/先round4再乘100。随机采样仅单seed、单步诊断，不能外推所有温度/步数。共11组数据集设置、51个CDR指标、49,065个区域预测，逐条匹配及聚合独立重算通过。',
        '在本次测试设置中，Kong的argmax iter1/bf16和SAb23H2的argmax iter2/fp32整体最接近论文。此为测试集协议敏感性核查，不证明恢复了论文未公开的实际参数；未逐CDR拼接最优值。现有270M两套评测继续按各自固定设置运行，未改写旧成绩。','',
        '| 数据集 | 采样 | iter | 精度 | H1 | H2 | H3 | L1 | L2 | L3 | 对论文MAE pp |','|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|',
        '| Kong论文 | [P] | 未披露 | 未披露 | 75.50 | 70.18 | 43.55 | — | — | — | — |',
        '| SAb23H2论文 | [P] | 未披露 | 未披露 | 74.80 | 68.60 | 36.80 | 81.10 | 80.10 | 73.70 | — |']
    for d in summary['results']:
        vals=[f"{d['modes'][m]['aar_percent']:.4f}" if m in d['modes'] else '—' for m in ['cdrh1','cdrh2','cdrh3','cdrl1','cdrl2','cdrl3']]
        lines.append('| '+' | '.join([d['dataset'],d['strategy'],str(d['iteration']),d['precision'],*vals,f"{d['mae_pp']:.4f}"])+' |')
    lines+=['',f"证据：[运行计划与哈希]({RUN/'plan.json'})、[完整指标]({RUN/'summary.json'})、[独立重算]({RUN/'independent_score_audit.json'})、[主干源码比对]({RUN/'backbone_source_comparison.json'})。论文：[本地PDF Table 1/2](/root/oph_paper/oph.pdf)。",end]
    block='\n'.join(lines)
    with (ROOT/'output/downstream_generation/v5_native_best_20261008/results_update.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);p=ROOT/'downstream/benchmark/EVALUATION_PROCESS.md';s=p.read_text()
        if start in s:
            a=s.index(start);b=s.index(end,a)+len(end);s=s[:a]+block+s[b:]
        else:
            head,tail=s.split('\n',1);s=head+'\n\n'+block+'\n'+tail
        temp=p.with_suffix('.audit.tmp');temp.write_text(s);temp.replace(p)
    print('Verified and published',sum(x['records'] for x in checked),'predictions')
if __name__=='__main__':main()
