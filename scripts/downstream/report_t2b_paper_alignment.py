"""Activate pllm/protenix_abtcr; python -B SCRIPT --run ABS [--watch].

Independently rescore saved cluster assignments, compare paper values, and
publish a separate reconstruction table only after all four workers succeed.
--watch waits in a detached process; it never submits or cancels GPU jobs.
"""
import argparse
import fcntl
import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
MODELS = ['NetTCR2.0', 'clusTCR', 'DeepRC', 'fixed_chain_50000']


def write(path, obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def independent(true, pred):
    pairs = Counter(zip(true, pred))
    a, b = Counter(true), Counter(pred)
    n = len(true)
    choose = lambda x: x*(x-1)/2
    ab = sum(choose(x) for x in pairs.values())
    aa, bb = sum(map(choose, a.values())), sum(map(choose, b.values()))
    expected = aa*bb/choose(n)
    ari = (ab-expected)/(.5*(aa+bb)-expected)
    mi = sum(v/n*math.log(v*n/(a[i]*b[j])) for (i,j),v in pairs.items())
    entropy = lambda counts: -sum(v/n*math.log(v/n) for v in counts.values())
    nmi = 2*mi/(entropy(a)+entropy(b))
    maxima = {j:max(v for (i,k),v in pairs.items() if k==j) for j in b}
    return dict(ari=ari, nmi=nmi, purity=sum(maxima.values())/n)


def build(run):
    import numpy as np
    import pandas as pd
    paper = pd.read_csv(run/'paper_kmeans_table2.csv').set_index('model')
    source = run/'author_source/dataset/clustering/TCRantigenData_unique_test.csv'
    true = pd.read_csv(source)['Antigen-epitope'].tolist()
    results, deltas, audit = {}, [], {}
    for name in MODELS:
        folder = run/name
        metrics = json.loads((folder/'metrics.json').read_text())
        labels_path = folder/'assignments.npz'
        assert hashlib.sha256(labels_path.read_bytes()).hexdigest() == metrics['assignments_sha256']
        curves = pd.read_csv(folder/'curve.csv')
        assert len(curves) == 76 and curves.groupby('arm').size().eq(19).all()
        labels = np.load(labels_path)
        assert len(labels.files) == 76
        errors = []
        for _, row in curves.iterrows():
            pred = labels[f"{row.arm}_k{int(row.k)}"]
            assert pred.shape == (9033,)
            scored = independent(true, pred.tolist())
            errors.extend(abs(scored[m]-float(row[m])) for m in scored)
        assert max(errors) < 1e-10, (name, max(errors))
        results[name] = metrics['means']['kmeans_raw_n10']
        audit[name] = dict(partitions=76, independent_max_error=max(errors),
                           assignment_sha256=metrics['assignments_sha256'])
        if name in paper.index:
            for m in ['ari','nmi','purity']:
                deltas.append(dict(model=name, metric=m, paper=float(paper.loc[name,m]),
                                   local=results[name][m], delta=results[name][m]-float(paper.loc[name,m])))
    pd.DataFrame(deltas).to_csv(run/'paper_comparison.csv',index=False)
    write(run/'independent_audit.json',audit)
    output = dict(status='complete', paper_reproduction_verified=False,
                  reason='Author evaluation code and unspecified parameters unresolved; baseline discrepancies remain',
                  primary_results=results, comparison=deltas, independent_audit=audit,
                  completed_utc=datetime.now(timezone.utc).isoformat())
    write(run/'summary.json',output)
    return paper, results


def publish(run, paper, results):
    raise RuntimeError('T2B removed from benchmark scope on 2026-10-14; publication disabled')
    rows = []
    for name in MODELS[:-1]:
        v = paper.loc[name]
        rows.append(f'| {name} [P] | {v.ari:.3f} | {v.nmi:.3f} | {v.purity:.3f} |')
        v = results[name]
        rows.append(f'| {name} 官方编码器 [L] | {v["ari"]:.6f} | {v["nmi"]:.6f} | {v["purity"]:.6f} |')
    v = results['fixed_chain_50000']
    rows.append(f'| Ours fixed-chain 75.6M / 50000 [L] | {v["ari"]:.6f} | {v["nmi"]:.6f} | {v["purity"]:.6f} |')
    table = '\n'.join(['| 方法 | mean ARI ↑ | mean NMI ↑ | mean purity ↑ |',
                       '|---|---:|---:|---:|', *rows])
    start, end = '<!-- t2b-paper-alignment-20261013:start -->','<!-- t2b-paper-alignment-20261013:end -->'
    for name in ['RESULTS.md','RESULTS_SUMMARY.md']:
        path = ROOT/'downstream/benchmark'/name
        content = path.read_text()
        title = '#### T2B 补充：完整 KMeans（本地重建）'
        note = '' if name == 'RESULTS_SUMMARY.md' else (
            '\n\n论文原表 [P] 与作者编码器接入本地完整 KMeans 的重建 [L]；raw features、n_init=10、seed=0，'
            '19个K取均值。n_init 是论文未披露的本地假设。baseline尚未复现通过，不作严格论文排名；'
            'Ours 使用本地GPU bf16提取特征；完整实验与各诊断分支均保留。'
            f'证据：[论文转录]({run}/paper_kmeans_table2.csv)、[逐项差异]({run}/paper_comparison.csv)、'
            f'[独立重算]({run}/independent_audit.json)、[完整结果]({run}/summary.json)。')
        block = start+'\n'+title+note+'\n\n'+table+'\n'+end+'\n\n'
        if start in content:
            a,b = content.index(start),content.index(end)+len(end)
            content = content[:a]+block.rstrip()+content[b:]
        else:
            # Insert immediately before T3 section (the anchor follows T2B).
            marker = '<a id="tcr-representation"></a>'
            if marker not in content:
                marker = '### T3 Representation'
            assert marker in content, name
            content = content.replace(marker,block+marker,1)
        tmp = path.with_suffix('.t2b.tmp');tmp.write_text(content);tmp.replace(path)
    stamp = datetime.now(timezone.utc).isoformat()
    updates = {
        ROOT/'downstream/benchmark/EVALUATION_PROCESS.md':
            f'2026-10-13 T2B完整KMeans重建4/4完成；304组簇标签独立ARI/NMI/Purity重算通过。作者baseline差异仍未消除，不宣称论文复现。结果已独立回填两版结果表；[核查产物]({run}/summary.json)。',
        ROOT/'downstream/benchmark/PROGRESS.md':
            f'2026-10-13 T2B完整KMeans本地重建结果已回填，原MiniBatchKMeans成绩保留；[核查]({run}/summary.json)。',
        ROOT/'PROJECT_PROCESS.md':
            f'2026-10-13 T2B本地评测4/4完成，结果与独立重算见[产物]({run}/summary.json)；严格论文复现仍未通过。',
        ROOT/'docs/EXPERIMENT_LOG.md':
            f'2026-10-13 T2B独立重建已完成并保存逐K标签与特征；[汇总与论文差异]({run}/summary.json)。',
    }
    for path, line in updates.items():
        content = path.read_text()
        if line not in content:
            head,tail = content.split('\n',1)
            content = head+'\n\n'+line+'\n'+tail
            if path.name == 'PROJECT_PROCESS.md':
                import re
                content = re.sub(r'Last updated: [^\n]+','Last updated: '+stamp,content,count=1)
            tmp=path.with_suffix('.t2b.tmp');tmp.write_text(content);tmp.replace(path)
    write(run/'published.json',dict(utc=stamp, status='published_as_local_reconstruction'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--watch',action='store_true')
    args=p.parse_args()
    while True:
        states={m:json.loads((args.run/m/'status.json').read_text()).get('status')
                if (args.run/m/'status.json').exists() else 'pending' for m in MODELS}
        write(args.run/'worker_status.json',states)
        if all(s=='success' for s in states.values()):break
        if 'failed' in states.values():raise RuntimeError(states)
        if not args.watch:return
        time.sleep(20)
    paper,results=build(args.run)
    lockpath=ROOT/'output/downstream_generation/v5_native_best_20261008/results_update.lock'
    with lockpath.open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        publish(args.run,paper,results)
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    main()
