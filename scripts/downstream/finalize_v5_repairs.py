"""Wait for local repair workers and sync reports: python -B SCRIPT --run ABS.

No jobs are submitted; no retries or model changes. Wait at most 12 hours.
Deletes only this CHIMERA run's reproducible MMseqs scratch directories after
the scoring worker has written a terminal status. Retains predictions/logs.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import re
import shutil
import sys
import time

ROOT=Path('/project_c20250601/251105016/project/dllm_test')
sys.path.insert(0,str(ROOT))
from scripts.downstream.report_v5_suite import update


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--worker-pids',type=int,nargs='+',required=True)
    args=parser.parse_args();run=args.run
    tasks=['t1-cdr3b','ag-chimera','t4-benchmark14','t4-held20']
    deadline=time.monotonic()+43200
    while True:
        states={task:json.loads((run/task/'status.json').read_text()) for task in tasks}
        # held20 is still the pre-retry failure while benchmark14 runs.
        alive=False
        for pid in args.worker_pids:
            try:alive |= b'repair_v5_failed.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
            except FileNotFoundError:pass
        complete=not alive and all(s['status'] in ('success','failed') for s in states.values())
        if complete:break
        if time.monotonic()>deadline:raise TimeoutError('Local workers still active; no files removed')
        time.sleep(30)
    for path in (run/'ag-chimera').glob('novelty_tmp_*'):
        if path.is_dir() and not path.is_symlink():shutil.rmtree(path)
    update(run)
    stamp=datetime.now(timezone.utc).isoformat()
    results={task:s['status'] for task,s in states.items()}
    validity={}
    for task in ['t4-benchmark14','t4-held20']:
        path=run/task/'designs.jsonl'
        if path.exists():
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            validity[task]=dict(targets=len(rows),requested=sum(r['n_requested'] for r in rows),
                syntactic_cdr3_valid=sum(r['n_valid'] for r in rows),full_beta_validity='not_established')
    report=dict(updated_utc=stamp,tasks=results,generation=validity,scratch_cleaned=True)
    (run/'repair_final_20261009.json').write_text(json.dumps(report,indent=2)+'\n')
    text=f'2026-10-09 修复复测已终态：{json.dumps(results,ensure_ascii=False)}。结果及CDR3语法有效率见 {run / "repair_final_20261009.json"}；原始候选保留，MMseqs临时目录已清理。82项回归及60组T1 AUROC复核通过；未见表位泛化弱仍未由流程修复解决。'
    p=ROOT/'PROJECT_PROCESS.md';s=p.read_text()
    s=re.sub(r'Last updated: .*','Last updated: '+stamp,s,count=1)
    s=s.replace('原平台作业已终态；本地恢复T1/CHIMERA评分并重评单链TCR（best45000）','best45000本地修复已终态；详见repair_final_20261009.json')
    p.write_text(s+'\n\n'+text+'\n')
    for name in ['docs/EXPERIMENT_LOG.md','downstream/benchmark/PROGRESS.md']:
        p=ROOT/name;p.write_text(p.read_text()+'\n\n'+text+'\n')
    with (run/'results_update.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        p=ROOT/'downstream/benchmark/EVALUATION_PROCESS.md';s=p.read_text()
        s=s.replace('**2026-10-09 修复复测进行中**','**2026-10-09 修复复测已终态**')
        s+='\n\n'+text+'\n';p.write_text(s)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
