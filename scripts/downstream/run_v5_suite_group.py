"""Execute a fixed task group: python -B SCRIPT --run ABS --group 0.

Continue independent tasks after a failure, never retry a completed/failed task.
Explicit STOP in the run directory prevents starting further tasks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/project_c20250601/251105016/project/dllm_test')
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--group',type=int,required=True)
    p.add_argument('--include-extra-assays',action='store_true');a=p.parse_args()
    from downstream.benchmark.ab_antigen.abbibench_scope import select_tasks
    plan=json.loads((a.run/'plan.json').read_text());tasks=select_tasks(plan['groups'][a.group],include_extra_assays=a.include_extra_assays)
    status={'group':a.group,'status':'running','results':[]};path=a.run/f'group_{a.group}.json'
    def save():path.write_text(json.dumps(status,indent=2)+'\n')
    save()
    for task in tasks:
        if task == 't2b':  # Removed from benchmark scope on 2026-10-14.
            continue
        if (a.run/'STOP').exists():status['status']='stopped';save();return
        folder=a.run/task;folder.mkdir(exist_ok=True)
        if (folder/'status.json').exists():
            status['results'].append({'task':task,'status':'already_attempted_not_retried'});save();continue
        for source,expected in plan['code_sha256'].items():
            if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=expected:raise RuntimeError('Evaluation code changed: '+source)
        with (folder/'run.log').open('x') as log:
            extra=['--include-extra-assays'] if a.include_extra_assays else []
            result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/downstream/run_v5_suite_task.py'),'--run',str(a.run),'--task',task,*extra],stdout=log,stderr=subprocess.STDOUT)
        if not (folder/'status.json').exists():
            (folder/'status.json').write_text(json.dumps({'status':'failed','error':'worker exited before task status','returncode':result.returncode})+'\n')
        status['results'].append({'task':task,'returncode':result.returncode,'finished_unix':time.time()});save()
        subprocess.run([sys.executable,'-B',str(ROOT/'scripts/downstream/report_v5_suite.py'),'--run',str(a.run)],check=True)
    status['status']='finished';save()
    if any(x.get('returncode',0)!=0 for x in status['results']):sys.exit(1)

if __name__=='__main__':main()
