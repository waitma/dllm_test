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

ROOT=Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--group',type=int,required=True);a=p.parse_args()
    plan=json.loads((a.run/'plan.json').read_text());tasks=plan['groups'][a.group]
    status={'group':a.group,'status':'running','results':[]};path=a.run/f'group_{a.group}.json'
    def save():path.write_text(json.dumps(status,indent=2)+'\n')
    save()
    for task in tasks:
        if (a.run/'STOP').exists():status['status']='stopped';save();return
        folder=a.run/task;folder.mkdir(exist_ok=True)
        if (folder/'status.json').exists():
            status['results'].append({'task':task,'status':'already_attempted_not_retried'});save();continue
        for source,expected in plan['code_sha256'].items():
            if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=expected:raise RuntimeError('Evaluation code changed: '+source)
        with (folder/'run.log').open('x') as log:
            result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/downstream/run_v5_suite_task.py'),'--run',str(a.run),'--task',task],stdout=log,stderr=subprocess.STDOUT)
        if not (folder/'status.json').exists():
            (folder/'status.json').write_text(json.dumps({'status':'failed','error':'worker exited before task status','returncode':result.returncode})+'\n')
        status['results'].append({'task':task,'returncode':result.returncode,'finished_unix':time.time()});save()
        subprocess.run([sys.executable,'-B',str(ROOT/'scripts/downstream/report_v5_suite.py'),'--run',str(a.run)],check=True)
    status['status']='finished';save()
    if any(x.get('returncode',0)!=0 for x in status['results']):sys.exit(1)

if __name__=='__main__':main()
