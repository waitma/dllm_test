"""Advance the approved sequence benchmark after the GPU smoke succeeds.

Run: python -m downstream.benchmark.ab_antigen.sequence_pipeline --watch
Sequential single-GPU submission only. STOP file pauses new submissions, never
cancels existing jobs. A failed job stops advancement for inspection.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import subprocess
import time
from downstream.benchmark.ab_antigen.sequence_data import ROOT,write_json
RUN=ROOT/'output/downstream_generation/ab_ag_sequence_v1_noesmc150000'


def volc(*args):
    r=subprocess.run(['bash',str(ROOT/'scripts/volc-no-proxy.sh'),'ml_task',*args,'-o','json'],cwd=ROOT,capture_output=True,text=True)
    if r.returncode:raise RuntimeError(r.stderr[-1500:])
    match=re.search(r'(\[\{.*\}\]|\{.*\})\s*$',r.stdout,re.S)
    if not match:raise ValueError('Missing JSON in Volc output')
    return json.loads(match.group(1))


def log_transition(task_id,key,status,yaml_path=None):
    import yaml
    yaml_path=yaml_path or str(ROOT/'eval_jobs/eval_ab_ag_sequence_v1_noesmc150000_smoke.yml')
    task_name=yaml.safe_load(Path(yaml_path).read_text())['TaskName']
    p=ROOT/'PROJECT_PROCESS.md';s=p.read_text();now=datetime.now(timezone.utc).isoformat()
    a=s.index('## Active Volc Training Tasks');b=s.find('\n## ',a+3)
    if b<0:b=len(s)
    block=s[a:b];block='\n'.join(line for line in block.split('\n') if not (line.startswith('|') and f'`{task_id}`' in line))
    if status not in ['Success','Failed','Killed']:
        marker='|---|---|---|---:|---:|---|'
        row=f'| `{task_id}` | `{task_name}`（eval） | c20250601 闲时 | 1 | sequence benchmark | {status}（{now}） |'
        if marker not in block:raise ValueError('Active table layout changed')
        block=block.replace(marker,marker+'\n'+row,1)
    block=re.sub(r'Last updated:.*',f'Last updated: {now}（AB–AG pipeline）',block,count=1)
    s=s[:a]+block+s[b:]
    s+=f'\n## {now[:10]} AB–AG sequence pipeline {status}\n\n- Task `{task_id}` ({key}), {status}, c20250601, Preemptible=true, one GPU. YAML `{yaml_path or "smoke YAML recorded above"}`.\n- Automated advancement of the approved no-ESMC150000 benchmark; no training task changes.\n'
    p.write_text(s)
    results=ROOT/'downstream/benchmark/RESULTS.md';text=results.read_text()
    state_path=RUN/'pipeline_state.json'
    if state_path.exists():
        state=json.loads(state_path.read_text())
        lines=['<!-- AB_AG_SEQUENCE_RUNTIME -->','',f'运行状态更新：{now}（平台状态，不等于指标通过）','',
            '| 作业 | Task ID | 平台状态 |','|---|---|---|',
            f'| smoke | `{state["smoke"]}` | {state.get("smoke_status","Submitted")} |']
        lines += [f'| {k} | `{v["id"]}` | {v["status"]} |' for k,v in state['jobs'].items()]
        lines += ['', '<!-- /AB_AG_SEQUENCE_RUNTIME -->']
        block='\n'.join(lines)
        if '<!-- AB_AG_SEQUENCE_RUNTIME -->' in text:
            text=re.sub(r'<!-- AB_AG_SEQUENCE_RUNTIME -->.*?<!-- /AB_AG_SEQUENCE_RUNTIME -->',lambda _:block,text,flags=re.S)
        else:text+='\n'+block+'\n'
        results.write_text(text)


def advance():
    if (RUN/'STOP').exists():return 'paused_by_STOP'
    plan=json.loads((RUN/'execution_plan.json').read_text())
    path=RUN/'pipeline_state.json';state=json.loads(path.read_text()) if path.exists() else {'smoke':plan['smoke_task'],'jobs':{}}
    smoke=volc('get','-i',state['smoke'])[0]
    if smoke['Status']!=state.get('smoke_status'):
        state['smoke_status']=smoke['Status'];write_json(path,state);log_transition(state['smoke'],'smoke',smoke['Status'])
    if smoke['Status']!='Success':return 'smoke_'+smoke['Status']
    for task in ['binding','spr','abbibench']:
        f=RUN/f'smoke_{task}/metrics.json'
        if not f.exists() or json.loads(f.read_text()).get('status')!='complete':raise ValueError('Smoke artifact missing: '+str(f))
    if not (RUN/'smoke_chimera/sequence_metrics.json').exists():raise ValueError('Generation smoke metrics missing')
    for job in plan['jobs']:
        key=job['key'];known=state['jobs'].get(key)
        if known:
            status=volc('get','-i',known['id'])[0]['Status']
            if status!=known['status']:
                known['status']=status;write_json(path,state);log_transition(known['id'],key,status,job['yaml'])
            if status in ['Failed','Killed']:return 'stopped_'+key+'_'+status
            if status!='Success':return key+'_'+status
            expected='sequence_metrics.json' if job['task']=='chimera' else 'metrics.json'
            if not (Path(job['output'])/expected).exists():raise ValueError('Platform Success without metrics: '+key)
            continue
        response=volc('submit','--conf',job['yaml'])
        tid=response['Id'];state['jobs'][key]={'id':tid,'status':'Submitted'}
        write_json(path,state);log_transition(tid,key,'Submitted',job['yaml'])
        return 'submitted_'+key
    return 'complete'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    while True:
        try:status=advance()
        except Exception as exc:
            write_json(RUN/'pipeline_error.json',{'error':str(exc),'time':datetime.now(timezone.utc).isoformat()});raise
        print(datetime.now(timezone.utc).isoformat(),status,flush=True)
        if not args.watch or status=='complete' or status.startswith(('stopped_','smoke_Failed','smoke_Killed','paused_')):break
        time.sleep(60)
