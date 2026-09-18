"""Persistently monitor the approved local TCR queues and audit finished jobs.

Activate pllm then protenix_abtcr. Run in a dedicated tmux session:
python scripts/downstream/monitor_tcr_49000_local.py --plan /abs/monitor_plan.json
Polls every 30 seconds, exits on a failure/dead producer or after all ten jobs
and their result audits finish. Never submits, cancels, kills or retries jobs.
Writes only its own new runtime directory and explicitly planned audit reports.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from scripts.downstream.run_tcr_local_short import ROOT, now, sha

DONE='executed_pending_result_audit'
PASSED={'passed','artifact_consistency_passed'}


def atomic(path,content):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(content,indent=2))
    temporary.replace(path)


def read_queues(plan):
    jobs={}
    for item in plan['queues']:
        state=json.loads(Path(item['status']).read_text())
        assert state['global_step']==49000 and state['checkpoint']==plan['checkpoint']
        assert state['plan_sha256']==item['plan_sha256'] and sha(state['plan'])==item['plan_sha256']
        assert state['runner_sha256']==item['runner_sha256'] and sha(item['runner'])==item['runner_sha256']
        if state['status'] not in ('running',DONE):
            raise RuntimeError(f'producer terminal error: {item["status"]}: {state.get("error",state["status"])}')
        if state['status']=='running':
            assert state['host']==socket.gethostname(),'cannot inspect a different host producer'
            cmdline=Path(f'/proc/{state["pid"]}/cmdline').read_bytes().decode().replace('\0',' ')
            if state['plan'] not in cmdline or Path(item['runner']).name not in cmdline:
                raise RuntimeError('producer missing or PID reused')
        for job in state['jobs']:
            assert job['key'] not in jobs,'duplicated local job'
            assert job['cloud_status_at_migration']=='Killed'
            assert sha(job['yaml'])==job['yaml_sha256'],'YAML changed during monitoring'
            jobs[job['key']]=job
    assert set(jobs)==set(plan['job_keys']) and len(jobs)==10
    return jobs


def verify_report(group):
    report=json.loads(Path(group['report']).read_text())
    assert report['status'] in PASSED and report['global_step']==49000
    assert report['auditor_sha256']==group['auditor_sha256']==sha(group['auditor'])
    if 'tracks' in report:
        assert set(report['tracks'])==set(group['members'])
        artifacts=list(report['tracks'].values())
    elif 'tasks' in report:
        assert set(report['tasks'])==set(group['members'])
        artifacts=list(report['tasks'].values())
    else:
        assert report['result']['dataset']==group['members'][0]
        artifacts=[report['result']]
    for artifact in artifacts:
        assert artifact['status'] in PASSED
        for field in ('artifact_sha256','prediction_files_sha256'):
            for path,expected in artifact.get(field,{}).items():
                assert sha(path)==expected,f'audited artifact drift: {path}'
        for name in ('summary','run_config'):
            if name+'_sha256' in artifact:
                assert sha(Path(artifact['output'])/(name+'.json'))==artifact[name+'_sha256']
    return report['status']


def collect(jobs,audited):
    results={}
    for key,job in jobs.items():
        if key not in audited:
            continue
        output=Path(job['result_dir'])
        if key.startswith('t1-'):
            payload=json.loads((output/'summary.json').read_text())
        elif key.startswith('t2'):
            payload=json.loads((output/'metrics.json').read_text())
            payload={k:v for k,v in payload.items() if k not in ('load_info','embedder')}
        elif key.startswith('t3'):
            full=json.loads((output/'fewshot.json').read_text())
            payload={k:full[k] for k in ('protocol','shots','seeds','auroc_by_shot',
                'n_epitopes_by_shot','n_train','n_test','n_universe','n_background','input_fields') if k in full}
        else:
            full=json.loads((output/'metrics.json').read_text())
            payload={k:full[k] for k in ('dataset','n_pmhc','candidate_count','greedy_count','summary','interpretation')}
        results[key]={'audit':audited[key],'result_dir':str(output),'metrics':payload}
    return {'global_step':49000,'updated_utc':now(),'tasks':results,
        'interpretation':'only the pinned 49000 run; local protocols and scientific limitations remain',
        'complete':len(results)==10}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text())
    assert plan['global_step']==49000
    assert len(plan['job_keys'])==len(set(plan['job_keys']))==10
    groups=plan['audit_groups']
    covered=[key for group in groups for key in group['jobs']]
    assert set(covered)==set(plan['job_keys']) and len(covered)==10
    for group in groups:
        assert sha(group['auditor'])==group['auditor_sha256']
        if Path(group['report']).exists():
            verify_report(group)
    read_queues(plan)
    if args.check_only:
        print('MONITOR_PLAN_CHECK_PASSED',flush=True)
        return
    output=Path(plan['monitor_dir'])
    output.mkdir(parents=True,exist_ok=True)
    with (output/'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        status_path=output/'status.json'
        if status_path.exists():
            raise FileExistsError('monitor already started; inspect it rather than replaying')
        state={'status':'running','global_step':49000,'host':socket.gethostname(),'pid':os.getpid(),
            'start_utc':now(),'plan':str(args.plan.resolve()),'plan_sha256':sha(args.plan),
            'monitor_sha256':sha(__file__),'audited':{},'polls':0,'active_audit':None}
        active=None

        def save():
            state['updated_utc']=now()
            atomic(status_path,state)

        save()
        try:
            while True:
                jobs=read_queues(plan)
                state['polls']+=1
                state['jobs']={key:{k:v for k,v in job.items() if k in
                    ('status','start_utc','end_utc','elapsed_seconds','exit_code','log','result_dir')} for key,job in jobs.items()}
                state['alerts']=[]
                state['gpu']=subprocess.check_output(['nvidia-smi','--id=0',
                    '--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True,timeout=10).strip()
                for key,job in jobs.items():
                    if job['status']=='running' and Path(job['log']).exists():
                        log=Path(job['log'])
                        state['jobs'][key]['log_bytes']=log.stat().st_size
                        state['jobs'][key]['log_age_seconds']=round(time.time()-log.stat().st_mtime,1)
                        if time.time()-log.stat().st_mtime>900:
                            state['alerts'].append(f'{key}: no log update for >900s; not proof of deadlock')
                        if key.startswith('t4-'):
                            designs=Path(job['result_dir'])/'designs.jsonl'
                            if designs.exists():
                                with designs.open() as handle:
                                    state['jobs'][key]['completed_pmhc_lines']=sum(line.endswith('\n') for line in handle)
                if active is not None and active['process'].poll() is not None:
                    code=active['process'].returncode
                    active['log'].close()
                    if code:
                        raise RuntimeError(f'audit failed: {active["group"]["name"]}, exit={code}')
                    quality=verify_report(active['group'])
                    for key in active['group']['jobs']:
                        state['audited'][key]={'status':quality,'report':active['group']['report']}
                    active=None
                    state['active_audit']=None
                if active is None:
                    for group in groups:
                        if all(key in state['audited'] for key in group['jobs']):
                            continue
                        if not all(jobs[key]['status']==DONE for key in group['jobs']):
                            continue
                        if Path(group['report']).exists():
                            quality=verify_report(group)
                            for key in group['jobs']:
                                state['audited'][key]={'status':quality,'report':group['report']}
                            continue
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
                        logfile=(output/(group['name']+'.log')).open('x')
                        proc=subprocess.Popen([sys.executable,group['auditor'],*group['args'],'--report',group['report']],
                            cwd=ROOT,env=env,stdout=logfile,stderr=subprocess.STDOUT)
                        active={'process':proc,'group':group,'log':logfile}
                        state['active_audit']={'name':group['name'],'pid':proc.pid,'log':str(logfile.name)}
                        break
                collected=collect(jobs,state['audited'])
                collected.update(checkpoint=plan['checkpoint'],checkpoint_sha256=plan['checkpoint_sha256'])
                atomic(output/'results_49000.json',collected)
                if len(state['audited'])==10 and all(j['status']==DONE for j in jobs.values()):
                    state['status']='all_tasks_executed_and_audits_passed'
                    save()
                    print(now(),state['status'],flush=True)
                    return
                save()
                print(now(),'executed',sum(j['status']==DONE for j in jobs.values()),
                    'audited',len(state['audited']),'running',[k for k,j in jobs.items() if j['status']=='running'],
                    'audit',state['active_audit'],'alerts',state['alerts'],flush=True)
                time.sleep(30)
        except Exception as exc:
            state.update(status='attention_required',error=repr(exc))
            save()
            raise
        finally:
            if active is not None and active['process'].poll() is None:
                state['unfinished_audit_pid']=active['process'].pid
                save()


if __name__=='__main__':
    main()
