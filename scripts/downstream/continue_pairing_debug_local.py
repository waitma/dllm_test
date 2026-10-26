"""Activate pllm; python -B SCRIPT to finish the two pending local controls.

Wait for the two already-running diagnostic processes; allow at most two GPU
diagnostics at once. Require canceled cloud duplicates and unchanged YAMLs.
Never retry a failed run or reuse an existing output folder. STOP prevents new
launches while existing processes are observed to completion.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/project_c20250601/251105016/project/dllm_test')
DEBUG=ROOT/'output/downstream_generation/pairing_sampler_debug_20261014'


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def live(receipt, folder):
    path=Path('/proc')/str(receipt['pid'])/'cmdline'
    if not path.exists():return False
    command=path.read_bytes().decode().replace('\0',' ')
    if folder not in command:raise RuntimeError('PID reused by another process')
    return True


def event(message):
    with (ROOT/'PROJECT_PROCESS.md').open('a') as stream:
        stream.write(f'\n{now()} pairing本地队列：{message}\n')
    print(now(),message,flush=True)


def foreign_gpu_users():
    """Reserve launches for this diagnostic and the verified matrix placeholder."""
    foreign=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            devices={os.readlink(fd) for fd in (path/'fd').iterdir()}
            if '/dev/nvidia7' not in devices:continue
            command=(path/'cmdline').read_bytes().decode().replace('\0',' ')
            placeholder=(path.name=='1909' and command.strip()=='python occupy.py'
                         and os.readlink(path/'cwd')==str(ROOT.parent/'flow_antibody_tcr'))
            diagnostic=(str(DEBUG/'runner_v1.py') in command or str(DEBUG/'runner_v2.py') in command)
            if not (placeholder or diagnostic):foreign.append(dict(pid=int(path.name),command=command))
        except OSError:continue
    return foreign


def main():
    import yaml
    lock=(DEBUG/'local_queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pending=[('fixed50000_gumbel_prob','t-20261014185159-2422w',
              ROOT/'eval_jobs/pairing_debug_fixed50000_gumbel_prob_20261014.yml'),
             ('fixed112000_legacy','t-20261014185500-5bm4j',
              ROOT/'eval_jobs/pairing_debug_fixed112000_legacy_20261014.yml')]
    configs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for _,_,p in pending}
    running={
        'fixed50000_legacy_dedicated':(DEBUG/'local_fixed50000_execution.json',None),
        'v7_native':(DEBUG/'local_v7_execution.json',None),
    }
    completed={};failed=False;start=time.monotonic()
    state=dict(pid=os.getpid(),started_utc=now(),yaml_hashes=configs)
    while running or pending:
        if time.monotonic()-start>14400:raise TimeoutError('Four-hour observation limit')
        for folder,(receipt_path,process) in list(running.items()):
            receipt=json.loads(receipt_path.read_text())
            if process is not None:
                code=process.poll()
                if code is None:continue
                receipt.update(exit_code=code,completed_utc=now());write(receipt_path,receipt)
            else:
                if live(receipt,folder):continue
                if 'exit_code' not in receipt:raise RuntimeError('Process ended without completion receipt: '+folder)
            result=json.loads((DEBUG/folder/'status.json').read_text())
            ok=receipt['exit_code']==0 and result['status']=='success'
            completed[folder]=dict(success=ok,exit_code=receipt['exit_code'])
            failed=failed or not ok
            del running[folder];event(f'{folder} 终态 {completed[folder]}；无自动重试。')
        foreign=foreign_gpu_users()
        while pending and len(running)<2 and not failed and not foreign and not (DEBUG/'STOP').exists():
            folder,task,path=pending[0]
            assert hashlib.sha256(path.read_bytes()).hexdigest()==configs[str(path)]
            query=subprocess.run(['volc','ml_task','get','-i',task,'-o','json'],capture_output=True,text=True,check=True)
            cloud=json.loads(query.stdout[query.stdout.index('[{'):])[0]
            assert cloud['Status']=='Killed',cloud
            assert not (DEBUG/folder).exists(),'Existing output cannot be reused'
            config=yaml.safe_load(path.read_text())
            env=dict(os.environ,COMP_CHAIN_ANARCI_N_JOBS='4',LOKY_MAX_CPU_COUNT='4')
            process=subprocess.Popen(['/bin/bash','-c',config['Entrypoint']],env=env,cwd=ROOT)
            receipt_path=DEBUG/f'local_{folder}_execution.json'
            write(receipt_path,dict(pid=process.pid,started_utc=now(),source=str(path),
                yaml_sha256=configs[str(path)],cloud_duplicate=cloud,output=str(DEBUG/folder)))
            running[folder]=(receipt_path,process);pending.pop(0)
            event(f'{folder} 启动，shell PID{process.pid}；同冻结YAML入口，云{task}已Killed；[产物]({DEBUG/folder})。')
        state.update(updated_utc=now(),status='failed' if failed else 'running',
                     running=list(running),pending=[v[0] for v in pending],completed=completed,
                     foreign_gpu_users=foreign)
        write(DEBUG/'local_queue_status.json',state)
        if not running and (failed or (DEBUG/'STOP').exists()):break
        if running or pending:time.sleep(30)
    state.update(status='failed' if failed else 'stopped' if pending else 'success',completed_utc=now())
    write(DEBUG/'local_queue_status.json',state)
    if failed:raise RuntimeError('A controlled run failed; outputs retained, no retries')


if __name__=='__main__':
    main()
