"""Four independent GPU lanes; bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_confirm_4gpu.sh --gpus 0,1,2,3."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import traceback
import uuid
from architecture_confirm.common import ROOT,JOBS,STEPS,EPOCHS,MICRO,GLOBAL,DATA_SHA,read,write,sha,digest,stamp,lock
from architecture_confirm.common import prepare,snapshot,source_binding,trial,valid_profile,qualified,completed

OUT=ROOT/'output/arch70m_confirm_4gpu_20261031'
SOURCE_SHA='3c9c424d816b9fda7adfba5047d9e075ac15dc036342b59461edd35031e50971'
STOP=threading.Event();ACTIVE={};ACTIVE_LOCK=threading.Lock()


def gpu_ids(value):
    ids=value.split(',')
    ids=[str(int(x)) if x.isdecimal() else x for x in ids]
    if len(ids)!=4 or len(set(ids))!=4 or any(not x or any(c.isspace() for c in x) for x in ids):
        raise argparse.ArgumentTypeError('Exactly four distinct GPU IDs are required, for example 0,1,2,3.')
    return ids


def plan(devices):
    return [dict(variant=v,seed=s,gpu=devices[i%4],stage='confirm',epochs=EPOCHS,
                 total_steps=STEPS,micro_batch=MICRO,global_batch=GLOBAL) for i,(v,s) in enumerate(JOBS)]


def resolve_devices(devices):
    from arch70m_seed_validation import cuda_device_id
    mapping={};uuids=[]
    for requested in devices:
        selector=str(int(cuda_device_id(requested)))
        value=subprocess.check_output(['nvidia-smi','--id='+requested,'--query-gpu=uuid','--format=csv,noheader'],text=True)
        rows=[x.strip() for x in value.splitlines() if x.strip()]
        if len(rows)!=1 or not rows[0].startswith('GPU-'):raise ValueError('Missing or ambiguous GPU: '+requested)
        mapping[requested]=dict(cuda_selector=selector,nvml_uuid=rows[0]);uuids.append(rows[0])
    if len(set(uuids))!=4 or len({x['cuda_selector'] for x in mapping.values()})!=4:
        raise ValueError('Four distinct physical GPUs are required; no jobs started.')
    return mapping


def worker_env(source,selector):
    blocked={'RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','MASTER_ADDR','MASTER_PORT',
             'CUDA_LAUNCH_BLOCKING','ARCH70M_FROZEN_SOURCE'}
    env={k:v for k,v in os.environ.items() if k not in blocked and not k.startswith(('NCCL_','TORCHELASTIC_'))}
    env.update(CUDA_VISIBLE_DEVICES=selector,CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONPATH=str(source),
        ARCH70M_PROJECT_ROOT=str(ROOT),CUBLAS_WORKSPACE_CONFIG=':4096:8',USE_TF='0',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    return env


def stop_children():
    STOP.set()
    with ACTIVE_LOCK:processes=list(ACTIVE.values())
    for process in processes:
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass


def parallel_lanes(jobs,function,callback=None):
    """One serial lane per GPU; result callbacks run on the parent thread."""
    lanes={gpu:[j for j in jobs if j['gpu']==gpu] for gpu in dict.fromkeys(j['gpu'] for j in jobs)}
    events=queue.Queue();results=[]
    def lane(items):
        for job in items:
            try:
                if STOP.is_set():raise RuntimeError('Four-GPU launch cancelled.')
                result=function(job)
            except Exception:
                result=dict(job,status='failed',exit_code=1,error=traceback.format_exc())
            events.put(result)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(lane,items) for items in lanes.values()]
        for _ in jobs:
            result=events.get();results.append(result)
            if callback is not None:callback(result)
        for future in futures:future.result()
    return results


def install_profile(out,shadow,variant,seed):
    src=trial(shadow,variant,seed);dst=trial(out,variant,seed)
    p=read(src/'profile.json');valid_profile(p,variant,seed,SOURCE_SHA)
    if (dst/'profile.json').exists():
        # The hash is part of any existing checkpoint binding: never overwrite it.
        valid_profile(read(dst/'profile.json'),variant,seed,SOURCE_SHA)
    else:
        for name in ('profile.json','profile_attempts.json','cuda_device.json'):write(dst/name,read(src/name))
        write(dst/'status.json',dict(status='qualified',step=0,total_steps=STEPS,updated=stamp()))
    return dict(cuda_device=read(src/'cuda_device.json'),actual_profile_sha256=sha(src/'profile.json'),
                canonical_profile_sha256=sha(dst/'profile.json'),profile=p)


def run_child(args,source,job,mapping,attempt,profile=False):
    v,s=job['variant'],job['seed'];selector=mapping[job['gpu']]['cuda_selector']
    if profile:
        target=attempt/'qualification'/('gpu_'+selector)/f'{v}_s{s}'
        (target/'data').mkdir(parents=True,exist_ok=True)
        # Each qualification has its own index/scratch path: no concurrent truncation.
        import shutil
        shutil.copyfile(args.output_root/'data/index.json',target/'data/index.json')
    else:target=args.output_root
    t=trial(args.output_root,v,s);t.mkdir(parents=True,exist_ok=True)
    mode='--profile-one' if profile else '--train-one'
    command=[sys.executable,'-B','-m','architecture_confirm.runner',mode,'--variant',v,'--seed',str(s),'--output-root',str(target)]
    path=attempt/'logs'/f'{v}_s{s}_{"qualification" if profile else "train"}.log';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as log:
        process=subprocess.Popen(command,cwd=source,env=worker_env(source,selector),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        with ACTIVE_LOCK:ACTIVE[process.pid]=process
        if STOP.is_set():
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        receipt=dict(pid=process.pid,command=command,gpu=selector,requested_gpu=job['gpu'],started=stamp(),log=str(path))
        write(t/('four_gpu_qualification_process.json' if profile else 'process.json'),receipt)
        print(json.dumps(dict(event='launch',variant=v,seed=s,gpu=selector,mode=mode,pid=process.pid)),flush=True)
        try:code=process.wait()
        finally:
            with ACTIVE_LOCK:ACTIVE.pop(process.pid,None)
    if code:
        error=path.read_text()[-5000:]
        status=read(t/'status.json') if (t/'status.json').exists() else dict(step=0)
        write(t/'status.json',dict(status,status='cancelled_by_user' if STOP.is_set() else 'failed',
              failed_mode=mode,exit_code=code,error=error,updated=stamp()))
        return dict(job,status='failed',exit_code=code,error=error)
    if profile:
        evidence=install_profile(args.output_root,target,v,s)
        write(t/'four_gpu_runtime.json',dict(evidence,**receipt,updated=stamp()))
        return dict(job,status='qualified',exit_code=0,evidence=evidence)
    if not completed(args.output_root,v,s,SOURCE_SHA):raise RuntimeError('Missing complete final confirmation receipt.')
    return dict(job,status='completed',training_exit_code=0,exit_code=0)


def validate_actual_devices(results):
    devices={};names=set()
    for result in results:
        if result['status']!='qualified':raise RuntimeError('Some GPU qualification failed; formal training was not launched.')
        device=result['evidence']['cuda_device'];gpu=result['gpu'];identity=device['cuda_uuid'];names.add(device['device'])
        if gpu in devices and devices[gpu]!=identity:raise ValueError('CUDA identity changed within one GPU lane.')
        devices[gpu]=identity
    if len(devices)!=4 or len(set(devices.values()))!=4:raise ValueError('Four independent actual CUDA devices are required.')
    if len(names)!=1:raise ValueError('Use four GPUs of the same model for this paired confirmation.')
    return devices


def execute(args):
    jobs=plan(args.gpus)
    if args.dry_run:
        print(json.dumps(dict(independent_single_gpu=True,parallel_gpus=4,output_root=str(args.output_root),
            source_sha256=SOURCE_SHA,assignments=jobs),indent=2));return
    from architecture_confirm.runner import publish
    if args.publish_only:
        print(publish(args.output_root,'refresh four-GPU confirmation report',not args.no_push));return
    mapping=resolve_devices(args.gpus)  # Reject a one-GPU developer node before mutation or GPU work.
    if digest(source_binding())!=SOURCE_SHA:raise ValueError('The frozen confirmation model/trainer/protocol changed.')
    branch=subprocess.check_output(['git','-C',str(ROOT),'branch','--show-current'],text=True).strip()
    if branch!='explore/arch70m-v8-20261028':raise ValueError('Use the exploration branch.')
    prepare(args.output_root)
    from experiments.arch70m.suite import verify_data
    verify_data(args.output_root)
    with lock(args.output_root/'controller.lock',blocking=False):
        for v,s in JOBS:
            with lock(trial(args.output_root,v,s)/'trial.lock',blocking=False):pass
        source,signature=snapshot(args.output_root)
        attempt=args.output_root/'four_gpu'/('attempt_'+stamp().replace(':','')+'_'+uuid.uuid4().hex[:8])
        attempt.mkdir(parents=True)
        launcher_files={name:sha(ROOT/name) for name in ('arch70m_confirm_4gpu.py','launch_arch70m_confirm_4gpu.sh')}
        manifest=dict(started=stamp(),controller_pid=os.getpid(),parallel_gpus=4,independent_single_gpu=True,
            source_sha256=signature,data_sha256=DATA_SHA,scheduler_files=launcher_files,scheduler_sha256=digest(launcher_files),
            assignments=jobs,devices=mapping,output_root=str(args.output_root),attempt=str(attempt),phase='qualification')
        write(attempt/'manifest.json',manifest);write(args.output_root/'four_gpu_execution.json',manifest)
        results=parallel_lanes(jobs,lambda job:run_child(args,source,job,mapping,attempt,True))
        actual=validate_actual_devices(results)
        profiles={f'{v}_s{s}':sha(trial(args.output_root,v,s)/'profile.json') for v,s in JOBS}
        write(args.output_root/'qualification.json',dict(status='passed',source_sha256=signature,data_sha256=DATA_SHA,
            micro_batch=MICRO,updated=stamp(),profiles=profiles,actual_cuda_devices=actual,attempt=str(attempt)))
        qualified(args.output_root,signature)
        write(attempt/'qualification.json',dict(status='passed',results=results,actual_cuda_devices=actual))
        write(args.output_root/'four_gpu_execution.json',dict(manifest,phase='training',qualified=stamp(),actual_cuda_devices=actual))
        def train_job(job):
            if completed(args.output_root,job['variant'],job['seed'],signature):return dict(job,status='completed',exit_code=0,training_exit_code=0,skipped=True)
            return run_child(args,source,job,mapping,attempt)
        def finished(result):
            message=f"{result['variant']} seed{result['seed']} {result['status']} on GPU{result['gpu']}"
            try:publication=publish(args.output_root,message,not args.no_push)
            except Exception:
                publication=dict(status='local_report_failed',error=traceback.format_exc())
                write(attempt/f"publication_{result['variant']}_s{result['seed']}.json",publication)
            print(json.dumps(dict(event='finished',**result,publication_status=publication['status'])),flush=True)
        results=parallel_lanes(jobs,train_job,finished)
        failed=[r for r in results if r['status']!='completed']
        from architecture_confirm.decision import decision
        verdict=decision(args.output_root)
        write(args.output_root/'four_gpu_execution.json',dict(manifest,phase='finished',finished=stamp(),failed_jobs=failed,decision=verdict['decision']))
        write(args.output_root/'continuation.json',dict(updated=stamp(),goal_complete=False,required_root_review=True,
            decision=verdict['decision'],next_action=verdict.get('next_action','Resolve incomplete confirmation trials.')))
        if failed:raise RuntimeError('Some confirmation trials failed; preserved their original errors.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',type=gpu_ids,default=gpu_ids(os.environ.get('ARCH_GPUS','0,1,2,3')))
    p.add_argument('--output-root',type=Path,default=OUT);p.add_argument('--dry-run',action='store_true')
    p.add_argument('--no-push',action='store_true');p.add_argument('--publish-only',action='store_true')
    args=p.parse_args();args.output_root=args.output_root.resolve()
    def stop(signum,frame):stop_children();raise KeyboardInterrupt('Four-GPU controller stopped by signal '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:execute(args)
    finally:stop_children()


if __name__=='__main__':main()
