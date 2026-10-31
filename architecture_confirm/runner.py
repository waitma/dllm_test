"""Run full confirmation locally: bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_confirm.sh --gpu 0."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from .common import ROOT,OUT,VARIANTS,SEEDS,JOBS,STEPS,EPOCHS,MICRO,GLOBAL,DATA_SHA,PREVIOUS_SHA
from .common import trial,read,write,sha,digest,stamp,lock,prepare,snapshot,source_binding,valid_profile,qualified,completed


def trainer():
    from experiments.arch70m import train
    from .models import build
    train.build=build;train.source_binding=source_binding
    return train


def profile_one(args):
    """Reuse the already tested GPU qualification math in a fresh worker process."""
    from architecture_hierarchy import runner as memory
    t=trial(args.output_root,args.variant,args.seed);signature=digest(source_binding())
    if (t/'profile.json').exists():valid_profile(read(t/'profile.json'),args.variant,args.seed,signature);return
    scratch=args.output_root/'qualification_jobs'
    (scratch/'data').mkdir(parents=True,exist_ok=True)
    shutil.copyfile(args.output_root/'data/index.json',scratch/'data/index.json')
    memory.SEED=args.seed;memory.STEPS=STEPS;memory.source_binding=source_binding;memory.trainer=trainer
    memory.valid_profile=lambda p,v,s:valid_profile(p,v,args.seed,s)
    memory.profile_one(argparse.Namespace(output_root=scratch,profile_one=args.variant))
    raw=scratch/'trials'/f'screen_{args.variant}_s{args.seed}'
    for name in ('profile.json','profile_attempts.json','cuda_device.json'):
        write(t/name,read(raw/name))
    write(t/'status.json',dict(status='qualified',step=0,total_steps=STEPS,updated=stamp()))


def publish(out,message,push):
    from .report import report
    from arch70m_seed_validation import push_results
    def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    with lock(ROOT/'output/arch70m_publish.lock'):
        if git('branch','--show-current')!='explore/arch70m-v8-20261028':raise ValueError('Wrong branch.')
        if git('diff','--cached','--name-only'):raise ValueError('Unrelated staged edits.')
        done,result=report(out)
        p=ROOT/'PROJECT_PROCESS.md'
        p.write_text(f'> Last updated: {stamp()} (UTC)。完整确认 {done}/9 完成；{message}；decision={result["decision"]}。A0/A5(B0)/B1×seed42/43/44，各3epoch/3828步、micro32/global512；原V8数据保持。报告 {ROOT}/docs/arch70m_results/confirmation_20261031/REPORT.md。未达到冻结判据不宣告稳定提升。\n\n'+p.read_text())
        git('add','--','PROJECT_PROCESS.md','docs/arch70m_results/confirmation_20261031');git('diff','--cached','--check')
        git('-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-m','arch70m confirmation: '+message)
        result=push_results() if push else dict(status='disabled')
        write(out/'publication.json',dict(result,commit=git('rev-parse','HEAD'),updated=stamp()))
        return result


def child(args,source,variant,seed,profile=False):
    t=trial(args.output_root,variant,seed);t.mkdir(parents=True,exist_ok=True)
    env={k:v for k,v in os.environ.items() if k not in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE',
        'GROUP_RANK','MASTER_ADDR','MASTER_PORT','ARCH70M_FROZEN_SOURCE','CUDA_LAUNCH_BLOCKING')
        and not k.startswith(('NCCL_','TORCHELASTIC_'))}
    env.update(CUDA_VISIBLE_DEVICES=args.gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONPATH=str(source),USE_TF='0',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    mode='--profile-one' if profile else '--train-one'
    command=[sys.executable,'-B','-m','architecture_confirm.runner',mode,'--variant',variant,'--seed',str(seed),'--output-root',str(args.output_root)]
    path=t/('qualification.log' if profile else 'console.log')
    with path.open('a') as log:
        start=log.tell();process=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
        write(t/('qualification_process.json' if profile else 'process.json'),dict(pid=process.pid,command=command,
              started=stamp(),gpu=args.gpu,log_start=start))
        print(dict(event='launch',variant=variant,seed=seed,mode=mode,pid=process.pid),flush=True)
        code=process.wait()
    if code:
        with path.open('rb') as log:
            log.seek(0,2);size=log.tell();log.seek(max(start,size-4000));error=log.read().decode(errors='replace')
        status=read(t/'status.json') if (t/'status.json').exists() else dict(step=0)
        write(t/'status.json',dict(status,status='failed',exit_code=code,failed_mode=mode,error=error,updated=stamp()))
        print(dict(event='failed',variant=variant,seed=seed,exit_code=code,error=error[-700:]),flush=True)
    return code


def execute(args):
    if not args.gpu.isdecimal():raise ValueError('Use numeric CUDA GPU selector.')
    if args.dry_run:
        print(dict(jobs=JOBS,epochs=EPOCHS,total_steps=STEPS,micro_batch=MICRO,global_batch=GLOBAL,gpu=args.gpu));return
    prepare(args.output_root)
    from experiments.arch70m.suite import verify_data
    verify_data(args.output_root)
    with lock(ROOT/'output'/f'arch70m_local_gpu{args.gpu}.lock',blocking=False),lock(args.output_root/'controller.lock',blocking=False):
        source,signature=snapshot(args.output_root)
        execution=dict(started=stamp(),controller_pid=os.getpid(),jobs=[list(j) for j in JOBS],epochs=EPOCHS,total_steps=STEPS,
            micro_batch=MICRO,global_batch=GLOBAL,source_sha256=signature,data_sha256=DATA_SHA,
            previous_source_sha256=PREVIOUS_SHA,gpu=args.gpu,phase='qualification')
        write(args.output_root/'execution.json',execution)
        for v,s in JOBS:
            path=trial(args.output_root,v,s)/'profile.json'
            if path.exists():valid_profile(read(path),v,s,signature)
            elif child(args,source,v,s,True):
                try:publish(args.output_root,f'{v} seed{s} qualification failed',not args.no_push)
                except Exception:write(args.output_root/'publication_error.json',dict(error=traceback.format_exc(),updated=stamp()))
                raise RuntimeError('Qualification failed; no unqualified training launched.')
        profiles={f'{v}_s{s}':sha(trial(args.output_root,v,s)/'profile.json') for v,s in JOBS}
        write(args.output_root/'qualification.json',dict(status='passed',source_sha256=signature,data_sha256=DATA_SHA,
            micro_batch=MICRO,updated=stamp(),profiles=profiles))
        qualified(args.output_root,signature)
        write(args.output_root/'execution.json',dict(execution,phase='training',qualified=stamp()))
        failures=[]
        for v,s in JOBS:
            if completed(args.output_root,v,s,signature):continue
            code=child(args,source,v,s)
            if code==0 and not completed(args.output_root,v,s,signature):raise RuntimeError('Missing complete final receipt.')
            if code:failures.append([v,s])
            try:publication=publish(args.output_root,f'{v} seed{s} '+('failed' if code else 'completed'),not args.no_push)
            except Exception:
                publication=dict(status='local_report_failed')
                write(args.output_root/'publication_error.json',dict(error=traceback.format_exc(),updated=stamp()))
            print(dict(event='finished',variant=v,seed=s,training_exit_code=code,publication_status=publication['status']),flush=True)
        from .decision import decision
        result=decision(args.output_root)
        write(args.output_root/'execution.json',dict(execution,phase='finished',finished=stamp(),failed_jobs=failures,decision=result['decision']))
        write(args.output_root/'continuation.json',dict(updated=stamp(),goal_complete=False,
            required_root_review=True,decision=result['decision'],report=str(ROOT/'docs/arch70m_results/confirmation_20261031/REPORT.md'),
            next_action=result.get('next_action','Resolve incomplete trials before statistical inference.')))
        if failures:raise RuntimeError('Incomplete confirmation jobs: '+str(failures))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',default='0');p.add_argument('--output-root',type=Path,default=OUT)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--no-push',action='store_true')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--profile-one',action='store_true');mode.add_argument('--train-one',action='store_true')
    p.add_argument('--variant',choices=VARIANTS);p.add_argument('--seed',type=int,choices=SEEDS)
    args=p.parse_args();args.output_root=args.output_root.resolve()
    if args.profile_one or args.train_one:
        if args.variant is None or args.seed is None:p.error('Worker requires variant and seed.')
        if args.profile_one:profile_one(args)
        else:
            qualified(args.output_root,digest(source_binding()))
            trainer().run(argparse.Namespace(output_root=str(args.output_root),stage='confirm',variant=args.variant,seed=args.seed,wandb=False))
    else:execute(args)


if __name__=='__main__':main()
