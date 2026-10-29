"""Run local micro32 trials: bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_hierarchy.sh --gpu 0."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from .common import ROOT, DEFAULT_OUT, VARIANTS, MICRO, GLOBAL, SEED, STEPS, DATA_SHA, PREVIOUS_SHA
from .common import read, write, sha, digest, stamp, lock, prepare, snapshot, source_binding, valid_profile, valid_qualification, completed


def trainer():
    from architecture_round2.runner import trainer as previous_trainer
    train = previous_trainer()
    train.source_binding = source_binding
    return train


def profile_one(args):
    import torch
    trial = args.output_root/'trials'/f'screen_{args.profile_one}_s{SEED}'
    signature = digest(source_binding())
    if (trial/'profile.json').exists():
        valid_profile(read(trial/'profile.json'), args.profile_one, signature)
        return
    trial.mkdir(parents=True, exist_ok=True)
    with lock(trial/'trial.lock', blocking=False):
        write(trial/'status.json', dict(status='profiling',step=0,total_steps=STEPS,pid=os.getpid(),updated=stamp()))
        train = trainer(); train.setup(SEED)
        torch.cuda.init()
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one CUDA device is required.')
        test = torch.ones(1,device='cuda:0')+1; torch.cuda.synchronize()
        if test.item() != 2: raise RuntimeError('Real CUDA kernel qualification failed.')
        write(trial/'cuda_device.json', dict(updated=stamp(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
              cuda_uuid=str(torch.cuda.get_device_properties(0).uuid),device=torch.cuda.get_device_name()))
        settings = read(ROOT/'experiments/arch70m/base.json')
        data = train.IndexedPreparedDataset(args.output_root/'data/index.json')
        collate = train.VFamilyCollator(train.FixedChainConfigProxy(settings),family_spec=settings['family_prediction'])
        clean = collate([data[i] for i in range(min(GLOBAL,len(data)))])
        model, optimizer = train.construct(settings,args.profile_one,SEED,torch.device('cuda:0'))
        scheduler = train.make_scheduler(optimizer,100,5)
        batch = train.profile_batch(clean,MICRO,settings['model']['max_length'],model.tokenizer)
        batch = train.corrupt(batch,model.tokenizer,torch.Generator().manual_seed(9100))
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); start = time.monotonic()
        updates = []
        for _ in range(3):
            value = train.update(model,batch,optimizer,scheduler,MICRO,torch.device('cuda:0'))
            if not all(math.isfinite(value[k]) for k in ('loss','family_loss','grad_norm')):
                raise RuntimeError('Non-finite qualification update.')
            updates.append(value)
        torch.cuda.synchronize()
        allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        free,total = torch.cuda.mem_get_info(); external = max(0,total-free-torch.cuda.memory_reserved())
        candidate = dict(status='passed',updates=3,micro_batch=MICRO,allocated_bytes=allocated,
            reserved_bytes=reserved,external_bytes=external,total_bytes=total,
            safe=reserved+external <= .85*total,seconds_per_step=(time.monotonic()-start)/3)
        profile = dict(variant=args.profile_one,micro_batch=MICRO,global_batch=GLOBAL,seed=SEED,
            grad_accum=16,final_micro_batch=32,length=2048,candidates=[candidate],updates=updates,
            source_sha256=signature,data_sha256=DATA_SHA,device=torch.cuda.get_device_name(),
            note='Maximum-length 3-update qualification; fresh process for formal training.')
        write(trial/'profile_attempts.json',profile)
        valid_profile(profile,args.profile_one,signature)
        write(trial/'profile.json',profile)
        write(trial/'status.json',dict(status='qualified',step=0,total_steps=STEPS,updated=stamp()))
        print(dict(event='profile_passed',variant=args.profile_one,**candidate),flush=True)


def publish(out,message,push=True):
    from .report import report
    from arch70m_seed_validation import push_results
    def git(*args): return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    with lock(ROOT/'output/arch70m_publish.lock'):
        if git('branch','--show-current') != 'explore/arch70m-v8-20261028': raise ValueError('Wrong branch.')
        if git('diff','--cached','--name-only'): raise ValueError('Unrelated staged changes.')
        done = report(out)
        p=ROOT/'PROJECT_PROCESS.md'
        p.write_text(f'> Last updated: {stamp()} (UTC)。层次注意力micro32探索 {done}/4 完成；{message}。仅seed43/1epoch/global512，四组共同micro32，原V8固定10%数据；报告 {ROOT}/docs/arch70m_results/hierarchy_m32/REPORT.md。\n\n'+p.read_text())
        git('add','--','PROJECT_PROCESS.md','docs/arch70m_results/hierarchy_m32');git('diff','--cached','--check')
        git('-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-m','arch70m hierarchy m32: '+message)
        result = push_results() if push else dict(status='disabled')
        write(out/'publication.json',dict(result,commit=git('rev-parse','HEAD'),updated=stamp()))
        return result


def child(args,source,variant,mode):
    trial=args.output_root/'trials'/f'screen_{variant}_s{SEED}';trial.mkdir(parents=True,exist_ok=True)
    env={k:v for k,v in os.environ.items() if k not in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE',
        'GROUP_RANK','MASTER_ADDR','MASTER_PORT','ARCH70M_FROZEN_SOURCE','CUDA_LAUNCH_BLOCKING')
        and not k.startswith(('NCCL_','TORCHELASTIC_'))}
    env.update(CUDA_VISIBLE_DEVICES=args.gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONPATH=str(source),USE_TF='0',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,'-B','-m','architecture_hierarchy.runner',mode,variant,'--output-root',str(args.output_root)]
    log_path=trial/('qualification.log' if mode=='--profile-one' else 'console.log')
    with log_path.open('a') as log:
        start=log.tell();process=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
        write(trial/('qualification_process.json' if mode=='--profile-one' else 'process.json'),
            dict(pid=process.pid,command=command,started=stamp(),gpu=args.gpu,log_start=start))
        print(dict(event='launch',mode=mode,variant=variant,pid=process.pid),flush=True)
        code=process.wait()
    if code:
        with log_path.open('rb') as log:
            log.seek(0,2);size=log.tell();log.seek(max(start,size-4000));error=log.read().decode(errors='replace')
        status=read(trial/'status.json') if (trial/'status.json').exists() else dict(step=0)
        write(trial/'status.json',dict(status,status='failed',error=error,exit_code=code,failed_mode=mode,updated=stamp()))
        print(dict(event='child_failed',mode=mode,variant=variant,exit_code=code,error=error[-700:]),flush=True)
    return code


def execute(args):
    if not args.gpu.isdecimal(): raise ValueError('Use a numeric CUDA GPU selector.')
    if args.dry_run:
        print(dict(variants=VARIANTS,gpu=args.gpu,micro_batch=MICRO,grad_accum=16,global_batch=GLOBAL,
              seed=SEED,epochs=1,output_root=str(args.output_root)));return
    prepare(args.output_root)
    from experiments.arch70m.suite import verify_data
    verify_data(args.output_root)
    with lock(ROOT/'output'/f'arch70m_local_gpu{args.gpu}.lock',blocking=False), lock(args.output_root/'controller.lock',blocking=False):
        source,signature=snapshot(args.output_root)
        execution=dict(started=stamp(),controller_pid=os.getpid(),variants=list(VARIANTS),micro_batch=MICRO,
            global_batch=GLOBAL,grad_accum=16,seed=SEED,epochs=1,source_sha256=signature,
            previous_source_sha256=PREVIOUS_SHA,data_sha256=DATA_SHA,gpu=args.gpu,phase='qualification')
        write(args.output_root/'execution.json',execution)
        for variant in VARIANTS:
            trial=args.output_root/'trials'/f'screen_{variant}_s{SEED}'
            if (trial/'profile.json').exists():valid_profile(read(trial/'profile.json'),variant,signature)
            elif child(args,source,variant,'--profile-one'):
                try:publish(args.output_root,variant+' qualification failed',not args.no_push)
                except Exception:write(args.output_root/'publication_error.json',dict(error=traceback.format_exc(),updated=stamp()))
                raise RuntimeError('All four variants must qualify before formal training: '+variant)
        profiles={v:sha(args.output_root/'trials'/f'screen_{v}_s{SEED}'/'profile.json') for v in VARIANTS}
        write(args.output_root/'qualification.json',dict(status='passed',updated=stamp(),profiles=profiles,
            micro_batch=MICRO,source_sha256=signature,data_sha256=DATA_SHA))
        write(args.output_root/'execution.json',dict(execution,phase='training',qualified=stamp()))
        print(dict(event='all_four_qualified',source_sha256=signature),flush=True)
        failures=[]
        for variant in VARIANTS:
            if completed(args.output_root,variant,signature):continue
            code=child(args,source,variant,'--train-one')
            if code==0 and not completed(args.output_root,variant,signature):raise RuntimeError('Missing completed receipt: '+variant)
            if code:failures.append(variant)
            try:publication=publish(args.output_root,variant+(' failed' if code else ' completed'),not args.no_push)
            except Exception:
                publication=dict(status='local_report_failed')
                write(args.output_root/'publication_error.json',dict(error=traceback.format_exc(),updated=stamp()))
            print(dict(event='finished',variant=variant,training_exit_code=code,publication_status=publication['status']),flush=True)
        write(args.output_root/'execution.json',dict(execution,phase='finished',finished=stamp(),failed_variants=failures))
        if failures:raise RuntimeError('Training failures: '+', '.join(failures))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',default='0');p.add_argument('--output-root',type=Path,default=DEFAULT_OUT)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--no-push',action='store_true')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--profile-one',choices=VARIANTS);mode.add_argument('--train-one',choices=VARIANTS)
    args=p.parse_args();args.output_root=args.output_root.resolve()
    if args.profile_one:profile_one(args)
    elif args.train_one:
        signature=digest(source_binding());valid_qualification(args.output_root,signature)
        trainer().run(argparse.Namespace(output_root=str(args.output_root),stage='screen',variant=args.train_one,seed=SEED,wandb=False))
    else:execute(args)


if __name__=='__main__':main()
