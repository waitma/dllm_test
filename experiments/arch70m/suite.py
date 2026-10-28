"""Independent GPU scheduling, frozen candidates and publication; run via run.sh."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import traceback
import uuid
from .common import ROOT, DEFAULT_OUT, DEFAULT_DATA, VARIANTS, read, write, sha, digest, source_binding, stamp, lock, commit_id

BRANCH = 'explore/arch70m-v8-20261028'

def assigned_trials(trials, machine_index, machine_count):
    if machine_count<1 or not 0<=machine_index<machine_count: raise ValueError('Invalid machine index/count')
    return [t for i,t in enumerate(trials) if i%machine_count==machine_index]

def snapshot(root):
    """Workers import a content-addressed copy, unaffected by subsequent worktree edits."""
    sources=source_binding(); signature=digest(sources)
    target=root/'sources'/signature
    with lock(root/'sources.lock'):
        if not target.exists():
            temporary=target.parent/('.pending_'+uuid.uuid4().hex); temporary.mkdir(parents=True)
            for name in sources:
                dest=temporary/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/name,dest)
            write(temporary/'provenance.json',dict(source_sha256=signature,files=sources,commit=commit_id(),created=stamp()))
            temporary.rename(target)
        if any(sha(target/n)!=h for n,h in sources.items()): raise ValueError('Source snapshot changed')
    return target, signature

def verify_data(root):
    """Rehash changed files; retain a stat-bound receipt to avoid repeating full scans."""
    index=read(root/'data/index.json'); refs=[]
    for shard in index['shards']:
        refs.extend([(shard['path'],shard['sha256']),(shard['index'],shard['index_sha256'])])
    for part in index['files'].values(): refs.extend((v['path'],v['sha256']) for v in part.values())
    with lock(root/'data/verify.lock'):
        receipt_path=root/'data/verified.json'; old=read(receipt_path) if receipt_path.exists() else {}; current={}
        for name,expected in refs:
            st=Path(name).stat(); mark=[st.st_size,st.st_mtime_ns,st.st_ctime_ns,expected]
            if old.get(name)!=mark and sha(name)!=expected: raise ValueError('Data checksum failed: '+name)
            current[name]=mark
        write(receipt_path,current)

def select(root):
    with lock(root/'selection.lock'):
        if (root/'selection.json').exists(): return read(root/'selection.json')
        values={}; bindings=[]; evidence={}
        for variant in VARIANTS:
            trial=root/'trials'/f'screen_{variant}_s42'
            if not (trial/'status.json').exists() or read(trial/'status.json')['status']!='completed':
                raise ValueError('Selection waits for all eight completed screening trials: '+variant)
            status=read(trial/'status.json'); path=trial/'metrics'/f"screen_{status['step']:06d}.json"
            values[variant]=read(path); bindings.append(read(trial/'binding.json')); evidence[variant]=sha(path)
        for b in bindings[1:]:
            for key in ('settings','source_sha256','data_sha256','seed','epochs','total_steps'):
                if b[key]!=bindings[0][key]: raise ValueError('Screening trials do not share '+key)
        base=values['A0']['generation']; scores={}
        for v in VARIANTS[1:]:
            g=values[v]['generation']; ce_keys=('conditional_ce','cdr_ce')
            if not all(k in g and k in base for k in ce_keys): raise ValueError('Generation task coverage incomplete')
            improvement=sum((base[k]-g[k])/max(base[k],1e-12) for k in ce_keys)/2
            length_ok=(g.get('conditional_length_mae') is None and base.get('conditional_length_mae') is None) or (
                g.get('conditional_length_mae') is not None and base.get('conditional_length_mae') is not None and
                g['conditional_length_mae']<=1.1*base['conditional_length_mae'])
            guarded=g.get('conditional_validity',0)>=base.get('conditional_validity',0)-.02 and length_ok
            scores[v]=dict(generation_improvement=improvement,generation_guard=guarded,
                representation_auroc=values[v]['representation']['macro_auroc'])
        g_order=sorted(scores,key=lambda v:(-scores[v]['generation_improvement'],v))
        r_order=sorted(scores,key=lambda v:(-scores[v]['representation_auroc'],v))
        eligible=[v for v in g_order if scores[v]['generation_guard']]
        g=(eligible or g_order)[0]; r=r_order[0]
        if r==g:
            r=min((v for v in scores if v!=g),key=lambda v:(g_order.index(v)+r_order.index(v),v))
        result=dict(created=stamp(),confirm_variants=['A0',g,r],seeds=[42,43,44],epochs=3,
            source_sha256=bindings[0]['source_sha256'],data_sha256=bindings[0]['data_sha256'],
            screening_evidence=evidence,scores=scores,generation_candidate=g,representation_candidate=r,
            generation_guard_fallback=not bool(eligible),selection_partition='screen',
            rule='Final epoch only; generation mean relative CE improvement with validity/length guards; representation AUROC; deduplicate by mean rank. Candidates are hypotheses, not confirmed wins.')
        write(root/'selection.json',result); return result

def publish(root, message, push=True):
    from .report import report
    with lock(ROOT/'output/arch70m_publish.lock'):
        report(root)
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
        if branch!=BRANCH: raise ValueError('Refusing to commit outside exploration branch: '+branch)
        paths=['docs/ARCH70M_EXPLORATION.md','docs/arch70m_results']
        subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
        # Never accidentally include unrelated staged changes in an automatic result commit.
        staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
        if any(p!=paths[0] and not p.startswith(paths[1]+'/') for p in staged): raise ValueError('Unrelated staged files; publication stopped')
        if staged:
            subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@localhost','commit','-m',message],cwd=ROOT,check=True)
        if push:
            # Honor the job's HTTP proxy instead of an unrelated global socks proxy.
            env=os.environ.copy(); proxy=env.get('https_proxy') or env.get('HTTPS_PROXY')
            if proxy:
                n=int(env.get('GIT_CONFIG_COUNT','0')); env['GIT_CONFIG_COUNT']=str(n+1)
                env[f'GIT_CONFIG_KEY_{n}']='http.proxy'; env[f'GIT_CONFIG_VALUE_{n}']=proxy
            subprocess.run(['git','push','origin','HEAD:refs/heads/'+BRANCH],cwd=ROOT,env=env,check=True)

def execute(args):
    root=Path(args.output_root).resolve(); root.mkdir(parents=True,exist_ok=True)
    if args.stage=='report':
        from .report import report
        report(root); return
    if args.stage=='select': print(json.dumps(select(root),indent=2)); return
    from .data import prepare
    with lock(root/'data_prepare.lock'): prepare(root,Path(args.data_root).resolve())
    if args.stage=='prepare': return
    verify_data(root); source,signature=snapshot(root)
    stages=('screen','confirm') if args.stage=='all' else (args.stage,)
    for stage in stages:
        if stage=='screen':
            names=args.trials.split(',') if args.trials else list(VARIANTS)
            if len(names)!=len(set(names)) or not set(names)<=set(VARIANTS): raise ValueError('Invalid/duplicate trial names')
            trials=[(stage,v,42) for v in names]
        else:
            choice=select(root)
            if choice['source_sha256']!=signature: raise ValueError('Confirmation must use the screened source snapshot')
            trials=[(stage,v,s) for v in choice['confirm_variants'] for s in choice['seeds']]
        trials=assigned_trials(trials,args.machine_index,args.machine_count)
        devices=args.gpus.split(',')
        if len(devices)!=len(set(devices)) or not all(devices): raise ValueError('GPU IDs must be unique')
        jobs=queue.Queue()
        for t in trials: jobs.put(t)
        def worker(gpu):
            failures=[]
            while True:
                try: stage,variant,seed=jobs.get_nowait()
                except queue.Empty: return failures
                name=f'{stage}_{variant}_s{seed}'; out=root/'trials'/name; out.mkdir(parents=True,exist_ok=True)
                env={k:v for k,v in os.environ.items() if not k.startswith(('NCCL_','TORCHELASTIC_')) and k not in
                     ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK','MASTER_ADDR','MASTER_PORT')}
                env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONPATH=str(source),CUBLAS_WORKSPACE_CONFIG=':4096:8',
                           USE_TF='0',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
                env.pop('CUDA_LAUNCH_BLOCKING',None)
                cmd=[sys.executable,'-B','-m','experiments.arch70m.train','--stage',stage,'--variant',variant,
                     '--seed',str(seed),'--output-root',str(root)]+(['--wandb'] if args.wandb else [])
                print(json.dumps(dict(stage='launch',trial=name,gpu=gpu,source=str(source))),flush=True)
                with (out/'console.log').open('a') as log:
                    process=subprocess.Popen(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
                    write(out/'process.json',dict(pid=process.pid,gpu=gpu,command=cmd,started=stamp()))
                    code=process.wait()
                status=read(out/'status.json') if (out/'status.json').exists() else {}
                if code and status.get('status')!='failed':
                    write(out/'status.json',dict(status='failed',exit_code=code,updated=stamp(),error='See console.log',step=status.get('step',0)))
                if code: failures.append(name)
                print(json.dumps(dict(stage='finished',trial=name,exit_code=code)),flush=True)
                try: publish(root,f'arch70m: record {name} '+('completed' if not code else 'failed'),not args.no_push)
                except Exception:
                    write(out/'publication_error.json',dict(error=traceback.format_exc(),updated=stamp()))
                    failures.append(name+':publication')
            return failures
        failures=[]
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            for f in as_completed([pool.submit(worker,d) for d in devices]): failures.extend(f.result())
        if failures: raise RuntimeError('Trials/publication failed: '+', '.join(failures))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('prepare','screen','select','confirm','all','report'),required=True)
    p.add_argument('--gpus',default='0'); p.add_argument('--trials')
    p.add_argument('--machine-index',type=int,default=0); p.add_argument('--machine-count',type=int,default=1)
    p.add_argument('--output-root',default=str(DEFAULT_OUT)); p.add_argument('--data-root',default=str(DEFAULT_DATA))
    p.add_argument('--wandb',action='store_true'); p.add_argument('--no-push',action='store_true',help='Keep result commits local')
    execute(p.parse_args())

if __name__=='__main__': main()
