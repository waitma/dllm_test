"""Frozen confirmation bindings; run python -B -m architecture_confirm.tests."""
from pathlib import Path
import math
import shutil
import uuid
from architecture_hierarchy.common import source_binding as previous_sources,prepare as previous_prepare
from experiments.arch70m.common import read,write,sha,digest,stamp,lock,commit_id

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_confirm_20261031')
PREVIOUS=Path('/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_hierarchy_m32_20261029')
PREVIOUS_SHA='eb2fda14bc98a3386637d7b590233538464f6b6e78125d002ea2e14cc4e8f443'
DATA_SHA='4beaac6b54a1f46492533aeb77c2de2c6ac2a0bd8266fd57a6bb312679435c51'
VARIANTS=('A0','B0','B1');SEEDS=(42,43,44);MICRO=32;GLOBAL=512;STEPS=3828;EPOCHS=3
JOBS=tuple((v,s) for s in SEEDS for v in VARIANTS)


def trial(out,variant,seed):return Path(out)/'trials'/f'confirm_{variant}_s{seed}'


def source_binding():
    sources=previous_sources()
    if digest(sources)!=PREVIOUS_SHA:raise ValueError('Historical frozen sources changed.')
    paths=list((ROOT/'architecture_confirm').glob('*.py'))+[ROOT/'architecture_confirm/protocol.json',ROOT/'launch_arch70m_confirm.sh']
    return dict(sources,**{str(p.relative_to(ROOT)):sha(p) for p in paths})


def prepare(out):
    out=Path(out).resolve()
    protected=[ROOT/'output'/n for n in ('arch70m_v8_20261028','arch70m_round2_20261029','arch70m_hierarchy_m32_20261029')]
    if any(out==p or p in out.parents for p in protected):raise ValueError('Original experiment outputs are read-only.')
    settings=previous_prepare(out);signature=digest(source_binding());protocol=read(ROOT/'architecture_confirm/protocol.json')
    expected=dict(variants=list(VARIANTS),seeds=list(SEEDS),epochs=EPOCHS,steps=STEPS,micro_batch=MICRO,global_batch=GLOBAL)
    if any(protocol.get(k)!=v for k,v in expected.items()):raise ValueError('Confirmation protocol differs.')
    for variant,receipt in protocol['screen_evidence'].items():
        path=Path(receipt['metrics_path'])
        if sha(path)!=receipt['sha256']:raise ValueError('Candidate selection evidence changed: '+variant)
    choice=dict(confirm_variants=list(VARIANTS),seeds=list(SEEDS),epochs=EPOCHS,micro_batch=MICRO,
        source_sha256=signature,data_sha256=DATA_SHA,previous_source_sha256=PREVIOUS_SHA,
        screen_evidence=protocol['screen_evidence'],decision_protocol=protocol['decision'])
    path=out/'selection.json'
    if path.exists() and read(path)!=choice:raise ValueError('Frozen candidate selection differs.')
    if not path.exists():write(path,choice)
    return settings


def snapshot(out):
    sources=source_binding();signature=digest(sources);target=Path(out)/'sources'/signature
    with lock(Path(out)/'sources.lock'):
        if not target.exists():
            pending=target.parent/('.pending_'+uuid.uuid4().hex);pending.mkdir(parents=True)
            for name in sources:
                dest=pending/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
            write(pending/'provenance.json',dict(created=stamp(),source_sha256=signature,commit=commit_id(),files=sources))
            pending.rename(target)
        if any(sha(target/n)!=h for n,h in sources.items()):raise ValueError('Frozen confirmation snapshot changed.')
    return target,signature


def valid_profile(profile,variant,seed,signature):
    expected=dict(variant=variant,seed=seed,micro_batch=MICRO,global_batch=GLOBAL,length=2048,
                  grad_accum=16,final_micro_batch=32,source_sha256=signature,data_sha256=DATA_SHA)
    if any(profile.get(k)!=v for k,v in expected.items()):raise ValueError('Profile binding differs.')
    candidates=profile.get('candidates',[])
    if len(candidates)!=1:raise ValueError('Missing common-micro GPU qualification.')
    p=candidates[0]
    if p.get('status')!='passed' or p.get('updates')!=3 or not p.get('safe') or p['reserved_bytes']+p['external_bytes']>.85*p['total_bytes']:
        raise ValueError('Unsafe or incomplete GPU qualification.')


def qualified(out,signature):
    q=read(Path(out)/'qualification.json');keys={f'{v}_s{s}' for v,s in JOBS}
    if q.get('status')!='passed' or q.get('source_sha256')!=signature or set(q.get('profiles',{}))!=keys:
        raise ValueError('All nine seed/architecture qualifications are required.')
    for v,s in JOBS:
        path=trial(out,v,s)/'profile.json'
        if sha(path)!=q['profiles'][f'{v}_s{s}']:raise ValueError('Qualified profile checksum changed.')
        valid_profile(read(path),v,s,signature)


def completed(out,variant,seed,signature):
    t=trial(out,variant,seed)
    if not (t/'status.json').exists() or read(t/'status.json')['status']!='completed':return False
    b=read(t/'binding.json');expected=dict(variant=variant,seed=seed,epochs=EPOCHS,total_steps=STEPS,
        micro_batch=MICRO,source_sha256=signature,data_sha256=DATA_SHA)
    if any(b.get(k)!=v for k,v in expected.items()) or read(t/'status.json')['step']!=STEPS:raise ValueError('Completed binding differs.')
    if b['settings']!=read(ROOT/'experiments/arch70m/base.json') or b['profile_sha256']!=sha(t/'profile.json'):
        raise ValueError('Completed settings/profile differs.')
    valid_profile(read(t/'profile.json'),variant,seed,signature)
    cp=t/f'checkpoints/step_{STEPS:06d}';r=read(cp/'receipt.json')
    if r['status']!='complete' or len(r['files'])!=6 or any(sha(cp/n)!=h for n,h in r['files'].items()):
        raise ValueError('Incomplete or corrupt final checkpoint.')
    state=read(cp/'state.json')
    if (state['step'],state['epoch'],state['next_batch'],state['seen_records'])!=(STEPS,EPOCHS,0,1959864):
        raise ValueError('Final epoch/data cursor differs.')
    if read(cp/'binding.json')!=b:raise ValueError('Checkpoint binding differs.')
    m=read(t/f'metrics/confirm_{STEPS:06d}.json');obs=read(t/f'metrics/confirm_{STEPS:06d}_observations.json')
    if m['step']!=STEPS or m['partition']!='confirm' or not obs['generation'] or not obs['representation']:
        raise ValueError('Missing final heldout observations.')
    if not all(math.isfinite(x) for x in [m['token']['all']['loss'],m['generation']['conditional_ce'],m['representation']['macro_auroc']]):
        raise ValueError('Non-finite final heldout metrics.')
    read(t/'ab_specificity/metrics.json')
    return True
