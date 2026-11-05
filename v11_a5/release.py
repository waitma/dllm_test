"""Build or verify an immutable V11 release; use launch_v11_a5_75m.sh.

Run: python -B -m v11_a5.release --run-root ABS [--config ABS].
Only new release paths are written. Parent source/data/checkpoints remain read-only.
"""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import uuid

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/models/v11_a5_75m_20261105/training_config.json'
SIZE_KEYS={'d_model','n_layers','n_heads','mlp_hidden_size'}
PLAN_SCHEMA='immune.v11.a5.training.v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())

def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def validate_config(config, reference):
    if (config['schema']!='immune.v11.a5.release.v1' or config['version']!='V11'
            or config['architecture']!='a5_all_global' or config['world_size']!=8
            or config['expected_parameters']!=75741903):raise ValueError('Unknown V11 release')
    for name in ['training','objective','recipe','family_prediction','initialization']:
        if config[name]!=reference[name]:raise ValueError('V11 differs from 1B '+name)
    if {k:v for k,v in config['model'].items() if k not in SIZE_KEYS}!={
            k:v for k,v in reference['model'].items() if k not in SIZE_KEYS}:
        raise ValueError('V11 must preserve the complete V8 layout')
    if {k:config['model'][k] for k in SIZE_KEYS}!={'d_model':768,'n_layers':8,'n_heads':12,'mlp_hidden_size':3072}:
        raise ValueError('V11 must use the 75M A5 dimensions')
    s=config['training']
    if s['micro_batch']*s['grad_accum']*config['world_size']!=s['global_batch']:
        raise ValueError('Effective batch differs')


def rebind_data(reference, model, data_root, staging):
    """Change only model-binding metadata; retain every record/index offset/split."""
    if sha(reference['data_index'])!=reference['data_index_sha256']:raise ValueError('Parent data index changed')
    old=read(reference['data_index']);old_binding=read(old['binding'])
    if sha(old['binding'])!=old['binding_sha256']:raise ValueError('Parent binding changed')
    new=deepcopy(old);binding=deepcopy(old_binding);binding['model']=deepcopy(model)
    binding_path=Path(data_root)/'binding.json'
    write(Path(staging)/'binding.json',binding)
    new['binding']=str(binding_path);new['binding_sha256']=sha(Path(staging)/'binding.json')
    write(Path(staging)/'index.json',new)
    for name in old:
        if name not in ['binding','binding_sha256'] and new[name]!=old[name]:raise ValueError('Data changed: '+name)
    if new['records']!={'train':6533037,'valid':71422}:raise ValueError('V11 must use the full 1B dataset')
    return new


def checked_plan(bundle):
    bundle=Path(bundle).resolve();launch=read(bundle/'release.json')
    if sha(bundle/'plan.json')!=launch['plan_sha256']:raise ValueError('V11 plan changed')
    plan=read(bundle/'plan.json')
    if plan['schema']!=PLAN_SCHEMA or plan['architecture']!='a5_all_global' or plan['version']!='V11':
        raise ValueError('Not a V11 A5 plan')
    if Path(plan['source_root']).resolve()!=bundle/'source':raise ValueError('Frozen source path differs')
    if plan['training']!=plan['reference_1b']['training']:raise ValueError('Training settings differ from 1B')
    if plan['expected_parameters']!=75741903 or plan['execution']['world_size']!=8:raise ValueError('V11 size/world differs')
    for path,digest in {**plan['sources'],**plan['assets']}.items():
        if sha(path)!=digest:raise ValueError('Frozen V11 dependency changed: '+path)
    if sha(plan['data_index'])!=plan['data_index_sha256']:raise ValueError('V11 data index changed')
    return plan,launch['plan_sha256']


def prepare(run_root, config_path=CONFIG):
    run_root=Path(run_root).resolve();config_path=Path(config_path).resolve()
    run_root.mkdir(parents=True,exist_ok=True)
    with (run_root/'package.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (run_root/'bundle').exists():
            p,_=checked_plan(run_root/'bundle')
            if p['release_config_sha256']!=sha(config_path):raise ValueError('Run root belongs to another V11 config')
            return run_root/'bundle'
        if (run_root/'train').exists():raise ValueError('Refusing to package over an existing train directory')
        config=read(config_path);reference_path=Path(config['reference_1b_plan'])
        if sha(reference_path)!=config['reference_1b_plan_sha256']:raise ValueError('Reference 1B plan changed')
        reference=read(reference_path);validate_config(config,reference)
        tmp=run_root/('.bundle-'+uuid.uuid4().hex);tmp.mkdir()
        final=run_root/'bundle';sources={}
        try:
            parent_source=Path(reference['source_root'])
            for path,expected in reference['sources'].items():
                path=Path(path);relative=path.relative_to(parent_source)
                if sha(path)!=expected:raise ValueError('Reference source changed: '+str(path))
                dest=tmp/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
                sources[str(final/'source'/relative)]=expected
            for path in sorted((ROOT/'v11_a5').glob('*.py')):
                dest=tmp/'source/v11_a5'/path.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
                sources[str(final/'source/v11_a5'/path.name)]=sha(dest)
            data=rebind_data(reference,config['model'],final/'data',tmp/'data')
            library=tmp/'runtime/libnccl.so.2';library.parent.mkdir()
            if sha(config['nccl_library'])!=config['nccl_sha256']:raise ValueError('NCCL runtime changed')
            shutil.copyfile(config['nccl_library'],library)
            credential=read(reference['wandb']['login_receipt'])
            wandb={k:credential[k] for k in ['credential_file','username','entity','project']}
            wandb.update(auth='existing_verified_account',run_id='v11a5-75m',name=run_root.name)
            plan={k:deepcopy(reference[k]) for k in ['objective','recipe','training','budget','family_prediction',
                'data_dir','initialization','target_pretraining_step','environment','smoke_indices']}
            plan.update(schema=PLAN_SCHEMA,version='V11',architecture='a5_all_global',run_root=str(run_root),
                source_root=str(final/'source'),model=config['model'],expected_parameters=config['expected_parameters'],
                data_index=str(final/'data/index.json'),data_index_sha256=sha(tmp/'data/index.json'),wandb=wandb,
                sources=sources,assets={str(final/'runtime/libnccl.so.2'):sha(library),
                    str(final/'data/binding.json'):sha(tmp/'data/binding.json')},
                execution=dict(mode='direct_train',world_size=8,validation='normal token/probe validation retained'),
                release_config_sha256=sha(config_path),reference_1b=dict(plan=str(reference_path),
                    plan_sha256=config['reference_1b_plan_sha256'],training=reference['training'],
                    data_index=reference['data_index'],data_index_sha256=reference['data_index_sha256'],
                    data_records=data['records'],source_files_copied=len(reference['sources'])))
            write(tmp/'plan.json',plan)
            write(tmp/'release.json',dict(plan_sha256=sha(tmp/'plan.json'),version='V11',source_files=len(sources)))
            write(tmp/'training_config.json',config)
            tmp.rename(final)
        except BaseException:
            shutil.rmtree(tmp,ignore_errors=True);raise
        checked_plan(final)
        return final


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',required=True,type=Path)
    p.add_argument('--config',default=CONFIG,type=Path);args=p.parse_args()
    print(prepare(args.run_root,args.config))
