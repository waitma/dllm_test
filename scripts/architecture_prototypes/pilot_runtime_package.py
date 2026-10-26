"""Undeployed source-package builder; never a training or GPU acceptance entry.

Activate pllm. CPU tests:
  python -m pytest scripts/tests/immune_llada/test_pilot_runtime_package.py -q
API: plan_runtime(base_snapshot, destination, additions={}), build_runtime(...),
verify_runtime(directory). A missing completed R20 archive returns a blocked plan.
Only an immutable parent plus fixed NEW script names can populate a runtime.
No CLI, model construction, data loading, submission or GPU operation is provided.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from scripts.architecture_prototypes import archive_refined_release as archive

SCHEMA = 'immune.pilot_runtime_source_package.v1'
NAME = 'r20_r27_pilot_v3_20261001'
MANIFEST = 'pilot_runtime_manifest.json'
INSPECTOR = 'scripts/architecture_prototypes/pilot_runtime_inspect.py'
REHEARSAL = 'SOURCE_COPY_REHEARSAL_ONLY.json'
ADDITIONS = frozenset({
    'scripts/architecture_prototypes/pilot_runtime_package.py', INSPECTOR,
    'scripts/architecture_prototypes/pilot_runtime_entry_v1.py',
    'scripts/architecture_prototypes/pilot_gpu_release_producer_v1.py',
})
RESOURCE_ALIASES = (
    'downstream/benchmark/data',
    'output/research/uptcr_20260930/released_tables/data/finetune/abpm',
)
require, sha, read_json = archive.require, archive.sha, archive.read_json


def aliases_for(original_root):
    return {name:str(Path(original_root)/name) for name in RESOURCE_ALIASES}


def verify_base(base_snapshot):
    base_snapshot=Path(base_snapshot).absolute()
    require(base_snapshot==base_snapshot.resolve() and base_snapshot.name==archive.NAME,
            'Use the canonical completed R20 snapshot')
    manifest=archive.verify_snapshot(base_snapshot)
    require(manifest.get('family')==list(archive.FAMILY) and
            set(manifest.get('terminal_job_ids',{}))==set(archive.FAMILY),
            'Parent archive lacks all eight terminal diagnostic identities')
    evidence=manifest.get('archive_evidence_sha256')
    require(type(evidence) is dict and evidence, 'Parent archive lacks original evidence bindings')
    for name,value in evidence.items():
        path=Path(name)
        require(path.is_absolute() and path.is_file() and sha(path)==value,
                'Parent archive evidence is missing or changed: '+name)
    require(archive.ENTRY in manifest['measured_source_sha256'], 'Parent refined entry is not profile-bound')
    return manifest


def plan_runtime(base_snapshot, destination, *, additions=None):
    base_snapshot,destination=Path(base_snapshot).absolute(),Path(destination).absolute()
    result=dict(schema=SCHEMA,ready=False,blockers=[],base_snapshot=str(base_snapshot),runtime_root=str(destination),
                gpu_accepted=False,data_accepted=False,training_started=False)
    require(base_snapshot==base_snapshot.resolve() and destination==destination.resolve(), 'Canonical paths required')
    if not base_snapshot.is_dir():
        result['blockers'].append('Completed R20 archive does not exist; no runtime directory may be created')
        return result
    parent=verify_base(base_snapshot)
    original_root=Path(parent['source_root'])
    require(original_root==original_root.resolve(), 'Parent source_root is noncanonical')
    require(destination==original_root/'output/immune_pilot_runtimes'/NAME, 'Use the fixed pilot runtime destination')
    if destination.exists():
        result['blockers'].append('Runtime destination exists; never overwrite an immutable package')
    base_files=parent['copied_files_sha256']
    added={}
    for name,path in (additions or {}).items():
        require(name in ADDITIONS, 'Addition name is outside the fixed whitelist')
        require(name not in base_files, 'Additions must never replace a parent file, including adapters or measured sources')
        path=Path(path).absolute()
        require(path==original_root/name and path==path.resolve() and path.is_file(),
                'Addition must be its explicit canonical original-root script')
        added[name]=sha(path)
    files={**base_files,**added}
    require(INSPECTOR in files and 'scripts/architecture_prototypes/pilot_runtime_package.py' in files,
            'Runtime requires the fixed package verifier and inspection script')
    aliases=aliases_for(original_root)
    for name,target in aliases.items():
        require(not any(Path(file).is_relative_to(name) or Path(name).is_relative_to(file) for file in files),
                'Resource alias overlaps copied source')
        if not Path(target).is_dir():
            result['blockers'].append('Missing required external resource directory: '+target)
    result.update(ready=not result['blockers'],source_root=str(original_root),
        base_manifest_sha256=sha(base_snapshot/'release_manifest.json'),base_copied_files_sha256=base_files,
        measured_source_sha256=parent['measured_source_sha256'],additions_sha256=added,
        runtime_files_sha256=files,external_resource_aliases=aliases,
        inherited_terminal_job_ids=parent['terminal_job_ids'])
    return result


def verify_runtime(directory, *, expected_destination=None):
    directory=Path(directory).absolute()
    require(directory==directory.resolve() and directory.is_dir(), 'Runtime directory must be canonical')
    manifest=read_json(directory/MANIFEST)
    require(manifest.get('schema')==SCHEMA and manifest.get('runtime_root')==str(expected_destination or directory)
            and manifest.get('status')=='source_package_only_not_gpu_or_data_acceptance'
            and all(manifest.get(key) is False for key in ('gpu_accepted','data_accepted','training_started')),
            'Runtime identity or source-only acceptance status differs')
    base=Path(manifest['base_snapshot']);parent=verify_base(base)
    require(sha(base/'release_manifest.json')==manifest.get('base_manifest_sha256'), 'Parent archive manifest changed')
    require(manifest.get('source_root')==parent['source_root'] and
            manifest.get('base_copied_files_sha256')==parent['copied_files_sha256'] and
            manifest.get('measured_source_sha256')==parent['measured_source_sha256'] and
            manifest.get('inherited_terminal_job_ids')==parent['terminal_job_ids'], 'Parent source/proof contract differs')
    added=manifest.get('additions_sha256',{})
    require(type(added) is dict and set(added)<=ADDITIONS and not set(added)&set(parent['copied_files_sha256']),
            'Invalid or replacing addition')
    files={**parent['copied_files_sha256'],**added}
    require(manifest.get('runtime_files_sha256')==files and INSPECTOR in files,
            'Runtime full source map differs')
    for name,value in files.items():
        require(sha(archive.relative_file(directory,name))==value, 'Runtime source SHA differs: '+name)
    actual={str(path.relative_to(directory)) for package in ('dllm','examples','scripts')
            for path in (directory/package).rglob('*') if path.is_file() and path.suffix in archive.SUFFIXES}
    require(actual==set(files), 'Runtime contains unrecorded or missing source/config files')
    aliases=aliases_for(parent['source_root'])
    require(manifest.get('external_resource_aliases')==aliases, 'Only the two exact original-root resource aliases are allowed')
    symlinks={str((Path(folder)/name).relative_to(directory))
              for folder,dirs,names in os.walk(directory,followlinks=False)
              for name in dirs+names if (Path(folder)/name).is_symlink()}
    require(symlinks==set(aliases), 'Runtime contains extra or missing resource/source aliases')
    for name,target in aliases.items():
        path=directory/name
        require(path.is_symlink() and os.readlink(path)==target and path.resolve()==Path(target).resolve()
                and path.is_dir(), 'Resource alias is missing, redirected or its target is unavailable')
    return manifest


def build_runtime(base_snapshot, destination, *, additions=None):
    plan=plan_runtime(base_snapshot,destination,additions=additions)
    require(plan['ready'], 'Runtime package blocked: '+'; '.join(plan['blockers']))
    destination=Path(plan['runtime_root']);base=Path(plan['base_snapshot'])
    destination.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.'+NAME+'.staging-',dir=destination.parent))
    try:
        for name,value in plan['runtime_files_sha256'].items():
            source=(base if name in plan['base_copied_files_sha256'] else Path(plan['source_root']))/name
            target=staging/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
            require(sha(target)==value, 'Source changed while copying runtime: '+name)
        for name,target in plan['external_resource_aliases'].items():
            alias=staging/name;alias.parent.mkdir(parents=True,exist_ok=True);alias.symlink_to(target,target_is_directory=True)
        manifest={key:value for key,value in plan.items() if key not in ('ready','blockers')}
        manifest.update(status='source_package_only_not_gpu_or_data_acceptance',
                        created_at=datetime.now(timezone.utc).isoformat())
        (staging/MANIFEST).write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
        verify_runtime(staging,expected_destination=destination)
        require(plan_runtime(base,destination,additions=additions)==plan, 'Parent/addition/resource changed before publication')
        require(not destination.exists(), 'Runtime appeared; never overwrite')
        staging.rename(destination)
        return verify_runtime(destination)
    finally:
        if staging.exists():shutil.rmtree(staging)


def inspect_fresh(directory, pool, *, mode='runtime', python=sys.executable):
    """Fixed CPU inspection only; never accepts a command, module or training mode."""
    require(mode in ('runtime','source-copy-rehearsal'), 'Unknown inspection mode')
    allowed={Path(sys.executable).resolve(),
        Path('/project_c20250601/251105016/conda/envs/pllm/bin/python').resolve(),
        Path('/project_c20250601/251105016/conda/envs/protenix_abtcr/bin/python').resolve()}
    require(Path(python).is_absolute() and Path(python).resolve() in allowed,
            'Inspection accepts only the current or two explicitly installed Python interpreters')
    directory,pool=Path(directory).resolve(),Path(pool).resolve()
    entry=archive.relative_file(directory,INSPECTOR)
    if mode=='runtime':verify_runtime(directory)
    result=subprocess.run([str(python),'-I',str(entry),'--inspection-mode',mode,'--pool-dir',str(pool)],
        cwd=directory,env={**os.environ,'PYTHONPATH':str(directory),'PYTHONDONTWRITEBYTECODE':'1',
                          'USE_TF':'0','CUDA_VISIBLE_DEVICES':''},capture_output=True,text=True,timeout=60)
    require(result.returncode==0,'CPU runtime inspection failed: '+result.stderr[-6000:])
    return json.loads(result.stdout)
