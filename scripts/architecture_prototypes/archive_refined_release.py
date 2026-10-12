"""Prepare a reviewable R20--R27 source archive, without routing or queue changes.

Activate pllm. Read-only report (also works while the eight jobs are running):
  USE_TF=0 python scripts/architecture_prototypes/archive_refined_release.py --pool-dir /absolute/pool
The optional --apply API/CLI refuses active jobs or incomplete original evidence.
It never submits jobs, trains, edits release_map, or reads biological datasets.
CPU tests: python -m pytest scripts/tests/immune_llada/test_archive_refined_release.py -q
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
NAME = 'r20_r27_20261001'
FAMILY = ('r20_refined_control', 'r21_refined_head_gate', 'r22_refined_fixed_gate',
    'r23_refined_layerscale', 'r24_refined_fixed_scale', 'r25_refined_late_fusion',
    'r26_refined_local_global', 'r27_refined_value_residual')
ENTRY = 'scripts/run_immune_refined.py'
REQUIRED = (ENTRY, 'scripts/run_immune_research.py', 'scripts/run_immune_architecture.py',
    'scripts/architecture_prototypes/expanded_fixtures.py',
    'dllm/pipelines/immune_llada/research_refined/model.py',
    'dllm/pipelines/llada/models/configuration_llada.py',
    'dllm/pipelines/immune_llada/v3/legacy_vocabulary.json')
SUFFIXES = {'.py', '.json', '.yaml', '.yml'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    result = json.loads(Path(path).read_text(), object_pairs_hook=pairs)
    json.dumps(result, allow_nan=False)
    require(type(result) is dict, 'Expected JSON object')
    return result


def relative_file(root, name):
    path = Path(name)
    require(isinstance(name, str) and not path.is_absolute() and '..' not in path.parts,
            'Source paths must be safe relative paths')
    result = root/path
    require(result.is_file() and not result.is_symlink() and result.resolve().is_relative_to(root),
            f'Missing, symlink or escaping source: {name}')
    return result


@dataclass(frozen=True)
class ArchiveRequest:
    source_root: Path
    pool: Path
    previous_snapshot: Path
    destination: Path


def canonical_request(request):
    values = [Path(value).absolute() for value in (request.source_root, request.pool,
              request.previous_snapshot, request.destination)]
    require(all(value == value.resolve() and not value.is_symlink() for value in values),
            'Use canonical nonsymlink absolute paths')
    result = ArchiveRequest(*values)
    require(result.destination == result.source_root/'output/immune_research_releases'/NAME,
            'Only the fixed R20--R27 snapshot destination is supported')
    require(result.previous_snapshot.name == 'r12_r19_20260930', 'Expected the prior R12--R19 snapshot')
    return result


def verify_snapshot(directory, *, expected_destination=None):
    directory = Path(directory).resolve()
    manifest = read_json(directory/'release_manifest.json')
    require(manifest.get('schema_version') == 1 and
            manifest.get('frozen_root') == str(expected_destination or directory), 'Snapshot schema or root differs')
    copied, measured = manifest.get('copied_files_sha256'), manifest.get('measured_source_sha256')
    require(type(copied) is dict and copied and type(measured) is dict and measured,
            'Snapshot lacks copied/measured source maps')
    require(all(copied.get(name) == value for name,value in measured.items()), 'Measured map not contained in archive')
    for name, value in copied.items():
        require(sha(relative_file(directory,name)) == value, f'Snapshot file SHA differs: {name}')
    return manifest


def import_inventory(source_root):
    """Fresh CPU-only import observation, supplemented by a full support-source tree."""
    code = '''import hashlib,importlib.metadata,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);sys.path.insert(0,str(root))
import torch
def forbidden(*a,**k):raise RuntimeError("Archive inspection must not initialize CUDA")
torch.cuda.init=forbidden;torch.cuda._lazy_init=forbidden
from scripts import run_immune_refined as runner
files={}
for module in list(sys.modules.values()):
 name=getattr(module,"__file__",None)
 if name:
  path=pathlib.Path(name).resolve()
  if path.is_file() and path.is_relative_to(root):files[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
versions={name:importlib.metadata.version(name) for name in ("torch","transformers","accelerate","safetensors","tokenizers")}
print(json.dumps(dict(imported_files_sha256=files,measured_source_sha256=runner.fingerprints(),external_versions=versions,scope="fresh runner import only; no model, forward, GPU or data loading"),sort_keys=True))
'''
    result = subprocess.run([sys.executable,'-c',code,str(source_root)], cwd=source_root,
        env={**os.environ,'USE_TF':'0','CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},
        capture_output=True,text=True,check=True,timeout=60)
    return json.loads(result.stdout)


def verify_original_profile(pool, variant):
    if str(ROOT) not in sys.path:
        sys.path.insert(0,str(ROOT))
    from scripts.enqueue_immune_refined import verify_profile_evidence
    return verify_profile_evidence(pool,variant)


def verify_original_diagnostic(pool, variant):
    if str(ROOT) not in sys.path:
        sys.path.insert(0,str(ROOT))
    from scripts.enqueue_immune_refined import verify_diagnostic
    # Archival preserves the experiment; a learning threshold is not archival certification.
    return verify_diagnostic(pool,variant,500,require_learning=False)


def source_inventory(root, previous):
    # Preserve the previous archive's support files and include all new Python
    # sources, including lazy imports. Runtime JSON configuration lives under dllm.
    names = set(previous['copied_files_sha256'])
    for package in ('dllm','examples','scripts'):
        names.update(str(path.relative_to(root)) for path in (root/package).rglob('*.py'))
    names.update(str(path.relative_to(root)) for path in (root/'dllm').rglob('*.json'))
    require(all(Path(name).suffix in SUFFIXES for name in names), 'Unexpected archive support file type')
    return {name:sha(relative_file(root,name)) for name in sorted(names)}


def plan_archive(request):
    request = canonical_request(request)
    blockers, bindings = [], {}
    def bound(path):
        value=read_json(path)
        bindings[str(path)]=sha(path)
        return value
    if request.destination.exists():
        blockers.append('Destination already exists; archives are never overwritten')
    pool_config=bound(request.pool/'pool.json')
    require(pool_config.get('source_root') == str(request.source_root), 'Pool source root differs')
    active=[]
    for state in ('pending','running'):
        for path in sorted((request.pool/state).glob('*.json')):
            active.append(dict(state=state,job_id=path.stem,job=bound(path)))
    if active:
        blockers.append(f'{len(active)} pending/running jobs remain; no snapshot write allowed')
    previous=verify_snapshot(request.previous_snapshot)
    bindings[str(request.previous_snapshot/'release_manifest.json')]=sha(request.previous_snapshot/'release_manifest.json')
    copied=source_inventory(request.source_root,previous)
    imports=import_inventory(request.source_root)
    measured=imports['measured_source_sha256']
    require(all(copied.get(name)==value for name,value in measured.items()), 'Measured source changed or missing from support copy')
    require(all(name in copied for name in REQUIRED), 'Required refined runtime dependency absent')
    require(ENTRY in measured, 'The refined entry must be profile-bound')
    observed=imports['imported_files_sha256']
    require(all(copied.get(name)==value for name,value in observed.items()), 'Observed import missing or changed in archive support tree')
    old=previous['copied_files_sha256']
    extra={name:dict(sha256=value,r12_status=('same' if old.get(name)==value else 'changed' if name in old else 'not_recorded'))
           for name,value in copied.items() if name not in measured}
    unmeasured_imports={name:extra[name] for name in sorted(observed) if name not in measured}
    if any(item['r12_status'] != 'same' for item in unmeasured_imports.values()):
        blockers.append('An imported dependency outside the old profile changed or lacks an R12 snapshot baseline')
    results={variant:[] for variant in FAMILY}
    for path in sorted((request.pool/'results').glob('*.json')):
        value=read_json(path);job=value.get('job',{})
        if job.get('variant') in FAMILY and job.get('steps')==500:
            bindings[str(path)]=sha(path)
            require(value.get('job_id')==path.stem, 'Result file/job identity differs')
            results[job['variant']].append((path,value))
    jobs={}
    for variant in FAMILY:
        directory=request.pool/'profiles'/variant
        try:
            proof,summary=bound(directory/'COMPLETE.json'),bound(directory/'summary.json')
            require(proof.get('source_sha256')==measured, 'Profile source map differs')
            require(proof.get('summary_sha256')==sha(directory/'summary.json'), 'Profile summary SHA differs')
            require(verify_original_profile(request.pool,variant)==summary, 'Original profile verification differs')
        except (ValueError,OSError,KeyError) as error:
            blockers.append(f'{variant} profile rejected: {error}')
        candidates=results[variant]
        if len(candidates)!=1:
            blockers.append(f'{variant}: require exactly one terminal original 500-step result, found {len(candidates)}')
            continue
        path,result=candidates[0];job=result['job']
        jobs[variant]=path.stem
        try:
            require(result.get('state')=='succeeded' and result.get('returncode')==0,
                    'Original 500-step task did not succeed')
            require(job.get('mode')=='train' and 'resume_from_checkpoint' not in job, 'Unexpected original diagnostic job protocol')
            directory=request.pool/'runs'/path.stem
            proof,summary=bound(directory/'COMPLETE.json'),bound(directory/'summary.json')
            require(proof.get('status')=='completed' and proof.get('global_step')==500 and
                    proof.get('summary_sha256')==sha(directory/'summary.json'), 'Incomplete diagnostic marker or bad SHA')
            require(summary.get('source_sha256')==measured and summary.get('variant')==variant,
                    'Diagnostic release/variant differs')
            require(verify_original_diagnostic(request.pool,variant)==summary,
                    'Original diagnostic verification selected another result')
        except (ValueError,OSError,KeyError,TypeError) as error:
            blockers.append(f'{variant} 500-step diagnostic rejected: {error}')
    return dict(schema='immune.refined_archive_plan.v1',ready=not blockers,blockers=blockers,
        source_root=str(request.source_root),pool=str(request.pool),frozen_root=str(request.destination),
        family=list(FAMILY),active_jobs=active,terminal_job_ids=jobs,
        measured_source_sha256=measured,copied_files_sha256=copied,evidence_sha256=bindings,
        dependency_audit=dict(import_observation=imports,unmeasured_imports=unmeasured_imports,
            all_unmeasured_support=extra,
            limitation='Supplemental source hashes were not part of the historical 39-source profile; matching R12 is a comparison, not a retroactive GPU certification. External packages are version-recorded, not vendored.'))


def archive_release(request, *, apply=False):
    request=canonical_request(request)
    plan=plan_archive(request)
    if not apply:
        return plan
    require(plan['ready'], 'Archive is blocked: '+'; '.join(plan['blockers']))
    with ExitStack() as stack:
        # Cooperate with existing architecture/refined enqueue helpers. Manual
        # queue writes do not honor these locks; recheck the full queue at publish.
        for name in ('architecture_enqueue.lock','refined_enqueue.lock'):
            lock=stack.enter_context((request.pool/name).open('a'))
            fcntl.flock(lock,fcntl.LOCK_EX)
        require(plan_archive(request)==plan, 'Archive inputs changed before copying')
        require(request.destination.parent.is_dir(), 'The existing releases parent must exist')
        staging=Path(tempfile.mkdtemp(prefix='.'+NAME+'.staging-',dir=request.destination.parent))
        try:
            for name,expected in plan['copied_files_sha256'].items():
                target=staging/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(relative_file(request.source_root,name),target)
                require(sha(target)==expected, f'Source changed while copying: {name}')
            manifest=dict(schema_version=1,created_at=datetime.now(timezone.utc).isoformat(),
                source_root=str(request.source_root),frozen_root=str(request.destination),family=list(FAMILY),
                measured_source_sha256=plan['measured_source_sha256'],copied_files_sha256=plan['copied_files_sha256'],
                archive_evidence_sha256=plan['evidence_sha256'],terminal_job_ids=plan['terminal_job_ids'],
                dependency_audit=plan['dependency_audit'],routing_changed=False)
            (staging/'release_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
            verify_snapshot(staging,expected_destination=request.destination)
            require(plan_archive(request)==plan, 'Queue/source/evidence changed before archive publication')
            require(not request.destination.exists(), 'Destination appeared; never overwrite')
            staging.rename(request.destination)
            return verify_snapshot(request.destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool-dir',required=True,type=Path)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args(argv)
    request=ArchiveRequest(ROOT,args.pool_dir,ROOT/'output/immune_research_releases/r12_r19_20260930',
                           ROOT/'output/immune_research_releases'/NAME)
    print(json.dumps(archive_release(request,apply=args.apply),indent=2,allow_nan=False))


if __name__=='__main__':
    main()
