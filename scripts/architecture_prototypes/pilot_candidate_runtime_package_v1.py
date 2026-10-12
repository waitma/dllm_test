"""Fixed R21/R24/R26 source package preparation, never GPU or data acceptance.

Activate pllm; USE_TF=0 CUDA_VISIBLE_DEVICES='' python -m pytest
scripts/tests/immune_llada/test_pilot_candidate_runtime_v1.py -q
API: plan_runtime(base, destination, variant=..., additions=...), build_runtime
with the same fixed arguments, and verify_runtime(directory). Only the three
canonical names and exactly 524 parent plus four new files are accepted.
No CLI, model, dataset, queue, or GPU operation is provided.
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
PARENT_SHA = '46399204c7018d00726e3ae08142bd4eb28475c8a9a2991b3a5c90725c8824f7'
MANIFEST = 'pilot_runtime_manifest.json'
PACKAGE = 'scripts/architecture_prototypes/pilot_candidate_runtime_package_v1.py'
INSPECTOR = 'scripts/architecture_prototypes/pilot_candidate_runtime_inspect_v1.py'
REHEARSAL = 'SOURCE_COPY_REHEARSAL_ONLY.json'
CANDIDATES = {
    'r21_refined_head_gate': dict(package_name='r21_pilot_v1_20261001',
        entry='scripts/architecture_prototypes/pilot_runtime_entry_r21_v1.py',
        producer='scripts/architecture_prototypes/pilot_gpu_release_producer_r21_v1.py'),
    'r24_refined_fixed_scale': dict(package_name='r24_pilot_v1_20261001',
        entry='scripts/architecture_prototypes/pilot_runtime_entry_r24_v1.py',
        producer='scripts/architecture_prototypes/pilot_gpu_release_producer_r24_v1.py'),
    'r26_refined_local_global': dict(package_name='r26_pilot_v1_20261001',
        entry='scripts/architecture_prototypes/pilot_runtime_entry_r26_v1.py',
        producer='scripts/architecture_prototypes/pilot_gpu_release_producer_r26_v1.py'),
}
RESOURCE_ALIASES = (
    'downstream/benchmark/data',
    'output/research/uptcr_20260930/released_tables/data/finetune/abpm',
)
require, sha, read_json = archive.require, archive.sha, archive.read_json


def candidate(variant):
    require(type(variant) is str and variant in CANDIDATES, 'Only fixed R21/R24/R26 candidates are supported')
    return CANDIDATES[variant]


def additions_for(variant):
    identity = candidate(variant)
    return frozenset((PACKAGE, INSPECTOR, identity['entry'], identity['producer']))


def aliases_for(original_root):
    return {name: str(Path(original_root)/name) for name in RESOURCE_ALIASES}


def verify_base(base_snapshot):
    base_snapshot = Path(base_snapshot).absolute()
    require(base_snapshot == base_snapshot.resolve() and base_snapshot.name == archive.NAME,
            'Use the canonical completed R20--R27 snapshot')
    require(sha(base_snapshot/'release_manifest.json') == PARENT_SHA, 'Fixed parent manifest SHA differs')
    manifest = archive.verify_snapshot(base_snapshot)
    source = Path(manifest['source_root'])
    require(source.is_absolute() and source == source.resolve()
            and base_snapshot == source/'output/immune_research_releases'/archive.NAME,
            'Parent source/snapshot identity differs')
    require(len(manifest['copied_files_sha256']) == 524 and len(manifest['measured_source_sha256']) == 39,
            'Parent must retain exactly 524 copied and 39 measured sources')
    require(manifest.get('family') == list(archive.FAMILY)
            and set(manifest.get('terminal_job_ids', {})) == set(archive.FAMILY),
            'Parent archive lacks all eight terminal diagnostic identities')
    evidence = manifest.get('archive_evidence_sha256')
    require(type(evidence) is dict and evidence, 'Parent archive lacks original evidence bindings')
    for name, value in evidence.items():
        path = Path(name)
        require(path.is_absolute() and path.is_file() and sha(path) == value,
                'Parent archive evidence is missing or changed: '+name)
    require(archive.ENTRY in manifest['measured_source_sha256'], 'Parent refined entry is not profile-bound')
    return manifest


def plan_runtime(base_snapshot, destination, *, variant, additions=None):
    identity = candidate(variant)
    base_snapshot, destination = Path(base_snapshot).absolute(), Path(destination).absolute()
    result = dict(schema=SCHEMA, ready=False, blockers=[], candidate_variant=variant,
        package_name=identity['package_name'], base_snapshot=str(base_snapshot), runtime_root=str(destination),
        gpu_accepted=False, data_accepted=False, training_started=False)
    require(base_snapshot == base_snapshot.resolve() and destination == destination.resolve(), 'Canonical paths required')
    if not base_snapshot.is_dir():
        result['blockers'].append('Completed R20--R27 archive does not exist; no runtime directory may be created')
        return result
    parent = verify_base(base_snapshot)
    original_root = Path(parent['source_root'])
    require(destination == original_root/'output/immune_pilot_runtimes'/identity['package_name'],
            'Use this candidate fixed pilot runtime destination')
    if destination.exists():
        result['blockers'].append('Runtime destination exists; never overwrite an immutable package')
    require(type(additions) is dict and set(additions) == additions_for(variant),
            'Exactly this candidate four fixed new additions are required')
    base_files = parent['copied_files_sha256']
    require(not set(additions)&set(base_files), 'Additions must never replace any parent file')
    added = {}
    for name, path in additions.items():
        path = Path(path).absolute()
        require(path == original_root/name and path == path.resolve() and path.is_file() and not path.is_symlink(),
                'Addition must be its explicit canonical original-root script')
        added[name] = sha(path)
    files = {**base_files, **added}
    require(len(files) == 528, 'Candidate runtime must bind exactly 528 source files')
    aliases = aliases_for(original_root)
    for name, target in aliases.items():
        require(not any(Path(file).is_relative_to(name) or Path(name).is_relative_to(file) for file in files),
                'Resource alias overlaps copied source')
        if not Path(target).is_dir():
            result['blockers'].append('Missing required external resource directory: '+target)
    result.update(ready=not result['blockers'], source_root=str(original_root),
        base_manifest_sha256=PARENT_SHA, base_copied_files_sha256=base_files,
        measured_source_sha256=parent['measured_source_sha256'], additions_sha256=added,
        runtime_files_sha256=files, external_resource_aliases=aliases,
        inherited_terminal_job_ids=parent['terminal_job_ids'])
    return result


def verify_runtime(directory, *, expected_destination=None):
    directory = Path(directory).absolute()
    require(directory == directory.resolve() and directory.is_dir(), 'Runtime directory must be canonical')
    manifest = read_json(archive.relative_file(directory, MANIFEST))
    variant = manifest.get('candidate_variant')
    identity = candidate(variant)
    require(manifest.get('schema') == SCHEMA and manifest.get('package_name') == identity['package_name']
            and manifest.get('status') == 'source_package_only_not_gpu_or_data_acceptance'
            and all(manifest.get(key) is False for key in ('gpu_accepted', 'data_accepted', 'training_started')),
            'Runtime identity or source-only acceptance status differs')
    base = Path(manifest['base_snapshot']); parent = verify_base(base)
    destination = Path(parent['source_root'])/'output/immune_pilot_runtimes'/identity['package_name']
    require(manifest.get('runtime_root') == str(destination), 'Candidate canonical runtime root differs')
    if expected_destination is None:
        require(directory == destination, 'Use only this candidate canonical directory')
    else:
        require(Path(expected_destination) == destination and directory.parent == destination.parent
                and directory.name.startswith('.'+identity['package_name']+'.staging-'),
                'Staging verification requires its fixed candidate destination and owned temporary directory')
    require(manifest.get('base_manifest_sha256') == PARENT_SHA
            and manifest.get('source_root') == parent['source_root']
            and manifest.get('base_copied_files_sha256') == parent['copied_files_sha256']
            and manifest.get('measured_source_sha256') == parent['measured_source_sha256']
            and manifest.get('inherited_terminal_job_ids') == parent['terminal_job_ids'],
            'Parent source/proof contract differs')
    added = manifest.get('additions_sha256')
    require(type(added) is dict and set(added) == additions_for(variant)
            and not set(added)&set(parent['copied_files_sha256']), 'Invalid or replacing candidate addition')
    files = {**parent['copied_files_sha256'], **added}
    require(len(files) == 528 and manifest.get('runtime_files_sha256') == files, 'Runtime full 528 source map differs')
    for name, value in files.items():
        require(sha(archive.relative_file(directory, name)) == value, 'Runtime source SHA differs: '+name)
    actual = {str(path.relative_to(directory)) for module in ('dllm', 'examples', 'scripts')
        for path in (directory/module).rglob('*') if path.is_file() and path.suffix in archive.SUFFIXES}
    require(actual == set(files), 'Runtime contains unrecorded or missing source/config files')
    aliases = aliases_for(parent['source_root'])
    require(manifest.get('external_resource_aliases') == aliases, 'Only two exact original-root resource aliases are allowed')
    symlinks = {str((Path(folder)/name).relative_to(directory))
        for folder, dirs, names in os.walk(directory, followlinks=False)
        for name in dirs+names if (Path(folder)/name).is_symlink()}
    require(symlinks == set(aliases), 'Runtime contains extra or missing resource/source aliases')
    for name, target in aliases.items():
        path = directory/name
        require(path.is_symlink() and os.readlink(path) == target and path.resolve() == Path(target).resolve()
                and path.is_dir(), 'Resource alias is missing, redirected or its target is unavailable')
    return manifest


def build_runtime(base_snapshot, destination, *, variant, additions=None):
    plan = plan_runtime(base_snapshot, destination, variant=variant, additions=additions)
    require(plan['ready'], 'Runtime package blocked: '+'; '.join(plan['blockers']))
    destination = Path(plan['runtime_root']); base = Path(plan['base_snapshot'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+plan['package_name']+'.staging-', dir=destination.parent))
    try:
        for name, value in plan['runtime_files_sha256'].items():
            source = (base if name in plan['base_copied_files_sha256'] else Path(plan['source_root']))/name
            target = staging/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            require(sha(target) == value, 'Source changed while copying runtime: '+name)
        for name, target in plan['external_resource_aliases'].items():
            alias = staging/name; alias.parent.mkdir(parents=True, exist_ok=True); alias.symlink_to(target, target_is_directory=True)
        manifest = {key: value for key, value in plan.items() if key not in ('ready', 'blockers')}
        manifest.update(status='source_package_only_not_gpu_or_data_acceptance', created_at=datetime.now(timezone.utc).isoformat())
        (staging/MANIFEST).write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
        verify_runtime(staging, expected_destination=destination)
        require(plan_runtime(base, destination, variant=variant, additions=additions) == plan,
                'Parent/addition/resource changed before publication')
        require(not destination.exists(), 'Runtime appeared; never overwrite')
        staging.rename(destination)
        return verify_runtime(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def inspect_fresh(directory, pool, *, mode='runtime', python=sys.executable):
    """Fixed CPU inspection; no arbitrary command, variant, model or execution mode."""
    require(mode in ('runtime', 'source-copy-rehearsal'), 'Unknown inspection mode')
    allowed = {Path(sys.executable).resolve(),
        Path('/vepfs-mlp2/c20250601/251105016/conda/envs/pllm/bin/python').resolve(),
        Path('/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python').resolve()}
    require(Path(python).is_absolute() and Path(python).resolve() in allowed,
            'Inspection accepts only current or two explicitly installed Python interpreters')
    directory, pool = Path(directory).resolve(), Path(pool).resolve()
    entry = archive.relative_file(directory, INSPECTOR)
    if mode == 'runtime':
        verify_runtime(directory)
    result = subprocess.run([str(python), '-I', str(entry), '--inspection-mode', mode, '--pool-dir', str(pool)],
        cwd=directory, env={**os.environ, 'PYTHONPATH': str(directory), 'PYTHONDONTWRITEBYTECODE': '1',
            'USE_TF': '0', 'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True, timeout=60)
    require(result.returncode == 0, 'CPU candidate inspection failed: '+result.stderr[-6000:])
    return json.loads(result.stdout)
