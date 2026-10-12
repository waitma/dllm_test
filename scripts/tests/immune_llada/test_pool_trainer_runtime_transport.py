"""CPU-only D01 staging, source preflight and inherited Python child checks.

Activate pllm; python -m pytest -q scripts/tests/immune_llada/test_pool_trainer_runtime_transport.py
Only temporary artificial repositories and stdlib sentinels execute. No real
runtime, resource aliases, model, attention diagnostic, GPU or corpus is opened.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.architecture_prototypes import pool_release_dispatcher as dispatch
from scripts.architecture_prototypes import stage_pool_trainer_runtime_routing as staging
from scripts.tests.immune_llada.test_pool_release_dispatcher import (
    CHILD_SENTINEL, load_module, make_release, site_python)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def trainer_stage(tmp_path):
    original, proposed = staging.proposed_sources()
    root = tmp_path/'temporary_repo'
    for name, text in original.items():
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    stage = tmp_path/'review_stage'
    manifest = staging.stage(stage)
    subprocess.run(['git', 'apply', '--check', str(stage/'routing.patch')], cwd=root, check=True, capture_output=True)
    subprocess.run(['git', 'apply', str(stage/'routing.patch')], cwd=root, check=True, capture_output=True)
    for name, text in proposed.items():
        assert (root/name).read_text() == text
        assert sha(root/name) == manifest['after_sha256'][name]
    for name in (staging.DISPATCHER, dispatch.STARTUP_GUARD):
        target = root/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dispatch.ROOT/name, target)
    return root, stage, original, manifest


def trainer_args(root):
    pool = root/'artificial_transport_pool'
    (pool/'runs').mkdir(parents=True, exist_ok=True)
    (pool/'pool.json').write_text(json.dumps({'source_root':str(root)}))
    return ['--variant', dispatch.TRAINER_DIAGNOSTIC, '--mode', 'profile-synthetic', '--steps', '1',
        '--profile-dir', str(pool/'profiles'/dispatch.TRAINER_DIAGNOSTIC), '--output-dir', str(pool/'runs/job')]


def make_artificial_runtime(root, sentinel=CHILD_SENTINEL):
    """A tiny metadata/source fixture; it binds no existing resources or GPU proof."""
    parent, _ = make_release(root, 'r20_r27_20261001')
    for name in ('scripts/architecture_prototypes/pilot_runtime_package.py',
                 'scripts/architecture_prototypes/pilot_runtime_inspect.py'):
        file = parent/name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('# artificial frozen support source\n')
    proof = root/'artificial_parent_evidence.json'
    proof.write_text('{}\n')
    parent_manifest = json.loads((parent/'release_manifest.json').read_text())
    parent_manifest.update(family=list(dispatch.MODEL_FAMILIES['r20_r27_20261001']),
        terminal_job_ids={variant:'artificial-terminal-'+variant for variant in dispatch.MODEL_FAMILIES['r20_r27_20261001']},
        archive_evidence_sha256={str(proof):sha(proof)})
    files = {str(file.relative_to(parent)):sha(file) for package in ('scripts', 'examples', 'dllm')
             for file in (parent/package).rglob('*.py')}
    parent_manifest['copied_files_sha256'] = files
    (parent/'release_manifest.json').write_text(json.dumps(parent_manifest))
    parent_sha = sha(parent/'release_manifest.json')
    bootstrap = root/staging.DISPATCHER
    bootstrap.write_text(bootstrap.read_text().replace(dispatch.RUNTIME_PARENT_SHA256, parent_sha))
    runtime = root/'output/immune_pilot_runtimes'/dispatch.RUNTIME_NAME
    for name in files:
        target = runtime/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent/name, target)
    (runtime/'examples').mkdir(exist_ok=True)
    added = {}
    for name in dispatch.RUNTIME_ADDITIONS:
        original = root/name
        original.parent.mkdir(parents=True, exist_ok=True)
        original.write_text(sentinel.replace('parents[1]', 'parents[2]') if name == dispatch.RUNTIME_ENTRY
                            else '# artificial producer source, never executed\n')
        target = runtime/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        added[name] = sha(target)
    aliases = {}
    for name in dispatch.RUNTIME_ALIASES:
        target = root/name
        target.mkdir(parents=True, exist_ok=True)
        alias = runtime/name
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.symlink_to(target, target_is_directory=True)
        aliases[name] = str(target)
    manifest = dict(schema=dispatch.RUNTIME_SCHEMA, runtime_root=str(runtime), source_root=str(root),
        base_snapshot=str(parent), base_manifest_sha256=parent_sha, base_copied_files_sha256=files,
        measured_source_sha256=parent_manifest['measured_source_sha256'],
        inherited_terminal_job_ids=parent_manifest['terminal_job_ids'], additions_sha256=added,
        runtime_files_sha256={**files, **added}, external_resource_aliases=aliases,
        status='source_package_only_not_gpu_or_data_acceptance', gpu_accepted=False,
        data_accepted=False, training_started=False)
    (runtime/dispatch.RUNTIME_MANIFEST).write_text(json.dumps(manifest))
    return runtime, manifest


def test_new_stage_preserves_28_models_and_adds_only_nonconstructible_d01(trainer_stage):
    root, _, original, manifest = trainer_stage
    registry = dispatch.read_registry(root/'dllm/pipelines/immune_llada/research/model.py')
    old = ast.literal_eval(next(node.value for node in ast.parse(original['dllm/pipelines/immune_llada/research/model.py']).body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'VARIANTS' for target in node.targets)))
    assert len(old) == len(dispatch.MODEL_VARIANTS) == 28
    assert {name:options for name, options in registry.items() if name != dispatch.TRAINER_DIAGNOSTIC} == old
    assert registry[dispatch.TRAINER_DIAGNOSTIC] == {'diagnostic_only':True}
    assert dispatch.DIAGNOSTIC not in registry and manifest['independent_of_historical_d00_stage'] is True
    model = next(node for node in ast.parse((root/'dllm/pipelines/immune_llada/research/model.py').read_text()).body
                 if isinstance(node, ast.ClassDef) and node.name == 'ResearchModel')
    method = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == '__new__')
    isolated = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='ResearchModel',
        bases=[ast.Name(id='object', ctx=ast.Load())], keywords=[], body=[method], decorator_list=[])], type_ignores=[]))
    namespace = {'VARIANTS':registry}
    exec(compile(isolated, 'isolated_d01_constructor_guard', 'exec'), namespace)
    with pytest.raises(ValueError, match='cannot construct'):
        namespace['ResearchModel'](variant=dispatch.TRAINER_DIAGNOSTIC)


@pytest.mark.parametrize('change', [dict(mode='train', data_dir='/invented'), dict(steps=2), dict(steps=True),
    dict(steps=1.), dict(data_dir='/invented'), dict(resume_from_checkpoint='/invented'),
    dict(shell='anything'), dict(target='/invented'), dict(variant='unknown')])
def test_new_disk_validator_accepts_only_exact_d01_job(trainer_stage, change):
    module = load_module(trainer_stage[0]/'scripts/immune_research_pool.py', 'temporary_trainer_pool')
    valid = dict(variant=dispatch.TRAINER_DIAGNOSTIC, mode='profile-synthetic', steps=1)
    module.validate(valid)
    with pytest.raises(ValueError):
        module.validate({**valid, **change})


def test_temporary_init_still_has_only_twelve_legacy_jobs(trainer_stage):
    root = trainer_stage[0]
    pool = root/'artificial_init_pool'
    process = subprocess.run([sys.executable, '-S', str(root/'scripts/immune_research_pool.py'),
        'init', '--pool-dir', str(pool)], cwd=root, capture_output=True, text=True, timeout=15)
    assert process.returncode == 0, process.stderr
    jobs = [json.loads(path.read_text()) for path in (pool/'pending').glob('*.json')]
    assert len(jobs) == 12 and {job['variant'] for job in jobs} == set(dispatch.MODEL_FAMILIES['r00_r11_20260930'])


@pytest.mark.parametrize('family', list(dispatch.MODEL_FAMILIES))
def test_existing_family_routes_still_resolve_and_exec_their_own_snapshot(trainer_stage, family):
    root = trainer_stage[0]
    frozen, entry = make_release(root, family)
    module = load_module(root/staging.DISPATCHER, 'preserved_family_dispatcher')
    for variant in dispatch.MODEL_FAMILIES[family]:
        assert module.resolve_route(['--variant', variant]) == (frozen, entry)
    process = subprocess.run([sys.executable, '-S', str(root/'scripts/run_immune_research.py'),
        '--variant', dispatch.MODEL_FAMILIES[family][0]], cwd=root, capture_output=True, text=True, timeout=15)
    assert process.returncode == 0, process.stderr
    observed = json.loads(process.stdout)
    assert observed['entry'] == str(entry) and observed['cwd'] == str(frozen) and observed['torch_loaded'] is False


def test_fixed_runtime_resolves_after_stdlib_source_preflight(trainer_stage):
    root = trainer_stage[0]
    runtime, _ = make_artificial_runtime(root)
    module = load_module(root/staging.DISPATCHER, 'temporary_trainer_dispatcher')
    before = set(sys.modules)
    assert module.resolve_route(trainer_args(root)) == (runtime, runtime/dispatch.RUNTIME_ENTRY)
    assert set(sys.modules) == before


@pytest.mark.parametrize('extra', [['--data-dir', '/invented'], ['--resume-from-checkpoint', '/invented'],
    ['--shell', 'anything'], ['--target', '/invented'], ['--steps', '2'], ['--mode', 'train']])
def test_old_worker_arguments_cannot_expand_fixed_runtime_transport(trainer_stage, extra):
    root = trainer_stage[0]
    process = subprocess.run([sys.executable, '-S', str(root/'scripts/run_immune_research.py'),
        *trainer_args(root), *extra], cwd=root, capture_output=True, text=True, timeout=15)
    assert process.returncode == 2 and 'Dispatch rejected' in process.stderr
    assert 'torch' not in process.stderr.lower() and not (root/'artificial_transport_pool/runs/job').exists()


@pytest.mark.parametrize('job', [dict(variant=dispatch.TRAINER_DIAGNOSTIC, mode='profile-synthetic', steps=2),
    dict(variant=dispatch.TRAINER_DIAGNOSTIC, mode='train', steps=1, data_dir='/invented')])
def test_actual_old_in_memory_validator_is_rejected_by_new_dispatcher(trainer_stage, job):
    root, _, original, _ = trainer_stage
    path = root/'scripts/artificial_old_worker.py'
    path.write_text(original['scripts/immune_research_pool.py'])
    old = load_module(path, 'old_trainer_worker_fixture')
    old.validate(job)
    pool = root/'artificial_old_worker_pool'
    (pool/'runs').mkdir(parents=True)
    command = old.command(pool, 'rejected', job)
    process = subprocess.run([command[0], '-S', *command[1:]], cwd=root, capture_output=True, text=True, timeout=15)
    assert process.returncode == 2 and 'Dispatch rejected' in process.stderr
    assert not (pool/'runs/rejected').exists() and 'torch' not in process.stderr.lower()


@pytest.mark.parametrize('kind', ['missing_runtime', 'changed_source', 'changed_addition', 'wrong_parent',
    'missing_addition', 'extra_source', 'wrong_alias', 'extra_alias', 'duplicate_manifest'])
def test_runtime_mutations_fail_before_entry_execution(trainer_stage, kind):
    root = trainer_stage[0]
    args = trainer_args(root)
    runtime, manifest = make_artificial_runtime(root)
    path = runtime/dispatch.RUNTIME_MANIFEST
    if kind == 'missing_runtime':
        shutil.rmtree(runtime)
    elif kind == 'changed_source':
        (runtime/'dllm/__init__.py').write_text('changed\n')
    elif kind == 'changed_addition':
        (root/dispatch.RUNTIME_ENTRY).write_text('changed live addition\n')
    elif kind == 'wrong_parent':
        manifest['base_manifest_sha256'] = '0'*64
    elif kind == 'missing_addition':
        manifest['additions_sha256'].pop(dispatch.RUNTIME_ENTRY)
    elif kind == 'extra_source':
        (runtime/'scripts/unrecorded.py').write_text('# unrecorded\n')
    elif kind == 'wrong_alias':
        alias = runtime/dispatch.RUNTIME_ALIASES[0]
        alias.unlink()
        alias.symlink_to(root, target_is_directory=True)
    elif kind == 'extra_alias':
        (runtime/'extra_alias').symlink_to(root, target_is_directory=True)
    elif kind == 'duplicate_manifest':
        path.write_text('{"schema":1,"schema":1}')
    if kind in ('wrong_parent', 'missing_addition'):
        path.write_text(json.dumps(manifest))
    process = subprocess.run([sys.executable, '-S', str(root/'scripts/run_immune_research.py'), *args],
        cwd=root, capture_output=True, text=True, timeout=15)
    assert process.returncode == 2 and 'Dispatch rejected' in process.stderr
    assert not process.stdout and not (root/'artificial_transport_pool/runs/job').exists()


def test_real_site_runtime_parent_and_both_python_children_inherit_isolation(trainer_stage, site_python, tmp_path):
    root = trainer_stage[0]
    runtime, _ = make_artificial_runtime(root)
    python, site = site_python
    foreign = tmp_path/'external_site_packages'
    for name in ('scripts', 'examples', 'dllm'):
        (foreign/name).mkdir(parents=True)
        (foreign/name/'__init__.py').write_text('raise RuntimeError("external regular package executed")\n')
    (foreign/'scripts/foreign_only.py').write_text('VALUE = "external"\n')
    (site/'foreign_source_paths.pth').write_text(str(root)+'\n'+str(foreign)+'\n')
    process = subprocess.run([str(python), str(root/'scripts/run_immune_research.py'), *trainer_args(root)],
        cwd=root, capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    records = [result, result['subprocess_child'], result['posix_spawn_child']]
    assert len({record['pid'] for record in records}) == 3
    for record in records:
        assert record['guard'] == dict(root=str(runtime), route=dispatch.TRAINER_ROUTE,
            guard_sha256=dispatch.STARTUP_GUARD_SHA256, pid=record['pid'])
        assert record['scripts_path'] == [str(runtime/'scripts')]
        assert record['examples_path'] == [str(runtime/'examples')]
        assert Path(record['dllm_file']).resolve() == runtime/'dllm/__init__.py'
        assert record['foreign_rejected'] and not record['torch_loaded']
        assert all(not Path(value).resolve().is_relative_to(root) or Path(value).resolve().is_relative_to(runtime)
                   for value in record['sys_path'] if value)


def test_live_measured_sources_remain_at_their_original_sha_and_help_is_stdlib():
    assert all(sha(dispatch.ROOT/name) == expected for name, expected in staging.BASE_SHA256.items())
    process = subprocess.run([sys.executable, '-S', str(dispatch.ROOT/staging.DISPATCHER), '--help'],
                             capture_output=True, text=True, timeout=15)
    assert process.returncode == 0 and '28 frozen model variants' in process.stdout
    assert 'd01' in process.stdout and process.stderr == ''
