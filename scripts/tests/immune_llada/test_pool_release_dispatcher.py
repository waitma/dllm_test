"""CPU-only staged patch tests; activate pllm and run pytest -q on this file.

Only temporary source copies and harmless standard-library sentinel entries are
executed. No Torch/model/adapter imports, CUDA, datasets, or real pool writes.
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.architecture_prototypes import pool_release_dispatcher as dispatch
from scripts.architecture_prototypes import stage_pool_diagnostic_routing as staging


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def staged(tmp_path):
    original, proposed = staging.proposed_sources()
    root = tmp_path/'temporary_repo'
    for name, text in original.items():
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    stage = tmp_path/'review_stage'
    manifest = staging.stage(stage)
    # Exercise the actual emitted unified diff, not only replacement helpers.
    subprocess.run(['git','apply','--check',str(stage/'routing.patch')], cwd=root, check=True, capture_output=True)
    subprocess.run(['git','apply',str(stage/'routing.patch')], cwd=root, check=True, capture_output=True)
    for name, text in proposed.items():
        assert (root/name).read_text() == text
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == manifest['after_sha256'][name]
    path = root/staging.DISPATCHER
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dispatch.__file__, path)
    guard = root/dispatch.STARTUP_GUARD
    guard.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dispatch.ROOT/dispatch.STARTUP_GUARD, guard)
    return root, stage, original, manifest


def make_release(root, family):
    frozen = root/'output/immune_research_releases'/family
    entry = frozen/dispatch.ENTRIES[family]
    entry.parent.mkdir(parents=True)
    (frozen/'examples').mkdir()
    (frozen/'dllm').mkdir()
    (frozen/'dllm/__init__.py').write_text('LOCATION = __file__\n')
    entry.write_text('import json,os,sys\nprint(json.dumps(dict(entry=__file__,cwd=os.getcwd(),'
                     'torch_loaded="torch" in sys.modules,argv=sys.argv[1:],pythonpath=os.environ.get("PYTHONPATH"))))\n')
    hashes = {dispatch.ENTRIES[family]: hashlib.sha256(entry.read_bytes()).hexdigest()}
    if family == 'r00_r11_20260930':
        manifest = dict(kind='immutable_legacy_source', source_root=str(root), copied_files=1, measured_source_sha256=hashes)
    else:
        manifest = dict(schema_version=1, source_root=str(root), frozen_root=str(frozen),
                        measured_source_sha256=hashes, copied_files_sha256=hashes)
    (frozen/'release_manifest.json').write_text(json.dumps(manifest))
    return frozen, entry


def test_patch_keeps_exact_28_models_and_one_explicit_diagnostic(staged):
    root, _, original, _ = staged
    registry_path = root/'dllm/pipelines/immune_llada/research/model.py'
    registry = dispatch.read_registry(registry_path)
    old_tree = ast.parse(original['dllm/pipelines/immune_llada/research/model.py'])
    old = ast.literal_eval(next(node.value for node in old_tree.body if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id=='VARIANTS' for target in node.targets)))
    assert len(old) == len(dispatch.MODEL_VARIANTS) == 28 and set(old) == set(dispatch.MODEL_VARIANTS)
    assert {key:value for key,value in registry.items() if key != dispatch.DIAGNOSTIC} == old
    assert len(registry) == 29 and registry[dispatch.DIAGNOSTIC] == {'diagnostic_only':True}
    model_node = next(node for node in ast.parse(registry_path.read_text()).body if isinstance(node,ast.ClassDef) and node.name=='ResearchModel')
    method = next(node for node in model_node.body if isinstance(node,ast.FunctionDef) and node.name=='__new__')
    fake_model = ast.ClassDef(name='ResearchModel',bases=[ast.Name(id='object',ctx=ast.Load())],keywords=[],body=[method],decorator_list=[])
    code = ast.fix_missing_locations(ast.Module(body=[fake_model],type_ignores=[]))
    namespace = {'VARIANTS':registry}
    exec(compile(code,'isolated_constructor_guard','exec'),namespace)
    with pytest.raises(ValueError,match='cannot construct'):
        namespace['ResearchModel'](variant=dispatch.DIAGNOSTIC)


def test_temporary_pool_init_does_not_enqueue_diagnostic(staged):
    root, _, _, _ = staged
    pool = root/'test_pool'
    process = subprocess.run([sys.executable,'-S',str(root/'scripts/immune_research_pool.py'),
        'init','--pool-dir',str(pool)],cwd=root,capture_output=True,text=True)
    assert process.returncode == 0, process.stderr
    jobs = [json.loads(path.read_text()) for path in (pool/'pending').glob('*.json')]
    assert len(jobs) == 12 and {job['variant'] for job in jobs} == set(dispatch.MODEL_FAMILIES['r00_r11_20260930'])
    assert all(job['variant'] != dispatch.DIAGNOSTIC for job in jobs)


@pytest.mark.parametrize('change',[dict(mode='train',data_dir='/invented'),dict(steps=2),dict(steps=True),dict(steps=1.),
    dict(data_dir='/invented'),dict(resume_from_checkpoint='/invented'),dict(shell='anything'),dict(variant='unknown')])
def test_future_disk_validate_rejects_invalid_diagnostic(staged,change):
    module = load_module(staged[0]/'scripts/immune_research_pool.py','temporary_pool_validator')
    valid = dict(variant=dispatch.DIAGNOSTIC,mode='profile-synthetic',steps=1)
    module.validate(valid)
    with pytest.raises(ValueError):module.validate({**valid,**change})


@pytest.mark.parametrize('family',list(dispatch.MODEL_FAMILIES))
def test_patched_stable_entry_executes_only_snapshot_sentinel(staged,family):
    root = staged[0]
    frozen, entry = make_release(root,family)
    variant = dispatch.MODEL_FAMILIES[family][0]
    environment = {**os.environ,'PYTHONPATH':str(root/'wrong_live_path')}
    process = subprocess.run([sys.executable,'-S',str(root/'scripts/run_immune_research.py'),
        '--variant',variant,'--mode','profile-synthetic','--steps','1'],cwd=root,env=environment,capture_output=True,text=True)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result['entry'] == str(entry) and result['cwd'] == str(frozen)
    assert result['pythonpath'] == os.pathsep.join((str((root/dispatch.STARTUP_GUARD).parent), str(frozen)))
    assert result['torch_loaded'] is False
    assert result['argv'] == ['--variant',variant,'--mode','profile-synthetic','--steps','1']


@pytest.mark.parametrize('argv',[[],['--variant'],['--variant','unknown'],['--variant','r00_a4_control','--variant=r00_a4_control'],
    ['--variant=r00_a4_control','--variant','r01_span_mask'],['--var','r00_a4_control']])
def test_unknown_missing_or_duplicate_cli_variant_is_rejected(argv):
    with pytest.raises(ValueError):dispatch.selected_variant(argv)


def diagnostic_args(root):
    pool = root/'diagnostic_pool'
    (pool/'runs').mkdir(parents=True)
    for name in dispatch.DIAGNOSTIC_SOURCES:
        path = root/name
        path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(dispatch.ROOT/name,path)
    return ['--variant',dispatch.DIAGNOSTIC,'--mode','profile-synthetic','--steps','1',
        '--profile-dir',str(pool/'profiles'/dispatch.DIAGNOSTIC),'--output-dir',str(pool/'runs'/'only_one')]


def test_diagnostic_resolves_to_pinned_adapter_without_importing_it(staged):
    root = staged[0]
    args = diagnostic_args(root)
    module = load_module(root/staging.DISPATCHER,'temporary_dispatcher')
    before = set(sys.modules)
    target_root, entry = module.resolve_route(args)
    assert target_root == root and entry == root/'scripts/architecture_prototypes/pool_attention_diagnostic.py'
    assert set(sys.modules) == before
    # Bound source is the qualified four-case implementation; no adapter import.
    tree = ast.parse(entry.read_text())
    function = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='fixed_cases')
    value = next(node.value for node in function.body if isinstance(node,ast.Return))
    assert [ast.literal_eval(generator.iter) for generator in value.generators] == [('math','efficient'),('float32','bfloat16')]
    assert len(value.generators[0].iter.elts)*len(value.generators[1].iter.elts) == 4


@pytest.mark.parametrize('extra',[['--data-dir','/invented'],['--resume-from-checkpoint','/invented'],['--steps','2'],
    ['--mode','train'],['--worker-spec','/invented'],['--case-timeout','999']])
def test_old_worker_transport_cannot_expand_diagnostic_cli(staged,extra):
    root = staged[0]
    args = diagnostic_args(root)+extra
    process = subprocess.run([sys.executable,'-S',str(root/'scripts/run_immune_research.py'),*args],
        cwd=root,capture_output=True,text=True)
    assert process.returncode == 2 and 'Dispatch rejected' in process.stderr
    assert 'torch' not in process.stderr.lower() and not (root/'diagnostic_pool/runs/only_one').exists()


@pytest.mark.parametrize('job',[dict(variant=dispatch.DIAGNOSTIC,mode='profile-synthetic',steps=2),
    dict(variant=dispatch.DIAGNOSTIC,mode='train',steps=1,data_dir='/invented')])
def test_actual_old_validator_acceptance_is_still_blocked_before_import(staged,job):
    root, _, original, _ = staged
    path = root/'scripts/old_worker_fixture.py'
    path.write_text(original['scripts/immune_research_pool.py'])
    module = load_module(path,'old_in_memory_worker_fixture')
    module.validate(job)  # Existing worker code reads the new registry dynamically.
    pool = root/'old_worker_pool'
    (pool/'runs').mkdir(parents=True)
    command = module.command(pool,'rejected',job)
    process = subprocess.run([command[0],'-S',*command[1:]],cwd=root,capture_output=True,text=True)
    assert process.returncode == 2 and 'Dispatch rejected' in process.stderr
    assert not (pool/'runs/rejected').exists() and 'torch' not in process.stderr.lower()


def test_changed_qualified_diagnostic_source_rejected(staged):
    root = staged[0]
    args = diagnostic_args(root)
    (root/'scripts/architecture_prototypes/attention_backends.py').write_text('changed')
    module = load_module(root/staging.DISPATCHER,'changed_diagnostic_dispatcher')
    with pytest.raises(ValueError,match='digest differs'):module.resolve_route(args)


def test_duplicate_registry_keys_rejected_by_patched_pool(staged):
    root = staged[0]
    path = root/'dllm/pipelines/immune_llada/research/model.py'
    path.write_text(path.read_text().replace("    'r00_a4_control': {},", "    'r00_a4_control': {},\n    'r00_a4_control': {},"))
    module = load_module(root/'scripts/immune_research_pool.py','duplicate_pool_validator')
    with pytest.raises(ValueError,match='Duplicate'):module.variants()
    with pytest.raises(ValueError,match='Duplicate'):dispatch.read_registry(path)


@pytest.mark.parametrize('kind',['missing','changed_file','outside_path','changed_root','missing_entry_hash','duplicate_manifest'])
def test_snapshot_integrity_errors_rejected(staged,kind):
    root = staged[0]
    family = 'r20_r27_20261001'
    module = load_module(root/staging.DISPATCHER,'invalid_release_dispatcher')
    if kind == 'missing':
        with pytest.raises(ValueError,match='missing'):module.frozen_entry(family)
        return
    frozen, entry = make_release(root,family)
    path = frozen/'release_manifest.json'
    manifest = json.loads(path.read_text())
    if kind == 'changed_file':entry.write_text('changed')
    elif kind == 'outside_path':manifest['copied_files_sha256']['../outside.py']='0'*64
    elif kind == 'changed_root':manifest['frozen_root']=str(root)
    elif kind == 'missing_entry_hash':manifest['copied_files_sha256']={}
    elif kind == 'duplicate_manifest':
        path.write_text('{"schema_version":1,"schema_version":1}')
        with pytest.raises(ValueError,match='Duplicate'):module.frozen_entry(family)
        return
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):module.frozen_entry(family)


def test_dispatcher_import_and_help_are_standard_library_only(staged):
    root = staged[0]
    process = subprocess.run([sys.executable,'-S',str(root/'scripts/run_immune_research.py'),'--help'],
        cwd=root,capture_output=True,text=True)
    assert process.returncode == 0 and '28 frozen model variants' in process.stdout
    assert process.stderr == ''
    assert all((dispatch.ROOT/name).read_bytes() and hashlib.sha256((dispatch.ROOT/name).read_bytes()).hexdigest()==expected
               for name,expected in staging.BASE_SHA256.items())


def test_stager_refuses_changed_base_and_existing_stage(staged,tmp_path):
    root, stage, _, _ = staged
    with pytest.raises(ValueError,match='fresh'):staging.stage(stage)
    with pytest.raises(ValueError,match='SHA differs'):staging.proposed_sources(root)


@pytest.fixture
def site_python(tmp_path):
    """A real site startup with artificial .pth; installed packages stay read-only."""
    environment = tmp_path/'python_with_test_site'
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', '--system-site-packages', str(environment)],
                   check=True, capture_output=True, text=True)
    site = environment/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages'
    return environment/'bin/python', site


def replace_entry(frozen, entry, text):
    entry.write_text(text)
    path = frozen/'release_manifest.json'
    manifest = json.loads(path.read_text())
    relative = str(entry.relative_to(frozen))
    for key in ('measured_source_sha256', 'copied_files_sha256'):
        if key in manifest:
            manifest[key][relative] = hashlib.sha256(entry.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))


CHILD_SENTINEL = '''import importlib,json,os,subprocess,sys,tempfile
from pathlib import Path
import scripts,examples,dllm
root=Path(__file__).resolve().parents[1]
data=dict(root=str(root),pid=os.getpid(),guard=getattr(sys,'_immune_release_isolation',None),
          scripts_path=list(scripts.__path__),examples_path=list(examples.__path__),
          dllm_file=dllm.__file__,sys_path=sys.path,torch_loaded='torch' in sys.modules)
try:
    importlib.import_module('scripts.foreign_only')
except ModuleNotFoundError:
    data['foreign_rejected']=True
else:
    data['foreign_rejected']=False
if '--child' not in sys.argv:
    data['subprocess_child']=json.loads(subprocess.check_output([sys.executable,__file__,'--child'],text=True))
    with tempfile.TemporaryDirectory() as directory:
        output=str(Path(directory)/'child.json')
        child=os.posix_spawn(sys.executable,[sys.executable,__file__,'--child'],dict(os.environ),
            file_actions=[(os.POSIX_SPAWN_OPEN,1,output,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)])
        _,status=os.waitpid(child,0)
        if os.waitstatus_to_exitcode(status)!=0:raise RuntimeError('posix_spawn child failed')
        data['posix_spawn_child']=json.loads(Path(output).read_text())
print(json.dumps(data))
'''


@pytest.mark.parametrize('family', [*dispatch.MODEL_FAMILIES, 'diagnostic'])
def test_real_site_parent_and_both_child_launches_are_isolated(staged, site_python, family, tmp_path):
    root = staged[0]
    if family == 'diagnostic':
        frozen, entry = root, root/'scripts/cpu_diagnostic_transport_sentinel.py'
        (root/'examples').mkdir()
        (root/'dllm/__init__.py').write_text('LOCATION = __file__\n')
        entry.write_text(CHILD_SENTINEL)
        args = diagnostic_args(root)
        module = load_module(root/staging.DISPATCHER, 'diagnostic_sentinel_dispatcher')
        environment = module.guarded_environment(root, args)
    else:
        frozen, entry = make_release(root, family)
        replace_entry(frozen, entry, CHILD_SENTINEL)
        environment = None
    python, site = site_python
    live_extra = root/'unselected_live_source'
    (live_extra/'scripts').mkdir(parents=True)
    (live_extra/'scripts/foreign_only.py').write_text('VALUE = "unselected_live"\n')
    foreign = tmp_path/'external_site_package'
    for name in ('scripts', 'examples', 'dllm'):
        (foreign/name).mkdir(parents=True)
        (foreign/name/'__init__.py').write_text('raise RuntimeError("external regular package executed")\n')
    (foreign/'scripts/foreign_only.py').write_text('VALUE = "external"\n')
    (site/'test_source_paths.pth').write_text(str(live_extra)+'\n'+str(foreign)+'\n')
    command = ([str(python), str(entry)] if family == 'diagnostic' else
        [str(python), str(root/'scripts/run_immune_research.py'), '--variant',
         dispatch.MODEL_FAMILIES[family][0], '--mode', 'profile-synthetic', '--steps', '1'])
    process = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    records = [result, result['subprocess_child'], result['posix_spawn_child']]
    assert len({record['pid'] for record in records}) == 3
    for record in records:
        assert record['guard'] == dict(root=str(frozen), route=family,
            guard_sha256=dispatch.STARTUP_GUARD_SHA256, pid=record['pid'])
        assert record['scripts_path'] == [str(frozen/'scripts')]
        assert record['examples_path'] == [str(frozen/'examples')]
        assert Path(record['dllm_file']).resolve() == frozen/'dllm/__init__.py'
        assert record['foreign_rejected'] and not record['torch_loaded']
        assert all(not Path(value).resolve().is_relative_to(root) or Path(value).resolve().is_relative_to(frozen)
                   for value in record['sys_path'] if value)


@pytest.mark.parametrize('name', ['scripts', 'examples', 'dllm'])
def test_pth_preloaded_project_module_exits_before_target(staged, site_python, name):
    root = staged[0]
    family = 'r12_r19_20260930'
    frozen, entry = make_release(root, family)
    marker = root/'must_not_run'
    replace_entry(frozen, entry, f'from pathlib import Path\nPath({str(marker)!r}).touch()\n')
    python, site = site_python
    (site/'test_preloaded_project.pth').write_text(
        f'import sys,types; sys.modules[{name!r}]=types.ModuleType({name!r})\n')
    process = subprocess.run([str(python), str(root/'scripts/run_immune_research.py'),
        '--variant', dispatch.MODEL_FAMILIES[family][0]], cwd=root, capture_output=True, text=True, timeout=30)
    assert process.returncode == 78 and 'Project modules were loaded' in process.stderr
    assert not marker.exists()


@pytest.mark.parametrize('change', [dict(IMMUNE_RELEASE_ROUTE='not_a_fixed_release'),
    dict(IMMUNE_RELEASE_GUARD_SHA256='0'*64)])
def test_guard_runtime_error_cannot_be_swallowed_by_site(staged, change):
    root = staged[0]
    family = 'r12_r19_20260930'
    frozen, _ = make_release(root, family)
    module = load_module(root/staging.DISPATCHER, 'guard_error_dispatcher')
    environment = {**module.guarded_environment(frozen, ['--variant', dispatch.MODEL_FAMILIES[family][0]]), **change}
    process = subprocess.run([sys.executable, '-c', 'print("TARGET_EXECUTED")'],
        cwd=frozen, env=environment, capture_output=True, text=True, timeout=15)
    assert process.returncode == 78 and 'Release startup rejected:' in process.stderr
    assert 'TARGET_EXECUTED' not in process.stdout


def test_fresh_bootstrap_rejects_missing_guard_and_changed_guard_source(staged):
    root = staged[0]
    family = 'r12_r19_20260930'
    make_release(root, family)
    argv = ['--variant', dispatch.MODEL_FAMILIES[family][0]]
    process = subprocess.run([sys.executable, '-S', str(root/staging.DISPATCHER), dispatch.GUARDED_LAUNCH, *argv],
        cwd=root, capture_output=True, text=True, timeout=15)
    assert process.returncode == 2 and 'did not complete' in process.stderr
    (root/dispatch.STARTUP_GUARD).write_text('raise RuntimeError("changed guard")\n')
    process = subprocess.run([sys.executable, '-S', str(root/staging.DISPATCHER), *argv],
        cwd=root, capture_output=True, text=True, timeout=15)
    assert process.returncode == 2 and 'Source digest differs' in process.stderr
