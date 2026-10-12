"""Software-only temporary parent archives and resource-path fixtures.

Activate pllm: python -m pytest scripts/tests/immune_llada/test_pilot_runtime_package.py -q
Fixture manifests identify software tests; no genuine archive/GPU/data certificate
is issued. No model construction, GPU, biological file, or live source is touched.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.architecture_prototypes import pilot_runtime_package as package
from scripts.architecture_prototypes.pilot_runtime_inspect import isolated_search_path,project_namespace_specs


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False)+'\n')


@pytest.fixture
def tree(tmp_path):
    original=tmp_path/'original'
    base=original/'output/immune_research_releases'/package.archive.NAME
    destination=original/'output/immune_pilot_runtimes'/package.NAME
    names=[package.archive.ENTRY,'dllm/model.py','scripts/architecture_prototypes/certified_pilot.py',
           'scripts/architecture_prototypes/validation_metrics.py',package.INSPECTOR,
           'scripts/architecture_prototypes/pilot_runtime_package.py']
    for name in names:
        for parent in (original,base):
            path=parent/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('SOFTWARE_TEST_ONLY = True\n')
    copied={name:package.sha(base/name) for name in names}
    evidence=original/'software_test_evidence.json';dump(evidence,{'software_test_only':True})
    dump(base/'release_manifest.json',dict(schema_version=1,source_root=str(original),frozen_root=str(base),
        family=list(package.archive.FAMILY),terminal_job_ids={name:'software_test_'+name for name in package.archive.FAMILY},
        archive_evidence_sha256={str(evidence):package.sha(evidence)},copied_files_sha256=copied,
        measured_source_sha256={name:copied[name] for name in (package.archive.ENTRY,'dllm/model.py')},
        software_test_only=True))
    for name in package.RESOURCE_ALIASES:(original/name).mkdir(parents=True)
    return SimpleNamespace(original=original,base=base,destination=destination,evidence=evidence)


def test_missing_completed_base_blocks_without_creating_runtime(tmp_path):
    base=tmp_path/package.archive.NAME;destination=tmp_path/'runtime'
    plan=package.plan_runtime(base,destination)
    assert not plan['ready'] and 'does not exist' in plan['blockers'][0]
    with pytest.raises(ValueError,match='blocked'):package.build_runtime(base,destination)
    assert not destination.exists()


def test_complete_parent_and_adapters_are_copied_byte_exact(tree):
    before=package.verify_base(tree.base)
    result=package.build_runtime(tree.base,tree.destination)
    assert package.verify_runtime(tree.destination)==result
    assert result['base_copied_files_sha256']==before['copied_files_sha256']
    assert result['measured_source_sha256']==before['measured_source_sha256']
    assert result['status']=='source_package_only_not_gpu_or_data_acceptance'
    for name in before['copied_files_sha256']:
        assert (tree.base/name).read_bytes()==(tree.destination/name).read_bytes()
        assert (tree.base/name).stat().st_ino!=(tree.destination/name).stat().st_ino


def test_live_registry_or_adapter_changes_cannot_replace_parent_bytes(tree):
    for name in (package.archive.ENTRY,'scripts/architecture_prototypes/validation_metrics.py'):
        (tree.original/name).write_text('LIVE_TREE_CHANGED = True\n')
    package.build_runtime(tree.base,tree.destination)
    for name in (package.archive.ENTRY,'scripts/architecture_prototypes/validation_metrics.py'):
        assert (tree.destination/name).read_bytes()==(tree.base/name).read_bytes()
        assert (tree.destination/name).read_bytes()!=(tree.original/name).read_bytes()


def test_fixed_new_addition_is_independently_bound(tree):
    name='scripts/architecture_prototypes/pilot_runtime_entry_v1.py'
    source=tree.original/name;source.write_text('SOFTWARE_TEST_NEW_ENTRY = True\n')
    result=package.build_runtime(tree.base,tree.destination,additions={name:source})
    assert result['additions_sha256']=={name:package.sha(source)}
    assert result['runtime_files_sha256'][name]==package.sha(tree.destination/name)
    assert not (tree.base/name).exists()


@pytest.mark.parametrize('kind',['unknown_name','existing_parent','wrong_source'])
def test_arbitrary_or_replacing_additions_are_rejected(tree,kind):
    name='scripts/architecture_prototypes/pilot_runtime_entry_v1.py'
    source=tree.original/name;source.write_text('SOFTWARE_TEST_NEW_ENTRY = True\n')
    if kind=='unknown_name':name='scripts/not_whitelisted.py'
    elif kind=='existing_parent':name=package.INSPECTOR;source=tree.original/name
    else:source=tree.original/package.archive.ENTRY
    with pytest.raises(ValueError):package.build_runtime(tree.base,tree.destination,additions={name:source})
    assert not tree.destination.exists()


def test_resource_aliases_resolve_to_original_certificate_paths_without_reading_content(tree):
    names=[package.RESOURCE_ALIASES[0]+'/tcr_binding_nm2025/artificial/test.csv',
           package.RESOURCE_ALIASES[1]+'/majority/test.csv']
    for name in names:
        path=tree.original/name;path.parent.mkdir(parents=True,exist_ok=True);path.touch()
    package.build_runtime(tree.base,tree.destination)
    assert all((tree.destination/name).resolve()==(tree.original/name).resolve() for name in names)
    observed=list((tree.destination/package.RESOURCE_ALIASES[0]/'tcr_binding_nm2025').rglob('test.csv'))
    assert len(observed)==1 and observed[0].resolve()==(tree.original/names[0]).resolve()


def test_missing_required_resource_directory_refuses_package(tree):
    (tree.original/package.RESOURCE_ALIASES[0]).rmdir()
    with pytest.raises(ValueError,match='Missing required external resource'):package.build_runtime(tree.base,tree.destination)
    assert not tree.destination.exists()


@pytest.mark.parametrize('damage',['source','manifest','original_evidence'])
def test_parent_drift_refuses_copy(tree,damage):
    if damage=='source':(tree.base/'dllm/model.py').write_text('changed parent')
    elif damage=='manifest':
        path=tree.base/'release_manifest.json';value=package.read_json(path);value['terminal_job_ids']={};dump(path,value)
    else:tree.evidence.write_text('changed original evidence')
    with pytest.raises(ValueError):package.build_runtime(tree.base,tree.destination)
    assert not tree.destination.exists()


@pytest.mark.parametrize('damage',['adapter_bytes','unlisted_code','extra_alias','unlisted_alias','redirect_alias','missing_alias','map_replacement','false_acceptance'])
def test_runtime_tampering_is_rejected(tree,damage):
    package.build_runtime(tree.base,tree.destination)
    manifest_path=tree.destination/package.MANIFEST
    if damage=='adapter_bytes':(tree.destination/'scripts/architecture_prototypes/validation_metrics.py').write_text('changed')
    elif damage=='unlisted_code':(tree.destination/'scripts/injected.py').write_text('unrecorded')
    elif damage=='extra_alias':
        value=package.read_json(manifest_path);value['external_resource_aliases']['unexpected']=str(tree.original);dump(manifest_path,value)
    elif damage=='unlisted_alias':(tree.destination/'undeclared_alias').symlink_to(tree.original,target_is_directory=True)
    elif damage=='false_acceptance':
        value=package.read_json(manifest_path);value['data_accepted']=True;dump(manifest_path,value)
    elif damage in ('redirect_alias','missing_alias'):
        alias=tree.destination/package.RESOURCE_ALIASES[0];alias.unlink()
        if damage=='redirect_alias':alias.symlink_to(tree.original,target_is_directory=True)
    else:
        value=package.read_json(manifest_path);value['measured_source_sha256']={};dump(manifest_path,value)
    with pytest.raises(ValueError):package.verify_runtime(tree.destination)


def test_copy_time_parent_drift_leaves_no_runtime_or_staging(tree,monkeypatch):
    original_copy=package.shutil.copyfile;done=[]
    def copy(src,dst):
        result=original_copy(src,dst)
        if not done:
            done.append(True);tree.evidence.write_text('changed during copy')
        return result
    monkeypatch.setattr(package.shutil,'copyfile',copy)
    with pytest.raises(ValueError):package.build_runtime(tree.base,tree.destination)
    assert not tree.destination.exists() and not list(tree.destination.parent.glob('.*.staging-*'))


def test_existing_runtime_is_never_overwritten(tree):
    package.build_runtime(tree.base,tree.destination)
    digest=package.sha(tree.destination/package.MANIFEST)
    with pytest.raises(ValueError,match='never overwrite'):package.build_runtime(tree.base,tree.destination)
    assert package.sha(tree.destination/package.MANIFEST)==digest


def test_inspection_cannot_accept_commands_or_training_modes(tmp_path):
    with pytest.raises(ValueError,match='Unknown inspection'):package.inspect_fresh(tmp_path,tmp_path,mode='train')
    with pytest.raises(ValueError,match='only the current'):package.inspect_fresh(tmp_path,tmp_path,python='/bin/sh')


def test_fresh_bootstrap_removes_live_pth_namespace_paths_and_keeps_runtime(tmp_path):
    original=tmp_path/'live';runtime=original/'output/runtime';external=tmp_path/'site-packages'
    paths=[str(original),str(original/'scripts'),str(runtime),str(external)]
    assert isolated_search_path(runtime,original,paths,[])==[str(runtime),str(external)]


def test_fresh_bootstrap_refuses_preloaded_project_modules(tmp_path):
    with pytest.raises(ValueError,match='fresh process'):
        isolated_search_path(tmp_path/'runtime',tmp_path/'live',[],['scripts.already_loaded'])


def test_project_namespace_locations_are_fixed_instead_of_merging_site_packages(tmp_path):
    from importlib.util import module_from_spec
    for name in ('scripts','examples'):(tmp_path/name).mkdir()
    specs=project_namespace_specs(tmp_path)
    assert set(specs)=={'scripts','examples'}
    for name,spec in specs.items():
        module=module_from_spec(spec)
        assert type(module.__path__) is list and module.__path__==[str(tmp_path/name)]
