"""Artificial filesystem/job evidence only; no GPU, corpus, or real snapshot write.

Activate pllm: python -m pytest scripts/tests/immune_llada/test_archive_refined_release.py -q
The production profile/diagnostic and import interfaces are mocked for tiny trees;
the archive's queue, file SHA, original-marker and copied-tree checks are real.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.architecture_prototypes import archive_refined_release as archive


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False)+'\n')


@pytest.fixture
def tree(tmp_path,monkeypatch):
    root=tmp_path/'repository';pool=root/'output/pool'
    releases=root/'output/immune_research_releases'
    previous=releases/'r12_r19_20260930';destination=releases/archive.NAME
    request=archive.ArchiveRequest(root,pool,previous,destination)
    for name in (*archive.REQUIRED,'dllm/imported_extra.py','scripts/lazy_support.py'):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('{}\n' if path.suffix=='.json' else 'SOFTWARE_TEST_ONLY = True\n')
    old_names=['dllm/pipelines/llada/models/configuration_llada.py','dllm/imported_extra.py','scripts/run_immune_research.py']
    for name in old_names:
        path=previous/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((root/name).read_bytes())
    old_map={name:archive.sha(previous/name) for name in old_names}
    dump(previous/'release_manifest.json',dict(schema_version=1,source_root=str(root),frozen_root=str(previous),
        measured_source_sha256={'scripts/run_immune_research.py':old_map['scripts/run_immune_research.py']},copied_files_sha256=old_map))
    measured_names=[name for name in archive.REQUIRED if 'configuration_llada.py' not in name]
    measured={name:archive.sha(root/name) for name in measured_names}
    def inventory(source):
        assert source==root
        return dict(measured_source_sha256={name:archive.sha(root/name) for name in measured_names},
            imported_files_sha256={name:archive.sha(root/name) for name in (archive.ENTRY,'dllm/imported_extra.py')},
            external_versions={'software_test_only':'0'},scope='mock import inventory')
    monkeypatch.setattr(archive,'import_inventory',inventory)
    dump(pool/'pool.json',{'source_root':str(root),'software_test_only':True})
    for state in ('pending','running','results'):(pool/state).mkdir(exist_ok=True)
    jobs={}
    for index,variant in enumerate(archive.FAMILY):
        job_id=f'artificial_{index}_{variant}';jobs[variant]=job_id
        dump(pool/'results'/f'{job_id}.json',dict(job_id=job_id,state='succeeded',returncode=0,
            job=dict(variant=variant,mode='train',steps=500,data_dir='/invented/not_read'),software_test_only=True))
        run=pool/'runs'/job_id
        dump(run/'summary.json',dict(variant=variant,source_sha256=measured,software_test_only=True))
        dump(run/'COMPLETE.json',dict(status='completed',global_step=500,summary_sha256=archive.sha(run/'summary.json'),software_test_only=True))
        profile=pool/'profiles'/variant
        dump(profile/'summary.json',dict(variant=variant,software_test_only=True))
        dump(profile/'COMPLETE.json',dict(status='passed',source_sha256=measured,summary_sha256=archive.sha(profile/'summary.json'),software_test_only=True))
    monkeypatch.setattr(archive,'verify_original_profile',lambda pool,variant:archive.read_json(pool/'profiles'/variant/'summary.json'))
    monkeypatch.setattr(archive,'verify_original_diagnostic',lambda pool,variant:archive.read_json(pool/'runs'/jobs[variant]/'summary.json'))
    return SimpleNamespace(root=root,pool=pool,previous=previous,destination=destination,request=request,jobs=jobs,
        measured=measured,first=archive.FAMILY[0],run=pool/'runs'/jobs[archive.FAMILY[0]])


@pytest.mark.parametrize('state',['pending','running'])
def test_active_jobs_refuse_without_any_archive_write(tree,state):
    dump(tree.pool/state/'active.json',{'variant':'unrelated_variant','steps':1})
    plan=archive.archive_release(tree.request)
    assert not plan['ready'] and len(plan['active_jobs'])==1
    before={str(path) for path in tree.root.rglob('*')}
    with pytest.raises(ValueError,match='pending/running'):archive.archive_release(tree.request,apply=True)
    assert {str(path) for path in tree.root.rglob('*')}==before
    assert not tree.destination.exists()


def test_successful_snapshot_copies_full_support_and_preserves_originals(tree):
    report=archive.archive_release(tree.request)
    assert report['ready'] and len(report['copied_files_sha256'])>len(report['measured_source_sha256'])
    manifest=archive.archive_release(tree.request,apply=True)
    assert manifest==archive.verify_snapshot(tree.destination)
    assert manifest['terminal_job_ids']==tree.jobs and manifest['routing_changed'] is False
    assert 'scripts/lazy_support.py' in manifest['copied_files_sha256']
    assert manifest['dependency_audit']['unmeasured_imports']['dllm/imported_extra.py']['r12_status']=='same'
    assert not (tree.pool/'release_map.json').exists()
    original=tree.root/'scripts/lazy_support.py';copied=tree.destination/'scripts/lazy_support.py'
    assert original.stat().st_ino!=copied.stat().st_ino
    before=copied.read_bytes();original.write_text('changed after archive')
    assert copied.read_bytes()==before


@pytest.mark.parametrize('damage',['profile_sha','profile_source','missing_complete','diagnostic_sha','missing_result','duplicate_result','failed_result'])
def test_incomplete_or_drifted_evidence_refuses_archive(tree,damage):
    profile=tree.pool/'profiles'/tree.first
    if damage=='profile_sha':dump(profile/'summary.json',{'changed':True})
    elif damage=='profile_source':
        proof=archive.read_json(profile/'COMPLETE.json');proof['source_sha256']={};dump(profile/'COMPLETE.json',proof)
    elif damage=='missing_complete':(tree.run/'COMPLETE.json').unlink()
    elif damage=='diagnostic_sha':dump(tree.run/'summary.json',{'changed':True})
    elif damage=='missing_result':(tree.pool/'results'/f'{tree.jobs[tree.first]}.json').unlink()
    elif damage=='duplicate_result':
        value=archive.read_json(tree.pool/'results'/f'{tree.jobs[tree.first]}.json');value['job_id']='duplicate'
        dump(tree.pool/'results/duplicate.json',value)
    else:
        path=tree.pool/'results'/f'{tree.jobs[tree.first]}.json';value=archive.read_json(path);value['state']='failed';dump(path,value)
    assert not archive.archive_release(tree.request)['ready']
    with pytest.raises(ValueError):archive.archive_release(tree.request,apply=True)
    assert not tree.destination.exists()


@pytest.mark.parametrize('interface',['verify_original_profile','verify_original_diagnostic'])
def test_original_verifier_rejection_is_not_bypassed(tree,monkeypatch,interface):
    def reject(*a,**k):raise ValueError('original verifier rejected')
    monkeypatch.setattr(archive,interface,reject)
    with pytest.raises(ValueError,match='original verifier rejected'):archive.archive_release(tree.request,apply=True)
    assert not tree.destination.exists()


def test_unprofiled_import_changed_since_r12_is_explicit_blocker(tree):
    (tree.root/'dllm/imported_extra.py').write_text('changed unprofiled dependency')
    plan=archive.archive_release(tree.request)
    assert not plan['ready']
    assert plan['dependency_audit']['unmeasured_imports']['dllm/imported_extra.py']['r12_status']=='changed'
    with pytest.raises(ValueError,match='imported dependency'):archive.archive_release(tree.request,apply=True)


@pytest.mark.parametrize('damage',['profiled_source','old_snapshot','missing_runtime_dependency'])
def test_source_integrity_is_checked_before_copy(tree,damage):
    if damage=='profiled_source':(tree.root/archive.ENTRY).write_text('changed profile-bound entry')
    elif damage=='old_snapshot':(tree.previous/'dllm/imported_extra.py').write_text('broken old snapshot')
    else:(tree.root/'dllm/pipelines/llada/models/configuration_llada.py').unlink()
    with pytest.raises((ValueError,FileNotFoundError)):archive.archive_release(tree.request,apply=True)
    assert not tree.destination.exists()


@pytest.mark.parametrize('change',['queued_job','source','original_summary'])
def test_copy_time_drift_leaves_no_published_or_staging_snapshot(tree,monkeypatch,change):
    real_copy=archive.shutil.copyfile;changed=[]
    def copy(src,dst):
        result=real_copy(src,dst)
        if not changed:
            changed.append(True)
            if change=='queued_job':dump(tree.pool/'pending/new.json',{'variant':tree.first,'steps':500})
            elif change=='source':(tree.root/'scripts/lazy_support.py').write_text('changed during copy')
            else:dump(tree.run/'summary.json',{'changed_during_copy':True})
        return result
    monkeypatch.setattr(archive.shutil,'copyfile',copy)
    with pytest.raises(ValueError):archive.archive_release(tree.request,apply=True)
    assert not tree.destination.exists() and not list(tree.destination.parent.glob('.*.staging-*'))


def test_existing_destination_is_never_overwritten(tree):
    tree.destination.mkdir();sentinel=tree.destination/'keep';sentinel.write_text('existing')
    with pytest.raises(ValueError,match='never overwritten'):archive.archive_release(tree.request,apply=True)
    assert sentinel.read_text()=='existing'


def test_missing_archived_dependency_fails_verification(tree):
    archive.archive_release(tree.request,apply=True)
    (tree.destination/'dllm/imported_extra.py').unlink()
    with pytest.raises(ValueError,match='Missing'):archive.verify_snapshot(tree.destination)


def test_source_symlink_is_rejected(tree):
    path=tree.root/'scripts/lazy_support.py';path.unlink();path.symlink_to(tree.root/archive.ENTRY)
    with pytest.raises(ValueError,match='symlink'):archive.archive_release(tree.request)
