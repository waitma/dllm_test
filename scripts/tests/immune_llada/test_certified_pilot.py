"""Temporary artificial data and interface mocks for the undeployed pilot adapter.

Activate pllm, then:
  USE_TF=0 python -m pytest scripts/tests/immune_llada/test_certified_pilot.py -q
The data gate is explicitly mocked only in wiring tests. Mock certificate files
contain no passing status; no real dataset is certified or trained, no GPU runs,
and no mock checkpoint is described as trained or semantically resumable.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.architecture_prototypes import certified_pilot as pilot
from scripts.architecture_prototypes import validation_selection as selection


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False)+'\n')


@pytest.fixture
def wiring(tmp_path, monkeypatch):
    data = tmp_path/'invented_prepared'
    manifest = dict(schema_version='immune_llada.semantic.v1', format='jsonl', dataset_version='mock_only',
        sources=['a','b'], splits={split:dict(records=0,shards=[]) for split in ('train','valid')})
    for split in ('train','valid'):
        for source, count in (('a',3),('b',2)):
            path=data/split/(source+'.jsonl');path.parent.mkdir(parents=True,exist_ok=True)
            rows=[dict(chains=['context','literal'],chain_roles=['antigen','tcr_beta'],source='mock/'+source,
                split=split,task_type='tcr',metadata={'v7_family_group':f'{split}/{source}/{i}'},identifiers={'id':str(i)}) for i in range(count)]
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            manifest['splits'][split]['shards'].append(dict(path=f'{split}/{source}.jsonl',source=source,split=split,
                records=count,sha256=pilot.base.sha(path)))
            manifest['splits'][split]['records']+=count
    dump(data/'dataset_manifest.json',manifest)
    for name in ('audit_certificate.json','downstream_scope_certificate.json'):
        dump(data/name,{'mock_only':True,'not_a_certificate':True})
    selected_path=tmp_path/'selection.json'
    selection.write_validation_selection(selected_path,data)
    variant='r20_refined_control';profile_dir=tmp_path/'pool'/'profiles'/variant
    sources={'mock_model_release.py':'a'*64}
    profile=dict(mock_only=True,variant=variant,config=asdict(pilot.ImmuneConfig(ablation='A4')),
        gpu='mock_A100',recommended_micro_batch=32,global_batch=256)
    dump(profile_dir/'summary.json',profile)
    dump(profile_dir/'COMPLETE.json',dict(mock_only=True,source_sha256=sources))
    events=[]
    def gate(directory):
        events.append('gate')
        assert directory==data.resolve()
        return json.loads((data/'dataset_manifest.json').read_text()), {'mock_only':True}
    def profile_check(pool,which):
        events.append('profile')
        assert pool==profile_dir.parent.parent and which==variant
        return deepcopy(profile)
    monkeypatch.setattr(pilot,'assert_ready',gate)
    monkeypatch.setattr(pilot.profile_checks,'verify_profile_evidence',profile_check)
    monkeypatch.setattr(pilot.refined,'fingerprints',lambda:dict(sources))
    monkeypatch.setattr(pilot,'protocol_sources',lambda:{'mock_adapter.py':'b'*64})
    monkeypatch.setattr(pilot,'metric_protocol',lambda:dict(schema='mock_metric_protocol',sha256='c'*64))
    request=pilot.PilotRequest(variant,data,selected_path,profile_dir,tmp_path/'future_run')
    return SimpleNamespace(request=request,data=data,manifest=manifest,profile=profile,sources=sources,events=events,tmp=tmp_path)


def checkpoint(wiring, provenance, step=1000, prior_stop=1000):
    directory=wiring.tmp/f'prior_{prior_stop}'/f'checkpoint-{step}'
    directory.mkdir(parents=True)
    saved=deepcopy(provenance);saved['pilot_stop_step']=prior_stop
    dump(directory/'run_provenance.json',saved)
    dump(directory/'trainer_state.json',dict(global_step=step,mock_only=True))
    dump(directory/'immune_config.json',saved['config'])
    variant=saved['variant']
    dump(directory/'research_config.json',dict(variant=variant,parent_variant=pilot.BASE_VARIANTS[variant],
        options=pilot.VARIANTS[variant],base=saved['config'],schema='immune_architecture.refined.v1'))
    tokenizer=pilot.make_tokenizer('A4')
    dump(directory/'vocabulary.json',[tokenizer.token(i) for i in range(tokenizer.vocab_size)])
    for name in ('model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth'):
        (directory/name).write_text('mock-only placeholder; not a trained checkpoint')
    return directory


def test_actual_gate_rejects_without_gpu_profile_selection_or_output(tmp_path,monkeypatch):
    data=tmp_path/'uncertified';dump(data/'dataset_manifest.json',{'mock_only':True})
    def forbidden(*args,**kwargs):pytest.fail('A downstream side effect was reached before data certification')
    monkeypatch.setattr(pilot.profile_checks,'verify_profile_evidence',forbidden)
    monkeypatch.setattr(pilot.selections,'load_verified_validation_selection',forbidden)
    monkeypatch.setattr(pilot,'runtime_gpu_name',forbidden)
    monkeypatch.setattr(pilot,'make_model',forbidden)
    req=pilot.PilotRequest('r20_refined_control',data,tmp_path/'missing_selection',tmp_path/'missing_profile',tmp_path/'not_created')
    with pytest.raises(FileNotFoundError,match='audit_certificate.json'):
        pilot.build_trainer(req)
    assert not req.output_dir.exists()


def test_pending_data_certificate_is_rejected_by_real_gate(tmp_path):
    data=tmp_path/'pending';dump(data/'dataset_manifest.json',{'mock_only':True})
    dump(data/'audit_certificate.json',{'status':'pending','pending':['unresolved'],'mock_only':True})
    req=pilot.PilotRequest('r20_refined_control',data,tmp_path/'selection',tmp_path/'profile',tmp_path/'output')
    with pytest.raises(ValueError,match='audit is incomplete'):
        pilot.preflight(req)
    assert not req.output_dir.exists()


def test_gate_failure_always_precedes_even_request_validation(wiring,monkeypatch):
    def rejected(directory):raise RuntimeError('mock gate rejection first')
    monkeypatch.setattr(pilot,'assert_ready',rejected)
    with pytest.raises(RuntimeError,match='gate rejection first'):
        pilot.preflight(replace(wiring.request,variant='invalid',steps=5000))


def test_mocked_preflight_binds_exact_selection_release_protocol_and_no_writes(wiring):
    checked=pilot.preflight(wiring.request)
    assert wiring.events==['gate','profile']
    assert not wiring.request.output_dir.exists()
    p=checked.provenance
    assert p['selection_sha256']==checked.selection['selection_sha256']
    assert p['validation_indices']=={s:v['source_local_indices'] for s,v in checked.selection['sources'].items()}
    assert p['source_sha256']==wiring.sources
    assert p['adapter_source_sha256']=={'mock_adapter.py':'b'*64}
    assert p['micro_batch']*p['grad_accum']==p['global_batch']==256
    assert p['max_steps']==10000 and p['pilot_stop_step']==1000
    assert p['learning_rate']==1e-4 and p['warmup_steps']==100 and p['scheduler']=='cosine'
    assert p['save_total_limit']==3 and p['save_only_model'] is False
    assert p['runtime_versions']['transformers']
    assert 'not yet GPU released' in p['profile_scope']
    assert json.loads((wiring.data/'audit_certificate.json').read_text())=={'mock_only':True,'not_a_certificate':True}


def test_exact_subset_matches_saved_indices_without_resampling(wiring):
    checked=pilot.preflight(wiring.request)
    subsets=pilot.exact_validation_subsets(wiring.data,checked.selection)
    for source,subset in subsets.items():
        expected=checked.selection['sources'][source]
        assert subset.indices==expected['source_local_indices']
        assert [record.metadata['v7_family_group'] for record in subset]==[row['family_group'] for row in expected['rows']]


def test_training_settings_keep_objective_budget_and_use_metric_requirements(wiring,monkeypatch):
    checked=pilot.preflight(wiring.request)
    monkeypatch.setattr(pilot,'TrainingArguments',lambda **kwargs:kwargs)
    args=pilot.training_arguments(wiring.request,checked.provenance)
    assert args['per_device_train_batch_size']==32 and args['gradient_accumulation_steps']==8
    assert args['per_device_eval_batch_size']==16
    assert args['max_steps']==10000 and args['optim']=='adamw_torch'
    assert args['bf16'] and args['disable_tqdm'] and not args['dataloader_persistent_workers']
    assert args['save_total_limit']==3 and not args['save_only_model']


def test_real_training_arguments_constructor_on_cpu(wiring,monkeypatch):
    checked=pilot.preflight(wiring.request)
    real_constructor=pilot.TrainingArguments
    # CPU construction only: preserve production kwargs and add the device override.
    monkeypatch.setattr(pilot,'TrainingArguments',lambda **kwargs:real_constructor(**kwargs,use_cpu=True))
    args=pilot.training_arguments(wiring.request,checked.provenance)
    assert args.device.type=='cpu' and args.world_size==1
    assert args.max_steps==10000 and args.save_total_limit==3
    assert args.bf16 and args.disable_tqdm and not args.dataloader_persistent_workers
    assert not wiring.request.output_dir.exists()


def test_metric_protocol_tracks_dynamic_module_constant_and_bytes(monkeypatch):
    protocol=pilot.metric_protocol()
    assert protocol['descriptor']==pilot.metrics.METRIC_PROTOCOL
    assert protocol['source_sha256']==pilot.base.sha(Path(pilot.metrics.__file__))
    monkeypatch.setattr(pilot.metrics,'METRIC_PROTOCOL','temporary_test_protocol')
    assert pilot.metric_protocol()['descriptor']=='temporary_test_protocol'


def test_factory_wires_token_trainer_and_callback_but_never_trains(wiring,monkeypatch):
    checked_model=SimpleNamespace(variant=wiring.request.variant,config=pilot.ImmuneConfig(),options=pilot.VARIANTS[wiring.request.variant])
    monkeypatch.setattr(pilot,'runtime_gpu_name',lambda:'mock_A100')
    monkeypatch.setattr(pilot,'make_model',lambda request:checked_model)
    monkeypatch.setattr(pilot,'training_arguments',lambda request,p:SimpleNamespace(world_size=1))
    class MockTokenTrainer:
        def __init__(self,**kwargs):self.kwargs=kwargs;self.callbacks=[]
        def add_callback(self,callback):self.callbacks.append(callback)
        def train(self,*args,**kwargs):pytest.fail('Factory must never train')
        def evaluate(self,*args,**kwargs):pytest.fail('Factory must never evaluate')
    monkeypatch.setattr(pilot.metrics,'TokenValidationTrainer',MockTokenTrainer)
    bundle=pilot.build_trainer(wiring.request)
    assert bundle.training_started is False and bundle.trainer.kwargs['model'] is checked_model
    assert len(bundle.trainer.kwargs['train_dataset'])==5
    assert bundle.trainer.run_provenance==bundle.preflight.provenance
    assert bundle.trainer.callbacks[0].step==1000
    assert not wiring.request.output_dir.exists()


@pytest.mark.parametrize('variant', ['r20_refined_control','r27_refined_value_residual'])
def test_actual_bridge_factory_rng_checkpointing_and_sampler_on_cpu(tmp_path,monkeypatch,variant):
    from dllm.pipelines.immune_llada.research_refined.model import RefinedModel
    config=pilot.ImmuneConfig(d_model=32,n_layers=2,n_heads=4,mlp_hidden_size=64,
        full_capacity=24,fragment_capacity=16,max_length=256)
    monkeypatch.setattr(pilot.base,'config_for',lambda args:config)
    original_to=torch.nn.Module.to
    moved=[]
    def cpu_only(model,device,*args,**kwargs):
        moved.append(device)
        assert device=='cuda'
        return original_to(model,'cpu',*args,**kwargs)
    monkeypatch.setattr(torch.nn.Module,'to',cpu_only)
    with torch.random.fork_rng(devices=[]):
        model=pilot.make_model(SimpleNamespace(variant=variant))
        assert isinstance(model,RefinedModel) and model.options==pilot.VARIANTS[variant]
        assert model.backbone.activation_checkpointing_strategy==pilot.base.ActivationCheckpointingStrategy.whole_layer
        assert torch.equal(torch.rand(5),torch.rand(5,generator=torch.Generator().manual_seed(42)))
    assert moved==['cuda']
    monkeypatch.setattr(torch.nn.Module,'to',original_to)
    args=pilot.TrainingArguments(output_dir=str(tmp_path/'trainer'),use_cpu=True,
        data_seed=42,report_to='none',disable_tqdm=True)
    trainer=pilot.metrics.TokenValidationTrainer(model=model,args=args,train_dataset=list(range(13)))
    sampler=trainer._get_train_sampler()
    assert isinstance(sampler,pilot.base.EpochRandomSampler)
    assert list(sampler)==torch.randperm(13,generator=torch.Generator().manual_seed(42)).tolist()


@pytest.mark.parametrize('field', ['manifest','selection','audit_certificate','scope_certificate','profile_summary','profile_complete','protocol','source','versions'])
def test_resume_rejects_any_provenance_change(wiring,monkeypatch,field):
    first=pilot.preflight(wiring.request)
    saved=checkpoint(wiring,first.provenance)
    paths={'manifest':wiring.data/'dataset_manifest.json','selection':wiring.request.selection_path,
        'audit_certificate':wiring.data/'audit_certificate.json','scope_certificate':wiring.data/'downstream_scope_certificate.json',
        'profile_summary':wiring.request.profile_dir/'summary.json','profile_complete':wiring.request.profile_dir/'COMPLETE.json'}
    if field in paths:
        path=paths[field];path.write_text(path.read_text()+' ')
    elif field=='protocol':monkeypatch.setattr(pilot,'metric_protocol',lambda:dict(schema='new_protocol'))
    elif field=='source':wiring.sources['new_model.py']='d'*64
    else:monkeypatch.setattr(pilot,'software_versions',lambda:{'transformers':'other'})
    with pytest.raises(ValueError):
        pilot.preflight(replace(wiring.request,steps=5000,resume_from_checkpoint=saved))


@pytest.mark.parametrize('prior_stop,completed,new_stop', [(1000,1000,5000),(1000,500,1000),(5000,2500,5000)])
def test_complete_stage_and_interrupted_stage_resume_metadata(wiring,prior_stop,completed,new_stop):
    first=pilot.preflight(wiring.request)
    saved=checkpoint(wiring,first.provenance,step=completed,prior_stop=prior_stop)
    continued=pilot.preflight(replace(wiring.request,steps=new_stop,resume_from_checkpoint=saved))
    assert continued.completed_checkpoint_step==completed
    assert continued.provenance['max_steps']==10000
    assert continued.provenance['pilot_stop_step']==new_stop
    a={k:v for k,v in first.provenance.items() if k!='pilot_stop_step'}
    b={k:v for k,v in continued.provenance.items() if k!='pilot_stop_step'}
    assert a==b


@pytest.mark.parametrize('prior_stop,completed,new_stop', [(1000,500,5000),(1000,1000,1000),(5000,5000,5000),(5000,500,1000)])
def test_invalid_cumulative_resume_rejected(wiring,prior_stop,completed,new_stop):
    initial=pilot.preflight(wiring.request)
    saved=checkpoint(wiring,initial.provenance,step=completed,prior_stop=prior_stop)
    with pytest.raises(ValueError):pilot.preflight(replace(wiring.request,steps=new_stop,resume_from_checkpoint=saved))


@pytest.mark.parametrize('filename', ['optimizer.pt','scheduler.pt','rng_state.pth','model.safetensors','immune_config.json','vocabulary.json'])
def test_incomplete_resumable_checkpoint_rejected(wiring,filename):
    first=pilot.preflight(wiring.request);saved=checkpoint(wiring,first.provenance)
    (saved/filename).unlink()
    with pytest.raises(ValueError,match='Missing|Incomplete'):
        pilot.preflight(replace(wiring.request,steps=5000,resume_from_checkpoint=saved))


@pytest.mark.parametrize('field,value', [('global_step',1000.5),('global_step','1000'),('pilot_stop_step',1000.0)])
def test_resume_does_not_coerce_malformed_step_metadata(wiring,field,value):
    initial=pilot.preflight(wiring.request);saved=checkpoint(wiring,initial.provenance)
    path=saved/('trainer_state.json' if field=='global_step' else 'run_provenance.json')
    content=json.loads(path.read_text());content[field]=value;dump(path,content)
    with pytest.raises(ValueError,match='exact integers'):
        pilot.preflight(replace(wiring.request,steps=5000,resume_from_checkpoint=saved))


@pytest.mark.parametrize('filename,mutation', [('immune_config.json','config'),('vocabulary.json','vocabulary'),('research_config.json','schema')])
def test_saved_checkpoint_model_contract_rejected(wiring,filename,mutation):
    first=pilot.preflight(wiring.request);saved=checkpoint(wiring,first.provenance)
    content=json.loads((saved/filename).read_text())
    if mutation=='config':content['n_layers']+=1
    elif mutation=='vocabulary':content.reverse()
    else:content['schema']='incorrect'
    dump(saved/filename,content)
    with pytest.raises(ValueError,match='configuration|vocabulary|architecture'):
        pilot.preflight(replace(wiring.request,steps=5000,resume_from_checkpoint=saved))


@pytest.mark.parametrize('change', ['invalid_variant','step500','fresh5000','existing_output','wrong_profile_path','world2','empty_source','wrong_selection_seed'])
def test_request_preflight_rejections_after_data_gate(wiring,monkeypatch,change):
    request=wiring.request
    if change=='invalid_variant':request=replace(request,variant='r00_a4_control')
    elif change=='step500':request=replace(request,steps=500)
    elif change=='fresh5000':request=replace(request,steps=5000)
    elif change=='existing_output':request.output_dir.mkdir()
    elif change=='wrong_profile_path':request=replace(request,profile_dir=request.profile_dir.parent/'elsewhere')
    elif change=='world2':monkeypatch.setenv('WORLD_SIZE','2')
    elif change=='empty_source':
        actual=pilot.selections.load_verified_validation_selection
        def altered(*args,**kwargs):
            selected=actual(*args,**kwargs);selected['sources']['a']['selected']['records']=0;return selected
        monkeypatch.setattr(pilot.selections,'load_verified_validation_selection',altered)
    else:
        request.selection_path.unlink();selection.write_validation_selection(request.selection_path,wiring.data,seed=7)
    with pytest.raises(ValueError):pilot.preflight(request)
    assert wiring.events[0]=='gate'


def test_runtime_gpu_mismatch_is_after_gates_before_model(wiring,monkeypatch):
    monkeypatch.setattr(pilot,'runtime_gpu_name',lambda:'different_GPU')
    monkeypatch.setattr(pilot,'make_model',lambda request:pytest.fail('Model must not be built'))
    with pytest.raises(ValueError,match='Runtime GPU differs'):pilot.build_trainer(wiring.request)
    assert wiring.events==['gate','profile']
