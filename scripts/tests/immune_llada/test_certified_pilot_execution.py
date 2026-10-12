"""Artificial lifecycle interfaces and a tiny real CPU checkpoint continuation.

Activate pllm, then USE_TF=0 python -m pytest scripts/tests/immune_llada/test_certified_pilot_execution.py -q
Admission interfaces are explicitly mocked only for positive lifecycle tests.
No passing data/GPU certificate is written; mock gate files say not_acceptance.
No GPU, corpus, submission, or production entry point is exercised.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from scripts.architecture_prototypes import certified_pilot_execution as execution


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False)+'\n')


@pytest.fixture
def wiring(tmp_path,monkeypatch):
    pilot=execution.pilot
    sources=dict(model={'mock_model':'a'*64},adapter={'mock_adapter':'b'*64},execution={'mock_executor':'c'*64})
    p=dict(kind='mock_only_not_admitted',variant='r20_refined_control',source_sha256=deepcopy(sources['model']),
        adapter_source_sha256=deepcopy(sources['adapter']),max_steps=10000,pilot_stop_step=1000)
    request=pilot.PilotRequest('r20_refined_control',tmp_path/'invented_data',tmp_path/'selection.json',
        tmp_path/'pool/profiles/r20_refined_control',tmp_path/'first_run')
    release=tmp_path/'not_a_gpu_acceptance'
    for name in ('GPU_RELEASE_COMPLETE.json','gpu_release_summary.json'):
        dump(release/name,dict(mock_only=True,not_acceptance=True))
    events=[]
    def checked(req):
        events.append('mock_base_preflight')
        assert not req.output_dir.exists()
        provenance={**deepcopy(p),'pilot_stop_step':req.steps}
        completed=None if req.resume_from_checkpoint is None else json.loads((req.resume_from_checkpoint/'trainer_state.json').read_text())['global_step']
        return pilot.PilotPreflight(req,{}, {},{},provenance,completed)
    def diagnostic(pool,variant,steps,require_learning=False):
        events.append('mock_diagnostic500')
        assert steps==500 and require_learning and variant==request.variant
        return dict(mock_only=True,not_diagnostic_evidence=True)
    def gpu(directory,base,binding):
        events.append('mock_gpu_admission')
        assert Path(directory)==release and binding==sources
        return dict(directory=str(release),complete_sha256=pilot.base.sha(release/'GPU_RELEASE_COMPLETE.json'),
            summary_sha256=pilot.base.sha(release/'gpu_release_summary.json'),contract={'mock_only':True})
    monkeypatch.setattr(pilot,'preflight',checked)
    monkeypatch.setattr(pilot.profile_checks,'verify_diagnostic',diagnostic)
    monkeypatch.setattr(execution,'source_binding',lambda:deepcopy(sources))
    real_gpu_check=execution.verify_gpu_release
    monkeypatch.setattr(execution,'verify_gpu_release',gpu)
    trainers=[]
    class FakeTrainer:
        def __init__(self,req,base):
            self.request,self.base=req,base
            self.state=SimpleNamespace(global_step=0)
            self.callbacks=[];self.calls=[];self.step_delta=0;self.failure=None;self.mutate_sources=False
        def add_callback(self,callback):self.callbacks.append(callback)
        def evaluate(self):
            self.calls.append('evaluate')
            if self.failure=='evaluate':raise RuntimeError('injected evaluation failure')
            return {'eval_loss':1.,'eval_absent_group':None}
        def save_metrics(self,name,metrics):dump(self.request.output_dir/f'{name}_results.json',metrics)
        def train(self,resume_from_checkpoint=None):
            self.calls.append(('train',resume_from_checkpoint))
            if self.failure=='train':raise RuntimeError('injected training failure')
            self.state.global_step=self.request.steps+self.step_delta
            directory=self.request.output_dir/f'checkpoint-{self.state.global_step}';directory.mkdir()
            for filename in execution.CHECKPOINT_FILES:
                (directory/filename).write_text('mock-only checkpoint payload, not trained state')
            dump(directory/'run_provenance.json',self.base.provenance)
            dump(directory/'trainer_state.json',{'global_step':self.state.global_step})
            for callback in self.callbacks:
                callback.on_save(SimpleNamespace(output_dir=str(self.request.output_dir)),self.state,None)
            if self.mutate_sources:sources['execution']['mock_executor']='changed'
            return SimpleNamespace(metrics={'train_loss':1.})
        def save_model(self):
            self.calls.append('save_model')
            if self.failure=='save':raise RuntimeError('injected save failure')
            (self.request.output_dir/'model.safetensors').write_text('mock-only final weights')
        def save_state(self):dump(self.request.output_dir/'trainer_state.json',{'global_step':self.state.global_step})
    def build(req):
        events.append('mock_build')
        base=checked(req);trainer=FakeTrainer(req,base);trainers.append(trainer)
        return pilot.PilotTrainerBundle(base,trainer)
    monkeypatch.setattr(pilot,'build_trainer',build)
    return SimpleNamespace(request=request,release=release,events=events,sources=sources,trainers=trainers,
        checked=checked,build=build,gpu=gpu,real_gpu_check=real_gpu_check,tmp=tmp_path)


def test_real_data_gate_precedes_diagnostic_gpu_builder_and_output(tmp_path,monkeypatch):
    data=tmp_path/'uncertified';dump(data/'dataset_manifest.json',{'mock_only':True})
    def forbidden(*args,**kwargs):pytest.fail('Downstream work ran before the original data gate')
    monkeypatch.setattr(execution.pilot.profile_checks,'verify_diagnostic',forbidden)
    monkeypatch.setattr(execution,'verify_gpu_release',forbidden)
    monkeypatch.setattr(execution.pilot,'build_trainer',forbidden)
    req=execution.pilot.PilotRequest('r20_refined_control',data,tmp_path/'selection',tmp_path/'profile',tmp_path/'output')
    with pytest.raises(FileNotFoundError,match='audit_certificate.json'):execution.execute(req,tmp_path/'gpu')
    assert not req.output_dir.exists()


def test_mock_gate_files_are_rejected_by_actual_gpu_verifier(wiring):
    with pytest.raises(ValueError,match='independent GPU'):
        wiring.real_gpu_check(wiring.release,wiring.checked(wiring.request),wiring.sources)


@pytest.mark.parametrize('change',['missing_evaluation','short_shape','short_capacity','no_peak','high_peak','nan_loss','no_optimizer','wrong_masks'])
def test_phase_contract_rejects_missing_or_invalid_measurements(change):
    # Plain artificial metadata only: no release/certificate is created or passed.
    p={'micro_batch':2,'config':{'max_length':64,'full_capacity':16,'fragment_capacity':8}}
    common={'status':'completed','input_shape':[2,64],'target_capacities':{'full_capacity':16,'fragment_capacity':8},
            'peak_allocated_bytes':40,'total_memory_bytes':100}
    phases={'train_optimizer_step':{**deepcopy(common),'losses':[1.], 'optimizer_steps':1,'finite_gradients':True},
            'four_mask_evaluation':{**deepcopy(common),'losses':[1.]*4,'mask_levels':[.25,.5,.75,1.],
                                   'trainer':'validation_metrics.TokenValidationTrainer'}}
    training=phases['train_optimizer_step']
    if change=='missing_evaluation':del phases['four_mask_evaluation']
    elif change=='short_shape':training['input_shape']=[2,32]
    elif change=='short_capacity':training['target_capacities']['full_capacity']=8
    elif change=='no_peak':del training['peak_allocated_bytes']
    elif change=='high_peak':training['peak_allocated_bytes']=86
    elif change=='nan_loss':training['losses']=[float('nan')]
    elif change=='no_optimizer':training['optimizer_steps']=0
    else:phases['four_mask_evaluation']['mask_levels']=[.5,1.]
    with pytest.raises(ValueError):execution.verify_gpu_phases(phases,p)


@pytest.mark.parametrize('gate',['diagnostic','gpu'])
def test_admission_failure_never_builds_or_creates_output(wiring,monkeypatch,gate):
    def rejected(*args,**kwargs):raise ValueError('required gate rejected')
    if gate=='diagnostic':monkeypatch.setattr(execution.pilot.profile_checks,'verify_diagnostic',rejected)
    else:monkeypatch.setattr(execution,'verify_gpu_release',rejected)
    with pytest.raises(ValueError,match='required gate'):execution.execute(wiring.request,wiring.release)
    assert not wiring.request.output_dir.exists() and 'mock_build' not in wiring.events


def test_fresh_then_cumulative_resume_binds_saved_checkpoint_sidecar(wiring):
    first=execution.execute(wiring.request,wiring.release)
    assert first['global_step']==1000
    assert wiring.trainers[0].calls==['evaluate',('train',None),'evaluate','save_model']
    checkpoint=wiring.request.output_dir/'checkpoint-1000'
    continued=replace(wiring.request,steps=5000,resume_from_checkpoint=checkpoint,output_dir=wiring.tmp/'continued')
    result=execution.execute(continued,wiring.release)
    assert result['global_step']==5000
    assert wiring.trainers[1].calls==[('train',str(checkpoint)),'evaluate','save_model']
    assert not (continued.output_dir/'initial_eval_results.json').exists()
    assert (continued.output_dir/'checkpoint-5000/execution_provenance.json').is_file()
    assert not (continued.output_dir/'FAILED.json').exists()
    for name,digest in result['artifact_sha256'].items():assert execution.pilot.base.sha(continued.output_dir/name)==digest


@pytest.mark.parametrize('change',['missing_sidecar','changed_sidecar','changed_gpu_bytes','source_changed','missing_optimizer','changed_optimizer','changed_model'])
def test_resume_requires_matching_execution_and_checkpoint_evidence(wiring,change):
    execution.execute(wiring.request,wiring.release)
    checkpoint=wiring.request.output_dir/'checkpoint-1000'
    if change=='missing_sidecar':(checkpoint/'execution_provenance.json').unlink()
    elif change=='changed_sidecar':dump(checkpoint/'execution_provenance.json',{'mock_only':'corrupt'})
    elif change=='changed_gpu_bytes':(wiring.release/'gpu_release_summary.json').write_text('{"mock_only": "changed"}')
    elif change=='source_changed':wiring.sources['execution']['mock_executor']='d'*64
    elif change=='missing_optimizer':(checkpoint/'optimizer.pt').unlink()
    elif change=='changed_optimizer':(checkpoint/'optimizer.pt').write_text('different artificial optimizer bytes')
    else:(checkpoint/'model.safetensors').write_text('different artificial model bytes')
    req=replace(wiring.request,steps=5000,resume_from_checkpoint=checkpoint,output_dir=wiring.tmp/'resume_rejected')
    with pytest.raises((ValueError,FileNotFoundError)):execution.execute(req,wiring.release)
    assert not req.output_dir.exists()


@pytest.mark.parametrize('failure',['evaluate','train','save','step_before','step_after','source_drift'])
def test_failure_has_no_completion_marker_and_keeps_stage_evidence(wiring,monkeypatch,failure):
    def build(req):
        bundle=wiring.build(req)
        if failure in ('evaluate','train','save'):bundle.trainer.failure=failure
        elif failure=='step_before':bundle.trainer.step_delta=-1
        elif failure=='step_after':bundle.trainer.step_delta=1
        else:bundle.trainer.mutate_sources=True
        return bundle
    monkeypatch.setattr(execution.pilot,'build_trainer',build)
    with pytest.raises((ValueError,RuntimeError)):execution.execute(wiring.request,wiring.release)
    assert not (wiring.request.output_dir/'COMPLETE.json').exists()
    failed=json.loads((wiring.request.output_dir/'FAILED.json').read_text())
    assert failed['status']=='failed' and failed['stage']


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.linear=torch.nn.Linear(2,3)
    def forward(self,features,labels):
        logits=self.linear(features)
        return {'loss':torch.nn.functional.cross_entropy(logits,labels),'logits':logits}


class TinyProvenanceTrainer(Trainer):
    def _save(self,output_dir=None,state_dict=None):
        super()._save(output_dir,state_dict)
        directory=Path(output_dir or self.args.output_dir)
        for name,value in [('run_provenance.json',self.run_provenance),('research_config.json',{'mock_only':True}),
            ('immune_config.json',{'tiny_cpu_only':True}),('vocabulary.json',['integer_class_0','integer_class_1','integer_class_2'])]:
            dump(directory/name,value)


def test_real_tiny_cpu_training_saves_and_restores_execution_sidecar(wiring,monkeypatch):
    from safetensors.torch import load_file
    previous_threads=torch.get_num_threads();torch.set_num_threads(1)
    dataset=[{'features':torch.tensor([1.,float(index)]),'labels':torch.tensor(index%3)} for index in range(4)]
    restored=[]
    class CheckRestored(TrainerCallback):
        def __init__(self,expected):self.expected=expected
        def on_train_begin(self,args,state,control,model=None,**kwargs):
            assert all(torch.equal(model.state_dict()[key],value) for key,value in self.expected.items())
            restored.append(state.global_step)
    def build(req):
        checked=wiring.checked(req)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42 if req.resume_from_checkpoint is None else 999)
            model=TinyModel()
        settings=TrainingArguments(output_dir=str(req.output_dir),use_cpu=True,max_steps=10000,
            per_device_train_batch_size=2,per_device_eval_batch_size=2,learning_rate=1e-4,
            warmup_steps=100,lr_scheduler_type='cosine',optim='adamw_torch',save_steps=1000,save_total_limit=3,
            eval_strategy='no',save_strategy='steps',logging_strategy='no',report_to='none',
            disable_tqdm=True,dataloader_num_workers=0,dataloader_pin_memory=False)
        trainer=TinyProvenanceTrainer(model=model,args=settings,train_dataset=dataset,eval_dataset=dataset)
        trainer.run_provenance=checked.provenance
        trainer.add_callback(execution.pilot.PilotStop(req.steps))
        if req.resume_from_checkpoint is not None:
            trainer.add_callback(CheckRestored(load_file(str(req.resume_from_checkpoint/'model.safetensors'))))
        return execution.pilot.PilotTrainerBundle(checked,trainer)
    monkeypatch.setattr(execution.pilot,'build_trainer',build)
    try:
        first=execution.execute(wiring.request,wiring.release)
        checkpoint=wiring.request.output_dir/'checkpoint-1000'
        req=replace(wiring.request,steps=5000,resume_from_checkpoint=checkpoint,output_dir=wiring.tmp/'tiny_continued')
        last=execution.execute(req,wiring.release)
        assert first['global_step']==1000 and last['global_step']==5000 and restored==[1000]
        assert (req.output_dir/'checkpoint-5000/optimizer.pt').is_file()
    finally:
        torch.set_num_threads(previous_threads)
