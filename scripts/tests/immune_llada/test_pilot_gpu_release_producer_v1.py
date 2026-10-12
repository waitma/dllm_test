"""Artificial CPU tests only; activate pllm and run pytest -q on this file.

Tiny CPU model/real Trainer and in-memory contract checks cannot issue GPU proof.
No runtime is built, GPU started, production evidence forged, or dataset read.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json
import importlib.machinery
import importlib.util
import sys

WORK_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(WORK_ROOT))
for name in ('scripts','examples'):
    spec = importlib.machinery.ModuleSpec(name,loader=None,is_package=True)
    spec.submodule_search_locations = [str(WORK_ROOT/name)]
    sys.modules[name] = importlib.util.module_from_spec(spec)

import pytest
import torch
from transformers import TrainingArguments

from scripts.architecture_prototypes import pilot_gpu_release_producer_v1 as producer
from scripts import run_immune_research as base
from dllm.pipelines.immune_llada.research_refined.model import RefinedModel
from dllm.pipelines.immune_llada.v3.model import ImmuneConfig


@pytest.fixture(autouse=True)
def cpu_only():
    assert producer.ROOT == WORK_ROOT and base.ROOT == WORK_ROOT
    assert not torch.cuda.is_initialized()
    prior = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(prior)
    assert not torch.cuda.is_initialized()


def config():
    return ImmuneConfig(d_model=16,n_layers=2,n_heads=2,mlp_hidden_size=32,
                       max_length=64,full_capacity=8,fragment_capacity=4)


def settings(path,micro,ga,oracle=False):
    return TrainingArguments(output_dir=str(path),use_cpu=True,
        per_device_train_batch_size=micro,per_device_eval_batch_size=2,gradient_accumulation_steps=ga,
        max_steps=10000,learning_rate=1e-4,warmup_steps=100,lr_scheduler_type='cosine',optim='adamw_torch',
        max_grad_norm=0. if oracle else 1.,seed=42,data_seed=42,bf16=False,remove_unused_columns=False,
        label_names=['labels'],eval_strategy='no',eval_steps=100,save_strategy='no',save_steps=100,
        save_total_limit=3,save_only_model=False,logging_steps=10,report_to='none',prediction_loss_only=True,
        dataloader_num_workers=0,dataloader_persistent_workers=False,disable_tqdm=True)


def make_model():
    torch.manual_seed(42)
    model = RefinedModel(config(),producer.VARIANT)
    from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy
    model.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
    torch.manual_seed(42)
    return model


def test_production_entry_rejects_live_root_before_import_or_output(tmp_path):
    with pytest.raises(ValueError,match='fixed derived runtime'):
        producer.produce(tmp_path/'profiles'/producer.VARIANT,tmp_path/'must_not_exist')
    assert not (tmp_path/'must_not_exist').exists()


def test_artificial_full_shape_both_capacities_padding_and_relation():
    cfg=config();collate=base.collator_for(cfg)
    a=producer.artificial_records(cfg,3);b=producer.artificial_records(cfg,3)
    assert [r.to_dict() for r in a] == [r.to_dict() for r in b]
    full=collate(a); observed=producer.observed_batch(full)
    assert observed['input_shape']==[3,64]
    assert observed['target_capacities']==dict(full_capacity=8,fragment_capacity=4)
    assert observed['padding_tokens']==0
    mixed=collate(producer.artificial_records(cfg,7,stress=False))
    assert (~mixed['attention_mask']).any()
    relation=mixed['eligible_mask'] & ~(mixed['residue_mask']|mixed['terminal_mask'])
    assert relation.any() and not (relation & mixed['target_slot_mask']).any()
    assert not (full['eligible_mask'] & ~(full['residue_mask']|full['terminal_mask'])).any()
    with pytest.raises(ValueError,match='both capacities'):
        producer.artificial_records(ImmuneConfig(max_length=12,full_capacity=8,fragment_capacity=4),1)


@pytest.mark.parametrize('ga',[8])
def test_real_hf_complete_window_gradient_oracle_and_three_adamw_updates(tmp_path,ga):
    cfg=config();collate=base.collator_for(cfg);batch=8;micro=batch//ga
    rows=producer.artificial_records(cfg,24)
    telemetry_plan = producer.fixed_plan(SimpleNamespace(profile={
        'config':asdict(ImmuneConfig()),'recommended_micro_batch':32}))
    validation={'procedural_mixed':producer.artificial_records(cfg,7,stress=False,offset=100),
                'full_capacity':producer.artificial_records(cfg,3,offset=200)}
    reference=producer.exercise(make_model(),settings(tmp_path/'oracle',micro,ga,True),rows,validation,
        collate,optimizer_steps=1,global_batch=8,oracle=True,
        progress_path=tmp_path/'oracle/producer_progress.json',progress_plan=telemetry_plan,progress_stage='gradient_oracle')
    assert reference['oracle']['passed'] and reference['oracle']['records']==8
    assert reference['oracle']['parameters'] and reference['adamw']['steps']==[1]
    # Original warmup starts at LR zero: a first window need not change weights.
    assert not any(reference['parameter_max_abs_changes'].values())
    gradients=reference.pop('_reference_gradients')
    observed=producer.exercise(make_model(),settings(tmp_path/'formal',micro,ga),rows,validation,
        collate,optimizer_steps=3,global_batch=8,reference_gradients=gradients,
        progress_path=tmp_path/'formal/producer_progress.json',progress_plan=telemetry_plan)
    assert [w['records'] for w in observed['windows']]==[8,8,8]
    assert [w['microsteps'] for w in observed['windows']]==[ga]*3
    assert reference['batches']==observed['batches'][:ga]
    assert observed['adamw']['steps']==[3] and observed['scheduler_last_epoch']==3
    assert observed['scheduler_horizon']==10000 and any(observed['parameter_max_abs_changes'].values())
    assert observed['first_window_preclip_match']['passed']
    assert observed['first_window_preclip_match']['gradient_scale']==pytest.approx(1.,abs=producer.GRAD_RTOL)
    assert reference['oracle']['rng_replay_equal']
    assert reference['oracle']['method'].startswith('independent_manual_corruption_logsoftmax')
    assert observed['windows'][0]['parameter_delta_l2']==0
    assert any(window['parameter_delta_l2']>0 for window in observed['windows'][1:])
    assert observed['adamw']['state_bytes']>0 and observed['adamw']['state_tensors']>0
    assert observed['memory']=={'cuda_measured':False}
    # These are tiny CPU telemetry observations, never GPU proof or a raw completed report.
    oracle_stage=json.loads((tmp_path/'oracle'/producer.STAGE_EVIDENCE).read_text())
    formal_stage=json.loads((tmp_path/'formal'/producer.STAGE_EVIDENCE).read_text())
    assert oracle_stage['gpu_accepted'] is formal_stage['gpu_accepted'] is False
    assert oracle_stage['phases']['gradient_oracle']['optimizer_steps']==1
    assert oracle_stage['phases']['gradient_oracle']['oracle']['gradient_scale']==pytest.approx(1.)
    assert formal_stage['phases']['train_optimizer_step']['optimizer_steps']==3
    assert formal_stage['phases']['train_optimizer_step']['records']==24
    assert formal_stage['phases']['four_mask_evaluation']['rng_preserved']
    evaluation=observed['evaluation']
    assert evaluation['rng_preserved'] and evaluation['rng_before_sha256']==evaluation['rng_after_sha256']
    assert evaluation['cpu_recount_oracle']['counts_exact']
    assert evaluation['callback_counts']==dict(prediction_steps=24,logs=1,evaluations=1)
    assert len(evaluation['raw_totals'])==8 and [2,64] in evaluation['actual_embedding_input_shapes']
    assert evaluation['metrics']['eval_mask100_full_capacity_relation_accuracy'] is None
    assert evaluation['metrics']['eval_records_per_mask']==10
    assert not list(tmp_path.rglob('GPU_RELEASE_COMPLETE.json'))
    assert not list(tmp_path.rglob('checkpoint-*'))
    producer.write_json(tmp_path/'cpu_case.json',dict(kind='artificial_cpu_interface_only',gpu_acceptance=False,
        ga=ga,global_batch=8,oracle_parameter_count=len(reference['oracle']['parameters']),
        oracle_max_abs_error=max(v['max_abs_error'] for v in reference['oracle']['parameters'].values()),
        first_window_inputs_match=reference['batches']==observed['batches'][:ga],
        optimizer_steps=observed['optimizer_steps'],adamw=observed['adamw'],
        scheduler_horizon=observed['scheduler_horizon'],scheduler_last_epoch=observed['scheduler_last_epoch'],
        any_parameter_changed=any(observed['parameter_max_abs_changes'].values()),
        rng_preserved=evaluation['rng_preserved'],source_records=evaluation['source_records'],
        actual_full_eval_shape=[2,64] in evaluation['actual_embedding_input_shapes']))


def test_incomplete_window_rejected_before_training(tmp_path):
    cfg=config()
    with pytest.raises(ValueError,match='full finite windows'):
        producer.exercise(make_model(),settings(tmp_path,4,2),producer.artificial_records(cfg,7),{},
            base.collator_for(cfg),optimizer_steps=1,global_batch=8)


def test_oracle_detects_native_ga_sum_regression(tmp_path,monkeypatch):
    from scripts.architecture_prototypes import validation_metrics as metrics
    original=metrics.TokenValidationTrainer.__init__
    def broken(self,*args,**kwargs):
        original(self,*args,**kwargs)
        self.model_accepts_loss_kwargs=True
    monkeypatch.setattr(metrics.TokenValidationTrainer,'__init__',broken)
    cfg=config()
    # The producer refuses an invalid scaling contract before executing train.
    with pytest.raises(ValueError,match='accumulation/runtime'):
        producer.exercise(make_model(),settings(tmp_path,2,4,True),producer.artificial_records(cfg,8),{},
            base.collator_for(cfg),optimizer_steps=1,global_batch=8,oracle=True)


def test_nonfinite_training_loss_rejected_without_complete(tmp_path):
    model=make_model()
    with torch.no_grad():
        next(model.parameters()).fill_(float('nan'))
    cfg=config()
    with pytest.raises(ValueError,match='Nonfinite training loss'):
        producer.exercise(model,settings(tmp_path,4,2,True),producer.artificial_records(cfg,8),{},
            base.collator_for(cfg),optimizer_steps=1,global_batch=8,oracle=True)
    assert not list(tmp_path.rglob('GPU_RELEASE_COMPLETE.json'))


@pytest.mark.parametrize('mutation',[{'cuda_execution':False},{'status':'failed'},{'sources':{'unexpected':1}},
                                  {'plan':{'variant':'r21'}}])
def test_worker_identity_or_non_gpu_report_rejected(mutation):
    plan={'variant':producer.VARIANT};sources={'runtime':'artificial_interface_only'}
    report=dict(schema=producer.SCHEMA,status='completed',cuda_execution=True,plan=plan,sources=sources)
    report.update(mutation)
    with pytest.raises(ValueError,match='identity/source/CUDA'):
        producer.validate_worker_report(report,plan,sources)


def test_evaluation_raw_population_corruption_rejected():
    from scripts.architecture_prototypes import validation_metrics as metrics
    cfg=config();collate=base.collator_for(cfg)
    datasets={'example':producer.artificial_records(cfg,1)}
    raw={f'eval_mask{int(r*100)}_example':[0]*len(metrics.FIELDS) for r in producer.RATIOS}
    with pytest.raises(ValueError,match='population differs'):
        producer.verify_evaluation(raw,{},datasets,collate,metrics,base)
    with pytest.raises(ValueError,match='Missing source/mask'):
        producer.verify_evaluation({}, {},datasets,collate,metrics,base)


def test_fixed_plan_is_r20_global256_three_steps_and_no_data_paths():
    context=SimpleNamespace(profile={'config':asdict(ImmuneConfig()),'recommended_micro_batch':32})
    plan=producer.fixed_plan(context)
    assert plan['variant']==producer.VARIANT and plan['global_batch']==256 and plan['grad_accum']==8
    assert plan['optimizer_steps']==3 and plan['scheduler_horizon']==10000
    assert producer.NAME=='r20_r27_pilot_v3_20261001'
    assert plan['timeout_seconds']==3600
    assert plan['phase_seconds']==dict(gradient_oracle=1200,train_optimizer_step=1800,four_mask_evaluation=600)
    tiny=plan['lifetime_preflight']
    assert tiny['timeout_seconds']==120 and tiny['counts_toward_formal'] is False
    assert [tiny[name] for name in ('micro_batch','grad_accum','global_batch','optimizer_steps')]==[1,8,8,1]
    assert [tiny['config'][name] for name in ('d_model','n_layers','n_heads','mlp_hidden_size','max_length','full_capacity','fragment_capacity')]==[16,2,2,32,64,8,4]
    assert plan['gradient_tolerance']==dict(atol=2e-5,rtol=.005,relative_l2=.005,scale=.005)
    assert plan['loss_tolerance']==dict(atol=2e-5,rtol=2e-5)
    assert plan['total_tolerance']==dict(atol=2e-4,rtol=2e-6)
    assert 'data_dir' not in plan and 'resume_from_checkpoint' not in plan
    for invalid in (3,16,64,True):
        context.profile['recommended_micro_batch']=invalid
        with pytest.raises(ValueError,match='microbatch'):
            producer.fixed_plan(context)


def packet_case():
    from scripts.architecture_prototypes import validation_metrics as metrics
    model=make_model();collate=base.collator_for(config())
    batch=collate(producer.artificial_records(config(),3,stress=False))
    logits=torch.zeros(*batch['input_ids'].shape,model.tokenizer.vocab_size)
    logits.scatter_(2,batch['input_ids'].unsqueeze(-1),3.)
    logits=logits.masked_fill(batch['target_slot_mask'].unsqueeze(-1)&~model.allowed_target[None,None],float('-inf'))
    model.denoise=lambda **kwargs:(logits,None)
    return model,batch,logits,metrics


@pytest.mark.parametrize('ratio',producer.RATIOS)
def test_compact_packet_independently_recounts_all_sixteen_fields(ratio):
    model,batch,logits,metrics=packet_case()
    native=metrics.selected_token_metrics(model,batch,ratio)['sample_totals'].sum(0).tolist()
    packet=producer.evaluation_packet(batch,logits,ratio)
    independent=producer.recount_packet(batch,packet,ratio,model.terminal_id,
        [model.tokenizer.special_id('<binding>'),model.tokenizer.special_id('<nonbinding>')],model.tokenizer.vocab_size)
    assert len(native)==len(independent)==16
    assert independent==pytest.approx(native,rel=producer.TOTAL_RTOL,abs=producer.TOTAL_ATOL)
    assert producer.cpu_summary(native,metrics)==metrics.summarize(native)
    assert packet['logits_shape']==list(logits.shape)
    # Impossible target vocabulary classes serialize as null, not nonfinite JSON.
    producer.digest(packet)
    if ratio==1.:
        assert independent[9]==4 and independent[10]==4 and independent[14]==2


@pytest.mark.parametrize('mutation',['hash','positions','labels','logits','shape','slots'])
def test_compact_packet_rejects_changed_input_mask_ce_or_endpoint(mutation):
    model,batch,logits,_=packet_case();packet=producer.evaluation_packet(batch,logits,1.)
    if mutation=='hash':packet['input_sha256']='0'*64
    elif mutation=='positions':packet['selected_positions'][0][1]+=1
    elif mutation=='labels':packet['selected_labels'][0]+=1
    elif mutation=='logits':packet['selected_logits'][0]=[None]*model.tokenizer.vocab_size
    elif mutation=='shape':packet['logits_shape'][1]+=1
    elif mutation=='slots':packet['slot_predictions'][0]=[]
    with pytest.raises((ValueError,IndexError)):
        producer.recount_packet(batch,packet,1.,model.terminal_id,
            [model.tokenizer.special_id('<binding>'),model.tokenizer.special_id('<nonbinding>')],model.tokenizer.vocab_size)


def test_independent_gradient_scale_rejects_ga_sum_even_with_clipping_appearance():
    oracle={'weight':torch.tensor([[.2,-.3],[.4,.1]])}
    good=producer.gradient_comparison(oracle,oracle);producer.verify_gradient_statistics(good)
    assert good['gradient_scale']==1 and good['relative_l2_error']==0
    with pytest.raises(ValueError,match='CE/GA gradient'):
        producer.gradient_comparison({name:value*8 for name,value in oracle.items()},oracle)
    corrupted=deepcopy(good);corrupted['gradient_scale']=8.
    with pytest.raises(ValueError,match='magnitude'):
        producer.verify_gradient_statistics(corrupted)


def test_existing_wrapper_output_failure_retains_no_success_or_complete(tmp_path,monkeypatch):
    monkeypatch.setattr(producer,'runtime_context',lambda *args:SimpleNamespace())
    def failure(*args):raise RuntimeError('artificial_cpu_failure_double')
    monkeypatch.setattr(producer,'worker',failure)
    output=tmp_path/'existing';output.mkdir();(output/'request.json').write_text('{"kind":"artificial_cpu_double"}')
    (output/'producer.log').write_text('artificial interface only\n')
    with pytest.raises(RuntimeError,match='artificial_cpu_failure'):
        producer.produce(tmp_path/'profile',output,producer.ROOT)
    assert (output/'FAILED.json').is_file()
    assert not (output/'PRODUCER_SUCCESS.json').exists()
    assert not (output/'GPU_RELEASE_COMPLETE.json').exists()
    assert 'cuda_attention_probe' not in Path(producer.__file__).read_text()


@pytest.mark.parametrize('phase',producer.PHASE_SECONDS)
def test_phase_budget_failure_keeps_small_cpu_observations_without_passing_artifacts(tmp_path,phase):
    plan=producer.fixed_plan(SimpleNamespace(profile={'config':asdict(ImmuneConfig()),'recommended_micro_batch':32}))
    limit=plan['phase_seconds'][phase]
    # Artificial CPU scalar double only; no CUDA metadata, model, or passing acceptance.
    value=dict(optimizer_steps=1,microsteps=8,records=256,training_loss=2.5,
        memory=dict(cuda_measured=False,elapsed_seconds=limit+.01),
        oracle=dict(passed=True,gradient_scale=1.,relative_l2_error=0.,parameters={}),
        _reference_gradients={'must_never_be_serialized':torch.ones(2)})
    producer.record_stage(tmp_path,plan,phase,'running',dict(optimizer_steps=1,microsteps=8,records=256),1.)
    with pytest.raises(producer.PhaseBudgetExceeded,match='exceeded fixed budget') as failure:
        producer.observed_phase_budget(tmp_path,plan,phase,value)
    assert failure.value.phase==phase and failure.value.limit_seconds==limit
    evidence=json.loads((tmp_path/producer.STAGE_EVIDENCE).read_text())
    assert evidence['gpu_accepted'] is evidence['raw_completed'] is False
    assert evidence['phases'][phase]['status']=='budget_failed'
    assert evidence['phases'][phase]['elapsed_seconds']==limit+.01
    assert evidence['phases'][phase]['records']==256
    assert len((tmp_path/producer.STAGE_EVIDENCE).read_bytes())<10000
    assert '_reference_gradients' not in (tmp_path/producer.STAGE_EVIDENCE).read_text()
    for name in ('gpu_release_raw.json','gpu_release_summary.json','PRODUCER_SUCCESS.json','GPU_RELEASE_COMPLETE.json'):
        assert not (tmp_path/name).exists()


def test_new_oracle_budget_can_observe_prior_elapsed_without_loosening_numbers(tmp_path):
    plan=producer.fixed_plan(SimpleNamespace(profile={'config':asdict(ImmuneConfig()),'recommended_micro_batch':32}))
    # Recorded elapsed from the negative v1 report is a scalar CPU test input, not rerun GPU evidence.
    observed=producer.observed_phase_budget(tmp_path,plan,'gradient_oracle',dict(
        optimizer_steps=1,microsteps=8,records=256,training_loss=2.5,
        memory=dict(cuda_measured=False,elapsed_seconds=701.6219)))
    assert observed['status']=='observed_within_budget' and observed['budget_seconds']==1200
    assert observed['memory']['cuda_measured'] is False
    assert plan['gradient_tolerance']==dict(atol=2e-5,rtol=.005,relative_l2=.005,scale=.005)


def test_produce_budget_failure_classification_retains_partial_and_no_candidate(tmp_path,monkeypatch):
    context=SimpleNamespace(profile={'config':asdict(ImmuneConfig()),'recommended_micro_batch':32})
    plan=producer.fixed_plan(context)
    monkeypatch.setattr(producer,'runtime_context',lambda *args:context)
    def negative_worker(context,output):
        producer.observed_phase_budget(output,plan,'gradient_oracle',dict(
            optimizer_steps=1,microsteps=8,records=256,training_loss=2.5,
            memory=dict(cuda_measured=False,elapsed_seconds=1200.01)))
    monkeypatch.setattr(producer,'worker',negative_worker)
    output=tmp_path/'cpu_negative';output.mkdir()
    with pytest.raises(producer.PhaseBudgetExceeded):
        producer.produce(tmp_path/'profile',output,producer.ROOT)
    failed=json.loads((output/'FAILED.json').read_text())
    assert failed['category']=='budget_exceeded' and failed['failed_stage']=='gradient_oracle'
    partial=json.loads((output/producer.STAGE_EVIDENCE).read_text())
    assert partial['phases']['gradient_oracle']['elapsed_seconds']==1200.01
    assert partial['phases']['gradient_oracle']['training_loss']==2.5
    assert partial['gpu_accepted'] is False
    for name in ('gpu_release_raw.json','gpu_release_summary.json','PRODUCER_SUCCESS.json','GPU_RELEASE_COMPLETE.json'):
        assert not (output/name).exists()
