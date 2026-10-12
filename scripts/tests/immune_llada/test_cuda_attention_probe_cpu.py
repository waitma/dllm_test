"""CPU-only: activate pllm, then pytest -q scripts/tests/immune_llada/test_cuda_attention_probe_cpu.py.

CUDA operations are forbidden or replaced by explicit test doubles. Mock CUDA
evidence below tests schema rejection only and is never an actual GPU result.
"""
from contextlib import contextmanager
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from scripts.architecture_prototypes import cuda_attention_probe as probe


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('This CPU test attempted a CUDA operation')
    for name in ('is_available','set_device','synchronize','empty_cache','reset_peak_memory_stats',
                 'get_device_properties','get_device_capability','Event'):
        monkeypatch.setattr(torch.cuda, name, forbidden)


@pytest.mark.parametrize('change', [dict(length=0),dict(length=4097),dict(batch_size=0),
    dict(warmup=2),dict(iterations=19),dict(backend='auto'),dict(precision='float16'),
    dict(mask='causal'),dict(bias_kind='none'),dict(padding=1),dict(heads=1.5),
    dict(batch_size=32,heads=32,length=4096)])
def test_invalid_or_unbounded_cases_rejected(change):
    with pytest.raises(ValueError):
        replace(probe.ProbeCase(), **change).validate()


def test_input_stream_is_private_and_backend_precision_independent():
    case = probe.ProbeCase(length=17, heads=3, head_dim=8, batch_size=2)
    before = torch.get_rng_state().clone()
    first = probe.make_inputs(case)
    other = probe.make_inputs(replace(case, backend='math', precision='float32'))
    assert torch.equal(before, torch.get_rng_state())
    assert probe.input_digest(first) == probe.input_digest(other)
    changed = probe.make_inputs(replace(case, seed=case.seed+1))
    assert probe.input_digest(first) != probe.input_digest(changed)
    assert all(value.device.type == 'cpu' for value in first.values())


@pytest.mark.parametrize('mask', ['global','chain','local'])
@pytest.mark.parametrize('kind', ['table','dense'])
@pytest.mark.parametrize('precision', ['float32','bfloat16'])
def test_cpu_math_oracle_retains_bias_gradients_and_mask_rules(mask, kind, precision):
    torch.set_num_threads(2)
    case = probe.ProbeCase(backend='math', precision=precision, length=17, heads=3,
                           head_dim=8, batch_size=2, mask=mask, bias_kind=kind, local_radius=2)
    state = probe.prepare_state(probe.make_inputs(case), 'cpu')
    q,k,v,bias = probe.compute_tensors(state, case)
    assert q.dtype == bias.dtype == probe.PRECISIONS[precision]
    assert bias.requires_grad
    allowed = probe.oracle.allowed_mask(state['groups'],state['valid'],mask,2)
    assert not (allowed & ~state['valid'][:,None,:]).any()
    if mask != 'global':
        assert not allowed[:,0,2].any()
        for row in range(case.batch_size):
            invalid_queries = ~state['valid'][row]
            assert torch.equal(allowed[row,invalid_queries],state['valid'][row].expand(int(invalid_queries.sum()),-1))
    first = probe.snapshot_step(state,case,'math')
    second = probe.snapshot_step(state,case,'math')
    comparison = probe.compare_snapshots(first,second,precision)
    assert comparison['passed'] and comparison['bias_gradient_present']
    assert first['bias_parameter_grad'].shape == state['parameter'].shape
    assert first['bias_parameter_grad'].dtype == torch.float32


@pytest.mark.parametrize('backend',['math','efficient'])
def test_one_forced_backend_and_original_bias_are_passed(monkeypatch,backend):
    recorded = []
    @contextmanager
    def context(choice):
        recorded.append(('backend',choice))
        yield
    def attention(q,k,v,**kwargs):
        recorded.append(('call',kwargs))
        return q
    monkeypatch.setattr(probe,'sdpa_kernel',context)
    monkeypatch.setattr(probe.F,'scaled_dot_product_attention',attention)
    q = torch.randn(1,2,3,4)
    bias = torch.randn(1,2,3,3,requires_grad=True)
    assert probe.forced_attention(q,q,q,bias,backend) is q
    assert recorded[0] == ('backend',probe.BACKENDS[backend])
    assert recorded[1][1] == {'attn_mask':bias,'dropout_p':0.,'is_causal':False}


def test_unsupported_is_not_retried_with_math(monkeypatch):
    calls=[]
    def failure(*args,**kwargs):
        calls.append(kwargs)
        raise RuntimeError('No available kernel. Aborting execution.')
    monkeypatch.setattr(probe.F,'scaled_dot_product_attention',failure)
    q = torch.ones(1,1,3,8)
    with pytest.raises(RuntimeError,match='No available kernel'):
        probe.forced_attention(q,q,q,torch.zeros(1,1,3,3),'efficient')
    assert len(calls) == 1


@pytest.mark.parametrize(('exception','expected'), [
    (RuntimeError('No viable backend for scaled_dot_product_attention'),'unsupported'),
    (torch.cuda.OutOfMemoryError('example only'),'oom'),
    (RuntimeError('CUDA out of memory.'),'oom'),
    (probe.NumericalMismatch('example'),'numerical_mismatch'),
    (probe.BackendMismatch('math was observed'),'backend_mismatch'),
    (subprocess.TimeoutExpired(['test'],1),'timeout'),
    (RuntimeError('unexpected programming bug'),'failed')])
def test_distinct_failure_categories(exception,expected):
    assert probe.classify_exception(exception) == expected


def test_backend_evidence_requires_expected_forward_backward_and_cuda_events():
    efficient=['aten::_scaled_dot_product_efficient_attention','aten::_scaled_dot_product_efficient_attention_backward']
    assert probe.observed_backend(efficient,['test-double-kernel'],'efficient') == 'efficient'
    for operators,kernels in [(efficient[:1],['test']), (efficient,[]),
        (['aten::_scaled_dot_product_attention_math'],['test']),
        (efficient+['aten::_scaled_dot_product_flash_attention'],['test'])]:
        with pytest.raises(probe.BackendMismatch):
            probe.observed_backend(operators,kernels,'efficient')


def test_nonfinite_diagnostics_are_serializable_and_small_gradient_norm_errors_detected():
    names=('output','q_grad','k_grad','v_grad','bias_parameter_grad')
    reference={name:torch.full((10000,),1e-4) for name in names}
    zeros={name:torch.zeros(10000) for name in names}
    assert not probe.compare_snapshots(reference,zeros,'bfloat16')['passed']
    nonfinite={name:torch.full((1,),float('nan')) for name in names}
    comparison=probe.compare_snapshots(nonfinite,nonfinite,'float32')
    assert not comparison['passed']
    json.dumps(comparison,allow_nan=False)


def test_cuda_unavailable_has_no_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    result=probe.run_cuda_case(probe.ProbeCase(),probe.fingerprints())
    assert result['status']=='unavailable'
    assert result['gpu_execution']=='not_started'
    assert result['timing'] is None and result['correctness'] is None


def test_dry_plan_and_single_case_never_spawn_or_query_cuda(monkeypatch,tmp_path):
    monkeypatch.setattr(probe,'run_case_process',lambda *a,**kw:pytest.fail('Dry run spawned a worker'))
    output=tmp_path/'plan'
    assert probe.main(['--dry-run','--output-dir',str(output),'--backends','efficient',
                       '--dtypes','bfloat16','--length','17']) == 0
    summary=json.loads((output/'summary.json').read_text())
    assert summary['status']=='dry_run_not_executed'
    assert summary['status_counts']=={'dry_run':1}
    assert summary['matched_comparisons']==[]
    command=summary['results'][0]['command']
    assert command[1]==str(Path(probe.__file__).resolve())
    assert '--worker-spec' in command and '--worker-result' in command
    with pytest.raises(FileExistsError):
        probe.execute_plan([probe.ProbeCase()],output,dry_run=True)


def test_subprocess_cases_are_sequential_isolated_and_failures_stay_nonpassing(monkeypatch,tmp_path):
    calls=[]
    def run(command,**kwargs):
        spec=json.loads(Path(command[command.index('--worker-spec')+1]).read_text())
        case=probe.ProbeCase(**spec['case'])
        result=probe.empty_result(case,'unsupported' if not calls else 'oom','test_double',spec['source_fingerprints'])
        assert set(kwargs) == {'stdout', 'timeout'}
        assert 0 < kwargs['timeout'] <= 10
        calls.append(command)
        probe.write_json(Path(command[command.index('--worker-result')+1]),result)
        return SimpleNamespace(returncode=2)
    monkeypatch.setattr(probe,'run_case_process',run)
    cases=[probe.ProbeCase(backend='math'),probe.ProbeCase(backend='efficient')]
    summary=probe.execute_plan(cases,tmp_path/'run',case_timeout=10,total_timeout=30)
    assert len(calls)==2 and calls[0]!=calls[1]
    assert summary['status_counts']=={'oom':1,'unsupported':1}
    assert all(row['child_process']['returncode']==2 and row['child_process']['cleanup_completed']
               and row['child_process']['protocol_valid'] for row in summary['results'])
    assert summary['matched_comparisons'][0]['status']=='not_comparable'
    assert 'total_median_speedup_math_over_efficient' not in summary['matched_comparisons'][0]


@pytest.mark.parametrize('returncode', [0, 1, -15, -9, 143])
def test_negative_json_with_abnormal_exit_is_execution_failure(monkeypatch,tmp_path,returncode):
    def run(command,**kwargs):
        spec=json.loads(Path(command[command.index('--worker-spec')+1]).read_text())
        case=probe.ProbeCase(**spec['case'])
        result=probe.empty_result(case,'unsupported','candidate_profile_and_gradients',spec['source_fingerprints'])
        probe.write_json(Path(command[command.index('--worker-result')+1]),result)
        return SimpleNamespace(returncode=returncode)
    monkeypatch.setattr(probe,'run_case_process',run)
    summary=probe.execute_plan([probe.ProbeCase()],tmp_path/'abnormal')
    result=summary['results'][0]
    assert result['status']=='failed' and result['phase']=='parent_execution'
    assert result['child_process']==dict(returncode=returncode,cleanup_completed=True,
        reported_status='unsupported',expected_returncode=2,protocol_valid=False)
    assert json.loads(next((tmp_path/'abnormal').glob('00_*/result.json')).read_text())==result


def test_timeout_consumes_remaining_budget_and_later_cases_do_not_spawn(monkeypatch,tmp_path):
    clock=[0.]
    calls=[]
    monkeypatch.setattr(probe.time,'monotonic',lambda:clock[0])
    def timeout(command,**kwargs):
        calls.append(command)
        assert kwargs['timeout']==5
        clock[0]=6
        raise subprocess.TimeoutExpired(command,kwargs['timeout'])
    monkeypatch.setattr(probe,'run_case_process',timeout)
    summary=probe.execute_plan([probe.ProbeCase(backend='math'),probe.ProbeCase()],
                              tmp_path/'timeout',case_timeout=10,total_timeout=5)
    assert len(calls)==1
    assert summary['status_counts']=={'not_run_budget':1,'timeout':1}


def test_passing_status_without_numeric_or_kernel_evidence_is_rejected():
    case=probe.ProbeCase()
    result=probe.empty_result(case,'passed','complete',{'test':'hash'})
    with pytest.raises(ValueError,match='CUDA/gradient'):
        probe.validate_result(result,case,{'test':'hash'})
    result.update(gpu_execution='completed',environment={'cuda_available':True},
                  correctness={'passed':True,'bias_gradient_present':True})
    with pytest.raises(ValueError,match='complete numerical'):
        probe.validate_result(result,case,{'test':'hash'})


def test_warmup_and_twenty_timed_steps_use_separate_cuda_events_with_test_doubles(monkeypatch):
    clock=[0.]
    class Event:
        def __init__(self,enable_timing):
            assert enable_timing
        def record(self):
            clock[0]+=.25
            self.at=clock[0]
        def elapsed_time(self,other):
            return other.at-self.at
    forward_calls=[]
    def forward(state,case):
        forward_calls.append(case.backend)
        return torch.tensor(1.),torch.tensor(1.)
    monkeypatch.setattr(torch.cuda,'Event',Event)
    for name in ('synchronize','empty_cache','reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda,name,lambda *args:None)
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda *args:100)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda *args:160)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda *args:180)
    monkeypatch.setattr(probe,'forward_loss',forward)
    monkeypatch.setattr(probe,'backward',lambda *args:(torch.tensor(1.),))
    case=probe.ProbeCase(warmup=3,iterations=20)
    timing,memory=probe.measure_candidate({},case)
    assert len(forward_calls)==23 and set(forward_calls)=={'efficient'}
    assert timing['iterations']==20 and timing['warmup']==3
    assert timing['total_ms']['median']==.5
    assert len(timing['raw_total_ms'])==20
    assert memory['incremental_peak_allocated_bytes']==60


def test_choice_lists_reject_unknowns_and_duplicates():
    for text in ('math,math','math,auto',''):
        with pytest.raises(Exception):
            probe.split_choices(text,probe.BACKENDS)
    arguments=probe.parser().parse_args(['--masks','global,chain,local','--bias-kinds','table,dense'])
    assert len(probe.cases_from_args(arguments))==24


def complete_test_double(case, sources):
    """In-memory schema fixture only; no GPU operation and no result file write."""
    result=probe.empty_result(case,'passed','complete',sources)
    snapshots={name:torch.ones(2) for name in ('output','q_grad','k_grad','v_grad','bias_parameter_grad')}
    operators=(['aten::_scaled_dot_product_attention_math'] if case.backend=='math' else
        ['aten::_scaled_dot_product_efficient_attention','aten::_scaled_dot_product_efficient_attention_backward'])
    result.update(gpu_execution='completed',environment={'cuda_available':True,'gpu':'TEST DOUBLE'},
        input_sha256='TEST-ONLY-HASH',correctness=probe.compare_snapshots(snapshots,snapshots,case.precision),
        backend_evidence={'requested':case.backend,'observed':case.backend,'operators':operators,
                          'cuda_kernel_names':['TEST-ONLY-NOT-A-CUDA-MEASUREMENT']},
        timing={'warmup':case.warmup,'iterations':case.iterations,'raw_total_ms':[1.]*case.iterations,
                **{key:probe.timing_summary([1.]*case.iterations) for key in ('forward_ms','backward_ms','total_ms')}},
        memory={'baseline_allocated_bytes':100,'peak_allocated_bytes':160,
                'incremental_peak_allocated_bytes':60,'peak_reserved_bytes':180})
    return result


def test_complete_result_schema_and_identity_checks():
    case=probe.ProbeCase()
    result=complete_test_double(case,{'test':'hash'})
    assert probe.validate_result(result,case,{'test':'hash'}) is result
    with pytest.raises(ValueError,match='identity'):
        probe.validate_result(result,replace(case,seed=case.seed+1),{'test':'hash'})
    result['timing']['iterations']=19
    with pytest.raises(ValueError,match='timing protocol'):
        probe.validate_result(result,case,{'test':'hash'})
    result['timing']['iterations']=case.iterations
    result['memory']['incremental_peak_allocated_bytes']=59
    with pytest.raises(ValueError,match='Inconsistent memory'):
        probe.validate_result(result,case,{'test':'hash'})


def test_pairing_requires_same_input_environment_and_successful_backends():
    a=complete_test_double(probe.ProbeCase(backend='math'),{'test':'hash'})
    b=complete_test_double(probe.ProbeCase(backend='efficient'),{'test':'hash'})
    assert probe.matched_comparisons([a,b])[0]['total_median_speedup_math_over_efficient']==1.
    b['input_sha256']='different'
    assert probe.matched_comparisons([a,b])[0]['status']=='not_comparable'
    b['input_sha256']=a['input_sha256']
    b['environment']['gpu']='another device'
    assert probe.matched_comparisons([a,b])[0]['status']=='not_comparable'
    b['environment']=a['environment']
    b['status']='unsupported'
    assert 'total_median_speedup_math_over_efficient' not in probe.matched_comparisons([a,b])[0]


def test_changed_source_rejected_before_cuda_query(monkeypatch):
    monkeypatch.setattr(probe,'fingerprints',lambda:{'new':'hash'})
    result=probe.run_cuda_case(probe.ProbeCase(),{'old':'hash'})
    assert result['status']=='failed'
    assert result['gpu_execution']=='not_started'
    assert 'source changed' in result['error']['message']
