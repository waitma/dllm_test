"""Run: python -m pytest scripts/tests/test_chimera_h3.py -q (activate pllm first)."""
from copy import deepcopy
import pytest
import torch

from downstream.benchmark.ab_antigen import chimera_h3 as task


def tiny_model(fixed=False):
    from dllm.pipelines.immune_llada.v3.model import ImmuneConfig
    from dllm.pipelines.immune_llada.research_refined.factorial import FactorialModel
    torch.set_num_threads(1)
    kwargs = dict(d_model=24, n_layers=2, n_heads=3, mlp_hidden_size=48,
                  max_length=64, full_capacity=32, ablation='V5')
    if fixed:
        from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig, FixedChainModel
        return FixedChainModel(FixedChainConfig(**kwargs, first_chain_capacity=16, second_chain_capacity=14)).eval()
    return FactorialModel(ImmuneConfig(**kwargs), 'cv_segment_global').eval()


def row(cid='a'):
    return dict(complex_id=cid, antigen='KLMNP', heavy='CAACF', light='DEFG',
                h3_start=1, h3_end=4, native_h3='AAC', h3_contact_indices=[2], input_errors=[])


@pytest.mark.parametrize('fixed', [False, True])
def test_no_target_or_structure_leak_and_scaffold_preserved(fixed):
    model = tiny_model(fixed)
    a = row()
    b = {**a, 'heavy': 'CGGGF', 'native_h3': 'GGG', 'h3_contact_indices': [],
         'epitope_residues': ['changed'], 'true_coords': 'must never reach model'}
    inputs, mask = task.model_inputs(model, a)
    other, other_mask = task.model_inputs(model, b)
    assert torch.equal(mask, other_mask)
    assert all(torch.equal(inputs[k], other[k]) for k in inputs)
    assert not {'labels', 'epitope_residues', 'true_coords'} & set(inputs)
    changed_ag, _ = task.model_inputs(model, {**a, 'antigen': 'KLMNQ'})
    assert not torch.equal(inputs['input_ids'], changed_ag['input_ids'])
    original = deepcopy(inputs)
    sequence, audit = task.sample_h3(model, inputs, mask, 42, steps=3)
    repeat, _ = task.sample_h3(model, inputs, mask, 42, steps=3)
    assert sequence == repeat and len(sequence) == 3
    assert all(torch.equal(inputs[k], original[k]) for k in inputs)
    assert audit['coordinates_provided'] is False


def test_best_of_k_is_per_metric_and_macro_per_complex():
    rows = [row('a'), row('b')]
    # Candidate 0 wins AAR; candidate 1 wins CAAR. Do not pool all candidates.
    pred = [dict(complex_id=cid, candidate=k, pred_sequence=seq, status='ok')
            for cid, seqs in [('a', ['AAG', 'GGC']), ('b', ['GGG', 'GGG'])]
            for k, seq in enumerate(seqs)]
    result = task.score(rows, pred, candidates=2)
    assert result['summary']['best_aar']['mean'] == pytest.approx(1/3)
    assert result['summary']['best_caar']['mean'] == .5
    assert result['n_complexes'] == 2 and result['n_attempts'] == 4
    assert result['structural_metrics'] is None
    assert 'best_chimera_s' not in result['summary']


def test_failed_attempt_is_retained_and_missing_ids_rejected():
    rows = [row('a'), row('b')]
    pred = [dict(complex_id='a', candidate=0, pred_sequence='AAC', status='ok'),
            dict(complex_id='b', candidate=0, pred_sequence='', status='input_incompatible', failure_reason='overflow')]
    result = task.score(rows, pred)
    assert result['summary']['best_aar']['mean'] == .5
    assert len(result['failed_attempts']) == 1
    for malformed in (pred[:1], pred + pred[:1], [{**pred[0], 'candidate': 1}, pred[1]]):
        with pytest.raises(ValueError, match='candidate IDs'):
            task.score(rows, malformed)


def test_native_suite_dispatch_uses_h3(monkeypatch, tmp_path):
    from downstream.v5_native_generation_eval import antigen
    seen = []
    monkeypatch.setattr(task, 'run', lambda *args, **kwargs: seen.append(kwargs))
    antigen('chimera', tmp_path, None, None, {})
    antigen('chimera:chothia', tmp_path, None, None, {})
    assert seen == [{'scheme': 'imgt'}, {'scheme': 'chothia'}]


def test_run_keeps_complexes_sharing_antigen(tmp_path):
    import json
    rows = [{**row(cid), 'split': 'test'} for cid in ('a', 'b')]
    data = tmp_path / 'data'
    data.mkdir()
    path = data / 'imgt.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    task.write(data / 'imgt.manifest.json', dict(protocol=task.PROTOCOL, counts={'test': 2},
               dataset_sha256=task.sha(path), official_source_sha256=task.official_modules()[-1]))
    out = tmp_path / 'run'
    task.run(out, tiny_model(), {'test_fixture': True}, directory=data, steps=1, candidates=2)
    predictions = [json.loads(line) for line in (out / 'predictions.jsonl').read_text().splitlines()]
    assert {(r['complex_id'], r['candidate']) for r in predictions} == {('a', 0), ('a', 1), ('b', 0), ('b', 1)}
    result = json.loads((out / 'metrics.json').read_text())
    assert result['n_complexes'] == 2 and result['n_attempts'] == 4
    with pytest.raises(FileExistsError):
        task.run(out, tiny_model(), {}, directory=data)
