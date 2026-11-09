"""Activate pllm; USE_TF=0 python -B -m pytest THIS_ABSOLUTE_FILE -q.

Exercise real collator metadata, full-region corruption, multi-chain routing,
fixed validation noise, and the next optimizer update after RNG/state restore.
"""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord, CHAIN_ROLE_TO_ID
from dllm.pipelines.immune_llada.fixed_chain.region_mlm import MODE_NAMES
from dllm.pipelines.immune_llada.fixed_chain.v9_bert import corrupt as legacy_corrupt
from dllm.pipelines.immune_llada.fixed_chain.v10_airgen_qk_swiglu_bias import (
    INPUT_FEATURES, V10Collator, V10QKSwiGLUBiasModel, corrupt)
from scripts.research.bert_mlm_eval import fixed_corruption, token_validation
from scripts.research.run_bert_mlm import capture_rng, restore_rng
from scripts.tests.immune_llada.test_v10_airgen_qk_swiglu_bias import config


def annotated(role):
    regions = dict(FR1='AA', CDR1='GC', FR2='TT', CDR2='WY', FR3='CC', CDR3='CAG', FR4='FG')
    return BioSeqChain(''.join(regions.values()), role, regions)


def record(roles=('antibody_heavy', 'antibody_light'), relation='binding', contexts=('antigen',)):
    chains = [BioSeqChain('ACDEFGHIK', role) for role in contexts] + [annotated(role) for role in roles]
    task = ('tcr_pmhc' if 'mhc' in contexts else 'tcr_epitope') if roles[0].startswith('tcr') else (
        'nanobody_antigen' if roles[0] == 'nanobody_vhh' else 'antibody_antigen')
    return BioSeqRecord(chains, task, 'region_test', labels={'relation': relation})


def collator():
    return V10Collator(replace(config(), second_chain_capacity=65, full_capacity=132))


@pytest.mark.parametrize('roles', [('antibody_heavy', 'antibody_light'), ('nanobody_vhh',), ('tcr_beta', 'tcr_alpha')])
@pytest.mark.parametrize('mode', MODE_NAMES[1:])
def test_whole_region_and_mutation_contract(roles, mode):
    c = collator()
    raw = c([record(roles)])
    # Force CDR3 so a native cysteine is definitely among the labels.
    raw['cdr_region_ids'].masked_fill_(raw['cdr_region_ids'].ne(3), 0)
    noisy = corrupt(raw, c.tokenizer, torch.Generator().manual_seed(25), mode=mode)
    span = noisy['span_mask']
    assert span.sum() == 3
    assert raw['chain_ids'][span].unique().numel() == 1
    assert raw['cdr_region_ids'][span].eq(3).all()
    assert raw['residue_mask'][span].all() and not raw['terminal_mask'][span].any()
    assert noisy['labels'][span].equal(raw['input_ids'][span])
    assert noisy['labels'][span].eq(c.tokenizer.encode_residues('C')[0]).any()
    assert noisy['input_ids'][span].ne(raw['input_ids'][span]).all()
    outside = ~span & ~raw['explicit_relation_mask']
    assert noisy['input_ids'][outside].equal(raw['input_ids'][outside])
    assert noisy['labels'][outside].eq(-100).all()
    assert noisy['input_ids'][raw['explicit_relation_mask']].eq(c.tokenizer.mask_token_id).all()
    if mode == 'span_mask':
        assert noisy['input_ids'][span].eq(c.tokenizer.mask_token_id).all()
    else:
        pool = torch.tensor(c.tokenizer.encode_residues('ADEFGHIKLMNPQRSTVWY'))
        assert torch.isin(noisy['input_ids'][span], pool).all()
    model = V10QKSwiGLUBiasModel(c.config)
    assert set(model.feature_args(noisy)) == set(INPUT_FEATURES)


@pytest.mark.parametrize('mode', MODE_NAMES)
def test_positive_negative_inputs_and_selection_are_identical(mode):
    c = collator()
    batches = [corrupt(c([record(relation=relation)]), c.tokenizer,
                       torch.Generator().manual_seed(35), mode=mode) for relation in ('binding', 'nonbinding')]
    positive, negative = batches
    for key in (*INPUT_FEATURES, 'span_mask', 'corruption_mode', 'cdr_region_ids'):
        assert torch.equal(positive[key], negative[key]), key
    assert negative['labels'].ne(-100).equal(negative['explicit_relation_mask'])
    if mode != 'mlm':
        assert positive['labels'].ne(-100).equal(positive['span_mask'] | positive['explicit_relation_mask'])


def test_strict_annotation_and_original_offset_synthetic_mask():
    c = collator()
    original = record()
    heavy = original.chains[-2]
    # CDR1 is partly synthetic; CDR2/3 remain actual observations at their
    # original sequence positions. EOS, unused slots and synthetic residues get0.
    synthetic = [0] * len(heavy.sequence)
    synthetic[2] = 1
    chains = [*original.chains[:-2], replace(heavy, metadata={'synthetic_residue_mask': synthetic}),
              replace(original.chains[-1], regions={'CDR3': 'CAG'})]
    raw = c([replace(original, chains=chains)])
    ids = raw['cdr_region_ids']
    assert not ids.eq(1).any() and ids.eq(2).sum() == 2 and ids.eq(3).sum() == 3
    assert not ids[~raw['residue_mask']].any()
    assert raw['position_ids'][ids.eq(3)].tolist() == [11, 12, 13]
    bad = replace(heavy, regions={**heavy.regions, 'FR1': 'AAA'})
    fallback = c([replace(original, chains=[original.chains[0], bad])])
    assert not fallback['cdr_region_ids'].any()
    for mode in (None, *MODE_NAMES):
        noise = corrupt(fallback, c.tokenizer, torch.Generator().manual_seed(5), mode=mode)
        old = legacy_corrupt(fallback, c.tokenizer, torch.Generator().manual_seed(5))
        assert noise['corruption_mode'].eq(0).all() and not noise['span_mask'].any()
        for key in old:
            assert torch.equal(old[key], noise[key]), key
    c.mode = 'generation'
    first = c.row(original)
    second = c.row(replace(original, chains=[original.chains[0], BioSeqChain('A', 'antibody_heavy'),
                                           BioSeqChain('G', 'antibody_light')]))
    assert first == second and not any(first['cdr_region_ids'])


@pytest.mark.parametrize('roles,contexts', [
    (('antibody_heavy', 'antibody_light'), ('antigen',)),
    (('antibody_heavy', 'antibody_light'), ('antigen', 'antigen')),
    (('nanobody_vhh',), ('antigen', 'antigen', 'antigen')),
    (('tcr_beta', 'tcr_alpha'), ('peptide',)),
    (('tcr_beta', 'tcr_alpha'), ('mhc', 'peptide')),
])
def test_actual_three_four_chain_layout_attention_and_gradients(roles, contexts):
    torch.set_num_threads(2)
    c = collator()
    raw = c([record(roles, contexts=contexts)])
    chains = raw['chain_ids'][raw['chain_ids'].ge(0)].unique()
    assert len(chains) == len(roles)+len(contexts)
    for role in set(contexts):
        mask = raw['role_ids'].eq(CHAIN_ROLE_TO_ID[role]) & raw['residue_mask']
        assert mask.sum() == 9 * contexts.count(role)
        assert raw['chain_ids'][mask].unique().numel() == contexts.count(role)
    model = V10QKSwiGLUBiasModel(c.config)
    noisy = corrupt(raw, c.tokenizer, torch.Generator().manual_seed(17), mode='span_mask')
    out = model(**noisy)
    out['loss_sum'].backward()
    for block in model.backbone.transformer.blocks:
        assert torch.isfinite(block.recognition_score_bias.grad).all()
        assert block.recognition_score_bias.grad.abs().sum() > 0
        if len(roles) == 2:
            assert block.pair_score_bias.grad.abs().sum() > 0


def test_mixture_and_per_record_validation_reproducible(monkeypatch):
    c = collator()
    raw = c([record()])
    repeated = {key: value.repeat(1000, 1) for key, value in raw.items()}
    noisy = corrupt(repeated, c.tokenizer, torch.Generator().manual_seed(92))
    counts = torch.bincount(noisy['corruption_mode'][:, 0], minlength=3)
    assert 600 < counts[0] < 800 and (counts[1:] > 100).all() and (counts[1:] < 200).all()
    mutated = noisy['span_mask'] & noisy['corruption_mode'].eq(2)
    assert noisy['input_ids'][mutated].ne(repeated['input_ids'][mutated]).all()
    assert torch.isin(noisy['input_ids'][mutated], torch.tensor(c.tokenizer.encode_residues('ADEFGHIKLMNPQRSTVWY'))).all()
    # Select one index of each actual branch for fixed-noise validation.
    rows = [record(), replace(record(relation='nonbinding'),
                             chains=[BioSeqChain('ACD', 'antigen'), *record().chains[1:]]), record()]
    seeds = []
    for mode in range(3):
        seeds.append(next(seed for seed in range(100)
                          if int(fixed_corruption(c([rows[mode]]), c.tokenizer, [seed], corrupt_fn=corrupt)['corruption_mode'][0,0]) == mode))
    together = fixed_corruption(c(rows), c.tokenizer, seeds, corrupt_fn=corrupt)
    for row, sample in enumerate(seeds):
        individual = fixed_corruption(c([rows[row]]), c.tokenizer, [sample], corrupt_fn=corrupt)
        length = individual['input_ids'].shape[1]
        for key in individual:
            assert torch.equal(together[key][row, :length], individual[key][0]), key
    # token_validation chooses indices0,1,2: force their noise seeds to the
    # three selected deterministic branches while retaining actual corruption.
    import scripts.research.bert_mlm_eval as evaluation
    original_fixed = evaluation.fixed_corruption
    monkeypatch.setattr(evaluation, 'fixed_corruption', lambda batch, tok, indices, **kw:
                        original_fixed(batch, tok, [seeds[i] for i in indices], **kw))
    monkeypatch.setattr(evaluation, 'read_samples', lambda _: [dict(source_group='test', record=r.to_dict()) for r in rows])
    metrics = token_validation(V10QKSwiGLUBiasModel(c.config), {'files': {'token_valid': {}}}, torch.device('cpu'))['all']
    assert metrics['span_eligible_records'] == 3 and metrics['span_unavailable_records'] == 0
    assert all(metrics[name+'_records'] == 1 for name in MODE_NAMES)
    assert sum(metrics[name+'_target_tokens'] for name in MODE_NAMES) == metrics['selected_tokens']
    assert metrics['cdr_span_tokens'] > 0


def test_training_update_reports_actual_mode_target_counts(tmp_path):
    from torch.nn.parallel import DistributedDataParallel
    from scripts.research.run_bert_mlm import update
    torch.set_num_threads(2)
    c = collator()
    batches = [corrupt(c([record(relation='nonbinding' if mode == 'span_mutation' else 'binding')]),
                       c.tokenizer, torch.Generator().manual_seed(31), mode=mode) for mode in MODE_NAMES]
    batch = {key: torch.cat([b[key] for b in batches]) for key in batches[0]}
    torch.distributed.init_process_group('gloo', init_method=(tmp_path/'rendezvous').as_uri(), rank=0, world_size=1)
    try:
        model = DistributedDataParallel(V10QKSwiGLUBiasModel(c.config))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        metrics = update(model, batch, optimizer, scheduler, micro=1, world=1, bf16=False)
    finally:
        torch.distributed.destroy_process_group()
    assert metrics['span_eligible_records'] == 3 and metrics['span_unavailable_records'] == 0
    assert all(metrics[mode+'_records'] == 1 for mode in MODE_NAMES)
    assert sum(metrics[mode+'_target_tokens'] for mode in MODE_NAMES) == metrics['selected_tokens']
    assert metrics['span_mutation_span_target_tokens'] == 0
    assert metrics['span_mask_span_target_tokens'] > 0


def test_rng_optimizer_next_step_exact_after_replay():
    torch.set_num_threads(2)
    torch.manual_seed(44)
    c = collator()
    raw = c([record(), record(('tcr_beta', 'tcr_alpha'), contexts=('mhc', 'peptide'))])
    model = V10QKSwiGLUBiasModel(c.config)
    model.backbone.set_activation_checkpointing('whole_layer')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    rng = torch.Generator().manual_seed(71)
    def step(batch):
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch)['loss_sum']/batch['labels'].ne(-100).sum()
        loss.backward()
        optimizer.step()
        return loss.detach().clone()
    step(corrupt(raw, c.tokenizer, rng, mode='span_mask'))
    saved_model, saved_opt, saved_rng = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict()), capture_rng(rng)
    expected_batch = corrupt(raw, c.tokenizer, rng, mode='span_mutation')
    expected_loss = step(expected_batch)
    expected_state = deepcopy(model.state_dict())
    model.load_state_dict(saved_model)
    optimizer.load_state_dict(saved_opt)
    restore_rng(saved_rng, rng)
    actual_batch = corrupt(raw, c.tokenizer, rng, mode='span_mutation')
    assert all(torch.equal(value, actual_batch[key]) for key, value in expected_batch.items())
    assert torch.equal(expected_loss, step(actual_batch))
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in expected_state.items())
