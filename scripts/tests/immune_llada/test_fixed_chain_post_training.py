"""Activate pllm/protenix_abtcr; python -m pytest this_file.py -q.

Check parent equivalence, conditional visibility, synthetic exclusion and real gradients.
"""
import pytest
import torch
from dllm.pipelines.immune_llada.fixed_chain import FixedChainConfig, FixedChainModel, FixedChainCollator
from dllm.pipelines.immune_llada.fixed_chain.post_training import PostTrainingModel, RECIPES
from dllm.pipelines.immune_llada.fixed_chain.post_training_data import PostTrainingCollator, observed_end_batch, relation_blind_features
from dllm.pipelines.immune_llada.data.records import BioSeqRecord, BioSeqChain


def fixture(recipe):
    torch.set_num_threads(1)
    cfg = FixedChainConfig(d_model=24, n_layers=4, n_heads=4, mlp_hidden_size=48,
                          max_length=128, first_chain_capacity=16, second_chain_capacity=12, full_capacity=30)
    torch.manual_seed(42)
    parent = FixedChainModel(cfg)
    model = PostTrainingModel(cfg, recipe)
    model.load_state_dict(parent.state_dict(), strict=True)
    model.install_adaptation()
    records = [BioSeqRecord([BioSeqChain('ACD', 'peptide'), BioSeqChain('CASSF', 'tcr_beta'),
                             BioSeqChain('CAVF', 'tcr_alpha')], 'tcr_epitope', 'test'),
               BioSeqRecord([BioSeqChain('EFG', 'peptide'), BioSeqChain('CASTF', 'tcr_beta'),
                             BioSeqChain('CATF', 'tcr_alpha')], 'tcr_epitope', 'test')]
    return parent, model, PostTrainingCollator(cfg)(records)


@pytest.mark.parametrize('recipe', [k for k, v in RECIPES.items() if not v.get('preference_weight')])
def test_every_recipe_has_finite_real_gradients_and_preserves_conditions(recipe):
    parent, model, batch = fixture(recipe)
    original = batch['input_ids'].clone()
    noised, selected, _ = model.training_corruption(batch)
    assert not (selected & ~batch['eligible_mask']).any()
    assert torch.equal(noised[~selected], original[~selected])
    loss = model(teacher=parent.eval().requires_grad_(False), **batch)['loss']
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    assert torch.equal(batch['input_ids'], original)


@pytest.mark.parametrize('recipe', ['continued', 'task_mix', 'lora', 'layerscale'])
def test_initial_forward_matches_parent(recipe):
    parent, model, batch = fixture(recipe)
    with torch.no_grad():
        a = parent.denoise(**parent.feature_args(batch))[0]
        b = model.denoise(**model.feature_args(batch))[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_continued_objective_is_exact_parent_control():
    parent, model, batch = fixture('continued')
    torch.manual_seed(123)
    a = parent(**batch)['loss']
    torch.manual_seed(123)
    b = model(**batch)['loss']
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_endpoint_reward_penalizes_early_eos_without_supervising_synthetic_sites():
    _, model, batch = fixture('endpoint')
    logits = torch.zeros((*batch['input_ids'].shape, model.tokenizer.vocab_size))
    selected = batch['eligible_mask'].clone()
    original = model.endpoint_loss(logits, batch, selected)
    first = (batch['residue_mask'][0] & batch['eligible_mask'][0]).nonzero()[0, 0]
    logits[0, first, model.terminal_id] = 8
    assert model.endpoint_loss(logits, batch, selected) > original
    # A completely unobserved example contributes no endpoint loss.
    assert model.endpoint_loss(logits, batch, torch.zeros_like(selected)) == 0
    # Synthetic holes invalidate the endpoint auxiliary for that chain.
    batch['residue_mask'][0, first] = False
    before = model.endpoint_loss(logits, batch, selected)
    logits[0, first, model.terminal_id] = -8
    torch.testing.assert_close(model.endpoint_loss(logits, batch, selected), before)


def test_late_fusion_and_global_topologies_have_distinct_actual_connectivity():
    for recipe in ('global', 'late_fusion'):
        _, model, batch = fixture(recipe)
        positions = batch['eligible_mask'][0].nonzero().flatten()
        a, b = int(positions[0]), int(positions[-1])
        allow = model.layer_allowed(0, batch['chain_ids'], batch['position_ids'], batch['attention_mask'])
        assert bool(allow[0, a, b]) == (recipe == 'global')
        allow = model.layer_allowed(3, batch['chain_ids'], batch['position_ids'], batch['attention_mask'])
        assert allow[0, a, b]


def test_lora_updates_only_adapters():
    _, model, batch = fixture('lora')
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(name.endswith(('.a', '.b')) for name in trainable)
    model(**batch)['loss'].backward()
    assert any(p.grad.abs().sum() > 0 for n, p in model.named_parameters() if n.endswith('.b'))


def test_screening_views_run_without_labels_entering_features():
    from scripts.research.run_fixed_chain_post_training import evaluate
    _, model, _ = fixture('task_mix')
    rows = [dict(chains=['ACD', 'CASSF', 'CAVF'], chain_roles=['peptide', 'tcr_beta', 'tcr_alpha'],
                 task_type='tcr_epitope', source='test', split='valid')]
    scores = evaluate(model, rows, torch.device('cpu'), batch_size=1)
    assert len(scores) == 6
    assert all(item['macro']['loss'] > 0 for item in scores.values())
    assert scores['first_chain']['macro']['chain_count'] == 1
    assert scores['second_chain']['macro']['chain_count'] == 1


def test_per_record_validation_sums_to_reported_source_metrics():
    from scripts.research.run_fixed_chain_post_training import evaluate
    from scripts.research.fixed_chain_validation import summarize
    _, model, _ = fixture('task_mix')
    rows = [dict(chains=['ACD', seq, 'CAVF'], chain_roles=['peptide', 'tcr_beta', 'tcr_alpha'],
                 task_type='tcr_epitope', source='test', split='valid') for seq in ['CASSF', 'CASTF']]
    result = evaluate(model, rows, torch.device('cpu'), semantic=True, collect_records=True)
    for view in result.values():
        if view is None:
            continue
        entries = view['records']['test']
        assert [x['row'] for x in entries] == [0, 1]
        assert entries[0]['cluster'] != entries[1]['cluster']
        assert summarize(torch.tensor([x['totals'] for x in entries], dtype=torch.float64)) == view['sources']['test']


def test_distinct_data_sources_with_same_record_alias_are_not_merged():
    from scripts.research.run_fixed_chain_post_training import assign_source_groups, source_of
    rows = [{'source': 'tcr_native'}, {'source': 'tcr_native'}]
    result = assign_source_groups(rows, {'train/tcr_native': 1, 'train/tcr_papers': 1}, 'train')
    assert list(map(source_of, result)) == ['tcr_native', 'tcr_papers']
    assert all(row['source'] == 'tcr_native' for row in result)


@pytest.mark.parametrize('recipe', ['repair', 'contrastive', 'distill', 'observed_cdr_distill', 'binding_aux'])
def test_two_forward_objectives_with_checkpointing_and_autocast(recipe):
    from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy
    parent, model, batch = fixture(recipe)
    model.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss = model(teacher=parent.eval().requires_grad_(False), **batch)['loss']
    loss.backward()
    gradient = model.backbone.transformer.blocks[0].base.q_proj.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_synthetic_endpoint_is_hidden_not_just_removed_from_loss():
    _, model, _ = fixture('observed_eos')
    row = BioSeqRecord([BioSeqChain('XXCASFXX', 'tcr_beta', metadata={'synthetic_residue_mask': [1,1,0,0,0,0,1,1]}),
                        BioSeqChain('CAVF', 'tcr_alpha')], 'tcr', 'test')
    batch = PostTrainingCollator(model.config)([row])
    result = observed_end_batch(batch, model.tokenizer.mask_token_id)
    assert batch['terminal_mask'].sum() == 2 and result['terminal_mask'].sum() == 1
    uncertain = batch['terminal_mask'] & ~batch['observed_terminal_mask']
    assert result['input_ids'][uncertain].eq(model.tokenizer.mask_token_id).all()
    assert result['labels'][uncertain].eq(-100).all() and not result['eligible_mask'][uncertain].any()
    for name in ('attention_mask', 'position_ids', 'chain_ids', 'role_ids', 'scope_ids', 'target_slot_mask'):
        assert torch.equal(result[name], batch[name])


def test_cdr_mapping_is_exact_and_excludes_synthetic_residues():
    _, model, _ = fixture('cdr_mix')
    chain = BioSeqChain('XCASSF', 'tcr_beta', regions={'FR1': 'X', 'CDR1': 'CA', 'FR2': 'S', 'CDR2': 'SF'},
                       metadata={'synthetic_residue_mask': [1,0,0,0,1,0]})
    row = BioSeqRecord([chain], 'tcr', 'test')
    batch = PostTrainingCollator(model.config)([row])
    assert batch['cdr_mapped_chains'].item() == 1 and batch['cdr_mask'].sum() == 3
    assert ''.join(model.tokenizer.token(int(x)) for x in batch['input_ids'][batch['cdr_mask']]) == 'CAF'
    chain.regions['CDR1'] = 'BAD'
    assert not PostTrainingCollator(model.config)([row])['cdr_mask'].any()


def test_binding_classifier_cannot_read_label_and_receives_gradients():
    from dataclasses import replace
    _, model, _ = fixture('binding_aux')
    row = BioSeqRecord([BioSeqChain('ACD', 'peptide'), BioSeqChain('CASSF', 'tcr_beta')],
                       'tcr_epitope', 'test', metadata={'relation_supervised': True}, labels={'relation': 'binding'})
    other = replace(row, labels={'relation': 'nonbinding'})
    collator = PostTrainingCollator(model.config)
    a, b = collator([row]), collator([other])
    fa, fb = relation_blind_features(model, a), relation_blind_features(model, b)
    assert all(torch.equal(fa[k], fb[k]) for k in fa)
    model(**collator([row, other]))['loss'].backward()
    assert model.binding_head.weight.grad is not None and model.binding_head.weight.grad.abs().sum() > 0


def test_frozen_teacher_distribution_has_no_gradient_and_reads_only_features():
    parent, model, batch = fixture('distill')
    parent.eval().requires_grad_(False)
    noised, _, _ = model.training_corruption(batch)
    features = {**model.feature_args(batch), 'input_ids': noised}
    target = model.teacher_distribution(parent, features)
    assert not target.requires_grad and torch.isfinite(target).all()
    torch.testing.assert_close(target.sum(-1), torch.ones_like(target[..., 0]))
    model(teacher=parent, **batch)['loss'].backward()
    assert all(p.grad is None for p in parent.parameters())


def test_observed_validation_scores_residues_but_not_template_endpoints():
    from scripts.research.run_fixed_chain_post_training import evaluate
    _, model, _ = fixture('task_mix')
    rows = [dict(chains=['XXCASFXX', 'CAVF'], chain_roles=['tcr_beta', 'tcr_alpha'],
                 task_type='tcr', source='test', split='valid',
                 chain_metadata={'0': {'synthetic_residue_mask': [1,1,0,0,0,0,1,1]}},
                 regions={'0': {'FR1': 'XX', 'CDR1': 'CA', 'FR2': 'S', 'CDR2': 'F', 'FR3': 'XX'}})]
    scores = evaluate(model, rows, torch.device('cpu'), semantic=True)
    assert scores['mask100']['macro']['chain_count'] == 2
    assert scores['observed_mask100']['macro']['chain_count'] == 1
    assert scores['observed_mask100']['macro']['aa_count'] == scores['mask100']['macro']['aa_count']
    assert scores['cdr']['macro']['aa_count'] == 3
    assert scores['cdr']['macro']['chain_count'] == 0


@pytest.mark.parametrize('recipe', ['task_mix', 'layerscale', 'lora', 'global', 'binding_aux'])
def test_post_training_export_preserves_adapted_logits_and_rejects_wrong_schema(recipe, tmp_path):
    _, model, batch = fixture(recipe)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    model(**batch)['loss'].backward()
    optimizer.step()
    before = model.denoise(**model.feature_args(batch))[0].detach()
    model.save_pretrained(tmp_path)
    restored = PostTrainingModel.from_pretrained(tmp_path)
    after = restored.denoise(**restored.feature_args(batch))[0].detach()
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    with pytest.raises(ValueError):
        FixedChainModel.from_pretrained(tmp_path)
