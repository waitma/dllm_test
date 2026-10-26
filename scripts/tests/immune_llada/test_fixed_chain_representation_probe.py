"""Activate pllm/protenix_abtcr; python -m pytest this_file.py -q.

Test label blindness and training-only probe selection with small analytic data.
"""
from dataclasses import replace
import numpy as np
import torch
from dllm.pipelines.immune_llada.data.records import BioSeqRecord, BioSeqChain
from scripts.tests.immune_llada.test_fixed_chain_post_training import fixture
from scripts.research.run_fixed_chain_representation_probe import RelationBlindProbeBackbone, fit_probe


def test_probe_input_is_identical_after_label_flip():
    _, model, _ = fixture('task_mix')
    backbone = RelationBlindProbeBackbone(model, {'step': 0})
    row = BioSeqRecord([BioSeqChain('ACD', 'peptide'), BioSeqChain('CASSF', 'tcr_beta')],
                      'tcr_epitope', 'test', labels={'relation': 'binding'},
                      metadata={'relation_supervised': True})
    other = replace(row, labels={'relation': 'nonbinding'})
    first, second = backbone.collator([row]), backbone.collator([other])
    for key in model.feature_args(first):
        assert torch.equal(first[key], second[key])
    assert first['input_ids'][first['relation_token_mask']].eq(model.tokenizer.mask_token_id).all()


def test_probe_hyperparameter_selection_does_not_read_screen_labels():
    rng = np.random.default_rng(42)
    rows = [dict(chains=['ACD', 'C'+str(i)], chain_roles=['peptide', 'tcr_beta'], source='test',
                 labels={'relation': 'binding' if i%2 else 'nonbinding'}) for i in range(100)]
    vectors = rng.normal(size=(100, 4))
    vectors[:, 0] += np.arange(100)%2 * 4
    a = fit_probe(vectors, vectors[:10], rows, rows[:10])
    changed = [{**row, 'labels': {'relation': 'nonbinding' if row['labels']['relation']=='binding' else 'binding'}}
               for row in rows[:10]]
    b = fit_probe(vectors, vectors[:10], rows, changed)
    assert a['selected_C'] == b['selected_C']
    assert a['inner_training_auroc'] == b['inner_training_auroc']
    assert [p['probability'] for p in a['predictions']] == [p['probability'] for p in b['predictions']]
