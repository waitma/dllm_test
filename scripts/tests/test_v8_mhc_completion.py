"""Run: USE_TF=0 python -B -m pytest scripts/tests/test_v8_mhc_completion.py -q."""
import hashlib
import json

import pytest
import torch

from scripts.tests.test_v7_mhc_completion import references
from scripts.data.complete_v7_mhc import plan_candidate, checked_replacement, encoded
from scripts.data.complete_v8_mhc import check_negative_batch
from scripts.data.update_mhc_inputs import identity, non_mhc
from dllm.pipelines.immune_llada.data.records import BioSeqRecord, BioSeqChain
from dllm.pipelines.immune_llada.fixed_chain.unbinding_mlm import UnbindingMLMCollator, corrupt
from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig


@pytest.mark.parametrize('has_mhc', [False, True])
def test_completed_negative_keeps_all_non_mhc_data_and_relation_only_loss(references, tmp_path, has_mhc):
    reference, b2m, pseudo = references
    chains = ([BioSeqChain('A' * 34, 'mhc')] if has_mhc else []) + [
        BioSeqChain('ACDEFGHIK', 'peptide'), BioSeqChain('LMNPQRST', 'tcr_beta'),
        BioSeqChain('VWYACD', 'tcr_alpha', metadata={'synthetic_residue_mask': [1, 1, 0, 0, 0, 1]})]
    row = BioSeqRecord(chains, 'tcr_pmhc' if has_mhc else 'tcr_epitope', 'trait',
                       split='train', labels={'relation': 'nonbinding'},
                       metadata={'relation_supervised': True}).to_dict()
    candidate = dict(current_shard='train/unbind_trait.jsonl', current_row=0,
        source='trait', split='train', old_row_sha256=hashlib.sha256(encoded(row)).hexdigest(),
        identity=identity(row, 'trait'), original_mhc=['A' * 34] if has_mhc else [],
        allele_names=['A*02:01'], evidence=[dict(allele='A*02:01')])
    plan = plan_candidate(candidate, reference, b2m, pseudo, {})
    assert plan['accepted']
    completed = checked_replacement(row, candidate, plan, 'parent', 'ledger', output_dir=tmp_path)
    assert non_mhc(completed) == non_mhc(row)
    assert completed['metadata']['mhc_update']['evidence_ledger'] == str(tmp_path / 'provenance/allele_candidates.jsonl')
    collator = UnbindingMLMCollator(FixedChainConfig())
    before, after = collator([row]), collator([completed])
    check_negative_batch(before, after, collator.tokenizer)
    noisy = corrupt(after, collator.tokenizer, torch.Generator().manual_seed(4))
    assert noisy['labels'].ne(-100).sum() == 2
    assert not (noisy['labels'].ne(-100) & after['residue_mask']).any()
    with pytest.raises(ValueError, match='feature changed'):
        changed = {k: v.clone() for k, v in after.items()}
        changed['input_ids'][changed['role_ids'].eq(5)] = collator.tokenizer.mask_token_id
        check_negative_batch(before, changed, collator.tokenizer)


def test_missing_allele_cannot_be_reconstructed_from_pseudo(references):
    reference, b2m, pseudo = references
    candidate = dict(allele_names=[''], evidence=[dict(allele='')], original_mhc=['A' * 34])
    plan = plan_candidate(candidate, reference, b2m, pseudo, {})
    assert not plan['accepted'] and plan['status'] == 'missing_allele'


def test_publish_refuses_missing_independent_acceptance(tmp_path, monkeypatch):
    from scripts.data import complete_v8_mhc as builder
    stage, output = tmp_path / 'stage', tmp_path / 'published'
    stage.mkdir()
    (stage / 'dataset_manifest.json').write_text('{}')
    digest = builder.sha(stage / 'dataset_manifest.json')
    for name in ('verification.json', 'reader_acceptance.json'):
        (stage / name).write_text(json.dumps(dict(status='passed', dataset_manifest_sha256=digest)))
    monkeypatch.setattr(builder, 'STAGE', stage)
    monkeypatch.setattr(builder, 'OUTPUT', output)
    with pytest.raises(FileNotFoundError):
        builder.publish()
    assert stage.exists() and not output.exists()
