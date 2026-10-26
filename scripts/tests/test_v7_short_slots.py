"""Activate pllm; USE_TF=0 python -B -m pytest scripts/tests/test_v7_short_slots.py -q."""
import pytest
from scripts.data.resize_v7_design_slots import CAPACITIES, overflows
from dllm.pipelines.immune_llada.fixed_chain import FixedChainConfig, FixedChainCollator
from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord


@pytest.mark.parametrize('role,slots', list(CAPACITIES.items()))
def test_eos_boundary_agrees_with_actual_reader(role, slots):
    collator = FixedChainCollator(FixedChainConfig(first_chain_capacity=167,
        second_chain_capacity=135, full_capacity=304))
    task = 'tcr' if role.startswith('tcr') else 'antibody'
    for length in (slots - 1, slots):
        seq = 'A' * length
        row = dict(chains=[seq], chain_roles=[role])
        record = BioSeqRecord([BioSeqChain(seq, role)], task, 'boundary_test')
        if length == slots:
            assert overflows(row)
            with pytest.raises(ValueError, match='chain_capacity_overflow'):
                collator([record])
        else:
            assert not overflows(row)
            batch = collator([record])
            assert int(batch['target_slot_mask'].sum()) == slots
            assert int(batch['terminal_mask'].sum()) == 1
            assert batch['input_ids'][batch['terminal_mask']].item() == collator.tokenizer.eos_token_id


def test_synthetic_scaffold_and_context_are_not_shortened():
    row = dict(chains=['A' * 500, 'X' * 167, 'A' * 134],
               chain_roles=['antigen', 'antibody_heavy', 'antibody_light'])
    assert overflows(row) == [dict(chain_index=1, role='antibody_heavy', residues=167, slots=167)]
    collator = FixedChainCollator(FixedChainConfig(first_chain_capacity=167,
        second_chain_capacity=135, full_capacity=304))
    record = BioSeqRecord([
        BioSeqChain('XXCAFXX', 'tcr_beta', metadata={'synthetic_residue_mask': [1,1,0,0,0,1,1]}),
        BioSeqChain('X' * 134, 'tcr_alpha', metadata={'synthetic_residue_mask': [1] * 134})],
        'tcr', 'scaffold_test')
    batch = collator([record])
    assert int(batch['target_slot_mask'].sum()) == 302
    assert int(batch['terminal_mask'].sum()) == 1
    assert int(batch['eligible_mask'].sum()) == 4  # Three observed residues and template EOS.
