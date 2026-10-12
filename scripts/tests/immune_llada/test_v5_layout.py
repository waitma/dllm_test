"""Run: python -m pytest scripts/tests/immune_llada/test_v5_layout.py -q."""
import pytest
import torch
from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord
from dllm.pipelines.immune_llada.v3.layout import ImmuneCollator
from dllm.pipelines.immune_llada.v3.model import ImmuneConfig, ImmuneModel
from dllm.pipelines.immune_llada.research_refined.factorial import FactorialModel


@pytest.mark.parametrize('task,chains,expected', [
 ('antibody', [('AC','antibody_heavy'),('DE','antibody_light')], '<prots><null><protd><unknown><prots><ab>AC.DE<protd>'),
 ('antibody_antigen', [('FG','antigen'),('AC','antibody_heavy'),('DE','antibody_light')], '<prots>FG<protd><binding><prots><ab>AC.DE<protd>'),
 ('nanobody', [('AC','nanobody_vhh')], '<prots><null><protd><unknown><prots><nb>AC<protd>'),
 ('nanobody_antigen', [('FG','antigen'),('AC','nanobody_vhh')], '<prots>FG<protd><binding><prots><nb>AC<protd>'),
 ('tcr', [('DE','tcr_alpha'),('AC','tcr_beta')], '<prots><null><protd><unknown><prots><tcr>AC.DE<protd>'),
 ('tcr', [('AC','tcr_beta')], '<prots><null><protd><unknown><prots><tcr>AC<protd>'),
 ('tcr_peptide', [('FG','peptide'),('AC','tcr_beta')], '<prots><pep>FG<protd><binding><prots><tcr>AC<protd>'),
 ('tcr_pmhc', [('HI','mhc'),('KL','mhc'),('FG','peptide'),('AC','tcr_beta'),('DE','tcr_alpha')], '<prots>HI.KL<protd><binding><prots><pep>FG<protd><binding><prots><tcr>AC.DE<protd>'),
])
def test_exact_object_layout(task, chains, expected):
    c = ImmuneCollator('V5', full_capacity=32)
    row = BioSeqRecord([BioSeqChain(*x) for x in chains], task, 'test', labels={'relation':'binding'})
    batch = c([row])
    tokens = batch['input_ids'][0].tolist()
    while tokens[-1] == c.tokenizer.mask_token_id:
        tokens.pop()
    assert ''.join(c.tokenizer.token(i) for i in tokens) == expected
    assert batch['terminal_mask'].sum() == 1
    assert batch['eligible_mask'][batch['terminal_mask']].all()


def test_synthetic_masks_and_backward():
    torch.set_num_threads(1)
    chain = BioSeqChain('XXACXDE', 'tcr_beta', metadata={'synthetic_residue_mask':[1,1,0,0,1,0,0]})
    record = BioSeqRecord([chain], 'tcr', 'test')
    c = ImmuneCollator('V5', full_capacity=32)
    b = c([record])
    assert b['residue_mask'].sum() == 4
    synthetic = b['input_ids'] == c.tokenizer.mask_token_id
    assert synthetic.sum() == 27
    assert not b['eligible_mask'][synthetic].any()
    assert b['labels'][synthetic].eq(-100).all()
    m = FactorialModel(ImmuneConfig(ablation='V5',d_model=24,n_layers=2,n_heads=3,mlp_hidden_size=48,max_length=64,full_capacity=32), 'cv_segment_global')
    out = m(**b, token_t=torch.ones_like(b['input_ids'],dtype=torch.float))
    assert torch.isfinite(out['loss'])
    out['loss'].backward()
    assert m.backbone.transformer.wte.weight.grad.abs().sum()>0
    assert m.mask_key(0)==('segment',)
    assert m.allowed_target[c.tokenizer.chain_separator_id()]
    assert not m.allowed_target[c.tokenizer.pad_token_id]


def test_generation_capacity_no_reference_leak():
    with pytest.raises(ValueError,match='not exact'):
        ImmuneCollator('V5', mode='generation', generation_lengths={'tcr_beta':9})
    c=ImmuneCollator('V5',mode='generation',full_capacity=32)
    a,b=[c([BioSeqRecord([BioSeqChain(seq,'tcr_beta')],'tcr','test')]) for seq in ('CAF','CASSGGGGF')]
    for key in ImmuneModel.feature_args(a):
        assert torch.equal(a[key],b[key]),key
    assert a['labels'].eq(-100).all()
    assert a['target_slot_mask'].sum()==32
    assert a['input_ids'][a['target_slot_mask']].eq(c.tokenizer.mask_token_id).all()


def test_downstream_cdr_hides_reference_and_preserves_context():
    from downstream.v5_layout import cdr_inputs, inputs
    torch.set_num_threads(1)
    model=FactorialModel(ImmuneConfig(ablation='V5',d_model=24,n_layers=2,n_heads=3,mlp_hidden_size=48,max_length=64,full_capacity=32),'cv_segment_global')
    records=[BioSeqRecord([BioSeqChain(seq,'antibody_heavy'),BioSeqChain('DE','antibody_light')],'antibody','test') for seq in ('CAAF','CGGF')]
    a,mask=cdr_inputs(model,records[0],'antibody_heavy',1,3)
    b,_=cdr_inputs(model,records[1],'antibody_heavy',1,3)
    for key in a:
        assert torch.equal(a[key],b[key])
    assert mask.sum()==2
    rep=inputs(model,[records[0]])
    assert rep['target_slot_mask'].sum()==32


@pytest.mark.parametrize('mode', ['train', 'representation'])
def test_cdr_only_pairs_preserve_full_scaffolds_as_masks(mode):
    from dllm.pipelines.immune_llada.data.records import CHAIN_ROLE_TO_ID
    chains = [
        BioSeqChain('XXXCAXSFXX', 'tcr_beta', metadata={
            'synthetic_residue_mask':[1,1,1,0,0,0,0,0,1,1]}),
        BioSeqChain('XXCAVFXXXX', 'tcr_alpha', metadata={
            'synthetic_residue_mask':[1,1,0,0,0,0,1,1,1,1]})]
    c = ImmuneCollator('V5', full_capacity=32, mode=mode)
    b = c([BioSeqRecord(chains,'tcr','test')])
    offset = 0
    slots = b['target_slot_mask'][0].nonzero().flatten()
    for chain in chains:
        positions = slots[offset:offset+len(chain.sequence)]
        offset += len(chain.sequence)+1
        # Grammar residue positions retain the prepared sequence's exact length.
        assert len(positions) == len(chain.sequence)
        synthetic = torch.tensor(chain.metadata['synthetic_residue_mask']).bool()
        ids = b['input_ids'][0,positions]
        assert ids[synthetic].eq(c.tokenizer.mask_token_id).all()
        expected = torch.tensor(c.tokenizer.encode_residues(chain.sequence))
        assert torch.equal(ids[~synthetic],expected[~synthetic])
        assert b['attention_mask'][0,positions].all()
        assert not b['eligible_mask'][0,positions[synthetic]].any()
        assert b['labels'][0,positions[synthetic]].eq(-100).all()
        assert not b['residue_mask'][0,positions[synthetic]].any()
    assert (b['input_ids']==c.tokenizer.chain_separator_id()).sum()==1
    assert (b['input_ids']==c.tokenizer.encode_residues('X')[0]).sum()==1


def test_fully_missing_partner_is_attended_mask_scaffold():
    c=ImmuneCollator('V5', full_capacity=32)
    row=BioSeqRecord([
        BioSeqChain('CAF','tcr_beta'),
        BioSeqChain('XXXXX','tcr_alpha',metadata={'synthetic_residue_mask':[1]*5})], 'tcr','test')
    b=c([row]); missing=b['input_ids']==c.tokenizer.mask_token_id
    assert missing.sum()==27
    assert b['attention_mask'][missing].all()
    assert b['labels'][missing].eq(-100).all()
    assert not b['eligible_mask'][missing].any()


def test_invalid_synthetic_provenance_fails():
    row=BioSeqRecord([BioSeqChain('XXCAF','tcr_beta',metadata={'synthetic_residue_mask':[1]})],'tcr','test')
    with pytest.raises(ValueError,match='provenance'):
        ImmuneCollator('V5', full_capacity=32)([row])
