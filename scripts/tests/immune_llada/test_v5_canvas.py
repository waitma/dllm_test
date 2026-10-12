"""Run python -m pytest scripts/tests/immune_llada/test_v5_canvas.py -q."""
import pytest
import torch
from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord
from dllm.pipelines.immune_llada.v3.layout import ImmuneCollator, GenerationRequest
from dllm.pipelines.immune_llada.v3.model import ImmuneConfig, ImmuneModel
from dllm.pipelines.immune_llada.research_refined.factorial import FactorialModel
from downstream.v5_layout import generate, parse_object


def record(beta='CAAF', alpha='CGF', **metadata):
    return BioSeqRecord([BioSeqChain('ACD','peptide'),BioSeqChain(beta,'tcr_beta',metadata=metadata),
                         BioSeqChain(alpha,'tcr_alpha')], 'tcr_epitope','test')


def tiny():
    torch.set_num_threads(1)
    return FactorialModel(ImmuneConfig(ablation='V5',d_model=24,n_heads=3,n_layers=2,
                                      mlp_hidden_size=48,max_length=64,full_capacity=24),'cv_segment_global')


def test_fully_masked_train_matches_generation_all_features():
    c=ImmuneCollator('V5',full_capacity=24)
    g=ImmuneCollator('V5',full_capacity=24,mode='generation')
    model=tiny()
    variants=[record(),record('CASSGGF','CAVVVWF'),record('XXXCAF','XXCF',synthetic_residue_mask=[1,1,1,0,0,0])]
    reference=model.feature_args(g([variants[0]]))
    for row in variants:
        batch=c([row])
        noised,_,_=model.corrupt(batch,torch.ones_like(batch['input_ids'],dtype=torch.float))
        features={**model.feature_args(batch),'input_ids':noised}
        for name in reference:
            assert torch.equal(reference[name],features[name]),name
        inference=g([row])
        for name in reference:
            assert torch.equal(reference[name],inference[name]),name


def test_reference_free_request_and_tail_terminal_supervision():
    c=ImmuneCollator('V5',full_capacity=24)
    b=c([record()]);t=c.tokenizer
    slots=b['target_slot_mask']
    assert slots.sum()==24
    assert b['attention_mask'][slots].all()
    assert (b['labels']==t.special_id('<protd>')).sum()==1
    assert (b['labels']==t.chain_separator_id()).sum()==1
    tail=slots & ~b['eligible_mask']
    assert tail.any() and b['input_ids'][tail].eq(t.mask_token_id).all()
    assert b['role_ids'][slots].eq(0).all() and b['scope_ids'][slots].eq(0).all()
    request=GenerationRequest((BioSeqChain('ACD','peptide'),),(('tcr_beta','full'),('tcr_alpha','full')))
    g=ImmuneCollator('V5',full_capacity=24,mode='generation')
    for key in ImmuneModel.feature_args(b):
        assert torch.equal(g([request])[key],g([record()])[key]),key


def test_split_does_not_change_object_attention_or_positions():
    model=tiny();c=ImmuneCollator.from_config(model.config)
    b=c([record()]);t=c.tokenizer
    dot=int((b['input_ids'][0]==t.chain_separator_id()).nonzero()[0])
    ids,pos=b['chain_ids'],b['position_ids']
    assert ids[0,dot-1]==ids[0,dot+1]
    assert pos[0,dot+1]==pos[0,dot]+1
    allowed=model.layer_allowed(0,ids,pos,b['attention_mask'])
    assert allowed[0,dot-1,dot+1]
    slots=b['target_slot_mask'][0].nonzero().flatten()
    assert ids[0,slots].unique().numel()==1
    assert allowed[0,slots[0],slots[-1]]
    # Headers belong to the same target object; relations remain outside it.
    header=int(slots[0])-2
    assert ids[0,header]==ids[0,slots[0]]
    assert allowed[0,header,slots[-1]]
    assert not allowed[0,slots[0],2]  # peptide context
    assert model.layer_allowed(1,ids,pos,b['attention_mask'])[0,slots[0],2]
    saved=[]
    handle=model.backbone.transformer.blocks[0].register_forward_pre_hook(
        lambda module,args: saved.append((args[1].detach().clone(),args[2].detach().clone())))
    try:
        with torch.no_grad():
            model.denoise(**model.feature_args(b))
            masked=b['input_ids'].masked_fill(b['input_ids']==t.chain_separator_id(),t.mask_token_id)
            model.denoise(**{**model.feature_args(b),'input_ids':masked})
    finally:
        handle.remove()
    for before,after in zip(saved[0],saved[1]):
        torch.testing.assert_close(before,after,atol=0,rtol=0)


def test_fixed_canvas_backward_and_padding_invariance():
    model=tiny(); c=ImmuneCollator.from_config(model.config)
    b=c([record()])
    out=model(**b,token_t=torch.ones_like(b['input_ids'],dtype=torch.float))
    assert torch.isfinite(out['loss'])
    out['loss'].backward()
    assert model.backbone.transformer.wte.weight.grad.abs().sum()>0
    model.eval()
    other=BioSeqRecord([BioSeqChain('ACDEFGHI','antigen'),BioSeqChain('CAAF','antibody_heavy')],'antibody_antigen','test')
    with torch.no_grad():
        a,_=model.denoise(**model.feature_args(b))
        both=c([record(),other]); bb,_=model.denoise(**model.feature_args(both))
    torch.testing.assert_close(a[0,:,model.allowed_target],bb[0,:a.shape[1],model.allowed_target],atol=1e-5,rtol=1e-5)


def test_capacity_errors_and_hidden_length_never_validated():
    with pytest.raises(ValueError,match='object_capacity_overflow'):
        ImmuneCollator('V5',full_capacity=4)([record()])
    with pytest.raises(ValueError,match='sequence_budget_overflow'):
        ImmuneCollator('V5',full_capacity=24,max_length=25)([record()])
    # A withheld over-budget reference is not read during generation.
    b=ImmuneCollator('V5',full_capacity=24,mode='generation')([record('C'*3000)])
    assert b['target_slot_mask'].sum()==24


def test_validation_accounts_for_learned_split_separately():
    from scripts.architecture_prototypes.validation_metrics import selected_token_metrics, summarize
    model = tiny().eval()
    batch = ImmuneCollator.from_config(model.config)([record()])
    result = summarize(selected_token_metrics(model, batch, 1.)['sample_totals'])
    assert result['split_count'] == 1
    assert result['relation_count'] == 0
    assert result['selected_count'] == result['aa_count'] + result['terminal_count'] + 1
    component = (result['aa_loss'] * result['aa_count']
                 + result['terminal_loss'] * result['terminal_count'] + result['split_loss'])
    assert result['loss'] * result['selected_count'] == pytest.approx(component)


def test_known_prefix_suffix_generation_has_unknown_end():
    g=ImmuneCollator('V5',full_capacity=24,mode='generation',target_roles=['tcr_alpha'])
    a=g([record()]);b=g([record(alpha='C'*100)])
    for key in ImmuneModel.feature_args(a):
        assert torch.equal(a[key],b[key]),key
    t=g.tokenizer; ids=a['input_ids'][a['target_slot_mask']].tolist()
    assert ids[:5]==t.encode_residues('CAAF')+[t.chain_separator_id()]
    assert all(x==t.mask_token_id for x in ids[5:])
    with pytest.raises(ValueError,match='target suffix'):
        ImmuneCollator('V5',mode='generation',target_roles=['tcr_beta'])([record()])


@pytest.mark.parametrize('sequence,status', [('AC.DE<protd>','ok'),('ACDE','capacity_exhausted'),
    ('AC.<protd>','invalid_chain_structure'),('AC..DE<protd>','invalid_chain_structure')])
def test_parser(sequence,status):
    t=ImmuneCollator('V5').tokenizer
    ids=[]
    for part in sequence.split('<protd>'):
        if ids: ids.append(t.special_id('<protd>'))
        ids.extend(t.chain_separator_id() if c=='.' else t.encode_residues(c)[0] for c in part)
    result=parse_object(t,ids,expected_chains=2)
    assert result['status']==status


def test_decoder_learns_end_and_ignores_tail():
    model=tiny();t=model.tokenizer
    desired=t.encode_residues('AC')+[t.chain_separator_id()]+t.encode_residues('DE')+[t.special_id('<protd>')]
    calls=[]
    def denoise(**features):
        calls.append(features['input_ids'].clone())
        logits=torch.full((*features['input_ids'].shape,t.vocab_size),-10.)
        for b in range(logits.shape[0]):
            slots=features['target_slot_mask'][b].nonzero().flatten()
            for j,slot in enumerate(slots):
                logits[b,slot,desired[j] if j<len(desired) else t.encode_residues('A')[0]]=30. if j<len(desired) else 0.
        return logits,None
    model.denoise=denoise
    request=GenerationRequest((),(('tcr_beta','full'),('tcr_alpha','full')))
    result=generate(model,[request],steps=4,expected_chains=2)[0]
    assert result['status']=='ok' and result['chains']==['AC','DE']
    assert result['terminal_offset']==5 and result['steps']==1
    assert t.mask_token_id in result['canvas_ids'][6:]
    assert model.training
    assert len(calls)==1


def test_v5_checkpoint_recomputation_matches_gradients():
    import copy
    from dllm.pipelines.llada.models.modeling_llada import ActivationCheckpointingStrategy
    first=tiny();second=copy.deepcopy(first)
    second.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
    b=ImmuneCollator.from_config(first.config)([record()])
    # Keep the separator visible in one pass, mask it in another; both passes
    # retain identical object boundaries through checkpoint recomputation.
    values=[]
    for model in (first,second):
        features=model.feature_args(b)
        logits,_=model.denoise(**features)
        noised=b['input_ids'].masked_fill(b['target_slot_mask'],model.tokenizer.mask_token_id)
        other,_=model.denoise(**{**features,'input_ids':noised})
        loss=logits[:,:,model.allowed_target].square().mean()+other[:,:,model.allowed_target].square().mean()
        loss.backward();values.append(loss.detach())
    torch.testing.assert_close(values[0],values[1],atol=0,rtol=0)
    for (name,a),(_,b) in zip(first.named_parameters(),second.named_parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad,b.grad,atol=1e-6,rtol=1e-5,msg=name)


def test_config_roundtrip_and_native_generate_dispatch():
    from tempfile import TemporaryDirectory
    from pathlib import Path
    model=tiny().eval()
    c=ImmuneCollator.from_config(model.config,mode='generation')
    request=GenerationRequest((),(('tcr_beta','full'),))
    batch=c([request])
    with TemporaryDirectory(prefix='v5-canvas-check-') as directory:
        model.save_pretrained(directory)
        restored=FactorialModel.from_pretrained(directory).eval()
        assert restored.config.full_capacity==24 and restored.config.ablation=='V5'
        with torch.no_grad():
            a,_=model.denoise(**model.feature_args(batch))
            b,_=restored.denoise(**restored.feature_args(batch))
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert not Path(directory).exists()
    # Native entry uses the same parser and rejects accidentally supplied ends.
    result=model.generate(batch,steps=1,expected_chains=1)
    assert len(result)==1 and 'status' in result[0]
    train=ImmuneCollator.from_config(model.config)([record()])
    with pytest.raises(ValueError,match='supplied terminal'):
        model.generate(train,steps=1)


def test_suffix_train_corruption_matches_generation():
    model=tiny()
    c=ImmuneCollator.from_config(model.config,target_roles=['tcr_alpha'])
    g=ImmuneCollator.from_config(model.config,target_roles=['tcr_alpha'],mode='generation')
    b=c([record()]);noised,_,_=model.corrupt(b,torch.ones_like(b['input_ids'],dtype=torch.float))
    train={**model.feature_args(b),'input_ids':noised}
    for k,v in model.feature_args(g([record()])).items():
        assert torch.equal(train[k],v),k


def test_generation_no_end_is_failure_not_silent_truncation():
    model=tiny();t=model.tokenizer
    def denoise(**features):
        logits=torch.zeros((*features['input_ids'].shape,t.vocab_size))
        logits[:,:,t.encode_residues('A')[0]]=10.
        return logits,None
    model.denoise=denoise
    request=GenerationRequest((),(('tcr_beta','full'),))
    result=generate(model,[request],steps=2)[0]
    assert result['status']=='capacity_exhausted' and result['chains']==[]


def test_representation_pools_by_protein_object():
    model=tiny().eval()
    b=ImmuneCollator.from_config(model.config,mode='representation')([record()])
    represented=model.represent(b)
    assert len(represented['chain_embeddings'][0])==2  # peptide object, receptor object (beta + alpha)



def test_multichain_context_uses_prots_protd_object_boundaries():
    model=tiny();c=ImmuneCollator.from_config(model.config)
    row=BioSeqRecord([BioSeqChain('AC','mhc'),BioSeqChain('DE','mhc'),
                     BioSeqChain('FG','peptide'),BioSeqChain('CAF','tcr_beta'),
                     BioSeqChain('CGF','tcr_alpha')],'tcr_pmhc','test')
    b=c([row]);t=c.tokenizer
    ids=b['input_ids'][0];groups=b['chain_ids'][0];positions=b['position_ids'][0]
    starts=(ids==t.special_id('<prots>')).nonzero().flatten().tolist()
    ends=(ids==t.special_id('<protd>')).nonzero().flatten().tolist()
    assert len(starts)==3
    for start,end in zip(starts[:2],ends[:2]):
        assert groups[start:end+1].unique().numel()==1
        assert torch.equal(positions[start:end+1],torch.arange(end-start+1))
        assert groups[end+1]==-1  # inter-object relation
    dots=(ids==t.chain_separator_id()).nonzero().flatten().tolist()
    allowed=model.layer_allowed(0,b['chain_ids'],b['position_ids'],b['attention_mask'])
    for dot in dots:
        assert allowed[0,dot-1,dot+1]
        assert groups[dot-1]==groups[dot+1]
    assert not allowed[0,starts[0],starts[1]]
    assert not allowed[0,starts[1],starts[2]]
    assert model.layer_allowed(1,b['chain_ids'],b['position_ids'],b['attention_mask'])[0,starts[0],starts[2]]


@pytest.mark.parametrize('roles,task', [
    (('antibody_heavy','antibody_light'),'antibody'),
    (('tcr_beta','tcr_alpha'),'tcr')])
def test_historical_sum_capacity_and_free_split(roles,task):
    import json
    from pathlib import Path
    from dllm.pipelines.immune_llada.data.grammar import FIXED_HEAVY_DECODER_SLOTS, FIXED_LIGHT_DECODER_SLOTS
    root=Path(__file__).resolve().parents[3]
    config=ImmuneConfig(**json.loads((root/'configs/models/v7_v5_layout_chain_global_20261007/research_config.json').read_text())['base'])
    assert config.full_capacity == FIXED_HEAVY_DECODER_SLOTS + FIXED_LIGHT_DECODER_SLOTS == 302
    c=ImmuneCollator.from_config(config)
    def row(a,b):
        return BioSeqRecord([BioSeqChain('A'*a,roles[0]),BioSeqChain('C'*b,roles[1])],task,'test')
    for a,b in ((166,134),(200,100)):
        batch=c([row(a,b)])
        slots=batch['target_slot_mask']
        assert slots.sum()==302
        assert batch['eligible_mask'][slots].sum()==302
        assert int(batch['input_ids'][slots][-1])==c.tokenizer.special_id('<protd>')
        assert batch['chain_ids'][slots].unique().numel()==1
    with pytest.raises(ValueError,match='object_capacity_overflow'):
        c([row(167,134)])
    g=ImmuneCollator.from_config(config,mode='generation')
    batch=g([row(166,134)])
    assert batch['input_ids'][batch['target_slot_mask']].eq(c.tokenizer.mask_token_id).all()
