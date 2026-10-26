"""Use the isolated teacher Python: python -B -m pytest THIS_FILE -q.

Verify multichain assembly against the installed official atom featurizer.
The normal pllm environment skips these tests because it has Transformers 4.
"""
import pytest
import torch

protein_utils=pytest.importorskip("transformers.models.esmfold2.protein_utils")
from scripts.research.stage1_teacher import prepare_multichain_features, trunk_arguments, fast_teacher_class


def test_single_chain_is_exactly_official_featurization():
    sequence="ACDEFGHIKLMNPQRSTVWY"
    official=protein_utils.prepare_protein_features(sequence)
    assembled=prepare_multichain_features([sequence])
    assert official.keys() == assembled.keys()
    assert all(torch.equal(v,assembled[k]) for k,v in official.items())


def test_multichain_atom_offsets_entities_and_local_residue_indices():
    sequences=["ACDEFGHIKLMNPQRSTVWY","GGH","ACDEFGHIKLMNPQRSTVWY"]
    features=prepare_multichain_features(sequences)
    assert features["asym_id"][0].tolist() == [0]*20+[1]*3+[2]*20
    assert features["entity_id"][0].tolist() == [1]*20+[2]*3+[1]*20
    assert features["sym_id"][0].tolist() == [0]*23+[1]*20
    assert features["residue_index"][0].tolist() == list(range(20))+list(range(3))+list(range(20))
    assert features["token_index"][0].tolist() == list(range(43))
    assert not features["token_bonds"].any()
    token_offset,atom_offset=0,0
    for sequence in sequences:
        part=protein_utils.prepare_protein_features(sequence)
        n=int(part["atom_attention_mask"].sum())
        assert torch.equal(features["atom_to_token"][:,atom_offset:atom_offset+n],part["atom_to_token"][:,:n]+token_offset)
        assert torch.equal(features["ref_space_uid"][:,atom_offset:atom_offset+n],part["ref_space_uid"][:,:n]+token_offset)
        assert torch.equal(features["distogram_atom_idx"][:,token_offset:token_offset+len(sequence)],part["distogram_atom_idx"]+atom_offset)
        atom_offset+=n
        token_offset+=len(sequence)
    assert features["ref_pos"].shape[1] % 32 == 0
    assert not features["atom_attention_mask"][:,atom_offset:].any()
    arguments=trunk_arguments(features)
    assert "distogram_atom_idx" not in arguments and "atom_inputs" in arguments
    assert arguments["atom_inputs"].ref_pos is features["ref_pos"]


def test_fast_disabled_msa_has_no_random_parameters_or_calls():
    from transformers import EsmFold2Config
    config=EsmFold2Config(esmc_config={"model_type":"esmc","hidden_size":32,
        "intermediate_size":64,"num_hidden_layers":1,"num_attention_heads":4,
        "num_key_value_heads":4},msa_encoder={"enabled":False},type="release")
    with torch.device("meta"):
        model=fast_teacher_class()(config)
    assert model.msa_encoder is None
    assert not any(name.startswith("msa_encoder.") for name in model.state_dict())
    assert model._build_msa_kwargs(msa=torch.ones(1,1,3)) is None
    # Release semantics retain the sequence-derived profile and nested per-loop
    # LM dropout, independent of the experimental model's top-level flags.
    assert model.config.lm_encoder.lm_dropout == .25
    assert model.config.lm_encoder.per_loop_lm_dropout is True
    features=prepare_multichain_features(["ACD"])
    res_type,profile,_,_,_,_=model._prepare_features(**{name:features[name] for name in (
        "res_type","attention_mask","msa","msa_attention_mask","deletion_mean",
        "ref_element","ref_atom_name_chars","atom_attention_mask","atom_to_token")})
    assert torch.equal(res_type,profile)
