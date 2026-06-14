from pathlib import Path

import torch

from dllm.pipelines.bioseq import (
    BioSeqCollator,
    BioSeqDiffusionConfig,
    BioSeqModelConfig,
    Esm2ProteinTokenizer,
    NoEncoderBioDiffusionModel,
    OphiuchusAbCollator,
    ProteinTokenizer,
    compute_diffusion_loss,
    load_jsonl,
    nanobody_row_to_example,
    oas_paired_row_to_example,
    ophiuchus_ab_model_config,
    ots_paired_row_to_example,
    processed_json_row_to_example,
    write_jsonl,
)


def test_bioseq_source_does_not_import_dllm_core():
    root = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq")
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text()
        if "dllm.core" in text:
            offenders.append(str(path))
    assert offenders == []


def test_collator_builds_chain_ids_and_loss_mask():
    tokenizer = ProteinTokenizer()
    collator = BioSeqCollator(tokenizer=tokenizer)
    batch = collator(
        [
            {
                "chains": ["ACD", "EFG"],
                "task_type": "antibody",
            }
        ]
    )

    assert batch["input_ids"].shape == batch["chain_ids"].shape
    assert batch["loss_mask"].sum().item() == 6
    assert set(batch["chain_ids"][0].tolist()) == {-1, 0, 1}


def test_ophiuchus_tokenizer_and_collator_are_esm2_compatible():
    tokenizer = Esm2ProteinTokenizer()
    assert tokenizer.vocab_size == 33
    assert tokenizer.cls_token_id == 0
    assert tokenizer.pad_token_id == 1
    assert tokenizer.eos_token_id == 2
    assert tokenizer.mask_token_id == 32
    assert tokenizer.encode_chain("ACD")[0] == [0, 5, 23, 13, 2]

    collator = OphiuchusAbCollator(tokenizer=tokenizer)
    batch = collator([{"chains": ["ACD", "EFG"], "task_type": "antibody"}])
    assert batch["input_ids"].shape == (1, 278)
    assert batch["chain_ids"][0, 0].item() == 0
    assert batch["chain_ids"][0, 149].item() == 0
    assert batch["chain_ids"][0, 150].item() == 1
    assert batch["input_ids"][0, 0].item() == tokenizer.cls_token_id
    assert batch["input_ids"][0, 150].item() == tokenizer.cls_token_id
    assert batch["loss_mask"].sum().item() == 6
    assert not batch["attention_mask"][0, 5]


def test_model_forward_and_diffusion_loss():
    tokenizer = ProteinTokenizer()
    collator = BioSeqCollator(tokenizer=tokenizer)
    batch = collator(
        [
            {"chains": ["ACDE", "FGHI"], "task_type": "antibody"},
            {"chains": ["KLMN", "PQRS"], "task_type": "ppi"},
        ]
    )
    config = BioSeqModelConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = NoEncoderBioDiffusionModel(config)
    output = model(
        input_ids=batch["input_ids"],
        chain_ids=batch["chain_ids"],
        attention_mask=batch["attention_mask"],
    )
    assert output.logits.shape[:2] == batch["input_ids"].shape
    assert output.logits.shape[-1] == tokenizer.vocab_size

    loss_result = compute_diffusion_loss(
        model=model,
        batch=batch,
        diffusion_config=BioSeqDiffusionConfig(mask_token_id=tokenizer.mask_token_id),
    )
    assert torch.isfinite(loss_result.loss)
    assert loss_result.masked_mask.any()


def test_ophiuchus_preset_can_be_scaled_down_for_local_forward():
    tokenizer = Esm2ProteinTokenizer()
    collator = OphiuchusAbCollator(tokenizer=tokenizer, chain_lengths=(8, 8))
    batch = collator([{"chains": ["ACDE", "FGHI"], "task_type": "antibody"}])
    preset = ophiuchus_ab_model_config()
    config = BioSeqModelConfig(
        vocab_size=preset.vocab_size,
        hidden_size=40,
        num_hidden_layers=2,
        num_attention_heads=5,
        intermediate_size=80,
        max_position_embeddings=16,
        dropout=preset.dropout,
        pad_token_id=preset.pad_token_id,
        mask_token_id=preset.mask_token_id,
        use_multimer_attention=preset.use_multimer_attention,
        token_dropout=preset.token_dropout,
        use_position_embeddings=preset.use_position_embeddings,
    )
    model = NoEncoderBioDiffusionModel(config)
    result = compute_diffusion_loss(
        model=model,
        batch=batch,
        diffusion_config=BioSeqDiffusionConfig(mask_token_id=tokenizer.mask_token_id),
    )
    assert torch.isfinite(result.loss)
    assert result.logits.shape == (1, 16, tokenizer.vocab_size)


def test_bioseq_adapters_normalize_local_data_schemas(tmp_path):
    oas = oas_paired_row_to_example(
        {
            "cleaned_chain1_seq": "EVQLVESGGGLVQPGGSLRLSCAAS",
            "cleaned_chain2_seq": "DIQMTQSPSSLSASVGDRVTITC",
            "chain1_anarci_type": "H",
            "chain2_anarci_type": "L",
            "chain1_CDR1": "GFTF",
            "chain2_CDR1": "RASQ",
            "source_file": "OAS",
            "split": "train",
            "cluster_id": "cluster_1",
        }
    )
    assert oas.task_type == "antibody"
    assert oas.chain_roles == ["antibody_heavy", "antibody_light"]
    assert oas.regions["0"]["CDR1"] == "GFTF"

    ots = ots_paired_row_to_example(
        {
            "cleaned_chain1_seq": "CASSLGQETQYF",
            "cleaned_chain2_seq": "CAVRDSNYQLIW",
            "chain1_anarci_type": "B",
            "chain2_anarci_type": "A",
            "source_file": "OTS",
            "split": "valid",
        }
    )
    assert ots.task_type == "tcr"
    assert ots.chain_roles == ["tcr_beta", "tcr_alpha"]

    nanobody = nanobody_row_to_example(
        {
            "cleaned_seq": "QVQLVESGGGLVQAGGSLRLSCAAS",
            "source": "vhhcorpus",
            "CDR3": "AADPLM",
            "split": "holdout",
        }
    )
    assert nanobody.task_type == "antibody"
    assert nanobody.chain_roles == ["nanobody_vhh"]

    ppi = processed_json_row_to_example(
        {
            "chains": ["ACDEFGHIK", "LMNPQRSTV"],
            "types": ["other", "other"],
            "targets": [0, 1],
            "source": "ppi",
        }
    )
    assert ppi.task_type == "ppi"

    tcr_pmhc = processed_json_row_to_example(
        {
            "chains": ["CASSLGQETQYF", "GLCTLVAML", "MHCSEQ"],
            "types": ["beta", "peptide", "mhc"],
            "targets": [0],
            "source": "vdjdb",
        }
    )
    assert tcr_pmhc.task_type == "tcr_pmhc"
    assert tcr_pmhc.chain_roles == ["tcr_beta", "peptide", "mhc"]

    output_path = tmp_path / "bioseq.jsonl"
    written = write_jsonl([oas, ots, nanobody, ppi, tcr_pmhc], output_path)
    rows = load_jsonl(output_path)
    assert written == 5
    assert rows[0]["schema_version"] == "bioseq.v1"
    batch = BioSeqCollator(tokenizer=ProteinTokenizer())(rows[:2])
    assert batch["input_ids"].shape[0] == 2
    assert batch["loss_mask"].any()
