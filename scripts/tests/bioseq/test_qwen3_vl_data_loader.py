from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dllm.pipelines.qwen3_vl_arch.data import (
    BioSeqChain,
    BioSeqQwenDataCollator,
    BioSeqRecord,
    BioSeqViewSampler,
    CsvBioSeqSource,
    CsvSourceConfig,
    Esm2SequenceTokenizer,
    HuggingFaceEsmTokenizerAdapter,
    JsonlSourceConfig,
    ProcessedJsonlSource,
    SequentialMultiSourceDataset,
    SourceWithWeight,
    TaskHomogeneousBatchDataset,
    WeightedMixtureDataset,
    bioseq_record_fingerprint,
    bioseq_task_group,
    oas_row_to_record,
    ots_row_to_record,
)
from dllm.pipelines.qwen3_vl_arch.data.sources import default_source_configs


def test_oas_and_ots_sources_orient_chain_roles(tmp_path):
    oas = tmp_path / "oas.csv"
    oas.write_text(
        "cleaned_chain1_seq,cleaned_chain2_seq,chain1_anarci_type,chain2_anarci_type,"
        "chain1_FR1,chain1_CDR1,chain1_FR2,chain1_CDR2,chain1_FR3,chain1_CDR3,chain1_FR4,"
        "chain2_FR1,chain2_CDR1,chain2_FR2,chain2_CDR2,chain2_FR3,chain2_CDR3,chain2_FR4\n"
        "DIQMTQSPSS,QVQLVQSGAE,L,H,D,IQ,M,TQ,SP,SS,,QV,QL,VQ,SG,AE,,\n"
    )
    oas_source = CsvBioSeqSource(CsvSourceConfig("oas", oas, oas_row_to_record))
    oas_record = next(iter(oas_source))
    assert oas_record.sequences == ["QVQLVQSGAE", "DIQMTQSPSS"]
    assert oas_record.chain_roles == ["antibody_heavy", "antibody_light"]

    ots = tmp_path / "ots.csv"
    ots.write_text(
        "cleaned_chain1_seq,cleaned_chain2_seq,chain1_anarci_type,chain2_anarci_type\n"
        "AAAAAA,CCCCCC,A,B\n"
    )
    ots_source = CsvBioSeqSource(CsvSourceConfig("ots", ots, ots_row_to_record))
    ots_record = next(iter(ots_source))
    assert ots_record.sequences == ["CCCCCC", "AAAAAA"]
    assert ots_record.chain_roles == ["tcr_beta", "tcr_alpha"]


def test_oas_label_schema_maps_heavy_light_regions_and_default_path():
    record = oas_row_to_record(
        {
            "cleaned_h_sequence": "AAACCCGGG",
            "cleaned_l_sequence": "TTTDDDFFF",
            "l_locus": "K",
            "h_fwr1": "AAA",
            "h_cdr1": "CCC",
            "h_fwr2": "GGG",
            "l_fwr1": "TTT",
            "l_cdr1": "DDD",
            "l_fwr2": "FFF",
            "h_v_call": "IGHV3-23*01",
            "h_j_call": "IGHJ4*02",
            "l_v_call": "IGKV1-5*03",
            "l_j_call": "IGKJ1*01",
            "source": "OAS",
            "split": "train",
        },
        split="train",
    )

    assert record is not None
    assert record.sequences == ["AAACCCGGG", "TTTDDDFFF"]
    assert record.chain_roles == ["antibody_heavy", "antibody_light"]
    assert record.chains[0].regions == {"FR1": "AAA", "CDR1": "CCC", "FR2": "GGG"}
    assert record.chains[1].regions == {"FR1": "TTT", "CDR1": "DDD", "FR2": "FFF"}
    assert record.chains[0].metadata["h_v_call"] == "IGHV3-23*01"
    assert record.chains[1].metadata["l_locus"] == "K"

    oas_config = next(config for config in default_source_configs(split="valid") if config.name == "oas")
    assert oas_config.path.name == "cleaned_merged_data_step_clustered_valid_oas_label.csv"


def test_processed_jsonl_source_keeps_multichain_roles(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "chains": ["CAVRPTSGGSYIPTF", "CASSHTGFQGELFF", "SLLMWITQV"],
                "types": ["alpha", "beta", "antigen"],
                "targets": [0, 1],
                "source": "vdjdb",
            }
        )
        + "\n"
    )
    source = ProcessedJsonlSource(JsonlSourceConfig("processed", path))
    record = next(iter(source))
    assert record.task_type == "tcr_epitope"
    assert record.chain_roles == ["tcr_alpha", "tcr_beta", "antigen"]
    assert record.metadata["targets"] == [0, 1]


def test_processed_ppi_source_normalizes_partner_roles(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "chains": ["AAAA", "CCCC"],
                "types": ["other", "other"],
                "targets": [0, 1],
                "source": "ppi",
            }
        )
        + "\n"
    )
    source = ProcessedJsonlSource(JsonlSourceConfig("processed", path))
    record = next(iter(source))
    assert record.task_type == "ppi"
    assert record.chain_roles == ["protein_a", "protein_b"]


def test_sequential_and_weighted_mixture_stream_records(tmp_path):
    oas = tmp_path / "oas.csv"
    oas.write_text(
        "cleaned_chain1_seq,cleaned_chain2_seq,chain1_anarci_type,chain2_anarci_type\n"
        "QVQLVQSGAE,DIQMTQSPSS,H,L\n"
    )
    ots = tmp_path / "ots.csv"
    ots.write_text(
        "cleaned_chain1_seq,cleaned_chain2_seq,chain1_anarci_type,chain2_anarci_type\n"
        "CCCCCC,AAAAAA,B,A\n"
    )
    oas_source = CsvBioSeqSource(CsvSourceConfig("oas", oas, oas_row_to_record))
    ots_source = CsvBioSeqSource(CsvSourceConfig("ots", ots, ots_row_to_record))

    seq_records = list(SequentialMultiSourceDataset([oas_source, ots_source]))
    assert [record.task_type for record in seq_records] == ["antibody", "tcr"]

    mixed = WeightedMixtureDataset(
        [SourceWithWeight(oas_source, 1.0), SourceWithWeight(ots_source, 1.0)],
        epoch_size=4,
        seed=7,
    )
    assert len(list(mixed)) == 4


def test_task_homogeneous_batch_dataset_groups_tasks_without_default_deduplication():
    ab_a = BioSeqRecord(
        chains=[BioSeqChain("HHHH", "antibody_heavy"), BioSeqChain("LLLL", "antibody_light")],
        task_type="antibody",
        source="unit",
    )
    ab_b = BioSeqRecord(
        chains=[BioSeqChain("HHHA", "antibody_heavy"), BioSeqChain("LLLA", "antibody_light")],
        task_type="antibody",
        source="unit",
    )
    tcr_a = BioSeqRecord(
        chains=[BioSeqChain("AAAA", "tcr_alpha"), BioSeqChain("BBBB", "tcr_beta")],
        task_type="tcr",
        source="unit",
    )
    tcr_b = BioSeqRecord(
        chains=[BioSeqChain("AAAC", "tcr_alpha"), BioSeqChain("BBBC", "tcr_beta")],
        task_type="tcr",
        source="unit",
    )
    batches = list(TaskHomogeneousBatchDataset([ab_a, tcr_a, ab_a, ab_b, tcr_b], batch_size=2, drop_last=True))
    assert [[bioseq_task_group(record) for record in batch] for batch in batches] == [
        ["antibody", "antibody"],
        ["tcr", "tcr"],
    ]
    assert [bioseq_record_fingerprint(record) for record in batches[0]].count(bioseq_record_fingerprint(ab_a)) == 2


def test_task_homogeneous_batch_dataset_can_deduplicate_as_defensive_option():
    ab_a = BioSeqRecord(
        chains=[BioSeqChain("HHHH", "antibody_heavy"), BioSeqChain("LLLL", "antibody_light")],
        task_type="antibody",
        source="unit",
    )
    ab_b = BioSeqRecord(
        chains=[BioSeqChain("HHHA", "antibody_heavy"), BioSeqChain("LLLA", "antibody_light")],
        task_type="antibody",
        source="unit",
    )
    batches = list(
        TaskHomogeneousBatchDataset(
            [ab_a, ab_a, ab_b],
            batch_size=2,
            drop_last=False,
            deduplicate_within_batch=True,
        )
    )
    assert len(batches) == 1
    assert [bioseq_record_fingerprint(record) for record in batches[0]] == [
        bioseq_record_fingerprint(ab_a),
        bioseq_record_fingerprint(ab_b),
    ]


def test_collator_samples_one_generation_view_per_batch():
    records = [
        BioSeqRecord(
            chains=[
                BioSeqChain("AAACCCGGG", "antibody_heavy", regions={"FR1": "AAA", "CDR1": "CCC", "FR2": "GGG"}),
                BioSeqChain("TTTDDDFFF", "antibody_light", regions={"FR1": "TTT", "CDR1": "DDD", "FR2": "FFF"}),
                BioSeqChain("MMMMM", "antigen"),
            ],
            task_type="antibody_antigen",
            source="unit",
        ),
        BioSeqRecord(
            chains=[
                BioSeqChain("VVVWWWYYY", "antibody_heavy", regions={"FR1": "VVV", "CDR1": "WWW", "FR2": "YYY"}),
                BioSeqChain("QQQEEEKKK", "antibody_light", regions={"FR1": "QQQ", "CDR1": "EEE", "FR2": "KKK"}),
                BioSeqChain("NNNNN", "antigen"),
            ],
            task_type="antibody_antigen",
            source="unit",
        ),
    ]
    collator = BioSeqQwenDataCollator(
        view_sampler=BioSeqViewSampler(
            allowed_views=["antigen_to_antibody", "antigen_fr_to_cdr"],
            seed=0,
        ),
        single_view_per_batch=True,
        require_homogeneous_task=True,
    )
    batch = collator(records)
    assert len(set(batch["view_names"])) == 1
    assert batch["view_names"][0] in {"antigen_to_antibody", "antigen_fr_to_cdr"}


def test_collator_can_reject_mixed_task_groups():
    records = [
        BioSeqRecord(
            chains=[BioSeqChain("HHHH", "antibody_heavy"), BioSeqChain("LLLL", "antibody_light")],
            task_type="antibody",
            source="unit",
        ),
        BioSeqRecord(
            chains=[BioSeqChain("AAAA", "tcr_alpha"), BioSeqChain("BBBB", "tcr_beta")],
            task_type="tcr",
            source="unit",
        ),
    ]
    collator = BioSeqQwenDataCollator(require_homogeneous_task=True)
    try:
        collator(records)
    except ValueError as exc:
        assert "mixed task groups" in str(exc)
    else:
        raise AssertionError("Expected mixed task groups to be rejected")


def test_collator_default_allows_mixed_tasks_with_per_record_views():
    records = [
        BioSeqRecord(
            chains=[BioSeqChain("HHHH", "antibody_heavy"), BioSeqChain("LLLL", "antibody_light")],
            task_type="antibody",
            source="unit",
        ),
        BioSeqRecord(
            chains=[
                BioSeqChain("HHHA", "antibody_heavy"),
                BioSeqChain("LLLA", "antibody_light"),
                BioSeqChain("ANTG", "antigen"),
            ],
            task_type="antibody_antigen",
            source="unit",
        ),
    ]
    collator = BioSeqQwenDataCollator(
        view_sampler=BioSeqViewSampler(allowed_views=["antigen_to_antibody"], seed=0),
    )

    batch = collator(records)

    assert batch["view_names"] == ["full_denoise", "antigen_to_antibody"]
    assert batch["task_groups"] == ["antibody", "antibody_antigen"]
    assert batch["task_types"] == ["antibody", "antibody_antigen"]
    assert batch["input_ids"].shape[0] == 2
    assert batch["diffusion_loss_mask"][0].sum().item() == len("HHHH") + len("LLLL")
    assert batch["diffusion_loss_mask"][1].sum().item() == len("HHHA") + len("LLLA")
    assert batch["fixed_context_mask"][1].sum().item() == len("ANTG")


def test_collator_falls_back_when_truncation_removes_view_targets():
    record = BioSeqRecord(
        chains=[
            BioSeqChain(
                "A" * 20 + "CCC" + "G" * 5,
                "antibody_heavy",
                regions={"FR1": "A" * 20, "CDR1": "CCC", "FR2": "G" * 5},
            ),
        ],
        task_type="antibody",
        source="unit",
    )
    collator = BioSeqQwenDataCollator(
        max_chain_length=16,
        view_sampler=BioSeqViewSampler(allowed_views=["single_cdr"], seed=0),
    )

    batch = collator([record])

    assert batch["view_names"] == ["full_denoise"]
    assert batch["diffusion_loss_mask"].sum().item() == 14


def test_view_sampler_uses_full_denoise_probability_for_single_records():
    record = BioSeqRecord(
        chains=[BioSeqChain("HHHH", "antibody_heavy"), BioSeqChain("LLLL", "antibody_light")],
        task_type="antibody",
        source="unit",
    )

    assert BioSeqViewSampler(seed=0).sample(record).name == "full_denoise"

    full_sampler = BioSeqViewSampler(seed=0, full_denoise_probability=1.0)
    assert [full_sampler.sample(record).name for _ in range(5)] == ["full_denoise"] * 5

    condition_sampler = BioSeqViewSampler(seed=0, full_denoise_probability=0.0)
    condition_views = {condition_sampler.sample(record).name for _ in range(10)}
    assert condition_views <= {"heavy_to_light", "light_to_heavy"}
    assert condition_views


def test_view_sampler_uses_full_denoise_probability_for_batches():
    records = [
        BioSeqRecord(
            chains=[BioSeqChain("HHHH", "antibody_heavy"), BioSeqChain("LLLL", "antibody_light")],
            task_type="antibody",
            source="unit",
        ),
        BioSeqRecord(
            chains=[BioSeqChain("HHHA", "antibody_heavy"), BioSeqChain("LLLA", "antibody_light")],
            task_type="antibody",
            source="unit",
        ),
    ]

    full_sampler = BioSeqViewSampler(
        allowed_views=["full_denoise", "heavy_to_light"],
        seed=0,
        full_denoise_probability=1.0,
    )
    assert [view.name for view in full_sampler.sample_batch(records)] == ["full_denoise", "full_denoise"]

    condition_sampler = BioSeqViewSampler(
        allowed_views=["full_denoise", "heavy_to_light"],
        seed=0,
        full_denoise_probability=0.0,
    )
    assert [view.name for view in condition_sampler.sample_batch(records)] == ["heavy_to_light", "heavy_to_light"]


def test_view_sampler_falls_back_to_full_denoise_without_condition_views():
    record = BioSeqRecord(
        chains=[BioSeqChain("AAAA", "protein_a"), BioSeqChain("CCCC", "protein_b")],
        task_type="ppi",
        source="unit",
    )
    sampler = BioSeqViewSampler(seed=0, full_denoise_probability=0.0)
    assert sampler.sample(record).name == "full_denoise"


def test_collator_builds_esm_compatible_encoder_and_diffusion_masks():
    record = oas_row_to_record(
        {
            "cleaned_chain1_seq": "HHHCCCAA",
            "cleaned_chain2_seq": "LLLCCCDD",
            "chain1_anarci_type": "H",
            "chain2_anarci_type": "L",
        },
        split="train",
    )
    assert record is not None

    collator = BioSeqQwenDataCollator(
        tokenizer=Esm2SequenceTokenizer(),
        view_sampler=BioSeqViewSampler(allowed_views=["heavy_to_light"]),
    )
    batch = collator([record])
    assert batch["input_ids"].shape[0] == 1
    assert batch["encoder_input_ids"].shape[:2] == (1, 2)
    assert batch["input_ids"][0, 0].item() == 0  # ESM <cls>
    assert batch["encoder_input_ids"][0, 0, 0].item() == 0
    assert batch["view_names"] == ["heavy_to_light"]
    assert batch["diffusion_loss_mask"].sum().item() == len("LLLCCCDD")
    assert batch["fixed_context_mask"].sum().item() == len("HHHCCCAA")
    assert batch["visible_mask"].sum().item() > batch["fixed_context_mask"].sum().item()


def test_antigen_to_antibody_view_targets_heavy_and_light_only():
    record = BioSeqRecord(
        chains=[
            BioSeqChain("HHHH", "antibody_heavy"),
            BioSeqChain("LLL", "antibody_light"),
            BioSeqChain("AAAAA", "antigen"),
        ],
        task_type="antibody_antigen",
        source="unit",
    )
    collator = BioSeqQwenDataCollator(view_sampler=BioSeqViewSampler(allowed_views=["antigen_to_antibody"]))
    batch = collator([record])
    assert batch["view_names"] == ["antigen_to_antibody"]
    assert batch["diffusion_loss_mask"].sum().item() == len("HHHH") + len("LLL")
    assert batch["fixed_context_mask"].sum().item() == len("AAAAA")


def test_heavy_antigen_to_light_view_targets_light_only():
    record = BioSeqRecord(
        chains=[
            BioSeqChain("HHHH", "antibody_heavy"),
            BioSeqChain("LLL", "antibody_light"),
            BioSeqChain("AAAAA", "antigen"),
        ],
        task_type="antibody_antigen",
        source="unit",
    )
    collator = BioSeqQwenDataCollator(view_sampler=BioSeqViewSampler(allowed_views=["heavy_antigen_to_light"]))
    batch = collator([record])
    assert batch["view_names"] == ["heavy_antigen_to_light"]
    assert batch["diffusion_loss_mask"].sum().item() == len("LLL")
    assert batch["fixed_context_mask"].sum().item() == len("HHHH") + len("AAAAA")


def test_antigen_to_nanobody_view_targets_vhh_only():
    record = BioSeqRecord(
        chains=[
            BioSeqChain("NNNN", "nanobody_vhh"),
            BioSeqChain("AAAAA", "antigen"),
        ],
        task_type="nanobody_antigen",
        source="unit",
    )
    collator = BioSeqQwenDataCollator(view_sampler=BioSeqViewSampler(allowed_views=["antigen_to_nanobody"]))
    batch = collator([record])
    assert batch["view_names"] == ["antigen_to_nanobody"]
    assert batch["diffusion_loss_mask"].sum().item() == len("NNNN")
    assert batch["fixed_context_mask"].sum().item() == len("AAAAA")


def test_antigen_fr_to_cdr_view_keeps_antigen_and_fr_fixed():
    record = BioSeqRecord(
        chains=[
            BioSeqChain(
                "AAACCCGGG",
                "antibody_heavy",
                regions={"FR1": "AAA", "CDR1": "CCC", "FR2": "GGG"},
            ),
            BioSeqChain(
                "TTTDDDFFF",
                "antibody_light",
                regions={"FR1": "TTT", "CDR1": "DDD", "FR2": "FFF"},
            ),
            BioSeqChain("MMMMM", "antigen"),
        ],
        task_type="antibody_antigen",
        source="unit",
    )
    collator = BioSeqQwenDataCollator(view_sampler=BioSeqViewSampler(allowed_views=["antigen_fr_to_cdr"]))
    batch = collator([record])
    assert batch["view_names"] == ["antigen_fr_to_cdr"]
    assert batch["diffusion_loss_mask"].sum().item() == len("CCCDDD")
    assert batch["fixed_context_mask"].sum().item() == len("AAAGGGTTTFFFMMMMM")


def test_default_view_profile_is_task_specific_for_antibody_antigen():
    record = BioSeqRecord(
        chains=[
            BioSeqChain("AAACCCGGG", "antibody_heavy", regions={"FR1": "AAA", "CDR1": "CCC", "FR2": "GGG"}),
            BioSeqChain("TTTDDDFFF", "antibody_light", regions={"FR1": "TTT", "CDR1": "DDD", "FR2": "FFF"}),
            BioSeqChain("MMMMM", "antigen"),
        ],
        task_type="antibody_antigen",
        source="unit",
    )
    views = BioSeqViewSampler().default_views_for_record(record)
    assert "antigen_to_antibody" in views
    assert "antigen_fr_to_cdr" in views
    assert "mhc_to_peptide_tcr" not in views
    assert "antibody_to_antigen" not in views


def test_esm2_tokenizer_matches_local_vocab_snapshot():
    vocab_path = Path("/c20250601/mj/model_weights/esm2/esm2_t33_650M_UR50D/vocab.txt")
    tokenizer = Esm2SequenceTokenizer()
    if not vocab_path.is_file():
        pytest.skip(f"local ESM2 vocab snapshot is not mounted: {vocab_path}")
    assert tuple(vocab_path.read_text().splitlines()) == tokenizer.tokens
    assert tokenizer.cls_token_id == 0
    assert tokenizer.pad_token_id == 1
    assert tokenizer.eos_token_id == 2
    assert tokenizer.mask_token_id == 32


def test_esmc_tokenizer_adapter_loads_local_tokenizer_json_without_auto_tokenizer():
    tokenizer_path = Path(
        "/vepfs-mlp2/c20250601/251105016/project/dllm_test/model_weights/esmc/ESMC-300M"
    )
    tokenizer = HuggingFaceEsmTokenizerAdapter.from_pretrained(tokenizer_path, local_files_only=True)

    assert tokenizer.cls_token_id == 0
    assert tokenizer.pad_token_id == 1
    assert tokenizer.eos_token_id == 2
    assert tokenizer.mask_token_id == 32
    assert tokenizer.vocab_size == 33
    token_ids, residue_mask = tokenizer.encode_chain("ACD", max_length=5)
    assert token_ids == [0, 5, 23, 13, 2]
    assert residue_mask == [0, 1, 1, 1, 0]


def test_full_denoise_keeps_antigen_and_pmhc_context_fixed():
    record = BioSeqRecord(
        chains=[
            BioSeqChain("AAA", "tcr_alpha"),
            BioSeqChain("BBBB", "tcr_beta"),
            BioSeqChain("CC", "peptide"),
            BioSeqChain("DDDDD", "mhc"),
            BioSeqChain("EEEEEE", "antigen"),
        ],
        task_type="tcr_pmhc",
        source="unit",
        metadata={"targets": [0, 1, 2, 3, 4]},
    )
    collator = BioSeqQwenDataCollator(view_sampler=BioSeqViewSampler(allowed_views=["full_denoise"]))
    batch = collator([record])
    assert batch["view_names"] == ["full_denoise"]
    assert batch["diffusion_loss_mask"].sum().item() == len("AAABBBB")
    assert batch["fixed_context_mask"].sum().item() == len("CC") + len("DDDDD") + len("EEEEEE")


def test_mhc_to_peptide_tcr_view_targets_tcr_and_peptide_only():
    record = BioSeqRecord(
        chains=[
            BioSeqChain("AAA", "tcr_alpha"),
            BioSeqChain("BBBB", "tcr_beta"),
            BioSeqChain("CC", "peptide"),
            BioSeqChain("DDDDD", "mhc"),
        ],
        task_type="tcr_pmhc",
        source="unit",
    )
    collator = BioSeqQwenDataCollator(view_sampler=BioSeqViewSampler(allowed_views=["mhc_to_peptide_tcr"]))
    batch = collator([record])
    assert batch["view_names"] == ["mhc_to_peptide_tcr"]
    assert batch["diffusion_loss_mask"].sum().item() == len("AAA") + len("BBBB") + len("CC")
    assert batch["fixed_context_mask"].sum().item() == len("DDDDD")


def test_tcr_mhc_to_peptide_view_targets_peptide_only():
    record = BioSeqRecord(
        chains=[
            BioSeqChain("AAA", "tcr_alpha"),
            BioSeqChain("BBBB", "tcr_beta"),
            BioSeqChain("CC", "peptide"),
            BioSeqChain("DDDDD", "mhc"),
        ],
        task_type="tcr_pmhc",
        source="unit",
    )
    collator = BioSeqQwenDataCollator(view_sampler=BioSeqViewSampler(allowed_views=["tcr_mhc_to_peptide"]))
    batch = collator([record])
    assert batch["view_names"] == ["tcr_mhc_to_peptide"]
    assert batch["diffusion_loss_mask"].sum().item() == len("CC")
    assert batch["fixed_context_mask"].sum().item() == len("AAA") + len("BBBB") + len("DDDDD")


def test_pmhc_to_tcr_view_targets_alpha_and_beta_only():
    record = BioSeqRecord(
        chains=[
            BioSeqChain("AAA", "tcr_alpha"),
            BioSeqChain("BBBB", "tcr_beta"),
            BioSeqChain("CC", "peptide"),
            BioSeqChain("DDDDD", "mhc"),
        ],
        task_type="tcr_pmhc",
        source="unit",
    )
    collator = BioSeqQwenDataCollator(view_sampler=BioSeqViewSampler(allowed_views=["pmhc_to_tcr"]))
    batch = collator([record])
    assert batch["view_names"] == ["pmhc_to_tcr"]
    assert batch["diffusion_loss_mask"].sum().item() == len("AAA") + len("BBBB")
    assert batch["fixed_context_mask"].sum().item() == len("CC") + len("DDDDD")


def test_pmhc_fr_to_cdr_view_keeps_pmhc_and_tcr_fr_fixed():
    record = BioSeqRecord(
        chains=[
            BioSeqChain("AAACCCGGG", "tcr_alpha", regions={"FR1": "AAA", "CDR1": "CCC", "FR2": "GGG"}),
            BioSeqChain("TTTDDDFFF", "tcr_beta", regions={"FR1": "TTT", "CDR1": "DDD", "FR2": "FFF"}),
            BioSeqChain("PPPP", "peptide"),
            BioSeqChain("MMMMM", "mhc"),
        ],
        task_type="tcr_pmhc",
        source="unit",
    )
    collator = BioSeqQwenDataCollator(view_sampler=BioSeqViewSampler(allowed_views=["pmhc_fr_to_cdr"]))
    batch = collator([record])
    assert batch["view_names"] == ["pmhc_fr_to_cdr"]
    assert batch["diffusion_loss_mask"].sum().item() == len("CCCDDD")
    assert batch["fixed_context_mask"].sum().item() == len("AAAGGGTTTFFFPPPPMMMMM")


def test_region_view_targets_cdr_only():
    record = oas_row_to_record(
        {
            "cleaned_chain1_seq": "AAACCCGGG",
            "cleaned_chain2_seq": "TTTDDDFFF",
            "chain1_anarci_type": "H",
            "chain2_anarci_type": "L",
            "chain1_FR1": "AAA",
            "chain1_CDR1": "CCC",
            "chain1_FR2": "GGG",
            "chain2_FR1": "TTT",
            "chain2_CDR1": "DDD",
            "chain2_FR2": "FFF",
        },
        split="train",
    )
    assert record is not None
    collator = BioSeqQwenDataCollator(view_sampler=BioSeqViewSampler(allowed_views=["fr_to_cdr"]))
    batch = collator([record])
    assert batch["view_names"] == ["fr_to_cdr"]
    assert batch["diffusion_target_mask"].sum().item() == len("CCCDDD")
    assert batch["fixed_context_mask"].sum().item() == len("AAAGGGTTTFFF")
