"""Validate the pinned five-task MINT notebook rebuild.

Run from ``dllm_test`` after activating ``pllm``::

    conda activate pllm
    python -m pytest -q scripts/tests/bioseq/test_mint_official_rebuild.py
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from downstream.mint_tasks.tasks import TASK_CONFIGS, get_task_datasets
from downstream.mint_tasks.finetune_general import (
    SimpleMLP,
    _baseline_provenance,
    _checkpoint_provenance,
    repetition_indices,
)
from downstream.mint_tasks.extract_embeddings import (
    _aligned_truncate_mutation_pairs,
    model_cache_name,
)
from downstream.mint_tasks.prepare_official_data import EXPECTED_OUTPUT_SHA256
from downstream.mint_tasks.run_local_fixed_baselines import (
    BASELINES,
    DEFAULT_PROTOCOL_TAG,
    effective_max_seq_length,
    extraction_command,
    finetune_command,
)
from downstream.mint_tasks.summarize_baselines import (
    collect_paper,
    select_hybrid,
)
from downstream.mint_tasks.split_protocols import (
    MUTATIONALPPI_FOLDS,
    MUTATIONALPPI_FOLDS_MANIFEST,
    MUTATIONALPPI_SPLIT_PROTOCOL_ID,
)
from scripts.downstream.validate_mint_ours_three_ppi import EXPECTED as OURS_THREE_PPI_EXPECTED
from scripts.downstream.extract_mint_embedding_shard import shard_bounds


ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
DATA_ROOT = ROOT / "data" / "downstream" / "mint_official"
MANIFEST = DATA_ROOT / "manifest.json"

EXPECTED_ROWS = {
    "gold": [59260, 163192, 52048],
    "humanppi": [26319, 234, 180],
    "yeastppi": [4945, 95, 394],
    "mutationalppi": [3406],
    "skempi": [6706],
}


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_registry_contains_only_selected_tasks() -> None:
    assert set(TASK_CONFIGS) == {
        "HumanPPI",
        "YeastPPI",
        "SKEMPI",
        "Bernett",
        "MutationalPPI",
    }
    with pytest.raises(KeyError):
        get_task_datasets("Pdb-bind", test_run=True)
    with pytest.raises(KeyError):
        get_task_datasets("MutationalPPI_cs", test_run=True)


def test_embedding_cache_is_namespaced_by_official_data_protocol() -> None:
    name = model_cache_name("esm2_t33_650M_UR50D", True, 3000)
    assert name == "esm2_t33_650M_UR50D_sep_mint_official_06694b7_t3000"
    local_name = model_cache_name(
        "esm2_t33_650M_UR50D",
        True,
        None,
        "localfixed-v1-l2048",
    )
    assert local_name.endswith("_localfixed-v1-l2048")


def test_three_headline_ppi_tasks_match_paper_head_protocol() -> None:
    for task in ("HumanPPI", "YeastPPI", "Bernett"):
        assert TASK_CONFIGS[task]["num_epochs"] == 100
        assert TASK_CONFIGS[task]["hidden_size"] == 640

    probe = SimpleMLP(input_size=1536, output_size=1)
    assert probe.project[0].in_features == 1536
    assert probe.project[0].out_features == 640
    assert probe.project[-1].in_features == 640
    assert probe.project[-1].out_features == 1


def test_three_headline_ppi_split_provenance_is_explicit() -> None:
    for task in ("HumanPPI", "YeastPPI"):
        _, _, _, metadata = get_task_datasets(task, test_run=True, return_metadata=True)
        assert metadata["split_protocol_id"].startswith("mint-public-")
        assert metadata["protocol_alignment"] == "paper_exact_public_fixed_split"
        assert metadata["paper_comparable"] is True
        provenance = _baseline_provenance(SimpleNamespace(), metadata, is_ours=True)
        assert provenance["paper_comparable"] is True

    _, _, _, gold = get_task_datasets("Bernett", test_run=True, return_metadata=True)
    assert gold["protocol_alignment"] == "public_notebook_train_count_mismatch"
    assert gold["paper_comparable"] is False
    assert "+173" in gold["protocol_notes"]


def test_step121000_checkpoint_provenance_is_machine_readable() -> None:
    checkpoint = (
        ROOT
        / "output/grammar_v2_esmc300m_integrated_llada_7l_step121000/best.pt"
    )
    provenance = _checkpoint_provenance(f"grammar:{checkpoint}")
    assert provenance["checkpoint_step"] == 121000
    assert provenance["checkpoint_sha256"] == (
        "51f0eee5fa7b127a0487a2ea8fdbc322bda6f6908ff2662087bd87d4980d4613"
    )


def test_formal_step121000_three_ppi_jobs_are_uncapped_and_protocol_locked() -> None:
    import yaml

    jobs = {
        "HumanPPI": "eval_mint_ours_step121000_humanppi_official_full.yml",
        "YeastPPI": "eval_mint_ours_step121000_yeastppi_official_full.yml",
        "Bernett": "eval_mint_ours_step121000_goldppi_official_full.yml",
    }
    assert set(OURS_THREE_PPI_EXPECTED) == set(jobs)
    for task, filename in jobs.items():
        payload = yaml.safe_load((ROOT / "eval_jobs" / filename).read_text())
        entrypoint = payload["Entrypoint"]
        assert f"TASK={task}" in entrypoint
        assert "--bs 96 --max_length 1024" in entrypoint
        assert "--batch_size 16 --rep 3 --max_seq_length 1024" in entrypoint
        assert "--max_train" not in entrypoint
        assert "--max_val" not in entrypoint
        assert "--max_test" not in entrypoint
        assert "--sep_chains" not in entrypoint
        assert payload["Preemptible"] is False


def test_embedding_extractor_supports_nonoverlapping_split_workers() -> None:
    source = (ROOT / "downstream/mint_tasks/extract_embeddings.py").read_text()
    assert 'if split_name not in args.splits:' in source
    assert 'choices=("train", "val", "test")' in source


def test_gold_train_shard_bounds_are_contiguous_and_complete() -> None:
    bounds = [shard_bounds(163192, index, 4) for index in range(4)]
    assert bounds == [
        (0, 40798),
        (40798, 81596),
        (81596, 122394),
        (122394, 163192),
    ]
    assert all(left[1] == right[0] for left, right in zip(bounds, bounds[1:]))
    test_bounds = [shard_bounds(52048, index, 4) for index in range(4)]
    assert test_bounds == [
        (0, 13012),
        (13012, 26024),
        (26024, 39036),
        (39036, 52048),
    ]


def test_fixed_split_mlp_repetitions_can_be_dispatched_without_seed_drift() -> None:
    assert list(repetition_indices(3, 0)) == [0, 1, 2]
    assert list(repetition_indices(1, 0)) == [0]
    assert list(repetition_indices(1, 1)) == [1]
    assert list(repetition_indices(1, 2)) == [2]
    with pytest.raises(ValueError):
        repetition_indices(0, 0)


def test_mutation_windows_are_truncated_in_alignment() -> None:
    wt = [["ABCDE", "LONGPARTNER"]]
    mutant = [["ABXDE", "LONGPARTNER"]]
    wt_out, mutant_out = _aligned_truncate_mutation_pairs(wt, mutant, 5)
    assert wt_out == [["ABC", "LON"]]
    assert mutant_out == [["ABX", "LON"]]
    with pytest.raises(RuntimeError):
        _aligned_truncate_mutation_pairs([["AA"]], [["AA", "BB"]], 5)


def test_local_rerun_covers_all_eight_figure2_models() -> None:
    assert DEFAULT_PROTOCOL_TAG == "localfixed-v1-l2048"
    assert [baseline["display"] for baseline in BASELINES] == [
        "MINT",
        "ESM2-150M",
        "ESM2-650M",
        "ESM-1b",
        "ESM2-3B",
        "ProGen2-Large",
        "ProtT5-UniRef",
        "ProtT5-BFD",
    ]
    assert len({baseline["stem"] for baseline in BASELINES}) == 8

    args = SimpleNamespace(
        task="MutationalPPI",
        max_seq_length=2048,
        repetitions=3,
        protocol_tag=DEFAULT_PROTOCOL_TAG,
    )
    esm1b = next(item for item in BASELINES if item["display"] == "ESM-1b")
    progen2 = next(
        item for item in BASELINES if item["display"] == "ProGen2-Large"
    )
    esm2 = next(item for item in BASELINES if item["display"] == "ESM2-650M")
    assert effective_max_seq_length(args, esm1b) == 1024
    assert effective_max_seq_length(args, progen2) == 1024
    assert effective_max_seq_length(args, esm2) == 2048
    esm1b_extract = extraction_command(args, esm1b)
    esm1b_head = finetune_command(args, esm1b)
    progen2_extract = extraction_command(args, progen2)
    progen2_head = finetune_command(args, progen2)
    assert esm1b_extract[esm1b_extract.index("--max_seq_length") + 1] == "1024"
    assert esm1b_head[esm1b_head.index("--max_seq_length") + 1] == "1024"
    assert progen2_extract[progen2_extract.index("--max_seq_length") + 1] == "1024"
    assert progen2_head[progen2_head.index("--max_seq_length") + 1] == "1024"


def test_global_manifest_is_complete_and_counts_are_frozen() -> None:
    manifest = _manifest()
    assert manifest["status"] == "validated"
    assert set(manifest["tasks"]) == set(EXPECTED_ROWS)
    for task, expected in EXPECTED_ROWS.items():
        task_manifest = manifest["tasks"][task]
        assert task_manifest["status"] == "validated"
        assert [item["rows"] for item in task_manifest["outputs"]] == expected
        for item in task_manifest["outputs"]:
            assert Path(item["path"]).is_file()
            assert item["sha256"] == EXPECTED_OUTPUT_SHA256[task][
                Path(item["path"]).name
            ]


def test_paper_mismatches_are_explicit_not_silently_equated() -> None:
    audit = _manifest()["paper_protocol_audit"]["tasks"]
    assert audit["humanppi"]["alignment"] == "exact"
    assert audit["yeastppi"]["alignment"] == "exact"
    assert audit["gold"]["alignment"] == "mismatch"
    assert audit["mutationalppi"]["alignment"] == (
        "data_size_exact_protocol_unresolved"
    )
    assert audit["skempi"]["alignment"] == "total_exact_fold_sizes_mismatch"


def test_mutationalppi_exposes_wild_type_and_mutant_pairs() -> None:
    train, validation, test = get_task_datasets("MutationalPPI", test_run=True)
    assert validation is None
    assert test is None
    assert len(train) == 20
    seq1, seq2, seq1_mut, seq2_mut, target = train[0]
    assert seq1 != seq1_mut or seq2 != seq2_mut
    assert target in (0.0, 1.0)


def test_mutationalppi_local_pair_group_folds_are_fixed_and_leak_free() -> None:
    manifest = json.loads(MUTATIONALPPI_FOLDS_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["status"] == "validated"
    assert manifest["protocol_id"] == MUTATIONALPPI_SPLIT_PROTOCOL_ID
    assert manifest["input"]["sha256"] == EXPECTED_OUTPUT_SHA256["mutationalppi"][
        "processed_data.csv"
    ]
    assert manifest["grouping"]["groups"] == 1490
    assert len(manifest["folds"]) == 10
    assert sum(fold["rows"] for fold in manifest["folds"]) == 3406

    assignments = pd.read_csv(MUTATIONALPPI_FOLDS)
    assert assignments["notebook_row"].tolist() == list(range(3406))
    assert sorted(assignments["fold"].unique().tolist()) == list(range(10))
    assert assignments.groupby("pair_group_sha256")["fold"].nunique().max() == 1

    _, _, _, metadata = get_task_datasets(
        "MutationalPPI",
        return_metadata=True,
    )
    assert metadata["method"] == "group_cv"
    assert metadata["split_protocol_id"] == MUTATIONALPPI_SPLIT_PROTOCOL_ID
    assert len(metadata["fold_ids"]) == 3406


def test_mint_paper_registry_contains_all_eight_figure2_models() -> None:
    registry = pd.read_csv(
        ROOT / "downstream/benchmark/outputs/external/paper_reported_baselines.csv"
    )
    rows = registry[
        (registry["benchmark"] == "mint_generalppi")
        & registry["task"].isin(
            [
                "HumanPPI",
                "YeastPPI",
                "Gold-standard PPI",
                "MutationalPPI",
                "SKEMPI",
            ]
        )
    ]
    assert len(rows) == 40
    assert set(rows["method"]) == {
        "MINT",
        "ESM2-150M",
        "ESM2-650M",
        "ESM-1b",
        "ESM2-3B",
        "ProGen2-Large",
        "ProtT5-UniRef",
        "ProtT5-BFD",
    }


def test_hybrid_table_never_uses_paper_fallback_for_local_tasks() -> None:
    hybrid = select_hybrid(collect_paper(), pd.DataFrame())
    assert len(hybrid) == 24
    assert set(hybrid["task"]) == {
        "HumanPPI",
        "YeastPPI",
        "Gold-standard PPI",
    }
    assert set(hybrid["source_label"]) == {"[P]"}


def test_skempi_complexes_do_not_leak_between_train_and_test() -> None:
    frame = pd.read_csv(
        DATA_ROOT / "SKEMPI_v2" / "processed_data.csv",
        usecols=["complex", "split_0", "split_1", "split_2"],
    )
    test_sets = []
    for split in ("split_0", "split_1", "split_2"):
        assert frame.groupby("complex")[split].nunique().max() == 1
        test_sets.append(set(frame.loc[frame[split] == "test", "complex"]))
    all_complexes = set(frame["complex"])
    assert set().union(*test_sets) == all_complexes
    assert not any(
        test_sets[left] & test_sets[right]
        for left in range(3)
        for right in range(left + 1, 3)
    )


def test_skempi_rng_state_is_committed_and_was_restored() -> None:
    task_manifest = _manifest()["tasks"]["skempi"]
    state = task_manifest["protocol"]["canonical_numpy_state"]
    assert Path(state["path"]).is_file()
    assert len(state["sha256"]) == 64
    assert task_manifest["execution_metadata"]["skempi_numpy_state_restored"] is True
