"""Tests for STRING actions -> grammar PPI mode rendering.

Run with::

    pytest scripts/tests/bioseq/test_string_actions_grammar.py -q
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
RAW = PROJECT_ROOT / "data/ppi_task_raw/processed/string_actions_smoke"
HUMAN_ACTIONS = PROJECT_ROOT / "data/ppi_task_raw/raw/stringdb_mint/9606.protein.actions.v11.0.txt.gz"
SEQUENCES_FA = PROJECT_ROOT / "data/ppi_task_raw/raw/stringdb_mint/protein.sequences.v12.0.fa"
CLUSTER_TSV = PROJECT_ROOT / "data/ppi_task_raw/raw/stringdb_mint/clu50.tsv"

from dllm.pipelines.qwen3_vl_arch.data import (  # noqa: E402
    BioSeqChain,
    BioSeqRecord,
    Esm2SequenceTokenizer,
    GrammarRenderer,
    GrammarTokenizer,
)
from dllm.pipelines.qwen3_vl_arch.data.grammar_builders import ppi_record  # noqa: E402


def rendered_tokens(record: BioSeqRecord) -> list[str]:
    tokenizer = GrammarTokenizer(Esm2SequenceTokenizer())
    row = GrammarRenderer(tokenizer).encode(record)
    return tokenizer.decode_tokens(row["input_ids"])


@pytest.fixture(scope="module")
def human_actions_smoke_dir() -> Path:
    if not HUMAN_ACTIONS.exists() or not SEQUENCES_FA.exists() or not CLUSTER_TSV.exists():
        pytest.skip("Missing human actions / sequences / clu50 for smoke test")
    if not (RAW / "manifest.json").exists():
        RAW.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/data/build_string_actions_splits.py"),
            "--actions-gz",
            str(HUMAN_ACTIONS),
            "--sequences-fa",
            str(SEQUENCES_FA),
            "--cluster-tsv",
            str(CLUSTER_TSV),
            "--output-dir",
            str(RAW),
            "--num-valid",
            "5000",
            "--physical-valid-links",
            "/dev/null",
        ]
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    return RAW


def test_catalysis_renders_target_fixed_actor_generated() -> None:
    record = ppi_record(
        "TARGETSEQ",
        "ACTORSEQ",
        split="train",
        relation="catalysis",
        source="mint_string_actions",
    )
    assert record is not None
    tokens = rendered_tokens(record)
    assert tokens[:4] == ["<prots>", "T", "A", "R"]
    assert "<catalysis>" in tokens
    assert tokens.index("<catalysis>") < tokens.index("A", tokens.index("<catalysis>"))


def test_human_actions_smoke_manifest(human_actions_smoke_dir: Path) -> None:
    manifest = json.loads((human_actions_smoke_dir / "manifest.json").read_text())
    assert manifest["policy_id"] == "mint_string_actions_v11"
    assert manifest["outputs"]["train_filtered"]["n_links"] > 0
    assert manifest["outputs"]["valid"]["n_links"] == 5000


def test_human_actions_links_have_mode_column(human_actions_smoke_dir: Path) -> None:
    links = human_actions_smoke_dir / "training_filtered.links.txt.gz"
    modes: set[str] = set()
    with gzip.open(links, "rt") as handle:
        for idx, line in enumerate(handle):
            parts = line.strip().split()
            assert len(parts) == 3, line
            modes.add(parts[2])
            if idx >= 1000:
                break
    assert "binding" in modes
    assert modes & {"catalysis", "activation", "inhibition", "reaction"}


def test_ppi_record_skips_chains_longer_than_1024() -> None:
    short = "A" * 100
    long = "C" * 1025
    assert ppi_record(short, short, split="train", relation="binding") is not None
    assert ppi_record(long, short, split="train", relation="catalysis") is None
    assert ppi_record(short, long, split="train", relation="reaction") is None
