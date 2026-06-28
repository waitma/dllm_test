"""Verify the ESM2 step-1 NaN root cause: an <unk>(3) residue label that is also a
forbidden target produces +inf cross-entropy.

Run:
    source activate /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
    cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
    python scripts/debug/verify_unk_forbidden_inf.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dllm.pipelines.qwen3_vl_arch.data import (  # noqa: E402
    BioSeqChain,
    BioSeqRecord,
    Esm2SequenceTokenizer,
    GrammarBioSeqCollator,
    GrammarTokenizer,
)
from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import (  # noqa: E402
    BioSeqDiffusionTransformerConfig,
    compute_masked_cross_entropy,
    forbidden_diffusion_target_token_ids,
)


def part1_numeric():
    print("=== Part 1: does an <unk> label at a corrupted position give inf CE? ===")
    config = BioSeqDiffusionTransformerConfig(
        vocab_size=49, hidden_size=16, num_hidden_layers=1, num_attention_heads=4,
        intermediate_size=32, max_position_embeddings=64, pad_token_id=1, mask_token_id=32,
    )
    forbidden = forbidden_diffusion_target_token_ids(config)
    print("forbidden target ids:", forbidden, "(3=<unk>)")
    # logits [B=1, S=2, V], labels: position 0 label=5 (Ala, valid), position 1 label=3 (unk)
    logits = torch.randn(1, 2, config.vocab_size)
    labels = torch.tensor([[5, 3]])
    loss = compute_masked_cross_entropy(logits, labels, forbidden_token_ids=forbidden)
    print(f"loss with an <unk>=3 label among targets: {loss.item()} finite={torch.isfinite(loss).item()}")
    labels_ok = torch.tensor([[5, 6]])
    loss_ok = compute_masked_cross_entropy(logits, labels_ok, forbidden_token_ids=forbidden)
    print(f"loss with only valid residue labels:      {loss_ok.item()} finite={torch.isfinite(loss_ok).item()}")


def part1b_bf16():
    print("\n=== Part 1b: same CE under bf16 autocast (training dtype) ===")
    config = BioSeqDiffusionTransformerConfig(
        vocab_size=49, hidden_size=16, num_hidden_layers=1, num_attention_heads=4,
        intermediate_size=32, max_position_embeddings=64, pad_token_id=1, mask_token_id=32,
    )
    forbidden = forbidden_diffusion_target_token_ids(config)
    if torch.cuda.is_available():
        logits = torch.randn(1, 2, config.vocab_size, device="cuda")
        labels = torch.tensor([[5, 3]], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = compute_masked_cross_entropy(logits, labels, forbidden_token_ids=forbidden)
        print(f"  bf16 loss with <unk>=3 label: {loss.item()} finite={torch.isfinite(loss).item()}")
    else:
        print("  (no GPU; skipping bf16 test)")


def part2_render():
    print("\n=== Part 2: does a non-standard residue char render to <unk>=3? ===")
    tok = Esm2SequenceTokenizer()
    print("unk_token_id =", tok.unk_token_id)
    for ch in ["A", "J", "*", "U", "X", "Z", "O", "B"]:
        ids = tok.encode_residues(ch)
        print(f"  residue {ch!r} -> id {ids[0]}  ({'UNK' if ids[0]==tok.unk_token_id else 'ok'})")

    # Render a record containing a 'J' residue and corrupt that position
    gtok = GrammarTokenizer(Esm2SequenceTokenizer())
    rec = BioSeqRecord(
        chains=[BioSeqChain("AAJAA", "antibody_heavy"), BioSeqChain("LLLLL", "antibody_light")],
        task_type="antibody", source="unit",
    )
    batch = GrammarBioSeqCollator(gtok)([rec])
    ids = batch["input_ids"][0].tolist()
    has_unk = 3 in ids
    print(f"  rendered decoder input_ids contains <unk>=3 residue: {has_unk}")


def part3_scan_real():
    print("\n=== Part 3: scan real training data for <unk>=3 residue tokens ===")
    args = SimpleNamespace(
        sources="oas,ots,tcr,ppi", batch_size=8,
        grammar_data_dir=PROJECT_ROOT / "data/bioseq_grammar_v1",
        oas_weight=3.9, ots_weight=3.6, ppi_weight=1.4, tcr_weight=1.0,
        split="train", source_seed=0, max_sequence_length=2112,
        num_workers=0, tokenizer_path=None,
    )
    from dllm.pipelines.qwen3_vl_arch.data import GrammarDataModule
    dm = GrammarDataModule.from_args(args)
    tok = dm.build_tokenizer()
    forbidden = {0, 1, 2, 3, 32}
    for seed in (0, 7, 42, 123, 2026):
        loader = dm.loader(tok, split="train", source_seed=seed, epoch_size=8000)
        it = iter(loader)
        per_tok = {f: 0 for f in forbidden}
        elig_forbidden_batches = 0
        seen = 0
        for i in range(400):
            try:
                batch = next(it)
            except StopIteration:
                break
            seen += 1
            ids = batch["input_ids"]
            # positions eligible to be a diffusion target/label
            elig = batch.get("diffusion_eligible_mask")
            if elig is None:
                elig = batch["residue_mask"]
            elig = elig.bool()
            hit = False
            for f in forbidden:
                c = ((ids == f) & elig).sum().item()
                if c:
                    per_tok[f] += c
                    hit = True
            if hit:
                elig_forbidden_batches += 1
        print(f"  seed={seed}: {seen} batches, eligible-position tokens by forbidden id: "
              f"{ {k:v for k,v in per_tok.items() if v} }  "
              f"(batches with any: {elig_forbidden_batches})")


if __name__ == "__main__":
    part1_numeric()
    part1b_bf16()
    part2_render()
    part3_scan_real()
