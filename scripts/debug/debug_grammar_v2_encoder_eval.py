"""Debug grammar-v2 encoder downstream eval vs v1.

Run:
  source activate /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
  python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/debug/debug_grammar_v2_encoder_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from downstream.grammar.common import (
    antibody_pair_record,
    build_grammar_collator,
    load_grammar_checkpoint,
    run_grammar_generate,
)
from downstream.grammar.masks import light_chain_generation_partial_mask
from downstream.grammar.metrics import extract_chain_sequence
from dllm.pipelines.qwen3_vl_arch.sampling_bioseq import (
    build_generation_mask,
    _model_logits,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def decode_encoder_chain(tokenizer, ids: list[int]) -> str:
    id_to_token = getattr(tokenizer.base_tokenizer, "id_to_token", {})
    out = []
    for i in ids:
        tok = id_to_token.get(int(i), "?")
        if tok in ("<cls>", "<eos>", "<pad>"):
            continue
        out.append(tok if len(tok) == 1 else "")
    return "".join(out)


def run_one(tag: str, ckpt: Path, heavy: str, light: str) -> None:
    print(f"\n==================== {tag} ====================", flush=True)
    model, tok = load_grammar_checkpoint(ckpt, device=DEV)
    print(f"model class: {type(model).__name__}", flush=True)
    col = build_grammar_collator(tok)
    rec = antibody_pair_record(heavy, light)
    batch = col([rec])
    batch = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in batch.items()}

    eii = batch.get("encoder_input_ids")
    if eii is not None:
        print(f"encoder_input_ids shape: {tuple(eii.shape)}", flush=True)
        for c in range(eii.shape[1]):
            res_sum = int(batch["encoder_residue_mask"][0, c].sum().item())
            chain_on = bool(batch["encoder_chain_mask"][0, c].item())
            seq = decode_encoder_chain(tok, eii[0, c].tolist())
            print(f"  chain[{c}] on={chain_on} resmask={res_sum} seq[:60]={seq[:60]}", flush=True)
        print(f"  heavy gt[:60]={heavy[:60]}", flush=True)
        print(f"  light gt[:60]={light[:60]}", flush=True)
    else:
        print("no encoder_input_ids in batch (no-encoder model)", flush=True)

    pm = light_chain_generation_partial_mask(batch, tok, prompt_residues=0)
    gen_mask = build_generation_mask(batch, pm)
    mask_id = int(model.config.mask_token_id)
    out_tokens = batch["input_ids"].clone().masked_fill(gen_mask, mask_id)
    ts = torch.ones(1, device=out_tokens.device)
    logits = _model_logits(
        model=model,
        batch=batch,
        output_tokens=out_tokens,
        generation_mask=gen_mask,
        mask_token_id=mask_id,
        timesteps=ts,
        cfg_scale=0.0,
        partial_mask=pm,
    )
    pred = logits.argmax(-1)
    gt = batch["input_ids"]
    correct = int(((pred == gt) & gen_mask).sum().item())
    total = int(gen_mask.sum().item())
    print(f"one-step light recon acc: {correct}/{total} = {correct / max(total, 1):.3f}", flush=True)

    ot, _ = run_grammar_generate(model, batch, partial_mask=pm, max_iter=32, sampling_strategy="argmax")
    gen_light = extract_chain_sequence(
        ot[0], batch["attention_mask"][0], batch["residue_mask"][0], tok, chain="light"
    )
    print(f"gen  light: {gen_light}", flush=True)
    print(f"gt   light: {light}", flush=True)


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "downstream" / "comp_chain" / "test_data_oas_holdout.csv")
    df = df[~df["h_sequence"].isna()]
    heavy = str(df["h_sequence"].iloc[0]).replace("-", "")
    light = str(df["l_sequence"].fillna("").iloc[0]).replace("-", "")
    print(f"heavy len={len(heavy)} light len={len(light)}", flush=True)

    run_one("v1_esmc300m", ROOT / "output/grammar_v1_esmc300m/latest.pt", heavy, light)
    run_one("v2_esmc300m", ROOT / "output/grammar_v2_esmc300m/latest.pt", heavy, light)
    run_one("v2_no_encoder_qwen0_6b", ROOT / "output/grammar_v2_no_encoder_qwen0_6b/latest.pt", heavy, light)


if __name__ == "__main__":
    main()
