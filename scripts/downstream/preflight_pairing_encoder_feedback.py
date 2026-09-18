"""Check actual fusion forward inputs before a full pairing rerun.

Run in protenix_abtcr on GPU:
  python scripts/downstream/preflight_pairing_encoder_feedback.py \
    --checkpoint /abs/snapshot --csv /abs/holdout.csv --prompt 0 --output /abs/gate.json
This verifies feedback and hidden-reference invariance, not generation quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from downstream.grammar.common import build_eval_collator, load_grammar_checkpoint
from downstream.grammar.light_chain_pairing import (
    GENERATION_PROTOCOL_VERSION,
    HeavyLightCsvDataset,
    _file_sha256,
    _protocol_source_hashes,
    generate_for_batch,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--prompt", type=int, choices=(0, 3), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assert torch.cuda.is_available(), "A real GPU model is required for this gate"
    device = torch.device("cuda")
    model, tokenizer = load_grammar_checkpoint(str(args.checkpoint), device=device)
    collator = build_eval_collator(model, tokenizer)
    dataset = HeavyLightCsvDataset(str(args.csv), end_index=2)
    samples = [dataset[i] for i in range(len(dataset))]
    assert len(samples) == 2
    original_denoise = model._denoise
    captures = []

    def checked_denoise(**kwargs):
        tokens = kwargs["input_ids"]
        encoder = kwargs["encoder_input_ids"]
        decoder_mask = int(model.config.mask_token_id)
        encoder_mask = int(model.config.encoder_mask_token_id)
        counts = []
        # Independent gather checks the tensors entering the real model, not
        # the helper that constructs them. ESMC chain and decoder IDs differ.
        for row in range(tokens.size(0)):
            for chain in range(2):
                positions = (kwargs["residue_mask"][row].bool() & kwargs["chain_ids"][row].eq(chain)).nonzero().flatten()
                inner = kwargs["position_ids_inner"][row, positions]
                dec = tokens[row, positions]
                residue = torch.isin(dec, model._residue_token_ids)
                mapped = model.llada_to_grammar_ids[dec]
                expected = torch.where(residue & dec.ne(decoder_mask), mapped, encoder_mask)
                slots = kwargs["encoder_residue_mask"][row, chain].nonzero().flatten()
                assert torch.equal(encoder[row, chain, slots[inner]], expected), (row, chain)
                if chain == 1:
                    counts.append(int(residue[args.prompt:].sum()))
        captures.append({
            "decoder": tokens.detach().cpu().clone(),
            "encoder": encoder.detach().cpu().clone(),
            "generated_visible_per_candidate": counts,
        })
        return original_denoise(**kwargs)

    model._denoise = checked_denoise
    all_captures = []
    generations = []
    for replace_suffix in (False, True):
        captures.clear()
        selected = [(h, l[:args.prompt] + "A" * (len(l) - args.prompt), meta)
                    if replace_suffix else (h, l, meta) for h, l, meta in samples]
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        rows = generate_for_batch(
            selected, model=model, collator=collator, tokenizer=tokenizer,
            device=device, num_seqs=2, max_iter=4, sampling_strategy="gumbel_argmax",
            temperature=1.0, light_prompt_tokens=args.prompt,
            light_length_mode="reference", length_prior=None, cfg_scale=0.0,
        )
        assert len(captures) == 4
        assert all(n == 0 for n in captures[0]["generated_visible_per_candidate"])
        assert all(n > 0 for n in captures[-1]["generated_visible_per_candidate"])
        all_captures.append(list(captures))
        generations.append([r["gen_l_sequence"] for r in rows])
    for original, perturbed in zip(*all_captures):
        assert torch.equal(original["decoder"], perturbed["decoder"])
        assert torch.equal(original["encoder"], perturbed["encoder"])
    assert generations[0] == generations[1]
    payload = {
        "passed": True, "protocol_version": GENERATION_PROTOCOL_VERSION,
        "light_prompt_tokens": args.prompt, "max_iter": 4, "num_heavies": 2, "num_seqs": 2,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _file_sha256(args.checkpoint / "model.safetensors"),
        "input_csv_sha256": _file_sha256(args.csv),
        "source_sha256": _protocol_source_hashes(),
        "gate_source_sha256": _file_sha256(Path(__file__)),
        "generated_visible_per_step": [c["generated_visible_per_candidate"] for c in all_captures[0]],
        "hidden_reference_invariance": True, "encoder_decoder_state_match": True,
        "gpu": torch.cuda.get_device_name(0),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
