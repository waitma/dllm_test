"""GPU gate for native AB jobs; run through the generated Volc preflight YAML.

Run: python scripts/downstream/preflight_ab_native.py --checkpoint /abs/ckpt --out-dir /abs/preflight
This checks execution/invariants, not benchmark quality; pairing uses only 4 steps.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

from downstream.grammar.ab_features import ROOT, embed_native_pairs, file_sha256, write_json
from downstream.grammar.ab_probes import prepare_task
from downstream.grammar.cdr_infill import SAbDabDataset, _build_cdr_partial_mask, _chain_role_for_mode, build_antibody_record
from downstream.grammar.common import run_grammar_generate
from downstream.grammar.light_chain_pairing import HeavyLightCsvDataset, generate_for_batch
from downstream.grammar.metrics import masked_token_accuracy
from examples.llada.load_fusion_checkpoint import load_fusion_for_eval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for this gate")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir / "passed.json").exists():
        raise FileExistsError("Use a fresh preflight directory")
    logging.basicConfig(level=logging.INFO)
    bundle = load_fusion_for_eval(args.checkpoint, device="cuda", max_length=1024)
    report = {"checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_sha256": file_sha256(args.checkpoint / "model.safetensors"),
              "gpu": torch.cuda.get_device_name(0), "d_model": bundle.d_model,
              "probes": {}, "pairing": {}, "cdr": {}}
    for task in ("specificity", "gdp_a1", "m396"):
        _, pairs, _, _ = prepare_task(task)
        # Test the actual production microbatch and include the longest record.
        sample = [max(pairs, key=lambda p: len(p[0]) + len(p[1]))] + pairs[:15]
        features = embed_native_pairs(bundle.model, bundle.collator, sample,
                                      device="cuda", batch_size=16)
        assert features.shape == (16, bundle.d_model)
        report["probes"][task] = {"shape": list(features.shape), "finite": True,
                                  "max_residues": max(sum(map(len, p)) for p in sample)}
        print(f"PASS probe {task}", flush=True)
    dataset = HeavyLightCsvDataset(str(ROOT / "data/downstream/comp_chain/test_data_oas_holdout.csv"))
    samples = [dataset[0], dataset[1]]
    for prompt in (0, 3):
        for cfg in (0.0, 1.0, 1.5):
            torch.manual_seed(42)
            rows = generate_for_batch(samples, model=bundle.model, collator=bundle.collator,
                tokenizer=bundle.grammar_tokenizer, device=torch.device("cuda"), num_seqs=8,
                max_iter=4, sampling_strategy="gumbel_argmax", temperature=1.0,
                light_prompt_tokens=prompt, light_length_mode="reference", length_prior=None,
                cfg_scale=cfg)
            assert len(rows) == 16
            for row in rows:
                assert row["generated_light_length"] == row["ref_light_length"]
                assert row["gen_l_sequence"][:prompt] == row["raw_l_sequence"][:prompt]
            key = f"p{prompt}_cfg{cfg:g}"
            pd.DataFrame(rows).to_csv(args.out_dir / f"{key}.csv", index=False)
            report["pairing"][key] = {"rows": len(rows), "max_iter": 4, "length_and_prefix_valid": True}
            print(f"PASS pairing {key}", flush=True)
    for name, modes in (("sabdab_kong", ("cdrh1", "cdrh2", "cdrh3")),
                        ("sab23h2_converted", ("cdrh1", "cdrh2", "cdrh3", "cdrl1", "cdrl2", "cdrl3"))):
        for mode in modes:
            dataset = SAbDabDataset(str(ROOT / "data/downstream/cdr_infilling" / name), mode, fold=0)
            heavy, light, target, _, _ = dataset[0]
            record = build_antibody_record(heavy, light)
            batch = bundle.collator([record])
            batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
            role, cdr = _chain_role_for_mode(mode)
            partial = _build_cdr_partial_mask(batch, bundle.grammar_tokenizer, record, role, cdr, target)
            mask = ~partial & batch["attention_mask"].bool() & batch["residue_mask"].bool()
            assert int(mask.sum()) == len(target)
            tokens, _ = run_grammar_generate(bundle.model, batch, partial_mask=partial,
                                             max_iter=1, sampling_strategy="argmax")
            aar = masked_token_accuracy(tokens, batch["labels"], mask)
            assert torch.isfinite(aar)
            report["cdr"][f"{name}/{mode}"] = {"target_residues": len(target), "finite": True}
            print(f"PASS CDR {name}/{mode}", flush=True)
    report.update(status="passed", quality_evaluation=False,
                  max_gpu_memory_bytes=torch.cuda.max_memory_allocated())
    write_json(args.out_dir / "passed.json", report)
    print("AB_NATIVE_GPU_PREFLIGHT_PASSED", flush=True)


if __name__ == "__main__":
    main()
