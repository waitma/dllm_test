#!/usr/bin/env python
"""Is the final decoder feature contaminated by batch padding?

The headline embedder pads every micro-batch to the longest row in that batch.
If padding were visible to the bidirectional attention, a sequence's pooled
final-layer feature would depend on *what else happened to share its batch* --
which would silently corrupt every representation number, and would do so
unevenly because ``_embed_single_decoder`` length-buckets its batches.

This embeds the same probe sequences three ways and compares the resulting
final-decoder vectors:

  alone   : batch of 1, no padding at all
  homog   : batched with similar-length sequences (little padding)
  padded  : batched with much longer sequences (heavy padding)

Any cosine below ~1.0 means padding leaks into the representation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_VALID = set("ACDEFGHIKLMNPQRSTVWY")

PROBES = [
    "CASSSRSGQPQHF",
    "CASSTRAGTEQFF",
    "CASSLGQAYEQYF",
]
FILLER_SHORT = ["CASSQETQYF", "CASRPGLAGGRPEQYF", "CASSLAPGATNEKLFF"]
# ~10x longer than the probes so `padded` batches are dominated by padding
FILLER_LONG = [
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYYMHWVRQAPGQGLEWMGIINPSGGSTSYAQKFQGRVTMTRDTSTSTVYMELSSLRSEDTAVYYCARDRGGYSSSWYFDLWGRGTLVTVSS",
    "EIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGSSPRTFGQGTKVEIKRTVAAPSVFIFPPS",
    "QVQLQESGPGLVKPSETLSLTCTVSGGSISSYYWSWIRQPPGKGLEWIGYIYYSGSTNYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCARDYYGSGSYYFDYWGQGTLVTVSSAS",
]


def sanitize(seq: str) -> str:
    s = "".join(str(seq or "").split()).upper().replace("J", "L")
    return "".join(c for c in s if c in _VALID) or "A"


@torch.no_grad()
def pooled_final(model, collator, seqs, device):
    """Pooled ``hidden_states[-1]`` over residue tokens, one row per input."""
    from dllm.pipelines.qwen3_vl_arch.data import BioSeqChain, BioSeqRecord

    records = [
        BioSeqRecord(
            chains=[BioSeqChain(sanitize(s), "tcr_beta")],
            task_type="tcr",
            source="batch_invariance",
        )
        for s in seqs
    ]
    batch = collator(records)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    hidden = model.last_hidden_state(**batch).float()
    w = batch["residue_mask"].bool().unsqueeze(-1).float()
    pooled = (hidden * w).sum(1) / w.sum(1).clamp(min=1.0)
    return pooled.cpu().numpy().astype(np.float64), int(batch["input_ids"].shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    from examples.llada.load_fusion_checkpoint import load_fusion_for_eval

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = load_fusion_for_eval(ckpt, device=device, max_length=1024)
    model, collator = bundle.model, bundle.collator

    print(f"\n=== {ckpt.parent.name}/{ckpt.name} ===")

    alone = []
    for s in PROBES:
        vec, width = pooled_final(model, collator, [s], device)
        alone.append(vec[0])
        print(f"  alone  '{s}'  padded_width={width}")
    alone = np.stack(alone)

    homog, w_homog = pooled_final(model, collator, PROBES + FILLER_SHORT, device)
    padded, w_padded = pooled_final(model, collator, PROBES + FILLER_LONG, device)
    print(f"  homog batch width={w_homog}   padded batch width={w_padded}")

    def cos(a, b):
        return float(
            (a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        )

    print(f"\n  {'probe':>4} {'cos(alone,homog)':>18} {'cos(alone,padded)':>19} "
          f"{'rel L2 drift (padded)':>22}")
    worst = 1.0
    for i, s in enumerate(PROBES):
        c_h = cos(alone[i], homog[i])
        c_p = cos(alone[i], padded[i])
        drift = float(
            np.linalg.norm(padded[i] - alone[i]) / (np.linalg.norm(alone[i]) + 1e-12)
        )
        worst = min(worst, c_p, c_h)
        print(f"  {i:>4} {c_h:>18.8f} {c_p:>19.8f} {drift:>22.6f}")

    print(
        f"\n  verdict: worst cosine = {worst:.8f} -> "
        + (
            "batch-invariant, padding is correctly masked"
            if worst > 0.9999
            else "PADDING LEAKS INTO THE REPRESENTATION"
        )
    )


if __name__ == "__main__":
    main()
