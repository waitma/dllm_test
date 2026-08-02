#!/usr/bin/env python
"""bf16 numerical parity: ESMC encoder with flash-attention OFF vs ON.

Gate for B3 (see SPEED_ANALYSIS.md lever 1 / ARCH_AUDIT gap 3): before enabling
``--encoder-use-flash-attn`` in the integrated training YAML, confirm the ESMC
``input_ids`` forward (the path training uses) gives numerically equivalent
features with flash on vs off under bf16 on the target GPU.

Run (env protenix_abtcr, needs `flash_attn` installed + an A100):

    LD_LIBRARY_PATH=$ENV/lib python scripts/tests/bioseq/check_esmc_flash_parity.py \
        --encoder /vepfs-mlp2/.../model_weights/esmc/ESMC-300M

Prints max/mean abs diff of last_hidden_state and exits non-zero if the max diff
exceeds --tol (default 5e-2, appropriate for bf16 attention kernels).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import load_local_esmc_encoder  # noqa: E402


def _has_flash() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="/c20250601/mj/model_weights/esmc/ESMC-300M")
    ap.add_argument("--tol", type=float, default=5e-2)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("SKIP: no CUDA (flash-attn needs a GPU)")
        return 0
    if not _has_flash():
        print("SKIP: flash_attn not installed — cannot verify parity. "
              "Install flash_attn in the training image, then enable "
              "--encoder-use-flash-attn only after this check passes.")
        return 0

    device = torch.device("cuda:0")
    torch.manual_seed(0)
    # ESMC vocab: standard AA ids 4..23 are safe residue tokens.
    ids = torch.randint(4, 24, (args.batch, args.seq_len), device=device)
    attn = torch.ones_like(ids)

    enc_off = load_local_esmc_encoder(args.encoder, use_flash_attn=False).to(device).eval()
    enc_on = load_local_esmc_encoder(args.encoder, use_flash_attn=True).to(device).eval()

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        h_off = enc_off(input_ids=ids, attention_mask=attn).last_hidden_state.float()
        h_on = enc_on(input_ids=ids, attention_mask=attn).last_hidden_state.float()

    diff = (h_off - h_on).abs()
    max_d, mean_d = float(diff.max()), float(diff.mean())
    ok = max_d <= args.tol
    print(f"last_hidden_state  max_abs_diff={max_d:.4e}  mean_abs_diff={mean_d:.4e}  tol={args.tol}")
    print("PARITY OK" if ok else "PARITY FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
