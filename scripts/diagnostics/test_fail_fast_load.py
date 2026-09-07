"""Verify the fusion loader's fail-fast guard: real ckpts pass, a hole raises.

Positive test  every headline checkpoint must still load (a false alarm here
               would break all downstream eval).
Negative test  drop one decoder weight from the state dict and confirm the
               loader refuses instead of scoring uninitialised memory.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from examples.llada import load_fusion_checkpoint as LFC  # noqa: E402

CKPTS = {
    "270m-bert": "output/protein_esmc_llada270m_bert_immune/checkpoint-49000",
    "270m-diff": "output/protein_esmc_llada270m_diffusion_immune/checkpoint-42000",
    "8b-bert": "output/protein_esmc_llada8b_bert_immune/checkpoint-43000",
    "8b-diff": "output/protein_esmc_llada8b_diffusion_immune/checkpoint-45000",
}


def positive(name: str, rel: str, device: str) -> bool:
    path = ROOT / rel
    if not path.is_dir():
        print(f"[skip] {name}: {path} missing")
        return True
    try:
        bundle = LFC.load_fusion_for_eval(path, device=device)
    except Exception:
        print(f"[FAIL] {name}: loader raised on a real checkpoint")
        traceback.print_exc()
        return False

    nan = [
        n for n, p in bundle.model.named_parameters()
        if p.is_floating_point() and bool(torch.isnan(p).any())
    ]
    n_par = sum(1 for _ in bundle.model.parameters())
    print(
        f"[ok]   {name}: loaded d_model={bundle.d_model} n_layers={bundle.n_layers} "
        f"params={n_par} residual_nan={len(nan)}"
    )
    del bundle
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return not nan


def negative(rel: str, device: str) -> bool:
    """Punch a hole in the state dict; the guard must raise, not warn."""

    path = ROOT / rel
    if not path.is_dir():
        print(f"[skip] negative: {path} missing")
        return True

    # load_file is imported inside load_fusion_for_eval, so patch the source
    # module rather than an attribute on LFC.
    import safetensors.torch as st

    real_load = st.load_file
    victim = "decoder.model.transformer.blocks.3.attn_out.weight"

    def holed(*args, **kwargs):
        state = real_load(*args, **kwargs)
        # The loader also pulls the ESMC encoder through load_file; only punch a
        # hole in the fusion checkpoint itself.
        candidates = [k for k in state if k.endswith("attn_out.weight")]
        if candidates:
            key = victim if victim in state else candidates[0]
            del state[key]
            print(f"       (dropped {key})")
        return state

    st.load_file = holed
    try:
        LFC.load_fusion_for_eval(path, device=device)
    except RuntimeError as exc:
        msg = str(exc)
        good = "uninitialised memory" in msg and "never" in msg
        print(f"[{'ok' if good else 'FAIL'}]   negative: raised -> {msg[:150]}")
        return good
    except Exception:
        print("[FAIL] negative: raised the wrong exception type")
        traceback.print_exc()
        return False
    else:
        print("[FAIL] negative: loader accepted a partially-loaded model")
        return False
    finally:
        st.load_file = real_load


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    names = args.only or list(CKPTS)
    ok = True
    for name in names:
        ok &= positive(name, CKPTS[name], args.device)
    ok &= negative(CKPTS["270m-bert"], args.device)

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
