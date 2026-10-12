"""Null controls for the AB light-chain pairing task (audit 2026-08-30).

The shipped ``*_metrics.json`` reports ``ref_immunomatch_mean`` only for the
*correct* heavy/light pairing (0.699).  Without the mispaired counterpart there
is no way to tell how much of a method's ImmunoMatch score reflects genuine
pairing specificity as opposed to "this merely looks like a real light chain".
That matters here because the published no-prompt rows reach ImmunoMatch 0.66
while matching the reference V gene at essentially chance level, which suggests
the metric is largely insensitive to which heavy chain a light chain is paired
with.

This script measures the missing anchor: the same 500 held-out reference light
chains scored against a derangement of the heavy chains.  Everything about the
light chain is then perfectly native and only the pairing is wrong, so the
resulting mean is the floor that any "native-looking" generator inherits for
free.

The chain-type split (ImmunoMatch ships separate kappa and lambda heads) is
taken from the dataset's own ``l_locus`` column rather than re-derived with
abnumber: the light sequences are unchanged from the reference, so their locus
is already known, and this keeps the control runnable without the ANARCI stack.

Run: python scripts/pairing_null_controls.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import RoFormerForSequenceClassification, RoFormerTokenizer

PROJECT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
GEN_DIR = PROJECT / "output" / "downstream_generation"
IMMUNOMATCH = PROJECT / "data" / "downstream" / "immunomatch"
# Any prompt3 run carries the same 500 reference heavy/light pairs; this one is
# only used as the source of that reference table.
REF_SOURCE = ("ours_fusion_8b_diff_45000_light_pairing_holdout500_prompt3_"
              "lenprior_iter124_n8.csv")

# Second control: do the same derangement to the *generated* light chains, to
# see how much of a generator's score survives when its heavy chain is replaced.
GENERATED = {
    "ours_8b_45000": REF_SOURCE,
    "ours_270m_42000": ("ours_fusion_270m_diff_42000_light_pairing_holdout500_"
                        "prompt3_lenprior_iter124_n8.csv"),
    "ophiuchus_official_cfg1.5": ("ophiuchus_ab/light_pairing_holdout500_prompt3_"
                                  "gumbel_argmax_iter124_cfg1.5_n8.csv"),
}


def _score(heavy: list[str], light: list[str], locus: list[str],
           batch_size: int = 64) -> np.ndarray:
    """ImmunoMatch pairing score per row, routing K/L to the matching head."""
    scores = np.full(len(heavy), np.nan)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for head, tag in (("kappa", "K"), ("lambda", "L")):
        idx = [i for i, t in enumerate(locus) if t == tag]
        if not idx:
            continue
        ckpt = str(IMMUNOMATCH / f"immunomatch-{head}")
        tok = RoFormerTokenizer.from_pretrained(ckpt, local_files_only=True)
        model = RoFormerForSequenceClassification.from_pretrained(
            ckpt, local_files_only=True).to(device).eval()
        print(f"  {head}: {len(idx)} rows")
        with torch.no_grad():
            for start in range(0, len(idx), batch_size):
                chunk = idx[start:start + batch_size]
                # ImmunoMatch treats each residue as its own token, so the
                # sequences must be space-separated before tokenisation.
                enc = tok([" ".join(heavy[i]) for i in chunk],
                          [" ".join(light[i]) for i in chunk],
                          return_tensors="pt", padding="max_length",
                          truncation=True, max_length=256).to(device)
                probs = torch.softmax(model(**enc).logits, dim=1)[:, 1]
                scores[chunk] = probs.float().cpu().numpy()
        del model
        torch.cuda.empty_cache()
    assert not np.isnan(scores).any(), "some rows had an unrecognised locus"
    return scores


def _derange(n: int, seed: int = 0) -> np.ndarray:
    """A permutation with no fixed point, so every pair really is mismatched."""
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(n)
        if not (perm == np.arange(n)).any():
            return perm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                        / "outputs" / "ab_pairing_null_controls.json"))
    ap.add_argument("--seeds", type=int, default=3,
                    help="number of independent derangements to average over")
    args = ap.parse_args()

    df = pd.read_csv(GEN_DIR / REF_SOURCE,
                     usecols=["h_sequence", "raw_l_sequence", "variant_idx", "l_locus"])
    uniq = (df[df.variant_idx == 0].drop_duplicates("h_sequence")
            .reset_index(drop=True))
    heavy = uniq["h_sequence"].tolist()
    light = uniq["raw_l_sequence"].tolist()
    locus = uniq["l_locus"].tolist()
    print(f"{len(uniq)} distinct held-out antibodies "
          f"(K={locus.count('K')}, L={locus.count('L')})")

    print("[correct pairing]")
    correct = _score(heavy, light, locus)

    mispaired_means = []
    for seed in range(args.seeds):
        print(f"[deranged pairing, seed {seed}]")
        perm = _derange(len(uniq), seed=seed)
        s = _score([heavy[i] for i in perm], light, locus)
        mispaired_means.append(float(s.mean()))

    out = {
        "purpose": ("ImmunoMatch floor for a perfectly native but mispaired light "
                    "chain, to separate pairing specificity from nativeness"),
        "dataset": "OAS holdout500, one row per distinct heavy chain",
        "n": len(uniq),
        "chain_type_source": "dataset l_locus column (light sequences are the references)",
        "reference_light": {
            "correct_pairing_mean": float(correct.mean()),
            "deranged_pairing_mean": float(np.mean(mispaired_means)),
            "deranged_pairing_per_seed": mispaired_means,
            "deranged_pairing_std_across_seeds": float(np.std(mispaired_means)),
            "specificity_headroom": float(correct.mean() - np.mean(mispaired_means)),
        },
        "generated_light": {},
    }

    # -- Control 2: same derangement applied to each generator's own output.
    # Both arms route K/L by the *reference* locus, so the absolute level carries
    # a small routing error (the generated light is a different locus from the
    # reference for a few percent of rows) but the correct-minus-deranged drop,
    # which is what this control is for, is unaffected.
    for tag, rel in GENERATED.items():
        gen = pd.read_csv(GEN_DIR / rel,
                          usecols=["h_sequence", "gen_l_sequence", "l_locus"])
        g_heavy = gen["h_sequence"].tolist()
        g_light = gen["gen_l_sequence"].tolist()
        g_locus = gen["l_locus"].tolist()
        print(f"[{tag}] correct heavy, {len(gen)} pairs")
        s_ok = _score(g_heavy, g_light, g_locus)
        # Derange over distinct heavy chains, then map back, so that all 8
        # samples of one heavy move together and none keeps its own heavy.
        order = list(dict.fromkeys(g_heavy))
        perm = _derange(len(order))
        swap = {h: order[perm[i]] for i, h in enumerate(order)}
        print(f"[{tag}] deranged heavy")
        s_bad = _score([swap[h] for h in g_heavy], g_light, g_locus)
        out["generated_light"][tag] = {
            "n": len(gen),
            "correct_heavy_mean": float(s_ok.mean()),
            "deranged_heavy_mean": float(s_bad.mean()),
            "drop": float(s_ok.mean() - s_bad.mean()),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    ref = out["reference_light"]
    print(f"\nreference light, correct pairing   {ref['correct_pairing_mean']:.4f}")
    print(f"reference light, deranged pairing  {ref['deranged_pairing_mean']:.4f} "
          f"(± {ref['deranged_pairing_std_across_seeds']:.4f} across {args.seeds} seeds)")
    print(f"specificity headroom               {ref['specificity_headroom']:.4f}\n")
    for tag, v in out["generated_light"].items():
        print(f"{tag:28s} correct={v['correct_heavy_mean']:.4f} "
              f"deranged={v['deranged_heavy_mean']:.4f} drop={v['drop']:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
