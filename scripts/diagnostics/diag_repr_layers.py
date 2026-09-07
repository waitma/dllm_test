#!/usr/bin/env python
"""Why is the BERT arm mediocre on representation tasks?

Runs the *headline* fusion embedding path (single-chain `tcr_beta` grammar
record -> clean ESMC-conditioned forward -> mean-pool over residues) on the T2
clustering set, but keeps **every** decoder layer instead of only the last one,
and additionally reports embedding geometry.

For each layer x post-processing combination we report the T2 basis-B metrics
(K-means ARI/NMI/Purity over a K sweep) plus a 25-way logistic probe, so a
"the readout is wrong" hypothesis can be separated from a "pretraining is
wrong" hypothesis.

Usage:
    python scripts/diagnostics/diag_repr_layers.py \
        --checkpoint output/protein_esmc_llada270m_bert_immune/checkpoint-49000 \
        --tag bert_270m_49000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BENCH = ROOT / "downstream" / "benchmark"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

T2_CSV = BENCH / "data" / "tcr_clustering_embed" / "tcrs.csv"
OUT_DIR = ROOT / "output" / "repr_diagnostics"

K_SWEEP = [10, 25, 40, 55, 70, 85, 100]
_VALID = set("ACDEFGHIKLMNPQRSTVWY")


# --------------------------------------------------------------------------
# forward pass keeping all layers
# --------------------------------------------------------------------------
def sanitize(seq: str) -> str:
    s = "".join(str(seq or "").split()).upper().replace("J", "L")
    s = "".join(c for c in s if c in _VALID)
    return s or "A"


@torch.no_grad()
def all_layer_hidden(
    model, batch: dict, disable_esmc: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror ``LLaDAEsmcFusion._denoise`` but return the full hidden tuple.

    Returns ``(hidden [L+1, B, S, D], token_condition [B, S, E])`` where the
    ESMC condition is the pre-projection encoder feature (encoder-only
    baseline). With ``disable_esmc`` the condition is still computed (so the
    encoder-only column stays available) but is **not** added to the decoder
    input, which measures how much the decoder relies on the ESMC path.
    """
    word_embeddings = model.decoder.get_input_embeddings()
    inputs_embeds = word_embeddings(batch["input_ids"])

    chain_token_condition = model.encode_chain_tokens(
        encoder_input_ids=batch["encoder_input_ids"],
        encoder_attention_mask=batch["encoder_attention_mask"],
        encoder_residue_mask=batch["encoder_residue_mask"],
        encoder_chain_mask=batch["encoder_chain_mask"],
        encoder_kwargs=None,
    )
    token_condition = model.gather_token_condition(
        chain_token_condition,
        chain_ids=batch["chain_ids"],
        position_ids_inner=batch["position_ids_inner"],
        attention_mask=batch["attention_mask"],
        encoder_residue_mask=batch["encoder_residue_mask"],
    )
    condition_mask = model.build_encoder_condition_mask(
        chain_ids=batch["chain_ids"],
        position_ids_inner=batch["position_ids_inner"],
        attention_mask=batch["attention_mask"],
        encoder_position_ids=None,
        residue_mask=batch["residue_mask"],
    )
    proj_dtype = model.condition_proj.weight.dtype
    cond = token_condition.to(dtype=proj_dtype)
    if model.condition_norm is not None:
        cond = model.condition_norm(cond)
    cond_h = model.condition_proj(cond).to(dtype=inputs_embeds.dtype)
    m = condition_mask.to(dtype=inputs_embeds.dtype).unsqueeze(-1)
    if disable_esmc:
        pass
    elif model.residue_cond_mode == "add":
        inputs_embeds = inputs_embeds + cond_h * m
    elif model.residue_cond_mode == "feature":
        inputs_embeds = inputs_embeds * (1.0 - m) + cond_h * m

    out = model.decoder(
        inputs_embeds=inputs_embeds,
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
        use_cache=False,
    )
    return out.hidden_states, token_condition


def embed_all_layers(model, collator, seqs, device, batch_size=64, disable_esmc=False):
    """-> (per_layer [L+1, N, D] float32, esmc [N, E] float32)."""
    from dllm.pipelines.qwen3_vl_arch.data import BioSeqChain, BioSeqRecord

    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    per_layer = None
    esmc = None

    t0 = time.time()
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        records = [
            BioSeqRecord(
                chains=[BioSeqChain(sanitize(seqs[j]), "tcr_beta")],
                task_type="tcr",
                source="repr_diag",
            )
            for j in idx
        ]
        batch = collator(records)
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        hidden, cond = all_layer_hidden(model, batch, disable_esmc=disable_esmc)
        # [B,S,1] float weights; every record has >=1 residue by construction
        w = batch["residue_mask"].bool().unsqueeze(-1).float()
        denom = w.sum(1).clamp(min=1.0)

        # one device->host copy per (layer, batch) instead of per (layer, sample)
        pooled = torch.stack(
            [(h.float() * w).sum(1) / denom for h in hidden]
        )  # [L+1, B, D]
        pooled_np = pooled.cpu().numpy().astype(np.float32)
        cond_np = ((cond.float() * w).sum(1) / denom).cpu().numpy().astype(np.float32)

        if per_layer is None:
            per_layer = np.zeros(
                (pooled_np.shape[0], len(seqs), pooled_np.shape[2]), dtype=np.float32
            )
            esmc = np.zeros((len(seqs), cond_np.shape[1]), dtype=np.float32)
        per_layer[:, idx, :] = pooled_np
        esmc[idx] = cond_np

        if (i // batch_size) % 20 == 0:
            done = min(i + batch_size, len(order))
            print(f"    {done}/{len(order)}  ({time.time() - t0:.0f}s)", flush=True)

    print(f"    embedding done in {time.time() - t0:.0f}s", flush=True)
    return per_layer, esmc


# --------------------------------------------------------------------------
# geometry + downstream proxies
# --------------------------------------------------------------------------
def geometry(x: np.ndarray, rng: np.random.Generator, n_sample: int = 2000) -> dict:
    """Anisotropy / effective-dimensionality descriptors of an embedding set."""
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    idx = rng.choice(len(xn), size=min(n_sample, len(xn)), replace=False)
    sub = xn[idx]
    sim = sub @ sub.T
    iu = np.triu_indices(len(sub), k=1)
    cos = sim[iu]

    xc = x - x.mean(0, keepdims=True)
    sv = np.linalg.svd(xc, compute_uv=False)
    ev = sv**2
    p = ev / (ev.sum() + 1e-12)
    eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))

    return {
        "mean_cos": float(cos.mean()),
        "p95_cos": float(np.percentile(cos, 95)),
        "std_cos": float(cos.std()),
        "eff_rank": round(eff_rank, 2),
        "top1_var_frac": float(p[0]),
        "top10_var_frac": float(p[:10].sum()),
        "norm_mean": float(np.linalg.norm(x, axis=1).mean()),
        "norm_cv": float(
            np.linalg.norm(x, axis=1).std() / (np.linalg.norm(x, axis=1).mean() + 1e-8)
        ),
    }


def postprocess(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        y = x
    elif mode == "center":
        y = x - x.mean(0, keepdims=True)
    elif mode == "zscore":
        mu, sd = x.mean(0, keepdims=True), x.std(0, keepdims=True)
        y = (x - mu) / (sd + 1e-6)
    else:
        raise ValueError(mode)
    return y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-8)


def cluster_scores(x: np.ndarray, true: np.ndarray, ks, seed=0) -> dict:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    aris, nmis, purs = [], [], []
    for k in ks:
        lab = MiniBatchKMeans(
            n_clusters=k, random_state=seed, n_init=3, batch_size=2048
        ).fit_predict(x)
        aris.append(adjusted_rand_score(true, lab))
        nmis.append(normalized_mutual_info_score(true, lab))
        df = pd.DataFrame({"c": lab, "t": true})
        purs.append(
            df.groupby("c")["t"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(df)
        )
    return {
        "ari_mean": round(float(np.mean(aris)), 4),
        "ari_best": round(float(np.max(aris)), 4),
        "nmi_mean": round(float(np.mean(nmis)), 4),
        "purity_mean": round(float(np.mean(purs)), 4),
    }


def probe_scores(x: np.ndarray, y: np.ndarray, seed=0) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    yy = le.fit_transform(y)
    xtr, xte, ytr, yte = train_test_split(
        x, yy, test_size=0.3, random_state=seed, stratify=yy
    )
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
    clf.fit(xtr, ytr)
    prob = clf.predict_proba(xte)
    return {
        "probe_auroc": round(
            float(roc_auc_score(yte, prob, multi_class="ovr", average="macro")), 4
        ),
        "probe_acc": round(float((clf.predict(xte) == yte).mean()), 4),
    }


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--modes", default="raw,center,zscore")
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--save-embeddings", action="store_true")
    ap.add_argument(
        "--disable-esmc",
        action="store_true",
        help="drop the ESMC condition from the decoder input (ablation)",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    from examples.llada.load_fusion_checkpoint import load_fusion_for_eval

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== loading {ckpt} ===", flush=True)
    bundle = load_fusion_for_eval(ckpt, device=device, max_length=1024)
    model, collator = bundle.model, bundle.collator
    print(
        f"    d_model={bundle.d_model} n_layers={bundle.n_layers} "
        f"enc_h={bundle.encoder_hidden}",
        flush=True,
    )

    df = pd.read_csv(T2_CSV).fillna("")
    seqs = df["cdr3b"].astype(str).tolist()
    true = df["epitope"].astype(str).to_numpy()
    uniq = list(dict.fromkeys(seqs))
    print(f"=== embedding {len(uniq)} unique CDR3b over {df['epitope'].nunique()} epitopes ===",
          flush=True)

    if args.disable_esmc:
        print("    [ablation] ESMC condition NOT added to decoder input", flush=True)
    per_layer, esmc = embed_all_layers(
        model,
        collator,
        uniq,
        device,
        batch_size=args.batch_size,
        disable_esmc=args.disable_esmc,
    )
    table = {s: i for i, s in enumerate(uniq)}
    row_idx = np.array([table[s] for s in seqs])
    per_layer = per_layer[:, row_idx, :]
    esmc = esmc[row_idx]
    print(f"    hidden tuple length = {per_layer.shape[0]} (n_layers+1 expected)")

    out_dir = OUT_DIR / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_embeddings:
        np.save(out_dir / "per_layer.npy", per_layer.astype(np.float32))
        np.save(out_dir / "esmc.npy", esmc.astype(np.float32))

    rng = np.random.default_rng(0)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    rows = []

    sources = [(f"layer{li}", per_layer[li]) for li in range(per_layer.shape[0])]
    sources.append(("esmc_encoder", esmc))

    for name, x in sources:
        geo = geometry(x, rng)
        for mode in modes:
            xp = postprocess(x, mode)
            rec = {"source": name, "post": mode, **geo}
            rec.update(cluster_scores(xp, true, K_SWEEP))
            if not args.skip_probe:
                rec.update(probe_scores(xp, true))
            rows.append(rec)
            print(
                f"  {name:14s} {mode:7s} ARI={rec['ari_mean']:.4f} "
                f"NMI={rec['nmi_mean']:.4f} Pur={rec['purity_mean']:.4f} "
                + (f"probe={rec.get('probe_auroc', float('nan')):.4f} " if not args.skip_probe else "")
                + f"| cos={geo['mean_cos']:+.4f} effrank={geo['eff_rank']:.1f}",
                flush=True,
            )

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "layer_sweep.csv", index=False)
    with (out_dir / "meta.json").open("w") as fh:
        json.dump(
            {
                "checkpoint": str(ckpt),
                "tag": args.tag,
                "d_model": bundle.d_model,
                "n_layers": bundle.n_layers,
                "n_units": int(len(df)),
                "k_sweep": K_SWEEP,
                "modes": modes,
                "disable_esmc": bool(args.disable_esmc),
            },
            fh,
            indent=2,
        )
    print(f"\n-> {out_dir}/layer_sweep.csv")


if __name__ == "__main__":
    main()
