"""Publication-quality architecture figure for the *current* BioSeq training setup.

Uses the `architecture-diagram` skill (Graphviz dot auto-layout). Scope = exactly
what we train today: BioSeqLLaDAEncoderDiffusionModel = ESMC per-chain encoder +
LLaDA decoder, bidirectional masked diffusion, loss on corrupted target residues.
No MoE / no-encoder / in-house-decoder alternatives (not in use).

Run:  python scripts/gen_arch_figure.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/vepfs-mlp2/c20250601/251105016/project/.cursor/skills/architecture-diagram/scripts")
from gv_diagram import (  # noqa: E402
    Graph,
    PALETTE,
    chain_parallel_block,
    leaf,
    render,
    stage,
)

OUT = "/vepfs-mlp2/c20250601/251105016/project/dllm_test/bioseq_model_architecture"

g = Graph(
    title="BioSeq Diffusion Model \u2014 Training Architecture",
    subtitle=("ESMC per-chain encoder + LLaDA denoiser &#183; bidirectional masked diffusion "
              "&#183; loss only on corrupted target residues"),
)

g.node("DATA", stage("Data pipeline", *PALETTE["data"], [
    ("BioSeqRecord", "chains &#183; task &#183; source"),
    ("Grammar render", "tokens + fixed / target masks"),
    ("Collate", "pad &#8594; per-chain streams [B, C, L]"),
    ("Diffusion corrupt", "t ~ U(&#949;, 1) &#8594; x_t, labels"),
], accent_last=PALETTE["accent"][1]))

enc_bar, enc_fill = PALETTE["enc"]
g.node("ENC", stage("ESMC encoder &#183; per-chain, trainable", enc_bar, enc_fill, [
    ("Corruption mirror", "x_t &#8594; encoder_input_ids [B, C, L]"),
    chain_parallel_block(
        enc_bar,
        caption="independent encode per chain &#183; flatten [B&#183;C, L] &#8594; reshape [B, C, L, E]",
    ),
    ("Gather to token stream", "chain feats &#8594; token_condition [B, S, E]"),
]))

dec_bar, dec_fill = PALETTE["dec"]
g.node("DEC", stage("LLaDA decoder &#183; bidirectional", dec_bar, dec_fill, [
    ("Fused input embedding", "wte(x_t) at grammar sites; residue sites &#8592; ESMC"),
    ("Transformer &#215; N", "RoPE &#183; RMSNorm &#183; SwiGLU &#183; is_causal=False"),
    ("Final norm + LM head", "&#8594; logits [B, S, V]"),
]))

g.node("LOSS", leaf(
    "Masked cross-entropy",
    "corrupted target residues only<br/>forbidden tokens masked",
    *PALETTE["loss"],
))

g.edge("DATA", "ENC", "x_t")
g.edge("ENC", "DEC", "token_condition")
g.edge("DATA", "DEC", "x_t (wte base)", dashed=True, muted=True, constraint=False)
g.edge("DEC", "LOSS", "logits")

render(g, OUT)
