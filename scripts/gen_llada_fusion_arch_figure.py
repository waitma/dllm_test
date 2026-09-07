"""Render the examples/llada ESMC->LLaDA-8B fusion training architecture.

Draws the variant that is actually trained by
``examples/llada/protein_pretrain_esmc.py`` (residue_cond_mode=add,
trainable ESMC-300M, pretrained LLaDA-8B decoder).
"""
from __future__ import annotations

import sys
from pathlib import Path

_SKILL = Path("/vepfs-mlp2/c20250601/251105016/project/.cursor/skills/architecture-diagram/scripts")
sys.path.insert(0, str(_SKILL))

from gv_diagram import Graph, PALETTE, chain_parallel_block, leaf, render, stage  # noqa: E402

OUT = "/vepfs-mlp2/c20250601/251105016/project/dllm_test/llada_fusion_architecture"

g = Graph(
    title="LLaDA-8B &#215; ESMC fusion &#8212; training architecture (examples/llada)",
    subtitle=(
        "pretrained LLaDA-8B masked-diffusion decoder &#183; trainable ESMC-300M "
        "residue encoder &#183; grammar-v2 immune sequences"
    ),
    rankdir="TB",
)

g.node("DATA", stage("Data &#183; grammar-v2", *PALETTE["data"], [
    ("Immune sources",
     "OAS H/L &#183; OTS &#945;/&#946; &#183; ASD Ab/Nb &#183; TRAIT &#183; tcr_native"),
    ("GrammarRenderer",
     "&lt;prots&gt; &lt;ab&gt; res&#8230; &lt;chainsep&gt; res&#8230; &lt;protd&gt;"),
    ("GrammarBioSeqCollator",
     "decoder stream [B,S] + per-chain encoder [B,C,L]"),
    ("RemapCollator",
     "grammar id &#8594; LLaDA id &#183; +42 tokens (25 &lt;res_X&gt;, 16 grammar, &lt;chainsep&gt;)"),
]))

g.node("NOISE", stage("Corruption &#8594; x_t", *PALETTE["accent"], [
    ("diffusion (default)",
     "per-sequence t &#126; U(&#949;,1) &#183; selected &#8594; 100% &lt;mask&gt;"),
    ("bert (ablation)",
     "fixed 15% selected &#183; 80 / 10 / 10"),
    ("Encoder mirror",
     "same corrupted positions masked on the ESMC stream &#8594; no target leak"),
]))

g.node("ENC", stage("ESMC-300M encoder &#183; trainable", *PALETTE["enc"], [
    chain_parallel_block(
        PALETTE["enc"][0],
        caption="flatten [B,C,L] &#8594; [B&#183;C,L] &#183; one forward &#183; reshape back",
    ),
    ("Per-chain residue features",
     "d_model 960 &#183; 30 layers &#183; vocab 64 &#183; no cross-chain mixing"),
    ("Gather to decoder positions",
     "[B,C,L,960] &#8594; token_condition [B,S,960]; specials zeroed"),
]))

g.node("CONN", stage("Condition connector", *PALETTE["out"], [
    ("condition_norm",
     "LayerNorm(960) &#183; rescales the small ESMC signal"),
    ("condition_proj",
     "Linear 960 &#8594; 4096, no bias &#183; cast to decoder dtype (bf16)"),
    ("Inject at residue sites only",
     "add: wte(x_t) + proj&#183;m &#183; feature: replace &#183; token: no ESMC"),
]))

g.node("DEC", stage("LLaDA-8B decoder &#183; bidirectional", *PALETTE["dec"], [
    ("inputs_embeds [B,S,4096]",
     "fused token embedding, no timestep input"),
    ("32 &#215; LLaDALlamaBlock",
     "RoPE &#183; RMSNorm &#183; SwiGLU (mlp 12288) &#183; is_causal=False"),
    ("LM head",
     "logits [B,S,126464] &#183; weights inherited from 8B pretraining"),
]))

g.node("LOSS", leaf(
    "Masked cross-entropy",
    "corrupted target positions only<br/>(labels = &#8722;100 elsewhere)",
    *PALETTE["loss"],
))

g.edge("DATA", "NOISE", "batch")
g.edge("NOISE", "ENC", "encoder x_t [B,C,L]")
g.edge("ENC", "CONN", "token_condition")
g.edge("CONN", "DEC", "cond_h [B,S,4096]")
g.edge("DEC", "LOSS", "logits")
g.edge("NOISE", "DEC", "decoder x_t [B,S] &#8594; wte",
       dashed=True, muted=True, constraint=False)

render(g, OUT)
