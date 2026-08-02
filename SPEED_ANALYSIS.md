# Training Speed Analysis (integrated LLaDA line)

> Profile of the integrated ESMC-300M + LLaDA run and a ranked list of speed
> levers. Evidence is from real run logs (`output/grammar_v2_*/logs/*rank0.log`)
> and the prior profiling in `PROJECT_PROCESS.md` (2026-07-05 / 07-08 speed
> sections). Method decisions mirror `.cursor/rules/volc-batch-sizing.mdc`.
>
> Last updated: 2026-07-09.

## Current config (integrated)
`train_jobs/qwen3_vl_bioseq_grammar_v2_esmc300m_integrated_llada.yml`:
ESMC-300M encoder (trainable, `--encoder-lr 2e-5`) + LLaDA decoder 28L/16H/3840,
`--batch-size 4 --grad-accum 2` on 2×8 = 16 GPU → global batch 128;
`--num-workers 1`, `--gradient-checkpointing`, `--qk-norm`, `--condition-norm`,
`--bf16`, `--find-unused-parameters`, `--max-sequence-length 2112`.

## Measured baseline (from real logs)
ESMC-300M + LLaDA, 8-GPU cmp500k run (same backbone), nw=1
(`output/grammar_v2_esmc300m_cmp500k_llada/logs/train_20260708_055104_rank0.log`):

| Signal | Value | Note |
|---|---|---|
| throughput | ~34–40 samples/s | 8 GPU, gb128, bs4 ga4 |
| `wait_s` (data load) | 0.43–0.66 / microstep | after nw0→nw1/2 fix (was ~1.03 at nw0) |
| `backward_s` | ~0.41 / microstep | includes DDP all-reduce on sync microsteps |
| `mem_peak` | 46.0 GB / 79.2 GB | **~33 GB headroom** at bs4 |
| corrupt_rate | ~0.50 | as designed (t~U(eps,1)) |

Two speedups already deployed (`PROJECT_PROCESS.md`): (1) `no_sync()` on
non-boundary grad-accum microsteps (saves (ga−1)/ga of gradient all-reduce,
math-identical); (2) `num_workers` + `persistent_workers` + `prefetch_factor=4`
to overlap the CPU-heavy `GrammarBioSeqCollator` with GPU compute.

## Ranked levers

### 1. ESMC flash-attention (low risk, high benefit) — gated on flash_attn + parity
The `--encoder-use-flash-attn` flag already exists and flows into
`BioSeqLLaDAEncoderDiffusionModel.from_esmc(use_flash_attn=...)`. Training runs
ESMC via the **`input_ids`** path (`encode_chain_tokens` →
`self.encoder(input_ids=..., attention_mask=...)`), which is flash-compatible
(the guard that rejects flash only triggers on the `inputs_embeds`/
`diffusion_state` path, which training does not use). The encoder forward is a
large slice of `forward_s`, so this is the cheapest real win.

**Status (B3)**: `load_local_esmc_encoder` now **guards** the flag — if
`flash_attn` is not importable it warns and falls back to non-flash SDPA, so the
flag is a safe no-op. `flash_attn` is **not** installed in the local
`protenix_abtcr` env (prebuilt wheel download over the proxy is prohibitively
slow), so the bf16 parity check
(`scripts/tests/bioseq/check_esmc_flash_parity.py`) currently SKIPs. **To turn it
on for real**: install `flash_attn` in the training image
(`airgen:v1`, torch 2.8 / cu12 / py312 → wheel
`flash_attn-*+cu12torch2.8cxx11abiTRUE-cp312`), run the parity check until it
prints `PARITY OK`, then add `--encoder-use-flash-attn` to the retrain YAML. The
guard means adding the flag before the package exists will not crash training —
it simply won't speed anything up.

### 2. Token-based dynamic batching (medium effort, high benefit) — P2, never done
`max_sequence_length=2112` but most records (single/paired receptors) are far
shorter, so fixed `batch_size=4` wastes compute/memory on padding. Pack each
microbatch to a **token budget** instead of a fixed sample count (still
task-homogeneous per the grammar loader). Expected: higher effective tokens/s at
equal memory. Touches `dllm/pipelines/qwen3_vl_arch/data/{mixture,datamodule}.py`;
must preserve DDP shard identity (num_workers issue).

### 3. Micro-batch fill (medium risk) — profile before changing
46/79 GB leaves headroom, but raising bs at fixed gb128 means bs8 ga1
(16 GPU). Historically bs8 OOM'd on the 300M cmp500k run at a *shorter* seq
length, and here seq=2112 + gradient-checkpointing pushes long-sequence batches
higher. Rule: profile a short run on the target flavor, keep `mem_peak` ≤~85%
(≈67 GB) before committing. Likely safe target: bs6 (needs gb rework) or keep bs4
and instead spend the headroom on lever 2.

### 4. Selective gradient checkpointing (medium) 
GC is **required** (OOM without it — verified). But it need not wrap every
module: checkpointing only the LLaDA decoder blocks (leaving the ESMC encoder
un-checkpointed, or vice versa) can recover compute while staying within memory.
Needs a per-tower GC flag; validate peak memory.

### 5. Do NOT push num_workers>1 under DDP
The infinite weighted Arrow stream is sharded per worker; `num_workers>1`
re-shards independently and can desync the first batch across ranks → NCCL
timeout (`trainer.py` warns; `nw=1` keeps shard identity of `nw=0`). Keep nw=1.
`find_unused_parameters=True` is kept (heterogeneous task batches); disabling it
was tried and abandoned (DDP crash risk), profiling showed no measurable gain.

## Recommendation
Implement lever 1 (flash-attn, = B3) with the integrated retrain (B4); it is
low-risk and reduces the dominant encoder forward cost. Schedule lever 2
(token-based batching) as the next structural improvement; treat levers 3–4 as
profile-gated tuning. Every batch-size change must be preceded by a short
`mem_peak` profile per the batch-sizing rule.
