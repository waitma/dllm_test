# Training Speed Analysis（current immune LLaDA line）

> This document separates the current prepared-data speed boundary from historical
> `grammar_v2` measurements. Historical numbers are retained as evidence only and
> must not be read as a current validation result.
>
> Last updated: 2026-09-12. v5 `immune_v5_receptor_completion` is published
> (counts: plan §4.2). The CPU loader table below is still **not** a v5
> measurement. The queued v5 8-GPU job (`t-20260912021346-5xq4s`) has not
> launched, so there is no live v5 DataLoader / GPU profile. The section
> 「v5 8-GPU proxy measurements」below is from matching historical v3 runs
> (same global batch 256, or the same per-device/ga load), not from that job.

## Current speed boundary

The current training entry is:

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py`

The active data implementation is:

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data`

The measured path to profile is:

```text
prepared semantic JSONL + manifest
  -> prepared loader / byte-offset decode
  -> current immune grammar rendering
  -> batch padding and tensor assembly
  -> per-chain ESMC input reconstruction
  -> diffusion/MLM masking
  -> GPU forward/backward
```

This line does not consume the deleted qwen `data` alias, deleted `training` modules, raw-CSV
runtime loader, infinite weighted Arrow stream, or model-ready token cache. The retained
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py`
and related model-layer files may still provide encoder/model primitives, but they do not define
the current data-loader speed boundary.

## Existing CPU prepared-loader baseline

The following values are prior prepared-loader measurements and are included for planning only:

| `num_workers` | throughput |
|---:|---:|
| 0 | **1367.76 samples/s** |
| 1 | **721.62 samples/s** |
| 2 | **1337.44 samples/s** |

These values are not a fresh run in this documentation pass. They do not establish the best
setting for GPU training, DDP, FSDP, or the current target hardware. In particular, do not infer
that `num_workers=1` or `num_workers=2` is universally correct from this table.

## v5 8-GPU proxy measurements (2026-09-12)

Worksheet (outside the repo):
`/vepfs-mlp2/c20250601/251105016/project/_jobmon/DECISION_NUMBERS.md`.
Decision recorded in `PROJECT_PROCESS.md` (2026-09-12 v5 entry) and
`examples/llada/PROTEIN_PRETRAIN_PROGRESS.md` §6.3 / §6.7:
`--save_top_k 3`, `--eval_steps 1000`, `ActiveDeadlineSeconds 950400` stay.

**Lesson:** do not copy one run's s/it onto another `per_device` / `ga` setup.
Read `training_args.bin` for `per_device × ga × world_size` first.

| Config | Evidence | Actual factors | Measured / inferred s/it |
|---|---|---|---|
| Historical v3 8-GPU | `t-20260901033935-h55f4`, `t-20260904014842-qgjbg` | `per_device=2`, `ga=16`, `world_size=8`, `max_length=1024`, global **256** | **2.94–2.95 s/it** (tqdm) |
| v3 4-GPU, same per-GPU load as v5 | `..._v3_4gpu` `checkpoint-42000/training_args.bin` | `per_device=4`, `ga=8`, `world_size=4` | **1.51–1.91 s/it** (tqdm) |
| Queued v5 8-GPU | `t-20260912021346-5xq4s` (not launched) | `per_device=4`, `ga=8`, 8 GPU, global **256** | **1.6–2.5 s/it** (center ~2.2) → train **4.0–5.8 days** |

Microbatch scaling is **sublinear**: `t(batch4)/t(batch2) ≈ 1.29`, so v5 step wall-clock
is about **0.65×** the v3 8-GPU 2.95 s/it (`0.5 × 1.29`: half as many accumulation
steps, each microstep only 1.29× slower).

**Eval lesson:** never extrapolate a single-source rate to the full valid split.

- `eval_oas_runtime=4.4792` at 2000 rows/source → OAS **446.5 samples/sec**.
  Source: `..._v3_8gpu_2m/checkpoint-105000`. That run **was already 8-GPU**;
  do not normalize by GPU count again.
- Same 8-GPU eval: `asd_antibody` is only **~114 samples/sec** (~3.9× slower)
  and is 44% of v5 valid.
- Per-source extrapolation on v5 valid **102,308** rows: one full eval
  **~8.7 min**; `--eval_steps 1000` × 200 evals **~29 h ≈ 1.21 days**.

## Historical grammar-v2 evidence

Earlier integrated ESMC + LLaDA logs under historical `output/grammar_v2_*` runs reported roughly
34–40 samples/s on 8 GPUs, with `wait_s` around 0.43–0.66 seconds per microstep after the old
worker/prefetch changes and a recorded peak near 46.0 GB on a 79.2 GB GPU. Those logs belong to
the deleted grammar-v2/Arrow training line. They remain useful only as historical context for
where CPU loading and encoder forward cost appeared, not as a reproducible current baseline.

The historical line also recorded two old optimizations: gradient all-reduce suppression on
non-boundary accumulation steps and worker prefetch overlap. These should be re-measured against
the current prepared loader before being described as deployed current behavior.

## Ranked current profiling plan

### 1. Profile the prepared loader and GPU overlap first

Run a short, representative profile using the current formal entry and current prepared-data
configuration. Record, with absolute artifact paths:

- prepared-loader startup/index-build time and steady-state samples/s;
- JSON decode, grammar rendering, padding, encoder reconstruction, and masking time;
- host-to-device wait time and GPU forward/backward time;
- peak CUDA memory, effective tokens/s, and batch shape/length distribution;
- DDP/FSDP rank synchronization and first-batch identity where applicable.

The profile must distinguish index construction from steady-state iteration. The prepared loader
still decodes semantic rows and performs runtime construction; it is not a zero-cost cache.

### 2. Gate worker/prefetch choices on the profile

Compare `num_workers=0`, `1`, and only then higher values on the target environment. Also compare
persistent workers and prefetch settings only when they are supported by the active loader. Do not
hard-code the historical `nw=1` rule: the old Arrow worker-sharding rationale does not establish
an equivalent invariant for the current prepared semantic loader.

For every candidate, verify that all ranks receive compatible batch shapes/tasks and that no
worker-specific ordering or shard identity causes DDP divergence. Select the setting by measured
GPU overlap and end-to-end samples/tokens per second, not CPU-only throughput alone.

### 3. Profile length-aware batching before increasing model batch size

The current grammar and per-chain encoder reconstruction make padding and sequence length a major
cost. Prefer length buckets or a token-budget batch policy if the active collator can preserve
record/task semantics and deterministic distributed sampling. First measure padding waste and
maximum encoder length; then compare fixed sample count with token-budget batches.

Every change must be short-run profiled on the target GPU. Keep effective global batch comparable
when comparing encoder variants, and record peak memory before committing a new batch size.

### 4. Measure gradient accumulation and checkpointing separately

Retain gradient accumulation or non-boundary synchronization changes only after checking their
actual communication and wall-clock effect on the current Trainer/DDP/FSDP path. Likewise, compare
whole-model and selective activation checkpointing with the current fusion architecture. Report
peak memory and tokens/s together; a faster microstep that reduces feasible batch size may be a
regression at the global-batch level.

### 5. Treat flash attention as environment-gated

The retained model layer exposes the ESMC flash-attention option, but the option is not evidence
that flash attention is installed or enabled in the current environment. If the package is
available, run numerical/parity checks and a short throughput comparison against the non-flash
path. If it is unavailable, the fallback must remain the documented behavior; do not report a
speedup or claim that the current baseline uses flash attention.

## Recommendation

The next speed action is a profile of the current prepared loader plus GPU overlap on the formal
`protein_pretrain_esmc.py` line. Use the result to choose workers, prefetch, length bucketing,
token budget, batch size, and checkpointing. Do not reuse deleted qwen data/training modules or
fix a worker count from historical Arrow measurements. The main agent must append the actual
command, exit code, elapsed time, and absolute profile artifact to
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md`; this documentation update
does not claim that the latest speed or full-model validation passed.
