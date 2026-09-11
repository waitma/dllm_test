# Training Speed Analysis（current immune LLaDA line）

> This document separates the current prepared-data speed boundary from historical
> `grammar_v2` measurements. Historical numbers are retained as evidence only and
> must not be read as a current validation result.
>
> Last updated: 2026-09-12. v5 `immune_v5_receptor_completion` is published
> (counts: plan §4.2). Epitope-source rows grow per-row tokens; the CPU loader
> table below is **not** a v5 measurement (v5 DataLoader / GPU throughput
> unmeasured).

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
