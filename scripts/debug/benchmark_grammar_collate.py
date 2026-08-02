"""Benchmark grammar collate and decompose training wait_s.

Run:
    source activate /vepfs-mlp2/c20250601/251105016/conda/envs/pllm
    python scripts/debug/benchmark_grammar_collate.py
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from dllm.pipelines.qwen3_vl_arch.data.esm_encoding import Esm2SequenceTokenizer
from dllm.pipelines.qwen3_vl_arch.data.grammar import (
    GrammarArrowSource,
    GrammarArrowSourceConfig,
    GrammarBioSeqCollator,
    GrammarRenderer,
    GrammarTokenizer,
)

DATA = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/bioseq_grammar_v1")


def parse_wait_decomposition(log_path: str, step: int = 1000) -> None:
    text = Path(log_path).read_text(errors="replace")
    rows: list[tuple[int, float, float | None, float | None]] = []
    for match in re.finditer(rf"batch_ready step={step} micro=(\d+) wait_s=([\d.]+)", text):
        micro = int(match.group(1))
        wait = float(match.group(2))
        pos = match.end()
        fw_match = re.search(
            rf"\[rank=0\] forward_done step={step} micro={micro} .* forward_s=([\d.]+)",
            text[pos : pos + 800],
        )
        bw_match = re.search(
            rf"\[rank=0\] backward_done step={step} micro={micro} .* backward_s=([\d.]+)",
            text[pos : pos + 1200],
        )
        rows.append(
            (
                micro,
                wait,
                float(fw_match.group(1)) if fw_match else None,
                float(bw_match.group(1)) if bw_match else None,
            )
        )

    print(f"\n=== wait decomposition: {Path(log_path).parent.parent.parent.parent.name} step={step} ===")
    collate_estimates: list[float] = []
    compute_times: list[float] = []
    for micro, wait, fwd, bwd in rows:
        if fwd is None or bwd is None:
            print(f"  micro={micro} wait={wait:.2f}s (missing fwd/bwd)")
            continue
        compute = fwd + bwd
        compute_times.append(compute)
        print(f"  micro={micro} wait={wait:.2f}s compute={compute:.2f}s (fwd={fwd:.2f}+bwd={bwd:.2f})")

    for index in range(1, len(rows)):
        _, wait, _, _ = rows[index]
        prev_fwd, prev_bwd = rows[index - 1][2], rows[index - 1][3]
        if prev_fwd is None or prev_bwd is None:
            continue
        collate_estimates.append(wait - prev_fwd - prev_bwd)

    if compute_times:
        print(f"  avg compute/micro: {sum(compute_times)/len(compute_times):.2f}s")
    if collate_estimates:
        print(f"  avg est collate+fetch/micro: {sum(collate_estimates)/len(collate_estimates):.2f}s")


def benchmark_collate() -> None:
    sources = ["oas", "ots", "tcr", "ppi"]
    tokenizer = GrammarTokenizer(Esm2SequenceTokenizer())
    collator = GrammarBioSeqCollator(tokenizer)
    renderer = GrammarRenderer(tokenizer)

    print("\n=== collate benchmark ===")
    for source_name in sources:
        cfg = GrammarArrowSourceConfig(name=source_name, path=DATA, split="train", weight=1.0)
        source = iter(GrammarArrowSource(cfg))
        records = [next(source) for _ in range(4)]

        start = time.time()
        repeats = 30
        for _ in range(repeats):
            collator(records)
        encode_and_collate = (time.time() - start) / repeats

        preencoded = [renderer.encode(record) for record in records]
        start = time.time()
        for _ in range(repeats):
            collator(preencoded)
        collate_only = (time.time() - start) / repeats

        batch = collator(records)
        print(
            f"  {source_name:>3}: encode+collate={encode_and_collate * 1000:5.1f}ms  "
            f"collate_only={collate_only * 1000:5.1f}ms  "
            f"decoder={tuple(batch['input_ids'].shape)} encoder={tuple(batch['encoder_input_ids'].shape)}"
        )


def main() -> None:
    logs = [
        "/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v1_esmc300m/wandb/wandb/run-20260622_030325-69j9jkq0/files/output.log",
        "/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_esmc600m_cmp500k/wandb/wandb/run-20260629_154629-44e2gs58/files/output.log",
        "/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_no_encoder_1b_cmp500k/wandb/wandb/run-20260629_150550-bk66ksgm/files/output.log",
    ]
    for log in logs:
        if Path(log).exists():
            parse_wait_decomposition(log)
    benchmark_collate()


if __name__ == "__main__":
    main()
