"""Run official GIANA clustering for Track-A generated/reference CDR3beta sets.

Install the isolated FAISS dependency and score with::

    /project_c20250601/251105016/conda/envs/protenix_abtcr/bin/pip install \
      --target /project_c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/GIANA/_deps faiss-cpu
    /project_c20250601/251105016/conda/envs/pllm/bin/python \
      /project_c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/run_public_beta.py score --with-giana

GIANA is invoked without V-gene information because the public generation
interface produces CDR3beta only.  Each model/target is clustered separately
with its own references, exactly as required by the benchmark protocol.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Mapping

import pandas as pd

from .track_a import GianaMetrics, Target, clean_sequence, coerce_generation_frame, is_valid_cdr3b


BENCH = Path(__file__).resolve().parents[1]
GIANA_REPO = BENCH / "baselines" / "GIANA"
GIANA_SCRIPT = GIANA_REPO / "GIANA4.1.py"
GIANA_DEPS = GIANA_REPO / "_deps"
GIANA_PYTHON = Path("/project_c20250601/251105016/conda/envs/protenix_abtcr/bin/python")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _eligible(sequence: str, start_position: int) -> bool:
    # Official GIANA help requires L >= startPosition + 7.  Track-A validity is
    # kept separate; short valid CDR3s remain valid but are GIANA-ineligible.
    return (
        is_valid_cdr3b(sequence)
        and len(sequence) >= start_position + 7
        and "_" not in sequence
        and "*" not in sequence
    )


def check_ready(python: Path = GIANA_PYTHON) -> tuple[bool, str]:
    if not GIANA_SCRIPT.is_file():
        return False, f"missing official GIANA source {GIANA_SCRIPT}"
    if not python.is_file():
        return False, f"missing GIANA runtime {python}"
    environment = os.environ.copy()
    if GIANA_DEPS.is_dir():
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(GIANA_DEPS), environment.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
    check = subprocess.run(
        [str(python), "-c", "import faiss, Bio, sklearn"],
        cwd=GIANA_REPO,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if check.returncode:
        return False, f"GIANA dependency import failed: {check.stdout.strip()}"
    return True, "official GIANA4.1 source and dependencies are ready"


def run_giana_block(
    generated: list[str],
    references: tuple[str, ...],
    work_dir: Path,
    *,
    python: Path = GIANA_PYTHON,
    threshold: float = 7.0,
    smith_waterman_threshold: float = 3.5,
    start_position: int = 3,
    threads: int = 1,
) -> GianaMetrics:
    """Cluster one generated/reference block and count reference-cluster hits."""

    # GIANA is executed with its own repository as cwd because it reads bundled
    # resources there.  Resolve all benchmark paths first so caller-provided
    # relative result directories still point to the intended files.
    work_dir = work_dir.resolve()
    generated_unique = list(dict.fromkeys(clean_sequence(x) for x in generated))
    generated_eligible = [x for x in generated_unique if _eligible(x, start_position)]
    reference_unique = list(dict.fromkeys(clean_sequence(x) for x in references))
    reference_eligible = [x for x in reference_unique if _eligible(x, start_position)]
    if not generated_eligible:
        return GianaMetrics(0, 0, 0, 0, "no_eligible_generated_sequences")
    if not reference_eligible:
        return GianaMetrics(0, 0, 0, len(generated_eligible), "no_eligible_references")

    ready, reason = check_ready(python)
    if not ready:
        return GianaMetrics(0, 0, 0, len(generated_eligible), f"unavailable: {reason}")

    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / "giana_input.tsv"
    output_path = work_dir / "giana_clusters.tsv"
    log_path = work_dir / "giana.log"
    with input_path.open("w") as handle:
        handle.write("CDR3\tsource\tsequence_id\n")
        for index, sequence in enumerate(generated_eligible, start=1):
            handle.write(f"{sequence}\tgenerated\tg{index:06d}\n")
        for index, sequence in enumerate(reference_eligible, start=1):
            handle.write(f"{sequence}\treference\tr{index:06d}\n")

    environment = os.environ.copy()
    if GIANA_DEPS.is_dir():
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(GIANA_DEPS), environment.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
    command = [
        str(python), str(GIANA_SCRIPT),
        "-f", str(input_path),
        "-O", str(output_path),
        "-o", str(work_dir),
        "-v",
        "-t", str(threshold),
        "-S", str(smith_waterman_threshold),
        "-T", str(start_position),
        "-N", str(threads),
    ]
    process = subprocess.run(
        command,
        cwd=GIANA_REPO,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text("$ " + " ".join(command) + "\n\n" + process.stdout)
    if process.returncode != 0 or not output_path.is_file():
        tail = " | ".join(process.stdout.splitlines()[-5:])
        return GianaMetrics(
            0, 0, 0, len(generated_eligible),
            f"failed_exit_{process.returncode}: {tail}",
        )

    records = []
    for line in output_path.read_text().splitlines():
        if not line or line.startswith("##"):
            continue
        fields = line.split("\t")
        if len(fields) < 4:
            continue
        records.append({
            "cdr3b": fields[0],
            "cluster": fields[1],
            "source": fields[2],
            "sequence_id": fields[3],
        })
    clustered = pd.DataFrame(records, columns=["cdr3b", "cluster", "source", "sequence_id"])
    if clustered.empty:
        return GianaMetrics(0, 0, 0, len(generated_eligible), "ok_no_clusters")

    reference_clusters = set(clustered.loc[clustered["source"] == "reference", "cluster"])
    generated_hits = clustered[
        (clustered["source"] == "generated")
        & (clustered["cluster"].isin(reference_clusters))
    ]
    covered_clusters = set(generated_hits["cluster"])
    covered_references = clustered[
        (clustered["source"] == "reference")
        & (clustered["cluster"].isin(covered_clusters))
    ]
    return GianaMetrics(
        generated_hit=int(generated_hits["sequence_id"].nunique()),
        reference_clusters_covered=len(covered_clusters),
        reference_sequences_covered=int(covered_references["sequence_id"].nunique()),
        generated_eligible=len(generated_eligible),
        status="ok",
    )


def run_all_giana(
    generations: pd.DataFrame,
    targets: Mapping[str, Target],
    work_root: Path,
    *,
    num_candidates: int = 1000,
    python: Path = GIANA_PYTHON,
    threads: int = 1,
) -> dict[tuple[str, str], GianaMetrics]:
    """Run official GIANA once per model-target block."""

    generations = coerce_generation_frame(generations)
    metrics: dict[tuple[str, str], GianaMetrics] = {}
    grouped = list(generations.groupby(["model", "target_id"], sort=True))
    for index, ((model, target_id), block) in enumerate(grouped, start=1):
        ordered = block.sort_values("rank")
        sequences = [
            clean_sequence(x)
            for x in ordered.loc[ordered["rank"] <= num_candidates, "cdr3b"].tolist()
            if is_valid_cdr3b(x)
        ]
        block_dir = work_root / _safe_name(str(model)) / _safe_name(str(target_id))
        result = run_giana_block(
            sequences,
            targets[str(target_id)].references,
            block_dir,
            python=python,
            threads=threads,
        )
        metrics[(str(model), str(target_id))] = result
        print(
            f"GIANA [{index}/{len(grouped)}] {model} {target_id}: "
            f"hit={result.generated_hit} status={result.status}",
            flush=True,
        )
    return metrics
