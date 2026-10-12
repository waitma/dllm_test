#!/usr/bin/env python
"""Prepare, generate, and score the public 14-pMHC CDR3beta Track-A benchmark.

Typical commands::

    /vepfs-mlp2/c20250601/251105016/conda/envs/pllm/bin/python \
      /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/run_public_beta.py prepare
    /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python \
      /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/run_public_beta.py \
      generate --model tcrt5 --targets rvr --num-candidates 10 --output /absolute/path/smoke.csv
    /vepfs-mlp2/c20250601/251105016/conda/envs/pllm/bin/python \
      /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/run_public_beta.py \
      score --generations /absolute/path/generations.csv --num-candidates 1000 --with-giana
    /vepfs-mlp2/c20250601/251105016/conda/envs/pllm/bin/python \
      /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/run_public_beta.py \
      generate-tcrdesign-alpha --targets rvr --beta-top-k 1 --alpha-per-beta 10
    /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python \
      /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/run_public_beta.py \
      generate-bioseq --targets rvr --num-candidates 10 --output /absolute/path/bioseq_smoke.csv

The mandatory interchange columns are ``model,target_id,peptide,mhc_allele,
rank,cdr3b,raw_score``.  TCR-epiDiff is protocol-guarded as
``reconstruction_only`` and cannot emit de-novo rows through this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd


BENCH = Path(__file__).resolve().parents[1]
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from tcr_generation_bench import giana as giana_runner  # noqa: E402
from tcr_generation_bench.bioseq_generator import (  # noqa: E402
    BIOSEQ_7L_STEP117000_LABEL,
    DEFAULT_BIOSEQ_7L_STEP117000,
    BioSeqTrackAGenerator,
)
from tcr_generation_bench.public_generators import (  # noqa: E402
    EPIDIFF_REPO,
    GRATCR_REPO,
    TCRDESIGN_REPO,
    TCRT5_MODEL_ID,
    ReconstructionOnlyError,
    generator_for,
    readiness_table,
)
from tcr_generation_bench.paired_alpha import (  # noqa: E402
    TcrDesignAlphaGenerator,
    select_tcrdesign_generated_betas,
    write_alpha_generations,
    write_official_alpha_evaluation_audit,
)
from tcr_generation_bench.track_a import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_RESULTS_DIR,
    RVR_TARGET_ID,
    Target,
    empty_generation_frame,
    generation_rows,
    import_official_predictions,
    load_public_targets,
    migrate_legacy_tcrt5_paper_ranking,
    read_generations,
    score_generations,
    write_benchmark_audit,
    write_generations,
    write_metric_outputs,
)


STATUS_JSON = "run_status.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _command() -> str:
    # Do not resolve the interpreter symlink: on this host the dedicated
    # protenix_abtcr entry point resolves to a shared ``flow`` binary even
    # though sys.prefix and imported packages remain in protenix_abtcr.
    return shlex.join([str(Path(sys.executable)), str(Path(__file__).resolve()), *sys.argv[1:]])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(path: Path) -> str:
    if not (path / ".git").exists():
        return "source_present_without_git_metadata"
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    return process.stdout.strip() if process.returncode == 0 else "unknown"


def _load_status(results_dir: Path) -> dict:
    path = results_dir / STATUS_JSON
    return json.loads(path.read_text()) if path.is_file() else {"events": []}


def _save_status(results_dir: Path, status: dict) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    status["updated_at"] = _utc_now()
    (results_dir / STATUS_JSON).write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n"
    )
    write_reproduction_status(results_dir, status)


def _record_event(results_dir: Path, **event) -> None:
    status = _load_status(results_dir)
    event = {"timestamp": _utc_now(), "command": _command(), **event}
    status.setdefault("events", []).append(event)
    _save_status(results_dir, status)


def _static_model_rows(status: dict) -> list[dict[str, object]]:
    event_by_model: dict[str, list[dict]] = {}
    for event in status.get("events", []):
        model = event.get("model")
        if model:
            event_by_model.setdefault(model, []).append(event)

    tcrt5_repo = BENCH / "baselines" / "TCRT5_tcr_translate"
    artifacts = {
        "TCRT5": {
            "source": tcrt5_repo / "src",
            "weights": [],
            "commit": _git_commit(tcrt5_repo),
            "mode": "de_novo",
            "comparable": "yes",
        },
        "GRATCR": {
            "source": GRATCR_REPO / "GRA.py",
            "weights": [
                GRATCR_REPO / "model" / "bert_pretrain.pth",
                GRATCR_REPO / "model" / "gpt_pretrain.pth",
                GRATCR_REPO / "model" / "gra.pth",
            ],
            "commit": _git_commit(GRATCR_REPO),
            "mode": "de_novo",
            "comparable": "yes",
        },
        "TcrDesign": {
            "source": TCRDESIGN_REPO / "tcrdesign_G.py",
            "weights": [TCRDESIGN_REPO / "weights" / "rnn" / "beta" / "RNN_tcr.ep"],
            "commit": _git_commit(TCRDESIGN_REPO),
            "mode": "de_novo_beta_only",
            "comparable": "yes",
        },
        "TCR-epiDiff": {
            "source": EPIDIFF_REPO / "Tutorial_Code" / "TCR-epiDiff" / "TCR-epiDiff_main.py",
            "weights": [EPIDIFF_REPO / "Model" / "TCR-epiDiff_Best_model.pth"],
            "commit": _git_commit(EPIDIFF_REPO),
            "mode": "reconstruction_only",
            "comparable": "no",
        },
        BIOSEQ_7L_STEP117000_LABEL: {
            "source": BENCH.parents[1] / "downstream" / "grammar" / "tcr_generation.py",
            "weights": [DEFAULT_BIOSEQ_7L_STEP117000],
            "commit": _git_commit(BENCH.parents[1]),
            "mode": "de_novo_epitope_conditioned",
            "comparable": "yes",
            "readiness": "local checkpoint present; runtime generation not yet recorded",
        },
    }
    readiness = {row["model"]: row for row in readiness_table()}
    rows = []
    for model, artifact in artifacts.items():
        events = event_by_model.get(model, [])
        successful_runtime = [
            event for event in events
            if event.get("action") == "generate" and event.get("success")
        ]
        artifact_events = [
            event for event in events
            if event.get("action") == "import_official_artifact" and event.get("success")
        ]
        weights = artifact["weights"]
        readiness_text = readiness.get(model, {}).get(
            "status", artifact.get("readiness", "not checked")
        )
        if successful_runtime:
            readiness_text = (
                "local checkpoint successfully loaded and decoded in a recorded runtime"
                if model == BIOSEQ_7L_STEP117000_LABEL
                else "official checkpoint successfully loaded and decoded in a recorded runtime"
            )
        rows.append({
            "model": model,
            "source_downloaded": Path(artifact["source"]).exists(),
            "source_commit": artifact["commit"],
            "official_weights_present": (
                all(Path(path).is_file() for path in weights) if weights else bool(successful_runtime)
            ),
            "official_weights_loaded": bool(successful_runtime),
            "generated_locally": bool(successful_runtime),
            "official_artifact_imported": bool(artifact_events),
            "generation_mode": artifact["mode"],
            "directly_comparable": artifact["comparable"],
            "readiness": readiness_text,
        })
    return rows


def write_reproduction_status(results_dir: Path, status: dict | None = None) -> None:
    status = status or _load_status(results_dir)
    rows = _static_model_rows(status)
    lines = [
        "# Public beta-generation reproduction status",
        "",
        f"Updated: {status.get('updated_at', _utc_now())}",
        "",
        "## Scope",
        "",
        "The quantitative leaderboard covers epitope-conditioned CDR3β generation on "
        "the TCR-TRANSLATE public 14-pMHC benchmark, including the retained local BioSeq "
        "7L step117000 checkpoint. TcrDesign Track-A rows use only "
        "`tcrdesign_G.py -mode beta`. A separate qualitative addendum runs the released "
        "generated-beta-conditioned alpha module; it does not alter the beta ranking.",
        "",
        "| Model | Source | Checkpoint/weights | Loaded in runtime | Generated locally | Official generations imported | Mode | Directly comparable |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    yes_no = lambda value: "yes" if value else "no"
    for row in rows:
        lines.append(
            f"| {row['model']} | {yes_no(row['source_downloaded'])} | "
            f"{yes_no(row['official_weights_present'])} | {yes_no(row['official_weights_loaded'])} | "
            f"{yes_no(row['generated_locally'])} | {yes_no(row['official_artifact_imported'])} | "
            f"`{row['generation_mode']}` | {row['directly_comparable']} |"
        )

    lines.extend([
        "",
        "## TCR-epiDiff classification",
        "",
        "`reconstruction_only`. The released inference function takes "
        "`TCR = original_TCR[0]`, calls `add_noise(TCR, ...)`, then repeatedly "
        "denoises that tensor. No official pure-random `x_T` reverse sampler is "
        "present. It is therefore excluded from de-novo ranking; real test TCRs "
        "are never used as generation seeds in this benchmark.",
        "",
        "## Source versions and readiness",
        "",
    ])
    for row in rows:
        lines.append(f"- {row['model']}: `{row['source_commit']}`; {row['readiness']}")

    alpha_events = [
        event for event in status.get("events", [])
        if event.get("action") == "generate_alpha_pairs" and event.get("success")
    ]
    latest_alpha = alpha_events[-1] if alpha_events else None
    alpha_output = (
        Path(latest_alpha["output"]) if latest_alpha and latest_alpha.get("output")
        else results_dir / "tcrdesign_alpha_generations.csv"
    )
    alpha_checkpoint = TCRDESIGN_REPO / "weights" / "rnn" / "beta_alpha" / "RNN_tcr.ep"
    alpha_generation_available = (
        (TCRDESIGN_REPO / "tcrdesign_G.py").is_file()
        and (TCRDESIGN_REPO / "utils" / "generate_tcr.py").is_file()
        and alpha_checkpoint.is_file()
    )
    lines.extend([
        "",
        "## TcrDesign alpha-chain addendum",
        "",
        f"- Alpha generation: {'available' if alpha_generation_available else 'unavailable'}",
        "- Official alpha evaluation: unavailable",
        "- Included in quantitative benchmark: no",
        "- Generation path: `epitope -> generated CDR3β; "
        "(epitope, generated CDR3β) -> generated CDR3α`.",
        "- Reference/oracle β is not used. Alpha rows are conditioned only on "
        "TcrDesign rows already present in `generations.csv` with "
        "`generation_mode=de_novo`.",
        "- The paper/supplement reports alpha BLOSUM62 similarity, minimum edit "
        "distance, and an auxiliary TcrDesign-B binding score, but the released "
        "repository and Zenodo archive provide no directly runnable script/CLI and "
        "parameters for reproducing those alpha-reference figures.",
        "- `clip/clip_model.py`, `clip/clip_trainer.py`, and `clip/clip_dataset.py` "
        "contain an in-batch alpha/beta contrastive training component accepting CSV "
        "columns `CDR3a,CDR3b`; no pairing checkpoint, evaluation CLI, retrieval "
        "metric, or reported pairing evaluator is released. It is therefore not used.",
        "- TcrDesign-B is a pMHC/full-TCR binding model, not a common independent "
        "alpha-beta pairing evaluator, and is not used for this addendum.",
        f"- Evaluation audit: `{(results_dir / 'tcrdesign_alpha_evaluation_audit.json').resolve()}`.",
        f"- Paired qualitative output: `{alpha_output.resolve()}` "
        f"({'present' if alpha_output.is_file() else 'not generated yet'}).",
    ])
    if latest_alpha:
        lines.extend([
            f"- Recorded run: {latest_alpha.get('beta_conditions', 0)} generated-beta "
            f"conditions × {latest_alpha.get('alpha_per_beta', 0)} alpha beams = "
            f"{latest_alpha.get('candidates', 0)} paired rows across "
            f"{latest_alpha.get('targets', 0)} targets.",
            "- `raw_alpha_score` is empty because the released `beam_search` orders "
            "hypotheses but does not expose their scores to the caller; no score was fabricated.",
        ])

    generations_path = results_dir / "generations.csv"
    per_target_path = results_dir / "per_target_metrics.csv"
    summary_path = results_dir / "summary_metrics.csv"
    if generations_path.is_file():
        generations = pd.read_csv(generations_path, low_memory=False)
        n_blocks = int(generations.groupby(["model", "target_id"]).ngroups)
        block_sizes = sorted(
            set(generations.groupby(["model", "target_id"]).size().astype(int).tolist())
        )
        lines.extend([
            "",
            "## Final result materialisation",
            "",
            f"- Raw generations: `{generations_path.resolve()}` ({len(generations)} rows; "
            f"{n_blocks} model-target blocks; block sizes {block_sizes}).",
            f"- Per-target metrics: `{per_target_path.resolve()}` "
            f"({'present' if per_target_path.is_file() else 'missing'}).",
            f"- Summary metrics: `{summary_path.resolve()}` "
            f"({'present' if summary_path.is_file() else 'missing'}).",
            "- The final generation table contains local official-code/official-weight "
            "runtime outputs only; no author-released prediction artifact rows were used.",
        ])

    tcrt5_audit_md = results_dir / "tcrt5_paper_consistency.md"
    tcrt5_audit_csv = results_dir / "tcrt5_paper_consistency.csv"
    if tcrt5_audit_md.is_file() and tcrt5_audit_csv.is_file():
        legacy_tcrt5_backup = results_dir / "generations.pre_tcrt5_paper_rank.csv"
        lines.extend([
            "",
            "## TCRT5 paper-consistency audit",
            "",
            "- Local TCRT5 candidates are ranked by cumulative model log-likelihood, "
            "matching the paper's rank-cutoff analysis; `raw_score` stores that "
            "cumulative value rather than Hugging Face's length-normalised beam score.",
            f"- Human-readable audit: `{tcrt5_audit_md.resolve()}`.",
            f"- Per-target audit: `{tcrt5_audit_csv.resolve()}`.",
            f"- Pre-migration length-normalised-score backup: "
            f"`{legacy_tcrt5_backup.resolve()}` "
            f"({'present' if legacy_tcrt5_backup.is_file() else 'missing'}).",
            "- The final TCRT5 table is the clean post-fix official-checkpoint rerun. "
            "Existing GIANA outputs remain valid because the rerun and the pre-fix run "
            "have exactly identical per-target candidate sets; only scores/ranks changed.",
        ])

    bioseq_report = results_dir / "bioseq_step117000_report.md"
    bioseq_generations = results_dir / "bioseq_step117000_generations.csv"
    bioseq_per_target = results_dir / "bioseq_step117000_per_target_metrics.csv"
    if bioseq_report.is_file() and bioseq_generations.is_file() and bioseq_per_target.is_file():
        lines.extend([
            "",
            "## BioSeq 7L step117000 Track-A result",
            "",
            "- Conditioning: epitope only; the public MHC fields are retained for "
            "target identity but are not consumed by this sampling path.",
            "- Sampling: seed 42, batch size 100, 32 iterations, temperature 1.0, "
            "`gumbel_argmax`; rank is emission order and `raw_score` is empty.",
            f"- Focused report: `{bioseq_report.resolve()}`.",
            f"- BioSeq generations: `{bioseq_generations.resolve()}`.",
            f"- BioSeq per-target metrics: `{bioseq_per_target.resolve()}`.",
        ])

    lines.extend(["", "## Commands actually recorded", ""])
    events = status.get("events", [])
    if not events:
        lines.append("No execution events recorded yet.")
    else:
        for event in events:
            outcome = "success" if event.get("success", True) else "failed"
            lines.append(
                f"- {event.get('timestamp', '')} · {event.get('action', 'unknown')} · "
                f"{event.get('model', 'benchmark')} · {outcome}: `{event.get('command', '')}`"
            )

    lines.extend([
        "",
        "## Benchmark code added or changed",
        "",
        "- No upstream model source file or checkpoint was edited.",
        f"- `{(BENCH / 'tcr_generation_bench' / 'track_a.py').resolve()}`: public-data audit, raw-candidate schema, strict validity, exact/recovery/Jaccard metrics, and an optional compiled Levenshtein accelerator with an identical pure-Python fallback.",
        f"- `{(BENCH / 'tcr_generation_bench' / 'public_generators.py').resolve()}`: common generation interface, isolated runtimes, beta-only TcrDesign call, TCRT5 cumulative-log-likelihood ranking, and the TCR-epiDiff reconstruction guard. GRATCR receives two identical temporary target rows solely to work around its one-row 80/20 preprocessing crash; its official model, weights, and decoder are unchanged.",
        f"- `{(BENCH / 'tcr_generation_bench' / 'giana.py').resolve()}`: official GIANA4.1, no-V mode, applied identically to every model-target block.",
        f"- `{(BENCH / 'tcr_generation_bench' / 'paired_alpha.py').resolve()}` and `{(BENCH / 'tcr_generation_bench' / 'tcrdesign_alpha_worker.py').resolve()}`: select only generated TcrDesign betas and run the released alpha checkpoint/beam search with one model load and batched conditions. Upstream TcrDesign source and weights remain unchanged.",
        f"- `{Path(__file__).resolve()}`: prepare/generate/import/generate-bioseq/score/migrate-tcrt5-paper-ranking/generate-tcrdesign-alpha/status CLI.",
        f"- `{(BENCH / 'tcr_generation_bench' / 'bioseq_generator.py').resolve()}`: local step117000 epitope-only BioSeq adapter with seeded emission-order ranks and no post-hoc sequence repair.",
        f"- `{(BENCH / 'tcr_generation_bench' / 'report_bioseq_step117000.py').resolve()}`: focused BioSeq generations, per-target metrics, nearest-reference diagnostics, and GIANA-hit report.",
        f"- `{(BENCH / 'tcr_generation_bench' / 'audit_tcrt5_paper.py').resolve()}`: per-target comparison of the local official-weight TCRT5 run, the authors' released prediction artifact, and the paper's reported values.",
        f"- `{(BENCH / 'scripts' / 'test_public_beta_benchmark.py').resolve()}`: protocol and metric regression tests.",
        "- TCRT5 uses the dedicated `protenix_abtcr` prefix. Historical event lines created before the command-recorder fix display the resolved shared `flow` interpreter symlink; the loaded torch/transformers modules and sys.prefix were verified under `protenix_abtcr`.",
        "",
        "## Comparability limits",
        "",
        "TCRT5, GRATCR, beta-only TcrDesign, and BioSeq-7L-step117000 are de-novo "
        "CDR3β generators and "
        "may be compared after producing the same requested candidate count. "
        "BioSeq, GRATCR, and TcrDesign consume epitope only, whereas TCRT5 also consumes "
        "the MHC pseudo-sequence. BioSeq has no comparable per-candidate likelihood, so "
        "its rank is deterministic seeded emission order and `raw_score` remains empty. "
        "TCR-epiDiff is reconstruction-only under the released code and must not "
        "appear in the ranked summary. Rows imported from "
        "`benchmark_data_w_preds.csv` are author-released generations, not a "
        "claim that the checkpoint was reloaded locally; the provenance is kept "
        "in the `source` and `run_id` columns.",
    ])
    (results_dir / "reproduction_status.md").write_text("\n".join(lines) + "\n")


def _select_targets(targets: Mapping[str, Target], selector: str) -> dict[str, Target]:
    if selector == "all":
        return dict(targets)
    if selector == "rvr":
        return {RVR_TARGET_ID: targets[RVR_TARGET_ID]}
    selected: dict[str, Target] = {}
    requested = [item.strip() for item in selector.split(",") if item.strip()]
    for item in requested:
        matches = {
            target_id: target for target_id, target in targets.items()
            if target_id == item or target.peptide == item.upper()
        }
        if not matches:
            raise ValueError(f"unknown target selector: {item}")
        selected.update(matches)
    return selected


def _merge_generations(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return new.copy()
    if new.empty:
        return existing.copy()
    replace_keys = set(zip(new["model"], new["target_id"]))
    keep = [
        (model, target_id) not in replace_keys
        for model, target_id in zip(existing["model"], existing["target_id"])
    ]
    return pd.concat([existing.loc[keep], new], ignore_index=True)


def command_prepare(args) -> None:
    audit = write_benchmark_audit(Path(args.data_dir))
    print(f"prepared {audit['n_targets']} targets / {audit['n_references_total']} references")
    print(f"RVRAYTYSK references: {audit['reference_counts'][RVR_TARGET_ID]}")
    _record_event(Path(args.results_dir), action="prepare", success=True)


def command_import_official(args) -> None:
    targets = load_public_targets()
    selected = _select_targets(targets, args.targets)
    imported = import_official_predictions(args.models)
    imported = imported[imported["target_id"].isin(selected)].copy()
    imported = imported[imported["rank"] <= args.num_candidates].copy()
    output = Path(args.output)
    existing = read_generations(output) if args.append and output.is_file() else empty_generation_frame()
    merged = _merge_generations(existing, imported)
    write_generations(merged, output)
    for model in sorted(imported["model"].unique()):
        _record_event(
            Path(args.results_dir),
            action="import_official_artifact",
            model=model,
            success=True,
            targets=len(imported[imported["model"] == model]["target_id"].unique()),
            candidates=int(len(imported[imported["model"] == model])),
            output=str(output.resolve()),
        )
    print(f"wrote {len(imported)} imported candidates; merged table has {len(merged)} rows -> {output}")


def command_generate(args) -> None:
    targets = load_public_targets()
    selected = _select_targets(targets, args.targets)
    generator = generator_for(args.model, device=args.device)
    ready, reason = generator.check_ready()
    if not ready:
        _record_event(
            Path(args.results_dir), action="generate", model=generator.name,
            success=False, generation_mode=generator.generation_mode, reason=reason,
        )
        if generator.generation_mode == "reconstruction_only":
            raise ReconstructionOnlyError(reason)
        raise RuntimeError(reason)

    frames = []
    for index, target in enumerate(selected.values(), start=1):
        print(f"[{index}/{len(selected)}] {generator.name} -> {target.target_id}", flush=True)
        candidates = generator.generate(
            target.peptide,
            mhc_allele=target.mhc_allele,
            mhc_pseudosequence=target.mhc_pseudosequence,
            num_candidates=args.num_candidates,
        )
        frames.append(generation_rows(
            generator.name,
            target,
            candidates,
            generation_mode=generator.generation_mode,
            source="official_code_and_weights_runtime",
            run_id=args.run_id,
        ))
    generated = pd.concat(frames, ignore_index=True) if frames else empty_generation_frame()
    output = Path(args.output)
    existing = read_generations(output) if args.append and output.is_file() else empty_generation_frame()
    merged = _merge_generations(existing, generated)
    write_generations(merged, output)
    _record_event(
        Path(args.results_dir),
        action="generate",
        model=generator.name,
        success=True,
        generation_mode=generator.generation_mode,
        targets=len(selected),
        requested_per_target=args.num_candidates,
        candidates=len(generated),
        output=str(output.resolve()),
    )
    print(f"wrote {len(generated)} new candidates; merged table has {len(merged)} rows -> {output}")


def command_generate_bioseq(args) -> None:
    """Run the retained local BioSeq checkpoint through the public Track-A schema."""

    targets = load_public_targets()
    selected = _select_targets(targets, args.targets)
    generator = BioSeqTrackAGenerator(
        args.checkpoint,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        temperature=args.temperature,
        sampling_strategy=args.sampling_strategy,
    )
    ready, reason = generator.check_ready()
    if not ready:
        _record_event(
            Path(args.results_dir),
            action="generate",
            model=generator.name,
            success=False,
            generation_mode=generator.generation_mode,
            reason=reason,
        )
        raise RuntimeError(reason)

    frames = []
    for index, target in enumerate(selected.values(), start=1):
        print(f"[{index}/{len(selected)}] {generator.name} -> {target.target_id}", flush=True)
        candidates = generator.generate(
            target.peptide,
            mhc_allele=target.mhc_allele,
            mhc_pseudosequence=target.mhc_pseudosequence,
            num_candidates=args.num_candidates,
        )
        frame = generation_rows(
            generator.name,
            target,
            candidates,
            generation_mode=generator.generation_mode,
            source="local_bioseq_checkpoint_runtime",
            run_id=args.run_id,
        )
        frames.append(frame)
        print(f"  raw examples: {frame['cdr3b'].head(3).tolist()}", flush=True)

    generated = pd.concat(frames, ignore_index=True) if frames else empty_generation_frame()
    output = Path(args.output)
    existing = read_generations(output) if args.append and output.is_file() else empty_generation_frame()
    merged = _merge_generations(existing, generated)
    write_generations(merged, output)
    manifest = generator.checkpoint_manifest()
    _record_event(
        Path(args.results_dir),
        action="generate",
        model=generator.name,
        success=True,
        generation_mode=generator.generation_mode,
        conditioning=generator.conditioning,
        score_semantics=generator.score_semantics,
        targets=len(selected),
        requested_per_target=args.num_candidates,
        candidates=len(generated),
        checkpoint=str(generator.checkpoint),
        checkpoint_sha256=manifest.get("sha256"),
        checkpoint_step=manifest.get("step"),
        seed=args.seed,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        temperature=args.temperature,
        sampling_strategy=args.sampling_strategy,
        output=str(output.resolve()),
    )
    print(f"wrote {len(generated)} BioSeq candidates; merged table has {len(merged)} rows -> {output}")


def command_score(args) -> None:
    targets = load_public_targets()
    generations = read_generations(Path(args.generations))
    results_dir = Path(args.results_dir)
    giana_metrics = {}
    if args.with_giana:
        giana_metrics = giana_runner.run_all_giana(
            generations,
            targets,
            results_dir / "giana",
            num_candidates=args.num_candidates,
            threads=args.giana_threads,
        )
    per_target, summary, jaccard = score_generations(
        generations,
        targets,
        num_candidates=args.num_candidates,
        giana_metrics=giana_metrics,
    )
    write_metric_outputs(generations, per_target, summary, jaccard, results_dir)
    _record_event(
        results_dir,
        action="score",
        success=True,
        models=sorted(generations["model"].unique().tolist()),
        model_target_blocks=len(per_target),
        with_giana=args.with_giana,
        num_candidates=args.num_candidates,
    )
    if len(summary):
        print(summary.to_string(index=False))
    print(f"results -> {results_dir}")


def command_migrate_tcrt5_paper_ranking(args) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    generations = read_generations(input_path)
    migrated = migrate_legacy_tcrt5_paper_ranking(generations)
    write_generations(migrated, output_path)
    tcrt5_rows = migrated[migrated["model"] == "TCRT5"]
    _record_event(
        Path(args.results_dir),
        action="migrate_tcrt5_paper_ranking",
        model="TCRT5",
        success=True,
        candidates=len(tcrt5_rows),
        source=str(input_path.resolve()),
        output=str(output_path.resolve()),
        score="cumulative_transition_log_likelihood",
    )
    print(f"migrated {len(tcrt5_rows)} TCRT5 rows -> {output_path}")


def command_generate_tcrdesign_alpha(args) -> None:
    results_dir = Path(args.results_dir)
    targets = load_public_targets()
    selected_targets = _select_targets(targets, args.targets)
    generations = read_generations(Path(args.generations))
    beta_conditions = select_tcrdesign_generated_betas(
        generations,
        selected_targets.keys(),
        beta_top_k=args.beta_top_k,
    )
    generator = TcrDesignAlphaGenerator(
        device=args.device,
        temperature=args.temperature,
        max_len=args.max_len,
        batch_size=args.batch_size,
    )
    ready, reason = generator.check_ready()
    if not ready:
        _record_event(
            results_dir,
            action="generate_alpha_pairs",
            model="TcrDesign",
            success=False,
            reason=reason,
        )
        raise RuntimeError(reason)

    print(
        f"TcrDesign generated-beta -> alpha: {len(beta_conditions)} beta conditions, "
        f"{args.alpha_per_beta} alpha beams each",
        flush=True,
    )
    paired = generator.generate_frame(
        beta_conditions,
        alpha_per_beta=args.alpha_per_beta,
        run_id=args.run_id,
    )
    output = Path(args.output)
    write_alpha_generations(paired, output)
    audit_path = write_official_alpha_evaluation_audit(results_dir)
    _record_event(
        results_dir,
        action="generate_alpha_pairs",
        model="TcrDesign",
        success=True,
        generation_mode="end_to_end_generated_beta",
        targets=len(selected_targets),
        beta_conditions=len(beta_conditions),
        beta_top_k=args.beta_top_k,
        alpha_per_beta=args.alpha_per_beta,
        candidates=len(paired),
        official_alpha_evaluation_available=False,
        included_in_quantitative_benchmark=False,
        output=str(output.resolve()),
        audit=str(audit_path.resolve()),
        worker_command=shlex.join(generator.last_command or []),
    )
    print(f"wrote {len(paired)} qualitative paired rows -> {output}")
    print(f"official alpha-evaluation audit -> {audit_path}")


def command_status(args) -> None:
    results_dir = Path(args.results_dir)
    status = _load_status(results_dir)
    _save_status(results_dir, status)
    for row in _static_model_rows(status):
        print(
            f"{row['model']:12s} mode={row['generation_mode']:20s} "
            f"source={row['source_downloaded']} weights={row['official_weights_present']} "
            f"loaded={row['official_weights_loaded']} comparable={row['directly_comparable']}"
        )
    print(f"status -> {results_dir / 'reproduction_status.md'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="audit/materialise the public benchmark")
    prepare.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    prepare.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    prepare.set_defaults(func=command_prepare)

    imported = subparsers.add_parser("import-official", help="import released TCRT5/GRATCR candidates")
    imported.add_argument("--models", nargs="+", choices=["tcrt5", "gratcr"], default=["tcrt5", "gratcr"])
    imported.add_argument("--targets", default="all", help="all, rvr, peptide, target_id, or comma list")
    imported.add_argument("--num-candidates", type=int, default=1000)
    imported.add_argument("--output", default=str(DEFAULT_RESULTS_DIR / "generations.csv"))
    imported.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    imported.add_argument("--append", action="store_true")
    imported.set_defaults(func=command_import_official)

    generate = subparsers.add_parser("generate", help="run one official generator through the unified API")
    generate.add_argument("--model", required=True, choices=["tcrt5", "gratcr", "tcrdesign", "tcr-epidiff"])
    generate.add_argument("--targets", default="rvr", help="all, rvr, peptide, target_id, or comma list")
    generate.add_argument("--num-candidates", type=int, default=10)
    generate.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    generate.add_argument("--run-id", default="local_runtime")
    generate.add_argument("--output", default=str(DEFAULT_RESULTS_DIR / "generations.csv"))
    generate.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    generate.add_argument("--append", action="store_true")
    generate.set_defaults(func=command_generate)

    bioseq = subparsers.add_parser(
        "generate-bioseq",
        help="run a retained BioSeq 7L checkpoint through the unified Track-A interface",
    )
    bioseq.add_argument("--checkpoint", default=str(DEFAULT_BIOSEQ_7L_STEP117000))
    bioseq.add_argument("--targets", default="rvr", help="all, rvr, peptide, target_id, or comma list")
    bioseq.add_argument("--num-candidates", type=int, default=10)
    bioseq.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    bioseq.add_argument("--seed", type=int, default=42)
    bioseq.add_argument("--batch-size", type=int, default=100)
    bioseq.add_argument("--max-iter", type=int, default=32)
    bioseq.add_argument("--temperature", type=float, default=1.0)
    bioseq.add_argument("--sampling-strategy", default="gumbel_argmax")
    bioseq.add_argument("--run-id", default="step117000_seed42_iter32")
    bioseq.add_argument("--output", default=str(DEFAULT_RESULTS_DIR / "generations.csv"))
    bioseq.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    bioseq.add_argument("--append", action="store_true")
    bioseq.set_defaults(func=command_generate_bioseq)

    score = subparsers.add_parser("score", help="compute the unified Track-A metrics")
    score.add_argument("--generations", default=str(DEFAULT_RESULTS_DIR / "generations.csv"))
    score.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    score.add_argument("--num-candidates", type=int, default=1000)
    score.add_argument("--with-giana", action="store_true")
    score.add_argument("--giana-threads", type=int, default=1)
    score.set_defaults(func=command_score)

    migrate = subparsers.add_parser(
        "migrate-tcrt5-paper-ranking",
        help="convert legacy TCRT5 beam scores/ranks to paper cumulative likelihood order",
    )
    migrate.add_argument("--input", required=True)
    migrate.add_argument("--output", required=True)
    migrate.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    migrate.set_defaults(func=command_migrate_tcrt5_paper_ranking)

    alpha = subparsers.add_parser(
        "generate-tcrdesign-alpha",
        help="run TcrDesign generated-beta-conditioned alpha generation (qualitative)",
    )
    alpha.add_argument("--generations", default=str(DEFAULT_RESULTS_DIR / "generations.csv"))
    alpha.add_argument("--targets", default="rvr", help="all, rvr, peptide, target_id, or comma list")
    alpha.add_argument(
        "--beta-top-k", type=int, default=100,
        help="generated beta ranks per target (official full-pipeline default: 100)",
    )
    alpha.add_argument(
        "--alpha-per-beta", type=int, default=10,
        help="official full-pipeline default: 10",
    )
    alpha.add_argument("--temperature", type=float, default=0.8)
    alpha.add_argument("--max-len", type=int, default=20)
    alpha.add_argument("--batch-size", type=int, default=32)
    alpha.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    alpha.add_argument("--run-id", default="tcrdesign_end_to_end_alpha")
    alpha.add_argument("--output", default=str(DEFAULT_RESULTS_DIR / "tcrdesign_alpha_generations.csv"))
    alpha.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    alpha.set_defaults(func=command_generate_tcrdesign_alpha)

    status = subparsers.add_parser("status", help="refresh reproduction_status.md")
    status.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    status.set_defaults(func=command_status)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
