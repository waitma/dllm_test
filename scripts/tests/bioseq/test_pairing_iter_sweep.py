"""CPU checks for pairing step-sweep provenance, counts and job configuration.

Run: python -m pytest scripts/tests/bioseq/test_pairing_iter_sweep.py -q
Uses synthetic CSV/JSON fixtures, without loading pretrained models or a GPU.
"""

import csv
import json
from pathlib import Path

import pytest
import yaml

from downstream.grammar.ab_features import file_sha256
from scripts.downstream import pairing_iter_sweep as sweep
from scripts.downstream.pairing_leakage_diagnostic import audit


def make_run(path, steps=8):
    path.mkdir()
    rows = []
    for i, heavy in enumerate(("ACDE", "FGHI")):
        for variant in range(2):
            rows.append({"h_sequence": heavy, "raw_l_sequence": "DIQ", "gen_l_sequence": "DIQ",
                         "_input_row_idx": str(i), "variant_idx": str(variant),
                         "target_light_length": "3", "ref_light_length": "3", "generated_light_length": "3",
                         "light_prompt_tokens": "0", "cfg_scale": "0.0",
                         "generation_protocol_version": "5", "light_length_mode": "reference"})
    with (path / "holdout500_n8.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {key: None for key in sweep.FIXED_KEYS}
    manifest.update(dataset_len=2, num_seqs=2, max_iter=steps, checkpoint_sha256="weights",
                    input_csv_sha256="data", source_sha256={"source.py": "sha"}, cfg_scale=0,
                    light_prompt_tokens=0, protocol_version=5, light_length_mode="reference")
    metrics = {key: 1.0 for key in sweep.SCORE_KEYS}
    metrics.update(total_pairs=4, evaluation_protocol=dict(manifest),
                   detailed_results=[{"index": i, "gen_valid": True,
                                      **{f"{k}_match": True for k in ("chain", "v_gene", "j_gene", "v_gene_family", "j_gene_family")}}
                                     for i in range(4)],
                   generation_audit=audit(path / "holdout500_n8.csv"))
    for name, value in (("holdout500_n8_manifest.json", manifest), ("holdout500_metrics.json", metrics)):
        (path / name).write_text(json.dumps(value))
    return manifest, rows, metrics


@pytest.mark.parametrize("steps", sweep.STEPS)
def test_all_step_budgets_accept_complete_results(tmp_path, steps):
    path = tmp_path / "run"
    manifest, rows, _ = make_run(path, steps)
    result = sweep.validate_result(path, {**manifest, "max_iter": 124}, rows, steps)
    assert result["max_iter"] == steps and result["n_candidates"] == 4
    assert result["metrics_sha256"] == file_sha256(path / "holdout500_metrics.json")


@pytest.mark.parametrize("corruption", ("checkpoint", "source", "prompt", "steps", "count", "denominator", "nan", "order"))
def test_rejects_protocol_drift_and_incomplete_metrics(tmp_path, corruption):
    path = tmp_path / "run"
    manifest, rows, metrics = make_run(path)
    reference = json.loads(json.dumps(manifest))
    if corruption == "checkpoint":
        manifest["checkpoint_sha256"] = "other"
    elif corruption == "source":
        manifest["source_sha256"]["source.py"] = "other"
    elif corruption == "prompt":
        manifest["light_prompt_tokens"] = 3
    elif corruption == "steps":
        metrics["evaluation_protocol"]["max_iter"] = 124
    elif corruption == "count":
        metrics["detailed_results"].pop()
    elif corruption == "denominator":
        metrics["overall_chain_match_rate"] = .5
    elif corruption == "nan":
        metrics["gen_immunomatch_mean"] = float("nan")
    else:
        metrics["detailed_results"][1]["index"] = 0
    (path / "holdout500_n8_manifest.json").write_text(json.dumps(manifest))
    (path / "holdout500_metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(ValueError):
        sweep.validate_result(path, reference, rows, 8)


def test_summary_never_calls_partial_or_invalid_matrix_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "OUT", tmp_path)
    monkeypatch.setattr(sweep, "STEPS", (8, 16))
    reference, _, _ = make_run(tmp_path / "reference", 124)
    make_run(tmp_path / "iter8", 8)
    plan = {"pinned_files": {}, "reference_dir": str(tmp_path / "reference"),
            "reference_manifest": reference, "reference_result": {}, "paper_reference": {}}
    (tmp_path / "experiment_manifest.json").write_text(json.dumps(plan))
    partial = sweep.summarize()
    assert not partial["complete"] and partial["pending"] == [16] and list(partial["results"]) == ["8"]
    make_run(tmp_path / "iter16", 16)
    assert sweep.summarize()["complete"]
    metrics_path = tmp_path / "iter16/holdout500_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["total_pairs"] = 3
    metrics_path.write_text(json.dumps(metrics))
    result = sweep.summarize()
    assert not result["complete"] and "16" in result["errors"]


def test_pinned_file_change_is_rejected(tmp_path):
    path = tmp_path / "source.py"
    path.write_text("original")
    plan = {"pinned_files": {str(path): file_sha256(path)}}
    sweep.verify_inputs(plan)
    path.write_text("modified")
    with pytest.raises(ValueError, match="changed"):
        sweep.verify_inputs(plan)


def test_six_jobs_are_independent_nonpreemptible_single_gpu():
    names = set()
    for steps in sweep.STEPS:
        path = sweep.ROOT / f"eval_jobs/eval_{sweep.TAG}_iter{steps}.yml"
        job = yaml.safe_load(path.read_text())
        assert job["Preemptible"] is False
        assert job["ResourceQueueName"] == "c20250601"
        assert job["TaskRoleSpecs"] == [{"RoleName": "worker", "RoleReplicas": 1, "Flavor": "ml.pni2.3xlarge"}]
        assert f"pairing_iter_sweep.py run --steps {steps}\n" in job["Entrypoint"]
        assert "conda/envs/pllm" in job["Entrypoint"] and "conda/envs/protenix_abtcr" in job["Entrypoint"]
        names.add(job["TaskName"])
    assert len(names) == len(sweep.STEPS)
