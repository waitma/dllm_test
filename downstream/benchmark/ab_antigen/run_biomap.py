"""Run the unchanged official PALM BioMap training/test entry on a Volc GPU.

Run in the isolated PALM environment (through eval_jobs YAML):
  python -m downstream.benchmark.ab_antigen.run_biomap --run-dir /abs/run

The prepared directory contains config.json, source.lock.json, assets.json, and
Code/ path aliases. This wrapper verifies provenance and collects official scores;
it does not implement a model, split, optimizer, or scoring formula.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import traceback

from .official import SOURCE_ROOT, digest, verify_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    report_path = run / "run_report.json"
    if report_path.exists():
        raise FileExistsError("Refusing to overwrite an existing run report")
    report = {"started_at": datetime.now(timezone.utc).isoformat(),
              "status": "verifying", "scores_produced": False,
              "protocol": "official BioMap seed=3; original CSV; original training and scoring",
              "limitations": ["Original CSV H/L column inconsistency retained",
                              "Single released default seed, not all paper repeats",
                              "Sample-random split, not unseen-antigen evaluation",
                              "Delta G label, not converted pKD"]}

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    try:
        import pandas as pd
        import torch

        spec = json.loads((run / "source.lock.json").read_text())
        report["source"] = verify_source(spec)
        source = SOURCE_ROOT / spec["id"]
        assets = json.loads((run / "assets.json").read_text())
        for asset in assets["files"].values():
            assert digest(Path(asset["path"])) == asset["sha256"], asset["path"]
        config_path = run / "config.json"
        config = json.loads(config_path.read_text())
        assert Path(config["data_loader"]["args"]["data_dir"]).resolve() == source / "ProcessedData/BioMap/biomap.csv"
        for key, member in [("heavy_dir", "Model_Zenodo/Heavy_roformer/pytorch_model.bin"),
                            ("light_dir", "Model_Zenodo/Light_roformer/pytorch_model.bin"),
                            ("antigen_dir", "esm2/pytorch_model.bin")]:
            assert Path(config["arch"]["args"][key]).resolve() == Path(assets["files"][member]["path"]).parent
        original = json.loads((source / "Code/config/common/bert_finetuning_er_common_biomap.json").read_text())
        # Only filesystem mappings may differ from the released experiment.
        normalized = json.loads(json.dumps(config))
        for group, key in [("data_loader", "data_dir"), ("data_loader", "antibody_tokenizer_dir"),
                           ("arch", "heavy_dir"), ("arch", "light_dir"), ("arch", "antigen_dir")]:
            normalized[group]["args"][key] = original[group]["args"][key]
        normalized["trainer"]["save_dir"] = original["trainer"]["save_dir"]
        assert normalized == original, "Non-path protocol change detected"
        assert torch.cuda.is_available(), "GPU execution must use a Volc job"
        assert not (run / "results").exists(), "Refusing to reuse a result directory"
        command = [sys.executable, str(source / "Code/bert_finetuning_er_main.py"),
                   "--config", str(config_path)]
        report.update(
            status="running", command=command, cwd=str(run / "Code"),
            source_lock_sha256=digest(run / "source.lock.json"),
            assets_manifest_sha256=digest(run / "assets.json"),
            config_sha256=digest(config_path),
            environment={"python": platform.python_version(), "gpu": torch.cuda.get_device_name(0),
                         **{p: importlib.metadata.version(p) for p in
                            ["torch", "numpy", "pandas", "scipy", "scikit-learn",
                             "transformers", "tokenizers", "huggingface-hub"]}},
        )
        save()
        with (run / "official.log").open("w") as log:
            result = subprocess.run(command, cwd=run / "Code", stdout=log,
                                    stderr=subprocess.STDOUT, timeout=21000)
        report["returncode"] = result.returncode
        assert result.returncode == 0, "Official entry failed; see official.log"
        predictions = list((run / "results").glob("datasplit/*/*/test_result.csv"))
        assert len(predictions) == 1, predictions
        frame = pd.read_csv(predictions[0])
        assert len(frame) == 170
        # Call the author's functions directly; no local metric implementation.
        sys.path.insert(0, str(source / "Code"))
        from model import metric
        scores = {name: float(getattr(metric, name)(frame.y_pred.to_numpy()[:, None],
                                                   frame.y_true.to_numpy()[:, None]))
                  for name in config["metrics"]}
        assert all(math.isfinite(v) for v in scores.values())
        checkpoints = list((run / "results").glob("checkpoints/*/*/model_best.pth"))
        assert len(checkpoints) == 1
        checkpoint = torch.load(checkpoints[0], map_location="cpu")
        verify_source(spec)
        report.update(status="completed", scores_produced=True, metrics=scores,
                      test_rows=len(frame), predictions=str(predictions[0]),
                      predictions_sha256=digest(predictions[0]),
                      best_checkpoint=str(checkpoints[0]), best_epoch=checkpoint["epoch"],
                      best_validation_pearson=float(checkpoint["monitor_best"]))
        (run / "metrics.json").write_text(json.dumps({
            "provenance": "R_official_code_single_seed", "metrics": scores,
            "n_test": len(frame), "best_epoch": checkpoint["epoch"],
            "source_commit": spec["commit"], "predictions_sha256": digest(predictions[0]),
            "limitations": report["limitations"],
        }, indent=2) + "\n")
    except Exception as error:
        report.update(status="failed", scores_produced=False,
                      error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        if (run / "official.log").exists():
            report["log_sha256"] = digest(run / "official.log")
        save()
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
