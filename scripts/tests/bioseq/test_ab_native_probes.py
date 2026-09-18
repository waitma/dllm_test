"""CPU checks: python -m pytest scripts/tests/bioseq/test_ab_native_probes.py -q."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("joblib", reason="Native probe runtime uses the existing protenix_abtcr environment")
pytest.importorskip("sklearn")

from dllm.pipelines.immune_llada.data import GrammarBioSeqCollator, GrammarTokenizer
from downstream.grammar.ab_features import embed_native_pairs
from downstream.grammar.ab_probes import ALL_COLS, FOLD_COL, SCORE_COLS, m396_splits, run_gdp, run_m396, run_specificity


def test_native_features_ab_grammar_full_sequences_and_global_pool():
    tokenizer = GrammarTokenizer()
    calls = []

    class Stub(torch.nn.Module):
        def last_hidden_state(self, **batch):
            calls.append(batch)
            assert not torch.is_grad_enabled()
            ids = batch["input_ids"]
            return torch.stack((ids.float(), torch.ones_like(ids).float()), dim=-1)

    pairs = [("A" * 245, "C" * 213), ("ACDE", "DIQMTQ")]
    features = embed_native_pairs(Stub(), GrammarBioSeqCollator(tokenizer, max_sequence_length=1024),
                                 pairs, device="cpu", batch_size=2)
    batch = calls[0]
    assert batch["residue_mask"].sum(1).tolist() == [458, 10]
    assert (batch["attention_mask"][1] == 0).any()
    assert batch["input_ids"].eq(tokenizer.special_id("<ab>")).sum().item() == 2
    assert not batch["input_ids"].eq(tokenizer.special_id("<tcr>")).any()
    assert not batch["input_ids"].eq(tokenizer.mask_token_id).any()
    assert features.shape == (2, 2)
    assert torch.equal(features[:, 1], torch.ones(2))
    for i in range(2):
        mask = batch["residue_mask"][i].bool()
        assert features[i, 0] == batch["input_ids"][i, mask].float().mean()


def test_native_long_input_fails_not_silently_truncated():
    with pytest.raises(ValueError, match="never grammar-truncated"):
        embed_native_pairs(torch.nn.Identity(), GrammarBioSeqCollator(GrammarTokenizer(), max_sequence_length=20),
                           [("A" * 245, "C" * 213)], device="cpu")


@pytest.mark.parametrize("log_epoch_metrics", [False, True])
def test_specificity_one_saved_head_per_fold_and_recomputable_predictions(tmp_path, log_epoch_metrics):
    df = pd.DataFrame({"name": [f"sample{i}" for i in range(30)], "label": np.tile([0, 1, 2], 10)})
    features = torch.randn(30, 4, generator=torch.Generator().manual_seed(4))
    folds = [(np.setdiff1d(np.arange(30), np.arange(k * 6, (k + 1) * 6)),
              np.arange(k * 6, (k + 1) * 6)) for k in range(5)]
    args = SimpleNamespace(out_dir=tmp_path, epochs=2, head_batch_size=4, lr=5e-5, seed=42, device="cpu",
                           log_epoch_metrics=log_epoch_metrics)
    result = run_specificity(df, features, folds, args)
    assert len(list(tmp_path.glob("fold_*/selected_head.pt"))) == 5
    pred = pd.read_csv(tmp_path / "oof_predictions.csv")
    assert len(pred) == pred.sample_id.nunique() == 30
    assert result["selected_epoch"] == 2
    from downstream.ophiuchus_eval.specificity import _metrics
    values = [_metrics(g.y_true, g.y_pred) for _, g in pred.groupby("fold")]
    for key, value in result["mean"].items():
        assert value == pytest.approx(np.mean([v[key] for v in values]))
    if log_epoch_metrics:
        import json
        curve = json.loads((tmp_path / "epoch_mean.json").read_text())
        assert len(curve) == 2
        assert curve[-1]["lr_next_step"] == 0
        for key, value in result["mean"].items():
            assert curve[-1][key] == pytest.approx(value)
        assert len(list(tmp_path.glob("fold_*/diagnostic_head_epoch_002.pt"))) == 5
    else:
        assert not list(tmp_path.glob("fold_*/epoch_metrics.json"))


def test_specificity_epoch_observation_does_not_change_training(tmp_path):
    from downstream.grammar.ab_probes import fit_selected_head, specificity_epoch_observer
    x = torch.randn(24, 4, generator=torch.Generator().manual_seed(8))
    y = torch.arange(24) % 3
    kwargs = dict(epochs=3, batch_size=4, lr=5e-5, seed=42, device="cpu")
    plain, plain_history = fit_selected_head(x[:18], y[:18], **kwargs)
    observer = specificity_epoch_observer(x[18:], y[18:], tmp_path, final_epoch=3)
    logged, logged_history = fit_selected_head(x[:18], y[:18], **kwargs, epoch_observer=observer)
    assert plain_history == logged_history
    for name, value in plain.state_dict().items():
        assert torch.equal(value, logged.state_dict()[name])


def test_grouped_m396_split_excludes_identical_sequences():
    df = pd.DataFrame({"index": range(2000)})
    groups = np.array([f"seq{i // 2}" for i in range(2000)])
    rows = list(m396_splits(df, groups, 2023))
    assert len(rows) == 12
    for mode, _, tr, te in rows:
        assert not set(tr) & set(te)
        assert len(tr) + len(te) == len(df)
        if mode == "sequence_grouped":
            assert not set(groups[tr]) & set(groups[te])


@pytest.mark.parametrize("existing_history", [False, True])
def test_gdp_reference_only_matches_reference_cv_and_preserves_history(tmp_path, monkeypatch, existing_history):
    import joblib
    from sklearn.linear_model import Ridge
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import GridSearchCV, PredefinedSplit, cross_val_predict
    from sklearn.preprocessing import PowerTransformer
    import downstream.grammar.ab_probes as probes

    rng = np.random.default_rng(9)
    features = rng.normal(size=(50, 4))
    df = pd.DataFrame({FOLD_COL: np.repeat(np.arange(5), 10)})
    col = ALL_COLS[0]
    df[col] = np.exp(features[:, 0] + rng.normal(scale=0.1, size=50))
    df.loc[[3, 27], col] = np.nan
    monkeypatch.setattr(probes, "ALL_COLS", [col])
    monkeypatch.setattr(probes, "LAMBDA_GRID", [0.01, 1.0])
    searches = []

    def track_search(*args, **kwargs):
        search = GridSearchCV(*args, **kwargs)
        searches.append(search)
        return search

    monkeypatch.setattr(probes, "GridSearchCV", track_search)
    folder = tmp_path / col
    historical_path = folder / "nested_oof.npz"
    if existing_history:
        folder.mkdir()
        historical_path.write_bytes(b"historical result must remain unchanged")

    result = run_gdp(df, torch.tensor(features), SimpleNamespace(out_dir=tmp_path))
    assert result["protocol"] == "native_ab_gdp_reference_only_v2"
    assert set(result["properties"][col]) == {"n", "legacy_reference"}
    assert len(searches) == 1  # No extra outer-fold hyperparameter searches.
    assert not list(folder.glob("outer_*"))
    if existing_history:
        assert historical_path.read_bytes() == b"historical result must remain unchanged"
    else:
        assert not historical_path.exists()

    valid = df[col].notna().to_numpy()
    y = df.loc[valid, col].to_numpy(dtype=float)
    fold = df.loc[valid, FOLD_COL].to_numpy(dtype=int)
    transformed = PowerTransformer().fit_transform(y[:, None]).ravel()
    cv = PredefinedSplit(fold)
    expected = GridSearchCV(Ridge(), {"alpha": [0.01, 1.0]}, scoring=make_scorer(probes.rho), cv=cv)
    expected.fit(features[valid], transformed)
    summary = result["properties"][col]["legacy_reference"]
    assert result["properties"][col]["n"] == int(valid.sum())
    assert summary["alpha"] == expected.best_params_["alpha"]
    assert summary["cv_mean_spearman"] == pytest.approx(expected.best_score_)
    saved = joblib.load(folder / "legacy_reference.joblib")
    assert saved["ridge"].alpha == summary["alpha"]
    np.testing.assert_allclose(saved["transformer_all_labels"].transform(y[:, None]).ravel(), transformed)
    artifact = np.load(folder / "legacy_oof.npz")
    np.testing.assert_array_equal(artifact["row_index"], np.flatnonzero(valid))
    np.testing.assert_array_equal(artifact["fold"], fold)
    np.testing.assert_allclose(artifact["y_true"], y)
    np.testing.assert_allclose(artifact["y_transformed"], transformed)
    np.testing.assert_allclose(artifact["prediction"], cross_val_predict(expected.best_estimator_, features[valid], transformed, cv=cv))
    assert np.isfinite(artifact["prediction"]).all()
    assert summary["pooled_oof_spearman"] == pytest.approx(probes.rho(y, artifact["prediction"]))
    fold_scores = [probes.rho(transformed[fold == k], artifact["prediction"][fold == k]) for k in range(5)]
    assert summary["cv_mean_spearman"] == pytest.approx(np.mean(fold_scores))


def test_m396_two_protocols_save_rescorable_predictions(tmp_path, monkeypatch):
    import downstream.grammar.ab_probes as probes
    rng = np.random.default_rng(12)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    sequences = ["".join(alphabet[(i // power) % 20] for power in (1, 20, 400)) for i in range(1000)]
    x = np.repeat(rng.normal(size=(1000, 4)), 2, axis=0)
    y = x @ rng.normal(size=(4, 5))
    df = pd.DataFrame({"h_sequence": np.repeat(sequences, 2), "l_sequence": "DIQMTQ"})
    for i, name in enumerate(SCORE_COLS):
        df[name] = y[:, i]
    monkeypatch.setattr(probes, "SPLITS", (0.01,))
    result = run_m396(df, torch.tensor(np.vstack((np.zeros((1, 4)), x))),
                     SimpleNamespace(out_dir=tmp_path, split_seed=2023, ridge_alpha=0.01))
    assert len(result["splits"]) == 2
    for row in result["splits"]:
        path = tmp_path / f"{row['split_protocol']}_0.01" / "predictions.npz"
        a = np.load(path)
        rescored = probes.score_multitarget(a["prediction"], a["y_true"])
        assert row["spearman_mean"] == pytest.approx(rescored["spearman_mean"])
        if row["split_protocol"] == "sequence_grouped":
            assert row["test_rows_sequence_seen_in_train"] == 0


def test_job_matrix_single_gpu_nonpreemptible_and_explicit_inputs():
    from pathlib import Path
    from scripts.downstream.prepare_ab_native_jobs import build_jobs
    checkpoint = Path("/abs/chosen_checkpoint")
    jobs = build_jobs(checkpoint, "ab_test")
    assert len(jobs) == 11
    assert len([key for key in jobs if key.startswith("pair-")]) == 6
    for key, job in jobs.items():
        assert job["Preemptible"] is False
        assert job["ResourceQueueName"] == "c20250601"
        assert job["TaskRoleSpecs"] == [{"RoleName": "worker", "RoleReplicas": 1, "Flavor": "ml.pni2.3xlarge"}]
        assert str(checkpoint) in job["Entrypoint"]
        if key.startswith("pair-"):
            assert "--light-length-mode reference" in job["Entrypoint"]
            assert "--cfg-scale" in job["Entrypoint"]
        elif key == "cdr-kong":
            assert "/sabdab_kong" in job["Entrypoint"]
            assert "--max-iter 1" in job["Entrypoint"]
