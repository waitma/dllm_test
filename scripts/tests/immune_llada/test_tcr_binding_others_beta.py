"""Test the same-others-row beta control: pytest -q this file in pllm.

Uses synthetic data and CPU only; no GPU or benchmark copies are needed.
"""

import pandas as pd
import pytest

from downstream.benchmark.tcr_binding import run_retrained_ours as runner


@pytest.mark.parametrize("track,data_track,mode", [
    ("cdr3b", "cdr3b", "cdr3b"),
    ("cdr3ab", "cdr3ab", "cdr3ab"),
    ("others_cdr3b", "cdr3ab", "cdr3b"),
])
def test_data_selection_is_distinct_from_input_mode(track, data_track, mode):
    assert runner.data_track_for_track(track) == data_track
    assert runner.input_mode_for_track(track) == mode
    assert ("CDR3A" in runner.input_columns(track)) == (mode == "cdr3ab")


def test_control_preserves_exact_source_rows_and_provenance(monkeypatch):
    frame = pd.DataFrame({
        "Epitope": ["GILGFVFTL", "SLLMWITQV"],
        "CDR3B": ["CASSQETQYF", "CASSLGQAYF"],
        "CDR3A": ["CAVRDSNYQLIW", "CAVGGGSNYQLIW"],
        "Affinity": [0, 1], "MHC": ["ignored", "ignored"],
    })
    calls = []

    def loader(**kwargs):
        calls.append(kwargs)
        return frame.copy(), b"unchanged official bytes", {"data_track": kwargs["track"]}

    monkeypatch.setattr(runner, "load_official_train", loader)
    monkeypatch.setattr(runner, "load_official_test", loader)
    frames, provenance = runner.load_protocol("AS", "others_cdr3b")
    assert all(call["track"] == "cdr3ab" for call in calls)
    assert len(calls) == 20
    assert provenance["data_track"] == "cdr3ab"
    assert provenance["input_mode"] == "cdr3b"
    for splits in frames.values():
        for loaded in splits.values():
            pd.testing.assert_frame_equal(loaded, frame)
    keys = runner.build_pair_universe(frames, "others_cdr3b")
    assert all(len(key) == 2 for key in keys)
    changed = frame.copy()
    changed["CDR3A"] = "different-alpha"
    changed["Affinity"] = 1 - changed["Affinity"]
    assert runner._unique_pairs(changed, "others_cdr3b") == set(keys)


def test_output_roots_and_rankings_do_not_mix_datasets(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "OUT_ROOT", tmp_path)
    roots = {runner._experiment_root("tag", "AS", track)
             for track in ("cdr3b", "cdr3ab", "others_cdr3b")}
    assert len(roots) == 3
    summary = runner.summarize("control", "AS", feature_identity="new",
                               track="others_cdr3b")
    assert summary["data_track"] == "cdr3ab"
    assert summary["input_columns"] == ["Epitope", "CDR3B"]
    runner.write_rankings(summary, "AS")  # Must not require/read the old leaderboard.
    assert "no larger beta-only dataset ranking" in capsys.readouterr().out


def test_unknown_track_is_rejected():
    with pytest.raises(ValueError, match="unsupported input track"):
        runner.data_track_for_track("typo")
