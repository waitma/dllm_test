"""Run: python -m pytest scripts/tests/bioseq/test_ab_fv_input.py -q."""
import json

import pytest

from downstream.grammar.ab_fv_input import MANIFEST, adapt_pairs


def test_m396_preserves_mutations_inside_fv_and_rejects_outside():
    rules = json.loads(MANIFEST.read_text())["m396"]
    h, l = rules["h"]["sequence"], rules["l"]["sequence"]
    mutant = ("A" if h[10] != "A" else "C")
    changed = h[:10] + mutant + h[11:]
    pairs, audit = adapt_pairs("m396", [(h, l), (changed, l)])
    assert tuple(map(len, pairs[0])) == (118, 108)
    assert pairs[1][0][10] == mutant
    assert audit["changed_pairs"] == 2
    with pytest.raises(ValueError, match="outside audited Fv"):
        adapt_pairs("m396", [(h[:-1] + "A", l)])


def test_unknown_overflow_rejected_and_short_inputs_unchanged():
    pairs = [("ACDE", "FGHI")]
    assert adapt_pairs("specificity", pairs)[0] == pairs
    with pytest.raises(ValueError, match="Unverified oversized"):
        adapt_pairs("specificity", [("A" * 200, "FGHI")])


def test_specificity_boundaries_are_exact_subsequences():
    rules = json.loads(MANIFEST.read_text())["overflow"]
    for rule in rules.values():
        pair = (rule["sequence"], "C") if rule["role"] == "h" else ("A", rule["sequence"])
        result, _ = adapt_pairs("specificity", [pair])
        value = result[0][0 if rule["role"] == "h" else 1]
        assert value == rule["sequence"][rule["start"]:rule["end"]]
        assert len(value) == rule["fv_length"]


def test_monitor_parses_real_yaml_without_escaped_newlines():
    from scripts.monitor_v2_eval_spot import yaml_meta, REPO
    path = REPO / "eval_jobs/eval_tcr_v2_esmc_v6_15000_20260927_t4-held20.yml"
    meta = yaml_meta(str(path))
    assert meta["tcr_tag"] == "tcr_v2_esmc_v6_15000_20260927"
    assert all("\\" not in p and "\n" not in p for p in meta["log_dirs"])


def test_t4_restart_archives_incomplete_and_protects_completed(tmp_path):
    from scripts.downstream.run_tcr_v2_generation_eval import prepare_output
    output = tmp_path / "run"
    output.mkdir()
    manifest = {"status": "running", "identity": {"checkpoint_sha256": "same"}}
    (output / "run_manifest.json").write_text(json.dumps(manifest))
    (output / "designs.jsonl").write_text("preserve me")
    prepare_output(output, {"checkpoint_sha256": "same"}, True)
    archived = list(tmp_path.glob("run.incomplete_*"))
    assert len(archived) == 1
    assert (archived[0] / "designs.jsonl").read_text() == "preserve me"
    manifest["status"] = "success"
    (output / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(FileExistsError, match="Completed"):
        prepare_output(output, {"checkpoint_sha256": "same"}, True)
