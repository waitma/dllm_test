"""Run in conda pllm: python -m pytest scripts/tests/test_ab_antigen_official.py -q.

Tests provenance rejection and upstream failure reporting using a tiny local git
repository. No model, biological dataset, or GPU is used.
"""
import subprocess

import pytest

from downstream.benchmark.ab_antigen.official import audit, digest, probe, verify_source


@pytest.fixture
def upstream(tmp_path):
    path = tmp_path / "upstream"
    path.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
    git("init", "-q")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "user.name", "Fixture")
    git("remote", "add", "origin", "https://example.invalid/official.git")
    (path / "entry.py").write_text("import argparse\nargparse.ArgumentParser().parse_args()\n")
    git("add", "entry.py")
    git("commit", "-qm", "fixture")
    spec = {"id": "upstream", "url": "https://example.invalid/official",
            "commit": git("rev-parse", "HEAD"), "release": "fixture",
            "protocol_status": "fixture", "files": {"entry.py": digest(path / "entry.py")},
            "cli_probes": [{"entry": "entry.py"}]}
    return tmp_path, path, spec, git


@pytest.mark.parametrize("mutation", ["commit", "remote", "dirty", "hash", "missing", "shadow"])
def test_rejects_changed_provenance(upstream, mutation):
    root, path, spec, git = upstream
    if mutation == "commit":
        spec["commit"] = "0" * 40
    elif mutation == "remote":
        git("remote", "set-url", "origin", "https://example.invalid/fork")
    elif mutation == "dirty":
        (path / "entry.py").write_text("raise RuntimeError('modified')\n")
    elif mutation == "hash":
        spec["files"]["entry.py"] = "0" * 64
    elif mutation == "missing":
        spec["files"]["missing.py"] = "0" * 64
    else:
        (path / "argparse.py").write_text("# module shadow\n")
    with pytest.raises(ValueError):
        verify_source(spec, root)


def test_help_is_never_inference(upstream):
    root, path, spec, git = upstream
    output = root / "logs"
    output.mkdir()
    result = probe(spec, output, source_root=root)
    assert result["probes"][0]["status"] == "cli_help_passed"
    assert result["inference_executed"] is False
    assert result["scores_produced"] is False
    assert not git("status", "--porcelain")


def test_upstream_failure_preserves_log(upstream):
    root, path, spec, git = upstream
    (path / "entry.py").write_text("raise RuntimeError('upstream failure')\n")
    git("add", "entry.py")
    git("commit", "-qm", "broken upstream")
    spec["commit"] = git("rev-parse", "HEAD")
    spec["files"]["entry.py"] = digest(path / "entry.py")
    output = root / "logs"
    output.mkdir()
    result = probe(spec, output, source_root=root)
    assert result["probes"][0]["status"] == "cli_help_failed"
    assert "upstream failure" in (output / "upstream_0.log").read_text()


def test_sparse_omission_is_not_upstream_missing(upstream):
    root, path, spec, git = upstream
    (path / "data.csv").write_text("x\n1\n")
    git("add", "data.csv")
    git("commit", "-qm", "upstream data")
    spec["commit"] = git("rev-parse", "HEAD")
    git("update-index", "--skip-worktree", "data.csv")
    (path / "data.csv").unlink()
    spec["required_upstream_files"] = ["data.csv", "absent.py"]
    result = audit(spec, root)
    assert result["status"] == "source_verified"
    assert result["required_files_absent_from_upstream_tree"] == ["absent.py"]
    assert result["static_checks_passed"] is False
