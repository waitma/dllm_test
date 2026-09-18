"""CPU checks: python -m pytest scripts/tests/bioseq/test_ab_specificity_budget_monitor.py -q."""
import pytest

from scripts.downstream.monitor_ab_specificity_budgets import (
    JOBS, PROCESS_END, PROCESS_START, RESULT_END, RESULT_START,
    close, publish, replace_owned, save_if_unchanged,
)


def test_numeric_guard_rejects_nonfinite_and_mismatch():
    close(1.0, 1.0 + 1e-12)
    for a, b in [(1, 2), (float("nan"), 1), (float("inf"), float("inf"))]:
        with pytest.raises(ValueError):
            close(a, b)


def test_owned_marker_guards():
    with pytest.raises(ValueError):
        replace_owned("unrelated document", RESULT_START, RESULT_END, "replacement")
    with pytest.raises(ValueError):
        replace_owned(RESULT_START * 2 + RESULT_END, RESULT_START, RESULT_END, "replacement")


def test_concurrent_change_is_preserved(tmp_path):
    path = tmp_path / "report.md"
    path.write_text("user changed content")
    with pytest.raises(RuntimeError, match="Concurrent"):
        save_if_unchanged(path, "original content", "replacement")
    assert path.read_text() == "user changed content"
    assert list(tmp_path.glob("*.tmp")) == []


def test_only_audited_success_is_published_and_only_its_active_rows_removed(tmp_path):
    result = tmp_path / "RESULTS.md"
    process = tmp_path / "PROJECT_PROCESS.md"
    preserved = "| baseline | keep original |\n| old checkpoint | keep original |"
    result.write_text("prefix\n" + RESULT_START + "\n" + preserved + "\n" + RESULT_END + "\nsuffix\n")
    rows = "\n".join(f"| `{job}` | active |" for job in JOBS.values())
    process.write_text(
        "> Last updated: old\n## Active Volc Training Tasks\nLast updated: old\n" + rows +
        "\n## Active Volc Evaluation Tasks\nLast updated: old\n" + rows +
        "\n## Unrelated history\nkeep history\n" + PROCESS_START + "\npending\n" + PROCESS_END + "\n")
    jobs = {str(e): {"JobId": job, "Status": "Success" if e == 300 else "Running", "End": "test"}
            for e, job in JOBS.items()}
    row = {"final_train_loss": .1, "final_eval_loss": .2,
           "metrics": {"accuracy": .7, "f1_macro": .6, "mcc": .5}}
    status = {"updated_utc": "2026-09-15T00:00:00+00:00", "jobs": jobs}
    publish(status, {300: row, 400: row}, tmp_path, result_path=result, process_path=process)
    text = result.read_text()
    assert preserved in text and text.startswith("prefix\n") and text.endswith("\nsuffix\n")
    assert "300轮 | 0.100000 | 0.200000 | 0.700000 | 0.600000 | 0.500000" in text
    assert "400轮 | 待完整验收" in text and "500轮 | 待完整验收" in text
    active = process.read_text().split("## Unrelated history")[0]
    assert JOBS[300] not in active and active.count(JOBS[400]) == 2 and active.count(JOBS[500]) == 2
    assert "keep history" in process.read_text()
    snapshots = (result.read_text(), process.read_text())
    publish(status, {300: row, 400: row}, tmp_path, result_path=result, process_path=process)
    assert snapshots == (result.read_text(), process.read_text())
