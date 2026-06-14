"""NbBench task registry for IRBench A1 (antibody/nanobody dimension).

Each task maps local ``hf_data/<name>/`` CSVs to a frozen-embedder + head-only
probe evaluation. Splits are taken from the official NbBench train/val/test files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
NBBENCH_DATA = PROJECT_ROOT / "data" / "nanobody_raw" / "nbbench" / "hf_data"


@dataclass(frozen=True)
class NbTask:
    name: str
    task_type: str          # "classification" | "regression"
    seq_columns: tuple[str, ...]
    label_column: str
    metric: str             # primary leaderboard metric name
    description: str = ""


# Head-only probing tasks with ready local CSV splits and simple scalar labels.
# VRClassification / CDRInfilling / Paratope need position-level or generative
# runners (see nbbench/README.md) and are excluded here.
NB_TASKS: dict[str, NbTask] = {
    "nanobody_type": NbTask(
        name="nanobody_type",
        task_type="classification",
        seq_columns=("seq",),
        label_column="label",
        metric="probe_acc",
        description="Nanobody type classification",
    ),
    "polyreaction": NbTask(
        name="polyreaction",
        task_type="classification",
        seq_columns=("seq",),
        label_column="label",
        metric="probe_auroc",
        description="Polyreactivity binary classification",
    ),
    "thermo-tm": NbTask(
        name="thermo-tm",
        task_type="regression",
        seq_columns=("seq",),
        label_column="label",
        metric="spearman",
        description="Melting temperature regression",
    ),
    "thermo-seq": NbTask(
        name="thermo-seq",
        task_type="regression",
        seq_columns=("seq",),
        label_column="label",
        metric="spearman",
        description="Thermostability sequence regression",
    ),
    "vhh_affinity-score": NbTask(
        name="vhh_affinity-score",
        task_type="regression",
        seq_columns=("seq",),
        label_column="score",
        metric="spearman",
        description="VHH binding affinity score regression",
    ),
    "hTNFa": NbTask(
        name="hTNFa",
        task_type="classification",
        seq_columns=("VHH_sequence", "Ag_sequence"),
        label_column="label",
        metric="probe_auroc",
        description="hTNFa binder classification (VHH + antigen)",
    ),
    "hIL6": NbTask(
        name="hIL6",
        task_type="classification",
        seq_columns=("VHH_sequence", "Ag_sequence"),
        label_column="label",
        metric="probe_auroc",
        description="hIL6 binder classification (VHH + antigen)",
    ),
    "SARS-CoV-2": NbTask(
        name="SARS-CoV-2",
        task_type="classification",
        seq_columns=("VHH_sequence", "Ag_sequence"),
        label_column="label",
        metric="probe_auroc",
        description="SARS-CoV-2 binder classification (VHH + antigen)",
    ),
}


def data_dir(task_name: str) -> Path:
    return NBBENCH_DATA / task_name


__all__ = ["NbTask", "NB_TASKS", "data_dir", "NBBENCH_DATA"]
