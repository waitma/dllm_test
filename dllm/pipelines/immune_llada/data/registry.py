"""Registry and path rules for the canonical immune data sources.

Use ``parse_sources`` for ``+``-separated or list-valued source selections and
``source_split_path`` to resolve a source directory to its split file.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .sources import (
    asd_antibody_row_to_record,
    asd_nanobody_row_to_record,
    oas_row_to_record,
    ots_row_to_record,
    tcr_native_row_to_record,
    tcr_papers_row_to_record,
    tcr_repertoire_row_to_record,
    trait_row_to_record,
)

DEFAULT_DATA_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test/data")
OAS_LABEL_FILE_TEMPLATE = "cleaned_merged_data_step_clustered_{split}_oas_label.csv"

_DEFAULT_PATHS = {
    "oas": DEFAULT_DATA_ROOT / "oas_previous_clean/splits",
    "ots": DEFAULT_DATA_ROOT / "ots_paired_clean/final",
    "asd_antibody": Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/asd/step6_final/antibody"),
    "asd_nanobody": Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/asd/step6_final/nanobody"),
    "trait": Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/trait/step4_final"),
    "tcr_native": DEFAULT_DATA_ROOT / "tcr_native/dataset",
    "tcr_papers": DEFAULT_DATA_ROOT / "tcr_papers_v2/dataset",
    "tcr_repertoire": DEFAULT_DATA_ROOT / "tcr_repertoire/dataset",
}

# The default recipe is exactly the seven-source active mix. ASD nanobody is
# registered and can be selected explicitly, but is intentionally not here.
DEFAULT_SOURCES = (
    "oas", "ots", "asd_antibody", "trait", "tcr_native", "tcr_papers", "tcr_repertoire",
)

SourceAdapter = Callable[..., object]
SOURCE_REGISTRY: dict[str, SourceAdapter] = {
    "oas": oas_row_to_record,
    "ots": ots_row_to_record,
    "asd_antibody": asd_antibody_row_to_record,
    "asd_nanobody": asd_nanobody_row_to_record,
    "trait": trait_row_to_record,
    "tcr_native": tcr_native_row_to_record,
    "tcr_papers": tcr_papers_row_to_record,
    "tcr_repertoire": tcr_repertoire_row_to_record,
}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    weight: float = 1.0


def parse_sources(value: str | list[str] | tuple[str, ...]) -> list[str]:
    """Parse source names, preserving first-seen order and removing duplicates.

    Strings use the legacy ``+`` separator.  The ``asd`` convenience token
    expands to ``asd_antibody`` followed by ``asd_nanobody``.
    """
    parts = [value] if isinstance(value, str) else list(value)
    tokens: list[str] = []
    for part in parts:
        tokens.extend(str(part).split("+"))
    expanded: list[str] = []
    for token in (item.strip() for item in tokens):
        if not token:
            continue
        if token == "asd":
            expanded.extend(("asd_antibody", "asd_nanobody"))
        else:
            expanded.append(token)
    result: list[str] = []
    seen: set[str] = set()
    for name in expanded:
        if name not in SOURCE_REGISTRY:
            raise ValueError(f"Unknown immune source {name!r}; known: {sorted(SOURCE_REGISTRY)}")
        if name not in seen:
            result.append(name)
            seen.add(name)
    if not result:
        raise ValueError(f"No dataset sources parsed from {value!r}")
    return result


def source_spec(name: str, *, path: Path | str | None = None, weight: float = 1.0) -> SourceSpec:
    if name not in SOURCE_REGISTRY:
        raise ValueError(f"Unknown immune source {name!r}; known: {sorted(SOURCE_REGISTRY)}")
    return SourceSpec(name, Path(path) if path is not None else _DEFAULT_PATHS[name], float(weight))


def _canonical_csv_split(split: str) -> str:
    normalized = str(split).strip().lower()
    if normalized in {"val", "valid", "validation"}:
        return "valid"
    return normalized or "train"


def source_split_path(spec: SourceSpec, split: str) -> Path:
    """Resolve a source file, including the OAS labelled filename convention."""
    if spec.path.is_file():
        return spec.path
    if spec.name == "oas":
        return spec.path / OAS_LABEL_FILE_TEMPLATE.format(split=_canonical_csv_split(split))
    return spec.path / f"{split}.csv"


__all__ = ["DEFAULT_SOURCES", "OAS_LABEL_FILE_TEMPLATE", "SOURCE_REGISTRY", "SourceSpec", "parse_sources", "source_spec", "source_split_path"]
