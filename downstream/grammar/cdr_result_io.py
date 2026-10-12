"""CDR observation artifacts and CPU-only rescoring; no model inference.

Run: python -m downstream.grammar.cdr_result_io --score-only <RUN_DIR>
Use --allow-partial only to inspect truncated/interrupted runs, never as complete scores.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import struct
from pathlib import Path

from downstream.benchmark.common.reporting import file_identity, object_sha256, sha256_file


def sample_id(source_sha256: str, fold: int, row: int, mode: str) -> str:
    return object_sha256({"source_sha256": source_sha256, "fold": fold, "row": row, "mode": mode})


def _float32(value: float) -> float:
    # Match masked_token_accuracy's float32 division before fold averaging.
    return struct.unpack("f", struct.pack("f", value))[0]


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("empty CDR result")
    folds = sorted({row["fold"] for row in rows})
    means = [statistics.mean(row["aar"] for row in rows if row["fold"] == fold) * 100 for fold in folds]
    return {"n_sequences": len(rows), "folds": folds, "fold_aar_percent": means,
            "average_aar": statistics.mean(row["aar"] for row in rows) * 100,
            "average_aar_all_folds": statistics.mean(means),
            "fold_sd_ddof0": statistics.pstdev(means)}


def checkpoint_identity(path: str | Path) -> dict:
    path = Path(path).resolve()
    weights = path / "model.safetensors" if path.is_dir() else path
    if not weights.is_file():
        raise FileNotFoundError(f"checkpoint weights missing: {weights}")
    parent = weights.parent
    return {"weights": file_identity(weights),
            "assets": [file_identity(p) for p in sorted(parent.glob("*.json"))]}


def _validate_manifest(manifest: dict) -> tuple[dict, dict[str, tuple]]:
    identity = manifest["identity"]
    if manifest.get("schema_version") != 1 or object_sha256(identity) != manifest["identity_sha256"]:
        raise ValueError("manifest identity mismatch")
    if not identity.get("checkpoint") or not identity.get("tokenizer") or not identity.get("code"):
        raise ValueError("missing checkpoint/tokenizer/code identity")
    expected = {}
    seen_folds = set()
    mode = identity["parameters"]["mode"]
    for source in identity["datasets"]:
        fold, count = source["fold"], source["n_rows"]
        if type(fold) is not int or fold in seen_folds or type(count) is not int or count <= 0:
            raise ValueError("empty/duplicate/invalid dataset fold")
        seen_folds.add(fold)
        for index in range(count):
            expected[sample_id(source["sha256"], fold, index, mode)] = (source["sha256"], fold, index, mode)
    if not expected:
        raise ValueError("empty CDR dataset")
    return identity, expected


def _validate_row(row: dict, manifest: dict, expected: dict) -> dict:
    if row.get("identity_sha256") != manifest["identity_sha256"]:
        raise ValueError("row checkpoint/dataset/parameter identity mismatch")
    coordinates = (row.get("source_sha256"), row.get("fold"), row.get("row"), row.get("mode"))
    if row.get("sample_id") not in expected or coordinates != expected[row["sample_id"]]:
        raise ValueError("unexpected sample or fold/mode/source identity mismatch")
    reference, prediction = row["reference_ids"], row["prediction_ids"]
    positions = row["masked_positions"]
    vocab = manifest["identity"]["tokenizer"]["tokens"]
    if (not reference or len(reference) != len(prediction) or len(reference) != len(positions)
            or any(type(p) is not int or p < 0 for p in positions) or positions != sorted(set(positions))):
        raise ValueError("empty target or invalid masked-token lengths/positions")
    if any(type(t) is not int or t < 0 or t >= len(vocab) for t in reference + prediction):
        raise ValueError("token outside the recorded vocabulary")
    if row.get("reference_tokens") != [vocab[t] for t in reference] or row.get("prediction_tokens") != [vocab[t] for t in prediction]:
        raise ValueError("token text/ID mismatch")
    correct = sum(a == b for a, b in zip(reference, prediction))
    aar = _float32(correct / len(reference))
    if row.get("correct") != correct or row.get("target_length") != len(reference):
        raise ValueError("incorrect AAR counts")
    if not math.isfinite(row.get("aar", float("nan"))) or abs(row["aar"] - aar) > 1e-7:
        raise ValueError("AAR differs from saved prediction/reference")
    return {**row, "aar": aar}


def score_artifacts(directory: str | Path, *, allow_partial: bool = False) -> dict:
    """Read-only scorer. Manifest + rows suffice; original weights/data are not loaded."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    _, expected = _validate_manifest(manifest)
    predictions = directory / "predictions.jsonl"
    if manifest.get("predictions_sha256") and sha256_file(predictions) != manifest["predictions_sha256"]:
        raise ValueError("predictions checksum mismatch")
    rows, seen = [], set()
    for line in predictions.read_text().splitlines():
        if not line.strip():
            raise ValueError("empty prediction line")
        row = _validate_row(json.loads(line), manifest, expected)
        if row["sample_id"] in seen:
            raise ValueError("duplicate sample_id")
        seen.add(row["sample_id"])
        rows.append(row)
    if not rows:
        raise ValueError("empty CDR predictions")
    missing = set(expected) - seen
    if missing and not allow_partial:
        raise ValueError(f"missing {len(missing)} predictions; use --allow-partial for diagnostic scoring")
    return {"schema_version": 1, "status": "partial" if missing else "complete",
            "identity_sha256": manifest["identity_sha256"],
            "predictions_sha256": sha256_file(predictions),
            "n_expected": len(expected), "n_missing": len(missing),
            "aggregation": "sequence_macro_then_fold_macro; fold_sd_ddof0; float32_per_sequence",
            **summarize_rows(rows)}


class CDRResultWriter:
    """Append observations to a fresh directory; seal only after validation."""

    def __init__(self, directory: str | Path, identity: dict):
        self.directory = Path(directory)
        self.manifest = {"schema_version": 1, "status": "running", "identity": identity,
                         "identity_sha256": object_sha256(identity)}
        _, self.expected = _validate_manifest(self.manifest)
        self.seen: set[str] = set()
        self.sealed = False
        self.directory.mkdir(parents=True, exist_ok=False)
        self._save_manifest()
        (self.directory / "predictions.jsonl").touch(exist_ok=False)

    def _save_manifest(self):
        temporary = self.directory / "manifest.json.tmp"
        temporary.write_text(json.dumps(self.manifest, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.directory / "manifest.json")

    def append(self, *, fold: int, row: int, reference_ids: list[int], prediction_ids: list[int],
               masked_positions: list[int], aar: float) -> None:
        if self.sealed:
            raise ValueError("CDR artifacts already sealed; use a new output directory")
        identity = self.manifest["identity"]
        source = next(s for s in identity["datasets"] if s["fold"] == fold)
        mode = identity["parameters"]["mode"]
        vocab = identity["tokenizer"]["tokens"]
        record = {"sample_id": sample_id(source["sha256"], fold, row, mode),
                  "identity_sha256": self.manifest["identity_sha256"],
                  "source_sha256": source["sha256"], "fold": fold, "row": row, "mode": mode,
                  "reference_ids": reference_ids, "prediction_ids": prediction_ids,
                  "reference_tokens": [vocab[t] for t in reference_ids],
                  "prediction_tokens": [vocab[t] for t in prediction_ids],
                  "masked_positions": masked_positions, "target_length": len(reference_ids),
                  "correct": sum(a == b for a, b in zip(reference_ids, prediction_ids)), "aar": aar}
        _validate_row(record, self.manifest, self.expected)
        if record["sample_id"] in self.seen:
            raise ValueError("duplicate sample_id")
        with (self.directory / "predictions.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        self.seen.add(record["sample_id"])

    def finish(self, *, allow_partial: bool = False) -> dict:
        if self.sealed:
            raise ValueError("CDR artifacts already sealed")
        result = score_artifacts(self.directory, allow_partial=allow_partial)
        self.manifest.update(status=result["status"], predictions_sha256=result["predictions_sha256"])
        self._save_manifest()
        (self.directory / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        self.sealed = True
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-only", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    print(json.dumps(score_artifacts(args.score_only, allow_partial=args.allow_partial), indent=2))


if __name__ == "__main__":
    main()
