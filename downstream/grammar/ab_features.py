"""Native, frozen antibody features for ESMC + LLaDA checkpoints.

Used by: python -m downstream.grammar.ab_probes --help
No Ophiuchus fixed windows, EOS padding, or TCR record construction is used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from downstream.grammar.common import antibody_pair_record

ROOT = Path(__file__).resolve().parents[2]
FEATURE_PROTOCOL = "ab_clean_full_hl_postllada_global_residue_mean_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def clean_sequence(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Antibody sequences must be nonempty strings")
    seq = "".join(value.split()).upper().replace("-", "").replace("J", "L")
    if not seq or set(seq) - set("ACDEFGHIKLMNPQRSTVWYBXZOU"):
        raise ValueError("Missing or invalid antibody sequence")
    return seq


@torch.no_grad()
def embed_native_pairs(model, collator, pairs, *, device, batch_size=16):
    """Clean AB forward; reject any truncation before global residue pooling."""
    outputs = []
    model.eval()
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        records = [antibody_pair_record(clean_sequence(h), clean_sequence(l)) for h, l in chunk]
        batch = collator(records)
        expected = torch.tensor([sum(len(c.sequence) for c in r.chains) for r in records])
        mask = batch["residue_mask"].bool() & batch["attention_mask"].bool()
        if not torch.equal(mask.sum(1).cpu(), expected):
            raise ValueError("Native AB input was truncated or lost residues")
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        hidden = model.last_hidden_state(**batch).float()
        mask = mask.to(device)
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        if not torch.isfinite(pooled).all():
            raise ValueError("Nonfinite post-LLaDA features")
        outputs.append(pooled.cpu())
        print(f"AB features {min(start + batch_size, len(pairs))}/{len(pairs)}", flush=True)
    return torch.cat(outputs)


def cached_features(pairs, *, checkpoint: Path, out_dir: Path, device: str,
                    batch_size: int, max_length: int):
    from examples.llada.load_fusion_checkpoint import load_fusion_for_eval, resolve_fusion_dir

    checkpoint = resolve_fusion_dir(checkpoint)
    pairs = [(clean_sequence(h), clean_sequence(l)) for h, l in pairs]
    sources = [Path(__file__), ROOT / "examples/llada/load_fusion_checkpoint.py",
               ROOT / "examples/llada/protein_fusion_model.py",
               ROOT / "dllm/pipelines/immune_llada/data/grammar.py",
               ROOT / "downstream/grammar/common.py",
               ROOT / "dllm/pipelines/llada/models/modeling_llada.py"]
    manifest = {
        "feature_protocol": FEATURE_PROTOCOL, "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint / "model.safetensors"),
        "checkpoint_json_sha256": {p.name: file_sha256(p) for p in sorted(checkpoint.glob("*.json"))},
        "pairs_sha256": json_hash(pairs), "n": len(pairs), "max_length": max_length,
        "batch_size": batch_size, "dtype": "bfloat16", "pool_accumulation": "float32",
        "torch_version": str(torch.__version__),
        "source_sha256": {str(p): file_sha256(p) for p in sources},
    }
    cache = out_dir / "features.pt"
    if cache.exists():
        data = torch.load(cache, map_location="cpu", weights_only=False)
        if data.get("manifest") != manifest:
            raise ValueError(f"Feature cache provenance mismatch: {cache}")
        features = data["features"]
        if features.ndim != 2 or len(features) != len(pairs) or not torch.isfinite(features).all():
            raise ValueError("Invalid cached feature tensor")
    else:
        bundle = load_fusion_for_eval(checkpoint, device=device, max_length=max_length)
        features = embed_native_pairs(bundle.model, bundle.collator, pairs,
                                      device=device, batch_size=batch_size)
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".pt.tmp")
        torch.save({"manifest": manifest, "features": features}, tmp)
        tmp.replace(cache)
        del bundle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(out_dir / "feature_manifest.json", {**manifest, "shape": list(features.shape)})
    return features, manifest
