"""Independent row/integrity/layout audit; run update_mhc_inputs.py --verify-only.

Use protenix_abtcr after activating pllm. Scans every rewritten row, compares all
non-MHC content with its parent, checks the A4 capacity for every row, renders
the longest row for every distinct MHC layout through the real collators, and
runs one tiny CPU model backward pass for each representation category.
"""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from scripts.data.update_mhc_inputs import RAW, sha, non_mhc, write_json
from dllm.pipelines.immune_llada.data.preprocessing.validators import validate_manifest, validate_prepared_row
from dllm.pipelines.immune_llada.data.grammar import GrammarRenderer, GrammarTokenizer
from dllm.pipelines.immune_llada.data.dataset import PreparedImmuneDataset
from dllm.pipelines.immune_llada.v3.layout import ImmuneCollator, observe, RECEPTORS
from dllm.pipelines.immune_llada.v3.model import ImmuneConfig, ImmuneModel


def a4_length(record):
    length = 2
    for chain in record.chains:
        obs = observe(chain)
        if obs.scope == "missing":
            length += 3
            continue
        target = chain.role in RECEPTORS
        cap = (256 if obs.scope == "full" else 64) if target else len(obs.sequence) + 1
        if len(obs.sequence) + 1 > cap:
            return None
        length += cap + 1
    return length


def verify(output: Path):
    manifest_file = output / "dataset_manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest_sha = sha(manifest_file)
    validate_manifest(manifest)
    parent = Path(manifest["mhc_update"]["parent"])
    if sha(parent / "dataset_manifest.json") != manifest["mhc_update"]["parent_manifest_sha256"]:
        raise ValueError("Parent manifest no longer matches")
    stats = Counter()
    samples = {}
    legacy_samples = {}
    long_indices = {}
    violations = Counter()
    for split, block in manifest["splits"].items():
        native_offset = 0
        for shard in block["shards"]:
            new_path = output / shard["path"]
            old_path = parent / shard["path"]
            if shard["source"] not in RAW:
                if new_path.resolve() != old_path.resolve():
                    raise ValueError("Unchanged shard must reference its original")
                stats["unchanged_source_records"] += shard["records"]
                continue
            h = sha(new_path)
            assert h == shard["sha256"]
            assert sha(old_path) == shard["parent_sha256"]
            n = 0
            with old_path.open() as old, new_path.open() as new:
                for i, before in enumerate(old):
                    row = json.loads(next(new))
                    assert non_mhc(json.loads(before)) == non_mhc(row)
                    record = validate_prepared_row(row)
                    n += 1
                    stats["rewritten_rows_checked"] += 1
                    mhcs = [c for c in record.chains if c.role == "mhc"]
                    for c in mhcs:
                        assert c.metadata["mhc_policy"] == "sequence_first_mhc_v1"
                        assert c.sequence and set(c.sequence) != {"X"}
                    length = a4_length(record)
                    if length is None or length > 2048:
                        violations["a4_capacity_or_length"] += 1
                    else:
                        stats["max_a4_tokens"] = max(stats["max_a4_tokens"], length)
                    key = tuple((len(c.sequence), c.metadata["mhc_representation"], c.metadata["sequence_scope"]) for c in mhcs)
                    ranking = sum(len(c.sequence) for c in record.chains)
                    if key not in samples or ranking > samples[key][0]:
                        samples[key] = (ranking, record)
                    legacy_fits = all(len(c.sequence) < ({"tcr_alpha": 135, "tcr_beta": 167}.get(c.role, 10**9))
                                      for c in record.chains)
                    if legacy_fits:
                        if key not in legacy_samples or ranking > legacy_samples[key][0]:
                            legacy_samples[key] = (ranking, record)
                    else:
                        # The original v6 includes a few receptors too long for
                        # its later fixed canvas. They are retained, not cropped
                        # or silently filtered by this MHC-only update.
                        stats["preexisting_legacy_receptor_canvas_overflow_rows"] += 1
                    if shard["source"] == "tcr_native" and any(len(c.sequence) > 34 for c in mhcs):
                        long_indices.setdefault(split, native_offset + i)
                assert next(new, None) is None
            assert n == shard["records"]
            if shard["source"] == "tcr_native":
                native_offset += n
            print("verified", output.name, shard["path"], n, flush=True)
    if violations:
        raise ValueError(dict(violations))
    collators = {a: ImmuneCollator(a, max_length=2048) for a in ("A1", "A4")}
    renderer = GrammarRenderer(GrammarTokenizer(), fixed_receptor_lengths=True)
    selected = {}
    for _, record in legacy_samples.values():
        encoded = renderer.encode(record)
        stats["max_legacy_tokens_in_boundary_samples"] = max(stats["max_legacy_tokens_in_boundary_samples"], len(encoded["input_ids"]))
        batch = collators["A1"]([record])
        assert batch["input_ids"].shape[0] == 1 and batch["eligible_mask"].any()
        stats["A1_boundary_layouts_passed"] += 1
    for _, record in samples.values():
        batch = collators["A4"]([record])
        assert batch["input_ids"].shape[0] == 1 and batch["eligible_mask"].any()
        stats["A4_boundary_layouts_passed"] += 1
        mhcs = [c for c in record.chains if c.role == "mhc"]
        category = "provided_long" if any(len(c.sequence) > 34 for c in mhcs) else "pseudo34" if mhcs else "missing"
        selected.setdefault(category, record)
    # Check indexed dataset loading as well as direct deserialization.
    for split, index in long_indices.items():
        dataset = PreparedImmuneDataset(output, split=split, source="tcr_native")
        actual = dataset[index]
        assert any(c.role == "mhc" and len(c.sequence) > 34 for c in actual.chains)
        collators["A4"]([actual])
        stats["prepared_loader_checks"] += 1
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = ImmuneModel(ImmuneConfig(ablation="A4", d_model=24, n_layers=1, n_heads=3,
                                   mlp_hidden_size=48, max_length=2048))
    backward = {}
    for category, record in selected.items():
        model.zero_grad(set_to_none=True)
        batch = collators["A4"]([record])
        output_model = model(**batch, token_t=torch.ones_like(batch["input_ids"], dtype=torch.float))
        assert torch.isfinite(output_model["loss"])
        output_model["loss"].backward()
        assert torch.isfinite(model.backbone.transformer.wte.weight.grad).all()
        backward[category] = {"finite_loss_and_gradients": True, "tokens": batch["input_ids"].shape[1]}
    assert sha(manifest_file) == manifest_sha
    validation = dict(status="passed", created_at=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=manifest_sha, counts=dict(stats), cpu_tiny_model_backward=backward,
        scope="data/loader/layout/backprop compatibility, not biological performance or decontamination certification")
    write_json(output / "mhc_validation.json", validation)
    report_path = output / "mhc_update_report.json"
    report = json.loads(report_path.read_text())
    report["status"] = "data_and_layout_validated"
    report["validation"] = str(output / "mhc_validation.json")
    write_json(report_path, report)
    print(json.dumps(validation, ensure_ascii=False), flush=True)
