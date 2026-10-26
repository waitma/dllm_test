"""Activate pllm; CUDA_VISIBLE_DEVICES='' USE_TF=0 python -B -m pytest THIS_FILE -q.

Exercise random initialization without checkpoint loaders, identity and resumption.
"""
from dataclasses import asdict
import json

import pytest
import torch

from scripts.research.stage1_ab500k_initialization import (
    build_scratch_model, shared_state_hash, canonical_hash, verify_scratch_initialization)
from scripts.research.run_stage1_ab500k import new_model, new_optimizer, DEFAULTS, prepare_batch
from scripts.research.report_stage1_ab500k import load_arm
from scripts.tests.immune_llada.test_stage1_ab500k_training import model_config, rows
from dllm.pipelines.immune_llada.fixed_chain.full_token_mlm import FullTokenMLMModel
from dllm.pipelines.immune_llada.fixed_chain.structure_mlm import StructureMLMModel


def plan():
    config = asdict(model_config())
    pair = dict(width=8, query_chunk=8)
    a = build_scratch_model(config, "mlm", 42, pair)
    return dict(model=config, pair=pair, training={**DEFAULTS, "backbone_lr": 1e-4},
                teacher_contract={"unit": True}, parent_checkpoint=None, parent_step=None,
                parent_model_sha256=None, parent_receipt_sha256=None,
                initialization=dict(schema="immune.stage1.random_init.v1", mode="scratch", seed=42,
                    config_sha256=canonical_hash(config), shared_state_sha256=shared_state_hash(a)))


def test_both_arms_use_identical_random_weights_and_no_checkpoint_load(monkeypatch):
    torch.set_num_threads(1)
    def forbidden(*args, **kwargs):
        raise AssertionError("A pretrained loader was called")
    monkeypatch.setattr(FullTokenMLMModel, "from_pretrained", forbidden)
    monkeypatch.setattr(StructureMLMModel, "from_bert", forbidden)
    spec = plan()
    a, d = (new_model(spec, arm, torch.device("cpu")).eval() for arm in ("mlm", "pair_kd"))
    assert shared_state_hash(a) == shared_state_hash(d) == spec["initialization"]["shared_state_sha256"]
    for key, tensor in a.state_dict().items():
        assert torch.equal(tensor, d.state_dict()[key])
    batch = prepare_batch(a, rows(a.config, 2), seed=10042, targets=False)[0]
    with torch.no_grad():
        assert torch.equal(a(**batch)["logits"], d(**batch)["logits"])
    assert not new_optimizer(a, spec["training"], "mlm").state
    assert not new_optimizer(d, spec["training"], "pair_kd").state
    different = build_scratch_model(spec["model"], "mlm", 43)
    assert shared_state_hash(different) != shared_state_hash(a)


@pytest.mark.parametrize("change", ["parent", "seed", "config", "fingerprint"])
def test_scratch_contract_rejects_mixed_or_modified_initialization(change):
    spec = plan()
    if change == "parent":
        spec["parent_checkpoint"] = "/unavailable/old-checkpoint"
    elif change == "seed":
        spec["initialization"]["seed"] = 43
    elif change == "config":
        spec["model"]["d_model"] = 64
    else:
        spec["initialization"]["shared_state_sha256"] = "0"*64
    with pytest.raises(ValueError):
        new_model(spec, "mlm", torch.device("cpu"))


def test_report_accepts_exact_ten_epochs_and_requires_final_validation(tmp_path):
    p = tmp_path/"mlm"
    p.mkdir()
    (p/"COMPLETE.json").write_text(json.dumps(dict(status="success", step=9770, seen_records=5000000)))
    (p/"run_provenance.json").write_text('{}')
    steps = sorted(set(range(0, 9771, 250)) | {9770})
    for step in steps:
        (p/f"validation_step{step}.json").write_text(json.dumps(dict(step=step, records=4096, masks=3)))
    settings = dict(max_steps=9770, global_batch=512, eval_every=250)
    assert len(load_arm(tmp_path, "mlm", settings)["validation"]) == len(steps)
    (p/"validation_step9770.json").unlink()
    with pytest.raises(ValueError, match="Missing"):
        load_arm(tmp_path, "mlm", settings)
