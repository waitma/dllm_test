"""Build AB500K students from random weights, without reading any checkpoint.

Activate pllm; run the absolute test_stage1_ab500k_scratch.py path with pytest.
The frozen plan records config, seed and a fingerprint of all shared tensors.
"""
import hashlib
import json

import torch

from dllm.pipelines.immune_llada.fixed_chain.full_token_mlm import FullTokenMLMModel
from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
from dllm.pipelines.immune_llada.fixed_chain.structure_mlm import StructureMLMModel, PairConfig


PAIR_PREFIXES = ("pair_encoder.", "layer_pair_encoders.", "pair_bias_outputs.", "distogram_head.")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def shared_state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith(PAIR_PREFIXES):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def build_scratch_model(config, arm, seed, pair=None):
    if arm not in {"mlm", "pair_kd"}:
        raise ValueError("Unknown scratch arm")
    torch.manual_seed(seed)
    spec = FixedChainConfig(**config)
    model = (FullTokenMLMModel(spec) if arm == "mlm" else
             StructureMLMModel(spec, PairConfig(**(pair or {}))))
    if arm == "pair_kd":
        model.parent_provenance = dict(initialization="scratch", seed=seed,
                                       checkpoint=None, inherited_parameters=0)
    return model


def verify_scratch_initialization(plan):
    spec = plan.get("initialization", {})
    if spec.get("mode") != "scratch" or spec.get("schema") != "immune.stage1.random_init.v1":
        raise ValueError("Invalid scratch initialization contract")
    if any(plan.get(key) is not None for key in (
            "parent_checkpoint", "parent_step", "parent_model_sha256", "parent_receipt_sha256")):
        raise ValueError("Scratch initialization must not reference parent weights")
    if spec.get("seed") != plan["training"]["seed"]:
        raise ValueError("Scratch initialization seed differs")
    if spec.get("config_sha256") != canonical_hash(plan["model"]):
        raise ValueError("Scratch model configuration changed")
    digest = spec.get("shared_state_sha256", "")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Missing shared random initialization fingerprint")
    FixedChainConfig(**plan["model"])
    return spec
