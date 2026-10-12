"""Compact chain-aware LLaDA; run pytest scripts/tests/immune_llada/test_v3.py.

Reuses the existing LLaDA blocks and absorbing-mask noise sampler. No ESMC
weights, reference-dependent feature masks, or independent length head.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from dllm.pipelines.llada.models.configuration_llada import ModelConfig
from dllm.pipelines.llada.models.modeling_llada import LLaDAModel
from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import sample_bioseq_diffusion_noise
from ..data.grammar import GrammarTokenizer
from ..data.records import CHAIN_ROLE_TO_ID
from .layout import SCOPES
from .vocabulary import make_tokenizer


@dataclass
class ImmuneConfig:
    ablation: str = "A4"
    d_model: int = 768
    n_layers: int = 8
    n_heads: int = 12
    mlp_hidden_size: int = 3072
    max_length: int = 2048
    full_capacity: int = 256
    fragment_capacity: int = 64
    schema_version: str = "immune_llada.noesmc.v3"

    def __post_init__(self):
        if self.schema_version != "immune_llada.noesmc.v3" or self.ablation not in {"A0", "A1", "A2", "A3", "A4", "V5"}:
            raise ValueError("Not a compatible compact v3 configuration")


class ImmuneModel(nn.Module):
    def __init__(self, config: ImmuneConfig):
        super().__init__()
        self.config = config
        self.tokenizer = make_tokenizer(config.ablation)
        self.terminal_id = self.tokenizer.special_id("<protd>") if config.ablation in {"A4", "V5"} else self.tokenizer.eos_token_id
        base = ModelConfig(d_model=config.d_model, n_layers=config.n_layers, n_heads=config.n_heads,
                           n_kv_heads=config.n_heads, mlp_hidden_size=config.mlp_hidden_size,
                           vocab_size=self.tokenizer.vocab_size, embedding_size=self.tokenizer.vocab_size,
                           max_sequence_length=config.max_length, rope=True, block_type="llama", activation_type="silu",
                           layer_norm_type="rms", weight_tying=False, init_device="cpu", rope_theta=500000.0,
                           init_fn="mitchell", init_std=0.02, scale_logits=False, bias_for_layer_norm=False,
                           attention_dropout=0.0, residual_dropout=0.0, embedding_dropout=0.0)
        self.backbone = LLaDAModel(base, init_params=True)
        self.role_embeddings = nn.Embedding(len(CHAIN_ROLE_TO_ID), config.d_model, padding_idx=0) if config.ablation not in {"A0","A1"} else None
        self.scope_embeddings = nn.Embedding(len(SCOPES), config.d_model, padding_idx=0) if config.ablation not in {"A0","A1"} else None
        self.chain_bias = nn.Parameter(torch.zeros(2, config.n_heads)) if config.ablation in {"A3", "A4", "V5"} else None
        for embedding in (self.role_embeddings, self.scope_embeddings):
            if embedding is not None:
                nn.init.normal_(embedding.weight, std=0.02)
                with torch.no_grad():
                    embedding.weight[0].zero_()
        allowed = torch.zeros(self.tokenizer.vocab_size, dtype=torch.bool)
        allowed[self.tokenizer.encode_residues("LAGVSERTIDPKQNFYMHWCXBUZO")] = True
        allowed[self.terminal_id] = True
        # The fixed V5 object header is context; content learns chain/end delimiters.
        if config.ablation == "V5":
            allowed[self.tokenizer.chain_separator_id()] = True
        self.register_buffer("allowed_target", allowed, persistent=False)

    def denoise(self, input_ids, attention_mask, role_ids, scope_ids, chain_ids, position_ids,
                target_slot_mask, *, output_hidden_states=False):
        """Explicit whitelist: clean labels, end masks and residue masks never enter forward features."""
        x = self.backbone.transformer.wte(input_ids)
        if self.role_embeddings is not None:
            x = x + self.role_embeddings(role_ids) + self.scope_embeddings(scope_ids)
        bias = None
        if self.chain_bias is not None:
            same = (chain_ids[:, :, None] == chain_ids[:, None, :]) & (chain_ids[:, :, None] >= 0) & (chain_ids[:, None, :] >= 0)
            bias = self.chain_bias[same.long()].permute(0, 3, 1, 2).contiguous()
        output = self.backbone(input_ids=None, input_embeddings=x, attention_mask=attention_mask,
                               attention_bias=bias, position_ids=position_ids if self.chain_bias is not None else None,
                               use_cache=False, output_hidden_states=output_hidden_states)
        # A0-A3 retain the legacy full-softmax training objective; vocabulary
        # reduction alone must not also silently change its allowed-token mask.
        logits = output.logits
        if self.config.ablation in {"A4", "V5"}:
            logits = logits.masked_fill(target_slot_mask[:, :, None] & ~self.allowed_target[None, None], float("-inf"))
        return logits, output.hidden_states[-1] if output_hidden_states else None

    @staticmethod
    def feature_args(batch):
        return {k: batch[k] for k in ("input_ids", "attention_mask", "role_ids", "scope_ids", "chain_ids", "position_ids", "target_slot_mask")}

    def forward(self, input_ids, labels, attention_mask, role_ids, scope_ids, chain_ids, position_ids,
                target_slot_mask, eligible_mask, residue_mask, terminal_mask, token_t=None):
        features = dict(input_ids=input_ids, attention_mask=attention_mask, role_ids=role_ids, scope_ids=scope_ids,
                        chain_ids=chain_ids, position_ids=position_ids, target_slot_mask=target_slot_mask)
        noise_batch = dict(input_ids=input_ids, attention_mask=attention_mask,
                           diffusion_eligible_mask=eligible_mask, diffusion_loss_mask=eligible_mask,
                           residue_mask=residue_mask)
        noised, targets, corrupted, _ = sample_bioseq_diffusion_noise(noise_batch, self.tokenizer.mask_token_id, token_t=token_t)
        if not torch.equal(targets[corrupted], labels[corrupted]):
            raise ValueError("Input/label mismatch in supervised target slots")
        logits, _ = self.denoise(**{**features, "input_ids": noised})
        per_token = F.cross_entropy(logits.transpose(1, 2).float(), targets, ignore_index=-100, reduction="none")
        selected = targets != -100

        def average(mask):
            mask = mask & selected
            return (per_token * mask).sum() / mask.sum().clamp_min(1)

        aa_selected, end_selected = residue_mask & selected, terminal_mask & selected
        sample_metrics = torch.stack(((per_token*aa_selected).sum(1), aa_selected.sum(1),
                                      (per_token*end_selected).sum(1), end_selected.sum(1)),dim=-1).detach()
        return {"loss": average(selected), "aa_loss": average(residue_mask).detach(),
                "terminal_loss": average(terminal_mask).detach(),
                "aa_count": (residue_mask & selected).sum().detach(),
                "terminal_count": (terminal_mask & selected).sum().detach(),"sample_metrics":sample_metrics}

    @torch.no_grad()
    def represent(self, batch):
        """Call with representation-mode layout; return residues, segment means and complex mean (V5: object segments)."""
        _, hidden = self.denoise(**self.feature_args(batch), output_hidden_states=True)
        mask = batch["residue_mask"].bool() & batch["attention_mask"].bool()
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        pool_chain_ids = batch["chain_ids"]  # V5 IDs are protein objects, not internal chains.
        chains = []
        for b in range(hidden.shape[0]):
            chains.append({int(uid): hidden[b][mask[b] & (pool_chain_ids[b] == uid)].mean(0)
                           for uid in pool_chain_ids[b][mask[b]].unique(sorted=True)})
        return {"residue_embeddings": hidden, "residue_mask": mask, "chain_embeddings": chains, "pooled": pooled}

    @torch.no_grad()
    def generate(self, batch, *, steps=32, temperature=0.0, seed=42, expected_chains=None):
        """Prototype mask-predict with revisable terminal; never consult clean target masks.

        Each iteration proposes all slot tokens. The first proposed terminal is
        provisional; it and its suffix remain MASK for the next iteration.
        Low-confidence prefix tokens are remasked on a decreasing schedule.
        """
        if self.config.ablation == "V5":
            from .v5_generation import generate_features
            rng = torch.Generator(device=next(self.parameters()).device).manual_seed(seed)
            return generate_features(self, self.feature_args(batch), steps=steps,
                                     temperature=temperature, generator=rng, expected_chains=expected_chains)
        if steps < 1 or temperature < 0:
            raise ValueError("Invalid generation schedule")
        features = self.feature_args(batch)
        slots = features["target_slot_mask"].bool()
        ids = features["input_ids"].clone().masked_fill(slots, self.tokenizer.mask_token_id)
        rng = torch.Generator(device=ids.device).manual_seed(seed)
        final = ids.clone()
        for step in range(steps):
            logits, _ = self.denoise(**{**features, "input_ids": ids})
            logits = logits.masked_fill(slots[:, :, None] & ~self.allowed_target[None, None], float("-inf"))
            probs = torch.softmax(logits.float() / (temperature if temperature else 1.0), -1)
            proposal = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1, generator=rng).reshape_as(ids) if temperature else logits.argmax(-1)
            proposal = torch.where(ids == self.tokenizer.mask_token_id, proposal, ids)
            confidence = probs.gather(-1, proposal.unsqueeze(-1)).squeeze(-1)
            final = torch.where(slots, proposal, features["input_ids"])
            ids = final.clone()
            if step == steps-1:
                break
            for b in range(ids.shape[0]):
                for uid in features["chain_ids"][b][slots[b]].unique():
                    positions = (slots[b] & (features["chain_ids"][b] == uid)).nonzero().flatten()
                    ends = (proposal[b, positions] == self.terminal_id).nonzero().flatten()
                    end = int(ends[0]) if len(ends) else len(positions)
                    ids[b, positions[end:]] = self.tokenizer.mask_token_id
                    count = math.ceil(end * (1-(step+1)/steps))
                    if count:
                        uncertain = confidence[b, positions[:end]].topk(count, largest=False).indices
                        ids[b, positions[uncertain]] = self.tokenizer.mask_token_id
        results = []
        for b in range(ids.shape[0]):
            chains = []
            for uid in features["chain_ids"][b][slots[b]].unique(sorted=True):
                positions = (slots[b] & (features["chain_ids"][b] == uid)).nonzero().flatten()
                tokens = final[b, positions].tolist()
                end = tokens.index(self.terminal_id) if self.terminal_id in tokens else len(tokens)
                status = "capacity_overflow" if end == len(tokens) else "empty_chain" if end == 0 else "terminated"
                chains.append({"chain_id": int(uid), "role_id": int(features["role_ids"][b, positions[0]]),
                               "sequence": "".join(self.tokenizer.token(t) for t in tokens[:end]),
                               "status": status, "capacity": len(tokens), "terminal_count": tokens.count(self.terminal_id)})
            results.append(chains)
        return {"chains": results, "input_ids": final}

    def save_pretrained(self, directory):
        from safetensors.torch import save_file
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "immune_config.json").write_text(json.dumps(asdict(self.config), indent=2)+"\n")
        (directory / "vocabulary.json").write_text(json.dumps([self.tokenizer.token(i) for i in range(self.tokenizer.vocab_size)], indent=2)+"\n")
        save_file({k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}, str(directory / "model.safetensors"))

    @classmethod
    def from_pretrained(cls, directory):
        from safetensors.torch import load_file
        directory = Path(directory)
        model = cls(ImmuneConfig(**json.loads((directory / "immune_config.json").read_text())))
        if json.loads((directory / "vocabulary.json").read_text()) != [model.tokenizer.token(i) for i in range(model.tokenizer.vocab_size)]:
            raise ValueError("Vocabulary fingerprint differs")
        model.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True)
        return model
