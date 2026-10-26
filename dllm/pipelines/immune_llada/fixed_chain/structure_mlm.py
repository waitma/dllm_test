"""Sequence-only, all-layer pair feedback initialized from full-token BERT.

After activating pllm:
  USE_TF=0 python -B -m pytest scripts/tests/immune_llada/test_structure_mlm.py -q
Teacher tensors and supervision masks never enter the encoder. Fixed chain
positions (including MASK slots) may exchange messages; only the KD objective
uses the independently validated observed-residue map.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .full_token_mlm import FullTokenMLMModel, RECIPE
from .mlm import selected_loss
from .model import FixedChainConfig
from ..data.records import CHAIN_ROLE_TO_ID


SCHEMA = "immune.fixed_chain.structure_bert_mlm.v1"


@dataclass(frozen=True)
class PairConfig:
    """All dimensions and feedback locations are checkpoint-bound."""

    width: int = 64
    distogram_bins: int = 64
    relative_clip: int = 32
    query_chunk: int = 16
    feedback: str = "all"  # all, global, none
    checkpoint_pair_chunks: bool = True
    bias_implementation: str = "dense"  # symmetric is valid only for this symmetric encoder
    autocast_pair_embeddings: bool = False
    symmetric_pair_chunk: int = 4096
    independent_layers: bool = False
    distogram_chunk_size: int = 0  # 0 preserves the original unchunked readout

    def __post_init__(self):
        if min(self.width, self.query_chunk, self.relative_clip) < 1 or self.distogram_bins < 2:
            raise ValueError("Invalid pair dimensions")
        if self.feedback not in {"all", "global", "none"}:
            raise ValueError("Feedback must be all, global or none")
        if self.bias_implementation not in {"dense", "symmetric"} or self.symmetric_pair_chunk < 1:
            raise ValueError("Invalid Pair execution configuration")
        if type(self.independent_layers) is not bool or self.distogram_chunk_size < 0:
            raise ValueError("Invalid per-layer Pair configuration")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


class ResiduePairEncoder(nn.Module):
    """Symmetric sequence pair features with a separate inter-chain position bin."""

    def __init__(self, hidden_size, config):
        super().__init__()
        self.config = config
        width = config.width
        self.num_roles = len(CHAIN_ROLE_TO_ID)
        self.single_norm = nn.LayerNorm(hidden_size)
        self.single_projection = nn.Linear(hidden_size, width)
        self.relative = nn.Embedding(config.relative_clip + 2, width)
        self.roles = nn.Embedding(2 * self.num_roles * self.num_roles, width)
        self.pair_mlp = nn.Sequential(
            nn.Linear(5 * width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width),
            nn.LayerNorm(width),
        )

    def project(self, hidden):
        return self.single_projection(self.single_norm(hidden))

    def pairs(self, left, right, position_left, position_right,
              chain_left, chain_right, role_left, role_right):
        same = chain_left.eq(chain_right)
        relative = (position_left - position_right).abs().clamp_max(self.config.relative_clip)
        relative = torch.where(same, relative, self.config.relative_clip + 1)
        roles = (torch.minimum(role_left, role_right) * self.num_roles
                 + torch.maximum(role_left, role_right)
                 + (~same).long() * self.num_roles * self.num_roles)
        relative_embedding, role_embedding = self.relative(relative), self.roles(roles)
        if self.config.autocast_pair_embeddings and torch.is_autocast_enabled(left.device.type):
            # The first Linear casts these embeddings anyway. Avoid an FP32 cat
            # temporary when the projected residue streams are already BF16.
            dtype = torch.get_autocast_dtype(left.device.type)
            relative_embedding, role_embedding = relative_embedding.to(dtype), role_embedding.to(dtype)
        return self.pair_mlp(torch.cat([
            left + right, (left - right).abs(), left * right,
            relative_embedding, role_embedding,
        ], dim=-1))

    def sparse(self, hidden, pair_indices, positions, chains, roles):
        projected = self.project(hidden)
        row, i, j = pair_indices.unbind(-1)
        return self.pairs(projected[row, i], projected[row, j], positions[row, i],
                          positions[row, j], chains[row, i], chains[row, j],
                          roles[row, i], roles[row, j])

    def attention_bias(self, hidden, positions, chains, roles, valid, output):
        if self.config.bias_implementation == "symmetric":
            return self._symmetric_attention_bias(hidden, positions, chains, roles, valid, output)
        projected = self.project(hidden)
        length = hidden.shape[1]
        chunks = []
        for start in range(0, length, self.config.query_chunk):
            stop = min(start + self.config.query_chunk, length)

            def chunk(left, right, pi, pj, ci, cj, ri, rj):
                # Only [B, query_chunk, S, width] pair intermediates are materialized.
                return output(self.pairs(left, right, pi, pj, ci, cj, ri, rj))

            args = (projected[:, start:stop, None], projected[:, None, :],
                    positions[:, start:stop, None], positions[:, None, :],
                    chains[:, start:stop, None], chains[:, None, :],
                    roles[:, start:stop, None], roles[:, None, :])
            if self.training and torch.is_grad_enabled() and self.config.checkpoint_pair_chunks:
                value = checkpoint(chunk, *args, use_reentrant=False)
            else:
                value = chunk(*args)
            allowed = valid[:, start:stop, None] & valid[:, None, :]
            chunks.append(value.masked_fill(~allowed[..., None], 0).permute(0, 3, 1, 2))
        return torch.cat(chunks, dim=2)

    def _symmetric_attention_bias(self, hidden, positions, chains, roles, valid, output):
        """Compute unordered pairs once, then mirror; preserve both gradients.

        This relies on sum/abs-difference/product, absolute relative position and
        symmetric role IDs in pairs(). Directed features require another backend.
        No teacher data or supervision masks enter this path.
        """
        projected = self.project(hidden)
        length = hidden.shape[1]
        i, j = torch.triu_indices(length, length, device=hidden.device).unbind(0)
        mapping = torch.empty((length, length), device=hidden.device, dtype=torch.long)
        ordered = torch.arange(len(i), device=hidden.device)
        mapping[i, j] = ordered
        mapping[j, i] = ordered
        chunks = []
        for start in range(0, len(i), self.config.symmetric_pair_chunk):
            def chunk(projected, positions, chains, roles, ci, cj):
                return output(self.pairs(projected[:, ci], projected[:, cj], positions[:, ci],
                    positions[:, cj], chains[:, ci], chains[:, cj], roles[:, ci], roles[:, cj]))
            stop = start + self.config.symmetric_pair_chunk
            args = (projected, positions, chains, roles, i[start:stop], j[start:stop])
            if self.training and torch.is_grad_enabled() and self.config.checkpoint_pair_chunks:
                values = checkpoint(chunk, *args, use_reentrant=False)
            else:
                values = chunk(*args)
            chunks.append(values)
        values = torch.cat(chunks, dim=1)
        bias = values[:, mapping.reshape(-1)].reshape(hidden.shape[0], length, length, -1)
        allowed = valid[:, :, None] & valid[:, None, :]
        return bias.masked_fill(~allowed[..., None], 0).permute(0, 3, 1, 2).contiguous()


class StructureMLMModel(FullTokenMLMModel):
    """Sequence student with shared or independent per-layer pair encoders."""

    def __init__(self, config, pair_config=None):
        super().__init__(config)
        self.pair_config = pair_config or PairConfig()
        if self.pair_config.independent_layers:
            self.layer_pair_encoders = nn.ModuleList([
                ResiduePairEncoder(config.d_model, self.pair_config)
                for _ in range(config.n_layers)
            ])
        else:
            self.pair_encoder = ResiduePairEncoder(config.d_model, self.pair_config)
        self.pair_bias_outputs = nn.ModuleList([
            nn.Linear(self.pair_config.width, config.n_heads, bias=False)
            for _ in range(config.n_layers)
        ])
        for output in self.pair_bias_outputs:
            nn.init.zeros_(output.weight)
        self.distogram_head = nn.Linear(self.pair_config.width, self.pair_config.distogram_bins)
        self.parent_provenance = None
        self.teacher_contract = None

    def uses_feedback(self, layer):
        return self.pair_config.feedback == "all" or (
            self.pair_config.feedback == "global" and layer % 2 == 1)

    def encoder_for_layer(self, layer):
        return (self.layer_pair_encoders[layer] if self.pair_config.independent_layers
                else self.pair_encoder)

    def denoise(self, input_ids, attention_mask, role_ids, scope_ids, chain_ids,
                position_ids, target_slot_mask, *, output_hidden_states=False):
        self._validate_target_features(role_ids, chain_ids, position_ids, target_slot_mask)
        if not attention_mask.bool().any(1).all():
            raise ValueError("Each record needs at least one attended token")
        hidden = self.backbone.transformer.emb_drop(
            self.backbone.transformer.wte(input_ids) + self.role_embeddings(role_ids)
            + self.scope_embeddings(scope_ids))
        biases = self._build_attention_biases(chain_ids, position_ids, attention_mask)
        # This is input-derived, independent of clean residues, noise_kind or KD eligibility.
        pair_valid = attention_mask.bool() & chain_ids.ge(0) & position_ids.ge(1)
        for layer, block in enumerate(self.backbone.transformer.blocks):
            bias = biases[self.mask_key(layer)]
            if self.uses_feedback(layer):
                dynamic = self.encoder_for_layer(layer).attention_bias(
                    hidden, position_ids, chain_ids, role_ids, pair_valid,
                    self.pair_bias_outputs[layer])
                # Preserve the original hard chain/padding topology after adding finite biases.
                hard_mask = ~self.layer_allowed(layer, chain_ids, position_ids, attention_mask)
                bias = (bias + dynamic.to(bias.dtype)).masked_fill(
                    hard_mask[:, None], torch.finfo(bias.dtype).min)
            if self.training and self.backbone.activation_checkpointing_strategy is not None:
                hidden = self.backbone._activation_checkpoint_fn(block, hidden, bias, position_ids)
            else:
                hidden = block(hidden, bias, position_ids)
        hidden = self.backbone.transformer.ln_f(hidden)
        logits = self.backbone.transformer.ff_out(hidden)
        logits = logits.masked_fill(~self.allowed_target[None, None], float("-inf"))
        return logits, hidden if output_hidden_states else None

    def forward(self, pair_indices=None, **batch):
        logits, hidden = self.denoise(**self.feature_args(batch), output_hidden_states=True)
        loss_sum = selected_loss(logits, batch["labels"])
        if pair_indices is None:
            pair_indices = torch.empty((0, 3), dtype=torch.long, device=hidden.device)
        if pair_indices.ndim != 2 or pair_indices.shape[1] != 3 or pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be int64 [N, batch/i/j]")
        if pair_indices.numel() and (pair_indices.min() < 0
                or pair_indices[:, 0].max() >= hidden.shape[0]
                or pair_indices[:, 1:].max() >= hidden.shape[1]):
            raise ValueError("Pair index out of bounds")
        # The last layer's encoder also reads out the final hidden state. Chunk
        # full-matrix supervision without keeping every MLP activation alive.
        encoder = self.encoder_for_layer(self.config.n_layers - 1)
        def readout(hidden, indices, positions, chains, roles):
            return self.distogram_head(encoder.sparse(hidden, indices, positions, chains, roles))
        chunks = []
        chunk_size = self.pair_config.distogram_chunk_size or max(1, len(pair_indices))
        for start in range(0, max(1, len(pair_indices)), chunk_size):
            args = (hidden, pair_indices[start:start+chunk_size],
                    batch["position_ids"], batch["chain_ids"], batch["role_ids"])
            if (self.training and torch.is_grad_enabled() and self.pair_config.checkpoint_pair_chunks
                    and self.pair_config.distogram_chunk_size):
                chunks.append(checkpoint(readout, *args, use_reentrant=False))
            else:
                chunks.append(readout(*args))
        return {"loss_sum": loss_sum, "mlm_count": batch["labels"].ne(-100).sum(),
                "logits": logits, "hidden_states": hidden,
                "pair_logits": torch.cat(chunks)}

    @classmethod
    def from_bert(cls, directory, pair_config=None):
        parent = FullTokenMLMModel.from_pretrained(directory)  # strict original loader
        model = cls(parent.config, pair_config)
        missing, unexpected = model.load_state_dict(parent.state_dict(), strict=False)
        expected = {name for name in model.state_dict() if name.startswith(
            ("pair_encoder.", "layer_pair_encoders.", "pair_bias_outputs.", "distogram_head."))}
        if set(missing) != expected or unexpected:
            raise ValueError("Unexpected parent parameter coverage")
        if any(not torch.equal(value, model.state_dict()[name])
               for name, value in parent.state_dict().items()):
            raise ValueError("Inherited BERT tensors differ")
        model.parent_provenance = {
            "checkpoint": str(Path(directory).resolve()),
            "model_sha256": file_sha256(Path(directory) / "model.safetensors"),
            "inherited_tensors": len(parent.state_dict()),
            "inherited_parameters": sum(p.numel() for p in parent.parameters()),
            "new_parameters": sum(p.numel() for n, p in model.named_parameters() if n in expected),
        }
        return model

    def save_research_config(self, directory):
        settings = dict(schema=SCHEMA, variant="fixed_chain_structure_bert_mlm",
                        options=self.options, base=asdict(self.config), recipe=RECIPE,
                        pair=asdict(self.pair_config), parent=self.parent_provenance,
                        teacher_contract=self.teacher_contract)
        (Path(directory) / "research_config.json").write_text(json.dumps(settings, indent=2) + "\n")

    @classmethod
    def from_pretrained(cls, directory):
        from safetensors.torch import load_file
        directory = Path(directory)
        settings = json.loads((directory / "research_config.json").read_text())
        if (settings.get("schema") != SCHEMA or settings.get("recipe") != RECIPE
                or settings.get("variant") != "fixed_chain_structure_bert_mlm"
                or settings.get("options") != {"residual_scale": .1, "segment_global": True}
                or settings["base"] != json.loads((directory / "immune_config.json").read_text())):
            raise ValueError("Structure BERT checkpoint contract differs")
        model = cls(FixedChainConfig(**settings["base"]), PairConfig(**settings["pair"]))
        vocab = [model.tokenizer.token(i) for i in range(model.tokenizer.vocab_size)]
        if json.loads((directory / "vocabulary.json").read_text()) != vocab:
            raise ValueError("Structure BERT vocabulary differs")
        model.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True)
        model.parent_provenance = settings["parent"]
        model.teacher_contract = settings["teacher_contract"]
        return model
