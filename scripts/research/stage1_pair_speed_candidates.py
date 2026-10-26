"""RAM-only equivalent Pair implementations for the speed exploration.

Activate pllm; use the absolute benchmark_stage1_pair_speed.py entrypoint.
Never patches a frozen training package or changes saved parameter names.
"""
import types

import torch
from torch.utils.checkpoint import checkpoint


def autocast_pairs(self, left, right, position_left, position_right,
                   chain_left, chain_right, role_left, role_right):
    """Cast embeddings before cat; the first autocast Linear casts them anyway."""
    same = chain_left.eq(chain_right)
    relative = (position_left-position_right).abs().clamp_max(self.config.relative_clip)
    relative = torch.where(same, relative, self.config.relative_clip+1)
    roles = (torch.minimum(role_left, role_right)*self.num_roles
             + torch.maximum(role_left, role_right)
             + (~same).long()*self.num_roles*self.num_roles)
    # Without autocast retain the original promotion semantics.
    dtype = torch.get_autocast_dtype(left.device.type) if torch.is_autocast_enabled(left.device.type) else None
    relative_embedding, role_embedding = self.relative(relative), self.roles(roles)
    if dtype is not None:
        relative_embedding, role_embedding = relative_embedding.to(dtype), role_embedding.to(dtype)
    return self.pair_mlp(torch.cat([
        left+right, (left-right).abs(), left*right, relative_embedding, role_embedding], dim=-1))


def install_cast(model):
    model.pair_encoder.pairs = types.MethodType(autocast_pairs, model.pair_encoder)


def install_compile(model, cast=False, precise=False):
    if cast:
        install_cast(model)
    options = {"emulate_precision_casts": True} if precise else {}
    model.pair_encoder.pairs = torch.compile(model.pair_encoder.pairs, dynamic=False, options=options)


def symmetric_attention_bias(self, hidden, positions, chains, roles, valid, output):
    """Evaluate unordered pairs once; original Pair features are exactly symmetric."""
    projected = self.project(hidden)
    length = hidden.shape[1]
    upper = torch.triu_indices(length, length, device=hidden.device)
    i, j = upper.unbind(0)
    mapping = torch.empty((length, length), device=hidden.device, dtype=torch.long)
    ordered = torch.arange(len(i), device=hidden.device)
    mapping[i, j] = ordered
    mapping[j, i] = ordered
    chunks = []
    for start in range(0, len(i), 4096):
        def chunk(projected, positions, chains, roles, ci, cj):
            return output(self.pairs(projected[:, ci], projected[:, cj], positions[:, ci],
                positions[:, cj], chains[:, ci], chains[:, cj], roles[:, ci], roles[:, cj]))
        args = (projected, positions, chains, roles, i[start:start+4096], j[start:start+4096])
        if self.training and torch.is_grad_enabled() and self.config.checkpoint_pair_chunks:
            values = checkpoint(chunk, *args, use_reentrant=False)
        else:
            values = chunk(*args)
        chunks.append(values)
    values = torch.cat(chunks, dim=1)
    bias = values[:, mapping.reshape(-1)].reshape(hidden.shape[0], length, length, -1)
    allowed = valid[:, :, None] & valid[:, None, :]
    return bias.masked_fill(~allowed[..., None], 0).permute(0, 3, 1, 2).contiguous()


def install_symmetric(model):
    model.pair_encoder.attention_bias = types.MethodType(symmetric_attention_bias, model.pair_encoder)
