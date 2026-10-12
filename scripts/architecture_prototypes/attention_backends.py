"""Pure random-tensor CPU attention checks; never imports project data/models.

Activate pllm; run:
python scripts/architecture_prototypes/attention_backends.py
pytest -q scripts/tests/immune_llada/test_attention_backends_cpu.py
This is an undeployed mathematical prototype, not a GPU performance result.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention
from torch.utils.checkpoint import checkpoint


def allowed_mask(groups, valid, mode, radius=2):
    """Integer group IDs are random metadata, with negative IDs for a prefix."""
    length = groups.shape[1]
    allowed = valid[:, None, :].expand(-1, length, -1)
    if mode in {"chain", "local"}:
        allowed = allowed & (groups[:, :, None] == groups[:, None, :])
        if mode == "local":
            positions = torch.arange(length, device=groups.device)
            distance = (positions[:, None] - positions[None, :]).abs()
            allowed = allowed & ((distance <= radius) | (groups[:, :, None] < 0))
        allowed = allowed | (~valid[:, :, None] & valid[:, None, :])
    if not allowed.any(-1).all():
        raise ValueError("Every query needs at least one allowed key")
    return allowed


def relation(groups):
    return ((groups[:, :, None] == groups[:, None, :]) &
            (groups[:, :, None] >= 0) & (groups[:, None, :] >= 0)).long()


def effective_bias(parameter, groups, allowed, kind):
    bias = parameter[relation(groups)].permute(0, 3, 1, 2) if kind == "table" else parameter
    return bias.masked_fill(~allowed[:, None], torch.finfo(bias.dtype).min)


def math_attention(q, k, v, bias):
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0., is_causal=False)


def chunked_attention(q, k, v, bias, chunk_size=5, recompute=False):
    outputs = []
    for start in range(0, q.shape[-2], chunk_size):
        stop = start + chunk_size
        inputs = (q[:, :, start:stop], k, v, bias[:, :, start:stop])
        outputs.append(checkpoint(math_attention, *inputs, use_reentrant=False)
                       if recompute else math_attention(*inputs))
    return torch.cat(outputs, dim=-2)


def flex_arbitrary_attention(q, k, v, parameter, groups, allowed, kind):
    # This eager CPU path verifies captured-parameter gradients. It is not the
    # compiled CUDA/Triton kernel and does not demonstrate GPU memory savings.
    minimum = torch.finfo(q.dtype).min
    if kind == "table":
        other_values, same_values = parameter[0], parameter[1]
        def score_mod(score, b, h, i, j):
            same = ((groups[b, i] == groups[b, j]) & (groups[b, i] >= 0)
                    & (groups[b, j] >= 0))
            # Avoid data-dependent scalar indexing in Dynamo's CPU trace.
            value = torch.where(same, same_values[h], other_values[h])
            return score + torch.where(allowed[b, i, j], value, minimum)
    else:
        def score_mod(score, b, h, i, j):
            return score + torch.where(allowed[b, i, j], parameter[b, h, i, j], minimum)
    return flex_attention(q, k, v, score_mod=score_mod)


def sample_inputs(kind="table", padding=True, dtype=torch.float32):
    generator = torch.Generator(device="cpu").manual_seed(8421)
    shape = (2, 3, 17, 8)
    q, k, v = [torch.randn(shape, generator=generator, dtype=dtype, device="cpu") for _ in range(3)]
    parameter = torch.randn((2, 3) if kind == "table" else (2, 3, 17, 17),
                            generator=generator, dtype=dtype, device="cpu") * .2
    groups = torch.tensor([[-1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                           [-1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]])
    valid = torch.ones(2, 17, dtype=torch.bool, device="cpu")
    if padding:
        valid[0, -3:] = False
        valid[1, -1:] = False
    weights = torch.randn(shape, generator=generator, dtype=dtype, device="cpu")
    return q, k, v, parameter, groups, valid, weights


def compare(method, kind="table", mode="global", padding=True, dtype=torch.float32):
    if method == "flex_eager":
        # Isolate static-shape CPU proofs. In this 2.8 build the eager wrapper
        # raises a Dynamo scalar-index error when different closures share its
        # compilation cache. Resetting is a TEST isolation, not a deployment fix.
        torch._dynamo.reset()
    values = sample_inputs(kind, padding, dtype)
    groups, valid, weights = values[-3:]
    allowed = allowed_mask(groups, valid, mode)
    results = []
    for implementation in ("math", method):
        q, k, v, parameter = [t.detach().clone().requires_grad_() for t in values[:4]]
        bias = effective_bias(parameter, groups, allowed, kind)
        if implementation == "math":
            out = math_attention(q, k, v, bias)
        elif implementation in {"chunked", "checkpoint_chunked"}:
            out = chunked_attention(q, k, v, bias, recompute=implementation == "checkpoint_chunked")
        elif implementation == "flex_eager":
            out = flex_arbitrary_attention(q, k, v, parameter, groups, allowed, kind)
        else:
            raise ValueError(implementation)
        gradients = torch.autograd.grad((out * weights).sum(), (q, k, v, parameter))
        results.append((out.detach(), [g.detach() for g in gradients]))
    (expected, expected_grads), (actual, actual_grads) = results
    atol = 2e-6 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(actual, expected, atol=atol, rtol=1e-5)
    for a, e in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, e, atol=atol, rtol=1e-5)
        assert torch.isfinite(a).all()
    return {"method": method, "bias_kind": kind, "mask": mode, "padding": padding,
            "dtype": str(dtype), "maximum_output_abs_error": (actual - expected).abs().max().item(),
            "maximum_q_k_v_bias_gradient_abs_errors": [(a - e).abs().max().item() for a, e in zip(actual_grads, expected_grads)],
            "bias_gradient_l1": actual_grads[-1].abs().sum().item(), "passed": True}


def cpu_dispatch_observation(trainable_bias):
    q, k, v, parameter, groups, valid, _ = sample_inputs("dense")
    q, k, v = [t.requires_grad_() for t in (q, k, v)]
    parameter.requires_grad_(trainable_bias)
    bias = effective_bias(parameter, groups, allowed_mask(groups, valid, "global"), "dense")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out.square().sum().backward()
    operators = sorted({event.key for event in profiler.key_averages()
                        if "scaled_dot_product" in event.key})
    return {"device": "cpu", "trainable_bias": trainable_bias, "operators": operators,
            "bias_gradient_exists": parameter.grad is not None}


def main():
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[2]
    out = root / "output/immune_attention_backend_review_20260930"
    out.mkdir(parents=True, exist_ok=True)
    rows = [compare(method, kind, mode, padding) for method in ("chunked", "checkpoint_chunked", "flex_eager")
            for kind in ("dense", "table") for mode in ("global", "chain", "local") for padding in (False, True)]
    report = {"status": "cpu_equivalence_passed_not_gpu_benchmark", "created_at": datetime.now(timezone.utc).isoformat(),
              "torch_version": torch.__version__, "device": "cpu", "random_seed": 8421,
              "dropout_p": 0., "shape": [2, 3, 17, 8], "cases": len(rows), "rows": rows,
              "cpu_dispatch": [cpu_dispatch_observation(flag) for flag in (False, True)],
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "limits": ["No GPU kernels executed or submitted.", "No project model/data/benchmark imported.",
                         "CPU Flex uses the eager higher-order implementation, not compiled CUDA kernels.",
                         "Each Flex case resets Dynamo; mixed-closure reuse failed in the initial probe.",
                         "No GPU speed or peak memory conclusions follow from this test."]}
    (out / "cpu_equivalence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases": len(rows), "status": report["status"], "cpu_dispatch": report["cpu_dispatch"]}))


if __name__ == "__main__":
    main()
