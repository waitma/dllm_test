"""CPU-only, undeployed attention-bias cache prototype for a later release.

After activating pllm, run the measurable allocation-count experiment:
USE_TF=0 python scripts/architecture_prototypes/bias_cache.py --output /tmp/bias-cache-counts.json
Tests: USE_TF=0 python -m pytest scripts/tests/immune_llada/test_architecture_bias_cache.py -q
Only explicitly bound model instances change; no live model, runner or class is patched.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import MethodType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch


def mask_key(model, layer):
    """Classify exactly the current ArchitectureModel.layer_allowed schedules."""
    if 'local_radius' in model.options and layer % 2 == 0:
        return ('local', model.options['local_radius'])
    if model.options.get('chain_late_fusion') and layer < model.config.n_layers//2:
        return ('chain',)
    return ('global',)


def denoise_with_bias_cache(self, input_ids, attention_mask, role_ids, scope_ids,
                            chain_ids, position_ids, target_slot_mask, *, output_hidden_states=False):
    """Reuse immutable attention biases only within this particular forward call.

    Cache values stay connected to chain_bias for autograd. They are passed as
    explicit checkpoint inputs; neither this dict nor any tensors enter model state.
    Casting inside ArchitectureBlock remains unchanged and is not cached here.
    """
    valid = attention_mask.bool()
    if not valid.any(1).all():
        raise ValueError('Each record needs at least one attended token')
    globally_unmasked = bool(valid.all())
    x = (self.backbone.transformer.wte(input_ids)+self.role_embeddings(role_ids)+
         self.scope_embeddings(scope_ids))
    x = self.backbone.transformer.emb_drop(x)
    same = ((chain_ids[:, :, None] == chain_ids[:, None, :]) &
            (chain_ids[:, :, None] >= 0) & (chain_ids[:, None, :] >= 0))
    base_bias = self.chain_bias[same.long()].permute(0, 3, 1, 2).contiguous()
    biases = {}
    first_values = None
    for index, block in enumerate(self.backbone.transformer.blocks):
        key = mask_key(self, index)
        if key not in biases:
            if key == ('global',) and globally_unmasked:
                biases[key] = base_bias
            else:
                allowed = self.layer_allowed(index, chain_ids, position_ids, attention_mask)
                biases[key] = base_bias.masked_fill(~allowed[:, None], torch.finfo(base_bias.dtype).min)
        bias = biases[key]
        if self.backbone.activation_checkpointing_strategy is not None and self.training:
            x, values = self.backbone._activation_checkpoint_fn(block, x, bias, position_ids, first_values)
        else:
            x, values = block(x, bias, position_ids, first_values)
        if index == 0 and self.options.get('value_residual'):
            first_values = values
    x = self.backbone.transformer.ln_f(x)
    logits = self.backbone.transformer.ff_out(x)
    logits = logits.masked_fill(target_slot_mask[:, :, None] & ~self.allowed_target[None, None], float('-inf'))
    return logits, x if output_hidden_states else None


def bind_bias_cache(model):
    """Opt in a single disposable model instance, without changing its class."""
    model.denoise = MethodType(denoise_with_bias_cache, model)
    return model


def count_bias_allocations(model, batch):
    """Count actual CPU masked_fill operations and distinct block-input storages.

    This observes allocation count/logical output bytes, not CUDA peak memory or
    process RSS. Strong references retain hook inputs for reliable storage counting;
    they intentionally make this unsuitable as an overall memory benchmark.
    """
    from torch.utils._python_dispatch import TorchDispatchMode

    if batch['input_ids'].device.type != 'cpu':
        raise ValueError('The allocation-count experiment is deliberately CPU-only')
    shape = (batch['input_ids'].shape[0], model.config.n_heads,
             batch['input_ids'].shape[1], batch['input_ids'].shape[1])

    class Counter(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.count = 0
            self.output_bytes = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if func == torch.ops.aten.masked_fill.Scalar and tuple(args[0].shape) == shape:
                self.count += 1
                self.output_bytes += result.numel()*result.element_size()
            return result

    inputs = []
    def record(module, args):
        inputs.append(args[1])
    handles = [block.register_forward_pre_hook(record) for block in model.backbone.transformer.blocks]
    counter = Counter()
    try:
        with counter:
            logits, hidden = model.denoise(**model.feature_args(batch), output_hidden_states=True)
    finally:
        for handle in handles:
            handle.remove()
    stats = dict(bias_shape=list(shape), masked_fill_count=counter.count,
                 masked_fill_output_bytes_total=counter.output_bytes,
                 block_invocations=len(inputs), unique_block_bias_storages=len({
                     bias.untyped_storage().data_ptr() for bias in inputs}),
                 bias_dtype=str(inputs[0].dtype))
    return stats, logits.detach(), hidden.detach()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from dllm.pipelines.immune_llada.research_next.model import ArchitectureModel, VARIANTS
    from dllm.pipelines.immune_llada.v3.model import ImmuneConfig
    from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy
    from scripts.run_immune_architecture import fingerprints
    from scripts.run_immune_research import collator_for, toy_rows

    torch.set_num_threads(1)
    sources = fingerprints()
    config = ImmuneConfig(ablation='A4', d_model=32, n_layers=8, n_heads=4,
                          mlp_hidden_size=64, max_length=256, full_capacity=24, fragment_capacity=16)
    collate = collator_for(config)
    rows = toy_rows()
    reports = []
    for variant in sorted(VARIANTS):
        for padded in (False, True):
            torch.manual_seed(23)
            original = ArchitectureModel(config, variant).train()
            original.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
            cached = bind_bias_cache(copy.deepcopy(original))
            batch = collate([rows[4], rows[5]] if padded else [rows[0], rows[0]])
            assert bool((~batch['attention_mask']).any()) == padded
            before, logits_before, hidden_before = count_bias_allocations(original, batch)
            after, logits_after, hidden_after = count_bias_allocations(cached, batch)
            if not torch.equal(logits_before, logits_after) or not torch.equal(hidden_before, hidden_after):
                raise AssertionError(f'Numerical mismatch for {variant}, padding={padded}')
            reports.append(dict(variant=variant, padded=padded, original=before, cached=after,
                                logits_exact=True, hidden_exact=True))
    if sources != fingerprints():
        raise RuntimeError('Live sources changed during CPU prototype validation')
    result = dict(kind='cpu_operation_and_storage_count', gpu_memory_measurement=False,
                  prototype_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_sha256=sources, torch_version=torch.__version__, measurements=reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output.resolve()), cases=len(reports),
                         live_sources_unchanged=True, gpu_memory_measurement=False)))


if __name__ == '__main__':
    main()
