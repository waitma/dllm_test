"""CPU-only checks for the new adapter, using fabricated tensors.

Activate pllm and use the pinned protenix_abtcr Python:
  python -B /absolute/path/test_v7_fsdp.py
  torchrun --standalone --nproc_per_node=2 /absolute/path/test_v7_fsdp.py --worker
No prepared data or CUDA access; temporary checkpoint files are removed on exit.
"""
from functools import partial
import importlib.util
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
BASE = ROOT/'output/v7_formal_eight_gpu_20261002/source_snapshot_v3'
sys.path.insert(0, str(BASE))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


entry = load('_fsdp_entry_test', ROOT/'scripts/research/run_v7_fsdp.py')
adapter = load('_fsdp_adapter_test', ROOT/'scripts/research/v7_fsdp_trainer.py')
import torch
from transformers import TrainingArguments, TrainerCallback
from dllm.pipelines.immune_llada.v3.model import ImmuneConfig
from dllm.pipelines.immune_llada.research_refined.factorial import FactorialModel
from dllm.pipelines.immune_llada.research_next.model import ArchitectureBlock
from scripts.architecture_prototypes import validation_metrics as metrics

torch.set_num_threads(1)


def model():
    torch.manual_seed(42)
    config = ImmuneConfig(d_model=32, n_layers=2, n_heads=4, mlp_hidden_size=64)
    return adapter.ForwardMetricModel(config, 'f011')


def batch(m, index=0):
    row = entry.synthetic_row(m.tokenizer, 516, index)
    return {k: v.unsqueeze(0) for k, v in row.items()}


def cpu_checks():
    m = model()
    reference = FactorialModel(m.config, 'f011')
    reference.load_state_dict(m.state_dict())
    x = batch(m)
    torch.manual_seed(51)
    a = m(**x)['loss']
    torch.manual_seed(51)
    b = reference(**x)['loss']
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.backward(); b.backward()
    for left, right in zip(m.parameters(), reference.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    m.eval(); reference.eval()
    for ratio in (.25, .5, .75, 1.):
        actual = m(validation_ratio=ratio, **x)
        expected = metrics.selected_token_metrics(reference, x, ratio)
        torch.testing.assert_close(actual['sample_totals'], expected['sample_totals'], rtol=0, atol=0)
    with TemporaryDirectory(prefix='v7_fsdp_cpu_') as directory:
        rows = [entry.synthetic_row(m.tokenizer, 516, i) for i in range(3)]
        args = TrainingArguments(output_dir=directory, use_cpu=True, report_to='none',
            per_device_eval_batch_size=2, per_device_train_batch_size=1,
            remove_unused_columns=False, label_names=['labels'], disable_tqdm=True,
            dataloader_drop_last=True, max_steps=2)
        collate = lambda values: {k: torch.stack([v[k] for v in values]) for k in values[0]}
        trainer = adapter.FSDPTrainer(model=m, args=args, train_dataset=rows,
            eval_dataset={'numeric': rows}, data_collator=collate)
        trainer.log = lambda *a, **k: None
        result = trainer.evaluate()
        assert result['eval_records_per_mask'] == 3 and result['eval_mask100_numeric_chain_count'] == 6
        assert result['eval_mask100_numeric_terminal_count'] == 6
        trainer.run_provenance = {'cpu_test': True}
        trainer.save_model(directory)
        reloaded = FactorialModel.from_pretrained(directory)
        for key, value in m.state_dict().items():
            torch.testing.assert_close(value, reloaded.state_dict()[key], rtol=0, atol=0)
    with TemporaryDirectory(prefix='v7_fsdp_native_resume_cpu_') as directory:
        rows = [entry.synthetic_row(m.tokenizer, 516, i) for i in range(3)]
        def make_trainer(name):
            args = TrainingArguments(output_dir=str(Path(directory)/name), use_cpu=True, report_to='none',
                per_device_eval_batch_size=2, per_device_train_batch_size=1, remove_unused_columns=False,
                label_names=['labels'], disable_tqdm=True, dataloader_drop_last=True, max_steps=4,
                warmup_steps=1, eval_strategy='steps', eval_steps=2, save_steps=2, seed=42, data_seed=42)
            trainer = adapter.FSDPTrainer(model=model(), args=args, train_dataset=rows,
                eval_dataset={'numeric': rows}, data_collator=collate)
            trainer.run_provenance = {'cpu_test': True}
            trainer.log = lambda *a, **k: None
            return trainer
        class StopAtTwo(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 2:
                    control.should_training_stop = control.should_save = control.should_evaluate = True
                return control
        interrupted = make_trainer('split')
        interrupted.add_callback(StopAtTwo())
        interrupted.train()
        resumed = make_trainer('split')
        resumed.train(resume_from_checkpoint=str(Path(directory)/'split/checkpoint-2'))
        assert resumed.native_rng_restore and resumed.native_optimizer_restore['step'] == 2
        continuous = make_trainer('continuous')
        continuous.train()
        for name, expected in continuous.model.state_dict().items():
            torch.testing.assert_close(resumed.model.state_dict()[name], expected, rtol=0, atol=0)
    print(json.dumps({'status': 'passed', 'checks': ['same_training_loss_and_gradients',
        'four_mask_metric_parity', 'three_record_eval_tail', 'native_model_metadata_roundtrip',
        'native_trainer_optimizer_scheduler_rng_resume_equals_continuous_cpu'],
        'cuda_used': False, 'corpus_read': False, 'temporary_checkpoints_removed': True}))


def worker_checks():
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl)
    torch.distributed.init_process_group('gloo')
    rank = torch.distributed.get_rank()
    reference, candidate = model(), model()
    policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={ArchitectureBlock})
    sharded = FSDP(candidate, auto_wrap_policy=policy, device_id=torch.device('cpu'), use_orig_params=True)
    apply_activation_checkpointing(sharded,
        checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        auto_wrap_policy=policy)
    assert len([m for m in sharded.modules() if isinstance(m, FSDP)]) == 3
    x = batch(reference, rank)
    optimizer = torch.optim.AdamW(sharded.parameters(), lr=1e-4)
    ordinary_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-4)
    torch.manual_seed(123+rank)
    ordinary_loss = reference(**x)['loss']; ordinary_loss.backward()
    for parameter in reference.parameters():
        torch.distributed.all_reduce(parameter.grad)
        parameter.grad /= 2
    ordinary_optimizer.step()
    torch.manual_seed(123+rank)
    loss = sharded(**x)['loss']; loss.backward(); optimizer.step()
    torch.testing.assert_close(loss, ordinary_loss, atol=1e-6, rtol=1e-6)
    with FSDP.state_dict_type(sharded, StateDictType.FULL_STATE_DICT,
                              FullStateDictConfig(offload_to_cpu=False, rank0_only=False)):
        weights = sharded.state_dict()
        for key, expected in reference.state_dict().items():
            torch.testing.assert_close(weights[key], expected, atol=2e-6, rtol=2e-5)
        optimizer_state = FSDP.optim_state_dict(sharded, optimizer)
    sharded.eval(); reference.eval()
    for ratio in (.25, .5, .75, 1.):
        with torch.no_grad():
            actual = sharded(validation_ratio=ratio, **x)
            expected = metrics.selected_token_metrics(reference, x, ratio)
        torch.testing.assert_close(actual['sample_totals'], expected['sample_totals'], atol=1e-4, rtol=2e-5)
    with TemporaryDirectory(prefix='v7_fsdp_distributed_cpu_') as directory:
        path = Path(directory)/'native.pt'
        torch.save({'model': weights, 'optimizer': optimizer_state}, path)
        restored = torch.load(path, weights_only=False)
        fresh = FSDP(model(), auto_wrap_policy=policy, device_id=torch.device('cpu'), use_orig_params=True)
        fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=1e-4)
        with FSDP.state_dict_type(fresh, StateDictType.FULL_STATE_DICT,
                                  FullStateDictConfig(offload_to_cpu=False, rank0_only=False)):
            fresh.load_state_dict(restored['model'])
            state = FSDP.optim_state_dict_to_load(fresh, fresh_optimizer, restored['optimizer'])
            fresh_optimizer.load_state_dict(state)
            for key, expected in weights.items():
                torch.testing.assert_close(fresh.state_dict()[key], expected, atol=0, rtol=0)
        assert fresh_optimizer.state and all(int(s['step']) == 1 for s in fresh_optimizer.state.values())
    torch.distributed.barrier()
    if rank == 0:
        print(json.dumps({'status': 'passed', 'world_size': 2, 'backend': 'gloo', 'device': 'cpu',
            'checks': ['actual_fsdp_sharding_and_checkpoint_wrapping', 'averaged_gradient_update_parity',
                       'four_mask_eval_inside_fsdp_forward', 'full_model_optimizer_roundtrip'],
            'gpu_or_multinode_acceptance': False, 'temporary_checkpoints_removed': True}))
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    worker_checks() if '--worker' in sys.argv else cpu_checks()
