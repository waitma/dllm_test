"""Test native HF old-model checkpoint recovery into the where model on CPU.

Activate conda pllm, hide CUDA and use the pinned protenix_abtcr Python:
python -B /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_v7_where_ddp_migration_cpu.py
Uses fabricated integer inputs and tiny models; all test checkpoints are removed.
This checks native single-process recovery, not actual world8 GPU admission.
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import test_v7_fsdp as original

ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
NEW = ROOT/'output/v7_formal_eight_gpu_20261002/where_continuation_20261006/source_snapshot_v1'
replacement = original.load('dllm.pipelines.immune_llada.research_refined._where_ddp_new_model',
                            NEW/'dllm/pipelines/immune_llada/research_refined/model.py')
transition = original.load('_where_ddp_transition', NEW/'scripts/research/v7_where_transition.py')
torch = original.torch


class FastModel(original.adapter.ForwardMetricModel):
    denoise = replacement.RefinedModel.denoise


def run():
    transition.model_expression_only(original.BASE/transition.MODEL_REL, NEW/transition.MODEL_REL)
    with TemporaryDirectory(prefix='v7_native_where_migration_') as directory:
        rows = [original.entry.synthetic_row(original.model().tokenizer, 516, i) for i in range(3)]
        collate = lambda values: {k: torch.stack([v[k] for v in values]) for k in values[0]}

        def trainer(name, fast=False):
            model = original.model()
            if fast:
                model.__class__ = FastModel
            args = original.TrainingArguments(output_dir=str(Path(directory)/name), use_cpu=True,
                report_to='none', per_device_train_batch_size=1, per_device_eval_batch_size=2,
                remove_unused_columns=False, label_names=['labels'], disable_tqdm=True,
                dataloader_drop_last=True, max_steps=4, warmup_steps=1, eval_strategy='steps',
                eval_steps=2, save_steps=2, seed=42, data_seed=42)
            value = original.adapter.FSDPTrainer(model=model, args=args, train_dataset=rows,
                eval_dataset={'numeric': rows}, data_collator=collate)
            value.run_provenance = {'synthetic_cpu_migration': True}
            value.log = lambda *a, **k: None
            return value

        class StopAtTwo(original.TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 2:
                    control.should_training_stop = control.should_save = control.should_evaluate = True
                return control

        old = trainer('old'); old.add_callback(StopAtTwo()); old.train()
        checkpoint = Path(directory)/'old/checkpoint-2'
        old_files = {p.name: transition.sha(p) for p in checkpoint.iterdir()}
        migrated = trainer('new', fast=True)
        migrated.train(resume_from_checkpoint=str(checkpoint))
        assert migrated.native_optimizer_restore['step'] == 2 and migrated.native_rng_restore
        assert migrated.state.global_step == 4 and migrated.lr_scheduler.last_epoch == 4
        assert {p.name: transition.sha(p) for p in checkpoint.iterdir()} == old_files
        continuous = trainer('continuous'); continuous.train()
        maximum = 0.
        for name, value in continuous.model.state_dict().items():
            other = migrated.model.state_dict()[name]
            torch.testing.assert_close(other, value, rtol=1e-5, atol=1e-7)
            maximum = max(maximum, float((other-value).abs().max()))
        assert all(int(state['step']) == 4 for state in migrated.optimizer.state_dict()['state'].values())
        first_parameters = torch.load(checkpoint/'optimizer.pt', map_location='cpu', weights_only=False)
        assert all(int(state['step']) == 2 for state in first_parameters['state'].values())
    print(json.dumps(dict(status='passed', scope='native_single_CPU_old_to_where_restore',
        maximum_parameter_error_vs_continuous_old=maximum, old_checkpoint_unchanged=True,
        optimizer_scheduler_rng_restored=True, final_step=4, data_read=False, cuda_used=False,
        temporary_checkpoints_removed=True)))


if __name__ == '__main__':
    run()
