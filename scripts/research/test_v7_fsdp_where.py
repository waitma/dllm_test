"""Check the where FSDP integration using fabricated CPU tensors only.

Activate conda pllm and use the pinned protenix_abtcr interpreter:
  python -B /project_c20250601/251105016/project/dllm_test/scripts/research/test_v7_fsdp_where.py
  python -m torch.distributed.run --standalone --nproc_per_node=2 /project_c20250601/251105016/project/dllm_test/scripts/research/test_v7_fsdp_where.py --worker
No corpus or CUDA access. All temporary model/optimizer checkpoints are removed.
"""
import ast
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
sys.path.insert(0, str(ROOT/'scripts/research'))
import test_v7_fsdp as original

fast = original.load('_v7_where_test_adapter', ROOT/'scripts/research/v7_fsdp_where_trainer.py')
original.adapter = fast
torch = original.torch


def source_check():
    source = original.BASE/'dllm/pipelines/immune_llada/research_refined/model.py'
    tree = ast.parse(source.read_text())
    expected = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'denoise')
    assignments = [n for n in ast.walk(expected) if isinstance(n, ast.Assign)
                   and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id == 'base_bias']
    assert len(assignments) == 1
    assignments[0].value = ast.parse('torch.where(same[:, None], self.chain_bias[1][None, :, None, None], '
                                     'self.chain_bias[0][None, :, None, None])', mode='eval').body
    actual = next(n for n in ast.walk(ast.parse(Path(fast.__file__).read_text()))
                  if isinstance(n, ast.FunctionDef) and n.name == 'denoise')
    assert ast.dump(actual, include_attributes=False) == ast.dump(expected, include_attributes=False)


def cpu_checks():
    source_check()
    max_error = 0.
    for padding in (0, 12):
        model = original.model()
        reference = original.FactorialModel(model.config, 'f011')
        reference.load_state_dict(model.state_dict(), strict=True)
        assert dict(model.named_parameters()).keys() == dict(reference.named_parameters()).keys()
        x = original.batch(model)
        if padding:
            x['attention_mask'][:, -padding:] = False
            x['target_slot_mask'][:, -padding:] = False
            x['chain_ids'][:, -padding:] = -1
        torch.manual_seed(51); actual = model(**x)['loss']
        torch.manual_seed(51); expected = reference(**x)['loss']
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.backward(); expected.backward()
        for (name, left), (other, right) in zip(model.named_parameters(), reference.named_parameters()):
            assert name == other and (left.grad is None) == (right.grad is None)
            if left.grad is not None:
                if name == 'chain_bias':
                    torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7)
                else:
                    torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
                max_error = max(max_error, float((left.grad-right.grad).abs().max()))
        model.eval(); reference.eval()
        for ratio in (.25, .5, .75, 1.):
            with torch.no_grad():
                a = model(validation_ratio=ratio, **x)['sample_totals']
                b = original.metrics.selected_token_metrics(reference, x, ratio)['sample_totals']
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    with TemporaryDirectory(prefix='v7_fsdp_where_resume_') as directory:
        rows = [original.entry.synthetic_row(original.model().tokenizer, 516, i) for i in range(3)]
        collate = lambda values: {k: torch.stack([v[k] for v in values]) for k in values[0]}

        def trainer(name):
            args = original.TrainingArguments(output_dir=str(Path(directory)/name), use_cpu=True,
                report_to='none', per_device_train_batch_size=1, per_device_eval_batch_size=2,
                remove_unused_columns=False, label_names=['labels'], disable_tqdm=True,
                dataloader_drop_last=True, max_steps=4, warmup_steps=1, eval_strategy='steps',
                eval_steps=2, save_steps=2, seed=42, data_seed=42)
            value = fast.FSDPTrainer(model=original.model(), args=args, train_dataset=rows,
                eval_dataset={'numeric': rows}, data_collator=collate)
            value.run_provenance = {'cpu_where_test': True}
            value.log = lambda *a, **k: None
            return value

        class StopAtTwo(original.TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 2:
                    control.should_training_stop = control.should_save = control.should_evaluate = True
                return control

        first = trainer('split'); first.add_callback(StopAtTwo()); first.train()
        resumed = trainer('split')
        resumed.train(resume_from_checkpoint=str(Path(directory)/'split/checkpoint-2'))
        continuous = trainer('continuous'); continuous.train()
        assert resumed.native_rng_restore and resumed.native_optimizer_restore['step'] == 2
        for name, value in continuous.model.state_dict().items():
            torch.testing.assert_close(resumed.model.state_dict()[name], value, rtol=0, atol=0)
        metrics = resumed.evaluate()
        assert metrics['eval_records_per_mask'] == 3 and metrics['eval_mask100_numeric_terminal_count'] == 6
        reloaded = fast.ForwardMetricModel.from_pretrained(str(Path(directory)/'split/checkpoint-4'))
        for name, value in resumed.model.state_dict().items():
            torch.testing.assert_close(reloaded.state_dict()[name], value, rtol=0, atol=0)
    print(json.dumps(dict(status='passed', maximum_gradient_error=max_error,
        checks=['single_expression_AST_change', 'padded_unpadded_loss_and_all_gradients',
                'four_mask_exact_metric_parity', 'native_trainer_resume_bitwise_equals_continuous',
                'native_optimizer_scheduler_rng_restore', 'three_record_eval_tail', 'model_reload'],
        cuda_used=False, corpus_read=False, temporary_checkpoints_removed=True)))


if __name__ == '__main__':
    if '--worker' in sys.argv:
        source_check()
        original.worker_checks()
    else:
        cpu_checks()
