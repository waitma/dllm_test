"""Focused CPU loader controls; no model forward, training or process group.

Run only these new cases after activation:
  source /project_c20250601/251105016/miniforge3/etc/profile.d/conda.sh
  conda activate pllm
  CUDA_VISIBLE_DEVICES='' USE_TF=0 python -B /project_c20250601/251105016/project/dllm_test/scripts/research/test_v7_formal_eval_tail.py --environment pllm --evidence-dir /project_c20250601/251105016/project/dllm_test/output/v7_formal_eval_tail_20261002/pllm

For worker, use /project_c20250601/251105016/conda/envs/protenix_abtcr/bin/python
with --environment worker and its own evidence directory. Eight real sampler
objects run as logical ranks in one CPU process. Concatenation replaces only
the network gather; the actual Accelerator loader remainder and metric-tail
trimming are exercised. This is not an eight-process or GPU acceptance test.
"""
import argparse
import ast
from collections import deque
import copy
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path('/project_c20250601/251105016/project/dllm_test')
DRIVER = ROOT / 'scripts/research/run_v7_formal.py'
PYTHONS = {'pllm': '/project_c20250601/251105016/conda/envs/pllm/bin/python',
           'worker': '/project_c20250601/251105016/conda/envs/protenix_abtcr/bin/python'}
OBSERVED = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def actual_override_class():
    from transformers import Trainer
    tree = ast.parse(DRIVER.read_text())
    original = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
                    and node.name == 'FormalBaseTrainer')
    method = next(node for node in original.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'get_eval_dataloader')
    # Compile the exact production method against its ultimate HF loader base.
    # The intermediate production classes do not override get_eval_dataloader.
    wrapper = ast.ClassDef(name='ActualEvalTailOverride', bases=[ast.Name(id='Trainer', ctx=ast.Load())],
                           keywords=[], body=[copy.deepcopy(method)], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
    namespace = {'Trainer': Trainer}
    exec(compile(module, str(DRIVER), 'exec'), namespace)
    OBSERVED['override_ast_sha256'] = hashlib.sha256(ast.dump(method).encode()).hexdigest()
    return namespace['ActualEvalTailOverride']


def integer_collator(rows):
    import torch
    return torch.tensor(rows, dtype=torch.long)


def make_trainer(directory, eval_dataset):
    import torch
    from transformers import TrainingArguments
    cls = actual_override_class()
    args = TrainingArguments(output_dir=str(directory), use_cpu=True,
        per_device_train_batch_size=4, per_device_eval_batch_size=4,
        dataloader_drop_last=True, dataloader_num_workers=0, dataloader_pin_memory=False,
        remove_unused_columns=False, report_to='none', disable_tqdm=True)
    return cls(model=torch.nn.Linear(1, 1), args=args, train_dataset=list(range(33)),
               eval_dataset=eval_dataset, data_collator=integer_collator)


class EvalTailControls(unittest.TestCase):
    def test_real_loader_shards_and_metric_tail_cover_integer_ids(self):
        import torch
        from accelerate.data_loader import BatchSamplerShard, DataLoaderShard
        receipts = []
        with TemporaryDirectory(prefix='formal-eval-tail-cpu-') as temporary:
            directory = Path(temporary)
            trainer = make_trainer(directory, list(range(13)))
            self.assertEqual(trainer.accelerator.num_processes, 1)
            for count in (13, 17):
                dataset = list(range(count))
                loader = trainer.get_eval_dataloader(dataset)
                self.assertIs(trainer.args.dataloader_drop_last, True)
                self.assertIs(loader.batch_sampler.drop_last, False)
                self.assertEqual(torch.cat(list(loader)).tolist(), dataset)
                sharded = [DataLoaderShard(dataset, device=torch.device('cpu'),
                    batch_sampler=BatchSamplerShard(loader.batch_sampler, num_processes=8,
                        process_index=rank, split_batches=False, even_batches=True),
                    collate_fn=integer_collator, num_workers=0, _drop_last=False)
                    for rank in range(8)]
                iterators = [iter(item) for item in sharded]
                per_rank, retained, steps = [[] for _ in range(8)], [], []
                try:
                    for batches in zip(*iterators):
                        for rank, batch in enumerate(batches):
                            per_rank[rank].append(batch.tolist())
                        # Actual eight shard outputs, assembled in rank order;
                        # no fictional num_samples and no emulated network call.
                        combined = torch.cat(batches)
                        gradient_state = trainer.accelerator.gradient_state
                        self.assertTrue(gradient_state.end_of_dataloader)
                        self.assertEqual(gradient_state.remainder, count)
                        selected = trainer.accelerator.gather_for_metrics(combined)
                        retained.extend(selected.tolist())
                        steps.append({'before_trim_ids': combined.tolist(),
                                      'actual_remainder': gradient_state.remainder,
                                      'retained_ids': selected.tolist()})
                finally:
                    # Complete every iterator so native GradientState releases
                    # its active dataloader references even if an assertion fails.
                    for iterator in iterators:
                        deque(iterator, maxlen=0)
                self.assertEqual(retained, dataset)
                self.assertEqual(len(set(retained)), count)
                self.assertEqual([len(items) for items in per_rank], [1] * 8)
                self.assertEqual(sum(1 for _ in retained), count)
                self.assertIsNone(trainer.accelerator.gradient_state.active_dataloader)
                receipts.append({'source_records': count, 'per_rank_batches': per_rank,
                                 'steps': steps, 'actual_unique_ids': sorted(set(retained)),
                                 'actual_count': len(retained), 'logical_rank_iterations': [len(x) for x in per_rank]})
            train_loader = trainer.get_train_dataloader()
            self.assertIs(trainer.args.dataloader_drop_last, True)
            self.assertIs(train_loader.batch_sampler.drop_last, True)
            training_batches = list(train_loader)
            self.assertEqual([len(batch) for batch in training_batches], [4] * 8)
            self.assertEqual(sum(map(len, training_batches)), 32)
            self.assertFalse(torch.cuda.is_initialized())
            self.assertFalse(torch.distributed.is_initialized())
            OBSERVED['coverage'] = receipts
            OBSERVED['training_loader'] = {'records': 33, 'observed_records': 32, 'drop_last': True,
                                            'batch_sizes': [len(batch) for batch in training_batches]}
        self.assertFalse(directory.exists())
        OBSERVED['temporary_output_removed'] = True

    def test_real_hf_loader_failure_restores_training_argument(self):
        with TemporaryDirectory(prefix='formal-eval-error-cpu-') as temporary:
            directory = Path(temporary)
            trainer = make_trainer(directory, None)
            with self.assertRaisesRegex(ValueError, 'requires an eval_dataset') as captured:
                trainer.get_eval_dataloader()
            self.assertIs(trainer.args.dataloader_drop_last, True)
            OBSERVED['exception_restore'] = {'type': type(captured.exception).__name__,
                'message': str(captured.exception), 'dataloader_drop_last_after': trainer.args.dataloader_drop_last}
        self.assertFalse(directory.exists())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', choices=tuple(PYTHONS), required=True)
    parser.add_argument('--evidence-dir', type=Path, required=True)
    args = parser.parse_args()
    if (sys.executable != PYTHONS[args.environment] or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or not args.evidence_dir.is_absolute()):
        raise ValueError('Expected environment executable, explicit empty CUDA mask and absolute evidence directory')
    args.evidence_dir.mkdir(parents=True, exist_ok=False)
    before = sha(DRIVER)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(EvalTailControls)
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    import accelerate
    import torch
    import transformers
    from accelerate.data_loader import BatchSamplerShard, DataLoaderShard
    sources = {str(Path(inspect.getsourcefile(obj)).resolve()): sha(inspect.getsourcefile(obj))
               for obj in (transformers.Trainer, accelerate.Accelerator, BatchSamplerShard, DataLoaderShard)}
    report = dict(at_utc=datetime.now(timezone.utc).isoformat(),
        status='passed_cpu_loader_boundary_only' if result.wasSuccessful() and sha(DRIVER) == before else 'failed',
        environment=args.environment, executable=sys.executable, tests_run=result.testsRun,
        failures=[text for _, text in result.failures], errors=[text for _, text in result.errors],
        driver_sha256=before, driver_unchanged=sha(DRIVER) == before,
        test_sha256=sha(__file__), installed_sources_sha256=sources,
        versions={'torch': torch.__version__, 'transformers': transformers.__version__, 'accelerate': accelerate.__version__},
        observed=OBSERVED, synthetic_integer_ids_only=True, model_forward=False, training=False,
        real_data_read=False, GPU_initialized=torch.cuda.is_initialized(),
        distributed_group_initialized=torch.distributed.is_initialized(),
        actual_processes=1, logical_batch_sampler_ranks=8,
        network_collective_executed=False, native_trainer_num_samples_not_used_as_coverage=True,
        limitation='Actual CPU loaders and shard/trim algorithms; no eight-process communication or GPU acceptance')
    (args.evidence_dir / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report['status'] == 'passed_cpu_loader_boundary_only' else 1


if __name__ == '__main__':
    raise SystemExit(main())
