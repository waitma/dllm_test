"""Run only new CPU restoration controls after activating conda pllm.

Use /project_c20250601/251105016/conda/envs/protenix_abtcr/bin/python -B
/project_c20250601/251105016/project/dllm_test/scripts/research/
test_formal_resume_observation.py. Fixtures are artificial CPU state only and
removed by TemporaryDirectory. No training, model construction, data or CUDA.
"""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
HELPER = ROOT/'scripts/research/formal_resume_observation.py'
DRIVER = ROOT/'scripts/research/run_v7_formal.py'


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import numpy as np
        from transformers import Trainer
        from accelerate.optimizer import AcceleratedOptimizer
        cls.torch, cls.np, cls.Trainer = torch, np, Trainer
        cls.AcceleratedOptimizer = AcceleratedOptimizer
        spec = importlib.util.spec_from_file_location('resume_control_helper', HELPER)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='formal_resume_fakecontrol_')
        self.addCleanup(self.temporary.cleanup)
        self.checkpoint = Path(self.temporary.name)/'artificial_checkpoint'
        self.checkpoint.mkdir()
        self.cuda_guard = patch.object(self.torch.cuda, 'is_available', return_value=False)
        self.cuda_guard.start()
        self.addCleanup(self.cuda_guard.stop)
        for name in ('get_rng_state_all', 'get_rng_state', 'set_rng_state_all', 'set_rng_state',
                     'init', '_lazy_init', 'manual_seed_all'):
            guard = patch.object(self.torch.cuda, name, side_effect=AssertionError('CUDA forbidden'))
            guard.start()
            self.addCleanup(guard.stop)
        self.parameter = self.torch.tensor([1., 2.], requires_grad=True)
        optimizer = self.torch.optim.AdamW([self.parameter], lr=0.001)
        # Populate artificial native AdamW state without forward/backward/step.
        optimizer.state[self.parameter] = {'step': self.torch.tensor(2.),
            'exp_avg': self.torch.tensor([0.1, 0.2]), 'exp_avg_sq': self.torch.tensor([0.01, 0.02])}
        scheduler = self.torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
        self.saved = {'optimizer.pt': optimizer.state_dict(), 'scheduler.pt': scheduler.state_dict(),
                      'rng_state_0.pth': self.module.rng_state(False)}
        for name, value in self.saved.items():
            self.torch.save(value, self.checkpoint/name)
        self.sidecar = {'binding_sha256': 'artificial-control-binding', 'step': 2,
            'world_size': 8, 'rank_acknowledgments': list(range(8)),
            'files': {name: self.module.sha(self.checkpoint/name) for name in self.saved}}
        (self.checkpoint/'formal_checkpoint.json').write_text(json.dumps(self.sidecar))
        self.native = {'optimizer_scheduler': self.Trainer._load_optimizer_and_scheduler,
                       'rng': self.Trainer._load_rng_state}
        self.observer = self.module.ResumeObservation(self.checkpoint,
            self.module.sha(self.checkpoint/'formal_checkpoint.json'), 0,
            'artificial-control-binding', self.native)
        node = next(node for node in ast.walk(ast.parse(DRIVER.read_text()))
                    if isinstance(node, ast.ClassDef) and node.name == 'FormalBaseTrainer')
        node.body = [node for node in node.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('_load_rng_state', '_load_optimizer_and_scheduler')]
        namespace = {'metrics': SimpleNamespace(TokenValidationTrainer=self.Trainer),
            'distributed': None, 'group': None, 'collective': lambda dist, group, call: [call()],
            'resume_observer': SimpleNamespace(observe=lambda *args: self.observer.observe(*args, include_cuda=False))}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(DRIVER), 'exec'), namespace)
        self.trainer = object.__new__(namespace['FormalBaseTrainer'])
        self.trainer.args = SimpleNamespace(world_size=8, device=self.torch.device('cpu'), process_index=0)
        self.trainer.is_deepspeed_enabled = self.trainer.is_fsdp_enabled = self.trainer.is_fsdp_xla_v1_enabled = False
        # Real Accelerate state_dict/load_state_dict delegation, no Accelerator init.
        from accelerate.utils import DistributedType
        wrapped = object.__new__(self.AcceleratedOptimizer)
        wrapped.optimizer = optimizer
        wrapped.accelerator_state = SimpleNamespace(distributed_type=DistributedType.NO)
        wrapped.device_placement = False
        self.trainer.optimizer, self.trainer.lr_scheduler = wrapped, scheduler

    def restore(self):
        self.trainer._load_optimizer_and_scheduler(str(self.checkpoint))
        self.trainer._load_rng_state(str(self.checkpoint))

    def test_actual_native_cpu_restore_and_eight_receipt_boundary(self):
        before = self.module.state_digest(self.module.rng_state(False))
        calls = []
        previous = sys.getprofile()
        def observe(frame, event, result):
            if event == 'call' and frame.f_code in [method.__code__ for method in self.native.values()]:
                calls.append(frame.f_code.co_name)
        try:
            sys.setprofile(observe)
            self.restore()
        finally:
            sys.setprofile(previous)
        self.assertEqual(calls, ['_load_optimizer_and_scheduler', '_load_rng_state'])
        self.assertEqual(before, self.module.state_digest(self.module.rng_state(False)))
        receipt = self.observer.receipt(3)
        # Eight logical metadata fixtures only; not eight actual processes.
        receipts = [dict(copy.deepcopy(receipt), rank=rank) for rank in range(8)]
        for rank, item in enumerate(receipts):
            event = next(event for event in item['events'] if event['kind'] == 'rng')
            event['matches'] = {f'rng_state_{rank}.pth': event['matches']['rng_state_0.pth']}
        self.assertFalse(self.module.complete_receipt(receipts)['continuous_trajectory_equivalence_tested'])
        with self.assertRaises(ValueError):
            self.module.complete_receipt(receipts[:7])
        receipts[7]['events'] = receipts[7]['events'][:1]
        with self.assertRaises(ValueError):
            self.module.complete_receipt(receipts)

    def test_none_passthrough_and_missing_observation(self):
        self.assertIsNone(self.trainer._load_optimizer_and_scheduler(None))
        self.assertIsNone(self.trainer._load_rng_state(None))
        self.assertEqual(self.observer.events, [])
        with self.assertRaises(ValueError):
            self.observer.receipt(3)

    def test_state_and_rank_rejection(self):
        self.restore()
        self.trainer.optimizer.param_groups[0]['lr'] = 0.2
        with self.assertRaisesRegex(ValueError, 'memory differs'):
            self.observer.observe('optimizer_scheduler', self.trainer, self.checkpoint,
                self.Trainer._load_optimizer_and_scheduler, include_cuda=False)
        self.trainer.args.process_index = 1
        with self.assertRaisesRegex(ValueError, 'rank differs'):
            self.observer.observe('rng', self.trainer, self.checkpoint,
                self.Trainer._load_rng_state, include_cuda=False)
        with self.assertRaisesRegex(ValueError, 'No actual resumed'):
            self.observer.receipt(2)
        self.assertNotEqual(self.module.state_digest([1]), self.module.state_digest((1,)))

    def test_file_source_and_rng_drift_rejection(self):
        self.restore()
        with (self.checkpoint/'optimizer.pt').open('ab') as stream:
            stream.write(b'control-drift')
        with self.assertRaisesRegex(ValueError, 'SHA differs'):
            self.observer.observe('optimizer_scheduler', self.trainer, self.checkpoint,
                self.Trainer._load_optimizer_and_scheduler, include_cuda=False)
        self.observer.sources['rng']['method_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'source changed'):
            self.observer.observe('rng', self.trainer, self.checkpoint,
                self.Trainer._load_rng_state, include_cuda=False)
        changed = copy.deepcopy(self.saved['rng_state_0.pth'])
        changed['cpu'][0] ^= 1
        self.assertNotEqual(self.module.state_digest(changed), self.module.state_digest(self.saved['rng_state_0.pth']))
        self.observer.sources['rng'] = self.module.method_source(self.Trainer._load_rng_state)
        with patch.object(self.observer, '_read', return_value=changed):
            with self.assertRaisesRegex(ValueError, 'memory differs'):
                self.observer.observe('rng', self.trainer, self.checkpoint,
                    self.Trainer._load_rng_state, include_cuda=False)

    def test_cold_import_and_parser_do_not_import_ml(self):
        import builtins
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in ('torch', 'numpy', 'transformers', 'accelerate'):
                raise AssertionError('Cold import attempted ML')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded):
            runpy.run_path(str(HELPER), run_name='cold_helper_control')
            driver = runpy.run_path(str(DRIVER), run_name='cold_driver_control')
            with self.assertRaises(SystemExit) as result:
                driver['parse_args'](['--help'])
            self.assertEqual(result.exception.code, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
