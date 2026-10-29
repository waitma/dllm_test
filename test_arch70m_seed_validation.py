"""CPU checks: python -B /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/test_arch70m_seed_validation.py."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

import arch70m_seed_validation as launcher


class LauncherTests(unittest.TestCase):
    def args(self, out):
        return launcher.parser().parse_args(['--output-root', str(out), '--gpus', '0,1,2,3,4,5,6,7'])

    def test_only_one_extra_seed_and_one_variant_per_gpu(self):
        jobs = launcher.plan(self.args('/tmp/arch70m_unused'))
        self.assertEqual([j['variant'] for j in jobs], list(launcher.VARIANTS))
        self.assertEqual([j['gpu'] for j in jobs], list('01234567'))
        self.assertEqual({j['seed'] for j in jobs}, {43})
        self.assertEqual({j['epochs'] for j in jobs}, {1})
        self.assertEqual({j['micro_batch'] for j in jobs}, {56})
        self.assertEqual({j['global_batch'] for j in jobs}, {512})
        for value in ('0,1,2,3', '0,1,2,3,4,5,6,6', '0,1,2,3,4,5,6,7,', '0,1,2,3,4,5,6, 7'):
            with self.assertRaises(argparse.ArgumentTypeError): launcher.gpu_ids(value)

    def test_worker_does_not_interpret_its_single_visible_gpu_as_eight(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-only-one'}), \
             patch('sys.argv', ['launcher', '--worker', 'A5', '--source', '/tmp/frozen']), \
             patch.object(launcher, 'profile_one') as profile:
            launcher.main()
        self.assertEqual(profile.call_count, 1)
        self.assertEqual(profile.call_args.args[0].worker, 'A5')

    def test_uuid_alias_and_missing_card_fail_before_launch(self):
        def aliased(cmd, **kwargs): return subprocess.CompletedProcess(cmd, 0, 'GPU-same\n', '')
        with patch.object(launcher.subprocess, 'run', side_effect=aliased):
            with self.assertRaises(ValueError): launcher.visible_uuids(list('01234567'))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'not-created'
            with patch.object(launcher, 'visible_uuids', side_effect=ValueError('missing card')), \
                 patch.object(launcher.subprocess, 'Popen') as start:
                with self.assertRaises(ValueError): launcher.execute(self.args(out))
                self.assertFalse(out.exists()); start.assert_not_called()

    def test_dry_run_has_no_gpu_or_output_side_effect(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'not-created'; args = self.args(out); args.dry_run = True
            with patch.object(launcher, 'visible_uuids') as gpu, patch('builtins.print'):
                launcher.execute(args)
            gpu.assert_not_called(); self.assertFalse(out.exists())

    def test_eight_subprocesses_have_isolated_devices_and_clean_env(self):
        from experiments.arch70m import suite
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'out'; out.mkdir(); source = Path(tmp)/'source'; source.mkdir()
            processes = []; counter = {}; guard = threading.Lock()
            def complete(root, variant):
                with guard:
                    counter[variant] = counter.get(variant, 0)+1
                    return counter[variant] > 1
            class Process:
                def __init__(self, cmd, **kwargs):
                    with guard:
                        self.pid = 9000+len(processes); processes.append((cmd, kwargs))
                def wait(self): return 0
            with patch.object(launcher, 'visible_uuids', return_value=['GPU-'+str(i) for i in range(8)]), \
                 patch.object(launcher, 'verify_reference', return_value=({'recipe': {}}, 1276)), \
                 patch.object(suite, 'verify_data'), patch.object(suite, 'snapshot', return_value=(source, launcher.SOURCE_SHA)), \
                 patch.object(launcher, 'completed', side_effect=complete), patch.object(launcher, 'publish') as publish, \
                 patch.object(launcher.subprocess, 'Popen', Process), patch('builtins.print'), \
                 patch.dict(os.environ, {'WORLD_SIZE': '8', 'LOCAL_RANK': '7', 'NCCL_DEBUG': 'INFO', 'TORCHELASTIC_RUN_ID': 'old'}):
                launcher.execute(self.args(out))
            self.assertEqual(len(processes), 8)
            self.assertEqual(publish.call_count, 9)
            mapping = {}
            for cmd, kwargs in processes:
                variant = cmd[cmd.index('--worker')+1]; env = kwargs['env']
                mapping[variant] = env['CUDA_VISIBLE_DEVICES']
                self.assertEqual(env['PYTHONPATH'], str(source))
                self.assertEqual(env['ARCH70M_FROZEN_SOURCE'], str(source))
                self.assertFalse(any(k in launcher.DISTRIBUTED_ENV or k.startswith(('NCCL_', 'TORCHELASTIC_')) for k in env))
                self.assertNotIn('torchrun', cmd)
            self.assertEqual(mapping, {v:'GPU-'+str(i) for i,v in enumerate(launcher.VARIANTS)})
            self.assertEqual(launcher.read(out/'seed43_validation/manifest.json')['seeds'], [43])

    def test_corrupt_completed_checkpoint_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); t = out/'trials/screen_A0_s43'; cp = t/'checkpoints/step_001276'; cp.mkdir(parents=True)
            launcher.write(t/'status.json', dict(status='completed', step=1276, total_steps=1276))
            hashes = {}
            for name in ('model.safetensors', 'optimizer.pt', 'scheduler.pt', 'rng.pt', 'state.json', 'binding.json'):
                (cp/name).write_text('test'); hashes[name] = launcher.sha(cp/name)
            launcher.write(cp/'receipt.json', dict(status='complete', files=hashes))
            launcher.write(t/'metrics/screen_001276.json', dict(step=1276, partition='screen'))
            self.assertTrue(launcher.completed(out, 'A0'))
            (cp/'model.safetensors').write_text('corrupted')
            with self.assertRaises(ValueError): launcher.completed(out, 'A0')

    def test_profile_resume_execs_frozen_screen_trainer_with_seed43(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); args = self.args(out); args.worker = 'A5'; args.source = launcher.ROOT
            t = out/'trials/screen_A5_s43'; t.mkdir(parents=True)
            launcher.write(t/'profile.json', dict(variant='A5', micro_batch=56, common_micro_seed_validation=43))
            with patch.object(launcher, 'sha', return_value=launcher.DATA_SHA), \
                 patch.object(launcher.os, 'execv') as execute:
                launcher.profile_one(args)
            cmd = execute.call_args.args[1]
            self.assertEqual(cmd[cmd.index('--seed')+1], '43')
            self.assertEqual(cmd[cmd.index('--stage')+1], 'screen')
            self.assertEqual(cmd[cmd.index('--variant')+1], 'A5')
            self.assertIn('experiments.arch70m.train', cmd)

    def test_report_keeps_both_seeds_and_excludes_pending_from_deltas(self):
        from experiments.arch70m import report as reporting
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out = root/'output'; out.mkdir()
            for seed in (42, 43):
                for variant in launcher.VARIANTS:
                    if seed == 43 and variant not in ('A0', 'A5'): continue
                    t = out/'trials'/f'screen_{variant}_s{seed}'
                    launcher.write(t/'status.json', dict(status='completed', step=1276, total_steps=1276))
                    metrics = dict(step=1276, partition='screen', token={'all': {'loss': .7}},
                        generation=dict(conditional_aar=.3, conditional_ce=3., conditional_validity=1., conditional_length_mae=5., cdr_aar=.8),
                        representation=dict(macro_auroc=.85, macro_auprc=.65))
                    launcher.write(t/'metrics/screen_001276.json', metrics)
            with patch.object(launcher, 'ROOT', root), patch.object(reporting, 'plot_curves'):
                self.assertEqual(launcher.report(out), 2)
            folder = root/launcher.REPORT_REL
            summary = launcher.read(folder/'summary.json')
            self.assertEqual(len(summary['trials']), 16)
            with (folder/'paired_vs_A0.csv').open() as f: rows = list(csv.DictReader(f))
            self.assertEqual({r['variant'] for r in rows if r['seed'] == '43'}, {'A0', 'A5'})


if __name__ == '__main__': unittest.main()
