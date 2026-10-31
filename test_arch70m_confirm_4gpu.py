"""CPU scheduler checks: python -B -m unittest /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/test_arch70m_confirm_4gpu.py."""
import argparse
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import arch70m_confirm_4gpu as launcher


class FourGPUConfirmationTests(unittest.TestCase):
    def setUp(self):launcher.STOP.clear()

    def test_four_distinct_ids_and_complete_nine_job_matrix(self):
        ids=launcher.gpu_ids('0,1,2,3');jobs=launcher.plan(ids)
        self.assertEqual(len(jobs),9);self.assertEqual({(j['variant'],j['seed']) for j in jobs},set(launcher.JOBS))
        self.assertEqual([sum(j['gpu']==g for j in jobs) for g in ids],[3,2,2,2])
        self.assertTrue(all(j['epochs']==3 and j['total_steps']==3828 and j['micro_batch']==32 and j['global_batch']==512 for j in jobs))
        for bad in ['0,1,2','0,0,2,3','0,00,2,3','0,1,2,3,','0, 1,2,3']:
            with self.assertRaises(argparse.ArgumentTypeError):launcher.gpu_ids(bad)

    def test_dry_run_never_queries_gpu_or_creates_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'absent';args=argparse.Namespace(gpus=['0','1','2','3'],output_root=out,dry_run=True)
            with patch.object(launcher,'resolve_devices') as gpu,redirect_stdout(io.StringIO()):launcher.execute(args)
            gpu.assert_not_called();self.assertFalse(out.exists())

    def test_nvml_alias_resolution_uses_numeric_cuda_selectors_and_rejects_duplicates(self):
        with patch('arch70m_seed_validation.cuda_device_id',side_effect=['0','1','2','3']), \
             patch.object(launcher.subprocess,'check_output',side_effect=['GPU-a\n','GPU-b\n','GPU-c\n','GPU-d\n']):
            devices=launcher.resolve_devices(['GPU-x','GPU-y','GPU-z','GPU-w'])
            self.assertEqual([d['cuda_selector'] for d in devices.values()],['0','1','2','3'])
        with patch('arch70m_seed_validation.cuda_device_id',side_effect=['0','1','2','3']), \
             patch.object(launcher.subprocess,'check_output',return_value='GPU-same\n'):
            with self.assertRaises(ValueError):launcher.resolve_devices(['0','1','2','3'])

    def test_workers_clear_distributed_environment_and_use_one_device(self):
        with patch.dict(os.environ,dict(WORLD_SIZE='4',RANK='2',LOCAL_RANK='2',NCCL_DEBUG='INFO',TORCHELASTIC_RUN_ID='old')):
            env=launcher.worker_env(Path('/tmp/frozen'),'3')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'3');self.assertEqual(env['CUDA_DEVICE_ORDER'],'PCI_BUS_ID')
        self.assertEqual(env['PYTHONPATH'],'/tmp/frozen')
        self.assertFalse(any(k in env for k in ['WORLD_SIZE','RANK','LOCAL_RANK','NCCL_DEBUG','TORCHELASTIC_RUN_ID']))

    def test_four_lanes_are_parallel_but_never_overlap_on_same_gpu(self):
        barrier=threading.Barrier(4);guard=threading.Lock();active=set();seen=set();maximum=[0];callback_threads=[]
        def work(job):
            with guard:
                self.assertNotIn(job['gpu'],active);active.add(job['gpu']);maximum[0]=max(maximum[0],len(active))
                first=job['gpu'] not in seen;seen.add(job['gpu'])
            if first:barrier.wait(timeout=5)
            time.sleep(.01)
            with guard:active.remove(job['gpu'])
            return dict(job,status='completed',exit_code=0)
        parent=threading.get_ident()
        results=launcher.parallel_lanes(launcher.plan(['0','1','2','3']),work,lambda r:callback_threads.append(threading.get_ident()))
        self.assertEqual(maximum[0],4);self.assertEqual(len(results),9)
        self.assertTrue(all(r['status']=='completed' for r in results));self.assertEqual(callback_threads,[parent]*9)

    def test_lane_failure_is_recorded_without_hiding_remaining_trials(self):
        def work(job):
            if (job['variant'],job['seed'])==('B1',43):raise RuntimeError('original training failure')
            return dict(job,status='completed',exit_code=0)
        results=launcher.parallel_lanes(launcher.plan(['0','1','2','3']),work)
        self.assertEqual(len(results),9);bad=[r for r in results if r['status']=='failed']
        self.assertEqual(len(bad),1);self.assertIn('original training failure',bad[0]['error'])

    def test_requalification_preserves_checkpoint_bound_profile_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'real';shadow=Path(tmp)/'shadow';v,s='A0',42
            original=dict(variant=v,seed=s,micro_batch=32,global_batch=512,length=2048,grad_accum=16,final_micro_batch=32,
                source_sha256=launcher.SOURCE_SHA,data_sha256=launcher.DATA_SHA,
                candidates=[dict(status='passed',updates=3,safe=True,reserved_bytes=70,external_bytes=1,total_bytes=100)])
            old=launcher.trial(out,v,s)/'profile.json';launcher.write(old,original);old_sha=launcher.sha(old)
            new=dict(original,device='fixture');src=launcher.trial(shadow,v,s)
            launcher.write(src/'profile.json',new);launcher.write(src/'cuda_device.json',dict(cuda_uuid='actual-cuda',device='fixture'))
            evidence=launcher.install_profile(out,shadow,v,s)
            self.assertEqual(launcher.sha(old),old_sha)
            self.assertEqual(evidence['canonical_profile_sha256'],old_sha)
            self.assertNotEqual(evidence['actual_profile_sha256'],old_sha)

    def test_actual_cuda_uuid_uniqueness_and_frozen_source_are_enforced(self):
        rows=[dict(gpu=str(i),status='qualified',evidence=dict(cuda_device=dict(cuda_uuid=str(i),device='same'))) for i in range(4)]
        self.assertEqual(len(launcher.validate_actual_devices(rows)),4)
        rows[3]['evidence']['cuda_device']['cuda_uuid']='0'
        with self.assertRaises(ValueError):launcher.validate_actual_devices(rows)
        self.assertEqual(launcher.digest(launcher.source_binding()),launcher.SOURCE_SHA)


if __name__=='__main__':unittest.main()
