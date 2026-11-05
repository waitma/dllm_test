"""Execute the real V11 shell with a fake interpreter to check child process flags.

Run: python -B -m unittest test_v11_a5_transport -v. No GPU, model, credentials,
network or formal training is used; only temporary launcher files are written.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest

ROOT=Path('/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore')


class TransportTests(unittest.TestCase):
    def test_actual_torchrun_child_gets_shm_and_no_stale_debug_blocking(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);fake=tmp/'python';run=tmp/'v11_run';captured=tmp/'child.json'
            fake.write_text('#!/usr/bin/python3\n'+textwrap.dedent('''
                import json,os,sys
                from pathlib import Path
                args=sys.argv[1:]
                if any(x.endswith('/v11_a5/prepare.py') for x in args):
                    root=Path(args[args.index('--run-root')+1]);(root/'bundle/runtime').mkdir(parents=True)
                    (root/'bundle/runtime/libnccl.so.2').symlink_to('/usr/lib/x86_64-linux-gnu/libc.so.6')
                elif '-m' in args:
                    keep=['NCCL_SHM_DISABLE','NCCL_P2P_DISABLE','NCCL_IB_DISABLE','NCCL_NET',
                          'NCCL_CUMEM_HOST_ENABLE','NCCL_DEBUG','NCCL_DEBUG_SUBSYS','NCCL_DEBUG_FILE',
                          'CUBLAS_WORKSPACE_CONFIG','IMMUNE_SDPA_BACKEND','CUDA_LAUNCH_BLOCKING']
                    Path(os.environ['V11_TEST_CAPTURE']).write_text(json.dumps(dict(args=args,environment={k:os.environ.get(k) for k in keep})))
                else:
                    sys.stdin.read()
            '''))
            fake.chmod(0o755)
            env=dict(os.environ,V11_PYTHON=str(fake),V11_TEST_CAPTURE=str(captured),
                NCCL_SHM_DISABLE='1',NCCL_P2P_DISABLE='0',NCCL_IB_DISABLE='0',CUDA_LAUNCH_BLOCKING='1')
            env.pop('LD_PRELOAD',None)
            result=subprocess.run(['bash',str(ROOT/'launch_v11_a5_75m.sh'),'--run-root',str(run)],
                env=env,text=True,capture_output=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            observed=json.loads(captured.read_text());settings=observed['environment']
            self.assertEqual(settings['NCCL_SHM_DISABLE'],'0')
            self.assertEqual(settings['NCCL_P2P_DISABLE'],'1')
            self.assertEqual(settings['NCCL_IB_DISABLE'],'1')
            self.assertEqual(settings['NCCL_NET'],'Socket')
            self.assertEqual(settings['NCCL_CUMEM_HOST_ENABLE'],'0')
            self.assertIsNone(settings['CUDA_LAUNCH_BLOCKING'])
            self.assertEqual(settings['CUBLAS_WORKSPACE_CONFIG'],':4096:8')
            self.assertEqual(settings['IMMUNE_SDPA_BACKEND'],'efficient')
            self.assertEqual(settings['NCCL_DEBUG_SUBSYS'],'INIT,ENV,GRAPH')
            self.assertEqual(settings['NCCL_DEBUG_FILE'],str(run/'nccl.%h.%p.log'))
            self.assertIn('--nproc_per_node=8',observed['args'])
            self.assertIn(str(run/'bundle/source/v11_a5/launch.py'),observed['args'])
            self.assertFalse((run/'train').exists())


if __name__=='__main__':unittest.main()
