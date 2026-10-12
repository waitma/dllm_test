"""Run under pllm: python -B /absolute/path/test_immune_training_container.py.

CPU subprocess fixture verifies standby cleanup and failure containment; no GPU,
corpus, real launcher, checkpoint, or platform job is used.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time
from unittest.mock import patch

import immune_training_container as module


def main():
    with TemporaryDirectory(prefix='training-container-cpu-') as temporary:
        root = Path(temporary).resolve()
        pool = root/'pool'
        for name in ('pending','running','finished','results','logs'):
            (pool/name).mkdir(parents=True)
        dummy = root/'dummy.py'
        dummy.write_text('"""CPU-only owned sleeper fixture."""\nimport time\ntime.sleep(60)\n')
        bundle = root/'output/v5_full_binding_20261007/fixture'
        launcher = bundle/'source/scripts/research/launch_v5_direct.py'
        launcher.parent.mkdir(parents=True)
        launcher.write_text('"""Not executed as a model launcher."""\n')
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        (bundle/'direct_launch.json').write_text(json.dumps({'runtime_dir':str(root/'runtime/attempt')}))
        (pool/'pending/job.json').write_text(json.dumps({'bundle':str(bundle),'launcher_sha256':sha(launcher)}))
        sources = {'helper':module.HELPER,'standby':dummy,'supervisor':Path(module.__file__)}
        (pool/'container.json').write_text(json.dumps({k:{'path':str(p),'sha256':sha(p)} for k,p in sources.items()}))
        processes = []
        spawn = subprocess.Popen
        def owned_spawn(command, **kwargs):
            if 'torch.distributed.run' in command:
                assert len(processes)==8 and all(p.poll() is not None for p in processes), 'Standby overlaps training'
                command = [sys.executable,'-I','-B','-c','raise SystemExit(9)']
            process = spawn(command,**kwargs)
            processes.append(process)
            return process
        def observe_failure():
            limit = time.monotonic()+15
            while time.monotonic() < limit:
                try:
                    state = json.loads((pool/'state.json').read_text())
                    result = json.loads((pool/'results/job.json').read_text())
                    if state['state']=='standby' and result['returncode']==9:
                        (pool/'STOP').touch()
                        return
                except (FileNotFoundError,json.JSONDecodeError):
                    pass
                time.sleep(.05)
            (pool/'STOP').touch()
        observer = threading.Thread(target=observe_failure)
        observer.start()
        with patch.object(module,'ROOT',root),patch.object(module,'STANDBY',dummy), \
             patch.object(sys,'argv',['supervisor','--pool-dir',str(pool),'--max-seconds','20']), \
             patch.object(module.subprocess,'check_output',return_value=''.join(f'{i}, GPU-fixture-{i}\n' for i in range(8))), \
             patch.object(module.subprocess,'Popen',side_effect=owned_spawn):
            module.main()
        observer.join()
        result = json.loads((pool/'results/job.json').read_text())
        assert result['returncode']==9 and result['state']=='failed'
        assert len(result['standby_cleanup'])==8
        assert len(processes)==17, 'Expected initial standby8, failed job1, restored standby8'
        assert all(p.poll() is not None for p in processes)
        assert not list((pool/'pending').glob('*.json')) and not list((pool/'running').glob('*.json'))
        assert (pool/'finished/job.json').exists()
        print(json.dumps({'status':'passed','owned_processes_reaped':17,
                          'failure_retains_container':True,'failed_ticket_retried':False,'gpu_used':False}))


if __name__=='__main__':
    main()
