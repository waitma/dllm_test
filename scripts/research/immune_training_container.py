"""Keep one allocated eight-GPU container available across training attempts.

Activate pllm, then submit the matching absolute train_jobs YAML. Publish a
reviewed {bundle, launcher_sha256} JSON ticket atomically in POOL/pending.
Failed tickets move to finished and are never retried. STOP ends the container;
PAUSE_TRAINING interrupts only its owned training group and keeps the allocation.
Standby processes are fully reaped before any eight-rank job starts.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import socket
import subprocess
import sys
import threading
import time

ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
HELPER = ROOT/'scripts/immune_research_pool.py'
STANDBY = ROOT/'scripts/gpu_standby_load.py'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool-dir', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=2591400)
    args = parser.parse_args()
    pool = args.pool_dir.resolve()
    settings = json.loads((pool/'container.json').read_text())
    for name in ('helper', 'standby', 'supervisor'):
        ref = settings[name]
        if hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest() != ref['sha256']:
            raise ValueError('Container helper source changed: '+name)
    helper = runpy.run_path(str(HELPER), run_name='_container_helpers')
    write, stop_process = helper['write_json'], helper['stop_process']
    listing = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    devices = [line.split(',')[1].strip() for line in listing.splitlines() if line.strip()]
    if len(devices) != 8 or len(set(devices)) != 8:
        raise ValueError('Exactly eight allocated GPUs are required')
    stopped = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.set())
    standby = []
    training = None
    deadline = time.monotonic()+args.max_seconds
    environment = dict(os.environ)
    for key in tuple(environment):
        if key in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','PYTHONPATH') or key.startswith(('TORCHELASTIC_','ACCELERATE_')):
            environment.pop(key)
    environment.update(USE_TF='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       PYTHONDONTWRITEBYTECODE='1')
    def release_standby():
        receipts = []
        for process in standby:
            stop_process(process)
            receipts.append(dict(pid=process.pid, returncode=process.returncode, reaped=True))
        standby.clear()
        return receipts
    with (pool/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while not stopped.is_set() and time.monotonic() < deadline and not (pool/'STOP').exists():
                if not standby:
                    for rank, device in enumerate(devices):
                        with (pool/'logs'/f'standby{rank}.log').open('a') as log:
                            standby.append(subprocess.Popen([sys.executable,'-I','-B','-u',str(STANDBY),
                                '--duty-cycle','0.7','--max-seconds',str(max(1,int(deadline-time.monotonic())))],
                                env={**environment,'CUDA_VISIBLE_DEVICES':device},stdout=log,stderr=subprocess.STDOUT,
                                start_new_session=True))
                write(pool/'state.json',dict(state='standby',time=time.time(),host=socket.gethostname(),
                                           pid=os.getpid(),standby_pids=[p.pid for p in standby]))
                tickets = sorted((pool/'pending').glob('*.json'))
                if not tickets or (pool/'PAUSE_TRAINING').exists():
                    stopped.wait(3)
                    continue
                ticket = tickets[0]
                active = pool/'running'/ticket.name
                ticket.rename(active)
                result = dict(ticket=str(active), started=time.time(), state='starting')
                try:
                    request = json.loads(active.read_text())
                    bundle = Path(request['bundle'])
                    if not bundle.is_absolute() or bundle != bundle.resolve() or bundle.parent != ROOT/'output/v5_full_binding_20261007':
                        raise ValueError('Only a reviewed V5 launch package may run here')
                    launcher = bundle/'source/scripts/research/launch_v5_direct.py'
                    if hashlib.sha256(launcher.read_bytes()).hexdigest() != request['launcher_sha256']:
                        raise ValueError('Ticket launcher changed')
                    result['standby_cleanup'] = release_standby()
                    command = [sys.executable,'-I','-B','-m','torch.distributed.run','--standalone','--nnodes=1',
                        '--nproc_per_node=8','--max_restarts=0','--no_python',sys.executable,'-I','-B','-u',
                        str(launcher),'--bundle',str(bundle)]
                    result['argv'] = command
                    config = json.loads((bundle/'direct_launch.json').read_text())
                    runtime = Path(config['runtime_dir'])
                    logfile = runtime.parent/('torchrun_'+runtime.name+'.log')
                    logfile.parent.mkdir(parents=True,exist_ok=True)
                    env = {**environment,'CUDA_VISIBLE_DEVICES':','.join(devices),'NCCL_DEBUG':'INFO',
                           'NCCL_CUMEM_HOST_ENABLE':'0','NCCL_IB_DISABLE':'1',
                           'NCCL_SOCKET_IFNAME':'lo','GLOO_SOCKET_IFNAME':'lo','TORCH_NCCL_ASYNC_ERROR_HANDLING':'1'}
                    with logfile.open('x') as log:
                        training = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,
                                                    start_new_session=True)
                        while training.poll() is None:
                            write(pool/'state.json',dict(state='training',time=time.time(),host=socket.gethostname(),
                                pid=os.getpid(),training_pid=training.pid,ticket=str(active),log=str(logfile),standby_pids=[]))
                            if stopped.is_set() or (pool/'STOP').exists() or (pool/'PAUSE_TRAINING').exists() or time.monotonic() >= deadline:
                                stop_process(training)
                                result['interrupted'] = True
                                break
                            stopped.wait(3)
                    result.update(returncode=training.wait(),state='succeeded' if training.returncode==0 else 'failed')
                except Exception as exc:
                    result.update(state='failed',error=repr(exc))
                finally:
                    if training is not None and training.poll() is None:
                        stop_process(training)
                    training = None
                    result['finished'] = time.time()
                    write(pool/'results'/active.name,result)
                    active.rename(pool/'finished'/active.name)
        finally:
            if training is not None:
                stop_process(training)
            release_standby()
            write(pool/'state.json',dict(state='exited',time=time.time(),host=socket.gethostname()))


if __name__ == '__main__':
    main()
