"""Explicit maintenance load for an already allocated GPU; not an experiment.

Worker-managed use only: CUDA_VISIBLE_DEVICES=<assigned GPU> python
scripts/gpu_standby_load.py --duty-cycle 0.7 --max-seconds 3600
SIGTERM stops the process and releases its CUDA context. No checkpoints are saved.
"""
import argparse
import json
import os
import signal
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duty-cycle', type=float, default=.7)
    parser.add_argument('--max-seconds', type=float, required=True)
    args = parser.parse_args()
    if not .05 <= args.duty_cycle <= .95 or args.max_seconds <= 0:
        raise ValueError('Duty cycle must be 0.05–0.95 and duration positive')
    import torch
    if torch.cuda.device_count() != 1:
        raise ValueError('Maintenance load requires exactly one assigned GPU')
    os.nice(10)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    torch.set_num_threads(1)
    a = torch.randn((4096, 4096), device='cuda', dtype=torch.float16)*.01
    b = torch.randn_like(a)*.01
    out = torch.empty_like(a)
    deadline = time.monotonic()+args.max_seconds
    last_log = 0.
    count = 0
    print(json.dumps(dict(kind='maintenance_only', duty_cycle=args.duty_cycle,
                          gpu=torch.cuda.get_device_name(), pid=os.getpid())), flush=True)
    with torch.no_grad():
        while not stop.is_set() and time.monotonic() < deadline:
            start = time.monotonic()
            while not stop.is_set() and time.monotonic()-start < .5*args.duty_cycle:
                torch.mm(a, b, out=out)
                torch.cuda.synchronize()
                count += 1
            now = time.monotonic()
            if now-last_log >= 60:
                print(json.dumps(dict(time=time.time(), kind='maintenance_only', multiplications=count,
                                      allocated_mib=torch.cuda.memory_allocated()/2**20)), flush=True)
                last_log = now
            stop.wait(max(0., .5-(time.monotonic()-start)))


if __name__ == '__main__':
    main()
