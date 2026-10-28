"""Shared experiment I/O; invoke the suite through experiments/arch70m/run.sh."""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / 'output/arch70m_v8_20261028'
DEFAULT_DATA = Path('/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/data/prepared/immune_v8_four_relations_20261026_r2')
VARIANTS = ('A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''): h.update(part)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(path)

def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, allow_nan=False) + '\n'); f.flush()

def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def source_binding():
    paths = list((ROOT / 'dllm').rglob('*.py'))
    paths += list((ROOT / 'experiments/arch70m').glob('*.py'))
    paths += list((ROOT / 'experiments/arch70m').glob('*.sh'))
    paths += [ROOT / 'experiments/arch70m/base.json', ROOT / 'experiments/__init__.py',
              ROOT / 'experiments/arch70m/variant_parameters.json']
    paths += [ROOT / 'scripts/research' / n for n in
              ('__init__.py', 'diffusion_cdr_tasks.py', 'bert_mlm_data.py', 'bert_mlm_eval.py', 'bert_mlm_objectives.py', 'run_bert_mlm.py')]
    paths += [ROOT/'scripts/__init__.py']
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths)) if p.exists()}

def commit_id():
    return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()

@contextlib.contextmanager
def lock(path, blocking=True):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try: yield
        finally: fcntl.flock(f, fcntl.LOCK_UN)
