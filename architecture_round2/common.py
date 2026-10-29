"""Frozen round-two I/O; run /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_round2.sh."""
from pathlib import Path
import shutil
import uuid
from experiments.arch70m.common import read, write, sha, digest, stamp, lock
from experiments.arch70m.common import source_binding as base_source_binding

ROOT = Path(__file__).resolve().parents[1]
BASE_OUT = Path('/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_v8_20261028')
DEFAULT_OUT = Path('/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_round2_20261029')
BASE_SOURCE_SHA = '053297983e0f2ece2aaad807032695fb5aebd1e0f221138a9bf0d202cc9b9ac4'
DATA_SHA = '4beaac6b54a1f46492533aeb77c2de2c6ac2a0bd8266fd57a6bb312679435c51'
VARIANTS = ('B1', 'B2', 'B3')
SEED = 43


def source_binding():
    base = base_source_binding()
    if digest(base) != BASE_SOURCE_SHA:
        raise ValueError('The original A0-A7 frozen source changed.')
    files = sorted((ROOT/'architecture_round2').glob('*.py'))
    files += [ROOT/'architecture_round2/protocol.json']
    return dict(base, **{str(p.relative_to(ROOT)): sha(p) for p in files})


def prepare(out):
    out = Path(out).resolve()
    if out == BASE_OUT.resolve(): raise ValueError('Round two requires a separate output directory.')
    original = BASE_OUT/'data/index.json'
    if sha(original) != DATA_SHA: raise ValueError('Original fixed 10% data index changed.')
    data = out/'data'; data.mkdir(parents=True, exist_ok=True)
    with lock(out/'data.lock'):
        if not (data/'index.json').exists(): shutil.copyfile(original, data/'index.json')
        if sha(data/'index.json') != DATA_SHA: raise ValueError('Round-two data differs from the original index.')
        if not (data/'verified.json').exists() and (original.parent/'verified.json').exists():
            shutil.copyfile(original.parent/'verified.json', data/'verified.json')
    settings = read(ROOT/'experiments/arch70m/base.json')
    reference = read(BASE_OUT/'trials/screen_A5_s43/binding.json')
    expected = dict(settings=settings, variant='A5', seed=SEED, epochs=1, micro_batch=56,
                    total_steps=1276, source_sha256=BASE_SOURCE_SHA, data_sha256=DATA_SHA)
    if any(reference.get(k) != v for k,v in expected.items()):
        raise ValueError('The matched A5 seed43 control has different settings.')
    return settings


def snapshot(out):
    files = source_binding(); signature = digest(files)
    target = Path(out)/'sources'/signature
    with lock(Path(out)/'sources.lock'):
        if not target.exists():
            temporary = target.parent/('.pending_'+uuid.uuid4().hex)
            temporary.mkdir(parents=True)
            for name in files:
                dest = temporary/name; dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/name, dest)
            write(temporary/'provenance.json', dict(created=stamp(), source_sha256=signature,
                  base_source_sha256=BASE_SOURCE_SHA, files=files))
            temporary.rename(target)
        if any(sha(target/name) != checksum for name,checksum in files.items()):
            raise ValueError('Round-two source snapshot changed.')
    return target, signature
