"""Protocol and immutable sources; run python -B -m architecture_hierarchy.tests."""
from pathlib import Path
import shutil
import uuid
from architecture_round2.common import source_binding as previous_sources, prepare as previous_prepare
from experiments.arch70m.common import read, write, sha, digest, stamp, lock, commit_id

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path('/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_hierarchy_m32_20261029')
BASE_OUT = Path('/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_v8_20261028')
PREVIOUS_OUT = Path('/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_round2_20261029')
PREVIOUS_SHA = '93518fa2a14bc029e808cefae7aef8c38d79804683bb5274af27ff5f4e67d65e'
DATA_SHA = '4beaac6b54a1f46492533aeb77c2de2c6ac2a0bd8266fd57a6bb312679435c51'
VARIANTS = ('B2', 'B3', 'B0', 'B1')
LABELS = {'B0':'A5 control', 'B1':'Global + learned scales',
          'B2':'Local then global', 'B3':'Local then global + learned scales'}
MICRO, GLOBAL, SEED, STEPS = 32, 512, 43, 1276


def source_binding():
    sources = previous_sources()
    if digest(sources) != PREVIOUS_SHA:
        raise ValueError('Original round-two sources changed.')
    paths = list((ROOT/'architecture_hierarchy').glob('*.py'))
    paths += [ROOT/'architecture_hierarchy/protocol.json', ROOT/'launch_arch70m_hierarchy.sh']
    return dict(sources, **{str(p.relative_to(ROOT)):sha(p) for p in paths})


def prepare(out):
    out = Path(out).resolve()
    if any(out == p or p in out.parents for p in (BASE_OUT, PREVIOUS_OUT)):
        raise ValueError('Preserve the original experiment output directories.')
    settings = previous_prepare(out)
    if settings['training']['global_batch'] != GLOBAL:
        raise ValueError('The original global batch differs.')
    protocol = read(ROOT/'architecture_hierarchy/protocol.json')
    expected = dict(micro_batch=MICRO, global_batch=GLOBAL, seed=SEED, steps=STEPS, epochs=1,
                    variants=list(VARIANTS), previous_source_sha256=PREVIOUS_SHA, data_sha256=DATA_SHA)
    if any(protocol.get(k) != v for k,v in expected.items()):
        raise ValueError('Protocol constants differ from the frozen manifest.')
    return settings


def snapshot(out):
    sources = source_binding(); signature = digest(sources)
    target = Path(out)/'sources'/signature
    with lock(Path(out)/'sources.lock'):
        if not target.exists():
            pending = target.parent/('.pending_'+uuid.uuid4().hex); pending.mkdir(parents=True)
            for name in sources:
                dest = pending/name; dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/name, dest)
            write(pending/'provenance.json', dict(source_sha256=signature, previous_source_sha256=PREVIOUS_SHA,
                  commit=commit_id(), created=stamp(), files=sources))
            pending.rename(target)
        if any(sha(target/name) != checksum for name,checksum in sources.items()):
            raise ValueError('Immutable source snapshot changed.')
    return target, signature


def valid_profile(profile, variant, signature):
    expected = dict(variant=variant, micro_batch=MICRO, global_batch=GLOBAL, seed=SEED,
                    length=2048, source_sha256=signature, data_sha256=DATA_SHA,
                    grad_accum=16, final_micro_batch=32)
    if any(profile.get(k) != v for k,v in expected.items()):
        raise ValueError('Profile protocol binding differs.')
    candidates = profile.get('candidates', [])
    if len(candidates) != 1:
        raise ValueError('Expected one common-micro qualification.')
    item = candidates[0]
    if (item.get('status') != 'passed' or item.get('updates') != 3 or not item.get('safe') or
            item['reserved_bytes']+item['external_bytes'] > .85*item['total_bytes']):
        raise ValueError('Unsafe or incomplete GPU qualification.')


def valid_qualification(out, signature):
    qualification = read(Path(out)/'qualification.json')
    expected = dict(status='passed', micro_batch=MICRO, source_sha256=signature, data_sha256=DATA_SHA)
    if any(qualification.get(k) != v for k,v in expected.items()) or set(qualification.get('profiles',{})) != set(VARIANTS):
        raise ValueError('Missing matching qualification for all four variants.')
    for variant in VARIANTS:
        path = Path(out)/'trials'/f'screen_{variant}_s{SEED}'/'profile.json'
        if sha(path) != qualification['profiles'][variant]:
            raise ValueError('Qualified profile checksum changed: '+variant)
        valid_profile(read(path),variant,signature)


def completed(out, variant, signature):
    trial = Path(out)/'trials'/f'screen_{variant}_s{SEED}'
    if not (trial/'status.json').exists() or read(trial/'status.json')['status'] != 'completed':
        return False
    binding = read(trial/'binding.json')
    expected = dict(variant=variant, seed=SEED, epochs=1, total_steps=STEPS, micro_batch=MICRO,
                    source_sha256=signature, data_sha256=DATA_SHA)
    if any(binding.get(k) != v for k,v in expected.items()):
        raise ValueError('Completed trial binding differs.')
    if binding['settings'] != read(ROOT/'experiments/arch70m/base.json'):
        raise ValueError('Completed trial settings differ.')
    profile = read(trial/'profile.json'); valid_profile(profile, variant, signature)
    if binding['profile_sha256'] != sha(trial/'profile.json'):
        raise ValueError('Completed profile checksum differs.')
    cp = trial/f'checkpoints/step_{STEPS:06d}'; receipt = read(cp/'receipt.json')
    if receipt['status'] != 'complete' or len(receipt['files']) != 6:
        raise ValueError('Incomplete final checkpoint.')
    if any(sha(cp/name) != checksum for name,checksum in receipt['files'].items()):
        raise ValueError('Final checkpoint checksum differs.')
    state = read(cp/'state.json')
    if state['step'] != STEPS or state['seen_records'] != 653288 or state['epoch'] != 1 or state['next_batch'] != 0:
        raise ValueError('Final data cursor differs.')
    if read(cp/'binding.json') != binding or read(trial/f'metrics/screen_{STEPS:06d}.json')['step'] != STEPS:
        raise ValueError('Final metric or checkpoint binding differs.')
    return True
