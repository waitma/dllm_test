"""Frozen factorial protocol and data: python -B -m architecture_attention.tests."""
from pathlib import Path
import shutil
import uuid
from architecture_confirm.common import source_binding as previous_sources
from architecture_hierarchy.common import valid_profile, completed as previous_completed
from experiments.arch70m.common import read, write, sha, digest, stamp, lock, commit_id

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_attention_20261102')
BASE = ROOT / 'output/arch70m_v8_20261028'
PREVIOUS_SHA = '3c9c424d816b9fda7adfba5047d9e075ac15dc036342b59461edd35031e50971'
DATA_SHA = '4beaac6b54a1f46492533aeb77c2de2c6ac2a0bd8266fd57a6bb312679435c51'
VARIANTS = ('C0', 'C1', 'C2', 'C3', 'A0')
LABELS = {'C0': 'A5 control', 'C1': 'A5 + QK RMS norm',
          'C2': 'A5 + unit output gate', 'C3': 'A5 + both factors', 'A0': 'Original architecture control'}
COUNTS = {'C0': 75741903, 'C1': 75742927, 'C2': 75815727, 'C3': 75816751, 'A0': 75741903}
MICRO, GLOBAL, SEED, STEPS = 32, 512, 43, 1276


def trial(out, variant): return Path(out) / 'trials' / f'screen_{variant}_s{SEED}'


def source_binding():
    sources = previous_sources()
    if digest(sources) != PREVIOUS_SHA: raise ValueError('Historical frozen sources changed.')
    files = list((ROOT / 'architecture_attention').glob('*.py'))
    files += [ROOT / 'architecture_attention/protocol.json', ROOT / 'launch_arch70m_attention.sh']
    return dict(sources, **{str(p.relative_to(ROOT)): sha(p) for p in files})


def validate_output(out):
    out = Path(out).resolve()
    protected = [ROOT / 'output' / n for n in ('arch70m_v8_20261028', 'arch70m_round2_20261029',
        'arch70m_hierarchy_m32_20261029', 'arch70m_confirm_20261031', 'arch70m_confirm_4gpu_20261031',
        'arch70m_local_recheck_20261102')]
    if any(out == p or p in out.parents or out in p.parents for p in protected):
        raise ValueError('Historical experiment outputs are read-only.')
    production = ROOT.parent / 'dllm_test'
    if out == production or production in out.parents or out in production.parents:
        raise ValueError('Production checkout is read-only.')
    return out


def prepare(out):
    out = validate_output(out)
    protocol = read(ROOT / 'architecture_attention/protocol.json')
    expected = dict(variants=list(VARIANTS), seed=SEED, epochs=1, steps=STEPS,
                    micro_batch=MICRO, global_batch=GLOBAL, data_sha256=DATA_SHA,
                    previous_source_sha256=PREVIOUS_SHA, parameters=COUNTS)
    if any(protocol.get(k) != v for k, v in expected.items()): raise ValueError('Frozen factorial protocol differs.')
    if sha(BASE / 'data/index.json') != DATA_SHA: raise ValueError('Original fixed data index changed.')
    with lock(out / 'data.lock'):
        dest = out / 'data'; dest.mkdir(parents=True, exist_ok=True)
        if not (dest / 'index.json').exists(): shutil.copyfile(BASE / 'data/index.json', dest / 'index.json')
        if sha(dest / 'index.json') != DATA_SHA: raise ValueError('New index is not byte-identical.')
        if not (dest / 'verified.json').exists(): shutil.copyfile(BASE / 'data/verified.json', dest / 'verified.json')
    settings = read(ROOT / 'experiments/arch70m/base.json')
    if settings['training']['global_batch'] != GLOBAL: raise ValueError('Global batch changed.')
    return settings


def snapshot(out):
    sources = source_binding(); signature = digest(sources)
    target = Path(out) / 'sources' / signature
    with lock(Path(out) / 'sources.lock'):
        if not target.exists():
            pending = target.parent / ('.pending_' + uuid.uuid4().hex); pending.mkdir(parents=True)
            for name in sources:
                dest = pending / name; dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, dest)
            write(pending / 'provenance.json', dict(files=sources, source_sha256=signature,
                  previous_source_sha256=PREVIOUS_SHA, commit=commit_id(), created=stamp()))
            pending.rename(target)
        if any(sha(target / name) != value for name, value in sources.items()):
            raise ValueError('Immutable snapshot changed.')
    return target, signature


def qualified(out, signature):
    q = read(Path(out) / 'qualification.json')
    if (q.get('status'), q.get('source_sha256'), q.get('data_sha256'), q.get('micro_batch')) != ('passed', signature, DATA_SHA, MICRO):
        raise ValueError('Missing common GPU qualification.')
    if set(q.get('profiles', {})) != set(VARIANTS): raise ValueError('All five variants must qualify.')
    for v in VARIANTS:
        p = trial(out, v) / 'profile.json'
        if sha(p) != q['profiles'][v]: raise ValueError('Profile checksum changed.')
        valid_profile(read(p), v, signature)


def completed(out, variant, signature):
    # The unchanged one-epoch receipt checker is variant-agnostic and uses the
    # same seed43/micro32/data cursor and six-file checkpoint contract.
    done = previous_completed(out, variant, signature)
    if done:
        t = trial(out, variant)
        status = read(t / 'status.json')
        state = read(t / 'checkpoints/step_001276/state.json')
        if status['step'] != STEPS or status['parameters'] != COUNTS[variant] or state['seen_tokens'] != 214482921:
            raise ValueError('Final step, model size or token budget changed.')
        metrics = read(t / 'metrics/screen_001276.json')
        if metrics['partition'] != 'screen' or (t / 'metrics/confirm_001276.json').exists():
            raise ValueError('Screen-only evaluation contract changed.')
    return done
