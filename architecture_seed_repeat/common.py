"""Frozen seed42 replication bindings: python -B -m architecture_seed_repeat.tests."""
import math
from pathlib import Path
import shutil
import uuid
from architecture_attention.common import source_binding as previous_sources, validate_output as previous_output
from architecture_attention.common import VARIANTS, LABELS, COUNTS
from experiments.arch70m.common import read, write, sha, digest, stamp, lock, commit_id

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_attention_20261102')
OUT = Path('/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_attention_seed42_20261103')
PREVIOUS_SHA = '28609d7f776013915561eb038396d54181dcee7b1c32d071be0a9da5dc12faad'
DATA_SHA = '4beaac6b54a1f46492533aeb77c2de2c6ac2a0bd8266fd57a6bb312679435c51'
SEED, MICRO, GLOBAL, STEPS = 42, 32, 512, 1276


def trial(out, variant, seed=SEED): return Path(out) / 'trials' / f'screen_{variant}_s{seed}'


def source_binding():
    sources = previous_sources()
    if digest(sources) != PREVIOUS_SHA: raise ValueError('The original seed43 source changed.')
    files = list((ROOT / 'architecture_seed_repeat').glob('*.py'))
    files += [ROOT / 'architecture_seed_repeat/protocol.json', ROOT / 'launch_arch70m_seed42.sh']
    return dict(sources, **{str(p.relative_to(ROOT)): sha(p) for p in files})


def validate_output(out):
    out = previous_output(out)
    if out == OLD or OLD in out.parents or out in OLD.parents:
        raise ValueError('The ongoing seed43 experiment is read-only.')
    return out


def prepare(out):
    out = validate_output(out)
    protocol = read(ROOT / 'architecture_seed_repeat/protocol.json')
    expected = dict(added_seed=SEED, reference_seed=43, variants=list(VARIANTS), epochs=1,
        steps=STEPS, micro_batch=MICRO, global_batch=GLOBAL, data_sha256=DATA_SHA, previous_source_sha256=PREVIOUS_SHA)
    if any(protocol.get(k) != v for k, v in expected.items()): raise ValueError('Seed replication protocol changed.')
    if sha(OLD / 'data/index.json') != DATA_SHA: raise ValueError('The fixed 10% data index changed.')
    if digest(previous_sources()) != PREVIOUS_SHA: raise ValueError('Frozen seed43 sources changed.')
    with lock(out / 'data.lock'):
        folder = out / 'data'; folder.mkdir(parents=True, exist_ok=True)
        if not (folder / 'index.json').exists(): shutil.copyfile(OLD / 'data/index.json', folder / 'index.json')
        if sha(folder / 'index.json') != DATA_SHA: raise ValueError('Replication data index is not identical.')
        if not (folder / 'verified.json').exists(): shutil.copyfile(OLD / 'data/verified.json', folder / 'verified.json')


def snapshot(out):
    sources = source_binding(); signature = digest(sources); target = Path(out) / 'sources' / signature
    with lock(Path(out) / 'sources.lock'):
        if not target.exists():
            pending = target.parent / ('.pending_' + uuid.uuid4().hex); pending.mkdir(parents=True)
            for name in sources:
                dest = pending / name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, dest)
            write(pending / 'provenance.json', dict(files=sources, source_sha256=signature,
                  previous_source_sha256=PREVIOUS_SHA, commit=commit_id(), created=stamp()))
            pending.rename(target)
        if any(sha(target / name) != value for name, value in sources.items()): raise ValueError('Immutable replication snapshot changed.')
    return target, signature


def valid_profile(profile, variant, signature):
    expected = dict(variant=variant, seed=SEED, micro_batch=MICRO, global_batch=GLOBAL, length=2048,
        grad_accum=16, final_micro_batch=32, source_sha256=signature, data_sha256=DATA_SHA)
    if any(profile.get(k) != v for k, v in expected.items()): raise ValueError('Seed42 GPU profile binding changed.')
    candidates = profile.get('candidates', [])
    if len(candidates) != 1: raise ValueError('One common-micro qualification is required.')
    p = candidates[0]
    if (p.get('status') != 'passed' or p.get('updates') != 3 or not p.get('safe') or
            p['reserved_bytes'] + p['external_bytes'] > .85 * p['total_bytes']):
        raise ValueError('Unsafe or incomplete GPU qualification.')


def qualified(out, signature):
    q = read(Path(out) / 'qualification.json')
    if (q.get('status'), q.get('seed'), q.get('source_sha256'), q.get('data_sha256')) != ('passed', SEED, signature, DATA_SHA):
        raise ValueError('Missing seed42 qualification.')
    if set(q.get('profiles', {})) != set(VARIANTS): raise ValueError('All five models must qualify.')
    for v in VARIANTS:
        p = trial(out, v) / 'profile.json'
        if sha(p) != q['profiles'][v]: raise ValueError('Qualified profile checksum changed.')
        valid_profile(read(p), v, signature)


def completed(out, variant, signature):
    t = trial(out, variant)
    if not (t / 'status.json').exists() or read(t / 'status.json')['status'] != 'completed': return False
    binding = read(t / 'binding.json')
    expected = dict(variant=variant, seed=SEED, epochs=1, total_steps=STEPS, micro_batch=MICRO,
                    source_sha256=signature, data_sha256=DATA_SHA)
    if any(binding.get(k) != v for k, v in expected.items()): raise ValueError('Completed seed42 binding differs.')
    if binding['settings'] != read(ROOT / 'experiments/arch70m/base.json'): raise ValueError('Training settings changed.')
    status = read(t / 'status.json')
    if status['step'] != STEPS or status['parameters'] != COUNTS[variant]: raise ValueError('Final step/model size changed.')
    valid_profile(read(t / 'profile.json'), variant, signature)
    if sha(t / 'profile.json') != binding['profile_sha256']: raise ValueError('Profile checksum differs.')
    checkpoint = t / 'checkpoints/step_001276'; receipt = read(checkpoint / 'receipt.json')
    if receipt['status'] != 'complete' or len(receipt['files']) != 6: raise ValueError('Incomplete final checkpoint.')
    if any(sha(checkpoint / name) != value for name, value in receipt['files'].items()): raise ValueError('Corrupt checkpoint file.')
    if read(checkpoint / 'binding.json') != binding: raise ValueError('Checkpoint binding changed.')
    state = read(checkpoint / 'state.json')
    if tuple(state[k] for k in ('step', 'epoch', 'next_batch', 'seen_records', 'seen_tokens')) != (STEPS, 1, 0, 653288, 214482921):
        raise ValueError('Final training data cursor changed.')
    m = read(t / 'metrics/screen_001276.json'); observations = read(t / 'metrics/screen_001276_observations.json')
    if m['step'] != STEPS or m['partition'] != 'screen' or not observations['generation'] or not observations['representation']:
        raise ValueError('Missing final screen evaluation.')
    if not all(math.isfinite(x) for x in (m['token']['all']['loss'], m['generation']['conditional_ce'], m['representation']['macro_auroc'])):
        raise ValueError('Non-finite screen evaluation.')
    return True


def predecessor_ready():
    from architecture_attention.common import completed as old_completed
    state = read(OLD / 'execution.json')
    if state['phase'] != 'finished' or state.get('failed_variants') or state['source_sha256'] != PREVIOUS_SHA:
        raise ValueError('Seed43 must finish cleanly before seed42 starts.')
    if not all(old_completed(OLD, v, PREVIOUS_SHA) for v in VARIANTS): raise ValueError('Seed43 results are incomplete.')
