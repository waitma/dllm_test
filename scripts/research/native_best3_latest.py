"""Keep three lowest-eval-loss native checkpoints plus the latest full state.

Activate conda pllm. Import retain() from the rank-zero native on_save callback,
only after all ranks have verified the new checkpoint. Run the companion
test_native_best3_latest.py with Python for temporary-directory checks.
No model imports, checkpoint slimming, or inferred ownership are used here.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import uuid


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def canonical(path):
    path = Path(path)
    require(path.is_absolute() and path == path.resolve()
            and not any(p.is_symlink() for p in (path, *path.parents)),
            'Canonical nonsymlink path required: '+str(path))
    return path


def atomic(path, value):
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def inspect_checkpoint(path, provenance_sha, sidecar_name):
    path = canonical(path)
    match = re.fullmatch(r'checkpoint-([1-9][0-9]*)', path.name)
    require(path.is_dir() and match is not None, 'Invalid native checkpoint directory')
    require(all(p.is_file() and not p.is_symlink() for p in path.iterdir()),
            'Checkpoint contains nonregular members')
    sidecar = json.loads((path/sidecar_name).read_text())
    require(sidecar['world_size'] == 8 and sidecar['writer_rank'] == 0
            and sidecar['rank_acknowledgments'] == list(range(8)), 'Incomplete native acknowledgments')
    require(binding_digest(json.loads((path/'run_provenance.json').read_text())) == provenance_sha,
            'Checkpoint owner binding differs')
    for name in ('trainer_state.json', 'run_provenance.json'):
        expected = sidecar['files'][name]
        if isinstance(expected, dict):
            require(set(expected) == {'bytes', 'sha256'}
                    and type(expected['bytes']) is int
                    and (path/name).stat().st_size == expected['bytes'],
                    'Checkpoint metadata inventory size differs')
            expected = expected['sha256']
        require(isinstance(expected, str) and sha(path/name) == expected,
                'Checkpoint metadata inventory differs')
    state = json.loads((path/'trainer_state.json').read_text())
    step = state['global_step']
    require(type(step) is int and step == int(match[1]) == sidecar['step'], 'Checkpoint step differs')
    values = [record['eval_loss'] for record in state['log_history']
              if record.get('step') == step and 'eval_loss' in record]
    require(values and all(type(v) in (float, int) and math.isfinite(v) for v in values),
            'Finite exact-step eval_loss required before any pruning')
    require(all(v == values[-1] for v in values), 'Conflicting exact-step validation metrics')
    return dict(path=str(path), step=step, eval_loss=float(values[-1]),
                sidecar_sha256=sha(path/sidecar_name), binding_sha256=provenance_sha)


def plan(owners, current, sidecar_name='formal_checkpoint.json'):
    """Validate every candidate before choosing anything to remove."""
    current = canonical(current)
    entries = []
    require(owners and isinstance(owners, dict), 'Explicit owner roots and provenance hashes required')
    for root, provenance_sha in owners.items():
        root = canonical(root)
        require(binding_digest(json.loads((root/'run_provenance.json').read_text())) == provenance_sha,
                'Run owner changed')
        for path in sorted(root.glob('checkpoint-*')):
            entries.append(inspect_checkpoint(path, provenance_sha, sidecar_name))
    require(entries and len({x['step'] for x in entries}) == len(entries), 'Duplicate or absent checkpoint steps')
    latest = max(entries, key=lambda x: x['step'])
    require(latest['path'] == str(current), 'Only the newest complete checkpoint may trigger retention')
    best = sorted(entries, key=lambda x: (x['eval_loss'], x['step']))[:3]
    keep = {x['path'] for x in best} | {latest['path']}
    return dict(schema='v7.native.best3_latest.v1', metric='eval_loss', mode='min', best_k=3,
                latest=latest['path'], best=[x['path'] for x in best], all_retained_states='full',
                owners=owners, candidates=entries, keep=[x for x in entries if x['path'] in keep],
                remove=[x for x in entries if x['path'] not in keep])


def retain(owners, current, sidecar_name='formal_checkpoint.json'):
    """Persist a decision and JSON evidence before deleting only owned nonwinners."""
    current = canonical(current)
    root = current.parent
    decision = plan(owners, current, sidecar_name)
    evidence = root/'checkpoint_retention_evidence'
    evidence.mkdir(exist_ok=True)
    for item in decision['candidates']:
        path = Path(item['path'])
        archive = evidence/(path.name+'_'+item['binding_sha256'][:12])
        archive.mkdir(exist_ok=True)
        for source in path.glob('*.json'):
            target = archive/source.name
            if target.exists():
                require(sha(target) == sha(source), 'Archived checkpoint evidence changed')
            else:
                target.write_bytes(source.read_bytes())
    receipt = evidence/(current.name+'_decision.json')
    require(not receipt.exists(), 'Retention decision already executed; inspect before retry')
    # Re-read metadata after archiving; a changing checkpoint is never pruned.
    require(plan(owners, current, sidecar_name) == decision, 'Checkpoint set changed during retention')
    atomic(receipt, {**decision, 'status': 'prepared'})
    for item in decision['remove']:
        path = canonical(item['path'])
        require(str(path.parent) in owners and path != current, 'Prune path is not owned')
        require(inspect_checkpoint(path, owners[str(path.parent)], sidecar_name) == item,
                'Prune candidate changed')
        shutil.rmtree(path)
    final = {**decision, 'status': 'applied'}
    atomic(receipt, final)
    atomic(root/'checkpoint_retention.json', final)
    return final
