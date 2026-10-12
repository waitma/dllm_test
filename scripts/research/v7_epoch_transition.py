"""Validate the explicit 25042-to-50082 schedule extension using real metadata.

Activate conda pllm. This module is imported by the independent frozen epoch-two
driver and pool wrapper. It never creates certificates or edits old provenance.
The parent final checkpoint must actually exist before a transition can be sealed.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
PARENT = ROOT/'output/v7_formal_eight_gpu_20261002/where_continuation_20261006'
DESTINATION = ROOT/'output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006'
ANCESTOR_RUN = ROOT/'output/v7_initial_training_20261005/run'
OLD_MANIFEST_SHA = '053f95d2237eed6f4bd45d53ea07126447c8ddaf4efa67f62c571f94361d0c5b'
DRIVER_REL = 'scripts/research/run_v7_formal.py'
ADDED = {'scripts/research/v7_epoch_transition.py', 'scripts/research/v7_epoch_schedule.py'}
POLICY = {'metric': 'eval_loss', 'mode': 'min', 'best_k': 3, 'keep_latest': True, 'full_state': True}
LR_EXTENSION = {'kind': 'explicit_warmup_cosine_restart', 'first_phase_steps': 25042,
                'total_steps': 50082, 'warmup_steps': 1000, 'data_epoch_steps': 25041,
                'target_data_epochs': 2, 'preserve_skip_loader_epoch': True}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_ref(reference):
    require(type(reference) is dict and set(reference) == {'path', 'sha256'}, 'Exact file reference required')
    path = Path(reference['path'])
    require(path.is_absolute() and path == path.resolve() and path.is_file() and not path.is_symlink()
            and sha(path) == reference['sha256'], 'Changed epoch transition reference: '+str(path))
    return json.loads(path.read_text())


def transition(args, new_binding=None):
    spec = read_ref({'path': str(args.transition), 'sha256': args.transition_sha256})
    require(spec['schema'] == 'v7.epoch2.ddp_transition.v1'
            and spec['old_source_manifest']['sha256'] == OLD_MANIFEST_SHA,
            'Unexpected parent release or migration type')
    require(spec['new_source_manifest'] == {'path': str(args.source_manifest), 'sha256': args.source_manifest_sha256}
            and spec['new_source_root'] == str(args.source_root) == str(DESTINATION/'source_snapshot_v1')
            and spec['new_output_dir'] == str(args.output_dir) == str(DESTINATION/'run')
            and spec['old_output_dir'] == str(PARENT/'run'), 'Unexpected continuation owner')
    request = read_ref(spec['user_request'])
    require(request['authorized_cumulative_target_step'] == 50082
            and request['evaluation_checkpoint_step'] == 50082, 'Two-epoch request differs')
    old = read_ref(spec['old_provenance'])
    require(old['source_manifest_sha256'] == OLD_MANIFEST_SHA and old['world_size'] == 8
            and old['variant'] == 'f011' and old['train_records'] == 6410749
            and old['schedule']['max_steps'] == 25042 and old['checkpoint_policy'] == POLICY,
            'Parent model, population, or fixed horizon differs')
    require(args.max_steps == 50082 and args.warmup_steps == 1000 and args.learning_rate == 1e-4
            and spec['lr_extension'] == LR_EXTENSION, 'Explicit scheduler extension differs')
    ancestor = read_ref(spec['ancestor_provenance'])
    require(spec['retention_owners'] == {str(PARENT/'run'): digest(old), str(ANCESTOR_RUN): digest(ancestor)},
            'Retention may only span the explicit original lineage')
    old_manifest = read_ref(spec['old_source_manifest'])
    new_manifest = read_ref(spec['new_source_manifest'])
    old_root = Path(old['source_root'])
    expected, changed = {}, []
    for filename, expected_sha in old_manifest['files'].items():
        path = Path(filename)
        require(sha(path) == expected_sha, 'Parent source changed')
        if path.is_relative_to(old_root):
            relative = str(path.relative_to(old_root))
            target = args.source_root/relative
            require(target.is_file() and not target.is_symlink(), 'Missing frozen continuation source')
            expected[str(target)] = sha(target)
            if expected[str(target)] != expected_sha:
                changed.append(relative)
        else:
            expected[str(path)] = expected_sha
    require(set(changed) == {DRIVER_REL}, 'Changes beyond the explicit driver extension')
    for relative in ADDED:
        path = args.source_root/relative
        require(path.is_file() and not path.is_symlink(), 'Missing epoch extension module')
        expected[str(path)] = sha(path)
    require(new_manifest['files'] == expected, 'Unexpected or omitted source bindings')
    anchor = read_ref(spec['anchor_sidecar'])
    require(anchor['step'] == 25042 and anchor['binding_sha256'] == digest(old)
            and anchor['world_size'] == 8 and anchor['rank_acknowledgments'] == list(range(8))
            and spec['anchor_checkpoint'] == str(PARENT/'run/checkpoint-25042'), 'Final parent checkpoint differs')
    if new_binding is not None:
        require(new_binding['checkpoint_policy'] == POLICY and new_binding['lr_extension'] == LR_EXTENSION
                and new_binding['transition'] == {'path': str(args.transition), 'sha256': args.transition_sha256},
                'Continuation policy not recorded')
        mutable = {'source_files', 'source_root', 'source_manifest_sha256', 'audit_execution',
                   'transition', 'schedule', 'lr_extension'}
        require({k: v for k, v in old.items() if k not in mutable} ==
                {k: v for k, v in new_binding.items() if k not in mutable},
                'Model/data/optimizer/sampling/precision changed beyond the request')
        require(new_binding['schedule'] == {**old['schedule'], 'max_steps': 50082}, 'Training schedule fields differ')
        for key in ('schema', 'status', 'metadata', 'repo_root', 'data_dir'):
            require(new_binding['audit_execution'][key] == old['audit_execution'][key], 'Parent audit semantics changed')
    return spec, old, anchor


def validate_migration_segment(plan, args, *, binding_digest, legacy):
    """Same-run resumes use the original gate; only the exact final parent migrates."""
    spec, old, anchor = transition(args)
    if str(args.resume_from_checkpoint) != spec['anchor_checkpoint']:
        return legacy(plan, args, binding_digest=binding_digest)
    work, target = plan['work'], plan['segment_steps']
    require(target == (args.stop_after or args.max_steps) and 25042 < target <= args.max_steps == 50082,
            'Invalid first continuation target')
    for key in ('source_root', 'repo_root', 'data_dir', 'output_dir', 'variant', 'seed',
                'micro_batch', 'grad_accum', 'global_batch', 'max_steps'):
        require(str(getattr(args, key)) == str(work[key]), 'New driver work differs: '+key)
    require(work['source_manifest'] == spec['new_source_manifest'], 'New source identity differs')
    for ticket in plan['tickets'].values():
        require(ticket['steps'] == target-25042 and ticket['data_dir'] == str(args.data_dir)
                and ticket['resume_from_checkpoint'] == spec['anchor_checkpoint'], 'Migration ticket differs')
    checkpoint = args.resume_from_checkpoint
    require(checkpoint.is_dir() and not checkpoint.is_symlink()
            and checkpoint.parent == PARENT/'run', 'Final parent checkpoint is absent or unowned')
    references = {str(checkpoint/'formal_checkpoint.json'): spec['anchor_sidecar']['sha256']}
    require(sha(checkpoint/'formal_checkpoint.json') == spec['anchor_sidecar']['sha256'], 'Actual anchor changed')
    metadata = {}
    for name in ('trainer_state.json', 'run_provenance.json'):
        path = checkpoint/name
        metadata[name] = read_ref({'path': str(path), 'sha256': anchor['files'][name]})
        references[str(path)] = anchor['files'][name]
    require(metadata['trainer_state.json']['global_step'] == 25042
            and metadata['run_provenance.json'] == old and anchor['binding_sha256'] == binding_digest(old),
            'Actual native parent metadata differs')
    for key in ('repo_root', 'data_dir', 'variant', 'seed', 'micro_batch', 'grad_accum', 'global_batch'):
        require(str(old[key]) == str(work[key]), 'Model/data/batch differs at transition: '+key)
    for key in ('warmup_steps', 'learning_rate', 'eval_every', 'save_every'):
        require(old['schedule'][key] == getattr(args, key), 'Original schedule field changed: '+key)
    require(all(sha(Path(path)) == value for path, value in references.items()), 'Anchor changed during metadata check')
    return dict(prior_step=25042, target_step=target, additional_updates=target-25042,
                checkpoint_metadata_sha256=references, migration='explicit_user_authorized_epoch2')
