"""Validate an explicit old-DDP to where-DDP checkpoint transition.

Activate conda pllm. Imported by the independent frozen continuation driver and
its slot wrapper. This reads metadata only: it never creates an audit, changes
old provenance, loads weights, stops a process, or admits a different objective.
The native driver still checks all checkpoint bytes and observes actual restore.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path

OLD_MANIFEST_SHA = '992bc3c5718b71c379c7a2ee4f39d8e5866f974266500d2c2ec69d5ab4db700b'
MODEL_REL = 'dllm/pipelines/immune_llada/research_refined/model.py'
DRIVER_REL = 'scripts/research/run_v7_formal.py'
ADDED = {'scripts/research/native_best3_latest.py', 'scripts/research/v7_where_transition.py'}
POLICY = {'metric': 'eval_loss', 'mode': 'min', 'best_k': 3, 'keep_latest': True, 'full_state': True}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_ref(ref):
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'Exact file reference required')
    path = Path(ref['path'])
    require(path.is_absolute() and path == path.resolve() and path.is_file() and not path.is_symlink()
            and sha(path) == ref['sha256'], 'Changed transition evidence: '+str(path))
    return json.loads(path.read_text())


def model_expression_only(old, new):
    tree = ast.parse(Path(old).read_text())
    denoise = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'denoise')
    assignments = [n for n in ast.walk(denoise) if isinstance(n, ast.Assign)
                   and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id == 'base_bias']
    require(len(assignments) == 1, 'Ambiguous original chain bias')
    assignments[0].value = ast.parse('torch.where(same[:, None], self.chain_bias[1][None, :, None, None], '
                                    'self.chain_bias[0][None, :, None, None])', mode='eval').body
    require(ast.dump(tree, include_attributes=False) == ast.dump(ast.parse(Path(new).read_text()),
                                                               include_attributes=False),
            'Model contains changes beyond the verified where expression')


def transition(args, new_binding=None):
    require(args.transition is not None and args.transition_sha256, 'Explicit transition identity required')
    spec = read_ref({'path': str(args.transition), 'sha256': args.transition_sha256})
    require(spec['schema'] == 'v7.where.ddp_transition.v1', 'Transition schema differs')
    require(spec['old_source_manifest']['sha256'] == OLD_MANIFEST_SHA, 'Unexpected original source release')
    require(spec['new_source_manifest'] == {'path': str(args.source_manifest),
                                          'sha256': args.source_manifest_sha256}, 'New release differs')
    require(spec['new_source_root'] == str(args.source_root)
            and spec['new_output_dir'] == str(args.output_dir), 'New transition destination differs')
    old = read_ref(spec['old_provenance'])
    require(old['source_manifest_sha256'] == OLD_MANIFEST_SHA and old['world_size'] == 8
            and old['variant'] == 'f011', 'Transition is restricted to the original f011 world8 run')
    old_manifest, new_manifest = read_ref(spec['old_source_manifest']), read_ref(spec['new_source_manifest'])
    old_root, new_root = Path(old['source_root']), args.source_root
    expected = {}
    changed = []
    for name, digest in old_manifest['files'].items():
        path = Path(name)
        require(sha(path) == digest, 'Original bound source changed')
        if path.is_relative_to(old_root):
            relative = str(path.relative_to(old_root)); target = new_root/relative
            expected[str(target)] = new_manifest['files'].get(str(target))
            require(target.is_file() and not target.is_symlink()
                    and sha(target) == expected[str(target)], 'New frozen source mismatch')
            if sha(target) != digest:
                changed.append(relative)
        else:
            expected[name] = digest
            require(new_manifest['files'].get(name) == digest, 'Original external gate/contract changed')
    require(set(changed) == {MODEL_REL, DRIVER_REL}, 'Unexpected changed original modules')
    for relative in ADDED:
        path = new_root/relative
        require(path.is_file() and not path.is_symlink(), 'Missing transition module')
        expected[str(path)] = sha(path)
    require(new_manifest['files'] == expected, 'New source manifest contains unexpected/omitted sources')
    model_expression_only(old_root/MODEL_REL, new_root/MODEL_REL)
    anchor = read_ref(spec['anchor_sidecar'])
    require(anchor['world_size'] == 8 and anchor['rank_acknowledgments'] == list(range(8))
            and spec['anchor_checkpoint'] == str(Path(spec['old_output_dir'])/f"checkpoint-{anchor['step']}"),
            'Original complete checkpoint identity differs')
    require(spec['old_output_dir'] == str(Path(spec['anchor_checkpoint']).parent), 'Original output identity differs')
    if new_binding is not None:
        require(new_binding['checkpoint_policy'] == POLICY
                and new_binding['transition'] == {'path': str(args.transition), 'sha256': args.transition_sha256},
                'Transition/policy not recorded in new provenance')
        mutable = {'source_files', 'source_root', 'source_manifest_sha256', 'audit_execution',
                   'checkpoint_policy', 'transition'}
        require({k:v for k,v in old.items() if k not in mutable} ==
                {k:v for k,v in new_binding.items() if k not in mutable},
                'Transition changed model/data/optimizer/sampling/schedule/precision')
        for key in ('schema', 'status', 'metadata', 'repo_root', 'data_dir'):
            require(new_binding['audit_execution'][key] == old['audit_execution'][key], 'Parent audit semantics changed')
    return spec, old, anchor


def validate_migration_segment(plan, args, *, binding_digest, legacy):
    """Reuse every original resume gate against the old checkpoint's true owner."""
    spec, old, anchor = transition(args)
    if str(args.resume_from_checkpoint) != spec['anchor_checkpoint']:
        return legacy(plan, args, binding_digest=binding_digest)
    old_plan, old_args = copy.deepcopy(plan), copy.copy(args)
    work = old_plan['work']
    for key in ('source_root', 'output_dir'):
        value = old['source_root'] if key == 'source_root' else spec['old_output_dir']
        work[key] = value; setattr(old_args, key, Path(value))
    work['source_manifest'] = spec['old_source_manifest']
    old_args.source_manifest = Path(spec['old_source_manifest']['path'])
    old_args.source_manifest_sha256 = spec['old_source_manifest']['sha256']
    result = legacy(old_plan, old_args, binding_digest=binding_digest)
    require(result['prior_step'] == anchor['step']
            and sha(Path(args.resume_from_checkpoint)/'formal_checkpoint.json') == spec['anchor_sidecar']['sha256'],
            'Actual migration anchor differs')
    return result
