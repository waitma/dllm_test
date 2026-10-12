"""Prepare an immutable 270M launch; activate pllm and run this file --help.

Copies the completed 75.6M run's frozen sources, preserving its data/layout and
audit. Only explicit model-size/startup changes are applied. No GPU job/ticket
is submitted. Profile and training packages use separate output identities.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
PARENT = ROOT/'output/v5_full_binding_20261007'
BASELINE = PARENT/'launch_v3'
CONFIG = ROOT/'configs/models/v5_noesmc_270m_20261010'
COUNT = 268590112


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def replace_once(path, before, after):
    text = path.read_text()
    if text.count(before) != 1:
        raise ValueError('Expected one source anchor: '+str(path)+' / '+before)
    path.write_text(text.replace(before, after, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('profile', 'train'), required=True)
    parser.add_argument('--micro-batch', type=int, default=64)
    args = parser.parse_args()
    if args.micro_batch not in (8, 16, 32, 64):
        raise ValueError('Micro batch must preserve global512 on eight ranks')
    bundle = PARENT/f'launch_270m_{args.stage}_bs{args.micro_batch}_20261010'
    if bundle.exists():
        raise ValueError('Do not overwrite/retry an existing package')
    original = json.loads((BASELINE/'source_manifest.json').read_text())['files']
    for name, expected in original.items():
        if sha(name) != expected:
            raise ValueError('Baseline source changed: '+name)
    bundle.mkdir()
    source = bundle/'source'
    shutil.copytree(BASELINE/'source', source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    driver = source/'scripts/research/run_v7_formal.py'
    text = driver.read_text().replace('check_v5_formal_parent.py', 'check_v5_270m_formal_parent.py')
    text = text.replace("scripts/research/v5_data_binding.py", "scripts/research/v5_270m_data_binding.py")
    driver.write_text(text)
    replace_once(driver, 'd_model=768, n_layers=8, n_heads=12,', 'd_model=1024, n_layers=16, n_heads=16,')
    replace_once(driver, 'mlp_hidden_size=3072, max_length=2048,', 'mlp_hidden_size=4096, max_length=2048,')
    replace_once(driver, '        model.backbone.set_activation_checkpointing',
                 f"        require(sum(p.numel() for p in model.parameters()) == {COUNT}, '270M parameter count differs')\n        model.backbone.set_activation_checkpointing")
    launcher = source/'scripts/research/launch_v5_direct.py'
    replace_once(launcher, 'd_model=768, n_layers=8, n_heads=12,', 'd_model=1024, n_layers=16, n_heads=16,')
    replace_once(launcher, 'mlp_hidden_size=3072, full_capacity=302,', 'mlp_hidden_size=4096, full_capacity=302,')
    replace_once(launcher, "    require(all(p.requires_grad for p in model.parameters()),",
                 f"    require(sum(p.numel() for p in model.parameters()) == {COUNT}, '270M profile parameter count differs')\n    require(all(p.requires_grad for p in model.parameters()),")
    replace_once(launcher, '    return launch, argv, driver, args', '''    require(launch['stage'] in ('profile', 'train'), 'Explicit launch stage required')
    if launch['stage'] == 'train':
        require(launch['profile_contract'] == json.loads(Path(launch['profile_bundle']).joinpath('direct_launch.json').read_text())['profile_contract'], 'Profile contract differs')
        for receipt in launch['profile_evidence']:
            require(sha(receipt['path']) == receipt['sha256'], 'Accepted profile evidence changed')
            report = json.loads(Path(receipt['path']).read_text())
            require(report.get('status') == 'passed' and report['parameter_count'] == 268590112
                    and report['micro_batch'] == args.micro_batch and report['sequence_length'] == 1218,
                    '270M accepted profile does not match this training')
        require(len(launch['profile_evidence']) == 8, 'Eight accepted profiles are required')
        complete = Path(launch['profile_bundle'])/'PROFILE_COMPLETE.json'
        require(sha(complete) == launch['profile_complete_sha256'], 'Profile completion changed')
    return launch, argv, driver, args''')
    replace_once(launcher, '    profile(args, directory, rank)', '''    if launch['stage'] == 'profile':
        profile(args, directory, rank)
    else:
        write_once(directory/f'rank{rank}_accepted_profile.json', launch['profile_evidence'][rank])''')
    replace_once(launcher, "        # The native driver verifies all sources", '''        if launch['stage'] == 'profile':
            if rank == 0:
                write_once(options.bundle/'PROFILE_COMPLETE.json', dict(status='passed', world_size=8,
                           parameter_count=268590112, micro_batch=args.micro_batch,
                           checkpoint_saved=False, corpus_training_started=False))
            return
        # The native driver verifies all sources''')
    # The original audit executables remain unchanged. New helper identities
    # retain the same full-content gate and bind one explicit larger preset.
    for name in ('check_v5_270m_formal_parent.py', 'v5_270m_data_binding.py'):
        shutil.copy2(ROOT/'scripts/research'/name, source/'scripts/research'/name)
    config = json.loads((BASELINE/'research_config.json').read_text())
    config['base'].update(d_model=1024, n_layers=16, n_heads=16, mlp_hidden_size=4096)
    write(bundle/'research_config.json', config)
    binding = json.loads((BASELINE/'data_binding.json').read_text())
    binding['model_config'] = {'path': str(bundle/'research_config.json'), 'sha256': sha(bundle/'research_config.json')}
    write(bundle/'data_binding.json', binding)
    from scripts.research.v5_270m_data_binding import verify_binding
    verify_binding(bundle/'data_binding.json', sha(bundle/'data_binding.json'),
                   Path(binding['data_dir']), 302, 'cv_segment_global', full_hash=False)
    files = {str(p): sha(p) for p in sorted(source.rglob('*.py'))}
    for name, expected in original.items():
        if not Path(name).is_relative_to(BASELINE/'source'):
            files[name] = expected
    for name in ('check_v5_270m_formal_parent.py', 'v5_270m_data_binding.py', 'prepare_v5_270m_launch.py'):
        path = ROOT/'scripts/research'/name
        files[str(path)] = sha(path)
    write(bundle/'source_manifest.json', {'files': files})
    argv = json.loads((BASELINE/'launch_training_args.json').read_text())
    output = ROOT/'output'/('immune_v5_270m_noesmc_300k_b512' if args.stage == 'train' else f'immune_v5_270m_profile_bs{args.micro_batch}_unused')
    changes = {'--source-root': source, '--source-manifest': bundle/'source_manifest.json',
               '--source-manifest-sha256': sha(bundle/'source_manifest.json'),
               '--data-binding': bundle/'data_binding.json', '--data-binding-sha256': sha(bundle/'data_binding.json'),
               '--output-dir': output, '--micro-batch': args.micro_batch, '--grad-accum': 64//args.micro_batch}
    for key, value in changes.items():
        argv[argv.index(key)+1] = str(value)
    write(bundle/'launch_training_args.json', argv)
    runtime = ROOT/'output/immune_v5_270m_noesmc_300k_b512_supervision'/f'{args.stage}_bs{args.micro_batch}'
    logical = {str(Path(path).relative_to(source)): value for path, value in files.items() if Path(path).is_relative_to(source)}
    contract = {'config': config, 'micro_batch': args.micro_batch, 'sequence_length': 1218,
                'parameter_count': COUNT, 'source_digest': hashlib.sha256(json.dumps(logical, sort_keys=True).encode()).hexdigest()}
    launch = {'schema': 'immune.v5.direct_launch.v1', 'stage': args.stage,
              'initialization': 'random_from_scratch', 'model_parameters_frozen': False,
              'runtime_dir': str(runtime), 'arguments_sha256': sha(bundle/'launch_training_args.json'),
              'launcher_sha256': sha(launcher), 'checkpoint_retention': 'best3_eval_loss_plus_latest_full',
              'data_audit_scope': binding['scope'], 'profile_contract': contract}
    if args.stage == 'train':
        profile_bundle = PARENT/f'launch_270m_profile_bs{args.micro_batch}_20261010'
        profile_launch = json.loads((profile_bundle/'direct_launch.json').read_text())
        if profile_launch['profile_contract'] != contract:
            raise ValueError('Profile and training source/config contracts differ')
        complete = json.loads((profile_bundle/'PROFILE_COMPLETE.json').read_text())
        if complete.get('status') != 'passed' or complete.get('world_size') != 8:
            raise ValueError('Eight-rank synthetic acceptance required')
        launch['profile_bundle'] = str(profile_bundle)
        launch['profile_complete_sha256'] = sha(profile_bundle/'PROFILE_COMPLETE.json')
        launch['profile_evidence'] = []
        for rank in range(8):
            path = Path(profile_launch['runtime_dir'])/f'profile_rank{rank}.json'
            report = json.loads(path.read_text())
            if report.get('status') != 'passed' or report['rank'] != rank or report['parameter_count'] != COUNT:
                raise ValueError('Invalid profile receipt')
            launch['profile_evidence'].append({'path': str(path), 'sha256': sha(path)})
    write(bundle/'direct_launch.json', launch)
    changed = []
    for path in sorted(source.rglob('*.py')):
        relative = path.relative_to(source)
        original_path = BASELINE/'source'/relative
        if not original_path.exists() or sha(path) != sha(original_path):
            changed.append(str(relative))
    write(bundle/'preparation.json', {'at_utc': datetime.now(timezone.utc).isoformat(),
          'stage': args.stage, 'source_baseline': str(BASELINE), 'changed_source_files': changed,
          'data_audit_unchanged': True, 'training_submitted': False, 'parameter_count': COUNT,
          'binding_check': 'stat_only_here; full_shard_hashes_mandatory_in_native_training_gate'})
    print(json.dumps({'bundle': str(bundle), 'stage': args.stage, 'parameter_count': COUNT,
                      'micro_batch': args.micro_batch, 'grad_accum': 64//args.micro_batch,
                      'changed_source_files': changed}, ensure_ascii=False))


if __name__ == '__main__':
    main()
