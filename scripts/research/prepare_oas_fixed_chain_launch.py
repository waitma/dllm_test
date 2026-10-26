"""Freeze the OAS-only scratch launch; no training job is submitted.

Activate pllm; python -B -m scripts.research.prepare_oas_fixed_chain_launch.
The immutable parent supplies the architecture, optimizer and native DDP driver.
"""
import json
from pathlib import Path
import shutil

from scripts.research.oas_pretraining_data import describe_selection
from scripts.research.v5_data_binding import sha, require

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
BASE = ROOT / 'output/v5_full_binding_20261007/launch_fixed_chain_75m_20261012'
BUNDLE = BASE.parent / 'launch_fixed_chain_75m_oas_queue012_20261015'
CONFIG = ROOT / 'configs/models/v5_fixed_chain_75m_oas_20261015'
OUTPUT = ROOT / 'output/immune_v5_fixed_chain_75m_oas_300k_b512_20261015'
TESTS = ('test_fixed_chain.py', 'test_fixed_chain_validation.py', 'test_oas_pretraining_data.py')


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def replace(path, old, new):
    content = path.read_text()
    require(content.count(old) == 1, 'Patch anchor differs: ' + old)
    path.write_text(content.replace(old, new))


def main():
    require(not BUNDLE.exists() and not CONFIG.exists(), 'Refusing to overwrite an OAS launch package')
    original = json.loads((BASE / 'source_manifest.json').read_text())['files']
    for filename, expected in original.items():
        require(sha(filename) == expected, 'Original frozen source changed: ' + filename)
    BUNDLE.mkdir()
    CONFIG.mkdir(parents=True)
    source = BUNDLE / 'source'
    shutil.copytree(BASE / 'source', source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(ROOT / 'scripts/research/oas_pretraining_data.py', source / 'scripts/research/oas_pretraining_data.py')
    for name in TESTS:
        shutil.copy2(ROOT / 'scripts/tests/immune_llada' / name, source / 'scripts/tests/immune_llada' / name)

    # Make source selection executable and bind it into every future checkpoint.
    driver = source / 'scripts/research/run_v7_formal.py'
    replace(driver, "    parser.add_argument('--source-manifest-sha256', required=True)",
            "    parser.add_argument('--source-manifest-sha256', required=True)\n"
            "    parser.add_argument('--data-selection', type=Path, required=True)\n"
            "    parser.add_argument('--data-selection-sha256', required=True)")
    replace(driver, "    args = parser.parse_args(argv)",
            "    args = parser.parse_args(argv)\n"
            "    require(args.data_selection.is_absolute() and args.data_selection == args.data_selection.resolve(),\n"
            "            'data-selection must be canonical absolute')")
    replace(driver, "def verify_sources(args, loaded=False):", "def verify_sources(args, loaded=False):\n"
            "    require(sha(args.data_selection) == args.data_selection_sha256, 'OAS selection SHA differs')")
    replace(driver, "    from torch.utils.data import Subset", "    from scripts.research.oas_pretraining_data import load_datasets")
    old = """        manifest = json.loads((args.data_dir/'dataset_manifest.json').read_text())
        def datasets():
            train = PreparedImmuneDataset(args.data_dir, split='train')
            valid, selection = {}, {}
            for source in sorted({item['source'] for item in manifest['splits']['valid']['shards']}):
                data = PreparedImmuneDataset(args.data_dir, split='valid', source=source)
                indices = random.Random(42).sample(range(len(data)), min(len(data), args.valid_per_source))
                valid[source], selection[source] = Subset(data, indices), indices
            return train, valid, selection"""
    new = """        data_selection = json.loads(args.data_selection.read_text())
        def datasets():
            # Both train and validation are restricted to manifest source='oas'.
            return load_datasets(args.data_dir, args.valid_per_source, data_selection)"""
    replace(driver, old, new)
    replace(driver, "'data_dir': str(args.data_dir), 'parent_metadata': audits[0], 'train_records': len(train),",
            "'data_dir': str(args.data_dir), 'parent_metadata': audits[0], 'train_records': len(train),\n"
            "            'initialization': 'random_from_scratch', 'trainable_parameters': 75601176,\n"
            "            'data_selection': data_selection, 'data_selection_sha256': args.data_selection_sha256,")

    shutil.copy2(BASE / 'research_config.json', BUNDLE / 'research_config.json')
    shutil.copy2(BASE / 'research_config.json', CONFIG / 'research_config.json')
    binding = json.loads((BASE / 'data_binding.json').read_text())
    binding['model_config'] = {'path': str(BUNDLE / 'research_config.json'),
                               'sha256': sha(BUNDLE / 'research_config.json')}
    write(BUNDLE / 'data_binding.json', binding)
    selection = describe_selection(Path(binding['data_dir']))
    require(selection['splits']['train']['records'] == 2485433
            and selection['splits']['valid']['records'] == 12551, 'Unexpected OAS parent counts')
    write(BUNDLE / 'data_selection.json', selection)
    write(CONFIG / 'data_selection.json', selection)

    files = {str(path): sha(path) for path in sorted(source.rglob('*.py'))}
    files.update({path: expected for path, expected in original.items()
                  if not Path(path).is_relative_to(BASE / 'source')})
    write(BUNDLE / 'source_manifest.json', {'files': files})
    argv = json.loads((BASE / 'launch_training_args.json').read_text())
    changes = {'--source-root': source, '--source-manifest': BUNDLE / 'source_manifest.json',
               '--source-manifest-sha256': sha(BUNDLE / 'source_manifest.json'),
               '--data-binding': BUNDLE / 'data_binding.json',
               '--data-binding-sha256': sha(BUNDLE / 'data_binding.json'), '--output-dir': OUTPUT}
    for key, value in changes.items():
        argv[argv.index(key) + 1] = str(value)
    argv += ['--data-selection', str(BUNDLE / 'data_selection.json'),
             '--data-selection-sha256', sha(BUNDLE / 'data_selection.json')]
    write(BUNDLE / 'launch_training_args.json', argv)

    # Record evaluator dependencies separately: changes must be reviewed before
    # evaluation, but cannot invalidate a running training checkpoint's sources.
    evaluation_files = {str(path): sha(path) for folder in ('downstream', 'scripts/downstream')
                        for path in sorted((ROOT / folder).rglob('*.py'))}
    checker = ROOT / 'scripts/research/check_oas_training_contract.py'
    contract = {'schema': 'immune.oas_evaluation_contract.v1', 'training_source_root': str(source),
                'training_source_manifest_sha256': sha(BUNDLE / 'source_manifest.json'),
                'model_loader': str(ROOT / 'downstream/fixed_chain_eval.py'),
                'checkpoint_check': str(ROOT / 'scripts/downstream/check_fixed_chain_eval.py'),
                'task_runner_reference': str(ROOT / 'scripts/downstream/run_fixed_chain_suite.py'),
                'pairing_reference': str(ROOT / 'scripts/downstream/run_fixed_chain_pairing_reference.py'),
                'unit_tests': {str(source / 'scripts/tests/immune_llada' / name):
                               sha(source / 'scripts/tests/immune_llada' / name) for name in TESTS},
                'checker': str(checker), 'checker_sha256': sha(checker),
                'evaluation_files': evaluation_files,
                'scope': 'Python version record, not an evaluation plan or result. New checkpoint/run plan required; '
                         'do not reuse historical checkpoint50000 report labels. External benchmark assets are '
                         'bound separately by the future task plan.'}
    write(BUNDLE / 'evaluation_contract.json', contract)
    write(CONFIG / 'evaluation_contract.json', contract)
    launch = json.loads((BASE / 'direct_launch.json').read_text())
    launch.update(runtime_dir=str(OUTPUT.with_name(OUTPUT.name + '_supervision') / 'startup'),
                  arguments_sha256=sha(BUNDLE / 'launch_training_args.json'),
                  launcher_sha256=sha(source / 'scripts/research/launch_v5_direct.py'),
                  evaluation_contract_sha256=sha(BUNDLE / 'evaluation_contract.json'),
                  data_audit_scope='Original full V5 read-only integrity audit; training/validation select OAS only.')
    write(BUNDLE / 'direct_launch.json', launch)
    training = {'schema': 'immune.oas_scratch_training.v1', 'queue': 'queue012', 'nodes': 1,
                'gpus_per_node': 8, 'flavor': 'ml.pni2.28xlarge', 'parameters': 75601176,
                'initialization': 'random_from_scratch', 'all_parameters_trainable': True,
                'pretrained_checkpoint': None, 'resume_checkpoint': None, 'lora': None,
                'data_selection': str(BUNDLE / 'data_selection.json'), 'train_records': 2485433,
                'valid_records': 12551, 'validation_sample_records': 256, 'validation_seed': 42,
                'seed': 42, 'micro_batch': 64, 'grad_accum': 1, 'global_batch': 512,
                'max_steps': 300000, 'learning_rate': 0.0001, 'warmup_steps': 2000,
                'scheduler': 'cosine', 'optimizer': 'adamw_torch', 'precision': 'bf16_autocast_fp32_parameters',
                'sampling': 'epoch_random_natural', 'eval_every': 1000, 'save_every': 1000,
                'output_dir': str(OUTPUT), 'bundle': str(BUNDLE),
                'launch_training_args': str(BUNDLE / 'launch_training_args.json'),
                'evaluation_contract': str(BUNDLE / 'evaluation_contract.json'),
                'profile_status': 'pending_on_allocated_8_gpus_before_formal_training',
                'training_submitted': False}
    write(CONFIG / 'training_config.json', training)
    changed = [str(path.relative_to(source)) for path in source.rglob('*.py')
               if not (BASE / 'source' / path.relative_to(source)).exists()
               or sha(path) != sha(BASE / 'source' / path.relative_to(source))]
    write(BUNDLE / 'preparation.json', {'status': 'prepared', 'parent': str(BASE),
          'builder': str(Path(__file__).resolve()), 'builder_sha256': sha(__file__),
          'changed_source_files': sorted(changed), 'training_submitted': False})
    print(BUNDLE)


if __name__ == '__main__':
    main()
