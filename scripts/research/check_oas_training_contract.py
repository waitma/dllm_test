"""Check the OAS launch and the recorded evaluation Python versions, without GPUs.

Activate pllm; python -B THIS_FILE --bundle ABS [--checkpoint ABS].
Run this before evaluation, then use check_fixed_chain_eval.py --run ABS.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(bundle, checkpoint=None):
    bundle = bundle.resolve()
    launch = json.loads((bundle / 'direct_launch.json').read_text())
    require(sha(bundle / 'launch_training_args.json') == launch['arguments_sha256'],
            'Launch arguments changed')
    contract_path = bundle / 'evaluation_contract.json'
    require(sha(contract_path) == launch['evaluation_contract_sha256'], 'Evaluation contract changed')
    contract = json.loads(contract_path.read_text())
    require(sha(__file__) == contract['checker_sha256'], 'Contract checker changed')
    argv = json.loads((bundle / 'launch_training_args.json').read_text())
    driver = runpy.run_path(str(bundle / 'source/scripts/research/run_v7_formal.py'))
    args = driver['parse_args'](argv)
    driver['verify_sources'](args)
    selection = json.loads(args.data_selection.read_text())
    require(sha(args.data_dir / 'dataset_manifest.json') == selection['manifest_sha256'],
            'Parent data manifest changed')
    require(selection['train_sources'] == selection['validation_sources'] == ['oas'],
            'Only OAS is permitted')
    for filename, expected in contract['evaluation_files'].items():
        require(sha(filename) == expected, 'Recorded evaluation Python changed: ' + filename)
    if checkpoint is not None:
        checkpoint = checkpoint.resolve()
        side = json.loads((checkpoint / 'formal_checkpoint.json').read_text())
        require(side['rank_acknowledgments'] == list(range(8)), 'Incomplete eight-rank checkpoint')
        for name in ('run_provenance.json', 'research_config.json', 'immune_config.json',
                     'vocabulary.json', 'trainer_state.json', 'model.safetensors'):
            require(sha(checkpoint / name) == side['files'][name], 'Checkpoint changed: ' + name)
        binding = json.loads((checkpoint / 'run_provenance.json').read_text())
        digest = hashlib.sha256(json.dumps(binding, sort_keys=True, allow_nan=False).encode()).hexdigest()
        require(digest == side['binding_sha256'], 'Checkpoint provenance digest differs')
        require(binding['source_root'] == str(args.source_root)
                and binding['source_manifest_sha256'] == args.source_manifest_sha256
                and binding['data_selection'] == selection
                and binding['data_selection_sha256'] == args.data_selection_sha256
                and binding['variant'] == 'fixed_chain_full_global',
                'Checkpoint was not trained with this OAS bundle')
    return {'status': 'passed', 'bundle': str(bundle),
            'train_records': selection['splits']['train']['records'],
            'valid_records': selection['splits']['valid']['records'],
            'evaluation_python_files': len(contract['evaluation_files']),
            'checkpoint': str(checkpoint) if checkpoint else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.bundle, args.checkpoint), indent=2))


if __name__ == '__main__':
    main()
