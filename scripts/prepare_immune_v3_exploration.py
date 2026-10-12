"""Write queue012 exploration YAMLs for manual submission; never submit jobs.

After activating pllm, run: python scripts/prepare_immune_v3_exploration.py
Use --data-dir /absolute/final-clean-data when the final cleaned dataset exists.
Real-data drafts always retain the production data and memory gates at runtime.
"""
import argparse
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from scripts.profile_immune_v3 import verify_memory_profile


class JobDumper(yaml.SafeDumper):
    """Keep shell entrypoints readable as literal blocks."""


def represent_string(dumper, value):
    return dumper.represent_scalar('tag:yaml.org,2002:str', value, style='|' if '\n' in value else None)


JobDumper.add_representer(str, represent_string)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-dir', type=Path, default=ROOT/'output/immune_v3_profile_20261001_queue012')
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data/prepared/immune_v7_clean_r2')
    parser.add_argument('--destination', type=Path, default=ROOT/'train_jobs/immune_v3_explore_queue012')
    args = parser.parse_args()
    summary = json.loads((args.profile_dir/'summary.json').read_text())
    entries = []
    template = ROOT/'train_jobs/immune_v3_profile_20261001_queue012.yml'
    for phase in ('01_synthetic_ready', '02_real_data_gated'):
        directory = args.destination/phase
        directory.mkdir(parents=True, exist_ok=True)
        for ablation in ('A0', 'A1', 'A2', 'A3', 'A4'):
            batch = summary[ablation]['recommended_micro_batch']
            if batch < 1 or 256 % batch:
                raise ValueError('Invalid global batch decomposition')
            measured = verify_memory_profile(args.profile_dir, ablation, batch)
            config = yaml.safe_load(template.read_text())
            synthetic = phase == '01_synthetic_ready'
            kind = 'synthetic20' if synthetic else 'gated-pilot10'
            name = f'immune-v3-{ablation.lower()}-{kind}-queue012'
            output_root = ROOT/'output/immune_v3_explore_queue012'/kind/ablation
            prefix = config['Entrypoint'].split('python -u scripts/profile_immune_v3.py')[0]
            # Unique output per submission; no overwriting another run/checkpoint.
            prefix += f'RUN_DIR={shlex.quote(str(output_root))}/$(date -u +%Y%m%dT%H%M%S%N)\n'
            if synthetic:
                command = ['python', '-u', 'scripts/explore_immune_v3_synthetic.py', '--steps', '20']
            else:
                # Fail before allocating a model when data are not yet certified.
                preflight = ('from pathlib import Path; '
                             'from examples.llada.immune_pretrain_v3 import assert_ready; '
                             f'assert_ready(Path({str(args.data_dir.resolve())!r}))')
                prefix += shlex.join(['python', '-c', preflight])+'\n'
                command = ['python', '-u', 'examples/llada/immune_pretrain_v3.py',
                           '--data-dir', str(args.data_dir.resolve()), '--max-steps', '10000',
                           '--pilot-stop-step', '10', '--seed', '42']
            command += ['--ablation', ablation, '--batch-size', str(batch), '--grad-accum', str(256//batch),
                        '--profile-dir', str(args.profile_dir.resolve())]
            config['Entrypoint'] = prefix+shlex.join(command)+' --output-dir "$RUN_DIR"\n'
            config['TaskName'] = name
            config['Description'] = (
                f'{ablation} {kind}; '+('synthetic only' if synthetic else 'DRAFT: requires passed data certificates')+
                f'; global batch 256 = bs {batch} x ga {256//batch} x 1 GPU; '
                f'measured 2048-token mem_peak {measured["max_allocated_gib"]:.2f}/{measured["total_gib"]:.2f} GiB; '
                '1x ml.pni2.3xlarge; no biological quality claim')
            config['Tags'] = ['immune', 'noesmc', 'v3', 'queue012', ablation.lower(), kind]
            path = directory/f'{ablation.lower()}.yml'
            path.write_text(yaml.dump(config, Dumper=JobDumper, sort_keys=False, allow_unicode=True))
            entries.append(dict(ablation=ablation, phase=phase, yaml=str(path.resolve()),
                                output_root=str(output_root), micro_batch=batch, grad_accum=256//batch,
                                global_batch=256, measured_peak_gib=measured['max_allocated_gib'],
                                optimizer_steps=20 if synthetic else 10,
                                status='prepared_not_submitted' if synthetic else 'draft_requires_data_preflight'))
    manifest = dict(queue='queue012', submitted=False, profile_dir=str(args.profile_dir.resolve()),
                    data_dir=str(args.data_dir.resolve()), data_readiness='not asserted by generation', jobs=entries)
    (args.destination/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
