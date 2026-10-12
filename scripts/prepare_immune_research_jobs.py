"""Generate 36 queue012 YAMLs, with no submission or GPU work.

After activating pllm: python scripts/prepare_immune_research_jobs.py
After final data cleaning: rerun with --data-dir /absolute/certified/data.
For selected 5000-step continuations: --resume-map /absolute/variant_checkpoint_map.json.
"""
import argparse
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from dllm.pipelines.immune_llada.research.model import VARIANTS
from scripts.prepare_immune_v3_exploration import JobDumper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data/prepared/immune_v7_mhc_sequence_v1')
    parser.add_argument('--destination', type=Path, default=ROOT/'train_jobs/immune_research_queue012')
    parser.add_argument('--profile-root', type=Path, default=ROOT/'output/immune_research_queue012/profiles')
    parser.add_argument('--resume-map', type=Path,
                        help='JSON object mapping selected variant names to checkpoint directories for the 5000-step stage')
    args = parser.parse_args()
    resume_map = json.loads(args.resume_map.read_text()) if args.resume_map else {}
    if not isinstance(resume_map, dict) or set(resume_map)-set(VARIANTS):
        raise ValueError('Resume map must contain registered variant names')
    for variant, checkpoint in resume_map.items():
        if not isinstance(checkpoint, str) or not Path(checkpoint).is_absolute():
            raise ValueError(f'Resume path must be absolute: {variant}')
    jobs = []
    for phase, mode, steps, deadline in [('01_profile_synthetic', 'profile-synthetic', 20, 7200),
                                          ('02_clean_data_pilots', 'train', 1000, 86400),
                                          ('03_extended_5000', 'train', 5000, 259200)]:
        destination = args.destination/phase
        destination.mkdir(parents=True, exist_ok=True)
        for variant, options in VARIANTS.items():
            config = yaml.safe_load((ROOT/'train_jobs/immune_v3_profile_20261001_queue012.yml').read_text())
            prefix = config['Entrypoint'].split('python -u scripts/profile_immune_v3.py')[0]
            output_root = ROOT/'output/immune_research_queue012'/phase/variant
            profile_dir = args.profile_root.resolve()/variant
            prefix += f'RUN_DIR={shlex.quote(str(output_root))}/$(date -u +%Y%m%dT%H%M%S%N)\n'
            if mode == 'train':
                gate = ('from pathlib import Path; from examples.llada.immune_pretrain_v3 import assert_ready; '
                        f'assert_ready(Path({str(args.data_dir.resolve())!r}))')
                prefix += shlex.join(['python', '-c', gate])+'\n'
            command = ['python', '-u', 'scripts/run_immune_research.py', '--variant', variant, '--mode', mode,
                       '--profile-dir', str(profile_dir), '--steps', str(steps)]
            if mode == 'train':
                command += ['--data-dir', str(args.data_dir.resolve())]
            resume = resume_map.get(variant) if phase == '03_extended_5000' else None
            if resume:
                command += ['--resume-from-checkpoint', resume]
            config['Entrypoint'] = prefix+shlex.join(command)+' --output-dir "$RUN_DIR"\n'
            config['TaskName'] = f'immune-research-{variant.replace("_", "-")}-{mode}-{steps}'
            config['Description'] = (f'{variant}: {mode}, cumulative {steps} optimizer steps; '+
                ('resume same scratch run' if resume else 'from scratch')+'; no ESMC; '
                'global batch 256 = measured bs x (256/bs) ga x 1 GPU; mem_peak pending fresh per-method '
                '2048-token profile; 1x ml.pni2.3xlarge A100-80GB; '+
                ('synthetic only; not a downstream result' if mode != 'train' else 'DRAFT: requires clean-data certificates and method profile'))
            config['Tags'] = ['immune', 'noesmc', 'research', 'queue012', variant.split('_')[0], mode]
            config['ActiveDeadlineSeconds'] = deadline
            path = destination/f'{variant}.yml'
            path.write_text(yaml.dump(config, Dumper=JobDumper, sort_keys=False, allow_unicode=True))
            jobs.append(dict(variant=variant, options=options, mode=mode, yaml=str(path.resolve()),
                             output_root=str(output_root), profile_dir=str(profile_dir), steps=steps,
                             resume_from_checkpoint=resume, source_exposures=steps*256 if mode == 'train' else None,
                             status='prepared_not_submitted' if mode != 'train' else 'gated_draft',
                             deadline_seconds=deadline, global_batch=256))
    (args.destination/'manifest.json').write_text(json.dumps(dict(submitted=False, queue='queue012',
        training_horizon=10000, warmup_steps=100, evaluation_every=100,
        data_dir=str(args.data_dir.resolve()), jobs=jobs), indent=2)+'\n')
    print(f'Prepared {len(jobs)} YAMLs in {args.destination}; no jobs submitted')


if __name__ == '__main__':
    main()
