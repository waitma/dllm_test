"""Freeze a no-target-storage AB500K run; activate pllm, then use teacher Python.

python -B THIS_FILE --run-root ABS_NEW_RUN
Writes source/config/launch scripts only. No teacher inference or submission.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from datetime import datetime, timezone
import gc
import json
import shlex
import shutil

import numpy as np
import torch
import yaml

from scripts.research.bert_mlm_data import sha, write_json, step_budget
from scripts.research.stage1_ab500k_initialization import build_scratch_model, shared_state_hash, canonical_hash


ROOT = Path('/project_c20250601/251105016/project/dllm_test')
PYTHON = '/project_c20250601/251105016/conda/envs/general_teacher_protenix_20261024/bin/python'


def freeze(run):
    if not run.is_absolute() or run.exists():
        raise ValueError("Use a new absolute output directory")
    base = json.loads((ROOT/'output/stage1_ab500k_scratch_fast_20261024/launch/plan.json').read_text())
    subset = json.loads(Path(base['subset_manifest']).read_text())
    rows = np.load(subset['files']['train_rows']['path'])
    lookup = {int(row): i for i, row in enumerate(rows)}
    maxima, longest = [-1, -1, -1], [None, None, None]
    # Upper proxy for atom-side memory, independent of total sequence length.
    heavy_atoms = dict(zip('ACDEFGHIKLMNPQRSTVWY', (5,6,8,9,11,4,10,8,9,8,8,8,7,9,11,6,7,7,14,12)))
    # Reuse existing sequence requests only to identify worst shapes; these
    # files contain input sequences/mappings, not any teacher features.
    with Path(subset['files']['requests']['path']).open() as stream:
        for line in stream:
            item = json.loads(line)
            if item['split'] != 'train':
                continue
            lengths = [len(c['sequence']) for c in item['request']['chains']]
            atom_count = sum(heavy_atoms[a] for c in item['request']['chains'] for a in c['sequence'])
            for i, length in enumerate((sum(lengths), max(lengths), atom_count)):
                if length > maxima[i]:
                    maxima[i], longest[i] = length, lookup[item['row_index']]
    del lookup
    training = {**base['training'], 'inter_weight': .01, 'confidence_weighting': True}
    pair = {**base['pair'], 'independent_layers': True, 'checkpoint_pair_chunks': True,
            'distogram_chunk_size': 4096}
    torch.set_num_threads(2)
    initial, parameters = None, {}
    for arm in ('mlm', 'pair_kd'):
        model = build_scratch_model(base['model'], arm, training['seed'], pair)
        digest = shared_state_hash(model)
        if initial is not None and digest != initial:
            raise ValueError('Shared random backbone differs')
        initial = digest
        parameters[arm] = sum(p.numel() for p in model.parameters())
        del model
        gc.collect()
    source, launch = run/'launch/source', run/'launch'
    paths = [ROOT/'scripts/__init__.py']
    paths += list((ROOT/'dllm').rglob('*.py')) + list((ROOT/'scripts/research').glob('*.py'))
    paths += [ROOT/'scripts/tests/immune_llada'/name for name in
              ('test_protenix_online.py', 'test_structure_mlm.py', 'test_confidence_distillation.py')]
    for path in sorted(paths):
        dest = source/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    assets = json.loads((ROOT/'output/general_teacher_validation_20261024/assets.json').read_text())
    teacher_assets = {x['path']: x['sha256'] for x in assets['weights'] if not x['name'].startswith('boltz')}
    native = Path(assets['sources']['protenix']['path'])
    teacher_assets.update({str(p): sha(p) for p in sorted(native.rglob('*.py'))})
    for name in ('components.cif', 'components.cif.rdkit_mol.pkl'):
        path = Path('/project_c20250601/251105016/project/Protenix-v2/data/common')/name
        teacher_assets[str(path)] = sha(path)
    teacher_contract = dict(model='protenix_mini_esm_v0.5.0',
        revision='85767b811c40ed46e73a9b39519cf6bfca8701ba', cycles=4,
        msa=False, templates=False, distogram_bins=64, esm_cache=False,
        coordinates_for_parity=dict(samples=1, steps=5),
        delivery='memory_only_no_target_cache', confidence_threshold_A=2.,
        confidence_symmetry='mean_of_both_directions', confidence_calibrated=False,
        pair_coverage='all_unordered_nonself_intra_and_inter', student_input='sequence_only')
    plan = dict(schema='immune.stage1.protenix_online.v1', created_utc=datetime.now(timezone.utc).isoformat(),
        source_root=str(source), run_root=str(run), output_root=str(run/'train'),
        sources={str(p): sha(p) for p in sorted(source.rglob('*.py'))},
        **{key: base[key] for key in ('subset_manifest', 'subset_manifest_sha256',
             'subset_validation', 'subset_validation_sha256', 'data_index', 'data_index_sha256')},
        model=base['model'], pair=pair, training=training, world_size=8,
        student_python=PYTHON, teacher_assets=teacher_assets, teacher_contract=teacher_contract,
        teacher_target_storage='none', cache_root=None, cache_manifest=None,
        parent_checkpoint=None, parent_step=None, parent_model_sha256=None, parent_receipt_sha256=None,
        student_profile_length=subset['student_max_length'], profile_train_ordinals=sorted(set(longest)),
        teacher_profile_maxima=dict(total_residues=maxima[0], single_chain_residues=maxima[1], heavy_atoms=maxima[2]),
        initialization=dict(schema='immune.stage1.random_init.v1', mode='scratch', seed=training['seed'],
            config_sha256=canonical_hash(base['model']), shared_state_sha256=initial),
        submitted=False)
    write_json(launch/'plan.json', plan)
    write_json(run/'initialization.json', dict(status='passed', parameters=parameters,
        shared_state_sha256=initial, inherited_parameters=0, teacher_target_storage='none',
        budget=step_budget(500000, 512, training['max_steps']), gpu_qualification='pending'))
    template = yaml.safe_load((ROOT/'train_jobs/stage1-ab500k-scratch-fast-20261024-train-pair-kd.yml').read_text())
    jobs = []
    q = shlex.quote
    for arm in ('mlm', 'pair_kd'):
        for qualify in (False, True):
            task = ('qualify_' if qualify else 'train_') + arm
            script = launch/(task+'.sh')
            command = (f'exec {q(PYTHON)} -B -m torch.distributed.run --standalone --nproc_per_node=8 '
                       f'{q(str(source/"scripts/research/run_stage1_protenix_online.py"))} '
                       f'--plan {q(str(launch/"plan.json"))} --arm {arm} '
                       + ('--qualify-only' if qualify else '--resume latest'))
            script.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
                'source /project_c20250601/251105016/miniforge3/etc/profile.d/conda.sh\n'
                'conda activate /project_c20250601/251105016/conda/envs/pllm\n'
                'export USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n'
                'export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 LAYERNORM_TYPE=torch\n'
                'export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false\n'
                f'export PYTHONPATH={q(str(source))}\ncd {q(str(source))}\n'
                + command + '\n')
            script.chmod(0o755)
            config = {**template, 'TaskName': run.name.replace('_', '-')+'-'+task.replace('_','-'),
                'Description': f'AB500K memory-only online {task}; global512; bs auto<=64 x accumulation x8; '
                    'joint teacher/student mem_peak pending profile; independent per-layer Pair; '
                    '9770 steps, random student; no teacher output storage.',
                'Entrypoint': 'bash '+str(script), 'ActiveDeadlineSeconds': 1209600,
                'Tags': ['bioseq', 'stage1-protenix-online', arm, '8gpu', 'nonpreemptible']}
            job = launch/(task+'.yml')
            job.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
            jobs.append(dict(task=task, shell=str(script), yaml=str(job), submitted=False))
    write_json(launch/'jobs.json', jobs)
    (launch/'README.md').write_text('# AB500K 在线蒸馏\n\n'
        '教师 feature、distogram、confidence、ESM embedding 均不落盘。只读取既有输入和权重，'
        '写学生 checkpoint、日志、验证统计与小型配置/代码快照。\n\n'
        'train_mlm.sh 与 train_pair_kd.sh 是两个训练入口。qualify_*.sh 只做联合显存与恢复检查；'
        '训练入口也会先执行这些检查。旧离线训练命令不能用于此实验。所有脚本固定8卡，自动续学生断点。\n\n'
        'YAML在同目录；正式任务通过 volc ml_task submit --conf ABS_YAML 提交。'
        '当前尚未通过8卡资格/提交；同一 arm 只能运行一个任务。\n')
    print(json.dumps(dict(run_root=str(run), parameters=parameters, teacher_target_storage='none',
        profile_ordinals=plan['profile_train_ordinals'], submitted=False), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    freeze(parser.parse_args().run_root)
