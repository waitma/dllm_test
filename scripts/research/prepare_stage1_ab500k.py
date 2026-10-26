"""Prepare independent, immutable AB500K shell/YAML jobs without submitting them.

Activate pllm; python -B -m scripts.research.prepare_stage1_ab500k --run-root ABS.
Build the fixed subset first with scripts.research.stage1_ab500k_data.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil

import yaml

from dllm.pipelines.immune_llada.fixed_chain.structure_mlm import PairConfig
from scripts.research.bert_mlm_data import sha, write_json


ROOT = Path(__file__).resolve().parents[2]
STORAGE = Path('/project_c20250601/251105016')
DEFAULT_RUN = ROOT / 'output/stage1_ab500k_pilot_20261022'
TASKS = tuple(f'teacher_cache_{i}' for i in range(4)) + ('check_cache', 'train_mlm', 'train_pair_kd', 'report')


def shell_text(plan_path, launcher, task, source):
    """Render an ordinary shell entrypoint with plan and launcher integrity checks."""
    checks = '\n'.join(f'{sha(p)}  {p}' for p in (plan_path, launcher))
    gpu = 4 if task.startswith('teacher_cache_') else 8 if task.startswith('train_') else 0
    return f'''#!/usr/bin/env bash
# AB500K {task}: one node, {gpu} GPU; user submits this job manually.
# Resume: rerun the identical teacher entry, or train with --resume latest.
# CPU admission: append --verify-only. No script submits or cancels cloud jobs.
set -euo pipefail
source {shlex.quote(str(STORAGE / 'miniforge3/etc/profile.d/conda.sh'))}
conda activate {shlex.quote(str(STORAGE / 'conda/envs/pllm'))}
cd {shlex.quote(str(source))}
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export IMMUNE_SDPA_BACKEND=efficient
ulimit -s 8192
sha256sum --status -c <<'AB500K_SHA'
{checks}
AB500K_SHA
exec {shlex.quote(str(STORAGE / 'conda/envs/pllm/bin/python'))} -I -B {shlex.quote(str(launcher))} --plan {shlex.quote(str(plan_path))} --task {task} "$@"
'''


def prepare(args):
    from scripts.research.stage1_ab500k_data import validate_subset
    run = Path(args.run_root).absolute()
    subset_path = run / 'data/subset.json'
    validation_path = run / 'data/validation.json'
    if validation_path.exists():
        validation = json.loads(validation_path.read_text())
        if validation.get('status') != 'passed' or validation.get('manifest_sha256') != sha(subset_path):
            raise ValueError('Existing subset deep-validation receipt is incompatible')
    else:
        validation = validate_subset(subset_path, workers=8)
        write_json(validation_path, validation)
    if getattr(args, 'validate_subset_only', False):
        print(json.dumps(validation), flush=True)
        return
    subset = json.loads(subset_path.read_text())
    if subset['counts'] != {'train': 500000, 'valid': 4096}:
        raise ValueError('Production launch requires exactly train500000/valid4096')
    parent = Path(args.parent).absolute()
    receipt = json.loads((parent / 'checkpoint_receipt.json').read_text())
    if receipt['step'] != 287000 or receipt['status'] != 'complete':
        raise ValueError('Expected complete BERT287000 parent')
    model = json.loads((parent / 'immune_config.json').read_text())
    contract = json.loads(Path(args.teacher_manifest).read_text())['teacher_contract']
    if sha(ROOT / 'scripts/research/stage1_teacher.py') != contract['fast_compat_source_sha256']:
        raise ValueError('Existing teacher implementation differs from its frozen contract')
    launch = run / 'launch'
    launch.mkdir(parents=True, exist_ok=False)
    source = launch / 'source'
    for directory in ('dllm', 'scripts', 'examples'):
        for origin in sorted((ROOT / directory).rglob('*.py')):
            destination = source / origin.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, destination)
    training = dict(max_steps=3000, global_batch=512, seed=42, backbone_lr=1e-5, pair_lr=1e-4,
        weight_decay=.01, warmup_steps=500, kd_warmup_steps=1000, intra_weight=.05,
        inter_weight=0., temperature=1., clip_norm=1., precision='bf16', eval_every=250,
        save_every=250, log_every=25, loader_workers=4)
    plan = dict(schema='immune.stage1.ab500k_pilot.v1', created_utc=datetime.now(timezone.utc).isoformat(),
        run_root=str(run), source_root=str(source), sources={str(p): sha(p) for p in sorted(source.rglob('*.py'))},
        subset_manifest=str(subset_path), subset_manifest_sha256=sha(subset_path),
        subset_validation=str(validation_path), subset_validation_sha256=sha(validation_path),
        data_index=subset['data_index'], data_index_sha256=subset['data_index_sha256'],
        parent_checkpoint=str(parent), parent_step=287000,
        parent_model_sha256=sha(parent / 'model.safetensors'),
        parent_receipt_sha256=sha(parent / 'checkpoint_receipt.json'), model=model,
        pair=asdict(PairConfig()), world_size=8, training=training,
        output_root=str(run / 'train'), cache_root=str(run / 'teacher_cache'),
        cache_manifest=str(run / 'cache_index/manifest.json'), teacher_contract=contract,
        model_dir=str(Path(args.model_dir).absolute()),
        teacher_python=str(STORAGE / 'conda/envs/stage1_esmfold2/bin/python'),
        student_python=str(STORAGE / 'conda/envs/pllm/bin/python'),
        storage_budget_bytes=150_000_000_000, worker_budget_bytes=8_750_000_000,
        auxiliary_budget_bytes=10_000_000_000, pilot_per_worker=128, replay_per_worker=2,
        teacher_quality_policy='exploratory_unvalidated', cache_coverage_policy='require_all_selected_records',
        report_output=str(run / 'comparison'), submitted=False)
    plan_path = launch / 'plan.json'
    write_json(plan_path, plan)
    launcher = source / 'scripts/research/launch_stage1_ab500k.py'
    template = yaml.safe_load((ROOT / 'train_jobs/stage1-teacher-pilot-cache-8gpu-20261022.yml').read_text())
    jobs, entries = [], {}
    for task in TASKS:
        script = launch / (task + '.sh')
        script.write_text(shell_text(plan_path, launcher, task, source))
        script.chmod(0o755)
        entries[task] = str(script)
        if task.startswith(('teacher_cache_', 'train_')):
            teacher = task.startswith('teacher_cache_')
            config = dict(template)
            config.update(TaskName='stage1-ab500k-' + task.replace('_', '-') + '-20261022',
                Entrypoint='bash ' + str(script), ResourceQueueName='ZENITH_MLP2_TASK', Preemptible=False,
                Description=('AB500K offline ESMFold2-Fast; 4 independent GPU workers, batch1; '
                    'mutually exclusive global shards; 35GB worker budget/job, 150GB overall; mem_peak pending pilot.'
                    if teacher else 'AB500K ' + task + '; BERT287000, 3000 steps, global batch512; '
                    'per-arm max safe micro profile + remainder accumulation x8 GPU; mem_peak pending profile; '
                    'fixed record masks and full-state resume; exploratory structural quality unvalidated.'),
                TaskRoleSpecs=[dict(RoleName='worker', RoleReplicas=1,
                    Flavor='ml.pni2.14xlarge' if teacher else 'ml.pni2.28xlarge')],
                Tags=['bioseq', 'stage1-ab500k', task, '4gpu' if teacher else '8gpu', 'nonpreemptible'],
                ActiveDeadlineSeconds=172800 if teacher else 604800)
            target = ROOT / 'train_jobs' / (config['TaskName'] + '.yml')
            if target.exists():
                raise FileExistsError(target)
            target.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
            jobs.append(dict(task=task, shell=str(script), yaml=str(target), yaml_sha256=sha(target),
                gpus=4 if teacher else 8, submitted=False,
                command='volc ml_task submit --conf ' + str(target), webpage_entrypoint=config['Entrypoint']))
    write_json(launch / 'submission.json', dict(submitted=False, jobs=jobs, entries=entries,
        plan=str(plan_path), plan_sha256=sha(plan_path),
        order=['teacher_cache_0..3 independently', 'check_cache', 'train_mlm', 'train_pair_kd', 'report']))
    readme = ['# AB500K 用户提交入口', '', '脚本与YAML由用户提交；准备工具不会自动提交GPU任务。', '',
        '网页创建任务时，必须先添加下列项目盘挂载；仅填写启动命令不会配置共享存储。四个教师和两个训练任务都需要相同挂载。', '',
        '```yaml', 'Type: Vepfs', 'VepfsId: vepfs-cnbj2c98dea54433',
        'SubPath: c20250601/251105016', 'MountPath: /project_c20250601/251105016', '```', '',
        '挂载需可读写。只有 /personal_fast_storage 和 /public_fast_storage 时，启动命令中的脚本不可见。下面六份任务YAML已包含此配置，CLI提交时使用对应YAML。', '',
        '先分别提交四个教师任务（各单节点4卡，可并行）；全部完成后运行缓存检查，再提交两组训练（各单节点8卡）。', '',
        'GPU运行前可在对应脚本后加 `--verify-only` 做CPU检查；训练在缓存未完成时明确返回not_ready。', '',
        '教师重跑同一入口可恢复已发布分片；训练重跑自动恢复latest，也可显式追加 `--resume latest`。', '',
        '需要停止训练时，在对应训练输出目录创建 `STOP`，本步完成后验证保存退出；恢复前移除该文件。', '',
        '| 任务 | GPU | 网页启动命令 |', '|---|---:|---|']
    readme += [f"| {j['task']} | {j['gpus']} | `bash {j['shell']}` |" for j in jobs]
    readme += ['', '缓存汇总与检查：', '', '```bash', 'bash ' + entries['check_cache'], '```', '',
        '两组完成后生成对照：', '', '```bash', 'bash ' + entries['report'], '```', '',
        '六份YAML与CLI命令见 ' + str(launch / 'submission.json') + '。', '',
        '教师pilot及学生8卡profile/恢复检查在用户提交后执行；此包不代表这些GPU检查已完成。', '']
    (launch / 'README.md').write_text('\n'.join(readme))
    print(json.dumps(dict(status='prepared_not_submitted', launch=str(launch), jobs=len(jobs), shells=len(entries))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', default=str(DEFAULT_RUN))
    parser.add_argument('--validate-subset-only', action='store_true', help='Deep CPU validation; write a bound receipt without freezing jobs')
    parser.add_argument('--parent', default=str(ROOT / 'output/bert_diffusion_edit_v7_from287000_20261019/checkpoint'))
    parser.add_argument('--teacher-manifest', default=str(ROOT / 'output/stage1_training_20261022/cache_index_teacher_pilot/manifest.json'))
    parser.add_argument('--model-dir', default=str(STORAGE / 'model_weights/esmfold2_fast/45fe8656f5b3ef493c17fcf9abe9a2968902e712'))
    prepare(parser.parse_args())
