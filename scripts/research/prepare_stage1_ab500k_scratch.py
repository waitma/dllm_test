"""Freeze a new randomly initialized AB500K comparison using existing targets.

Activate pllm; python -B THIS_FILE --base-plan ABS_PLAN --run-root ABS_NEW_RUN
Optional --max-steps defaults to 9770 (ten complete epochs). No job submission.
Use --pair-speed symmetric_no_recompute for the qualified speed candidate.
Only model configuration is reused; no old student weights or optimizer are read.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
import shutil

import torch
import yaml

from scripts.research.bert_mlm_data import sha, write_json, step_budget
from scripts.research.prepare_stage1_ab500k import shell_text
from scripts.research.stage1_ab500k_initialization import (
    build_scratch_model, shared_state_hash, canonical_hash, verify_scratch_initialization)


def execution_pair_config(pair, mode):
    """Bind execution choices without changing model weights or KD supervision."""
    if mode not in {"baseline", "symmetric", "symmetric_no_recompute"}:
        raise ValueError("Unknown Pair speed mode")
    result = deepcopy(pair)
    if mode != "baseline":
        result.update(bias_implementation="symmetric", autocast_pair_embeddings=True,
                      symmetric_pair_chunk=4096, checkpoint_pair_chunks=mode == "symmetric")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-plan", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--max-steps", type=int, default=9770)
    parser.add_argument("--pair-speed", choices=("baseline", "symmetric", "symmetric_no_recompute"),
                        default="baseline")
    args = parser.parse_args()
    if not args.base_plan.is_absolute() or not args.run_root.is_absolute():
        raise ValueError("Use absolute paths")
    if args.max_steps <= 500:
        raise ValueError("Budget must exceed the 500-step warmup")
    root = Path(__file__).resolve().parents[2]
    run, launch = args.run_root, args.run_root/"launch"
    if run.exists():
        raise FileExistsError("Refusing to replace a scratch experiment: " + str(run))
    plan = json.loads(args.base_plan.read_text())
    old_source = Path(plan["source_root"])
    for path, digest in plan["sources"].items():
        if sha(path) != digest:
            raise ValueError("Base frozen package changed: " + path)
    plan["pair"] = execution_pair_config(plan["pair"], args.pair_speed)
    torch.set_num_threads(2)
    config, seed = plan["model"], plan["training"]["seed"]
    a = build_scratch_model(config, "mlm", seed, plan["pair"])
    initial_hash = shared_state_hash(a)
    d = build_scratch_model(config, "pair_kd", seed, plan["pair"])
    if shared_state_hash(d) != initial_hash:
        raise ValueError("Shared random tensors differ between arms")
    for name, value in a.state_dict().items():
        if not torch.equal(value, d.state_dict()[name]):
            raise ValueError("Initial shared tensor differs: " + name)
    if any(torch.count_nonzero(x.weight) for x in d.pair_bias_outputs):
        raise ValueError("Initial pair feedback must be zero")
    audit = dict(status="passed", device="cpu", initialized_from="random_no_checkpoint_read",
        seed=seed, shared_state_sha256=initial_hash, shared_tensors=len(a.state_dict()),
        parameters={"mlm": sum(p.numel() for p in a.parameters()),
                    "pair_kd": sum(p.numel() for p in d.parameters())},
        inherited_parameters=0, all_feedback_outputs_zero=True,
        gpu_qualification="pending_user_submission")
    del a, d
    gc.collect()
    source = launch/"source"
    shutil.copytree(old_source, source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    modified = [
        "scripts/research/run_stage1_ab500k.py", "scripts/research/launch_stage1_ab500k.py",
        "scripts/research/report_stage1_ab500k.py", "scripts/research/stage1_ab500k_initialization.py",
        "scripts/research/prepare_stage1_ab500k_scratch.py",
        "scripts/tests/immune_llada/test_stage1_ab500k_scratch.py"]
    if args.pair_speed != "baseline":
        modified += ["dllm/pipelines/immune_llada/fixed_chain/structure_mlm.py",
                     "scripts/research/stage1_pair_speed_candidates.py",
                     "scripts/tests/immune_llada/test_stage1_pair_speed_candidates.py",
                     "scripts/tests/immune_llada/test_stage1_pair_speed_integrated.py"]
    for name in modified:
        shutil.copy2(root/name, source/name)
    for key in ("parent_checkpoint", "parent_step", "parent_model_sha256", "parent_receipt_sha256"):
        plan[key] = None
    plan.update(created_utc=datetime.now(timezone.utc).isoformat(), run_root=str(run),
        source_root=str(source), output_root=str(run/"train"), report_output=str(run/"comparison"),
        initialization=dict(schema="immune.stage1.random_init.v1", mode="scratch", seed=seed,
            config_sha256=canonical_hash(config), shared_state_sha256=initial_hash),
        cache_reused=True, submitted=False)
    if args.pair_speed != "baseline":
        plan["pair_execution"] = dict(mode=args.pair_speed,
            qualification="CPU integration checked; actual GPU profile and 8rank replay required at launch",
            reference="stage1_pair_speed_20261024", teacher_or_target_changes=False)
    plan["training"].update(max_steps=args.max_steps, backbone_lr=1e-4, pair_lr=1e-4)
    plan["sources"] = {str(p): sha(p) for p in sorted(source.rglob("*.py"))}
    verify_scratch_initialization(plan)
    write_json(launch/"plan.json", plan)
    audit.update(plan_sha256=sha(launch/"plan.json"),
                 budget=step_budget(500000, 512, args.max_steps))
    (run/"diagnostics").mkdir()
    write_json(run/"diagnostics/initialization.json", audit)
    template = yaml.safe_load((root/"train_jobs/stage1-teacher-pilot-cache-8gpu-20261022.yml").read_text())
    jobs = []
    for task in ("train_mlm", "train_pair_kd", "report"):
        script = launch/(task+".sh")
        script.write_text(shell_text(launch/"plan.json", source/"scripts/research/launch_stage1_ab500k.py", task, source))
        script.chmod(0o755)
        if task == "report":
            continue
        config = deepcopy(template)
        config.update(TaskName=run.name.replace('_', '-')+'-'+task.replace('_', '-'),
            Entrypoint='bash '+str(script), ResourceQueueName='ZENITH_MLP2_TASK', Preemptible=False,
            Description=f'AB500K scratch {task}; no inherited student weights; {args.max_steps} steps; global512; '
                f'backbone/pair LR1e-4; same random backbone seed42; pair speed={args.pair_speed}; GPU profile/replay pending.',
            TaskRoleSpecs=[dict(RoleName='worker', RoleReplicas=1, Flavor='ml.pni2.28xlarge')],
            Tags=['bioseq', 'stage1-ab500k', 'scratch', task, '8gpu', 'nonpreemptible'],
            ActiveDeadlineSeconds=604800)
        path = root/'train_jobs'/(config['TaskName']+'.yml')
        if path.exists():
            raise FileExistsError(path)
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        jobs.append(dict(task=task, shell=str(script), yaml=str(path), yaml_sha256=sha(path), submitted=False))
    write_json(launch/"jobs.json", jobs)
    (launch/"README.md").write_text(
        '# AB500K 从零初始化对照\n\n'
        '纯MLM与原有共享Pair＋KD架构；不加载旧学生checkpoint。共用随机主干seed42，'
        'Pair出口零初始化；教师缓存沿用，教师是已有冻结模型。\n\n'
        f'Pair执行模式：{args.pair_speed}；具体开关绑定plan.pair与checkpoint research_config。'
        '链内distogram KD目标保持不变，不新增跨链或原始Pair State对齐。\n\n'
        f'每组单节点8卡，global512，{args.max_steps}步；主干/Pair峰值LR均1e-4，'
        'warmup500、cosine降至峰值10%，KD权重0.05/warmup1000。每250步验证和保存，终点额外验证保存。\n\n'
        f'```bash\nbash {launch}/train_mlm.sh\nbash {launch}/train_pair_kd.sh\n```\n\n'
        '两个命令分别提交8卡任务。必须挂载 /project_c20250601/251105016；'
        '可在命令后追加 --verify-only 做CPU预检。作业内先执行GPU profile及8rank恢复短测。\n\n'
        f'新输出为 {run}/train。首次从随机权重step0开始；以后原命令只自动续接本实验latest，'
        '不能用旧实验保存点续接新初始化方案。旧实验保持不变。\n\n'
        f'结果汇总：bash {launch}/report.sh。云端YAML路径记录于 {launch}/jobs.json；本程序不提交任务。\n')
    print(json.dumps(dict(status="prepared", launch=str(launch), audit=audit, jobs=jobs)), flush=True)


if __name__ == "__main__":
    main()
