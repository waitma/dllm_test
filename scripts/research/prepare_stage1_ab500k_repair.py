"""Freeze corrected AB500K student entrypoints without altering old runs.

Activate pllm; python -B THIS_FILE --run-root ABS_RUN.
Creates launch_training_fix_20261023 and new train/report directories in the plan.
Old step-zero checkpoints retain their original code/plan identity; no submit.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from datetime import datetime, timezone
import json
import shutil

from scripts.research.bert_mlm_data import sha, write_json
from scripts.research.prepare_stage1_ab500k import shell_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    run = args.run_root.resolve()
    old = run/"launch_mlm_independent"
    new = run/"launch_training_fix_20261023"
    plan = json.loads((old/"plan.json").read_text())
    for path, digest in plan["sources"].items():
        if sha(path) != digest:
            raise ValueError("Old frozen package changed: " + path)
    for arm in ("mlm", "pair_kd"):
        latest = Path(plan["output_root"])/arm/"latest.json"
        if latest.exists() and json.loads(latest.read_text())["step"] != 0:
            raise ValueError("A later old checkpoint exists; review migration before creating this fresh-start repair")
    if new.exists():
        raise ValueError("Refusing to replace an existing frozen repair package")
    source = new/"source"
    shutil.copytree(old/"source", source)
    modified = ("scripts/research/run_stage1_ab500k.py",
                "scripts/research/launch_stage1_ab500k.py",
                "scripts/research/report_stage1_ab500k.py",
                "scripts/research/check_stage1_ab500k_loader.py",
                "scripts/tests/immune_llada/test_stage1_ab500k_training.py")
    for relative in modified:
        shutil.copy2(root/relative, source/relative)
    plan.update(created_utc=datetime.now(timezone.utc).isoformat(), source_root=str(source),
                output_root=str(run/"train_fix_20261023"), report_output=str(run/"report_fix_20261023"))
    plan["sources"] = {str(source/Path(path).relative_to(old/"source")):
                       sha(source/Path(path).relative_to(old/"source")) for path in plan["sources"]}
    for relative in modified:
        plan["sources"][str(source/relative)] = sha(source/relative)
    write_json(new/"plan.json", plan)
    for task in ("train_mlm", "train_pair_kd", "report"):
        script = new/(task+".sh")
        script.write_text(shell_text(new/"plan.json", source/"scripts/research/launch_stage1_ab500k.py", task, source))
        script.chmod(0o755)
    write_json(new/"repair.json", dict(schema="immune.stage1.ab500k.repair.v1",
        old_plan=str(old/"plan.json"), old_plan_sha256=sha(old/"plan.json"),
        new_plan_sha256=sha(new/"plan.json"), modified_sources=list(modified),
        initialization="fresh BERT287000 with the same seed; old checkpoints only at step zero",
        fixes=["duplicate metric keyword names", "post-NCCL fork loader", "silent rank-zero logging errors"],
        frozen_originals_unchanged=True, submitted=False))
    (new/"README.md").write_text(
        "# AB500K训练修复入口\n\n"
        "修复首步metrics重复关键字、NCCL后的fork DataLoader；使用spawn、120秒读批超时、分阶段日志和跨rank日志异常传播。\n\n"
        "两组必须都用本目录的入口，单节点8卡，挂载/project_c20250601/251105016。先确认旧任务退出，避免重复占卡。\n\n"
        f"```bash\nbash {new}/train_mlm.sh\nbash {new}/train_pair_kd.sh\n```\n\n"
        "两个命令分别提交，保留原BERT287000起点、seed42、同一子集、缓存及3000步/global512。"
        f"输出为{plan['output_root']}；旧训练与checkpoint0不改。本修复包首次从相同初始化重新开始，"
        "不直接恢复旧代码身份的checkpoint；以后本包仍自动resume latest。GPU profile/8rank恢复短测将在新作业内重做。\n")
    print(json.dumps(dict(status="prepared", launch=str(new), submitted=False)), flush=True)


if __name__ == "__main__":
    main()
