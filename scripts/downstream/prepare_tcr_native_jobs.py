"""Prepare independent non-preemptible TCR jobs; never submit automatically.

Run in pllm: python scripts/downstream/prepare_tcr_native_jobs.py --stage binding
Each job requires a successful matching GPU gate. Submit the printed YAMLs and
record every task ID in PROJECT_PROCESS.md. Existing configurations are protected.
"""
from __future__ import annotations

import argparse
import copy
import json
import shlex
from pathlib import Path

import yaml

from scripts.downstream.tcr_eval_pin import PIN, ROOT

TAG = PIN.tag
OUT = PIN.out
CKPT = PIN.ckpt


def config_for(key, command):
    config = copy.deepcopy(yaml.safe_load((ROOT / "eval_jobs/eval_tcr_others_longab_v5_29000_nonspot.yml").read_text()))
    runtime = "/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr"
    entry = f"""set -euo pipefail
source /vepfs-mlp2/c20250601/251105016/miniforge3/etc/profile.d/conda.sh
conda activate /vepfs-mlp2/c20250601/251105016/conda/envs/pllm
conda activate {runtime}
export LD_LIBRARY_PATH={runtime}/lib:${{LD_LIBRARY_PATH:-}}
export PYTHONPATH={ROOT}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/vepfs-mlp2/c20250601/251105016/conda/cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TCR_EVAL_TAG={PIN.tag}
export TCR_EVAL_CKPT={CKPT}
export TCR_EVAL_STEP={PIN.step}
export TCR_EVAL_SHA={PIN.sha}
cd {ROOT}
mkdir -p {OUT}/{key}
"""
    config.update(TaskName=f"{TAG}-{key}".replace("_", "-"),
        Description=f"TCR v5 {PIN.step} {key}; audited runtime inputs; independent non-preemptible GPU",
        Entrypoint=entry + command, Tags=["tcr", f"v5-{PIN.step}", "nonspot"],
        Preemptible=False, ActiveDeadlineSeconds=43200)
    return config


def binding_jobs():
    gate_path = OUT / "binding_preflight/passed.json"
    gate = json.loads(gate_path.read_text())
    if gate["status"] != "passed" or int(gate["global_step"]) != PIN.step:
        raise ValueError("binding GPU gate must pass first")
    jobs = {}
    for track, report in gate["tracks"].items():
        key = f"t1-{track}"
        # Re-verify checkpoint and exact input protocol at job startup, before any
        # costly extraction. A changed source tree must obtain a fresh gate.
        guard = (
            "import json; from pathlib import Path; "
            "from downstream.benchmark.tcr_binding.retrained_protocol import sha256_file; "
            "from downstream.benchmark.tcr_binding.query_protocol import TCRBindingQueryProtocol; "
            f"assert sha256_file(Path({str(CKPT)!r})/'model.safetensors') == {gate['checkpoint_sha256']!r}; "
            f"assert TCRBindingQueryProtocol(input_mode={report['input_protocol']['input_mode']!r}).sha256 == {report['input_protocol_sha256']!r}"
        )
        cmd = f"python -c {shlex.quote(guard)}\n"
        cmd += f"""python -u downstream/benchmark/tcr_binding/run_retrained_ours.py \\
  --checkpoint {CKPT} --tag ours_{TAG}_{track} --track {track} \\
  --neg-source AS --fold all --eval-set all --seed 0 --cdr3-format junction --max-length 1024 \\
  --embedding-batch-size 16 --feature-chunk-size 512 --head-hidden 256 --head-batch-size 512 \\
  --dropout 0.3 --learning-rate 0.001 --weight-decay 0.00001 \\
  --max-epochs 200 --patience 15 --val-fraction 0.1 \\
  2>&1 | tee {OUT}/{key}/run.log
"""
        jobs[key] = config_for(key, cmd)
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["binding", "repr", "repr-v2", "generation", "suite-audit"], required=True)
    args = parser.parse_args()
    if args.stage == "binding":
        jobs = binding_jobs()
    elif args.stage == "repr":
        raise ValueError("repr v1 superseded after scoring audit; use --stage repr-v2")
    elif args.stage == "repr-v2":
        from scripts.downstream.run_tcr_native_repr import identity
        gate = json.loads((OUT / "repr_preflight_v2/passed.json").read_text())
        if gate["status"] != "passed":
            raise ValueError("T2/T3 GPU gate must pass first")
        for key, report in gate['tasks'].items():
            if identity(key) != report['identity']:
                raise ValueError(f"{key}: source/data drift since v2 gate")
        jobs = {key + '-v2': config_for(key + '-v2',
            "python -c 'import torch; assert torch.cuda.is_available()'\n"
            "python -m pytest -q scripts/tests/immune_llada/test_tcr_scoring_audit.py\n"
            f"python -u scripts/downstream/run_tcr_native_repr.py --task {key} "
            f"2>&1 | tee {OUT}/{key}-v2/run.log\n") for key in gate["tasks"]}
    elif args.stage == 'suite-audit':
        from scripts.downstream.preflight_tcr_suite import verify_gates
        from downstream.grammar.ab_features import file_sha256
        verify_gates()
        script = ROOT / 'scripts/downstream/preflight_tcr_suite.py'
        guard = f"from pathlib import Path; from downstream.grammar.ab_features import file_sha256; assert file_sha256(Path({str(script)!r})) == {file_sha256(script)!r}"
        command = "python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'\n"
        command += f"python -c {shlex.quote(guard)}\n"
        tests = ['test_tcr_binding_query.py', 'test_tcr_binding_others_beta.py',
                 'test_tcr_binding_long.py', 'test_tcr_native_features.py',
                 'test_tcr_generation_v5.py', 'test_tcr_scoring_audit.py']
        command += 'python -m pytest -q ' + ' '.join('scripts/tests/immune_llada/' + t for t in tests)
        command += ' scripts/tests/bioseq/test_sampling_bioseq.py\n'
        command += f'python -u {script} --out-dir {OUT}/suite_audit_cloud 2>&1 | tee {OUT}/suite-audit/run.log\n'
        config = config_for('suite-audit', command)
        config.update(Description='TCR T1-T4 cloud execution/scoring gate; not quality evaluation', ActiveDeadlineSeconds=3600)
        jobs = {'suite-audit': config}
    else:
        gate = json.loads((OUT / "generation_preflight/passed.json").read_text())
        if gate["status"] != "passed":
            raise ValueError("beta generation GPU gate must pass first")
        jobs = {f"t4-{task}": config_for(f"t4-{task}",
            f"python -u scripts/downstream/run_tcr_native_generation.py --task {task} "
            f"2>&1 | tee {OUT}/t4-{task}/run.log\n") for task in ("benchmark14", "held20")}
    plan = OUT / f"{args.stage}_job_plan.json"
    paths = [ROOT / "eval_jobs" / f"eval_{TAG}_{key}.yml" for key in jobs]
    if plan.exists() or any(path.exists() for path in paths):
        raise FileExistsError("Refusing to overwrite existing job plan/YAML")
    for (key, config), path in zip(jobs.items(), paths):
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        print(f"bash scripts/volc-no-proxy.sh ml_task submit --conf {path}")
    plan.write_text(json.dumps({"checkpoint": str(CKPT), "global_step": PIN.step,
        "preemptible": False, "gpus_per_job": 1, "queue": "c20250601",
        "stage": args.stage, "jobs": [str(p) for p in paths],
        "status": "prepared_not_submitted"}, indent=2))


if __name__ == "__main__":
    main()
