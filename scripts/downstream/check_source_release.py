"""Check first-party downstream source visibility and imports from an isolated copy.

Run: python scripts/downstream/check_source_release.py --smoke-copy
Copies git-visible source only (including unstaged repairs), never data or weights.
Uses the current CPU Python environment, not an installation of the training stack.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = (
    "downstream/grammar/cdr_infill.py", "downstream/grammar/cdr_result_io.py",
    "downstream/grammar/common.py", "downstream/grammar/masks.py", "downstream/grammar/metrics.py",
    "downstream/grammar/ab_fv_input.py", "downstream/grammar/ab_probes.py", "downstream/grammar/ab_features.py",
    "downstream/ophiuchus_eval/probe_common.py", "downstream/ophiuchus_eval/affinity_m396.py",
    "downstream/ophiuchus_eval/developability.py", "downstream/ophiuchus_eval/specificity.py",
    "downstream/benchmark/common/reporting.py", "downstream/benchmark/common/baseline_provenance.py",
    "downstream/benchmark/common/provenance.py", "downstream/benchmark/common/fewshot.py",
    "downstream/benchmark/tcr_binding/original_protocol.py", "downstream/benchmark/tcr_binding/query_protocol.py",
    "downstream/benchmark/tcr_binding/retrained_protocol.py", "downstream/benchmark/tcr_binding/run_uptcr_ours.py",
    "downstream/benchmark/scripts/summarize_nm2025.py", "downstream/benchmark/scripts/summarize_tcr_representation_paper6.py",
    "downstream/benchmark/scripts/summarize_tcr_representation.py", "downstream/benchmark/scripts/audit_baseline_provenance.py",
    "downstream/benchmark/requirements-cpu.txt", "downstream/benchmark/configs/integrity_20260930.json",
    "downstream/tasks/AB_CDR_INFILLING.md", "downstream/tasks/TCR_T1_BINDING.md", "downstream/tasks/TCR_T3_REPRESENTATION.md",
    "scripts/downstream/prepare_ab_native_jobs.py", "scripts/downstream/prepare_v2_immune_v6_eval_jobs.py",
    "scripts/downstream/prepare_v2_noesmc_150000_eval_jobs.py", "scripts/downstream/run_immune_fusion_gen.sh",
    "eval_jobs/eval_immune_270m_42000_pairing_reference_v3.yml",
    "scripts/downstream/audit_result_integrity.py", "scripts/downstream/check_source_release.py",
    "scripts/tests/test_downstream_reporting.py", "scripts/tests/bioseq/test_cdr_result_io.py",
    "scripts/tests/bioseq/test_cdr_artifacts_integration.py",
)


def visible_sources() -> set[str]:
    output = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT)
    return set(output.decode().split("\0")) - {""}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-copy", action="store_true")
    args = parser.parse_args(argv)
    visible = visible_sources()
    missing = [name for name in REQUIRED if name not in visible or not (ROOT / name).is_file()]
    if missing:
        raise RuntimeError(f"required source ignored or missing: {missing}")
    protected = ["downstream/benchmark/outputs/synthetic.json", "downstream/benchmark/data/synthetic.csv",
                 "downstream/benchmark/baselines/third_party_vendor/model.py", "downstream/grammar/weights.safetensors"]
    ignored = subprocess.run(["git", "check-ignore", "--stdin"], input="\n".join(protected) + "\n",
                             text=True, capture_output=True, cwd=ROOT, check=False)
    if set(ignored.stdout.splitlines()) != set(protected):
        raise RuntimeError("data/vendor/weight ignore protection regressed")
    if args.smoke_copy:
        with tempfile.TemporaryDirectory(prefix="downstream-source-") as temporary:
            copy_root = Path(temporary)
            for name in visible:
                source = ROOT / name
                if source.is_file() and not source.is_symlink() and (source.suffix in {".py", ".sh"} or name in REQUIRED):
                    target = copy_root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
            env = {**os.environ, "PYTHONPATH": str(copy_root), "CUDA_VISIBLE_DEVICES": "", "USE_TF": "0",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "2"}
            smoke = """
import importlib, pathlib, sys
root = pathlib.Path.cwd()
for name in ['downstream.benchmark.common.reporting', 'downstream.benchmark.common.fewshot',
             'downstream.benchmark.tcr_binding.query_protocol', 'downstream.benchmark.tcr_binding.retrained_protocol',
             'downstream.benchmark.scripts.summarize_nm2025', 'downstream.benchmark.scripts.summarize_tcr_representation_paper6',
             'downstream.grammar.cdr_result_io', 'downstream.grammar.ab_probes', 'scripts.downstream.audit_result_integrity']:
    importlib.import_module(name)
for name, module in list(sys.modules.items()):
    if name.startswith(('downstream.', 'dllm.', 'scripts.')) and getattr(module, '__file__', None):
        assert pathlib.Path(module.__file__).resolve().is_relative_to(root), (name, module.__file__)
print('source-copy imports: pass')
"""
            subprocess.run([sys.executable, "-c", smoke], cwd=copy_root, env=env, check=True)
            subprocess.run([sys.executable, "-m", "pytest", "-q", "-o", "addopts=",
                            "scripts/tests/test_downstream_reporting.py", "scripts/tests/bioseq/test_cdr_result_io.py",
                            "scripts/tests/bioseq/test_cdr_artifacts_integration.py"],
                           cwd=copy_root, env=env, check=True)
    print(json.dumps({"status": "pass", "required_files": len(REQUIRED), "source_copy": args.smoke_copy}))


if __name__ == "__main__":
    main()
