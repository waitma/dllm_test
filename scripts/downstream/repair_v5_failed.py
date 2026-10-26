"""Recover saved scores or rerun TCR: python -B SCRIPT --run ABS --phase scores|generation.

Preserves original attempts and raw predictions. No training or derived ckpt.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT=Path('/project_c20250601/251105016/project/dllm_test')
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'downstream/benchmark'))
from downstream.v5_native_eval import sha
from scripts.downstream.run_v5_suite_task import write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--phase',choices=['scores','generation'],required=True)
    args=parser.parse_args();run=args.run
    from scripts.downstream.report_v5_suite import update
    model=identity=None
    if args.phase=='generation':
        from downstream.v5_native_eval import load
        model,identity=load(run/'checkpoint')
        # dllm is now bound to the verified frozen package. Evaluation helpers
        # and their data roots must come from the live benchmark repository.
        sys.path.remove(identity['source_root'])
    tasks=['t1-cdr3b','ag-chimera'] if args.phase=='scores' else ['t4-benchmark14','t4-held20']
    failed=[]
    for task in tasks:
        out=run/task
        old=json.loads((out/'status.json').read_text())
        if old['status']=='success':continue
        if old['status']!='failed':raise ValueError('Task must be terminal failed before recovery')
        history=run/'attempt_history'/(task+'_before_repair_20261009')
        if history.exists():
            history=history.with_name(history.name+'_'+str(time.time_ns()))
        history.parent.mkdir(exist_ok=True)
        if args.phase=='generation':
            shutil.move(str(out),str(history));out.mkdir()
        else:
            history.mkdir();shutil.copy2(out/'status.json',history/'status.json')
            if (out/'run.log').exists():shutil.copy2(out/'run.log',history/'run.log')
        status=dict(status='running',task=task,started_unix=time.time(),repair='20261009',original_attempt=str(history))
        write(out/'status.json',status);update(run)
        try:
            if task=='t1-cdr3b':
                from tcr_binding import run_retrained_ours as binding
                binding.OUT_ROOT=out/'protocol_outputs'
                source=ROOT/'downstream/benchmark/outputs/tcr_binding_nm2025_retrained/comparison_AS.csv'
                shutil.copy2(source,binding.OUT_ROOT/'comparison_AS.csv')
                summary=next(binding.OUT_ROOT.rglob('summary.json'))
                binding.write_rankings(json.loads(summary.read_text()),'AS')
                write(out/'repair.json',dict(baseline_path=str(source),baseline_sha256=sha(source),summary_sha256=sha(summary),inference_rerun=False))
            elif task=='ag-chimera':
                from downstream.benchmark.ab_antigen.sequence_metrics import main as score
                before=sha(out/'predictions.jsonl')
                score(out)
                assert sha(out/'predictions.jsonl')==before
                write(out/'repair.json',dict(predictions_sha256=before,inference_rerun=False,numbering_errors=str(out/'numbering.errors.json')))
            else:
                from downstream.v5_native_generation_eval import tcr_generation
                write(out/'model_identity.json',identity)
                files=['downstream/v5_single_chain_generation.py','downstream/v5_native_generation_eval.py',
                    'downstream/benchmark/tcr_generation_bench/scoring.py','downstream/benchmark/tcr_generation_bench/result_io.py',
                    'downstream/benchmark/data/tcr_generation_bench/eval_conditional.json',
                    'scripts/downstream/run_tcr_native_generation.py','scripts/downstream/repair_v5_failed.py']
                write(out/'repair.json',dict(protocol='single_chain_no_dot_v2',
                    files_sha256={str(ROOT/f):sha(ROOT/f) for f in files},
                    original_raw=str(run/'attempt_history'/(task+'_before_repair_20261009')/'designs.jsonl')))
                tcr_generation(task[3:],out,model,identity)
            status.update(status='success',finished_unix=time.time())
            for pattern in ('**/*.pt','**/*.joblib','**/*features*.npy','**/*.tmp'):
                for path in out.glob(pattern):path.unlink()
        except Exception as error:
            status.update(status='failed',error=repr(error),traceback=traceback.format_exc(),finished_unix=time.time())
            traceback.print_exc();failed.append(task)
        finally:
            write(out/'status.json',status);update(run)
    if failed:raise RuntimeError('Repair failed: '+','.join(failed))


if __name__=='__main__':main()
