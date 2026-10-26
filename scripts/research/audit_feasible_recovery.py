"""Audit one stable recovery snapshot without evaluating intermediate GDP scores.

Activate pllm/protenix_abtcr; from repository root run:
python -B scripts/research/audit_feasible_recovery.py --run ABS_RUN --arm pool_only --out ABS_NEW --min-step 100 --wait
Checks actual parameter/AdamW tensors, frozen final norm, and saved screen metrics.
Waits only while the original training PID is live; never resumes or restarts it.
"""
import argparse
from datetime import datetime, timezone
import io
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--arm', choices=['pool_only','feasible_token'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--min-step', type=int, default=100)
    parser.add_argument('--wait', action='store_true')
    args=parser.parse_args();run=args.run.resolve();folder=run/args.arm
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',stage='wait_snapshot',pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'status.json',state)
    try:
        while True:
            ready=(folder/'recovery_latest.json').exists()
            if ready:
                meta=json.loads((folder/'recovery_latest.json').read_text())
                ready=meta['step']>=args.min_step
            if ready:
                raw=(folder/'recovery_latest.pt').read_bytes()
                digest=hashlib.sha256(raw).hexdigest()
                if digest==meta['sha256']:break
            assert args.wait,'Requested snapshot not ready'
            training=json.loads((run/'status.json').read_text())
            pid=training['pid'];command=Path(f'/proc/{pid}/cmdline')
            assert training['status']=='running' and command.exists() and b'train_feasible_final_pilot.py' in command.read_bytes(),training
            time.sleep(10)
        state.update(stage='tensor_audit',snapshot_step=meta['step']);write(out/'status.json',state)
        import torch
        torch.set_num_threads(4)
        bundle=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
        del raw
        assert bundle['arm']==args.arm and bundle['step']==meta['step']
        plan=json.loads((run/'plan.json').read_text())
        assert sha(run/'plan.json')==bundle['plan_sha256']
        assert sha(run/'schedule.json')==bundle['schedule_sha256']==plan['schedule_sha256']
        assert bundle['source_sha256']==plan['source_sha256']
        for path,expected in bundle['source_sha256'].items():assert sha(path)==expected,path
        from downstream.fixed_chain_eval import load
        parent,identity=load(plan['parent_checkpoint'],device='cpu')
        assert identity==plan['parent_identity']
        prefix=tuple(f'backbone.transformer.blocks.{i}.' for i in range(2,8))
        parameters=dict(parent.named_parameters())
        ordered=[n for n in parameters if n.startswith(prefix)]
        norm='backbone.transformer.ln_f.weight'
        assert set(ordered)==set(bundle['optimized_parameter_names'])==set(plan['optimized_parameter_names'])
        delta=bundle['parameters'];assert set(delta)==set(ordered)|{norm}
        assert torch.equal(delta[norm],parameters[norm]),'Final norm changed'
        changed=[]
        for name,value in delta.items():
            assert value.shape==parameters[name].shape and torch.isfinite(value).all(),name
            if not torch.equal(value,parameters[name]):changed.append(name)
        assert changed and set(changed)<=set(ordered)
        optimizer=bundle['optimizer'];groups=optimizer['param_groups']
        assert len(groups)==1
        ids=groups[0]['params'];assert len(ids)==len(ordered)==54
        assert set(optimizer['state'])==set(ids)
        for name,param_id in zip(ordered,ids):
            value=optimizer['state'][param_id]
            assert int(value['step'])==meta['step'],(name,value['step'])
            for slot in ['exp_avg','exp_avg_sq']:
                assert value[slot].shape==parameters[name].shape and torch.isfinite(value[slot]).all(),(name,slot)
        assert groups[0]['lr']==plan['learning_rate']*min(meta['step']/plan['warmup'],1.)
        history=json.loads((folder/'validation.json').read_text())
        baseline=json.loads((run/'baseline_validation.json').read_text())
        assert history[0]==baseline
        matching=[x for x in history if x['step']==meta['step']];assert len(matching)==1
        current=matching[0]
        report=dict(status='passed',snapshot_sha256=digest,snapshot_step=meta['step'],arm=args.arm,
            parent_identity=identity,plan_sha256=bundle['plan_sha256'],source_sha256=bundle['source_sha256'],
            auditor_sha256=sha(Path(__file__)),final_norm_bitwise_original=True,updated_parameter_count=len(changed),
            changed_parameter_names=changed,optimizer_steps_all_equal=True,optimizer_parameter_order=ordered,
            baseline_screen=baseline,current_screen=current,
            spectral_relative_change=current['spectral']/baseline['spectral']-1,
            token_relative_change=current['normalized_token_mse']/baseline['normalized_token_mse']-1,
            interpretation='Intermediate heldout monitors and actual saved tensors only; no GDP/cross-task inference or checkpoint selection. Snapshot bytes audited in memory; live recovery file may later be replaced.')
        write(out/'verification.json',report)
        state.update(status='success',stage='complete',finished_utc=datetime.now(timezone.utc).isoformat())
        print(json.dumps({k:report[k] for k in ['snapshot_step','updated_parameter_count','baseline_screen','current_screen','spectral_relative_change','token_relative_change']},indent=2),flush=True)
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());raise
    finally:write(out/'status.json',state)


if __name__=='__main__':main()
