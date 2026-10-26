"""Repeat updates5501-5520 twice from the same retention recovery bundle.

Activate pllm/protenix_abtcr and run from repository root:
python -B scripts/research/audit_retention_recovery_repeatability.py --out ABS_NEW
Diagnostic only: no formal checkpoint or completed training output is modified.
Reuses the original forward/loss functions by compiling their unchanged ASTs.
"""
import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    status = dict(status='running', pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'status.json', status)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from downstream.final_layer_preservation_eval import load
    from downstream.fixed_chain_eval import FixedChainBackbone
    from downstream.fixed_chain_layer_eval import block_hidden
    torch.set_num_threads(4)
    run = ROOT/'output/diagnostics/final_layer_retention_pilot_20261018'
    plan = json.loads((run/'plan.json').read_text())
    for group in ['source_sha256', 'control_artifact_sha256', 'replay_source_sha256']:
        for path, digest in plan[group].items(): assert sha(path) == digest, path
    bundle_meta = json.loads((run/'tcr_cdr_replay/recovery_latest.json').read_text())
    assert sha(bundle_meta['bundle']) == bundle_meta['sha256']
    recovered = torch.load(bundle_meta['bundle'], map_location='cpu', weights_only=False)
    assert recovered['global_step'] == 5500
    initial = Path(plan['initial_checkpoint'])
    model, identity = load(initial)
    from dllm.pipelines.immune_llada.fixed_chain.layout import FixedChainCollator
    from dllm.pipelines.immune_llada.data.records import BioSeqRecord
    from downstream.final_retention_objectives import RetentionReplay
    train_names = set(recovered['ordered_parameter_names'])
    for name, parameter in model.named_parameters(): parameter.requires_grad_(name in train_names)
    backbone = FixedChainBackbone(model, identity)
    collator = FixedChainCollator.from_config(model.config, mode='train')
    data = ROOT/'output/diagnostics/final_layer_preservation_pilot_20261018'
    groups = {split:[json.loads(line) for line in (data/f'{split}.jsonl').read_text().splitlines()] for split in ['train','screen']}
    cal = np.load(plan['calibration'])
    spectrum = np.load(run/'spectrum.npz'); eigen, vh, weights = [spectrum[key] for key in ['eigenvalues','axes','weights']]
    denominator = float(np.mean(eigen*weights))
    mu_s,mu_t,rms_s,rms_t,rotation = [torch.as_tensor(cal[key], device='cuda', dtype=torch.float32) for key in ['mu_s','mu_t','rms_s','rms_t','rotation']]
    axes = torch.tensor(vh.T, device='cuda', dtype=torch.float32)
    weight = torch.tensor(weights, device='cuda', dtype=torch.float32)
    original = ROOT/'scripts/research/train_final_layer_retention_pilot.py'
    main_ast = next(node for node in ast.parse(original.read_text()).body if isinstance(node,ast.FunctionDef) and node.name=='main')
    definitions = [node for node in main_ast.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))]
    namespace = dict(locals())
    exec(compile(ast.Module(body=definitions,type_ignores=[]), str(original), 'exec'), namespace)
    replay = RetentionReplay(ROOT, model, backbone, groups, sha)
    replay.weights = json.loads((run/'replay_normalization.json').read_text())['weights']
    schedule = json.loads((run/'schedule.json').read_text())
    tcr_schedule = json.loads((run/'tcr_schedule.json').read_text())
    original5520 = next(row for row in [json.loads(line) for line in (run/'recovery_audit_20261019/tcr_cdr_replay/train.jsonl').read_text().splitlines()] if row['step']==5520)
    runs = []
    previous_parameters = None
    differences = []
    for trial in range(2):
        model.load_state_dict(recovered['trained_parameters'], strict=False)
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=plan['lr'], weight_decay=.01)
        # load_state_dict may retain tensor objects; independently reload for each trial.
        optimizer_bundle = torch.load(bundle_meta['bundle'], map_location='cpu', weights_only=False)
        optimizer.load_state_dict(optimizer_bundle['optimizer']); del optimizer_bundle
        assert {int(v['step'].item()) for v in optimizer.state.values()} == {5200}
        for name,parameter in model.named_parameters():
            if name in train_names: assert torch.equal(parameter.detach().cpu(), recovered['trained_parameters'][name])
        logs = []
        for local_step in range(501,521):
            step = 5000+local_step
            torch.manual_seed(91000+step)
            indices = schedule[local_step-1]; rows = [groups['train'][i] for i in indices]
            optimizer.zero_grad(set_to_none=True)
            batch = namespace['device'](collator(rows))
            with torch.autocast('cuda',dtype=torch.bfloat16): ce = model(**batch)['loss']
            h,t,m = namespace['clean'](rows,True,True)
            rep,detail = namespace['objective'](h,t,m,True,'token_all_direction')
            raw = replay.raw('train',indices,tcr_schedule[local_step-1],(local_step-1)%3,True)
            retention = replay.weighted(raw); loss = rep+ce+retention
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            assert torch.isfinite(loss) and torch.isfinite(gradient)
            optimizer.step()
            logs.append(dict(step=step,ce=float(ce.detach()),representation_loss=float(rep.detach()),
                total_loss=float(loss.detach()),gradient_norm=float(gradient),replay_weighted=float(retention.detach()),
                replay_raw={key:float(value.detach()) for key,value in raw.items()},**detail))
        write(out/f'trial{trial}.json', logs); runs.append(logs)
        parameters = {name:p.detach().cpu().clone() for name,p in model.named_parameters() if name in train_names}
        if previous_parameters is not None:
            for name,value in parameters.items():
                delta = value-previous_parameters[name]
                differences.append(dict(name=name,bitwise_equal=torch.equal(value,previous_parameters[name]),
                    max_abs=float(delta.abs().max()),rms=float(delta.square().mean().sqrt())))
        previous_parameters = parameters
        print('trial',trial,'5520',logs[-1],flush=True)
        del optimizer
    comparison = {key:dict(original=original5520[key],trial0=runs[0][-1][key],trial1=runs[1][-1][key])
                  for key in ['ce','representation_loss','total_loss','gradient_norm','replay_weighted']}
    report = dict(status='passed',starting_parameters_and_optimizer_verified=True,original5520=comparison,
        two_trials_logs_bitwise_equal=runs[0]==runs[1],parameter_differences=differences,
        torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        source_sha256={str(original):sha(original),str(Path(__file__).resolve()):sha(__file__)},
        bundle_sha256=bundle_meta['sha256'],limitations='Repeatability diagnostic; no formal endpoint or task score.')
    write(out/'results.json',report)
    status.update(status='success',finished_utc=datetime.now(timezone.utc).isoformat())
    write(out/'status.json',status)
    print('Repeated logs equal',report['two_trials_logs_bitwise_equal'],'max parameter delta',max(row['max_abs'] for row in differences),flush=True)


if __name__ == '__main__': main()
