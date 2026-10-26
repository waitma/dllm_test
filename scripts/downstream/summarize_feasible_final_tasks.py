"""Verify the paired feasible-target endpoints and all ten actual task evaluations.

Activate pllm/protenix_abtcr; from repository root run:
python -B scripts/downstream/summarize_feasible_final_tasks.py --out ABS_NEW --wait
Waits only on verified live task drivers. Compares identical TCR attention modes,
checks exact frozen final RMSNorm tensors, and binds every task to its checkpoint.
AB original baseline is native, while both new candidates use no-mask keys.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import sha, write


def live(pid, fragment):
    path = Path(f'/proc/{pid}/cmdline')
    return path.exists() and fragment.encode() in path.read_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    root = ROOT/'output/diagnostics'
    run = root/'feasible_final_pilot_20261019'
    cross = root/'feasible_final_cross_task_20261019'
    initial = root/'final_layer_joint_cross_task_20261018'
    parent = ROOT/'output/downstream_generation/fixed_chain_75m_step50000_20261013'
    state = dict(status='running', stage='wait_tasks', pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'status.json', state)
    hashes = {}

    def read(path):
        hashes[str(path)] = sha(path)
        return json.loads(path.read_text())

    def check_report(path):
        report = read(path); assert report['status'] == 'passed'
        for source, digest in report['source_sha256'].items(): assert sha(source) == digest, source
        return report

    try:
        if args.wait:
            launch = read(root/'feasible_final_cross_launch_20261019.json')
            while True:
                complete = True
                for driver in launch['processes']:
                    lane, pid = driver['lane'], driver['pid']
                    path = cross/f'driver_{lane}/status.json'
                    if not path.exists():
                        assert live(pid, 'run_feasible_final_cross_tasks.py'); complete = False; continue
                    current = json.loads(path.read_text()); assert current['pid'] == pid
                    if current['status'] != 'success':
                        assert current['status'] == 'running' and live(pid, 'run_feasible_final_cross_tasks.py'), current
                        complete = False
                if complete: break
                time.sleep(10)
        state['stage'] = 'verify_and_compare'; write(out/'status.json',state)
        assert read(run/'status.json')['status'] == 'success'
        plan = read(run/'plan.json')
        assert plan['steps'] == 1000 and plan['arms'] == ['pool_only','feasible_token']
        for path,digest in plan['source_sha256'].items(): assert sha(path)==digest,path
        from safetensors import safe_open
        import torch
        arms = ['pool_only','feasible_token']; identities = {}
        norm_key = 'backbone.transformer.ln_f.weight'
        with safe_open(str(parent/'checkpoint/model.safetensors'),framework='pt',device='cpu') as handle:
            original_norm = handle.get_tensor(norm_key)
        original_metrics = read(root/'fixed50000_depth_20261018/metrics.json')
        GDP = dict(original_native_block2=original_metrics['native_block2'],
                   original_native_final=original_metrics['native_final_norm'],
                   original_no_mask_final=original_metrics['no_mask_keys_final_norm'])
        for arm in arms:
            side = read(run/arm/'checkpoint.json')
            assert side['steps'] == 1000 and sha(side['parameter_file']) == side['sha256']
            for path,digest in side['source_sha256'].items(): assert sha(path)==digest,path
            assert sha(run/arm/'optimizer.pt')==side['optimizer_sha256']
            verified = read(run/arm/'verification.json')
            assert verified['status']=='passed' and len(verified['oof_checks'])==10
            assert verified['reloaded_features_bitwise_equal'] and verified['final_norm_bitwise_original']
            assert side['frozen_serialized_names']==[norm_key]
            assert norm_key not in side['optimizer_parameter_names']
            with safe_open(side['parameter_file'],framework='pt',device='cpu') as handle:
                assert set(handle.keys())==set(side['trainable_names'])
                assert torch.equal(handle.get_tensor(norm_key),original_norm)
            GDP[arm] = {mode:read(run/arm/folder/'metrics.json') for mode,folder in [('native','gdp_a1'),('no_mask_keys','gdp_no_mask_keys')]}
            identities[arm] = side['sha256']
            for name in ['t3broad_native','t3broad_no_mask_keys','cdr_native','specificity_no_mask_keys','m396_no_mask_keys']:
                folder = cross/arm/name
                assert read(folder/'status.json')['status'] == 'success'
                report = check_report(folder/'comparison_verification.json')
                identity = report['model_identity'] if name == 'cdr_native' else read(folder/'execution.json')['identity']
                assert identity['adaptation_sha256'] == side['sha256']
        summary = dict(GDP=GDP,TCR={},AB={},CDR={})
        for mode in ['native','no_mask_keys']:
            folders = dict(original=parent/'t3broad' if mode=='native' else initial/'t3broad_parent_no_mask_keys')
            folders.update({arm:cross/arm/('t3broad_'+mode) for arm in arms})
            data = {}
            for name, folder in folders.items():
                if folder != parent/'t3broad':
                    assert read(folder/'execution.json')['attention_mode'] == mode
                    check_report(folder/'comparison_verification.json')
                data[name] = read(folder/'fewshot.json')
            reference = data['original']
            for current in data.values():
                for key in ['protocol','shots','seeds','agg','n_train','n_test','n_epitopes_by_shot']:
                    assert current[key] == reference[key]
                assert len(current['episodes']) == len(reference['episodes']) == 775
                for a,b in zip(current['episodes'], reference['episodes']):
                    for key in ['k','epitope','seed','support_indices']: assert a[key] == b[key]
            summary['TCR'][mode] = []
            for shot in reference['shots']:
                scores = {name:d['auroc_by_shot'][str(shot)]['mean'] for name,d in data.items()}
                summary['TCR'][mode].append(dict(shot=shot,scores=scores,
                    token_minus_pool=scores['feasible_token']-scores['pool_only'],
                    token_minus_original=scores['feasible_token']-scores['original']))
        for task in ['specificity','m396']:
            folders = dict(original=parent/task)
            folders.update({arm:cross/arm/(task+'_no_mask_keys') for arm in arms})
            data = {name:read(folder/'metrics.json') for name,folder in folders.items()}
            if task == 'specificity':
                summary['AB'][task] = {key:dict(scores={name:d['mean'][key] for name,d in data.items()},
                    token_minus_pool_folds=[a[key]-b[key] for a,b in zip(data['feasible_token']['folds'],data['pool_only']['folds'])])
                    for key in ['accuracy','f1_macro','mcc']}
            else:
                rows = []
                for i, original in enumerate(data['original']['splits']):
                    for current in data.values():
                        for key in ['split_protocol','train_fraction','train_index_sha256','test_index_sha256']:
                            assert current['splits'][i][key] == original[key]
                    scores = {name:d['splits'][i]['spearman_mean'] for name,d in data.items()}
                    rows.append(dict(protocol=original['split_protocol'],fraction=original['train_fraction'],scores=scores,
                                     token_minus_pool=scores['feasible_token']-scores['pool_only']))
                summary['AB'][task] = rows
        folders = dict(original=parent/'cdr-kong/kong')
        folders.update({arm:cross/arm/'cdr_native/kong' for arm in arms})
        cdr = {name:read(folder/'metrics.json')['modes'] for name,folder in folders.items()}
        for region in ['cdrh1','cdrh2','cdrh3']:
            scores = {name:d[region]['airgen_aar_percent'] for name,d in cdr.items()}
            summary['CDR'][region] = dict(scores=scores,token_minus_pool_pp=scores['feasible_token']-scores['pool_only'],
                token_minus_pool_folds_pp=[a-b for a,b in zip(cdr['feasible_token'][region]['fold_means'],cdr['pool_only'][region]['fold_means'])])
        summary['limitations'] = ['Single training seed; no independent stability confirmation.',
            'TCR compares identical attention modes; AB original is native while both new arms use no-mask keys.',
            'TCR supports and aggregation verified; distance predictions not independently recomputed.',
            'Two arms start from original weights; only token loss differs within this pair. No exact numerical trajectory reproducibility claim.']
        for path,digest in hashes.items(): assert sha(path)==digest,path
        write(out/'results.json',summary)
        hashes[str(Path(__file__))] = sha(__file__)
        write(out/'verification.json',dict(status='passed',task_count=10,actual_parameter_sha256=identities,
            same_attention_tcr_comparisons=True,original_final_norm_tensor_verified=True,source_sha256=hashes))
        state.update(status='success',stage='complete')
    except BaseException:
        state.update(status='failed',error=traceback.format_exc()); raise
    finally:
        state['updated_utc'] = datetime.now(timezone.utc).isoformat(); write(out/'status.json',state)


if __name__ == '__main__': main()
