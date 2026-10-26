"""Verify and compare the ten fixed6000 retention evaluations as one suite.

Activate pllm/protenix_abtcr; run from repository root:
python -B scripts/downstream/summarize_final_retention_tasks.py --out ABS_NEW --wait
Without --wait, all ten task evaluations and the endpoint audit must be complete.
TCR original/CE1/control/replay comparisons hold attention mode fixed. AB probes
hold no-mask attention fixed for CE1/control/replay; original AB uses native.
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
    run = root/'final_layer_retention_pilot_20261018'
    cross = root/'final_layer_retention_cross_task_20261018'
    audit = root/'final_layer_retention_audit_20261018'
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
            while True:
                supervision = json.loads((root/'final_layer_retention_recovery_supervision_v2_20261019.json').read_text())
                if supervision['status'] == 'success': break
                assert supervision['status'] == 'running' and live(supervision['pid'], 'continue_retention_after_recovery_v2.py'), supervision
                time.sleep(10)
            while True:
                complete = True
                for lane, pid in supervision['queue_pids'].items():
                    path = cross/f'driver_{lane}/status.json'
                    if not path.exists():
                        assert live(pid, 'run_final_retention_cross_tasks.py'); complete = False; continue
                    current = json.loads(path.read_text()); assert current['pid'] == pid
                    if current['status'] != 'success':
                        assert current['status'] == 'running' and live(pid, 'run_final_retention_cross_tasks.py'), current
                        complete = False
                if complete: break
                time.sleep(10)
        state['stage'] = 'verify_and_compare'; write(out/'status.json',state)
        endpoint = check_report(audit/'verification.json')
        assert endpoint['fixed_endpoint'] == 6000
        recovery = read(run/'recovery_audit_v2_20261019/recovery.json')
        assert recovery['status'] == 'success' and recovery['overlapping5520_numerical_checks_passed']
        arms = ['ab_control','tcr_cdr_replay']; identities = {}
        for arm in arms:
            side = read(run/arm/'checkpoint.json')
            assert side['steps'] == 6000 and sha(side['parameter_file']) == side['sha256']
            identities[arm] = side['sha256']
            for name in ['t3broad_native','t3broad_no_mask_keys','cdr_native','specificity_no_mask_keys','m396_no_mask_keys']:
                folder = cross/arm/name
                assert read(folder/'status.json')['status'] == 'success'
                report = check_report(folder/'comparison_verification.json')
                identity = report['model_identity'] if name == 'cdr_native' else read(folder/'execution.json')['identity']
                assert identity['adaptation_sha256'] == side['sha256']
        summary = dict(GDP=read(audit/'results.json')['GDP'], TCR={}, AB={}, CDR={})
        for mode in ['native','no_mask_keys']:
            folders = dict(original=parent/'t3broad' if mode=='native' else initial/'t3broad_parent_no_mask_keys',
                           ce1_5000=initial/('t3broad_joint_native' if mode=='native' else 't3broad_no_mask_keys'))
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
                    replay_minus_control=scores['tcr_cdr_replay']-scores['ab_control'],
                    replay_minus_original=scores['tcr_cdr_replay']-scores['original']))
        for task in ['specificity','m396']:
            folders = dict(original=parent/task, ce1_5000=initial/(task+'_no_mask_keys'))
            folders.update({arm:cross/arm/(task+'_no_mask_keys') for arm in arms})
            data = {name:read(folder/'metrics.json') for name,folder in folders.items()}
            check_report(initial/(task+'_no_mask_keys')/'comparison_verification.json')
            if task == 'specificity':
                summary['AB'][task] = {key:dict(scores={name:d['mean'][key] for name,d in data.items()},
                    replay_minus_control_folds=[a[key]-b[key] for a,b in zip(data['tcr_cdr_replay']['folds'],data['ab_control']['folds'])])
                    for key in ['accuracy','f1_macro','mcc']}
            else:
                rows = []
                for i, original in enumerate(data['original']['splits']):
                    for current in data.values():
                        for key in ['split_protocol','train_fraction','train_index_sha256','test_index_sha256']:
                            assert current['splits'][i][key] == original[key]
                    scores = {name:d['splits'][i]['spearman_mean'] for name,d in data.items()}
                    rows.append(dict(protocol=original['split_protocol'],fraction=original['train_fraction'],scores=scores,
                                     replay_minus_control=scores['tcr_cdr_replay']-scores['ab_control']))
                summary['AB'][task] = rows
        folders = dict(original=parent/'cdr-kong/kong',ce1_5000=initial/'cdr_native/kong')
        folders.update({arm:cross/arm/'cdr_native/kong' for arm in arms})
        cdr = {name:read(folder/'metrics.json')['modes'] for name,folder in folders.items()}
        check_report(initial/'cdr_native/comparison_verification.json')
        for region in ['cdrh1','cdrh2','cdrh3']:
            scores = {name:d[region]['airgen_aar_percent'] for name,d in cdr.items()}
            summary['CDR'][region] = dict(scores=scores,replay_minus_control_pp=scores['tcr_cdr_replay']-scores['ab_control'],
                replay_minus_control_folds_pp=[a-b for a,b in zip(cdr['tcr_cdr_replay'][region]['fold_means'],cdr['ab_control'][region]['fold_means'])])
        summary['limitations'] = ['Single training seed; no independent stability confirmation.',
            'TCR compares identical attention modes; AB original is native while CE1/control/replay use no-mask keys.',
            'TCR supports and aggregation verified; distance predictions not independently recomputed.',
            'Intervention resumed from exact5500 state; subsequent numerical trajectory is not bitwise deterministic.']
        write(out/'results.json',summary)
        hashes[str(Path(__file__))] = sha(__file__)
        write(out/'verification.json',dict(status='passed',task_count=10,actual_parameter_sha256=identities,
            same_attention_tcr_comparisons=True,source_sha256=hashes))
        state.update(status='success',stage='complete')
    except BaseException:
        state.update(status='failed',error=traceback.format_exc()); raise
    finally:
        state['updated_utc'] = datetime.now(timezone.utc).isoformat(); write(out/'status.json',state)


if __name__ == '__main__': main()
