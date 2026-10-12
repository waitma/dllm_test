"""Read-only verification of a newly saved V5 checkpoint and validation metrics.

Activate pllm; python -B /absolute/path/review_v5_checkpoint.py --step 1000.
Reads saved artifacts only: no model/tensor loading, dataset access, training,
evaluation reruns, pruning, or process control. Existing reviews are not replayed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import runpy

ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
RUN = ROOT/'output/immune_v5_object302_noesmc_300k_b512'
SUP = ROOT/'output/immune_v5_object302_noesmc_300k_b512_supervision'
BUNDLE = ROOT/'output/v5_full_binding_20261007/launch_v3'
HELPER = ROOT/'scripts/research/review_v7_formal_checkpoint.py'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_removed_archives(retention, helper):
    """Check metadata preserved after native retention removed a full checkpoint."""
    require, read = helper['require'], helper['read']
    records = []
    for candidate in retention['remove']:
        checkpoint = Path(candidate['path'])
        require(checkpoint.parent == RUN and not checkpoint.exists(), 'Unexpected retained or foreign removal')
        archive = RUN/'checkpoint_retention_evidence'/f"checkpoint-{candidate['step']}_{candidate['binding_sha256'][:12]}"
        sidecar = archive/'formal_checkpoint.json'
        require(sha(sidecar) == candidate['sidecar_sha256'], 'Pruned checkpoint receipt changed')
        side = read(sidecar)
        require(side['step'] == candidate['step'] and side['binding_sha256'] == candidate['binding_sha256'],
                'Pruned checkpoint archive binding differs')
        expected = {name: digest for name, digest in side['files'].items() if name.endswith('.json')}
        expected['formal_checkpoint.json'] = candidate['sidecar_sha256']
        require({p.name for p in archive.iterdir()} == set(expected), 'Pruned metadata inventory differs')
        for name, digest in expected.items():
            path = archive/name
            require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                    'Pruned checkpoint metadata changed: '+name)
        records.append({'step': candidate['step'], 'archive': str(archive),
                        'binding_sha256': candidate['binding_sha256'], 'files': expected})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, required=True)
    options = parser.parse_args()
    step = options.step
    helper = runpy.run_path(str(HELPER), run_name='_saved_metadata_helpers')
    require, read, equal, average = (helper[name] for name in ('require','read','equal','average'))
    require(0 < step <= 300000 and step % 1000 == 0, 'Expected a scheduled new checkpoint')
    target = SUP/f'checkpoint{step}_root_review.json'
    require(not target.exists(), 'Already reviewed; reuse the existing evidence')
    checkpoint = RUN/f'checkpoint-{step}'
    side = read(checkpoint/'formal_checkpoint.json')
    binding = read(checkpoint/'run_provenance.json')
    binding_sha = hashlib.sha256(json.dumps(binding,sort_keys=True,allow_nan=False).encode()).hexdigest()
    require(binding == read(RUN/'run_provenance.json'), 'Checkpoint provenance differs from run')
    require(side['schema']=='v7.formal.native_checkpoint.v1' and side['step']==step
            and side['binding_sha256']==binding_sha and side['world_size']==8
            and side['writer_rank']==0 and side['rank_acknowledgments']==list(range(8)),
            'Incomplete or mismatched native save acknowledgment')
    require(binding['source_root']==str(BUNDLE/'source') and binding['config']['ablation']=='V5'
            and binding['config']['full_capacity']==302 and binding['variant']=='cv_segment_global'
            and binding['data_dir']==str(ROOT/'data/prepared/immune_v5_receptor_completion')
            and binding['world_size']==8 and binding['micro_batch']==8 and binding['grad_accum']==8
            and binding['global_batch']==512 and binding['schedule']['max_steps']==300000
            and binding['checkpoint_retention']=='best3_eval_loss_plus_latest_full_state', 'Wrong experiment')
    require(sha(BUNDLE/'source_manifest.json')==binding['source_manifest_sha256'], 'Source manifest changed')
    saved = side['saved_parameter_digest']
    require(saved['numel']==75601176 and side['rank_parameter_digests']==[
        dict(rank=rank,**saved) for rank in range(8)], 'Actual rank/saved parameters disagree')
    names = {'model.safetensors','optimizer.pt','scheduler.pt','training_args.bin','trainer_state.json',
        'immune_config.json','research_config.json','vocabulary.json','run_provenance.json',
        *(f'rng_state_{rank}.pth' for rank in range(8))}
    require(set(side['files'])==names, 'Full 17-file native checkpoint inventory differs')
    sizes = {}
    for name in sorted(names):
        path = checkpoint/name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size>0,
                'Missing or invalid native checkpoint file: '+name)
        require(sha(path)==side['files'][name], 'Saved file hash differs: '+name)
        sizes[name] = path.stat().st_size
    state, metrics = helper['evaluation'](checkpoint/'trainer_state.json',step)
    require(all(not isinstance(v,float) or math.isfinite(v) for v in metrics.values()), 'Nonfinite evaluation')
    sources = {name:len(indices) for name,indices in binding['validation_indices'].items()}
    require(metrics['eval_records_per_mask']==sum(sources.values())
            and metrics['eval_record_evaluations']==4*sum(sources.values()), 'Validation population differs')
    core = helper['CORE']+('split_loss','split_accuracy')
    denominators = {**helper['DENOMINATOR'],'split_loss':'split_count','split_accuracy':'split_count'}
    checks = 0
    for level in helper['LEVELS']:
        for source in sources:
            prefix = f'eval_{level}_{source}_'
            counts = {key:metrics[prefix+key+'_count'] for key in ('selected','aa','terminal','relation','split')}
            require(counts['selected']==sum(counts[k] for k in ('aa','terminal','relation','split')),
                    'Selected target partition differs')
            total_loss = sum((metrics[prefix+k+'_loss'] or 0)*counts[k] for k in ('aa','terminal','relation','split'))
            equal(metrics[prefix+'loss'],total_loss/counts['selected'] if counts['selected'] else None,prefix+'loss')
            for metric in core:
                count = metrics[prefix+denominators[metric]]
                require((metrics[prefix+metric] is None)==(count==0),'Undefined population differs')
        for metric in core:
            pairs = [(metrics[f'eval_{level}_{s}_{metric}'],metrics[f'eval_{level}_{s}_{denominators[metric]}']) for s in sources]
            count = sum(n for _,n in pairs)
            micro = sum(v*n for v,n in pairs if v is not None)/count if count else None
            for weighting,expected in [('token_micro',micro),('source_macro',average([v for v,_ in pairs]))]:
                equal(metrics[f'eval_{level}_{weighting}_{metric}'],expected,level+'/'+weighting+'/'+metric)
                checks += 1
    for weighting in ('token_micro','source_macro'):
        for metric in core:
            equal(metrics[f'eval_mask_mean_{weighting}_{metric}'],average([
                metrics[f'eval_{level}_{weighting}_{metric}'] for level in helper['LEVELS']]),metric)
            checks += 1
    equal(metrics['eval_loss'],metrics['eval_mask100_source_macro_loss'],'checkpoint selection loss')
    retention_path = RUN/'checkpoint_retention_evidence'/f'checkpoint-{step}_decision.json'
    retention = read(retention_path)
    require(retention['status']=='applied' and retention['latest']==str(checkpoint)
            and retention['best_k']==3 and retention['metric']=='eval_loss'
            and retention['all_retained_states']=='full', 'Retention contract differs')
    best = sorted(retention['candidates'],key=lambda x:(x['eval_loss'],x['step']))[:3]
    keep = {x['path'] for x in best}|{str(checkpoint)}
    require(set(retention['best'])=={x['path'] for x in best}
            and {x['path'] for x in retention['keep']}==keep, 'Best3 plus latest selection differs')
    current = read(RUN/'checkpoint_retention.json')
    require(current==retention, 'A newer save is active; inspect before reviewing this checkpoint')
    require({str(p) for p in RUN.glob('checkpoint-*') if p.is_dir()}==keep,
            'Actual retained checkpoint directories differ')
    require(all(not Path(x['path']).exists() for x in retention['remove']), 'Owned nonwinners were not pruned')
    removed_archives = verify_removed_archives(retention, helper)
    archive = RUN/'checkpoint_retention_evidence'/f'checkpoint-{step}_{binding_sha[:12]}'
    for path in checkpoint.glob('*.json'):
        require(sha(path)==sha(archive/path.name),'Checkpoint metadata archive differs')
    history = []
    for path in sorted(SUP.glob('checkpoint*_root_review.json')):
        value = read(path)
        if value['step'] < step:
            history.append((value['step'],path,value))
    previous = max(history,key=lambda item:item[0]) if history else None
    if previous:
        require(previous[2]['binding_sha256']==binding_sha and previous[2]['parameter_digest']!=saved,
                'Run binding changed or parameters did not update')
    summary = {level:{weighting:{key:metrics[f'eval_{level}_{weighting}_{key}'] for key in core}
               for weighting in ('token_micro','source_macro')} for level in ('mask_mean','mask100')}
    report = dict(schema='immune.v5.saved_checkpoint_review.v1',status='passed',
        at_utc=datetime.now(timezone.utc).isoformat(),step=step,planned_steps=300000,
        training_complete=False,full_checkpoint_hashes_rechecked=True,native_inventory_count=17,
        native_file_sizes=sizes,binding_sha256=binding_sha,parameter_digest=saved,
        eight_rank_parameter_digests_equal=True,optimizer_scheduler_rng_loaded_by_root=False,
        checkpoint_resume_executed=False,native_callback_validated_optimizer_scheduler_rng=True,
        source_manifest_sha256=binding['source_manifest_sha256'],source_root=binding['source_root'],
        validation_source_record_counts=sources,metrics=summary,reaggregated_metric_fields=checks,
        separator_in_total_loss=True,endpoint_unit='V5 receptor object, not each component chain',
        retention={'best':retention['best'],'latest':retention['latest'],'kept':sorted(keep),
                   'decision':str(retention_path),'sha256':sha(retention_path)},
        removed_checkpoint_archives_verified=removed_archives,
        native_receipt={'path':str(checkpoint/'formal_checkpoint.json'),'sha256':sha(checkpoint/'formal_checkpoint.json')},
        previous_review=str(previous[1]) if previous else None,
        reviewer={'path':str(Path(__file__).resolve()),'sha256':sha(Path(__file__).resolve()),
                  'helper':str(HELPER),'helper_sha256':sha(HELPER)})
    with target.open('x') as stream:
        json.dump(report,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'review':str(target),'sha256':sha(target),'step':step,'metric_checks':checks,
                      'eval_loss':metrics['eval_loss'],'retained':sorted(keep)},ensure_ascii=False))


if __name__=='__main__':
    main()
