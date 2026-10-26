"""Verify the completed existing-model evaluation and its published artifacts.

Activate pllm, then run this absolute file with --root pointing to
/project_c20250601/251105016/project/dllm_test/output/stage1_downstream_9770_20261025.
This reads saved artifacts only; no inference, fitting, or training is performed.
Scientific scope and interpretation remain in the linked review/task documents.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    assert root.is_absolute()
    repo = root.parents[1]
    checks = {}
    paths = {}
    plan = read(root / 'gdp_fixed_ridge/plan.json')
    identity = read(root / 'comparison_identity_audit.json')
    assert identity['status'] == 'passed'
    for name, digest in identity['plan_sha256'].items():
        assert sha(root / name) == digest
    for arm, spec in plan['arms'].items():
        checkpoint = Path(spec['checkpoint'])
        assert sha(checkpoint / 'model.safetensors') == spec['model_sha256'] == identity['model_sha256'][arm]
        assert sha(checkpoint / 'research_config.json') == spec['research_config_sha256']
        assert read(checkpoint / 'run_provenance.json') == spec['provenance']
        for scope, shape in [('gdp_fixed_ridge', [246, 768]), ('experimental_structure', [1177, 768])]:
            folder = root / scope
            receipt = read(folder / f'{arm}_features_receipt.json')
            assert receipt['status'] == 'complete' and receipt['shape'] == shape
            assert receipt['model_sha256'] == spec['model_sha256'] and receipt['relation_hidden']
            assert sha(folder / f'{arm}_features.npz') == receipt['features_sha256']
            if scope == 'gdp_fixed_ridge':
                assert receipt['parameter_grad_enabled'] is False and receipt['residue_counts_verified']
            else:
                assert receipt['trainable_parameters'] == 0 and receipt['counts_verified'] and receipt['full_pair_forward']
        inp = read(root / f'{arm}_input_verification.json')
        assert inp['status'] == 'passed' and inp['native_candidates'] == 15882 and inp['structure_inputs'] == 1177
        assert inp['model_sha256'] == spec['model_sha256']
        for key in ['all_original_sequences_match', 'selected_target_sequences_match', 'mask_partition_covers_every_target_once', 'no_labels_in_forward_features', 'structure_pooling_excludes_special_tokens']:
            assert inp[key]
    checks['same_checkpoints_full_frozen_features_and_input_coverage'] = True
    gdp = read(root / 'gdp_fixed_ridge/verification.json')
    assert gdp['status'] == 'passed' and gdp['groups_verified'] == 800 and gdp['full_data_readouts_refit'] == 50
    assert gdp['max_prediction_error'] < 1e-10
    assert sha(root / 'gdp_fixed_ridge/predictions.csv') == gdp['predictions_sha256']
    checks['gdp_complete_predictions_independently_verified'] = True
    native = read(root / 'native_partner/metrics.json')
    assert native['status'] == 'complete' and all(native['verification'].values())
    assert (native['candidate_count'], native['query_groups'], native['retrieval_groups']) == (15882, 1000, 999)
    assert (len(native['summaries']), len(native['recovery']), len(native['paired_differences'])) == (48, 80, 72)
    for arm in plan['arms']:
        receipt = read(root / f'native_partner/{arm}_complete.json')
        assert receipt['status'] == 'complete' and receipt['items'] == 15882
        assert receipt['scores_sha256'] == native['original_scores'][arm] == sha(root / f'native_partner/{arm}_scores.jsonl')
    checks['partner_all_candidates_readouts_and_recovery_complete'] = True
    structure = read(root / 'experimental_structure/metrics.json')
    assert structure['status'] == 'complete' and structure['pairs'] == 25588 and structure['queries'] == 583
    assert structure['verification']['all_pair_filters_checked'] and structure['verification']['query_gallery_cluster_disjoint']
    assert structure['verification']['independent_biopython_geometry_pairs'] == 20
    assert structure['verification']['geometry_max_abs_error'] < 1e-10
    checks['experimental_geometry_and_common_readout_complete'] = True
    source_count = 0
    for name in ['gdp', 'structure', 'structure_homology', 'partner', 'partner_homology']:
        published = repo / 'downstream/benchmark/outputs' / f'stage1_frozen_9770_{name}_20261025/metrics.json'
        artifact = read(published)
        assert artifact['status'] == ('verified' if name == 'gdp' else 'complete')
        for source in artifact['sources'].values():
            assert sha(source['path']) == source['sha256'], source['path']
            source_count += 1
        paths[str(published)] = sha(published)
        if name == 'partner':
            assert len(artifact['supporting_recovery_context']['summaries']) == 40
            for source in artifact['supporting_recovery_context']['outputs'].values():
                assert sha(source['path']) == source['sha256']
    checks['published_source_hashes'] = source_count
    for scope in ['native_partner', 'experimental_structure', 'homology_sensitivity/structure', 'homology_sensitivity/partner']:
        folder = root / scope
        artifact = read(folder / 'metrics.json')
        for name, digest in artifact['outputs'].items():
            assert sha(folder / name) == digest
    for task in ['structure', 'partner']:
        folder = root / 'homology_sensitivity' / task
        sensitivity = read(folder / 'metrics.json')
        p = read(folder / 'plan.json')
        assert p['thresholds'] == [.9, .95] and p['inference'] is False and p['training'] is False
        assert sha(p['original_metrics']['path']) == p['original_metrics']['sha256']
        assert sorted(x['cutoff'] for x in sensitivity['coverage']) == [.9, .95]
    assert read(root / 'shared_homology/receipt.json')['status'] == 'complete'
    checks['both_homology_thresholds_and_original_results_preserved'] = True
    runtime = read(root / 'runtime_terminal_evidence.json')
    assert runtime['inference_session']['exit_code'] == runtime['scoring_session']['exit_code'] == 0
    assert read(root / 'native_partner/finish_scoring_status.json')['phase'] == 'complete'
    for pid in [runtime['inference_driver_pid'], runtime['pair_pid'], runtime['scoring_pid']]:
        assert not Path(f'/proc/{pid}').exists()
    checks['original_runtime_terminal'] = True
    figures = []
    for folder, stem in [('gdp_fixed_ridge', 'gdp_fewshot_zh'), ('gdp_fixed_ridge', 'gdp_deltas_zh'), ('experimental_structure', 'experimental_structure_retrieval_zh'), ('native_partner', 'partner_retrieval_zh')]:
        for suffix in ['png', 'pdf']:
            path = root / folder / f'{stem}.{suffix}'
            assert path.stat().st_size > 1000
            figures.append(dict(path=str(path), sha256=sha(path)))
    checks['standalone_plot_artifacts'] = len(figures)
    result_text = (repo / 'downstream/benchmark/RESULTS.md').read_text()
    section = result_text.split('<a id="stage1-native-partner-results-20261025"></a>')[1].split('<a id="stage1-experimental-structure-20261025"></a>')[0]
    table_lines = [line for line in section.splitlines() if line.startswith('| ') and not line.startswith('| 方向') and not line.startswith('| 目标链') and not line.startswith('| identity')]
    # Numeric cells are checked against the corresponding source rows below,
    # independently of the publishing command and its formatting loop.
    partner_pub = read(repo / 'downstream/benchmark/outputs/stage1_frozen_9770_partner_20261025/metrics.json')
    sensitivity = read(root / 'homology_sensitivity/partner/metrics.json')
    scores = {'伙伴可见': 'observed_partner', '伙伴遮蔽': 'masked_partner', '上下文增益': 'context_gain'}
    number_count = 0
    recovery_seen = set()
    for line in table_lines:
        cells = [x.strip() for x in line.strip('|').split('|')]
        if cells[0] in ['H→L', 'L→H']:
            target = 'l' if cells[0] == 'H→L' else 'h'
            rows = [next(x for x in native['summaries'] if x['cohort'] == 'human_16' and x['arm'] == arm and x['target'] == target and x['region'] == cells[1] and x['score'] == scores[cells[2]]) for arm in ['mlm', 'pair_kd']]
            expected = [row[k] for row in rows for k in ['recall1', 'recall5', 'mrr']]
            actual = cells[3:]
        elif cells[0] in ['H', 'L']:
            target, region = cells[0].lower(), cells[1]
            key = (target, region)
            if key not in recovery_seen:
                rows = [next(x for x in native['recovery'] if x['cohort'] == 'human_16' and x['arm'] == arm and x['target'] == target and x['region'] == region and x['condition'] == 'observed_partner') for arm in ['mlm', 'pair_kd']]
                expected = [row[k] for k in ['mean_nll', 'mean_accuracy'] for row in rows]
                recovery_seen.add(key)
            else:
                rows = [next(x for x in partner_pub['supporting_recovery_context']['summaries'] if x['cohort'] == 'human_16' and x['arm'] == arm and x['target'] == target and x['region'] == region) for arm in ['mlm', 'pair_kd']]
                expected = [row[k] for k in ['mean_logprob_gain', 'mean_accuracy_gain'] for row in rows]
            actual = cells[2:]
        elif cells[0].startswith('<'):
            cutoff = int(cells[0][1:-1]) / 100
            target = 'l' if cells[2] == 'H→L' else 'h'
            rows = [next(x for x in sensitivity['summaries'] if x['cutoff'] == cutoff and x['arm'] == arm and x['target'] == target and x['region'] == 'all' and x['score'] == scores[cells[3]]) for arm in ['mlm', 'pair_kd']]
            assert int(cells[1]) == rows[0]['queries'] == rows[1]['queries']
            expected = [row[k] for k in ['recall1', 'recall5', 'mrr'] for row in rows]
            actual = cells[4:]
        else:
            raise AssertionError(cells)
        assert len(actual) == len(expected)
        assert all(abs(float(a) - e) <= 0.00000051 for a, e in zip(actual, expected)), cells
        number_count += len(actual)
    assert number_count == 200
    checks['published_partner_numeric_cells_verified'] = number_count
    review = repo / 'docs/STAGE1_TRAINED_MODEL_EVALUATION_REVIEW_20261025.md'
    for text in [section, review.read_text()]:
        for target in re.findall(r'\]\((/[^)\s]+)\)', text):
            local = Path(target.split('#')[0])
            if local == root / 'completion_audit.json':
                continue
            assert local.exists(), target
    checks['new_result_and_completion_review_links_exist'] = True
    output = dict(status='passed',created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Existing final-step models: GDP frozen readout and label-efficiency, native partner retrieval/recovery, experimental structure retrieval; no new pretraining',
        checks=checks,published_metrics=paths,figures=figures,verifier_sha256=sha(__file__),
        interpretation='Evaluation complete; evidence does not establish overall Pair plus intra-chain KD superiority',
        remaining_required_work=[],optional_not_executed=['Non-germline-specific recovery','Learned experimental contact probes','TCR cross-domain extension','New pretraining ablations'])
    (root / 'completion_audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status='passed',checks=checks),ensure_ascii=False))


if __name__ == '__main__':
    main()
