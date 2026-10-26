"""Audit full benchmark input lengths without fitting heads or using test scores.

Activate pllm/protenix_abtcr; python -B SCRIPT --run /absolute/evaluation/run.
Writes input_audit.json; original rows, labels and partitions are never edited.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'downstream/benchmark'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    from downstream.bert_full_token_eval import load, sha
    model, identity = load(args.run / 'checkpoint', 'cpu')
    from downstream.grammar import ab_probes
    from downstream.grammar.ab_fv_input import adapt_pairs, MANIFEST
    from downstream.grammar.tcr_features import TCRRepresentationProtocol
    from tcr_binding import run_retrained_ours as binding
    from tcr_binding.run_uptcr_ours import load_frames, DEFAULT_DATA, COUNTS
    from scripts.downstream.run_tcr_native_repr import inputs
    from downstream.benchmark.ab_antigen import sequence_eval
    from downstream.benchmark.ab_antigen.abbibench_scope import select_rows
    import torch
    torch.set_num_threads(4)
    checks, paths = {}, set()
    caps = (model.config.first_chain_capacity - 1, model.config.second_chain_capacity - 1)

    def lengths(name, pairs):
        pairs = list(pairs)
        maximum = [max(len(p[i]) for p in pairs) for i in (0, 1)]
        bad = [(len(a), len(b)) for a, b in pairs if len(a) > caps[0] or len(b) > caps[1]]
        checks[name] = dict(n_pairs=len(pairs), maximum=maximum, capacities=caps,
                            over_capacity=len(bad), examples=bad[:5])
        print(name, checks[name], flush=True)

    for task in ('specificity', 'gdp_a1', 'm396'):
        _, pairs, files, _ = ab_probes.prepare_task(task)
        paths.update(files)
        if task != 'gdp_a1':
            pairs, _ = adapt_pairs(task, pairs)
            paths.add(MANIFEST)
        lengths(task, pairs)
    archive = binding._verify_data_archive()
    paths.add(Path(archive['path']))
    for track in ('cdr3b', 'others_cdr3b', 'cdr3ab', 'others_longab'):
        frames, _ = binding.load_protocol('AS', track)
        columns = binding.input_columns(track)[1:]
        pairs = set()
        for fold in frames.values():
            for frame in fold.values():
                pairs.update(frame[columns].itertuples(index=False, name=None))
        protocol = binding.TCRBindingQueryProtocol(input_mode=binding.input_mode_for_track(track))
        rendered = []
        for pair in sorted(pairs):
            record = protocol.record('GILGFVFTL', *pair)
            rendered.append(tuple(c.sequence for c in record.chains[1:]))
        lengths('t1-' + track, rendered)
    for task in ('t2a', 't3broad', 't3deep'):
        frames, files, paired = inputs(task)
        paths.update(files)
        cols = ['cdr3b', 'cdr3a'] if paired else ['cdr3b']
        pairs = set()
        for frame in frames:
            pairs.update(frame[cols].itertuples(index=False, name=None))
        protocol = TCRRepresentationProtocol(input_mode='cdr3ab' if paired else 'cdr3b')
        lengths(task, [tuple(c.sequence for c in protocol.record(*pair).chains) for pair in sorted(pairs)])
    for setting in COUNTS:
        frames, provenance = load_frames(DEFAULT_DATA, setting)
        pairs = set()
        for split, (frame, _) in frames.items():
            paths.add(Path(provenance[split]['path']))
            pairs.update(frame[['FV_beta', 'FV_alpha']].itertuples(index=False, name=None))
        lengths('uptcr-' + setting, pairs)
    for name in ('binding', 'spr', 'abbibench'):
        source = sequence_eval.DATA / (name + '.jsonl')
        paths.add(source)
        rows = sequence_eval.read_rows(source)
        if name == 'abbibench':
            rows = select_rows(rows)
        lengths('ag-' + name, [(r['heavy'], r['light']) for r in rows])
        checks['ag-' + name]['max_antigen'] = max(len(r['antigen']) for r in rows)
        assert max(len(r['antigen']) for r in rows) + sum(caps) + 32 <= model.config.max_length
    result = dict(status='passed' if all(c['over_capacity'] == 0 for c in checks.values()) else 'failed',
                  identity=identity, checks=checks,
                  data_sha256={str(p): sha(p) for p in sorted(paths)})
    (args.run / 'input_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    if result['status'] != 'passed':
        raise ValueError('Input capacity mismatch; inspect input_audit.json before submission')


if __name__ == '__main__':
    main()
