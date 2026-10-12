"""Recompute deterministic argmax bounds for the exact six-toy/four-mask protocol.

After activating pllm:
  USE_TF=0 python scripts/audit_immune_toy_identifiability.py --output-dir /absolute/output
Pass --source-root /absolute/frozen/release to reproduce an archived source version.
Uses actual collator/mask features on CPU. No model, GPU, or checkpoint is loaded.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def bound(batch, groups, selected):
    correct = total = 0
    lower_nll = 0.
    for indices in groups:
        for position in range(batch['input_ids'].shape[1]):
            counts = Counter(int(batch['labels'][index, position]) for index in indices if selected[index, position])
            if counts:
                count = sum(counts.values())
                total += count
                correct += max(counts.values())
                lower_nll += sum(-value*math.log(value/count) for value in counts.values())
    return dict(maximum_correct=correct, target_count=total,
                accuracy_upper_bound=correct/total if total else None,
                empirical_token_ce_lower_bound=lower_nll/total if total else None)


def audit():
    from scripts import run_immune_architecture as runner
    from dllm.pipelines.immune_llada.v3.model import ImmuneModel
    base = runner.base
    collate = base.collator_for(base.ImmuneConfig(ablation='A4'))
    batch = collate(base.toy_rows())
    levels = {}
    for name, ratio in runner.MASK_LEVELS:
        selected = base.fixed_validation_mask(batch, ratio).bool() & batch['eligible_mask'].bool() & batch['attention_mask'].bool()
        features = ImmuneModel.feature_args(batch)
        features['input_ids'] = batch['input_ids'].masked_fill(selected, collate.tokenizer.mask_token_id)
        groups = defaultdict(list)
        for index in range(len(batch['input_ids'])):
            key = hashlib.sha256(json.dumps({key: value[index].tolist() for key, value in features.items()}, sort_keys=True).encode()).hexdigest()
            groups[key].append(index)
        groups = list(groups.values())
        aa = selected & batch['residue_mask']
        terminal = selected & batch['terminal_mask']
        levels[name] = dict(mask_ratio=ratio, input_groups=groups, selected_count=int(selected.sum()),
                            aa_count=int(aa.sum()), terminal_count=int(terminal.sum()),
                            all_targets=bound(batch, groups, selected), aa_only=bound(batch, groups, aa),
                            terminal_only=bound(batch, groups, terminal))
    return dict(time=time.time(), protocol='fixed_per_record_selected_tokens_v1',
                description='Group all seven actual denoise features after exact fixed_validation_mask. Maximize label counts per identical-input group and position. Bounds assume input-deterministic argmax, as in eval-mode denoise. AA/end optima need not be simultaneously attainable.',
                feature_names=list(features), levels=levels,
                mask_mean_aa_upper_bound=sum(level['aa_only']['accuracy_upper_bound'] for level in levels.values())/len(levels),
                mask_mean_all_targets_upper_bound=sum(level['all_targets']['accuracy_upper_bound'] for level in levels.values())/len(levels),
                source_sha256=runner.fingerprints(), audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = audit()
    destination = args.output_dir/'four_mask_identifiability.json'
    temporary = destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(destination)
    print(json.dumps(dict(output=str(destination.resolve()), mask_mean_aa_upper_bound=result['mask_mean_aa_upper_bound'],
                          mask_mean_all_targets_upper_bound=result['mask_mean_all_targets_upper_bound'],
                          groups={name: level['input_groups'] for name, level in result['levels'].items()})))
