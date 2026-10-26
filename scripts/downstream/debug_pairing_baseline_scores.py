"""Activate pllm/protenix_abtcr, hide CUDA, then python -B SCRIPT.

Re-score saved fixed-chain candidates on CPU with the original ImmunoMatch
implementation. Reuse the saved K/L allocation only after exact row checking;
never alter original runs. Save per-row scores for paired sampler comparisons.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
GEN = ROOT / 'output/downstream_generation'
SCORER = ROOT / 'downstream/comp_chain/eval_scripts'
sys.path.insert(0, str(SCORER))


def main():
    import numpy as np
    import pandas as pd
    import torch
    from immunomatch_score import pairing_scores_batches, _immunomatch_ckpt
    assert not torch.cuda.is_available(), 'CPU-only audit; set CUDA_VISIBLE_DEVICES empty'
    torch.set_num_threads(4)
    output = GEN / 'pairing_sampler_debug_20261014/baseline_cpu_scores'
    output.mkdir(exist_ok=False)
    os.chdir(output)
    started = time.time()
    for name, directory in [
        ('fixed50000', 'fixed_chain_75m_step50000_pairing_reference_20261013'),
        ('fixed112000', 'fixed_chain_75m_best112000_20261014/pairing-reference'),
    ]:
        source = GEN / directory
        frame = pd.read_csv(source / 'generated.csv', keep_default_na=False)
        assert len(frame) == 4000
        scores = np.full(4000, np.nan)
        files = {}
        for kind in ('kappa', 'lambda'):
            path = source / f'generated_gen_l_sequence_{kind}.csv'
            part = pd.read_csv(path, keep_default_na=False)
            indices = part['_immunomatch_row_id'].to_numpy(dtype=int)
            assert len(set(indices)) == len(indices)
            assert np.isnan(scores[indices]).all()
            for column in ('h_sequence', 'raw_l_sequence', 'gen_l_sequence'):
                assert part[column].tolist() == frame.iloc[indices][column].tolist(), column
            copy = output / f'{name}_{kind}.csv'
            copy.write_bytes(path.read_bytes())
            files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            result = pairing_scores_batches(str(copy), 'h_sequence', 'gen_l_sequence', _immunomatch_ckpt(kind))
            assert result['_immunomatch_row_id'].tolist() == indices.tolist()
            scores[indices] = result['pairing_scores'].to_numpy(dtype=float)
        # Both historical runs have 100% recognized light chains.
        assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
        old = json.loads((source / 'metrics.json').read_text())['gen_immunomatch_mean']
        mean = float(scores.mean())
        assert abs(mean-old) < 1e-4, (name, mean, old)
        report = dict(status='passed', device='cpu', count=4000, gen_l_sequence=scores.tolist(),
                      mean=mean, historical_mean=old, mean_difference=mean-old,
                      source_files=files, elapsed_seconds=time.time()-started)
        (output / f'{name}.json').write_text(json.dumps(report, indent=2)+'\n')
        print(name, mean, 'historical', old, flush=True)


if __name__ == '__main__':
    main()
