"""Check tokenwise shallow targets subject to frozen-head and weighted-norm constraints.

Activate pllm then protenix_abtcr; from repository root run:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 USE_TF=0 python -B scripts/downstream/audit_token_head_norm_targets.py --out ABS_NEW_DIR
Target-only diagnostic on observed GDP residues, not a trained model or masked
generation test. Mapping is fitted on OAS only; existing fixed GDP probes are reused.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', stage='prepare', pid=os.getpid(),
                 started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'status.json', state)
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    torch.set_num_threads(4)
    diagnostic = ROOT/'output/diagnostics'
    original = diagnostic/'fixed50000_depth_20261018'
    cal_path = diagnostic/'final_layer_coordinate_pilot_20261018/calibration.npz'
    head_path = diagnostic/'head_nullspace_targets_20261018/calibration.npz'
    sources = [Path(__file__), cal_path, head_path, original/'features.npz',
               ROOT/'downstream/fixed_chain_eval.py', ROOT/'downstream/fixed_chain_layer_eval.py',
               ROOT/'downstream/grammar/ab_probes.py', ROOT/'data/downstream/dev/GDPa1_v1.2_20250814.csv']
    hashes = {str(p):sha(p) for p in sources}
    assert json.loads((original/'verification.json').read_text())['status'] == 'passed'
    from downstream.fixed_chain_eval import load, FixedChainBackbone
    from downstream.fixed_chain_layer_eval import block_hidden
    checkpoint = ROOT/'output/downstream_generation/fixed_chain_75m_step50000_20261013/checkpoint'
    model, identity = load(checkpoint)
    from dllm.pipelines.immune_llada.data.records import BioSeqRecord, BioSeqChain
    from downstream.grammar.ab_probes import prepare_task, run_gdp
    from downstream.ophiuchus_eval.developability import ALL_COLS, FOLD_COL
    backbone = FixedChainBackbone(model, identity, batch_size=8)
    norm = model.backbone.transformer.ln_f
    head = model.backbone.transformer.ff_out
    assert getattr(norm, 'bias', None) is None and getattr(head, 'bias', None) is None
    w = head.weight.detach().double()
    gamma = norm.weight.detach().double()
    assert w.shape == (50, 768) and gamma.abs().min() > 0
    cached_head = np.load(head_path)
    np.testing.assert_array_equal(w.cpu().numpy(), cached_head['head'])
    np.testing.assert_array_equal(gamma.cpu().numpy(), cached_head['gamma'])
    # In pre-affine coordinates z=h/gamma, the head is C=W diag(gamma).
    c = w*gamma[None, :]
    _, singular, vh = torch.linalg.svd(c, full_matrices=False)
    tolerance = max(c.shape)*torch.finfo(torch.float64).eps*singular[0]
    rank = int((singular > tolerance).sum())
    basis = vh[:rank].T.contiguous()
    torch.testing.assert_close(c-(c@basis)@basis.T, torch.zeros_like(c), atol=1e-11, rtol=0)
    cal = np.load(cal_path)
    mapping = {k:torch.as_tensor(cal[k], device='cuda', dtype=torch.float64)
               for k in ['mu_s', 'mu_t', 'rms_s', 'rms_t', 'rotation']}
    names = ['mapped_early', 'norm_only', 'head_and_norm']
    write(out/'plan.json', dict(parent_identity=identity, source_sha256=hashes,
        rank=rank, null_dimension=768-rank, variants=names, batch_size=8,
        attention='native', precision='BF16 original forward, FP32 baseline pool; float64 target construction and pool/Ridge',
        mapping='existing OAS4096 pooled early-to-final Procrustes; applied tokenwise without GDP fitting',
        constraint='preserve observed per-token raw outputs of all 50 head rows and ||h/gamma||, in float64 algebra',
        norm_eps=float(norm.eps), labels_used_to_construct_targets=False,
        limitation='No student training, no inference change, no masked-context or CDR claim. Preserves actual numerical weighted norm, not proof of exact BF16 RMSNorm realizability. Same-CV Ridge alpha selection.'))
    np.savez_compressed(out/'calibration.npz', head=w.cpu().numpy(), gamma=gamma.cpu().numpy(),
                        row_basis=basis.cpu().numpy(), singular_values=singular.cpu().numpy())
    df, pairs, _, _ = prepare_task('gdp_a1')
    records = [BioSeqRecord([BioSeqChain(h, 'antibody_heavy'), BioSeqChain(l, 'antibody_light')],
                 task_type='antibody', source='native_v5_probe') for h, l in pairs]
    chunks = {name:[] for name in names+['native_block2', 'native_final_norm']}
    checks = []; token_stats = []
    for start in range(0, len(records), 8):
        batch = backbone.collator(records[start:start+8])
        batch = {k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
        mask = batch['residue_mask'] & batch['attention_mask']
        lengths = mask.sum(1).tolist()
        assert lengths == [len(h)+len(l) for h,l in pairs[start:start+8]]
        captured = {}
        def hook(module, inputs, output):
            captured['early'] = block_hidden(output).detach()
        handle = model.backbone.transformer.blocks[1].register_forward_hook(hook)
        try:
            hidden, _ = backbone._final_hidden(batch)
        finally:
            handle.remove()
        def original_pool(h):
            return (h.float().masked_fill(~mask[..., None], 0).sum(1)/mask.sum(1, keepdim=True)).cpu().numpy()
        chunks['native_block2'].append(original_pool(captured['early']))
        chunks['native_final_norm'].append(original_pool(hidden))
        early = captured['early'][mask].double()
        h = hidden[mask].double()
        desired = mapping['mu_s']+((early-mapping['mu_t'])/mapping['rms_t'])@mapping['rotation']*mapping['rms_s']
        z = h/gamma
        row = (z@basis)@basis.T
        radius = torch.linalg.vector_norm(z-row, dim=1, keepdim=True)
        q = desired/gamma
        q_null = q-(q@basis)@basis.T
        q_norm = torch.linalg.vector_norm(q, dim=1, keepdim=True)
        q_null_norm = torch.linalg.vector_norm(q_null, dim=1, keepdim=True)
        assert q_norm.min() > 1e-12 and q_null_norm.min() > 1e-12
        z_norm = torch.linalg.vector_norm(z, dim=1, keepdim=True)
        values = dict(mapped_early=desired,
                      norm_only=gamma*q*(z_norm/q_norm),
                      head_and_norm=gamma*(row+q_null*(radius/q_null_norm)))
        target = values['head_and_norm']
        base_logits = h@w.T
        target_logits = target@w.T
        target_norm = torch.linalg.vector_norm(target/gamma, dim=1, keepdim=True)
        torch.testing.assert_close(target_logits, base_logits, atol=1e-9, rtol=1e-10)
        torch.testing.assert_close(target_norm, z_norm, atol=1e-9, rtol=1e-10)
        checks.append(dict(start=start, tokens=len(h),
            max_raw_head_error=float((target_logits-base_logits).abs().max()),
            max_weighted_norm_error=float((target_norm-z_norm).abs().max()),
            null_direction_min_norm=float(q_null_norm.min())))
        token_stats.append(dict(tokens=len(h),
            original_weighted_rms_max=float((z_norm/(768**0.5)).max()),
            original_weighted_rms_above_one=int((z_norm/(768**0.5)>1+1e-7).sum()),
            parent_row_energy_fraction=float(row.square().sum()/z.square().sum()),
            target_relative_distance=float((target-h).square().sum()/h.square().sum())))
        for name, value in values.items():
            chunks[name].append(torch.stack([part.mean(0) for part in value.split(lengths)]).cpu().numpy())
        state.update(stage='extract', rows=min(start+8, len(records))); write(out/'status.json', state)
    features = {k:np.concatenate(v) for k,v in chunks.items()}
    baseline = np.load(original/'features.npz')
    for name in ['native_block2', 'native_final_norm']:
        np.testing.assert_array_equal(features[name], baseline[name])
    np.savez_compressed(out/'features.npz', **features)
    write(out/'token_checks.json', dict(checks=checks, token_stats=token_stats,
        tokens=sum(x['tokens'] for x in checks), all50_raw_head_and_weighted_norm_preserved=True))
    del model, backbone, head, norm, hidden, captured, batch, early, h, desired, z, row, q, q_null, values, target
    torch.cuda.empty_cache()
    results = {}; oof_checks = []
    for name in names:
        state.update(stage='ridge', variant=name); write(out/'status.json', state)
        folder = out/name; folder.mkdir()
        metric = run_gdp(df, torch.from_numpy(features[name]), SimpleNamespace(out_dir=folder))
        results[name] = metric; write(folder/'metrics.json', metric); write(out/'results.json', results)
        for prop in ALL_COLS:
            saved = np.load(folder/prop/'legacy_oof.npz')
            old_path = original/'native_block2'/prop/'legacy_oof.npz'
            hashes[str(old_path)] = sha(old_path)
            old = np.load(old_path)
            valid = df[prop].notna().to_numpy()
            np.testing.assert_array_equal(saved['row_index'], np.flatnonzero(valid))
            np.testing.assert_array_equal(saved['fold'], df.loc[valid, FOLD_COL])
            for key in ['row_index', 'fold', 'y_true', 'y_transformed']:
                np.testing.assert_array_equal(saved[key], old[key])
            scores = [float(spearmanr(saved['y_transformed'][saved['fold']==k], saved['prediction'][saved['fold']==k]).statistic) for k in range(5)]
            assert abs(np.mean(scores)-metric['properties'][prop]['legacy_reference']['cv_mean_spearman']) < 1e-12
            oof_checks.append(dict(variant=name, property=prop, fold_scores=scores))
        print(name, {p:v['legacy_reference']['cv_mean_spearman'] for p,v in metric['properties'].items()}, flush=True)
    for path,digest in hashes.items():
        assert sha(Path(path)) == digest, path
    write(out/'verification.json', dict(status='passed', source_sha256=hashes, oof_checks=oof_checks,
        original_block2_and_final_pools_bitwise_equal=True, actual_head_and_gamma_match=True,
        observed_token_head_norm_checks_passed=True, student_trained=False, generation_tested=False))
    state.update(status='success', stage='complete', finished_utc=datetime.now(timezone.utc).isoformat())
    write(out/'status.json', state)


if __name__ == '__main__':
    main()
