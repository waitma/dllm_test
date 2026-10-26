"""Prepare OAS-only token targets with fixed pooled semantics and head/norm constraints.

Activate pllm then protenix_abtcr. From repository root run:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 USE_TF=0 python -B scripts/research/prepare_feasible_final_targets.py --out ABS_NEW_DIR
Reuses OAS-fitted complementary512 nullspace targets. No GDP rows or labels are
opened. Saves float32 ragged token targets, original input coordinates, and exact
float64 constraint diagnostics. This preparation does not train a student.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import sha, write, key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', stage='prepare', pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'status.json', state)
    import numpy as np
    import torch
    torch.set_num_threads(4)
    d = ROOT/'output/diagnostics'
    data = d/'final_layer_preservation_pilot_20261018'
    sources = {}
    def pin(path):
        sources[str(path)] = sha(path); return path
    def array(path):
        return np.load(pin(path))
    groups = {s:[json.loads(line) for line in pin(data/f'{s}.jsonl').read_text().splitlines()] for s in ['train', 'screen']}
    manifest = json.loads(pin(data/'data.json').read_text())
    for split, rows in groups.items():
        assert sha(data/f'{split}.jsonl') == manifest['files'][split]['sha256']
        assert len(rows) == {'train':4096, 'screen':128}[split]
    assert not set(map(key, groups['train'])) & set(map(key, groups['screen']))
    base_cal = array(d/'final_layer_coordinate_pilot_20261018/calibration.npz')
    null_cal = array(d/'head_nullspace_targets_20261018/calibration.npz')
    mix_cal = array(d/'global_teacher_targets_20261018/early256_global256/calibration.npz')
    early_cal = array(d/'multilayer_teacher_targets_20261018/oas_fitted_targets.npz')
    global_cal = array(d/'global_teacher_targets_20261018/global_pca.npz')
    global_train = array(d/'global_teacher_targets_20261018/all_global_oas.npy')
    mixed_train = array(d/'global_teacher_targets_20261018/early256_global256/oas_targets.npy')
    null_train = array(d/'head_nullspace_targets_20261018/null_mixed512_parent_rowspace/oas_targets.npy')
    screen_teacher = array(d/'final_layer_complementary_teacher_pilot_20261018/screen_teacher_features.npz')
    for directory in ['head_nullspace_targets_20261018', 'global_teacher_targets_20261018']:
        assert json.loads(pin(d/directory/'verification.json').read_text())['status'] == 'passed'
    def semantic(early, global_):
        e = (early-early_cal['early_mean'])@early_cal['early_axes'].T
        g = (global_-global_cal['mean'])@global_cal['axes'].T
        return np.concatenate([e[:,:256]/mix_cal['early_scale'], g[:,:256]/mix_cal['global_scale']], axis=1)
    coordinates = dict(train=semantic(base_cal['teacher'], global_train),
                       screen=semantic(screen_teacher['early'], screen_teacher['all_global']))
    np.testing.assert_allclose(mix_cal['mean']+coordinates['train']@mix_cal['rotation']*mix_cal['scale'], mixed_train, atol=1e-9, rtol=1e-9)
    from downstream.fixed_chain_eval import load, FixedChainBackbone
    from downstream.fixed_chain_layer_eval import block_hidden
    checkpoint = ROOT/'output/downstream_generation/fixed_chain_75m_step50000_20261013/checkpoint'
    model, identity = load(checkpoint)
    from dllm.pipelines.immune_llada.data.records import BioSeqRecord
    backbone = FixedChainBackbone(model, identity, batch_size=16)
    head = model.backbone.transformer.ff_out
    norm = model.backbone.transformer.ln_f
    assert getattr(head, 'bias', None) is None and getattr(norm, 'bias', None) is None
    w = head.weight.detach().double(); gamma = norm.weight.detach().double()
    np.testing.assert_array_equal(w.cpu().numpy(), null_cal['head'])
    np.testing.assert_array_equal(gamma.cpu().numpy(), null_cal['gamma'])
    _, singular, vh = torch.linalg.svd(w*gamma[None,:], full_matrices=False)
    assert int((singular>max(w.shape)*torch.finfo(torch.float64).eps*singular[0]).sum()) == 50
    basis = vh.T.contiguous()
    row_unweighted = torch.tensor(null_cal['row_basis'], device='cuda', dtype=torch.float64)
    rotation = torch.tensor(null_cal['rotation'], device='cuda', dtype=torch.float64)
    mean = torch.tensor(null_cal['mean'], device='cuda', dtype=torch.float64)
    scale = float(null_cal['null_scale'])
    pin(Path(__file__)); pin(ROOT/'downstream/fixed_chain_eval.py'); pin(ROOT/'downstream/fixed_chain_layer_eval.py')
    write(out/'plan.json', dict(created_utc=state['started_utc'], parent_identity=identity,
        data_manifest=str(data/'data.json'), source_sha256=sources, batch_size=16, attention='native',
        semantic_target='OAS-only early256/global256 coordinates, existing fitted rotation into complete head nullspace; original raw-head row component',
        precision='BF16 original forward, FP32 baseline pool, float64 constraint solve; stored token targets float32',
        initialization='original per-token null vector; common mean residual correction followed by per-token radius projection',
        max_iterations=1000, mean_coordinate_tolerance=1e-8,
        selection='all4096 train and128 heldout screen; no target selection or fitting on screen; no GDP read',
        limitation='Clean observed AB only. Does not establish learnability, masked decoding, or TCR preservation. No student checkpoint.'))
    np.savez(out/'calibration.npz', head=w.cpu().numpy(), gamma=gamma.cpu().numpy(), row_basis=basis.cpu().numpy())
    all_checks = {}; artifacts = {}
    for split, rows in groups.items():
        folder = out/split; folder.mkdir()
        lengths_all = [sum(map(len, row['chains'])) for row in rows]
        offsets = np.concatenate([[0], np.cumsum(lengths_all)]).astype(np.int64)
        target_file = folder/'token_targets.npy'
        target_map = np.lib.format.open_memmap(target_file, mode='w+', dtype=np.float32, shape=(int(offsets[-1]),768))
        positions = np.empty(int(offsets[-1]), dtype=np.int16)
        token_ids = np.empty(int(offsets[-1]), dtype=np.int16)
        pools = {name:[] for name in ['original', 'early', 'desired', 'realized_float64', 'stored_float32']}
        checks = []
        for start in range(0, len(rows), 16):
            records = [BioSeqRecord.from_prepared_dict(row) for row in rows[start:start+16]]
            batch = backbone.collator(records)
            batch = {k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
            mask = batch['residue_mask'] & batch['attention_mask']
            lengths = mask.sum(1).tolist()
            assert lengths == lengths_all[start:start+16]
            captured = []
            hook = model.backbone.transformer.blocks[1].register_forward_hook(lambda mod, inp, output:captured.append(block_hidden(output).detach()))
            try:
                hidden, _ = backbone._final_hidden(batch)
            finally:
                hook.remove()
            def pool32(x):
                return (x.float().masked_fill(~mask[...,None],0).sum(1)/mask.sum(1,keepdim=True)).cpu().numpy()
            parent_pool = pool32(hidden); early_pool = pool32(captured[0])
            np.testing.assert_array_equal(early_pool, (base_cal['teacher'] if split=='train' else screen_teacher['early'])[start:start+16])
            if split == 'train':
                np.testing.assert_array_equal(parent_pool, base_cal['student'][start:start+16])
            h = hidden[mask].double(); z = h/gamma
            row = (z@basis)@basis.T; y = z-row
            radius = torch.linalg.vector_norm(y, dim=1, keepdim=True)
            sample = torch.repeat_interleave(torch.arange(len(lengths), device='cuda'), torch.tensor(lengths, device='cuda'))
            counts = torch.tensor(lengths, device='cuda', dtype=torch.float64)[:,None]
            def mean_tokens(value):
                return torch.zeros((len(lengths), value.shape[1]), device='cuda', dtype=torch.float64).index_add_(0,sample,value)/counts
            parent_mean = mean_tokens(h)
            semantic_batch = torch.tensor(coordinates[split][start:start+16], device='cuda', dtype=torch.float64)
            desired = mean+((parent_mean-mean)@row_unweighted)@row_unweighted.T+semantic_batch@rotation*scale
            if split == 'train':
                # Stored old targets used FP32 pooled parent; current constraints
                # use the exact float64 token mean. Quantify that numerical change.
                np.testing.assert_allclose(desired.cpu().numpy(), null_train[start:start+16], atol=2e-6, rtol=1e-6)
            desired_z = desired/gamma
            mean_row = mean_tokens(row)
            torch.testing.assert_close((desired_z@basis)@basis.T, mean_row, atol=1e-10, rtol=1e-10)
            desired_null = desired_z-mean_row
            norm_desired = torch.linalg.vector_norm(desired_null, dim=1)
            upper = mean_tokens(radius).squeeze(1)
            lower = torch.stack([(2*x.max()-x.sum()).clamp_min(0)/len(x) for x in radius.split(lengths)])
            assert bool(((norm_desired>lower+1e-8)&(norm_desired<upper-1e-8)).all()), 'Infeasible pooled null norm'
            for iteration in range(1000):
                residual = desired_null-mean_tokens(y)
                if float(residual.abs().max())<1e-8:
                    break
                shifted = y+residual[sample]
                shifted_norm = torch.linalg.vector_norm(shifted, dim=1, keepdim=True)
                assert shifted_norm.min()>1e-12
                y = shifted*(radius/shifted_norm)
            residual_max = float((desired_null-mean_tokens(y)).abs().max())
            assert residual_max<1e-8, (split,start,residual_max)
            target = gamma*(row+y)
            target32 = target.float()
            torch.testing.assert_close(target@w.T,h@w.T,atol=1e-9,rtol=1e-10)
            torch.testing.assert_close(torch.linalg.vector_norm(target/gamma,dim=1),torch.linalg.vector_norm(z,dim=1),atol=1e-9,rtol=1e-10)
            torch.testing.assert_close(mean_tokens(target),desired,atol=2e-8,rtol=1e-8)
            lo,hi = int(offsets[start]),int(offsets[start+len(lengths)])
            target_map[lo:hi] = target32.cpu().numpy()
            positions[lo:hi] = mask.nonzero()[:,1].cpu().numpy()
            token_ids[lo:hi] = batch['input_ids'][mask].cpu().numpy()
            pools['original'].append(parent_pool); pools['early'].append(early_pool)
            pools['desired'].append(desired.cpu().numpy())
            pools['realized_float64'].append(mean_tokens(target).cpu().numpy())
            pools['stored_float32'].append(mean_tokens(target32.double()).cpu().numpy())
            checks.append(dict(start=start, rows=len(lengths), tokens=len(h), iterations=iteration+1,
                mean_coordinate_error=residual_max, null_radius_bound_max_ratio=float((norm_desired/upper).max()),
                float64_head_error=float((target@w.T-h@w.T).abs().max()),
                float64_norm_error=float((torch.linalg.vector_norm(target/gamma,dim=1)-torch.linalg.vector_norm(z,dim=1)).abs().max()),
                float32_head_error=float((target32.double()@w.T-h@w.T).abs().max()),
                float32_norm_error=float((torch.linalg.vector_norm(target32.double()/gamma,dim=1)-torch.linalg.vector_norm(z,dim=1)).abs().max())))
            state.update(stage='construct', split=split, rows=start+len(lengths)); write(out/'status.json',state)
            if start%256==0: print(split,start+len(lengths), 'max_iterations',max(x['iterations'] for x in checks),flush=True)
        target_map.flush(); del target_map
        np.savez(folder/'coordinates.npz', offsets=offsets, positions=positions, token_ids=token_ids)
        np.savez(folder/'pooled_features.npz', **{k:np.concatenate(v) for k,v in pools.items()})
        write(folder/'checks.json', checks)
        # Validate persisted array shape, dtype and every token; checks above use
        # exactly the float32 array written, and the final file hash binds it.
        persisted = np.load(target_file, mmap_mode='r')
        assert persisted.shape == (offsets[-1],768) and persisted.dtype == np.float32
        assert np.isfinite(persisted).all()
        all_checks[split] = dict(rows=len(rows), tokens=int(offsets[-1]), max_iterations=max(x['iterations'] for x in checks),
            max_mean_error=max(x['mean_coordinate_error'] for x in checks),
            max_float32_head_error=max(x['float32_head_error'] for x in checks),
            max_float32_norm_error=max(x['float32_norm_error'] for x in checks))
        artifacts[split] = {str(path):sha(path) for path in [target_file,folder/'coordinates.npz',folder/'pooled_features.npz',folder/'checks.json']}
    for path,digest in sources.items(): assert sha(Path(path))==digest,path
    write(out/'verification.json',dict(status='passed',parent_identity=identity,source_sha256=sources,
        artifact_sha256=artifacts, splits=all_checks, pooled_target_reconstructed=True,
        original_train_early_final_and_screen_early_bitwise=True, all_token_constraints_passed_float64=True,
        stored_targets_dtype='float32; measured approximation errors reported, not exact constraints',
        train_screen_overlap=False, gdp_read=False, student_trained=False))
    state.update(status='success',stage='complete',finished_utc=datetime.now(timezone.utc).isoformat());write(out/'status.json',state)
    print(json.dumps(all_checks,indent=2),flush=True)


if __name__=='__main__':
    main()
