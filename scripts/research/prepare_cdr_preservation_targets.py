"""Cache original-model full-CDR conditional logits on existing OAS train/screen.

Activate pllm/protenix_abtcr; run from repository root:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 USE_TF=0 python -B scripts/research/prepare_cdr_preservation_targets.py --out ABS_NEW_DIR
Caches native-coordinate full-span masks for all three heavy CDRs. This is
training preparation, not evidence that generation retention has been fixed.
"""
import argparse
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
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    status = dict(status='running', stage='prepare', pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'status.json', status)
    import numpy as np
    import torch
    torch.set_num_threads(4)
    from downstream.fixed_chain_eval import load, FixedChainBackbone
    data = ROOT/'output/diagnostics/final_layer_preservation_pilot_20261018'
    prior = ROOT/'output/diagnostics/final_layer_cdr_span_preservation_v2_20261018'
    manifest = json.loads((data/'data.json').read_text())
    assert json.loads((prior/'verification.json').read_text())['status'] == 'passed'
    groups = {}
    pinned = [Path(__file__), data/'data.json', prior/'parent_predictions.npz',
              ROOT/'downstream/fixed_chain_eval.py']
    for split in ['train', 'screen']:
        path = data/f'{split}.jsonl'
        assert sha(path) == manifest['files'][split]['sha256']
        groups[split] = [json.loads(line) for line in path.read_text().splitlines()]
        pinned.append(path)
    hashes = {str(path): sha(path) for path in pinned}
    checkpoint = ROOT/'output/downstream_generation/fixed_chain_75m_step50000_20261013/checkpoint'
    model, identity = load(checkpoint)
    backbone = FixedChainBackbone(model, identity)
    from dllm.pipelines.immune_llada.data.records import BioSeqRecord, CHAIN_ROLE_TO_ID
    regions = ['CDR1', 'CDR2', 'CDR3']
    order = ['FR1', 'CDR1', 'FR2', 'CDR2', 'FR3', 'CDR3', 'FR4']
    plan = dict(created_utc=status['started_utc'], parent_identity=identity,
        source_sha256=hashes, regions=regions, attention='native', batch_size=16,
        precision='BF16 forward; float32 saved logits', labels_used=False,
        train_rows=len(groups['train']), screen_rows=len(groups['screen']),
        scope='Full-span conditional predictions; no sampled intermediate trajectories. No downstream rows used, no model training or evaluation claim.',
        coordinates='native role_ids + real residue order, including chains beyond legacy AirGen grid capacity',
        storage='NPZ row,region,position; -1 padded positions, zero padded logits; lengths define valid cells. Only common finite vocabulary support is stored.')
    write(out/'plan.json', plan)
    prior_logits = np.load(prior/'parent_predictions.npz')
    finite_ids = None
    checks, artifacts = [], {}
    for split, rows in groups.items():
        spans = []
        for row in rows:
            hi = row['chain_roles'].index('antibody_heavy')
            reg = row['regions'][str(hi)]
            assert ''.join(reg[k] for k in order) == row['chains'][hi]
            cursor, current = 0, {}
            for key in order:
                if key in regions:
                    assert reg[key]
                    current[key] = list(range(cursor, cursor+len(reg[key])))
                cursor += len(reg[key])
            spans.append(current)
        max_len = max(len(s[r]) for s in spans for r in regions)
        shape = (len(rows), len(regions), max_len)
        positions = np.full(shape, -1, dtype=np.int32)
        gold = np.full(shape, -1, dtype=np.int16)
        lengths = np.zeros(shape[:2], dtype=np.int32)
        cached = None
        for start in range(0, len(rows), 16):
            chunk = rows[start:start+16]
            cpu = backbone.collator([BioSeqRecord.from_prepared_dict(r) for r in chunk])
            batch = {k: v.cuda() if torch.is_tensor(v) else v for k,v in cpu.items()}
            for ri, region in enumerate(regions):
                selected = torch.zeros_like(batch['input_ids'], dtype=torch.bool)
                for j, row in enumerate(chunk):
                    hi = row['chain_roles'].index('antibody_heavy')
                    observed = ((batch['role_ids'][j] == CHAIN_ROLE_TO_ID['antibody_heavy']) & batch['residue_mask'][j]).nonzero().flatten()
                    assert batch['input_ids'][j,observed].tolist() == model.tokenizer.encode_residues(row['chains'][hi])
                    target = observed[spans[start+j][region]]
                    selected[j,target] = True
                    length = len(target)
                    lengths[start+j,ri] = length
                    positions[start+j,ri,:length] = target.cpu().numpy()
                    gold[start+j,ri,:length] = batch['input_ids'][j,target].cpu().numpy()
                noised = batch['input_ids'].clone()
                noised[selected] = model.tokenizer.mask_token_id
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    logits, _ = model.denoise(**{**model.feature_args(batch), 'input_ids': noised})
                chosen = logits[selected].float()
                support = torch.isfinite(chosen)
                assert torch.equal(support, support[:1].expand_as(support))
                assert torch.isneginf(chosen[~support]).all()
                ids = support[0].nonzero().flatten().cpu().numpy()
                if finite_ids is None:
                    finite_ids = ids
                    assert len(finite_ids) == 26
                np.testing.assert_array_equal(ids, finite_ids)
                if cached is None:
                    cached = np.zeros((*shape, len(ids)), dtype=np.float32)
                active = chosen[:,ids].cpu().numpy()
                offset = 0
                for j in range(len(chunk)):
                    length = int(lengths[start+j,ri])
                    cached[start+j,ri,:length] = active[offset:offset+length]
                    offset += length
                assert offset == len(active)
                if split == 'screen':
                    prefix = region+'__span'
                    seq = prior_logits[prefix+'_sequence_index']
                    mask = (seq >= start) & (seq < start+len(chunk))
                    np.testing.assert_array_equal(chosen.cpu().numpy(), prior_logits[prefix+'_logits'][mask])
                    np.testing.assert_array_equal(batch['input_ids'][selected].cpu().numpy(), prior_logits[prefix+'_gold'][mask])
                    checks.append(dict(region=region, start=start, rows=len(chunk), bitwise_equal=True))
            status.update(stage='cache', split=split, completed=start+len(chunk), total=len(rows))
            write(out/'status.json', status)
            if start % 256 == 0:
                print(split, start+len(chunk), len(rows), flush=True)
        path = out/f'{split}_targets.npz'
        np.savez_compressed(path, logits=cached, native_positions=positions, gold_ids=gold,
                            lengths=lengths, finite_vocab_ids=finite_ids)
        reloaded = np.load(path)
        for name, value in [('logits', cached), ('native_positions', positions), ('gold_ids', gold), ('lengths', lengths)]:
            np.testing.assert_array_equal(reloaded[name], value)
        artifacts[split] = dict(path=str(path), sha256=sha(path), rows=len(rows), regions=len(regions),
                                masked_tokens=int(lengths.sum()), max_mask_length=max_len)
    for path, digest in hashes.items():
        assert sha(Path(path)) == digest, path
    write(out/'verification.json', dict(status='passed', all_source_regions_reconstructed=True,
        source_and_row_order_pinned=True, native_heavy_token_mapping_verified=True,
        finite_support_verified=True, all_screen_logits_match_previous_diagnostic=checks,
        serialized_arrays_bitwise_equal=True, artifacts=artifacts))
    status.update(status='success', stage='complete', finished_utc=datetime.now(timezone.utc).isoformat())
    write(out/'status.json', status)
    print(json.dumps(artifacts), flush=True)


if __name__ == '__main__':
    main()
