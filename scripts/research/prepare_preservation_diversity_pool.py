"""Prepare a nested8192 OAS pool for a later controlled diversity experiment.

Activate pllm; from repository root:
python -B scripts/research/prepare_preservation_diversity_pool.py --out ABS_NEW_DIR
No model execution or GDP reads. The old4096 records must match the exact
prefix of the expanded pool. Existing128 OAS screen records remain unchanged.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import write,sha,key


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'output/fixed_chain_exploration_20261014/data'
    old=ROOT/'output/diagnostics/final_layer_preservation_pilot_20261018'
    manifest=json.loads((source/'manifest.json').read_text())
    source_train=source/'train.jsonl'
    assert sha(source_train)==manifest['files']['train']['sha256']
    unique={}
    with source_train.open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['source']!='oas_paired' or set(row['chain_roles'])!={'antibody_heavy','antibody_light'} or len(row['chains'])!=2:continue
            pair=dict(zip(row['chain_roles'],row['chains']))
            if len(pair['antibody_heavy'])>=187 or len(pair['antibody_light'])>=154:continue
            if any(any(v.get('synthetic_residue_mask',[])) for v in row.get('chain_metadata',{}).values()):continue
            unique.setdefault(key(row),row)
    rows=list(unique.values());random.Random(20261018).shuffle(rows)
    assert len(rows)==8192
    original=[json.loads(line) for line in (old/'train.jsonl').read_text().splitlines()]
    assert len(original)==4096 and rows[:4096]==original
    prefix=''.join(json.dumps(row,separators=(',',':'))+'\n' for row in rows[:4096])
    assert prefix==(old/'train.jsonl').read_text()
    target=out/'train8192.jsonl';target.write_text(''.join(json.dumps(row,separators=(',',':'))+'\n' for row in rows))
    screen=out/'screen.jsonl';screen.write_bytes((old/'screen.jsonl').read_bytes())
    screen_rows=[json.loads(line) for line in screen.read_text().splitlines()]
    assert len(screen_rows)==128 and not set(map(key,rows)) & set(map(key,screen_rows))
    record=dict(created_utc=datetime.now(timezone.utc).isoformat(),schema='same prepared BioSeqRecord JSONL as original pilot',
                status='prepared_only_no_training_started',source_train=str(source_train),source_train_sha256=sha(source_train),
                parent_manifest_sha256=sha(source/'manifest.json'),script_sha256=sha(Path(__file__)),
                train=dict(path=str(target),records=8192,sha256=sha(target)),
                screen=dict(path=str(screen),records=128,sha256=sha(screen)),
                original4096_exact_prefix=True,original_train_sha256=sha(old/'train.jsonl'),
                additional_unique_pairs=4096,screen_unchanged=True,
                selection='original seed20261018 shuffle, extended from4096 to8192, no downstream labels read',
                intended_comparison='same checkpoint/optimizer/update budget/loss; only OAS pool diversity changes; not yet run')
    write(out/'manifest.json',record);print(json.dumps(record,indent=2))


if __name__=='__main__':main()
