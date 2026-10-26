"""Small, read-only full-matrix teacher/student diagnostic; never updates weights.

Activate pllm. Run this absolute file with prepare or student; run teacher
with the isolated stage1_esmfold2 Python. Every command accepts --plan ABS_JSON
and --output ABS_DIRECTORY. prepare fixes six validation examples and three
existing anti-lysozyme Fab references before any predictions are inspected.
Use plot_stage1_cross_chain.py separately to compute metrics and figures.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path('/project_c20250601/251105016/project/dllm_test')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def announce(phase, **kwargs):
    print(json.dumps(dict(phase=phase, **kwargs), ensure_ascii=False), flush=True)


def structure_reference(pdb_id, sequences, output):
    import requests
    from Bio import Align
    from Bio.PDB import PDBParser
    from Bio.Data.PDBData import protein_letters_3to1
    directory = output/'structures'
    directory.mkdir(exist_ok=True)
    path = directory/(pdb_id+'.pdb')
    url = f'https://files.rcsb.org/download/{pdb_id.upper()}.pdb'
    if not path.exists():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        path.write_text(response.text)
    model = PDBParser(QUIET=True).get_structure(pdb_id, path)[0]
    aligner = Align.PairwiseAligner(mode='local', match_score=2, mismatch_score=-4,
                                   open_gap_score=-5, extend_gap_score=-.5)
    options = []
    for query in sequences:
        candidates = []
        for chain in model:
            residues = [r for r in chain if r.resname in protein_letters_3to1 and r.id[0]==' ']
            observed = ''.join(protein_letters_3to1[r.resname] for r in residues)
            if not observed:
                continue
            alignment = aligner.align(query, observed)[0]
            coords = np.full((len(query),3), np.nan, dtype=np.float32)
            mapping = []
            matches = aligned = 0
            for qi, oi in alignment.indices.T:
                if qi < 0 or oi < 0:
                    continue
                aligned += 1
                if query[qi] != observed[oi]:
                    continue
                matches += 1
                residue = residues[oi]
                atom = 'CA' if query[qi]=='G' else 'CB'
                if atom in residue:
                    coords[qi] = residue[atom].coord
                    mapping.append([int(qi), str(residue.id)])
            if matches/len(query)>=.95 and matches/max(1,aligned)==1:
                candidates.append(dict(chain=chain.id, coverage=matches/len(query),
                                       coords=coords, mapping=mapping))
        if not candidates:
            raise ValueError(f'No exact high-coverage coordinate mapping for {pdb_id}')
        options.append(candidates)
    pairs = []
    for h in options[0]:
        for l in options[1]:
            if h['chain']==l['chain']:
                continue
            distances=np.linalg.norm(h['coords'][:,None]-l['coords'][None,:],axis=-1)
            pairs.append((int(np.sum(distances<8)),h['chain'],l['chain'],h,l))
    # Identical sequence copies may occur in the asymmetric unit. Choose their
    # physical Fab pairing by observed contacts, independently of predictions.
    pairs.sort(key=lambda x:(-x[0],x[1],x[2]))
    contacts, _, _, h, l = pairs[0]
    if contacts < 1:
        raise ValueError('Reference H/L chains do not form a contact interface')
    coords=np.concatenate([h['coords'],l['coords']])
    np.savez_compressed(directory/(pdb_id+'_coordinates.npz'), coordinates=coords,
                        distances=np.linalg.norm(coords[:,None]-coords[None,:],axis=-1))
    return dict(pdb_id=pdb_id,url=url,path=str(path),sha256=sha(path),model_index=0,
        representative_atom='CB; CA for glycine; missing atoms masked',
        heavy_chain=h['chain'],light_chain=l['chain'],heavy_coverage=h['coverage'],
        light_coverage=l['coverage'],heavy_mapping=h['mapping'],light_mapping=l['mapping'],
        inter_contacts_lt8=contacts,
        candidate_pairs=[dict(heavy=x[1],light=x[2],contacts_lt8=x[0]) for x in pairs],
        copy_selection='max observed contacts among exact sequence copies, before teacher inference',
        interpretation='illustrative existing benchmark structures; no unseen-training claim')


def prepare(args, plan):
    import sqlite3
    import requests
    from scripts.research.stage1_ab500k_cache import decode_record
    from scripts.research.stage1_data import make_teacher_request
    from dllm.pipelines.immune_llada.data.records import BioSeqRecord, BioSeqChain
    from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
    from safetensors.torch import save_file
    if (args.output/'examples.json').exists():
        raise ValueError('Completed example manifest already exists; reuse it for inference')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'sparse_original').mkdir(exist_ok=True)
    cm=json.loads(Path(plan['cache_manifest']).read_text())
    con=sqlite3.connect(Path(cm['database']).as_uri()+'?mode=ro&immutable=1',uri=True)
    rows=con.execute("SELECT row_index,shard,offset,length FROM targets WHERE split='valid' ORDER BY row_index").fetchall()
    selected=sorted(np.random.default_rng(20261024).choice(len(rows),6,replace=False).tolist())
    examples=[]
    for number, ordinal in enumerate(selected,1):
        row,path,offset,length=rows[ordinal]
        with open(path,'rb') as f:
            f.seek(offset); cache=decode_record(f.read(length))
        key=f'valid_{number:02d}'
        save_file(cache['tensors'],str(args.output/'sparse_original'/(key+'.safetensors')))
        examples.append(dict(id=key,group='oas_valid',row_index=row,ordinal=ordinal,
            request=cache['request'],teacher_seed=cache['metadata']['seed'],
            lengths=[len(c['sequence']) for c in cache['request']['chains']]))
    con.close()
    config=FixedChainConfig(**examples[0]['request']['config'])
    wanted={'1jhl','1mlc','1vfb'}
    reference_rows={}
    for path in sorted((ROOT/'data/downstream/cdr_infilling/sabdab/cdrh3').glob('fold_*/test.json')):
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if row['pdb_id'] in wanted:
                reference_rows.setdefault(row['pdb_id'],(row,path))
    if set(reference_rows)!=wanted:
        raise ValueError('Missing preselected structure reference')
    for pdb_id,(row,path) in sorted(reference_rows.items()):
        sequences=[row['heavy_chain_seq'],row['light_chain_seq']]
        reference=structure_reference(pdb_id,sequences,args.output)
        record=BioSeqRecord([BioSeqChain(s,r,metadata={'sequence_scope':'fv'})
            for s,r in zip(sequences,['antibody_heavy','antibody_light'])],
            'antibody','structure_diagnostic',split='valid',identifiers={'pdb_id':pdb_id})
        request=make_teacher_request(record,config,
            inter_evidence=[dict(chains=[0,1],evidence=f'Experimental PDB {pdb_id} mapped Fab chains')])
        examples.append(dict(id='pdb_'+pdb_id,group='structure_reference',request=request,
            teacher_seed=(42+int(request['request_sha256'][:8],16))%(2**31),lengths=list(map(len,sequences)),
            reference=reference,sequence_source=str(path),sequence_source_sha256=sha(path)))
    sources=args.output/'sources';sources.mkdir(exist_ok=True)
    remote={}
    for name in ('model','prepare_input'):
        url=f'https://raw.githubusercontent.com/Biohub/esm/main/esm/models/esmfold2/{name}.py'
        response=requests.get(url,timeout=60);response.raise_for_status()
        path=sources/(name+'.py');path.write_text(response.text)
        remote[name]=dict(url=url,sha256=sha(path))
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),plan=str(args.plan),
        plan_sha256=sha(args.plan),source_root=plan['source_root'],cache_manifest_sha256=sha(plan['cache_manifest']),
        seed=20261024,selection='6 uniform valid rows, seed fixed before inference; 3 predeclared anti-lysozyme Fab references',
        examples=examples,official_source_snapshots=remote,
        bins=dict(count=64,reference_range_angstrom=[2,22],
            short_distance_score='sum softmax bins 0..18 (approx. 8 A under official conditioning convention)',
            limitation='Checkpoint stores class count, not physical bin edges; use bin-index score and threshold sensitivity, not calibrated exact distance.'),
        limitations=['Existing structural references may be in teacher/student historical data; illustrative, not held-out generalization.',
                    'OAS valid samples lack experimental coordinates; teacher agreement is not structural truth.',
                    'Current student was trained with intra-chain-only KD; no new training in this diagnostic.'])
    write_json(args.output/'examples.json',manifest)
    announce('prepared',examples=[dict(id=x['id'],lengths=x['lengths']) for x in examples])


def teacher(args,plan):
    import torch
    from safetensors.torch import save_file,load_file
    from scripts.research.stage1_ab500k_teacher import OfflineTeacher
    from scripts.research import stage1_teacher as helper
    torch.set_num_threads(4)
    manifest=json.loads((args.output/'examples.json').read_text())
    directory=args.output/'teacher_full';directory.mkdir(exist_ok=True)
    announce('loading_teacher',device=torch.cuda.get_device_name(0))
    teacher_model=OfflineTeacher(plan,'cuda')
    measurements=[]
    for ex in manifest['examples']:
        started=time.monotonic()
        sequences=[c['sequence'] for c in ex['request']['chains']]
        torch.manual_seed(ex['teacher_seed']);torch.cuda.manual_seed_all(ex['teacher_seed'])
        features=helper.prepare_multichain_features(sequences,'cuda')
        kwargs=helper.trunk_arguments(features)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            result=teacher_model.model(**kwargs,num_loops=plan['teacher_contract']['num_loops'])
            logits=result.distogram_logits[0].float().cpu().contiguous()
            latent=result.pair_states[0].half().cpu().contiguous()
        length=sum(ex['lengths'])
        assert logits.shape==(length,length,64) and latent.shape==(length,length,256)
        assert torch.isfinite(logits).all() and torch.isfinite(latent).all()
        symmetry=float((logits-logits.transpose(0,1)).abs().max())
        assert symmetry<1e-5
        path=directory/(ex['id']+'.safetensors')
        save_file({'distogram_logits':logits,'pair_states':latent},str(path))
        row=dict(id=ex['id'],seconds=time.monotonic()-started,symmetry_max_abs=symmetry,
            logits_shape=list(logits.shape),pair_shape=list(latent.shape),bytes=path.stat().st_size,
            sha256=sha(path),peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30)
        if ex['group']=='oas_valid':
            old=load_file(str(args.output/'sparse_original'/(ex['id']+'.safetensors')))
            pairs=old['teacher_pair_indices'];selected=logits[pairs[:,0],pairs[:,1]]
            logp=selected.log_softmax(-1);logq=old['teacher_logits'].log_softmax(-1)
            row['sparse_replay_mean_kl']=float((logp.exp()*(logp-logq)).sum(-1).mean())
            row['sparse_replay_max_abs']=float((selected-old['teacher_logits']).abs().max())
        measurements.append(row)
        write_json(directory/'manifest.json',dict(teacher_contract=teacher_model.contract,
            examples_manifest_sha256=sha(args.output/'examples.json'),records=measurements,
            scope='full matrices for diagnosis, separate from production sparse cache'))
        announce('teacher_complete',**row)
        del result,features,kwargs,logits,latent
        torch.cuda.empty_cache()


def student(args,plan):
    import torch
    from safetensors.torch import save_file
    from dllm.pipelines.immune_llada.data.records import BioSeqRecord
    from dllm.pipelines.immune_llada.fixed_chain.structure_mlm import StructureMLMModel
    from dllm.pipelines.immune_llada.fixed_chain.full_token_mlm import FullTokenMLMCollator
    from scripts.research.run_stage1_ab500k import prepare_batch
    from dllm.pipelines.immune_llada.fixed_chain.full_token_mlm import FullTokenMLMModel
    torch.set_num_threads(4)
    manifest=json.loads((args.output/'examples.json').read_text())
    directory=args.output/'student_full';directory.mkdir(exist_ok=True)
    measurements=[]
    for arm,step in [('pair_kd',0),('pair_kd',3000),('mlm',3000)]:
        checkpoint=Path(plan['output_root'])/arm/f'checkpoint-{step}'
        receipt=json.loads((checkpoint/'checkpoint_receipt.json').read_text())
        digest=sha(checkpoint/'model.safetensors')
        if receipt['files']['model.safetensors']!=digest:
            raise ValueError('Student checkpoint checksum mismatch')
        cls=StructureMLMModel if arm=='pair_kd' else FullTokenMLMModel
        model=cls.from_pretrained(checkpoint).cuda().float().eval().requires_grad_(False)
        announce('student_loaded',arm=arm,step=step)
        for ex in manifest['examples']:
            record=BioSeqRecord.from_prepared_dict(ex['request']['record'])
            ordinal=ex.get('row_index',int(ex['request']['request_sha256'][:7],16))
            noisy,_,_=prepare_batch(model,[(ordinal,record,None)],seed=41016,targets=False)
            clean=FullTokenMLMCollator(model.config)([record])
            positions=torch.tensor([p for c in ex['request']['chains'] for p in c['student_positions']],device='cuda')
            n=sum(ex['lengths']);nh=ex['lengths'][0]
            pi,pj=torch.meshgrid(positions,positions,indexing='ij')
            indices=torch.stack([torch.zeros(n*n,dtype=torch.long,device='cuda'),pi.flatten(),pj.flatten()],-1)
            tensors={};metrics=dict(id=ex['id'],arm=arm,step=step,checkpoint_sha256=digest,context=[])
            for view,raw in [('clean',clean),('masked',noisy)]:
                batch={k:v.cuda() for k,v in raw.items()}
                with torch.inference_mode():
                    logits,hidden=model.denoise(**model.feature_args(batch),output_hidden_states=True)
                    if arm=='pair_kd':
                        latent=model.pair_encoder.sparse(hidden,indices,batch['position_ids'],batch['chain_ids'],batch['role_ids'])
                        dist=model.distogram_head(latent).reshape(n,n,64)
                        tensors[view+'_logits']=dist.cpu().contiguous()
                        if view=='clean':
                            tensors['pair_states']=latent.reshape(n,n,64).cpu().contiguous()
                    if view=='masked':
                        for target_chain,(a,b) in enumerate(((0,nh),(nh,n))):
                            target_pos=positions[a:b]
                            labels=batch['labels'][0,target_pos]
                            valid=labels.ne(-100)
                            blocked={k:v.clone() for k,v in batch.items()}
                            partner=positions[nh:] if target_chain==0 else positions[:nh]
                            blocked['attention_mask'][0,partner]=0
                            isolated,_=model.denoise(**model.feature_args(blocked),output_hidden_states=False)
                            if valid.any():
                                ce=torch.nn.functional.cross_entropy(logits[0,target_pos][valid],labels[valid],reduction='sum')
                                ce_block=torch.nn.functional.cross_entropy(isolated[0,target_pos][valid],labels[valid],reduction='sum')
                                metrics['context'].append(dict(target_chain=target_chain,count=int(valid.sum()),
                                    ce_sum=float(ce),partner_hidden_ce_sum=float(ce_block),
                                    partner_benefit_nll=float((ce_block-ce)/valid.sum())))
            if tensors:
                path=directory/f'{ex["id"]}_{arm}_{step}.safetensors'
                save_file(tensors,str(path))
                metrics['file_sha256']=sha(path)
            measurements.append(metrics)
            write_json(directory/'manifest.json',dict(records=measurements,
                mask_seed=41016,precision='float32',context_intervention='mask partner residue attention keys; preserve target MLM noise and labels',
                limitation='Inference intervention can be out of distribution; sensitivity does not establish benefit or structural accuracy.'))
            announce('student_complete',id=ex['id'],arm=arm,step=step)
        del model
        torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','teacher','student'])
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output=args.output.resolve();args.plan=args.plan.resolve()
    sys.path.insert(0,str(args.output/'dependencies'))
    os.environ.setdefault('USE_TF','0')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ.setdefault('IMMUNE_SDPA_BACKEND','math')
    plan=json.loads(args.plan.read_text())
    sys.path.insert(0,plan['source_root'])
    globals()[args.command](args,plan)


if __name__=='__main__':
    main()
