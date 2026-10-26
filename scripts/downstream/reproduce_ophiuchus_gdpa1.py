"""Fresh CPU reproduction using AirGen model, pooling, loader and Ridge source.

Activate pllm then protenix_abtcr; python -B scripts/downstream/reproduce_ophiuchus_gdpa1.py
--out ABS_NEW_DIRECTORY. Runs all 246 pairs from official weights, without reading
historical embeddings. Frozen source copies and source hashes accompany all outputs.
Only checkpoint/path loading is adapted; original model/forward/pooling/CV execute unchanged.
"""
import argparse
import ast
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
from pathlib import Path
import shutil
import sys
import time
import types
import warnings

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(4*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    air = ROOT.parent/'airgen/AirGen-Dev'
    source = air/'downstream/dev/finetune_dev.py'
    mint = air/'src/byprot/models/lm/modules/mint'
    ckpt = ROOT/'model_weights/ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt'
    csv = ROOT/'data/downstream/dev/GDPa1_v1.2_20250814.csv'
    tokenizer = ROOT/'model_weights/esm2/esm2_t33_650M_UR50D'
    snap = out/'source'
    snap.mkdir()
    files = [source, *sorted(mint.rglob('*.py')), *sorted(tokenizer.glob('*.json')), tokenizer/'vocab.txt', Path(__file__)]
    hashes = {str(f): sha(f) for f in files}
    shutil.copy2(source, snap/'finetune_dev.py')
    shutil.copytree(mint, snap/'mint', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(__file__, snap/'runner.py')
    config = json.loads((tokenizer/'config.json').read_text())
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), checkpoint=str(ckpt),
                checkpoint_sha256=sha(ckpt), data=str(csv), data_sha256=sha(csv),
                source_sha256=hashes, device='cpu', threads=16, batch_size=8, dtype='float32',
                fresh_embeddings=True, use_multimer=True, sep_chains=True,
                layer=33, pooling='AirGen FlabWrapper.forward unchanged',
                properties=['AC-SINS_pH7.4','HIC','PR_CHO','Titer','Tm2'],
                protocol='AirGen fixed folds / all-label PowerTransformer / Ridge GridSearchCV',
                adaptations=['local tokenizer/config/checkpoint paths',
                             'namespace-isolated original mint imports',
                             'strict state_dict loading; frozen eval model',
                             'encode each unique source row once; filter property missing labels afterwards',
                             'CPU batch8 instead of original batch64; no autocast'])
    write(out/'plan.json', plan)
    started = time.time()
    write(out/'status.json', dict(status='running', phase='load', started_unix=started))
    import numpy as np
    import pandas as pd
    import scipy.stats
    import sklearn
    import torch
    from torch import nn
    from torch.utils.data import Dataset
    from transformers import EsmTokenizer
    from sklearn.linear_model import Ridge
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import GridSearchCV, PredefinedSplit, cross_val_predict
    from sklearn.preprocessing import PowerTransformer
    import joblib
    torch.set_num_threads(plan['threads'])
    pkg = types.ModuleType('gdpa1_original_mint')
    pkg.__path__ = [str(snap/'mint')]
    sys.modules[pkg.__name__] = pkg
    ESM2 = importlib.import_module(pkg.__name__+'.model.esm2').ESM2
    searches = []
    def capture_search(*a, **kw):
        search = GridSearchCV(*a, **kw)
        searches.append(search)
        return search
    names = {'DevDataset','MyCollateFn','FlabWrapper','gaussian_transform','spearman_corr','cross_validate'}
    nodes = [n for n in ast.parse((snap/'finetune_dev.py').read_text()).body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == names
    env = dict(np=np, pd=pd, scipy=scipy, torch=torch, nn=nn, Dataset=Dataset,
               EsmTokenizer=EsmTokenizer, Ridge=Ridge, GridSearchCV=capture_search,
               PredefinedSplit=PredefinedSplit, make_scorer=make_scorer, PowerTransformer=PowerTransformer)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(snap/'finetune_dev.py'), 'exec'), env)
    model = ESM2(num_layers=config['num_hidden_layers'], embed_dim=config['hidden_size'],
                 attention_heads=config['num_attention_heads'], token_dropout=config['token_dropout'],
                 use_multimer=True)
    state = torch.load(ckpt, map_location='cpu', weights_only=False)['state_dict']
    expected_keys = set(model.state_dict())
    matching = []
    for prefix in ('model.net.model.', 'model.', ''):
        if all(k.startswith(prefix) for k in state):
            candidate = {k[len(prefix):]: v for k,v in state.items()}
            if set(candidate) == expected_keys:
                matching.append((prefix, candidate))
    if len(matching) != 1:
        write(out/'checkpoint_key_diagnostic.json', dict(keys=list(state), expected=sorted(expected_keys)))
        raise ValueError('Checkpoint keys do not have one exact prefix-only mapping')
    prefix, converted = matching[0]
    incompatible = model.load_state_dict(converted, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    write(out/'checkpoint_load.json', dict(prefix_removed=prefix, strict=True,
                                          loaded_keys=len(converted), missing=[], unexpected=[]))
    del state, converted, matching, candidate
    wrapper = env['FlabWrapper'].__new__(env['FlabWrapper'])
    nn.Module.__init__(wrapper)
    wrapper.model, wrapper.sep_chains = model, True
    wrapper.eval().requires_grad_(False)
    assert not any(p.requires_grad for p in wrapper.parameters())
    collator = env['MyCollateFn'](tokenizer_path=str(tokenizer))
    df = pd.read_csv(csv)
    hs = df.vh_protein_sequence.str.replace('J','L',regex=False).tolist()
    ls = df.vl_protein_sequence.str.replace('J','L',regex=False).tolist()
    chunks, timings = [], []
    for start in range(0, len(df), plan['batch_size']):
        end = min(start+plan['batch_size'], len(df))
        records = [(h,l,0.0,0) for h,l in zip(hs[start:end],ls[start:end])]
        chains, chain_ids, _, _ = collator(records)
        assert chains.shape == (end-start,278)
        for row,(h,l) in enumerate(zip(hs[start:end],ls[start:end])):
            for seq,offset,width in [(h,0,150),(l,150,128)]:
                expected = [model.cls_idx]+[model.alphabet.get_idx(c) for c in seq]+[model.eos_idx]
                assert len(expected)<=width
                expected += [model.eos_idx]*(width-len(expected))
                assert chains[row,offset:offset+width].tolist()==expected
        t = time.time()
        with torch.inference_mode():
            vector = wrapper(chains,chain_ids).cpu()
        assert vector.shape == (end-start,2560) and torch.isfinite(vector).all()
        chunks.append(vector)
        timings.append(dict(start=start,end=end,seconds=time.time()-t))
        if start == 0:
            torch.save(dict(input_ids=chains,chain_ids=chain_ids,features=vector),out/'first_batch.pt')
        write(out/'status.json', dict(status='running',phase='features',completed=end,total=len(df),
                                     elapsed_seconds=time.time()-started,last_batch_seconds=timings[-1]['seconds']))
        print(f'Fresh official features: {end}/{len(df)}, batch {timings[-1]["seconds"]:.2f}s',flush=True)
    x = torch.cat(chunks).numpy()
    np.savez_compressed(out/'features.npz',features=x,row_index=np.arange(len(df)))
    paper={'AC-SINS_pH7.4':0.511,'HIC':0.550,'PR_CHO':0.460,'Titer':0.359,'Tm2':None}
    properties={}
    for prop in plan['properties']:
        dataset=env['DevDataset'](str(csv),prop)
        indices=dataset.data.index.to_numpy()
        y=np.asarray(dataset.target); folds=np.asarray(dataset.fold)
        assert dataset.heavy==[hs[i] for i in indices] and dataset.light==[ls[i] for i in indices]
        with warnings.catch_warnings(record=True) as ws, redirect_stdout(io.StringIO()):
            warnings.simplefilter('always')
            env['cross_validate'](x[indices],y,folds,scale_all=True)
        search=searches[-1]
        transformer=PowerTransformer().fit(y[:,None])
        yt=transformer.transform(y[:,None]).ravel()
        pred=cross_val_predict(search.best_estimator_,x[indices],yt[:,None],cv=PredefinedSplit(folds)).ravel()
        fold_scores=[float(scipy.stats.spearmanr(yt[folds==k],pred[folds==k]).statistic) for k in range(5)]
        assert abs(float(np.mean(fold_scores))-float(search.best_score_))<1e-10
        folder=out/prop;folder.mkdir()
        np.savez_compressed(folder/'oof.npz',row_index=indices,fold=folds,y_true=y,y_transformed=yt,prediction=pred)
        joblib.dump(dict(ridge=search.best_estimator_,transformer_all_labels=transformer),folder/'ridge.joblib')
        properties[prop]=dict(n=len(y),cv_mean_spearman=float(search.best_score_),alpha=float(search.best_params_['alpha']),
                              fold_scores=fold_scores,paper=paper[prop],warning_count=len(ws),
                              warnings=sorted(set(str(w.message) for w in ws)))
        print(f'{prop}: {search.best_score_:.9f}, alpha={search.best_params_["alpha"]}',flush=True)
    assert sha(csv)==plan['data_sha256'] and sha(ckpt)==plan['checkpoint_sha256']
    assert all(sha(f)==digest for f,digest in hashes.items())
    result=dict(status='passed',protocol=plan['protocol'],device='cpu',precision='float32',
                plan_sha256=sha(out/'plan.json'),features_shape=list(x.shape),
                checkpoint_strict_load=True,all_input_tokens_verified=True,fresh_embeddings=True,
                properties=properties,timings=timings,elapsed_seconds=time.time()-started,
                versions=dict(torch=torch.__version__,numpy=np.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__))
    write(out/'metrics.json',result)
    write(out/'status.json',dict(status='success',elapsed_seconds=result['elapsed_seconds']))
    print(f'COMPLETE: {out}',flush=True)


if __name__ == '__main__':
    main()
