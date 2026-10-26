"""Activate pllm/protenix_abtcr; python -B SCRIPT --run ABS --model MODEL.

Independent T2B reconstruction: original published rows and author encoders,
full KMeans, explicit unresolved defaults, saved embeddings/assignments.
Models: NetTCR2.0, clusTCR, DeepRC, fixed_chain_50000. No legacy output writes.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def data(run):
    import pandas as pd
    source = run/'author_source/dataset/clustering/TCRantigenData_unique_test.csv'
    df = pd.read_csv(source)
    local = pd.read_csv(ROOT/'downstream/benchmark/data/tcr_clustering_embed/tcrs.csv')
    assert len(df) == 9033 and df.CDR3b.nunique() == 9033
    labels = df['Antigen-epitope'].astype(str)
    assert labels.nunique() == 25
    assert df.CDR3b.tolist() == local.cdr3b.tolist()
    # Verify that legacy shortening of label strings does not merge categories.
    assert labels.str.rsplit('_', n=1).str[-1].tolist() == local.epitope.tolist()
    assert local.epitope.nunique() == labels.nunique()
    return df, labels.to_numpy(), source


def features(args, df, source, folder):
    import numpy as np
    cache = folder/'embeddings.npy'
    identity = dict(model=args.model, dataset_sha256=sha(source), n_rows=len(df))
    if cache.exists():
        saved = json.loads((folder/'feature_manifest.json').read_text())
        assert all(saved[k] == v for k, v in identity.items())
        assert saved['embedding_sha256'] == sha(cache)
        return np.load(cache), saved
    author = args.run/'author_source/src/tcrembedding'
    started = time.time()
    if args.model == 'NetTCR2.0':
        m = module(author/'NetTCR2_0/embedding.py', 'paper_nettcr')
        enc = m.EmbeddingNetTCR2()
        enc.load_data(str(source))
        x = enc.embed('CDR3b')
        identity['encoder'] = 'author EmbeddingNetTCR2, default length=30, flatten'
    elif args.model == 'clusTCR':
        # Author tools.py uses an absolute profiles import; add its original folder.
        sys.path.insert(0, str(author/'clusTCR'))
        from profiles.properties import OPTIMAL
        from tools import make_profiles
        x = make_profiles(df.CDR3b, OPTIMAL, int(df.CDR3b.str.len().max()), 1)
        identity['encoder'] = 'author make_profiles, OPTIMAL properties, dynamic max length'
    elif args.model == 'DeepRC':
        m = module(author/'DeepRC/get_feature.py', 'paper_deeprc')
        x = m.GetFeatures().compute_features(df.CDR3b.tolist())
        identity['encoder'] = 'author GetFeatures defaults, onehot plus positional features, flatten'
    elif args.model == 'fixed_chain_50000':
        import torch
        torch.set_num_threads(4)
        from downstream.fixed_chain_eval import load, FixedChainBackbone
        checkpoint = ROOT/'output/downstream_generation/fixed_chain_75m_step50000_20261013/checkpoint'
        model, model_id = load(checkpoint, args.device)
        backbone = FixedChainBackbone(model, model_id)
        from downstream.grammar.tcr_features import TCRRuntimeEmbedder
        enc = TCRRuntimeEmbedder(backbone=backbone, batch_size=8)
        identity.update(checkpoint=model_id, encoder=enc.name,
                        input_protocol=enc._load_info, device=args.device,
                        precision='bf16_autocast' if args.device == 'cuda' else 'float32')
        chunks = []
        partial = folder/'chunks'; partial.mkdir(exist_ok=True)
        binding = dict(identity, chunk_size=128)
        bindfile = partial/'binding.json'
        if bindfile.exists():
            assert json.loads(bindfile.read_text()) == binding, 'Changed partial embedding identity'
        else:
            write(bindfile, binding)
        for start in range(0, len(df), 128):
            path = partial/f'{start:05d}.npy'
            if path.exists():
                chunk = np.load(path)
            else:
                chunk = enc.embed(df.CDR3b.iloc[start:start+128].tolist())
                tmp = path.with_suffix('.tmp')
                with tmp.open('wb') as f:
                    np.save(f, chunk)
                tmp.replace(path)
            assert chunk.shape == (min(128, len(df)-start), backbone.hidden)
            chunks.append(chunk)
            write(folder/'progress.json', dict(stage='embedding', rows=start+len(chunk),
                  total=len(df), elapsed_sec=time.time()-started))
            print(args.model, 'embedded', start+len(chunk), '/', len(df), flush=True)
        x = np.concatenate(chunks)
    else:
        raise ValueError(args.model)
    x = np.asarray(x).reshape(len(df), -1)
    assert np.isfinite(x).all()
    np.save(cache, x)
    identity.update(shape=list(x.shape), dtype=str(x.dtype),
                    embedding_sha256=sha(cache), elapsed_sec=time.time()-started)
    write(folder/'feature_manifest.json', identity)
    return x, identity


def scores(labels, true):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.metrics.cluster import contingency_matrix
    c = contingency_matrix(true, labels)
    return dict(ari=float(adjusted_rand_score(true, labels)),
                nmi=float(normalized_mutual_info_score(true, labels)),
                purity=float(c.max(axis=0).sum()/len(true)),
                fraction_clusters_purity_gt_09=float(np.mean(c.max(axis=0)/c.sum(axis=0) > .9)))


def score_checks():
    import numpy as np
    # Known mixed partition: purity 5/6; relabelling must not change any metric.
    truth = np.array([0, 0, 0, 1, 1, 1])
    pred = np.array([0, 0, 1, 1, 2, 2])
    a = scores(pred, truth)
    assert abs(a['purity'] - 5/6) < 1e-12
    assert a == scores(pred*7+19, truth*3+5)
    assert scores(truth, truth)['ari'] == 1
    assert scores(truth, truth)['nmi'] == 1
    # Independent pair-count ARI formula from the supplement, with ordered pairs.
    from sklearn.metrics.cluster import pair_confusion_matrix
    tn, fp, fn, tp = pair_confusion_matrix(truth, pred).ravel().astype(float)
    independent = 2*(tp*tn-fn*fp)/((tp+fn)*(fn+tn)+(tp+fp)*(fp+tn))
    assert abs(a['ari']-independent) < 1e-12
    return dict(perfect_partition=True, label_permutation=True, known_purity=True,
                pair_formula_ari=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--features-only', action='store_true')
    args = p.parse_args()
    p.error('T2B KMeans was removed from evaluation scope on 2026-10-14; historical artifacts are retained')
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from threadpoolctl import threadpool_limits
    folder = args.run/args.model
    folder.mkdir(exist_ok=True)
    def status(state, **kw):
        write(folder/'status.json', dict(status=state, utc=datetime.now(timezone.utc).isoformat(), **kw))
    try:
        status('running')
        for item in json.loads((args.run/'sources.json').read_text())['files']:
            assert sha(item['path']) == item['sha256'], item['path']
        df, true, source = data(args.run)
        checks = score_checks()
        write(folder/'checks.json', dict(metrics=checks, dataset_rows_order_labels=True))
        with threadpool_limits(limits=4):
            x, identity = features(args, df, source, folder)
            if args.features_only:
                status('features_complete'); return
            configs = [
                ('kmeans_raw_n10', False, 10, False),
                ('kmeans_raw_n1', False, 1, False),
                ('kmeans_l2_n10', True, 10, False),
                ('legacy_minibatch_l2', True, 3, True),
            ]
            rows, saved_labels = [], {}
            for arm, normalize, n_init, mini in configs:
                xx = x.astype(np.float32) if mini else x.astype(np.float64)
                if normalize:
                    xx = xx/(np.linalg.norm(xx, axis=1, keepdims=True)+1e-8)
                for k in range(10, 101, 5):
                    ctor = MiniBatchKMeans if mini else KMeans
                    kw = dict(batch_size=2048) if mini else dict(algorithm='lloyd', max_iter=300, tol=1e-4)
                    start = time.time()
                    obj = ctor(n_clusters=k, init='k-means++', n_init=n_init, random_state=0, **kw)
                    labels = obj.fit_predict(xx)
                    row = dict(arm=arm, k=k, **scores(labels, true), seconds=time.time()-start)
                    rows.append(row)
                    saved_labels[f'{arm}_k{k}'] = labels
                    pd.DataFrame(rows).to_csv(folder/'curve.csv', index=False)
                    print(args.model, arm, k, row['ari'], row['purity'], flush=True)
            np.savez_compressed(folder/'assignments.npz', **saved_labels)
            frame = pd.DataFrame(rows)
            means = frame.groupby('arm')[['ari','nmi','purity','fraction_clusters_purity_gt_09']].mean().to_dict('index')
            write(folder/'metrics.json', dict(model=args.model, means=means, feature_manifest=identity,
                  sklearn_version=sklearn.__version__, primary_arm='kmeans_raw_n10',
                  unresolved=['author scoring pipeline unavailable in pinned repo',
                              'paper does not specify n_init or global normalization'],
                  protocol_alignment='partial; reconstruction, not verified paper reproduction',
                  controls='n_init=1 and L2 are diagnostics; no best-arm selection',
                  script_sha256=sha(__file__), assignments_sha256=sha(folder/'assignments.npz')))
        status('success')
    except Exception as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
