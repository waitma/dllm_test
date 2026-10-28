"""Frozen screening/confirmation tasks; invoked by run.sh and the single-GPU trainer."""
from collections import defaultdict
from contextlib import nullcontext
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from dllm.pipelines.immune_llada.data.records import BioSeqRecord, CHAIN_ROLE_TO_ID
from dllm.pipelines.immune_llada.fixed_chain.v_family import VFamilyCollator
from dllm.pipelines.immune_llada.fixed_chain.model import RESIDUES
from dllm.pipelines.immune_llada.fixed_chain.generation import parse_chain
from scripts.research.bert_mlm_eval import token_validation
from .data import RECEPTORS, REGIONS
from .common import sha, write

def rows(ref):
    if sha(ref['path']) != ref['sha256']: raise ValueError('Evaluation binding changed')
    return [json.loads(x) for x in Path(ref['path']).read_text().splitlines() if x]

def amp(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()

def make_tasks(model, samples):
    collator = VFamilyCollator(model.config, family_spec=model.family_spec, hide_relation=True)
    tasks = []; exclusions = defaultdict(int)
    for sample in samples:
        record = BioSeqRecord.from_prepared_dict(sample['record'])
        if record.labels.get('relation') == 'nonbinding': exclusions['nonbinding'] += 1; continue
        clean = collator([record])
        for chain in record.chains:
            if chain.role not in RECEPTORS: continue
            if any(chain.metadata.get('synthetic_residue_mask', [])):
                exclusions['synthetic_chain'] += 1; continue
            region = clean['role_ids'][0].eq(CHAIN_ROLE_TO_ID[chain.role]) & clean['attention_mask'][0]
            aa = region & clean['residue_mask'][0]
            if not aa.any(): exclusions['empty_chain'] += 1; continue
            common = dict(source=sample['source_group'], key=sample['receptor_key'], role=chain.role,
                          clean=clean, truth=clean['input_ids'][0].clone(), true_length=int(aa.sum()))
            tasks.append(dict(common, kind='conditional', target=region, score_mask=aa))
            regions = chain.regions
            if not regions or ''.join(regions.get(k, '') for k in REGIONS) != chain.sequence:
                exclusions['no_exact_cdr_annotation'] += 1; continue
            positions = region.nonzero().flatten(); mask = torch.zeros_like(region); offset = 0
            for name in REGIONS:
                end = offset + len(regions.get(name, ''))
                if name.startswith('CDR'): mask[positions[offset:end]] = True
                offset = end
            if mask.any(): tasks.append(dict(common, kind='cdr', target=mask, score_mask=mask))
    return tasks, dict(exclusions)

def task_batch(model, tasks, device):
    length = max(t['truth'].numel() for t in tasks); features = {}
    for key in model.feature_args(tasks[0]['clean']):
        fill = model.tokenizer.pad_token_id if key == 'input_ids' else -1 if key == 'chain_ids' else 0
        features[key] = torch.cat([F.pad(t['clean'][key], (0, length-t['truth'].numel()), value=fill) for t in tasks]).to(device)
    target = torch.stack([F.pad(t['target'], (0, length-t['truth'].numel())) for t in tasks]).to(device)
    score = torch.stack([F.pad(t['score_mask'], (0, length-t['truth'].numel())) for t in tasks]).to(device)
    truth = features['input_ids'].clone(); features['input_ids'] = truth.masked_fill(target, model.tokenizer.mask_token_id)
    return features, target, score, truth

@torch.no_grad()
def generation(model, reference, device, batch_size=8):
    tasks, excluded = make_tasks(model, rows(reference)); observations = []
    legal_aa = torch.tensor(model.tokenizer.encode_residues(RESIDUES), device=device)
    legal_chain = torch.cat([legal_aa, torch.tensor([model.tokenizer.eos_token_id], device=device)])
    for start in range(0, len(tasks), batch_size):
        part = tasks[start:start+batch_size]; features, targets, scoring, truth = task_batch(model, part, device)
        with amp(device): logits, _ = model.denoise(**features)
        ce = [float(F.cross_entropy(logits[i, scoring[i]].float(), truth[i, scoring[i]])) for i in range(len(part))]
        for step in range(16):
            if step:
                with amp(device): logits, _ = model.denoise(**features)
            for i, task in enumerate(part):
                pending = (targets[i] & features['input_ids'][i].eq(model.tokenizer.mask_token_id)).nonzero().flatten()
                if not len(pending): continue
                if task['kind'] == 'conditional' and parse_chain(model.tokenizer, features['input_ids'][i, targets[i]])['status'] == 'ok': continue
                legal = legal_chain if task['kind'] == 'conditional' else legal_aa
                probabilities = logits[i, pending][:, legal].float().softmax(-1)
                confidence, prediction = probabilities.max(-1)
                remaining = int(targets[i].sum()) * (15-step) // 16
                count = max(0, len(pending)-remaining)
                chosen = confidence.topk(count).indices
                features['input_ids'][i, pending[chosen]] = legal[prediction[chosen]]
        if not torch.equal(features['input_ids'][~targets], truth[~targets]): raise ValueError('Decoder modified conditions')
        for i, task in enumerate(part):
            prediction = features['input_ids'][i]
            common = dict(source=task['source'], key=task['key'], role=task['role'], kind=task['kind'], ce=ce[i],
                          target_tokens=int(scoring[i].sum()), aar=float(prediction[scoring[i]].eq(truth[i, scoring[i]]).float().mean()))
            if task['kind'] == 'conditional':
                parsed = parse_chain(model.tokenizer, prediction[targets[i]])
                common.update(valid=int(parsed['status'] == 'ok'), status=parsed['status'], sequence=parsed.get('sequence'),
                              length_error=abs(parsed['length']-task['true_length']) if parsed['status']=='ok' else None)
                if not common['valid']: common['aar'] = 0.
                else:
                    reference=''.join(model.tokenizer.token(int(t)) for t in truth[i,scoring[i]])
                    common['aar']=sum(a==b for a,b in zip(parsed['sequence'],reference))/len(reference)
            observations.append(common)
    return summarize_generation(observations), observations, excluded

def summarize_generation(observations):
    groups = defaultdict(list)
    for x in observations: groups[x['kind'], x['source']].append(x)
    result = {'sources': {}}
    for (kind, source), group in sorted(groups.items()):
        value = dict(records=len(group), ce=float(np.mean([x['ce'] for x in group])), aar=float(np.mean([x['aar'] for x in group])))
        if kind == 'conditional':
            valid = [x['length_error'] for x in group if x['valid']]
            value.update(validity=float(np.mean([x['valid'] for x in group])), length_mae=float(np.mean(valid)) if valid else None,
                         missing_eos=sum(x['status']=='capacity_exhausted' for x in group))
        result['sources'].setdefault(kind, {})[source] = value
    for kind, groups in result['sources'].items():
        for metric in ('ce', 'aar', 'validity', 'length_mae'):
            values = [x[metric] for x in groups.values() if x.get(metric) is not None]
            if values: result[kind+'_'+metric] = float(np.mean(values))
    return result

@torch.no_grad()
def representation(model, references, device, batch_size=16):
    samples = {s: rows(references['probe_'+s]) for s in ('train', 'valid')}
    collate = VFamilyCollator(model.config, family_spec=model.family_spec, hide_relation=True)
    vectors = {}
    for split, items in samples.items():
        chunks = []
        for start in range(0, len(items), batch_size):
            batch = {k:v.to(device) for k,v in collate([x['record'] for x in items[start:start+batch_size]]).items()}
            with amp(device): chunks.append(model.receptor_vectors(batch).float().cpu().numpy())
        vectors[split] = np.concatenate(chunks) if chunks else np.empty((0, 4*model.config.d_model))
    details = []; sources = {}; exclusions = []
    for source in sorted({x['source_group'] for x in samples['valid']}):
        ids = {s: [i for i,x in enumerate(items) if x['source_group']==source] for s,items in samples.items()}
        labels = {s: np.array([int(samples[s][i]['record']['labels']['relation']=='binding') for i in ids[s]]) for s in ids}
        if any(len(set(y)) != 2 for y in labels.values()): exclusions.append(source); continue
        estimator = make_pipeline(StandardScaler(), LogisticRegression(C=.1,class_weight='balanced',max_iter=2000,random_state=42))
        estimator.fit(vectors['train'][ids['train']], labels['train'])
        probabilities = estimator.predict_proba(vectors['valid'][ids['valid']])[:,1]
        estimator.fit(vectors['train'][ids['train']], np.random.default_rng(918).permutation(labels['train']))
        null = estimator.predict_proba(vectors['valid'][ids['valid']])[:,1]
        sources[source] = dict(auroc=float(roc_auc_score(labels['valid'], probabilities)), auprc=float(average_precision_score(labels['valid'], probabilities)),
                               shuffled_auroc=float(roc_auc_score(labels['valid'], null)), train_records=len(ids['train']), valid_records=len(ids['valid']))
        details.extend(dict(key=samples['valid'][i]['receptor_key'],source=source,label=int(y),probability=float(p))
                       for i,y,p in zip(ids['valid'],labels['valid'],probabilities))
    if not sources: raise ValueError('No eligible two-class held-out probe sources')
    return dict(sources=sources, macro_auroc=float(np.mean([x['auroc'] for x in sources.values()])),
                macro_auprc=float(np.mean([x['auprc'] for x in sources.values()])), excluded_single_class_sources=exclusions), details

def evaluate(model, index, partition, device, output, step, tasks=False):
    refs = index['files'][partition]; was_training = model.training; model.eval()
    try:
        result = dict(step=step, partition=partition, token=token_validation(model, {'files':refs}, device, batch_size=8))
        if tasks:
            result['generation'], generated, result['generation_excluded'] = generation(model, refs['task_valid'], device)
            result['representation'], probes = representation(model, refs, device)
            write(Path(output) / f'{partition}_{step:06d}_observations.json', dict(generation=generated, representation=probes))
        write(Path(output) / f'{partition}_{step:06d}.json', result)
        return result
    finally: model.train(was_training)
