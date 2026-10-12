"""Model-independent interfaces for the four sequence benchmark subsets.

CPU CLI: python -m downstream.benchmark.ab_antigen.interfaces validate --task binding --data DATA.jsonl
CPU scoring: ... evaluate --task binding --data DATA.jsonl --predictions PRED.jsonl --out-dir DIR
Export: ... export --task chimera --data DATA.jsonl --out-dir DIR
Python adapters implement score_pair(PairInput) and/or generate(GenerationInput).
No labels or reference antibodies are passed to generation adapters.
"""
from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Protocol

TASKS=('binding','spr','abbibench','chimera')
AA=frozenset('ACDEFGHIKLMNPQRSTVWY')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def read_jsonl(path):
    rows=[]
    with Path(path).open() as f:
        for line_number,line in enumerate(f,1):
            if not line.strip():continue
            try:row=json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_number}: invalid or incomplete JSONL; preserve/repair before resuming') from exc
            if not isinstance(row,dict):raise ValueError(f'{path}:{line_number}: expected an object')
            rows.append(row)
    return rows


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def finite(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def unique_index(rows,key='id'):
    result={}
    for row in rows:
        value=row.get(key)
        if not isinstance(value,str) or not value:raise ValueError(f'Missing string {key}')
        if value in result:raise ValueError(f'Duplicate {key}: {value}')
        result[value]=row
    return result


def load_dataset(path,task,*,split=None,assay=None):
    if task not in TASKS:raise ValueError(f'Unknown task: {task}')
    if assay and task!='abbibench':raise ValueError('--assay applies only to AbBiBench')
    rows=read_jsonl(path);unique_index(rows)
    selected=[]
    for row in rows:
        if row.get('dataset')!=task:raise ValueError(f'{row["id"]}: dataset/task mismatch')
        for field in ['antigen','heavy','light']:
            seq=row.get(field)
            if not isinstance(seq,str) or not seq or set(seq)-AA:
                raise ValueError(f'{row["id"]}: invalid {field}')
        if task=='chimera':
            if row.get('split') not in ['train','val','test']:raise ValueError('CHIMERA needs an explicit split')
        else:
            y=finite(row.get('label'),'label')
            if task=='binding' and y not in (0,1):raise ValueError('Binding label must be 0/1')
            if task in ['binding','spr']:
                for field in ['antibody_key','lineage']:
                    if not isinstance(row.get(field),str) or not row[field]:raise ValueError(f'Missing {field}')
            if task=='spr':
                if row.get('endpoint')!='log10_kd':raise ValueError('SPR ranking expects audited log10_kd endpoint')
                for field in ['source_doi','assay']:
                    if not row.get(field):raise ValueError(f'SPR needs {field} to isolate experimental sources')
            if task=='abbibench':
                if not row.get('assay'):raise ValueError('Missing assay')
                for field in ['wt_heavy','wt_light']:
                    seq=row.get(field)
                    if not isinstance(seq,str) or not seq or set(seq)-AA:raise ValueError(f'Invalid {field}')
        if split and row.get('split')!=split:continue
        if assay and row.get('assay')!=assay:continue
        selected.append(row)
    if not selected:raise ValueError('No rows match the requested task/split/assay')
    return selected


@dataclass(frozen=True)
class PairInput:
    id: str
    antigen: str
    heavy: str
    light: str


@dataclass(frozen=True)
class GenerationInput:
    id: str
    antigen: str
    seed: int
    candidate: int
    condition: str
    antigen_key: str
    condition_antigen_key: str


class PairScorer(Protocol):
    def score_pair(self, request: PairInput) -> float:
        """Return a finite, higher-is-preferred conditional score."""
        ...


class PairGenerator(Protocol):
    def generate(self, request: GenerationInput) -> dict:
        """Return heavy/light, EOS flags, token_valid and a failure reason."""
        ...


def prediction_inputs(rows):
    return [PairInput(r['id'],r['antigen'],r['heavy'],r['light']) for r in rows]


def generation_inputs(rows,*,candidates=20,seed=42):
    if candidates<1:raise ValueError('candidates must be positive')
    targets={}
    for r in rows:
        if r.get('split')!='test':raise ValueError('Generation interface requires a frozen test split')
        # Use the canonical key assigned by the data builder and reject inconsistent aliases.
        key=r.get('antigen_key')
        if not isinstance(key,str) or not key:raise ValueError('Missing antigen_key')
        if key in targets and targets[key]!=r['antigen']:raise ValueError('Antigen key maps to multiple sequences')
        targets[key]=r['antigen']
    if len(set(targets.values()))!=len(targets):raise ValueError('Multiple keys for one antigen sequence')
    keys=sorted(targets)
    if len(keys)<2:raise ValueError('Condition ablation requires two distinct antigens')
    requests=[]
    for i,key in enumerate(keys):
        target_seed=seed+int(key[:8],16)%1000000
        for condition,condition_key in [('correct',key),('replaced',keys[(i+1)%len(keys)])]:
            for candidate in range(candidates):
                identity=fingerprint([key,condition,candidate])
                requests.append(GenerationInput(identity,targets[condition_key],target_seed+candidate,
                    candidate,condition,key,condition_key))
    return requests


class PredictionJournal:
    """Strict append-only resume, bound to dataset/config/model provenance."""
    def __init__(self,out_dir,manifest):
        self.path=Path(out_dir);self.path.mkdir(parents=True,exist_ok=True)
        manifest_path=self.path/'interface_manifest.json';self.predictions_path=self.path/'interface_predictions.jsonl'
        if manifest_path.exists():
            if json.loads(manifest_path.read_text())!=manifest:raise ValueError('Resume provenance mismatch')
        else:
            if self.predictions_path.exists():raise ValueError('Predictions without a manifest cannot be resumed')
            save_json(manifest_path,manifest)
        self.rows=unique_index(read_jsonl(self.predictions_path)) if self.predictions_path.exists() else {}

    def append(self,row):
        unique_index([row])
        if row['id'] in self.rows:raise ValueError('Refusing to overwrite an existing prediction')
        line=json.dumps(row,allow_nan=False)+'\n'
        with self.predictions_path.open('a') as f:
            f.write(line);f.flush()
        self.rows[row['id']]=row


def run_adapter(task,rows,adapter,out_dir,*,model_identity,mode='zero_shot',candidates=20,seed=42):
    if not model_identity:raise ValueError('A pinned model identity is required')
    if mode not in ['zero_shot','frozen_probe']:raise ValueError('Unknown prediction mode')
    if task in ['abbibench','chimera'] and mode!='zero_shot':raise ValueError('This track does not use a supervised probe')
    unique_index(rows)
    if task not in TASKS:raise ValueError('Unknown task')
    requests=generation_inputs(rows,candidates=candidates,seed=seed) if task=='chimera' else prediction_inputs(rows)
    manifest=dict(interface_version=1,task=task,dataset_sha256=fingerprint(rows),model_identity=model_identity,
        mode=mode,candidates=candidates if task=='chimera' else None,seed=seed,
        conditioning_protocol='antigen_sequence_only' if task=='chimera' else 'antigen_paired_vh_vl',
        score_definition='variant_minus_wt' if task=='abbibench' else 'adapter_score',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    journal=PredictionJournal(out_dir,manifest);expected={r.id for r in requests}
    if set(journal.rows)-expected:raise ValueError('Unexpected prediction IDs in resume cache')
    source=unique_index(rows);wt_cache={}
    for request in requests:
        if request.id in journal.rows:
            if task=='chimera':validate_generated(journal.rows[request.id])
            else:finite(journal.rows[request.id].get('score'),'cached score')
            continue
        if task=='chimera':
            generated=adapter.generate(request)
            if set(generated)-{'heavy','light','heavy_eos','light_eos','token_valid','failure_reason'}:
                raise ValueError('Unexpected generator output fields')
            result=dict(id=request.id,antigen_key=request.antigen_key,condition_antigen_key=request.condition_antigen_key,
                condition=request.condition,candidate=request.candidate,seed=request.seed,**generated)
            validate_generated(result)
        else:
            score=finite(adapter.score_pair(request),'score');result=dict(id=request.id,score=score)
            if task=='abbibench':
                row=source[request.id];wt_key=(request.antigen,row['wt_heavy'],row['wt_light'])
                if wt_key not in wt_cache:
                    wt_cache[wt_key]=finite(adapter.score_pair(PairInput('wt:'+fingerprint(wt_key),*wt_key)),'wt_score')
                result.update(raw_score=score,wt_score=wt_cache[wt_key],score=score-wt_cache[wt_key])
        journal.append(result)
    return [journal.rows[r.id] for r in requests]


def validate_generated(row):
    for field in ['heavy','light']:
        if not isinstance(row.get(field),str):raise ValueError(f'Generation result missing {field}')
    for field in ['heavy_eos','light_eos','token_valid']:
        if not isinstance(row.get(field),bool):raise ValueError(f'{field} must be boolean')
    actual=all(row[c] and set(row[c])<=AA and row[c+'_eos'] for c in ['heavy','light'])
    if bool(actual)!=row['token_valid']:raise ValueError('token_valid contradicts sequence/EOS evidence')
    if not actual and not row.get('failure_reason'):raise ValueError('Failed generation needs a failure_reason')


def evaluate(task,rows,predictions,*,mode='zero_shot',allow_partial=False,candidates=20,seed=42):
    if task not in TASKS:raise ValueError('Unknown task')
    if mode not in ['zero_shot','frozen_probe']:raise ValueError('Unknown mode')
    if task in ['abbibench','chimera'] and mode!='zero_shot':raise ValueError(f'{task} v1 is zero-shot only')
    reference=unique_index(rows);predicted=unique_index(predictions)
    requests=generation_inputs(rows,candidates=candidates,seed=seed) if task=='chimera' else prediction_inputs(rows)
    expected={r.id for r in requests};extra=set(predicted)-expected;missing=expected-set(predicted)
    if extra:raise ValueError(f'{len(extra)} unknown prediction IDs')
    if missing and not allow_partial:raise ValueError(f'{len(missing)} missing predictions; explicit --allow-partial required')
    report=dict(interface_version=1,task=task,mode=mode,status='partial' if missing else 'complete',
        expected=len(expected),scored=len(predicted),missing_ids=sorted(missing),
        dataset_sha256=fingerprint(rows),predictions_sha256=fingerprint(predictions),
        provenance='project_sequence_protocol_not_official_baseline_reproduction')
    if task=='chimera':
        indexed={r.id:r for r in requests}
        for key,row in predicted.items():
            validate_generated(row);req=indexed[key]
            for field in ['antigen_key','condition_antigen_key','condition','candidate','seed']:
                if row.get(field)!=getattr(req,field):raise ValueError(f'Generation request/result mismatch: {field}')
        report['conditions']={}
        for condition in ['correct','replaced']:
            selected=[r for r in predicted.values() if r['condition']==condition]
            report['conditions'][condition]=dict(attempts=len(selected),
                token_valid_rate=sum(r['token_valid'] for r in selected)/len(selected) if selected else None,
                unique_valid_pairs=len({(r['heavy'],r['light']) for r in selected if r['token_valid']}))
        report['reference_metrics_status']='use_sequence_metrics_for_numbering_recovery_diversity_novelty'
        report['interpretation']='token validity is not numbered paired validity or binding success'
        return report
    from downstream.benchmark.ab_antigen.sequence_eval import group_metrics
    joined=[]
    for key,row in reference.items():
        if key not in predicted:continue
        p=predicted[key];score=finite(p.get('score'),'score')
        if task=='binding' and mode=='frozen_probe' and not 0<=score<=1:raise ValueError('Binding probe score must be a probability')
        r={**row,'score':score}
        if task in ['binding','spr']:
            suffix='|'+r['source_doi']+'|'+r['assay'] if task=='spr' else ''
            r['group_ab']=r['antibody_key']+suffix;r['group_ag']=r['lineage']+suffix
        joined.append(r)
    if task=='abbibench':
        if mode!='zero_shot':raise ValueError('AbBiBench v1 is zero-shot only')
        report['per_assay']=group_metrics(joined,group_key='assay')
    else:
        report['fixed_antibody']=group_metrics(joined,group_key='group_ab',classification=task=='binding')
        report['fixed_antigen']=group_metrics(joined,group_key='group_ag',classification=task=='binding')
    if task=='spr':
        quantitative=[]
        for r in joined:
            prediction=predicted[r['id']]
            if r.get('pkd_status')=='verified' and r.get('pkd') is not None and prediction.get('pkd_prediction') is not None:
                quantitative.append((finite(r['pkd'],'pkd'),finite(prediction['pkd_prediction'],'pkd_prediction')))
        if mode!='frozen_probe' and quantitative:raise ValueError('pKD predictions belong to the supervised probe track')
        errors=[p-y for y,p in quantitative]
        report['pkd_regression']=dict(n=len(errors),excluded=len(joined)-len(errors),
            status='evaluated_verified_subset' if errors else 'unavailable_no_verified_quantitative_pairs',
            mae=sum(abs(e) for e in errors)/len(errors) if errors else None,
            rmse=math.sqrt(sum(e*e for e in errors)/len(errors)) if errors else None)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['validate','export','evaluate'])
    parser.add_argument('--task',choices=TASKS,required=True);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--split',choices=['train','val','test']);parser.add_argument('--assay')
    parser.add_argument('--predictions',type=Path);parser.add_argument('--out-dir',type=Path)
    parser.add_argument('--mode',choices=['zero_shot','frozen_probe'],default='zero_shot')
    parser.add_argument('--allow-partial',action='store_true');parser.add_argument('--candidates',type=int,default=20)
    parser.add_argument('--reference-metrics',action='store_true',help='CHIMERA: run ANARCI/reference recovery/diversity/novelty scoring on CPU')
    parser.add_argument('--seed',type=int,default=42);args=parser.parse_args()
    if args.reference_metrics and (args.task!='chimera' or args.command!='evaluate'):
        parser.error('--reference-metrics requires evaluate --task chimera')
    rows=load_dataset(args.data,args.task,split=args.split or ('test' if args.task=='chimera' else None),assay=args.assay)
    if args.command=='validate':
        print(json.dumps(dict(task=args.task,rows=len(rows),dataset_sha256=fingerprint(rows),status='valid_no_inference')));return
    if args.out_dir is None:parser.error('--out-dir is required')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    if args.command=='export':
        inputs=generation_inputs(rows,candidates=args.candidates,seed=args.seed) if args.task=='chimera' else prediction_inputs(rows)
        destination=args.out_dir/'inputs.jsonl'
        if destination.exists():raise FileExistsError(destination)
        with destination.open('w') as f:
            for request in inputs:f.write(json.dumps(asdict(request))+'\n')
        save_json(args.out_dir/'export_manifest.json',dict(task=args.task,dataset_sha256=fingerprint(rows),requests=len(inputs),candidates=args.candidates,seed=args.seed));return
    if args.predictions is None:parser.error('--predictions is required')
    report=evaluate(args.task,rows,read_jsonl(args.predictions),mode=args.mode,allow_partial=args.allow_partial,candidates=args.candidates,seed=args.seed)
    destination=args.out_dir/'interface_metrics.json'
    if destination.exists() and json.loads(destination.read_text())!=report:raise FileExistsError('Refusing to replace different metrics')
    save_json(destination,report)
    if args.reference_metrics:
        if args.task!='chimera':parser.error('--reference-metrics applies only to CHIMERA')
        from downstream.benchmark.ab_antigen.sequence_metrics import main as score_generation
        canonical=args.out_dir/'predictions.jsonl'
        content=''.join(json.dumps(row,allow_nan=False)+'\n' for row in read_jsonl(args.predictions))
        if canonical.exists() and canonical.read_text()!=content:raise FileExistsError('Reference-scoring input mismatch')
        temporary=canonical.with_suffix('.jsonl.tmp');temporary.write_text(content);temporary.replace(canonical)
        score_generation(args.out_dir,reference_path=args.data)


if __name__=='__main__':main()
