"""Audit saved screening generations without running inference or changing metrics.

After activating pllm, run with the exploration environment:
python /absolute/path/analyze_generation_coverage.py --output-dir /absolute/path/report
Only the screening split is read; confirmation results remain untouched.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dllm.pipelines.immune_llada.data.records import BioSeqRecord

ROOT=Path('/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_v8_20261028')
REGIONS=('FR1','CDR1','FR2','CDR2','FR3','CDR3','FR4')
RECEPTORS=('antibody_heavy','antibody_light','tcr_alpha','tcr_beta','nanobody_vhh')


def run(output):
    evidence={}
    def read(path):
        raw=path.read_bytes()
        evidence[str(path)]=hashlib.sha256(raw).hexdigest()
        return raw
    index=json.loads(read(ROOT/'data/index.json'))
    reference=index['files']['screen']['task_valid']
    raw=read(Path(reference['path']))
    assert hashlib.sha256(raw).hexdigest()==reference['sha256']
    lookup={}
    for line in raw.splitlines():
        sample=json.loads(line)
        for chain in BioSeqRecord.from_prepared_dict(sample['record']).chains:
            if chain.role not in RECEPTORS or any(chain.metadata.get('synthetic_residue_mask', [])):
                continue
            key=(sample['source_group'],sample['receptor_key'],chain.role)
            value=(chain.sequence,chain.regions)
            if key in lookup:
                assert lookup[key][0]==chain.sequence, 'Conflicting receptor sequence'
            lookup[key]=value
    rows=[]; summaries=[]; predictions={}
    for variant in [f'A{i}' for i in range(8)]:
        trial=ROOT/'trials'/f'screen_{variant}_s42'
        status=json.loads(read(trial/'status.json'))
        assert status['status']=='completed' and status['step']==1276
        observations=json.loads(read(trial/'metrics/screen_001276_observations.json'))['generation']
        published=json.loads(read(trial/'metrics/screen_001276.json'))['generation']
        grouped=defaultdict(list)
        for row in observations:
            if row['kind']!='conditional':
                continue
            truth,regions=lookup[(row['source'],row['key'],row['role'])]
            assert len(truth)==row['target_tokens'], 'Reference differs from evaluated target'
            prediction=row['sequence'] or ''
            valid=bool(row['valid'])
            key=(variant,row['source'],row['key'],row['role'])
            assert key not in predictions, 'Duplicate task identity'
            predictions[key]=prediction if valid else ''
            covered=min(len(prediction),len(truth)) if valid else 0
            matches=sum(a==b for a,b in zip(truth,prediction)) if valid else 0
            aar=matches/len(truth)
            assert abs(aar-row['aar'])<1e-12, 'Published AAR does not reproduce'
            result=dict(variant=variant,source=row['source'],receptor_key=row['key'],role=row['role'],
                        valid=valid,true_length=len(truth),generated_length=len(prediction) if valid else 0,
                        reference_coverage=covered/len(truth),aar=aar,
                        covered_accuracy=matches/covered if covered else None,
                        shorter_than_half=valid and len(prediction)<.5*len(truth),
                        length_mae=abs(len(prediction)-len(truth)) if valid else None)
            if regions and ''.join(regions.get(k,'') for k in REGIONS)==truth:
                position=0; indices=defaultdict(list)
                for region in REGIONS:
                    end=position+len(regions.get(region,''))
                    indices['cdr' if region.startswith('CDR') else 'framework'].extend(range(position,end))
                    position=end
                for name,ids in indices.items():
                    result[name+'_conditional_aar']=sum(valid and i<len(prediction) and prediction[i]==truth[i] for i in ids)/len(ids) if ids else None
            rows.append(result); grouped[row['source']].append(result)
        by_source=[]
        fields=['true_length','generated_length','reference_coverage','aar','covered_accuracy','shorter_than_half','length_mae','cdr_conditional_aar','framework_conditional_aar']
        for source,values in sorted(grouped.items()):
            mean={k:statistics.mean(x[k] for x in values if x.get(k) is not None) for k in fields if any(x.get(k) is not None for x in values)}
            mean.update(variant=variant,source=source,tasks=len(values),valid_tasks=sum(x['valid'] for x in values),
                        covered_accuracy_tasks=sum(x['covered_accuracy'] is not None for x in values))
            assert abs(mean['aar']-published['sources']['conditional'][source]['aar'])<1e-12
            summaries.append(mean); by_source.append(mean)
        assert abs(statistics.mean(x['aar'] for x in by_source)-published['conditional_aar'])<1e-12
    matched=[]
    for source in sorted({r['source'] for r in summaries}):
        base_values=[]; candidate_values=[]; count=0
        for key,baseline in predictions.items():
            v,s,receptor,role=key
            if v!='A0' or s!=source:
                continue
            candidate=predictions['A5',s,receptor,role]
            truth=lookup[s,receptor,role][0]
            length=min(len(baseline),len(candidate),len(truth))
            if not length:
                continue
            base_values.append(sum(baseline[i]==truth[i] for i in range(length))/length)
            candidate_values.append(sum(candidate[i]==truth[i] for i in range(length))/length)
            count+=1
        matched.append(dict(source=source,tasks=count,A0_shared_accuracy=statistics.mean(base_values),
                            A5_shared_accuracy=statistics.mean(candidate_values)))
    output.mkdir(parents=True,exist_ok=True)
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'stage':'screen_diagnostic_only',
            'definitions':{'aar':'Position matches / full reference length; invalid predictions score zero.',
                           'reference_coverage':'min(generated length, reference length) / reference length; invalid predictions score zero.',
                           'covered_accuracy':'Position matches / covered reference positions; undefined when coverage is zero. Descriptive, not a matched-position comparison.',
                           'cdr_conditional_aar':'CDR positions in whole-chain generation, scored with exact reference FR/CDR annotations; missing residues count wrong.',
                           'length_mae':'Valid generations only; residues.',
                           'shared_accuracy':'Compare A0/A5 on the same per-task reference prefix covered by both predictions; only tasks with nonzero shared coverage.',
                           'relation':'Per task, AAR = coverage * covered accuracy when coverage is positive; means need not factorize.',
                           'limitation':'Post hoc diagnostic on selected screening results, one training seed; does not establish significance or causal mechanism.'},
            'source_file_sha256':evidence,'by_source':summaries,'A0_A5_shared_positions':matched}
    (output/'coverage_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,values in [('per_task.csv',rows),('by_source.csv',summaries),('shared_positions.csv',matched)]:
        with (output/name).open('w') as f:
            writer=csv.DictWriter(f,sorted({k for row in values for k in row}),lineterminator='\n'); writer.writeheader(); writer.writerows(values)
    selected={(r['variant'],r['source']):r for r in summaries}
    sources=sorted({r['source'] for r in summaries})
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','svg.hashsalt':'arch70m-coverage','pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(16,6))
    fig.subplots_adjust(left=.06,right=.98,top=.75,bottom=.29,wspace=.25)
    fig.text(.06,.93,'Why does whole-chain AAR improve?',fontsize=22,weight='bold',color='#243447')
    fig.text(.06,.87,'A0 baseline vs A5 all-global attention | final screening step 1276 | seed 42',fontsize=12)
    shared={r['source']:r for r in matched}
    for ax,(key,title,scale) in zip(axes,[('generated_length','Generated length (residues)',1),('reference_coverage','Reference positions covered (%)',100),('shared_accuracy','Accuracy at shared covered positions (%)',100)]):
        x=np.arange(len(sources))
        for variant,offset,color in [('A0',-.18,'#334155'),('A5',.18,'#CC3311')]:
            vals=[(shared[s][variant+'_shared_accuracy'] if key=='shared_accuracy' else selected[variant,s][key])*scale for s in sources]
            bars=ax.bar(x+offset,vals,.33,label=variant,color=color)
            ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
        if key=='generated_length':
            ax.scatter(x,[selected['A0',s]['true_length'] for s in sources],marker='D',c='#228833',s=34,label='Reference',zorder=4)
        ax.set_title(title,fontsize=12,loc='left',weight='bold')
        ax.set_xticks(x,['ASD Ab','OAS','OTS','TCR native'])
        ax.set_ylim(0,135 if key=='generated_length' else 110 if key=='reference_coverage' else 100)
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    axes[0].legend(frameon=False,loc='upper left',bbox_to_anchor=(0,-.15),ncol=3)
    fig.text(.06,.13,'Third panel: each A0/A5 pair is scored on the same reference positions covered by both outputs; later missing positions are excluded.',fontsize=10)
    fig.text(.06,.08,'AAR remains the original full-reference metric. This post hoc audit neither changes selection rules nor uses the confirmation split.',fontsize=10,color='#64748B')
    for ext in ('png','pdf','svg'):
        fig.savefig(output/f'coverage_diagnostic.{ext}',dpi=180)
    svg=output/'coverage_diagnostic.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    print(json.dumps({'output':str(output),'variants':8,'tasks_checked':len(rows),'published_aar_reproduced':True,'shared_positions':matched}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(); assert args.output_dir.is_absolute()
    run(args.output_dir)
