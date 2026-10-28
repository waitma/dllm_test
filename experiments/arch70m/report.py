"""Export complete local curves and a continuously updated lab book: run.sh --stage report."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype']='none'
matplotlib.rcParams['svg.hashsalt']='arch70m-v8'
import matplotlib.pyplot as plt
from .common import ROOT, VARIANTS, read, write, stamp, digest

def records(path):
    if not path.exists(): return []
    result=[]
    for line in path.read_text().splitlines():
        try: result.append(json.loads(line))
        except json.JSONDecodeError: break  # A live writer may not have flushed the last line yet.
    return result

def flatten(value,prefix=''):
    result={}
    for k,v in value.items():
        key=prefix+k
        if isinstance(v,dict): result.update(flatten(v,key+'.'))
        elif isinstance(v,(str,int,float,bool)) or v is None: result[key]=v
    return result

def export_csv(path,rows):
    columns=sorted({k for r in rows for k in r}) or ['trial','curve','step']
    with path.open('w') as f:
        w=csv.DictWriter(f,columns,lineterminator='\n'); w.writeheader(); w.writerows(rows)

def save_figure(fig,folder,name):
    for ext in ('png','svg','pdf'):
        path=folder/f'{name}.{ext}'; fig.savefig(path,dpi=140)
        if ext=='svg': path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')

def plot_curves(trials,folder):
    fig,axes=plt.subplots(3,3,figsize=(17,12),constrained_layout=True)
    labels=['Train selected-token CE','Validation selected-token CE','Conditional generation CE',
            'CDR infilling CE','Conditional AAR (invalid = 0)','Frozen probe macro AUROC',
            'Peak allocated GPU memory (GiB)','Train CE vs consumed tokens','Train CE vs training GPU hours']
    for name,trial in trials.items():
        train,evals=trial['train'],trial['eval']
        if train:
            steps=np.array([r['step'] for r in train]); losses=np.array([r['loss'] for r in train])
            smooth=np.array([losses[max(0,i-24):i+1].mean() for i in range(len(losses))])
            line=axes.flat[0].plot(steps,smooth,label=name,lw=1.2)[0]; color=line.get_color()
            axes.flat[0].plot(steps,losses,color=color,alpha=.12,lw=.4)
            axes.flat[6].plot(steps,[r['allocated_bytes']/2**30 for r in train],label=name,lw=1)
            axes.flat[7].plot([r['seen_tokens'] for r in train],smooth,label=name,lw=1)
            axes.flat[8].plot([r['gpu_hours'] for r in train],smooth,label=name,lw=1)
        for axis,field in ((1,'token.all.loss'),(2,'generation.conditional_ce'),(3,'generation.cdr_ce'),
                           (4,'generation.conditional_aar'),(5,'representation.macro_auroc')):
            points=[flatten(x) for x in evals if x['partition']=='screen']
            points=[x for x in points if field in x]
            if points: axes.flat[axis].plot([x['step'] for x in points],[x[field] for x in points],label=name,marker='.',lw=1)
    for i,ax in enumerate(axes.flat):
        ax.set_title(labels[i]); ax.set_xlabel('Optimizer step' if i<7 else 'Consumed nonpad tokens' if i==7 else 'Training GPU hours')
        ax.grid(alpha=.2)
    handles,names=axes.flat[0].get_legend_handles_labels()
    if handles: fig.legend(handles,names,loc='outside lower center',ncol=4,fontsize=8)
    fig.suptitle('V8 architecture exploration | raw training CE faint; trailing mean window 25 bold\nValidation and task curves are unsmoothed. Empty panels mean measurements are pending.')
    save_figure(fig,folder,'curves')
    plt.close(fig)
    # Confirmation summaries use matched optimizer steps; do not extrapolate partial seeds.
    confirmed={}
    for name,t in trials.items():
        if not name.startswith('confirm_'): continue
        variant=name.split('_')[1]; confirmed.setdefault(variant,[]).append(t)
    if not confirmed: return
    fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
    for variant,ts in confirmed.items():
        if len(ts)!=3 or any(t['status'].get('status')!='completed' for t in ts): continue
        maps=[{r['step']:r['loss'] for r in t['train']} for t in ts]; shared=sorted(set.intersection(*(set(m) for m in maps)))
        x=np.array([[m[s] for s in shared] for m in maps]); mean=x.mean(0); sd=x.std(0,ddof=1)
        ax.plot(shared,mean,label=variant); ax.fill_between(shared,mean-sd,mean+sd,alpha=.15)
    ax.set(xlabel='Optimizer step',ylabel='Raw training CE',title='Confirmation: mean ± sample SD over 3 seeds; no smoothing')
    if ax.lines: ax.legend()
    ax.grid(alpha=.2)
    save_figure(fig,folder,'confirmation_seeds')
    plt.close(fig)

def report(root):
    root=Path(root); folder=ROOT/'docs/arch70m_results'; folder.mkdir(parents=True,exist_ok=True)
    trials={}; summary=[]; curve_rows=[]
    names=[f'screen_{v}_s42' for v in VARIANTS]
    if (root/'selection.json').exists():
        choice=read(root/'selection.json'); write(folder/'selection.json',choice)
        names += [f'confirm_{v}_s{s}' for v in choice['confirm_variants'] for s in choice['seeds']]
    for name in names:
        p=root/'trials'/name; status=read(p/'status.json') if (p/'status.json').exists() else {'status':'pending','step':0}
        train=records(p/'train.jsonl'); evals=records(p/'eval.jsonl')
        trials[name]=dict(status=status,train=train,eval=evals)
        counts=read(ROOT/'experiments/arch70m/variant_parameters.json')
        row=dict(trial=name,status=status['status'],step=status.get('step',0),total_steps=status.get('total_steps'),parameters=counts[name.split('_')[1]])
        if (p/'profile.json').exists():
            profile=read(p/'profile.json'); row.update(micro_batch=profile['micro_batch'],grad_accum=profile['grad_accum'],
                                                       final_micro_batch=profile.get('final_micro_batch',profile['micro_batch']))
            write(folder/(name+'_profile.json'),profile)
        if train:
            row.update(train_loss=train[-1]['loss'],training_gpu_hours=train[-1]['gpu_hours'],seen_tokens=train[-1]['seen_tokens'],
                peak_allocated_gib=max(r['allocated_bytes'] for r in train)/2**30,
                median_step_seconds=float(np.median([r['seconds_per_step'] for r in train])),
                records_per_train_second=sum(r.get('records',512) for r in train)/max(1e-9,sum(r['seconds_per_step'] for r in train)))
        if evals: row.update({'last_eval.'+k:v for k,v in flatten(evals[-1]).items()})
        if (p/'ab_specificity/metrics.json').exists():
            row.update({'ab_specificity.'+k:v for k,v in flatten(read(p/'ab_specificity/metrics.json')).items()})
        summary.append(row)
        for split,values in (('train',train),('eval',evals)):
            for value in values: curve_rows.append(dict(trial=name,curve=split,**flatten(value)))
    write(folder/'summary.json',dict(updated=stamp(),output_root=str(root),trials=summary))
    export_csv(folder/'summary.csv',summary); export_csv(folder/'all_curves.csv',curve_rows)
    plot_curves(trials,folder)
    if (root/'selection.json').exists():
        from .statistics import confirmation
        stats=confirmation(root)
        write(folder/'confirmation.json',stats)
    text=['<!-- ARCH70M_RESULTS_BEGIN -->',f'自动记录更新时间：{stamp()}。原始日志：`{root}`。',
          '', '| Trial | Status | Step | Micro × accumulation | Last train CE | Peak allocated GiB |',
          '|---|---|---:|---|---:|---:|']
    for r in summary:
        fmt=lambda k: f'{r[k]:.4f}' if r.get(k) is not None else '—'
        batch=(f"{r['micro_batch']} × {r['grad_accum']-1} + {r['final_micro_batch']}" if 'micro_batch' in r else '—')
        text.append(f"| {r['trial']} | {r['status']} | {r['step']} / {r.get('total_steps') or '—'} | {batch} | {fmt('train_loss')} | {fmt('peak_allocated_gib')} |")
    text+=['','![完整实验曲线](arch70m_results/curves.png)','',
           '原始逐步曲线与分来源指标：[CSV](arch70m_results/all_curves.csv)；汇总：[JSON](arch70m_results/summary.json)。',
           '图中训练 CE 同时展示原始值和 25 步移动平均；验证指标不平滑。训练 GPU 小时不含评估、保存和 profile，不能当作端到端成本。',
           '未完成试验不参与候选选择或显著性判断。三种子确认必须从头训练；筛选 checkpoint 不作为确认初始化。',
           '<!-- ARCH70M_RESULTS_END -->']
    path=ROOT/'docs/ARCH70M_EXPLORATION.md'; book=path.read_text() if path.exists() else '# V8 70M architecture exploration\n\n'
    a=book.find('<!-- ARCH70M_RESULTS_BEGIN -->'); b=book.find('<!-- ARCH70M_RESULTS_END -->')
    if a>=0 and b>=a: book=book[:a]+'\n'.join(text)+book[b+len('<!-- ARCH70M_RESULTS_END -->'):]
    else: book+='\n'+'\n'.join(text)+'\n'
    path.write_text(book)
    return summary
