"""Plot full-matrix diagnostics after inspect_stage1_cross_chain completes.

Activate pllm, then run this absolute file with --output ABS_DIAGNOSTIC_DIRECTORY.
Creates readable PNG/PDF heatmaps, per-example metrics and a summary. Structural
references are illustrative, not a new held-out benchmark or model-selection set.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager
import numpy as np
from safetensors.numpy import load_file
from scipy.special import log_softmax
from scipy.stats import spearmanr, rankdata


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def short_score(logits,bins=19):
    return np.exp(log_softmax(logits.astype(np.float64),axis=-1))[...,:bins].sum(-1)


def pair_kl(teacher,student,mask):
    p=log_softmax(teacher.astype(np.float64),axis=-1)
    q=log_softmax(student.astype(np.float64),axis=-1)
    return float((np.exp(p)*(p-q)).sum(-1)[mask].mean())


def contact_metrics(distance,score):
    valid=np.isfinite(distance)
    label=(distance[valid]<8).astype(int)
    s=score[valid]
    k=min(min(distance.shape),len(s))
    ap,auc=ranking_metrics(label,s)
    return dict(pairs=len(s),contacts=int(label.sum()),prevalence=float(label.mean()),
        average_precision=ap,roc_auc=auc,
        precision_at_min_chain_length=float(label[np.argsort(-s,kind='stable')[:k]].mean()),top_k=k)


def ranking_metrics(label,score):
    """Threshold-grouped average precision and tie-corrected Mann-Whitney AUROC."""
    label=np.asarray(label);score=np.asarray(score)
    positives=int(label.sum());negatives=len(label)-positives
    if not positives or not negatives:
        raise ValueError('Contact ranking requires both positive and negative pairs')
    order=np.argsort(-score,kind='stable');y=label[order];s=score[order]
    ends=np.r_[np.flatnonzero(np.diff(s)!=0),len(s)-1]
    tp=np.cumsum(y)[ends]
    ap=np.sum((tp/(ends+1))*np.diff(np.r_[0,tp])/positives)
    auc=(rankdata(score,method='average')[label.astype(bool)].sum()-positives*(positives+1)/2)/(positives*negatives)
    return float(ap),float(auc)


def feature_pc(array,other=None):
    n=len(array)
    both=array if other is None else np.concatenate([array,other])
    mean=both.mean(0);centered=both-mean
    _,v=np.linalg.eigh(centered.T@centered/max(1,len(both)-1))
    direction=v[:,-1]
    direction*=1 if direction[np.argmax(np.abs(direction))]>=0 else -1
    projection=centered@direction
    return projection[:n],None if other is None else projection[n:]


def save(fig,output,name):
    fig.savefig(output/(name+'.png'),dpi=160,facecolor='white')
    fig.savefig(output/(name+'.pdf'),facecolor='white')
    plt.close(fig)


def heat(ax,array,title,*,full=None,cmap='magma',vmin=0,vmax=1):
    im=ax.imshow(array,origin='upper',aspect='auto',interpolation='nearest',cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set_title(title,fontsize=17,pad=10)
    if full is None:
        ax.set_xlabel('轻链残基位置',fontsize=15)
        ax.set_ylabel('重链残基位置',fontsize=15)
        ax.set_xticks([0,array.shape[1]-1],[1,array.shape[1]])
        ax.set_yticks([0,array.shape[0]-1],[1,array.shape[0]])
    else:
        nh=full;n=array.shape[0]
        ax.axhline(nh-.5,color='cyan',lw=1.3);ax.axvline(nh-.5,color='cyan',lw=1.3)
        ticks=[(nh-1)/2,nh+(n-nh-1)/2]
        ax.set_xticks(ticks,['重链 H','轻链 L']);ax.set_yticks(ticks,['重链 H','轻链 L'])
        ax.set_xlabel('残基位置',fontsize=15);ax.set_ylabel('残基位置',fontsize=15)
    ax.tick_params(labelsize=14)
    return im


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output
    font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font_path)
    font=FontProperties(fname=font_path)
    plt.rcParams.update({'font.family':font.get_name(),'font.size':16,'axes.unicode_minus':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    manifest=json.loads((output/'examples.json').read_text())
    arrays={};metrics=[]
    for ex in manifest['examples']:
        key=ex['id'];nh,nl=ex['lengths'];n=nh+nl
        teacher=load_file(str(output/'teacher_full'/(key+'.safetensors')))
        initial=load_file(str(output/'student_full'/(key+'_pair_kd_0.safetensors')))
        trained=load_file(str(output/'student_full'/(key+'_pair_kd_3000.safetensors')))
        assert teacher['distogram_logits'].shape==(n,n,64)
        i,j=np.indices((n,n));same=(i<nh)==(j<nh)
        masks={'intra':same&(i<j),'intra_sep6':same&(j-i>=6),'inter':(i<nh)&(j>=nh)}
        entry=dict(id=key,group=ex['group'],lengths=ex['lengths'],kl={},structure={})
        t=short_score(teacher['distogram_logits'])
        s0=short_score(initial['masked_logits']);s1=short_score(trained['masked_logits'])
        sc0=short_score(initial['clean_logits']);sc1=short_score(trained['clean_logits'])
        arr=dict(teacher=t,initial=s0,trained=s1,initial_clean=sc0,trained_clean=sc1)
        for view in ('clean','masked'):
            entry['kl'][view]={}
            for name,mask in masks.items():
                entry['kl'][view][name]={str(step):pair_kl(teacher['distogram_logits'],data[view+'_logits'],mask)
                    for step,data in [(0,initial),(3000,trained)]}
        inter=masks['inter']
        entry['inter_teacher_student_spearman']={str(step):float(spearmanr(t[inter],s[inter]).statistic)
            for step,s in [(0,s0),(3000,s1)]}
        entry['inter_short_score']={name:dict(mean=float(s[inter].mean()),max=float(s[inter].max()),std=float(s[inter].std()))
            for name,s in [('teacher',t),('student0',s0),('student3000',s1)]}
        if ex['group']=='structure_reference':
            distance=np.load(output/'structures'/(ex['reference']['pdb_id']+'_coordinates.npz'))['distances'][:nh,nh:]
            arr['truth']=np.where(np.isfinite(distance),(distance<8).astype(float),np.nan)
            for name,logits in [('teacher',teacher['distogram_logits']),('student0',initial['clean_logits']),('student3000',trained['clean_logits'])]:
                entry['structure'][name]=contact_metrics(distance,short_score(logits)[:nh,nh:])
            entry['bin_sensitivity_teacher']={str(k):contact_metrics(distance,short_score(teacher['distogram_logits'],k)[:nh,nh:]) for k in (18,19,20)}
            tp=teacher['pair_states'][:nh,nh:].astype(np.float32)
            ip=initial['pair_states'][:nh,nh:].astype(np.float32)
            sp=trained['pair_states'][:nh,nh:].astype(np.float32)
            pc,_=feature_pc(tp.reshape(-1,tp.shape[-1]))
            p0,p1=feature_pc(ip.reshape(-1,64),sp.reshape(-1,64))
            arr.update(teacher_pc=pc.reshape(nh,nl),initial_pc=p0.reshape(nh,nl),trained_pc=p1.reshape(nh,nl))
        arrays[key]=arr;metrics.append(entry)
        print(json.dumps(dict(phase='metrics',id=key,inter_kl=entry['kl']['masked']['inter'],structure=entry['structure'])),flush=True)
    # Keep the displayed numerical maps even when large full-bin caches are local.
    np.savez_compressed(output/'visualization_arrays.npz',
        **{f'{key}__{name}': value.astype(np.float32)
           for key, maps in arrays.items() for name, value in maps.items()})
    write_json(output/'metrics.json',metrics)
    with (output/'metrics.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['id','group','view','region','step0_KL','step3000_KL'])
        for row in metrics:
            for view,regions in row['kl'].items():
                for region,steps in regions.items():
                    writer.writerow([row['id'],row['group'],view,region,steps['0'],steps['3000']])
    valid=[x for x in manifest['examples'] if x['group']=='oas_valid']
    references=[x for x in manifest['examples'] if x['group']=='structure_reference']
    fig,axes=plt.subplots(2,3,figsize=(17,11),layout='constrained')
    for ex,ax in zip(valid,axes.flat):
        nh,nl=ex['lengths']
        im=heat(ax,arrays[ex['id']]['teacher'],f'{ex["id"]} · H={nh}, L={nl}',full=nh)
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7,label='短距离概率分数（0–1）')
    fig.suptitle('教师完整 Distogram：对角块为链内，非对角块为跨链\n颜色为前19个距离bin的概率和；不是 attention 权重',fontsize=21)
    save(fig,output,'teacher_full_examples')
    fig,axes=plt.subplots(6,3,figsize=(17,25),layout='constrained')
    for ex,row_axes in zip(valid,axes):
        nh=ex['lengths'][0];arr=arrays[ex['id']]
        for ax,key,title in zip(row_axes,['teacher','initial','trained'],['教师完整序列','学生 step0 · 固定 MASK','学生 step3000 · 同一 MASK']):
            im=heat(ax,arr[key][:nh,nh:],f'{ex["id"]} | {title}')
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.5,label='短距离概率分数（统一 0–1 色标）')
    fig.suptitle('六个样例的 H–L 跨链区域：训练前后与教师对照\n当前学生训练仅接受链内 KD，跨链区域从未直接监督',fontsize=21)
    save(fig,output,'cross_chain_comparison')
    fig,axes=plt.subplots(3,4,figsize=(22,13),layout='constrained')
    for ex,row_axes in zip(references,axes):
        nh=ex['lengths'][0];arr=arrays[ex['id']]
        for ax,key,title in zip(row_axes,['truth','teacher','initial_clean','trained_clean'],['实验接触 · Cβ距离<8Å','教师预测','学生 step0 · 完整序列','学生 step3000 · 完整序列']):
            matrix=arr[key] if key=='truth' else arr[key][:nh,nh:]
            im=heat(ax,matrix,f'{ex["reference"]["pdb_id"].upper()} | {title}')
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.6,label='实验列为0/1；预测列为短距离概率分数')
    fig.suptitle('三个实验结构：跨链热图是否对应真实 H–L 接触？\n预先固定的抗体 Fab 案例；仅作诊断，不代表未见样本泛化',fontsize=21)
    save(fig,output,'structure_validation')
    fig,axes=plt.subplots(3,3,figsize=(17,14),layout='constrained')
    for ex,row_axes in zip(references,axes):
        arr=arrays[ex['id']]
        shared=max(np.abs(arr['initial_pc']).max(),np.abs(arr['trained_pc']).max())
        for ax,key,title in zip(row_axes,['teacher_pc','initial_pc','trained_pc'],['教师 Pair Feature · 独立PC1','学生 step0 · 共同PC1','学生 step3000 · 共同PC1']):
            limit=np.abs(arr[key]).max() if key=='teacher_pc' else shared
            im=heat(ax,arr[key],f'{ex["reference"]["pdb_id"].upper()} | {title}',cmap='coolwarm',vmin=-limit,vmax=limit)
            fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle('跨链 Pair Feature 的第一主成分投影\n学生前后共享投影轴；教师为独立投影。颜色不是概率或注意力',fontsize=21)
    save(fig,output,'pair_feature_examples')
    student=json.loads((output/'student_full/manifest.json').read_text())
    context={}
    for arm,step in [('pair_kd',0),('mlm',3000),('pair_kd',3000)]:
        records=[r for r in student['records'] if r['arm']==arm and r['step']==step and r['id'].startswith('valid_')]
        counts=sum(c['count'] for r in records for c in r['context'])
        ce=sum(c['ce_sum'] for r in records for c in r['context'])/counts
        hidden=sum(c['partner_hidden_ce_sum'] for r in records for c in r['context'])/counts
        context[f'{arm}_{step}']=dict(selected_aa_tokens=counts,full_context_nll=ce,
            partner_hidden_nll=hidden,partner_benefit_nll=hidden-ce)
    summary=dict(examples=9,oas_validation_examples=6,structure_examples=3,
        masked_kl_macro={region:{step:float(np.mean([r['kl']['masked'][region][step] for r in metrics if r['group']=='oas_valid']))
            for step in ['0','3000']} for region in ['intra','intra_sep6','inter']},
        structural_macro={name:{metric:float(np.mean([r['structure'][name][metric] for r in metrics if r['group']=='structure_reference']))
            for metric in ['average_precision','precision_at_min_chain_length','prevalence']} for name in ['teacher','student0','student3000']},
        partner_context=context,limitations=manifest['limitations']+[
            'Short-distance score is bins0..18; exact checkpoint bin edges are not serialized. Reference convention and 18/19/20-bin sensitivity are reported.',
            'PCA colors are arbitrary feature coordinates, not contact probability or attention.',
            'Partner masking is an inference intervention with possible distribution shift; six examples are not a population estimate.'])
    write_json(output/'summary.json',summary)
    fig,axes=plt.subplots(1,3,figsize=(19,6),layout='constrained')
    x=np.arange(3);width=.35
    for offset,step,label in [(-width/2,'0','step0'),(width/2,'3000','step3000')]:
        axes[0].bar(x+offset,[summary['masked_kl_macro'][k][step] for k in ['intra','intra_sep6','inter']],width,label=label)
    axes[0].set_xticks(x,['链内全部','链内间隔≥6','跨链 H–L']);axes[0].set_ylabel('教师→学生 KL（越低越接近）');axes[0].legend()
    axes[0].set_title('6条验证样例：完整位置 KD')
    names=['teacher','student0','student3000']
    axes[1].bar(x,[summary['structural_macro'][k]['average_precision'] for k in names])
    axes[1].axhline(summary['structural_macro']['teacher']['prevalence'],ls='--',color='gray',label='接触占比参考')
    axes[1].set_xticks(x,['教师','学生0','学生3000']);axes[1].set_ylim(0,1);axes[1].set_ylabel('跨链接触 Average Precision');axes[1].legend(fontsize=13)
    axes[1].set_title('3个实验结构：接触识别')
    names=['pair_kd_0','mlm_3000','pair_kd_3000']
    axes[2].bar(x,[context[k]['partner_benefit_nll'] for k in names])
    axes[2].axhline(0,color='gray',lw=1);axes[2].set_xticks(x,['训练前','纯MLM','Pair＋链内KD'])
    axes[2].set_ylabel('隐藏伙伴后 NLL − 原 NLL');axes[2].set_title('伙伴上下文帮助程度\n正值：保留伙伴恢复更好')
    fig.suptitle('区分“拟合教师”“识别真实接触”和“利用伙伴恢复序列”',fontsize=21)
    save(fig,output,'diagnostic_summary')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
