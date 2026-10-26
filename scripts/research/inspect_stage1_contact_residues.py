"""Inspect observed H/L contacts and their cached teacher feature vectors, CPU only.

Activate pllm, then run this absolute file with --cache ABS_DIAGNOSTIC_DIRECTORY
and --output ABS_WRITABLE_DIRECTORY. Requires the full teacher tensors produced
by inspect_stage1_cross_chain.py. Never updates training data or model weights.
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from safetensors.numpy import load_file
from scipy.special import softmax


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, output, name):
    fig.savefig(output/(name+'.png'), dpi=150, facecolor='white')
    fig.savefig(output/(name+'.pdf'), facecolor='white')
    plt.close(fig)


def csv_write(path, rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),
                         'font.size':16, 'axes.unicode_minus':False, 'pdf.fonttype':42})
    examples=json.loads((args.cache/'examples.json').read_text())['examples']
    summaries=[]
    for ex in examples:
        if ex['group']!='structure_reference': continue
        pdb=ex['reference']['pdb_id']; nh,nl=ex['lengths']
        seq=[c['sequence'] for c in ex['request']['chains']]
        path=args.cache/'teacher_full'/(ex['id']+'.safetensors')
        data=load_file(str(path))
        z=data['pair_states'].astype(np.float32)
        raw=z[:nh,nh:]; reverse=z[nh:,:nh].transpose(1,0,2)
        head=raw+reverse  # Exact symmetrization used by the distogram head.
        logits=data['distogram_logits'][:nh,nh:].astype(np.float64)
        prob=softmax(logits,axis=-1); short=prob[...,:19].sum(-1)
        coordpath=args.cache/'structures'/(pdb+'_coordinates.npz')
        distance=np.load(coordpath)['distances'][:nh,nh:]
        contact=np.isfinite(distance)&(distance<8)
        hi=np.flatnonzero(contact.any(1)); li=np.flatnonzero(contact.any(0))
        mapping=[]
        for role in ('heavy','light'):
            mapping.append({int(i):ast.literal_eval(r) for i,r in ex['reference'][role+'_mapping']})
        def pdb_res(chain,index):
            item=mapping[chain][int(index)]
            return str(item[1])+item[2].strip()
        def label(chain,index):
            return f'{"H" if chain==0 else "L"}{index+1}{seq[chain][index]}'
        def row(i,j,kind):
            return dict(pdb=pdb,kind=kind,h_seq_position=int(i+1),h_aa=seq[0][i],
                h_pdb_chain=ex['reference']['heavy_chain'],h_pdb_residue=pdb_res(0,i),
                l_seq_position=int(j+1),l_aa=seq[1][j],
                l_pdb_chain=ex['reference']['light_chain'],l_pdb_residue=pdb_res(1,j),
                cb_distance_angstrom=float(distance[i,j]),is_contact=bool(contact[i,j]),
                teacher_short_bin_score=float(short[i,j]),teacher_modal_bin=int(prob[i,j].argmax()))
        all_contacts=[row(int(i),int(j),'contact') for i,j in zip(*np.where(contact))]
        csv_write(args.output/(pdb+'_all_contacts.csv'),all_contacts)
        # Predefined structural rule: max contact count, ties by sequence position.
        hotspot=int(np.argmax(contact.sum(1)))
        positive=np.flatnonzero(contact[hotspot])
        candidates=[int(j) for j in li if np.isfinite(distance[hotspot,j]) and distance[hotspot,j]>=12]
        candidates.sort(key=lambda j:(distance[hotspot,j],j))
        negative=np.array(candidates[:len(positive)],dtype=int)
        assert len(negative)>0
        selected=np.r_[positive,negative];k=len(positive)
        rows=[row(hotspot,int(j),'contact' if r<k else 'far_control') for r,j in enumerate(selected)]
        csv_write(args.output/(pdb+'_selected_pairs.csv'),rows)
        mean=head.reshape(-1,256).mean(0);std=head.reshape(-1,256).std(0)
        standardized=(head[hotspot,selected]-mean)/np.maximum(std,1e-6)
        ci,cj=np.where(contact)
        np.savez_compressed(args.output/(pdb+'_contact_features.npz'),
            selected_pair_indices_0based=np.stack([np.full(len(selected),hotspot),selected],axis=1),
            selected_is_contact=np.arange(len(selected))<k,
            raw_z_hl=raw[hotspot,selected],raw_z_lh=reverse[hotspot,selected],
            symmetric_head_input=head[hotspot,selected],standardized_head_input=standardized,
            channel_mean=mean,channel_std=std,distogram_probabilities=prob[hotspot,selected].astype(np.float32),
            distances=distance[hotspot,selected],all_contact_indices_0based=np.stack([ci,cj],axis=1),
            all_contact_z_hl=raw[ci,cj],all_contact_symmetric_head_input=head[ci,cj],
            interface_h_indices_0based=hi,interface_l_indices_0based=li,
            interface_distances=distance[np.ix_(hi,li)],interface_short_scores=short[np.ix_(hi,li)])

        fig,axes=plt.subplots(1,2,figsize=(25,15),layout='constrained')
        cut=distance[np.ix_(hi,li)];score=short[np.ix_(hi,li)]
        for ax,values,title,cmap,vmin,vmax in zip(axes,[cut,score],
                ['实验结构：数字为接触对距离（Å）','教师：短距离概率分数；白框标出实验接触'],
                ['viridis_r','magma'],[3,0],[20,1]):
            im=ax.imshow(values,aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest')
            ax.set_xticks(np.arange(len(li)),[label(1,int(j)) for j in li],rotation=90,fontsize=15)
            ax.set_yticks(np.arange(len(hi)),[label(0,int(i)) for i in hi],fontsize=15)
            ax.set_xlabel('轻链：真实输入序列位置＋氨基酸',fontsize=19)
            ax.set_ylabel('重链：真实输入序列位置＋氨基酸',fontsize=19)
            ax.set_title(title,fontsize=21,pad=15)
            for a,b in zip(*np.where(cut<8)):
                if ax is axes[0]:
                    ax.text(b,a,f'{cut[a,b]:.1f}',ha='center',va='center',fontsize=13,color='black')
                else:
                    ax.add_patch(plt.Rectangle((b-.48,a-.48),.96,.96,fill=False,edgecolor='white',linewidth=1.1))
            fig.colorbar(im,ax=ax,shrink=.7,label='距离（显示上限20 Å）' if ax is axes[0] else '前19 bins概率和')
        fig.suptitle(f'{pdb.upper()}：把真实 H–L 接触位置展开\n仅保留至少参与一次接触的残基；轴位置不连续，编号非IMGT',fontsize=25)
        save(fig,args.output,pdb+'_interface_residues')

        fig,axes=plt.subplots(1,2,figsize=(26,max(10,len(selected)*.56)),
            gridspec_kw={'width_ratios':[1.6,1]},layout='constrained')
        labels=[f'{"接触" if r<k else "远距"}  {label(0,hotspot)}–{label(1,int(j))}  {distance[hotspot,j]:.2f} Å'
            for r,j in enumerate(selected)]
        im=axes[0].imshow(standardized,aspect='auto',cmap='coolwarm',vmin=-3,vmax=3,interpolation='nearest')
        axes[0].set_yticks(np.arange(len(selected)),labels,fontsize=15)
        axes[0].set_xticks([0,63,127,191,255],[1,64,128,192,256])
        axes[0].set_xlabel('256个特征通道（不是残基位置）',fontsize=19)
        axes[0].set_title('教师 Z[i,j]＋Z[j,i]：逐通道标准化展示',fontsize=21)
        fig.colorbar(im,ax=axes[0],shrink=.6,label='相对该结构全部H–L对的通道标准分')
        im=axes[1].imshow(prob[hotspot,selected],aspect='auto',cmap='magma',vmin=0,vmax=1,interpolation='nearest')
        axes[1].set_yticks(np.arange(len(selected)),[f'{short[hotspot,j]:.3f}' for j in selected],fontsize=16)
        axes[1].set_ylabel('对应行的短距离分数',fontsize=18)
        axes[1].set_xticks([0,18,31,47,63]);axes[1].axvline(18.5,color='cyan',lw=2)
        axes[1].set_xlabel('距离类别 bin（0–63）；青线左侧为短距离分数',fontsize=18)
        axes[1].set_title('同一残基对的64类距离分布',fontsize=21)
        fig.colorbar(im,ax=axes[1],shrink=.6,label='每个距离类别的概率')
        for ax in axes: ax.axhline(k-.5,color='green',lw=3);ax.tick_params(axis='x',labelsize=16)
        fig.suptitle(f'{pdb.upper()}：固定 {label(0,hotspot)}，比较它接触和不接触的轻链位置\n上方{len(positive)}对实验距离<8 Å；下方{len(negative)}对≥12 Å；按实验几何选择，未按预测筛选',fontsize=24)
        save(fig,args.output,pdb+'_contact_feature_vectors')
        entry=dict(pdb=pdb,contacts=int(contact.sum()),interface_h_residues=len(hi),interface_l_residues=len(li),
            hotspot=label(0,hotspot),selected_pairs=rows,
            full_contact_short_score_mean=float(short[contact].mean()),
            full_noncontact_short_score_mean=float(short[np.isfinite(distance)&~contact].mean()),
            selected_contact_short_score_mean=float(short[hotspot,positive].mean()),
            selected_control_short_score_mean=float(short[hotspot,negative].mean()),
            source_teacher_sha256=digest(path),source_coordinates_sha256=digest(coordpath))
        summaries.append(entry)
        print(json.dumps({k:v for k,v in entry.items() if k!='selected_pairs'},ensure_ascii=False),flush=True)
    manifest=dict(selection='max contact-count H residue; all its <8 A contacts; same H paired with closest >=12 A interface L residues as controls',
        contact='CB distance <8 A; CA for Gly; finite coordinate pairs only',
        numbering='1-based actual sequence positions; CSV includes mapped PDB chain/residue identifiers; not IMGT',
        features='raw Z_ij and Z_ji saved; plots use Z_ij+Z_ji, standardized per channel over all H-L pairs in that structure',
        scores='softmax distance bins0..18; not exact calibrated P(distance<8 A)',
        limitations=['structure-informed case diagnosis, not held-out probe performance',
            'feature channels have no assigned biological meaning; visualization clips standardized values to [-3,3]',
            'noncontact controls keep H residue but do not match L amino-acid identity',
            'CB-based contact convention does not imply a specific bond or atomistic interaction'],
        examples=summaries)
    (args.output/'summary.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
