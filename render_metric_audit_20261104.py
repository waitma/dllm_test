"""Render the metric audit and annotate current conclusions, without rewriting results.

After metric_audit_20261104.py --cpu/--gpu/--overlap, activate pllm and run:
MPLCONFIGDIR=/tmp/arch70m_metric_plot output/arch70m_env/bin/python -B render_metric_audit_20261104.py
All repository writes hold the shared publication lock. No training is started.
"""
from collections import Counter
from datetime import datetime, timezone
import csv
import fcntl
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore')
OUT=ROOT/'docs/arch70m_results/metric_audit_20261104'
read=lambda name:json.loads((OUT/name).read_text())
key=lambda x:(x['source'],x['key'],x['role'])


def render():
    cpu,gpu,overlap=[read(x) for x in ['cpu_audit.json','gpu_audit.json','training_overlap.json']]
    endpoints=[x for x in cpu['endpoints'] if x['variant']=='A0']
    a0=[x for x in gpu['results'] if x['variant']=='A0']
    mean=lambda xs:float(np.mean(xs))
    cdr={name:mean([x['cdr_regions'][name]['source_macro_chain_aar'] for x in a0]) for name in ['CDR1','CDR2','CDR3']}
    initial=mean([x['initial_representation']['macro_auroc'] for x in endpoints])
    final=mean([x['final_representation']['macro_auroc'] for x in endpoints])
    full=mean([x['generation']['conditional_aar'] for x in endpoints])
    all_cdr=mean([x['generation']['cdr_aar'] for x in endpoints])
    role_cdr3={role:mean([d['aar'] for x in a0 for d in x['details'] if d['role']==role and d['region']=='CDR3'])
        for role in sorted({d['role'] for d in a0[0]['details']})}
    seen_full={key(x) for x in overlap['exact_full_chain_matches']}
    seen_cdr3={key(x) for x in overlap['exact_cdr3_matches']}
    cdr_ids={key(d) for d in a0[0]['details']}
    generation_rows=[]
    for x in gpu['results']:
        for d in x['details']:
            generation_rows.append(dict(d,full_chain_seen_in_training=key(d) in seen_full,cdr3_seen_in_training=key(d) in seen_cdr3))
    srcs=['tcr_papers','trait']
    auc_rows=[]
    for source in srcs:
        auc_rows.append(dict(source=source,
            random_backbone=mean([x['initial_representation']['sources'][source]['auroc'] for x in endpoints]),
            trained_backbone=mean([x['final_representation']['sources'][source]['auroc'] for x in endpoints]),
            missing_alpha_only=cpu['baselines'][source]['missing_alpha_only']['auroc'],
            layout_only=cpu['baselines'][source]['layout']['auroc']))
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    ax=axes[0];vals=[full,all_cdr,cdr['CDR1'],cdr['CDR2'],cdr['CDR3'],role_cdr3['antibody_heavy']]
    bars=ax.bar(range(6),np.array(vals)*100,color=['#426e86','#56a29a','#a7c9bd','#a7c9bd','#e0a35b','#bd7056'])
    ax.set_xticks(range(6),['Full\nchain','All\nCDRs','CDR1','CDR2','CDR3','Ab heavy\nCDR3']);ax.set_ylim(0,105);ax.set_ylabel('AAR (%)')
    ax.set_title('A0: different reconstruction tasks')
    ax.bar_label(bars,labels=[f'{x*100:.1f}' for x in vals],padding=3,fontsize=9)
    ax.text(.02,-.21,'Full: unknown length; CDR: visible framework + known positions.',transform=ax.transAxes,fontsize=8)
    ax=axes[1];xx=np.arange(2);width=.24
    for j,(field,label,color) in enumerate([('random_backbone','Random backbone + trained probe','#93abb6'),
             ('trained_backbone','Trained A0 + trained probe','#426e86'),('missing_alpha_only','Missing-alpha flag only','#bd7056')]):
        vv=[x[field] for x in auc_rows];b=ax.bar(xx+(j-1)*width,vv,width,label=label,color=color)
        ax.bar_label(b,fmt='%.3f',padding=2,fontsize=8)
    ax.axhline(.5,color='grey',linestyle='--',lw=.8);ax.set_ylim(0,1.04);ax.set_xticks(xx,srcs);ax.set_ylabel('AUROC');ax.set_title('High AUROC precedes pretraining')
    ax.legend(fontsize=7,loc='lower right')
    ax=axes[2];b=ax.bar(['Binding\n(n=7)','Nonbinding\n(n=26)'],[6/7*100,0],color=['#bd7056','#426e86'])
    ax.bar_label(b,labels=['6 / 7','0 / 26'],padding=4);ax.set_ylim(0,105);ax.set_ylabel('Records without observed alpha chain (%)');ax.set_title('tcr_papers label / missingness shortcut')
    fig.suptitle('Metric audit: same checkpoints, unchanged V8 inputs; A0 seeds 42 / 43',fontsize=13)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'audit.{ext}',dpi=180)
    svg=OUT/'audit.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    with (OUT/'cdr_regions.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(generation_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(generation_rows)
    with (OUT/'probe_controls.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(auc_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(auc_rows)
    timestamp=datetime.now(timezone.utc).isoformat(timespec='seconds')
    lines=['# 指标审计：高 CDR AAR / AUROC 的含义与评测缺口','',f'完成时间：{timestamp}。只读检查现有数据与最终权重，没有训练、改写输入或修改冻结评分器。','',
        '用户的质疑成立：三项数字衡量不同任务；AUROC 还受到明确的缺链捷径影响，现有隔离只按受体组合做精确去重。之前把高 AUROC 作为强表征证据的说法需要降级。此审计没有发现 CDR 评分掺入框架/padding 或关系标签直接进入编码器的错误。','',
        '## 1. 评分口径与 CDR 分区域复算','',
        '| 指标 | A0 两 seed 均值 | 任务条件 |','|---|---:|---|',
        f'| 整链 AAR | {full:.2%} | 117 条链/4 来源；整条定容区域全遮蔽，生成 EOS 和长度，按完整参考长度计分 |',
        f'| 全 CDR AAR | {all_cdr:.2%} | 64 条链：OAS 抗体 H/L 各16，OTS TCR β/α 各16；只遮 CDR，框架/另一条链/EOS 保留 |',
        *[f'| {name} AAR | {value:.2%} | 相同64条 CDR 补全任务，按来源等权平均链级 AAR |' for name,value in cdr.items()],
        f"| 抗体重链 CDR3 | {role_cdr3['antibody_heavy']:.2%} | OAS 中16条重链，每个种子固定同一组 |",'',
        'CDR 是保留框架、已知各 CDR 位置/长度的补全。框架可提供 V/J 相关信息；CDR1/2 很高不等于未知长度整链生成好，也不等于 CDR3 或新结合特异性好。64条任务不是64条抗体：一半来自 OTS TCR。','',
        'A0 整链提前 EOS：seed42/43 的来源宏平均预测长度分别为37.10/11.49，参考114.18；参考覆盖分别33.04%/10.17%。A5 同口径为95.39%/94.04%。原评分把未输出的尾部计为不匹配，因此整链 AAR 显著更低；已覆盖位置准确率与长度问题都应分开报告。','',
        '即使只比较相同 OAS/OTS 来源，A0 整链 AAR 仍分别只有13.72%/4.38%，CDR 为79.80%/79.62%。因此来源差异不是唯一原因，遮蔽/长度条件是实质差异。','',
        '## 2. AUROC 的明确捷径','',
        f'A0 随机权重加监督线性探针的宏 AUROC 已为 {initial:.4f}，训练后为 {final:.4f}（增加{final-initial:.4f}）。随机的是骨干；探针仍在 probe_train 用真实标签训练。不能把随机骨干基线称为随机猜测。','',
        '当前探针只有 tcr_papers 与 TRAIT 两个来源，分别独立拟合 StandardScaler + LogisticRegression，再等权取两个 AUROC。train383、valid106；tcr_papers valid33中仅7正例，TRAIT valid73中9正例。不是通用蛋白/抗体表征排行榜。','',
        '| 来源 | 未训练 A0 + 探针 | 训练 A0 + 探针 | 仅缺 α 链指示器 | 仅布局/长度特征 + 探针 |','|---|---:|---:|---:|---:|',
        *[f"| {x['source']} | {x['random_backbone']:.4f} | {x['trained_backbone']:.4f} | {x['missing_alpha_only']:.4f} | {x['layout_only']:.4f} |" for x in auc_rows], '',
        'tcr_papers train：30正例中12缺 α 链，112负例全部观测到 α 链；valid：7正例中6缺 α 链，26负例全部观测到 α 链。缺链在 receptor_vectors 中产生零向量，输入本身也保留观测缺失模式。这是标签与采样/缺失模式相关的捷径，不是把 labels 字段直接喂给模型。元数据子来源与正负类也不平衡，详见 cpu_audit.json。','',
        '只保留两链均有观测的 tcr_papers valid，原 A0 探针 AUROC seed42/43变成0.6538/0.6923；但剩下27条且只有1正例，不能把这个事后子集作为可靠的新泛化估计。TRAIT 对应72条、8正例，A0 为0.7051/0.7012。','',
        '## 3. 全部固定10%训练记录的精确重叠检查','',
        f"实际读取 {overlap['scanned_selected_training_records']:,} 条已选训练记录。评测与预训练完整受体组合的精确重叠为0，probe_train/probe_valid组合重叠也为0。但是：",'',
        f"- 117条整链任务中 {len(seen_full)} 条目标单链在训练中出现过；其搭配链不同，因而未被组合去重排除。",
        f"- 64条 CDR 任务中 {len(cdr_ids&seen_full)} 条完整目标单链在训练中出现过；{len(seen_cdr3)} 条目标 CDR3 在训练中出现过。",
        '- CDR3 重叠按角色：OAS L=13、H=1，OTS α=4、β=2；不能将这些任务统称为未见过序列的泛化验证。',
        '- probe_train/valid 无相同非空 β 链；TRAIT 有1条非空 α 链重复。空 α 链不能当作相同蛋白序列重复计数。',
        '- 此检查仅是精确身份检查，没有证明同源、克隆型、V/J 或研究来源隔离；重复也不足以单独解释全部高分。','',
        '## 4. 已验证的计算与复现限制','',
        '- 117条整链和64条 CDR 的索引与真实序列对齐；CDR 分母只含被遮蔽的真实残基，框架/EOS/padding 未计分。',
        '- 489条探针记录逐条翻转关系标签后，全部7个编码器输入特征与 pooling residue_mask 完全一致；没有直接关系标签穿透。',
        '- 10个模型端点及初始评测的40个来源 AUROC/AUPRC，均由保存的逐样本概率独立重算一致；所有保存整链 AAR 也由预测序列和完整参考长度重算一致。',
        '- A0 seeds42/43 与 A5 seed42重新推理，各64条 CDR评分与原产物一致；A0 seed43、A5 seed42全181条观察也完全一致。',
        '- A0 seed42有2/117整链解码与原产物不同（原始来源均为ASD，CDR0差异）。该细小复现问题未定位，不将本次A0 seed42整链推理声称为完全精确重放；原始长度统计仍来自原始保存序列。详情 replay_A0_42.json / gpu_audit.json。', '',
        '## 5. 结论与后续使用规则','',
        '保留所有历史原始数字，但名称必须写明“未知长度整链重建”“已知框架/位置的CDR补全”“来源内小样本结合探针”。旧 AUROC 架构排序暂停用于泛化能力结论；A5 对 A0 的长度/整链重建改善只保留为当前冻结任务上的描述性结果。C2 等候选需要在控制缺链模式、研究来源及单链/CDR3/同源隔离后的评测再比较，现有带捷径探针不足以支持稳定表征提升。','',
        '后续修正应另建带版本的评测清单与评分报告，按原V8 reader/collator读取，保留原始记录形式及历史权重。应分别报告 CDR1/2/3 与 H/L/β/α，加入随机骨干、缺链指示器和布局基线，并为新泛化结论准备独立样本。此次未启动新训练、没有改变实验筛选/显著性门槛，也不把事后子集当作新确认集。','',
        f'![审计图]({OUT}/audit.png)','',
        f'[CPU输入/概率检查]({OUT}/cpu_audit.json) · [GPU逐区域核对]({OUT}/gpu_audit.json) · [训练精确重叠]({OUT}/training_overlap.json) · [逐CDR CSV]({OUT}/cdr_regions.csv) · [AUROC控制CSV]({OUT}/probe_controls.csv)','',
        f'复现代码：{ROOT}/metric_audit_20261104.py 与 {ROOT}/render_metric_audit_20261104.py。原seed42/43绑定源码SHA依然为258dd11e8895c3b6d4fbb9c00e526de67e6bc9ccbef43948970790f5a00d8fed / 28609d7f776013915561eb038396d54181dcee7b1c32d071be0a9da5dc12faad。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    summary=dict(created=timestamp,a0_full_aar=full,a0_cdr_aar=all_cdr,a0_regions=cdr,a0_cdr3_by_role=role_cdr3,
        a0_initial_macro_auroc=initial,a0_final_macro_auroc=final,cdr_exact_chain_overlap=len(cdr_ids&seen_full),
        cdr_exact_cdr3_overlap=len(seen_cdr3),decision='Existing probe ranking cannot establish representation generalization',
        inputs_unchanged=True,training_started=False)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    note=f'> 2026-11-04 指标审计更正：CDR是已知框架/位置的补全，不能与未知长度整链AAR直接比较；tcr_papers仅缺α链指示器AUROC已0.9286，A0随机骨干宏AUROC已0.8206。64条CDR任务中10条整链、20条CDR3与固定10%训练子集精确重复；现有AUROC排序不能作为表征泛化提升证据。详见 {OUT}/REPORT.md。\n\n'
    review=ROOT/'docs/arch70m_results/review_20261104/REPORT.md'
    old=review.read_text()
    if not old.startswith('> 2026-11-04 指标审计更正'):review.write_text(note+old)
    process=ROOT/'PROJECT_PROCESS.md'
    process.write_text(f'> Last updated: {timestamp}。完成用户提出的A0指标疑点审计：489条标签翻转输入不变、40组概率指标复算一致、3份既有权重逐CDR核对、653288条训练精确重叠检查；确认缺α链捷径（仅该指示器AUROC0.9286）及目标单链/CDR3隔离缺口。旧表征泛化结论降级，未训练新任务。报告 {OUT}/REPORT.md。\n\n'+process.read_text())
    plan=ROOT/'BIOSEQ_MODEL_PLAN.md'
    plan.write_text(note+plan.read_text())
    guide=ROOT/'PROJ_GUIDE.md'
    guide.write_text(f'> 2026-11-04 审计入口 {ROOT}/metric_audit_20261104.py --cpu / --gpu / --overlap；报告 {OUT}/REPORT.md。历史指标三种任务口径必须区分，保留所有旧数字。来源内AUROC存在缺链捷径，不能据此认定架构表征更强。后续评测独立版本并控制缺链/单链/CDR3/同源隔离，原V8形式及冻结源码不改。\n\n'+guide.read_text())
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    with (ROOT/'output/arch70m_publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        render()
