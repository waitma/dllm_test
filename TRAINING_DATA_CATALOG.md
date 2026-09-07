# Current Training Data Catalog

审计日期：2026-07-23；canonical v2 执行更新：2026-08-04

> 🔴 **2026-08-28：标题里的 "Current" 已经不指现役训练。**
> 本文件通篇描述 `data/bioseq_grammar_v1` 的 7 源 Arrow 配方（`oas, ots, nanobody,
> tcr_piste, tcr_pmhc_fulllength, ppi, neutralization`，7L step389500 lineage）。
> 现役训练走的是完全另一条线：
> `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py`
> 直读 CSV 的七源 mix（`oas, ots, asd_antibody, trait, tcr_native, tcr_papers,
> tcr_repertoire`），**不经过 `bioseq_grammar_v1`，也不经过 `immune_receptor_v2`**。
> 两个 7 源配方只有 `oas`/`ots` 重合，其余五源互不相同，容易混淆。
>
> 现役语料的实测行数、残基占比、长度上限口径与新增源，见
> `/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md`
> 「当前训练语料实测快照（2026-08-28）」。去污染阈值与代价见
> `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/audit_2026_08_27/RETRAIN_PLAN.md`。
> 本文件保留为 step389500 的 lineage 事实。

项目根：
`/vepfs-mlp2/c20250601/251105016/project/dllm_test`

本文件只描述当前 7L checkpoint 实际使用的训练数据，以及下一版数据整理应解决的
问题。历史数据资产、候选数据和下游数据分别列出，不能因为它们已经落盘就视为已经
参与训练。

## 2026-08-04 canonical v2 执行更新

本文件第 1 节之后继续描述 step389500 的历史 7 源训练事实；它不能用来判断新数据
是否已进入训练。下一版 AB/TCR 候选数据已 canonicalize 到
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/immune_receptor_v2`，
完整权威报告为
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/IMMUNE_RECEPTOR_DATA_V2.md`。

- TCR union 922,479，strict core 41,489；antibody union 470,949，
  antibody-antigen interaction core 308,267；19,987 条 canonical property 中
  19,981 条满足 export-core。
- 九套 split 均通过 disjoint audit；源内 train/test 仅作 provenance。
- benchmark exact/near quarantine、cluster-level decontamination、train-only
  negatives、严格 OAS/OTS group resplit 与 immutable SHA256 manifests 已物化。
  candidate recipe `ir2recipe_1e3eac44551e88aedc6e` 含 real train 3,077,769、
  valid 39,138、test 38,565，另有 3,970 条 train-only TCR synthetic negatives。
- 当前是 `canonicalized=true`、`split_ready=true`、
  `technical_data_export_ready=true`，但因 SAbDab2/CATNAP rights、runtime views 与
  sampling weights 尚未关闭，仍为 `export_ready=false`、`training_ready=false`、
  `training_started=false`。不能直接把 canonical JSONL 接入现有 recipe，也不能声称
  step389500 已经见过这些新增 Ab-Ag records。
- 早先“SAbDab2 完整归档不含 `abag_split.csv`”是损坏下载导致的误判，已由官方
  MD5 校验的 876,381,859-byte 归档纠正。

## 0. 下一版 active scope：免疫受体

2026-07-23 起，下一版 foundation-training recipe 收敛为四个训练平面：

| training plane | biological object | 必须保留的条件 |
|---|---|---|
| antibody pairing | antibody heavy + light | H/L 链身份、FR/CDR、V/J、pair/cluster |
| TCR pairing | TCR α + β | α/β 链身份、FR/CDR、V(D)J、pair/clone/donor |
| antibody recognition | H + L + antigen chain(s) | antigen 实际序列、binding/affinity assay、complex/target id |
| TCR recognition | α/β 或 CDR3 + epitope + optional MHC/B2M | peptide、MHC allele/sequence scope、binding label、donor/assay |

下一版主训练配方明确不包含：

- nanobody/VHH/单域抗体，包括 `nanobody` shard 和 FLAb 的 AVIDa/VHH 数据；
- MINT、STRING 和通用 PPI，包括 `ppi`、`mint_ppi`、`mint_actions`；
- 当前 `neutralization` shard：它没有 antigen sequence，且 runtime 没有真正使用
  neutralization relation；
- 只有 target 名称、没有可追溯 target sequence mapping 的 antibody assay；
- TCRdb2.0 之类不成对、无 specificity 的 bulk repertoire。

MINT 和 nanobody 的历史 checkpoint/benchmark artifact 不删除，但不再参与下一版
训练数据权重、validation macro 或模型能力主张。

这里必须区分 checkpoint lineage：当前 step389500 已经读过 nanobody、PPI 和名义
neutralization。直接从它续训可以得到“后续只喂免疫受体数据”的 checkpoint，但不能
声称它从未见过这些排除源。若需要严格的 immune-receptor-only lineage，应从共同模型
初始化或尚未加入这些源的 checkpoint 重新训练。

### 0.1 本地可用数据审计

抗体侧：

| source | 本地审计结果 | 下一步状态 |
|---|---|---|
| OAS paired | source train 2,486,442；H/L 双链 | strict benchmark-clean export 1,498,849：1,468,754 / 15,295 / 14,800 |
| SAbDab2 `abag_split.csv` | paired VH/VL、排除 VHH/VNAR 并过滤 resolved protein/peptide antigen 后为 6,412 component records；3,363 single-polymer core + 3,049 multi-component aux | 已进入 canonical v2 与共同 recognition decontamination；官方 split 仅作 provenance，rights review 仍待完成 |
| FLAb/AbRank | 342,356 rows；76,515 rows 有合法 H/L/Ag 且适配当前 1024 单链上限；其中 75,483 rows 有 `fitness` | strict sequence-conditioned 子集已 canonicalize，并进入共同 receptor/antigen cluster export |
| FLAb/Kothiwal | 709 assay rows，去重后 385 个唯一 H/L/Ag，10 个 antigen sequence | 已按 assay/value provenance canonicalize，并进入共同 cluster export |
| 其他 FLAb binding | 多数只有固定 target 名称，未逐行保存 antigen sequence | 建立 versioned target-sequence map 后再考虑 |
| CoV-AbDab | 当前表有 variant/target 名称但没有 antigen sequence，且正负字段可同时出现 | 不进入下一版主 mix |

SAbDab2 v2 adapter 的严格输入为官方完整归档中的 `abag_split.csv`，原始归档固定在
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/sabdab2_ml/raw/splits.tar.gz`。
官方 `ab_ag_split`、`ab_cluster` 与 `agclusters` 全部保留为 provenance，但不会直接
作为下一版 train/validation/test；v2 另生成 antibody-disjoint、antigen-disjoint 与
joint-hard manifest。损坏旧下载以 `.invalid.5bc49f5d1e96` 留存，不得作为输入。

AbRank 位于
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/raw/flab/FLAb/data/binding/AbRank_dataset.csv.zip`。
其中 192,559 条 RBD-escape row 的 `Ag_seq` 实际是 mutation expression，不是氨基酸
序列；AlphaSeq 71,834 条和 AbCoV 1,377 条主要使用超过当前 1024 上限的 full spike。
这些记录不能靠静默截断进入模型，应先映射到 assay 实际 domain/epitope。

TCR 侧：

| source | 本地审计结果 | 下一步状态 |
|---|---|---|
| OTS paired | source train 2,102,715；含少量非 α/β locus 组合 | strict α/β benchmark-clean export 1,475,277：1,445,804 / 14,732 / 14,741 |
| PISTE | active train 221,020；同时有 CDR3β、peptide、HLA pseudosequence、正负标签 | 保留，但重做 group/unseen-epitope split |
| full TCR-pMHC | active train 57,913；full α/β + peptide + MHC/B2M | 保留，但先按 `TCR_hash`/clonotype/donor/epitope 重建 split |
| legacy `tcr` | train 145,510；其中 126,618 条有 β/αβ + epitope；全部为 positive | 从 VDJdb/MIRA/McPAS 原始字段重建，不直接复用信息丢失 Arrow |
| TDC Weber | 47,182 rows，23,595 positive / 23,587 negative；无 HLA | 候选 relation source；先去 benchmark overlap 和 label conflict |
| TEIM | 45,603 rows，基本为 VDJdb/McPAS/nCoV/STCRDab positive 汇总 | 作为 provenance/interface 补充，不按独立 45,603 再叠加 |

PISTE random-train 的 `(CDR3β, peptide, HLA_type)` 284,144 条全部唯一且无标签冲突；
一旦丢掉 HLA，只有 209,632 个 pair，并出现 13,901 个正负冲突。这说明 MHC/HLA 不能
在 canonical merge 时被抹掉。

PISTE、TDC、TEIM 和 legacy TCR 的 positive `(CDR3β, epitope)` 集合分别为
25,985 / 23,550 / 45,545 / 129,408；简单相加是 224,488，合并后只有 142,456，
重复 membership 为 82,032。TEIM 与 legacy alone exact overlap 40,912。PISTE 与
TDC 共享的 4,593 个无 HLA pair 中，1,192 个 label set 不一致。因此它们必须先按
`assay + HLA + donor + source record` 保留 provenance，再做 canonical union；不能
把各源行数直接相加，也不能多数投票覆盖冲突。

### 0.2 第一版免疫受体采样配方

在数据 adapter、split 和 renderer 修复完成后，建议先按 training plane 采样，而不是
按源行数采样：

| plane | 初始 record share | plane 内主要 views |
|---|---:|---|
| antibody pairing | 30% | H+L joint denoise；H→L；L→H |
| TCR pairing | 30% | α+β joint denoise；α→β；β→α |
| antibody recognition | 20% | Ag→H+L；Ag+H→L；Ag+L→H；H+L+Ag→relation |
| TCR recognition | 20% | pMHC→α/β；pMHC+α→β；pMHC+β→α；TCR+pMHC→relation |

当 Ab–Ag 和 pMHC split/label 质量稳定后，再评估是否调到四个 plane 各 25%。每个
plane 内应先采 view，再采 source/record；按 target-token budget 组 batch，并限制小源
的重复 exposure，避免 2,489 个结构复合物被无限循环当成数百万条独立证据。

## 1. 当前事实的判定顺序

当前训练数据事实按以下优先级判定：

1. checkpoint 内保存的 `args`；
2. `datasets.load_from_disk(...)` 返回的实际行数；
3. 当前训练 YAML 和
   `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/bioseq_grammar_v1/manifest.json`；
4. build/dedup report；
5. 历史说明文档。

Hugging Face shard 中的部分 `dataset_info.json::num_examples` 仍保留去污染前行数，
不能用它代替 `len(load_from_disk(...))`。

本次固定的训练快照为：

- checkpoint：
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_esmc300m_integrated_llada_7l_step389500/best.pt`
- checkpoint step：`389500`
- data root：
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/bioseq_grammar_v1`
- recipe：
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_grammar_v2_esmc300m_integrated_llada_7l_resume_queue012.yml`
- model type：`llada`
- global batch：`6 × 24 = 144`
- planned max steps：`500000`
- max grammar length：`2112`
- max single-chain length：`1024`
- source shuffle buffer：`0`
- within-batch exact dedup：关闭

## 2. 当前 7 源训练配方

活跃 train 总计 **16,044,966** 条；有 valid shard 的 6 源总计 **156,024**
条。source 采样概率为 `weight / 12.5`，与数据源行数无关。

| source | active train | valid | weight / share | Arrow roles | 运行时 fixed → target | 平均 target tokens |
|---|---:|---:|---:|---|---|---:|
| `oas` | 2,484,758 | 12,553 | 3.0 / 24% | heavy, light | 无 fixed；H+L 联合去噪 | 235.5 |
| `ots` | 2,085,414 | 10,619 | 3.0 / 24% | beta, alpha | 无 fixed；α+β 联合去噪 | 230.4 |
| `nanobody` | 10,922,486 | 58,311 | 2.0 / 16% | VHH | 无 fixed；VHH 去噪 | 124.7 |
| `tcr_piste` | 221,020 | 71,036 | 1.5 / 12% | HLA, epitope, CDR3β | HLA+peptide+relation → CDR3β | 17.5 |
| `tcr_pmhc_fulllength` | 57,913 | 590 | 1.5 / 12% | MHC, B2M, peptide, α, β | MHC+B2M+peptide → full α+β | 589.4 |
| `ppi` | 261,901 | 2,915 | 1.0 / 8% | protein A, protein B | A+`binding` → B | 428.4 |
| `neutralization` | 11,474 | — | 0.5 / 4% | H+L 或 VHH | **实际为普通 H+L/VHH 无条件去噪** | 220.1 |

这里的 “target tokens” 包含运行时会被 diffusion corruption/loss 覆盖的残基和结构
token。所有记录都满足当前 `2112` grammar 上限和单链 `1024 aa` 上限；没有依靠
runtime truncation。

按 record 采样份额看：

- 48% 是 OAS/OTS 免疫受体双链联合去噪；
- 24% 是 pMHC 条件下的 TCR/CDR3β 去噪；
- 8% 是通用 PPI 条件去噪；
- 20% 是 nanobody 加名义 neutralization 的无条件受体去噪。

只有 32% records 运行时带显式 partner+relation context，而且 relation 仍是 fixed
input。当前 **step389500 checkpoint** 的 antibody-antigen sequence-conditioned
records 为 0；canonical v2 虽已生成候选 Ab-Ag records，但尚未 export 或进入任何
checkpoint。

### 2.1 去污染和精确重复

当前 `train` 是相对 2026-07-08 downstream bank 去污染后的 promoted shard；
`train_prededup` 保留原数据用于回滚。这里的 “dedup” 主要表示
**train-versus-downstream decontamination**，不是任意意义上的全库内部去重。

| source | pre-decontamination | active train | removed | active exact duplicate rows |
|---|---:|---:|---:|---:|
| `oas` | 2,486,442 | 2,484,758 | 1,684 | 0 |
| `ots` | 2,102,715 | 2,085,414 | 17,301 | 0 |
| `nanobody` | 11,525,884 | 10,922,486 | 603,398 | 0 |
| `tcr_piste` | 284,144 | 221,020 | 63,124（downstream 63,117 + valid overlap 7） | 11 |
| `tcr_pmhc_fulllength` | 57,913 | 57,913 | 0（另在建库前删除旧 bank 命中 15,324） | 0 |
| `ppi` | 319,429 | 261,901 | 57,528 | 0 |
| `neutralization` | 12,346 | 11,474 | 872 | 157 |

对应报告位于：

- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/dedup/reports`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/dedup/DEDUP_COVERAGE.md`

### 2.2 计划训练中的重复采样强度

按 500,000 steps、global batch 144 的计划，模型将读取 72,000,000 条记录。

| source | 期望读取条数 | 相对 active train 的期望 passes |
|---|---:|---:|
| `oas` | 17,280,000 | 6.95× |
| `ots` | 17,280,000 | 8.29× |
| `nanobody` | 11,520,000 | 1.05× |
| `tcr_piste` | 8,640,000 | 39.09× |
| `tcr_pmhc_fulllength` | 8,640,000 | 149.19× |
| `ppi` | 5,760,000 | 21.99× |
| `neutralization` | 2,880,000 | 251.00× |

因此 weight 不是“数据量占比”。当前最小的两个源被循环数百次，而最大的 nanobody
源在完整计划中才刚好约一遍。

### 2.3 上游来源与 split

| source | 当前上游输入 | 当前 split 口径 |
|---|---|---|
| `oas` | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/oas_previous_clean/splits/cleaned_merged_data_step_clustered_{train,valid,holdout}_oas_label.csv` | 项目已有 antibody cluster split |
| `ots` | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ots_paired_clean/final/{train,valid,holdout}.csv` | 项目已有 paired-TCR cluster split |
| `nanobody` | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/nanobody_processed/step7_clean/{train,valid,holdout}.csv` | 继承上游 split；已删除 `nbbench_*` 且限制 90–160 aa |
| `tcr_piste` | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/raw/piste_tcr_epitope_hla/PISTE/data/random/{train,val}_data.csv` | PISTE author random split，不是 unseen-epitope split |
| `tcr_pmhc_fulllength` | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_pmhc_fulllength/records_{train,valid,holdout}.jsonl` | 当前为 record-random split；见 P1-A |
| `ppi` | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi/string_model_org_90_90_split` | `bernett_string_90_90_hf` train/valid；test 不进 shard |
| `neutralization` | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/raw/covabdab_neutralization/CoV-AbDab_080224.csv` | 原数据无 official split；当前全放 train |

## 3. 模型实际学到的 supervision

### 3.1 `relation` 字段不等于 relation prediction

当前 renderer 的事实是：

- OAS/OTS 虽在 Arrow 中保存 `relation=binding`，运行时不放 relation token；
- nanobody 的 `unknown` 不进入 runtime grammar；
- neutralization 的 `neutralization` 不进入 runtime grammar；
- PISTE、全长 TCR-pMHC 和 PPI 会放 relation token，但 relation token全部是
  **fixed context**，不属于 diffusion target。

活跃 PISTE 的 relation 分布为 `binding=20,192`、`nonbinding=200,828`
（约 9.1% versus 90.9%）；全长 TCR-pMHC 和 PPI 全是 `binding`。OAS/OTS
保存的 `binding`、nanobody 的 `unknown`、neutralization 的 `neutralization`
都不会成为运行时 relation token。

因此当前模型学习的是：

- 给定某个 relation，生成/去噪 partner；
- 通过上下文梯度学习 relation embedding；

而不是：

- 给定多条链，预测它们是否 binding/nonbinding；
- 给定抗体和抗原，预测 neutralization；
- 直接生成 relation token。

如果目标是“多链免疫关系学习”，下一版必须增加
`fixed chains → target relation token` 的 view，并保留正负标签；否则下游 relation
分类只能依靠表示空间中的间接信号。

### 3.2 当前 view 仍然过于单一

- OAS：只有 H+L 联合全去噪，没有 `H→L` 和 `L→H`。
- OTS：只有 α+β 联合全去噪，没有 `α→β` 和 `β→α`。
- PPI：只有 `A→B`，没有 `B→A` 或 relation prediction。
- PISTE：`binding/nonbinding` 是输入条件，目标始终是 CDR3β。
- 全长 TCR-pMHC：只有 `pMHC→α+β`。
- neutralization：没有 antigen/virus sequence，不能形成真正的多实体关系 view。

### 3.3 同 task 混批导致的 token supervision 失衡

`tcr_piste` 和 `tcr_pmhc_fulllength` 都进入 `tcr_pmhc` buffer，且 source weight
同为 1.5，所以 record 采样约 50:50。但二者平均 target tokens 分别为 17.5 和
589.4。在同一 token-normalized batch 中，全长源约占 **97.1%** 的目标 token，
PISTE 的正负关系监督被显著稀释。

当前 validation 也汇总为一个 token-weighted loss。按 valid source 权重和平均长度
估算，PISTE 虽占约 12.5% valid records，对总 validation loss denominator 的贡献
不足 1%。因此当前 best checkpoint 选择并不敏感于 PISTE relation 能力。

## 4. 必须先解决的问题

### P0-A：downstream decontamination bank 已经过期

当前 bank 生成于 2026-07-08：

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/dedup/banks/manifest.json`

但以下 canonical benchmark artifact 在 2026-07-21 才固定：

- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/mint_official`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/data/tcr_binding_nm2025`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/data/tcr_beta_public_benchmark`

只读复核得到：

- Public TCR Track-A 有 1,306 个唯一参考 CDR3β；旧 bank 只包含其中 88 个。
- 活跃 OTS/PISTE/全长 TCR 源合计命中其中 **395 / 1,306（30.2%）**：
  - OTS：1,637 条 train rows，覆盖 282 个 public references；
  - PISTE：219 条 train rows，覆盖 21 个 references；
  - 全长 TCR-pMHC：198 条 train rows，覆盖 158 个 references。
- 当前 canonical MINT Gold PPI test 与活跃 `ppi` 有 **498 个 exact pair**
  重叠，其中 test label 为正 488、负 10。
- 另有 6,547 条活跃 PPI train rows 至少包含一条与 Gold test 完全相同的链，
  共涉及 686 条唯一 test chains。

结论：

- 不能继续写“所有当前 benchmark 均已去污染”；
- 当前 Public Track-A 和 MINT Gold 结果必须标记为对 **post-bank benchmark**
  存在训练污染风险；
- 下一版训练前必须基于当前 canonical benchmark roots 重建 bank，并重新执行
  domain-specific similarity decontamination；上述 exact overlap 只是下界。

另一个问题是 CDR3 key 的表示不一致：OTS 的 `chain*_cdr3` 是去掉首尾保守锚点的
内部 loop（例如 `ASS...`），NM2025/Public artifact 保存的是完整 `CASS...F`。
因此“完整 CDR3 已出现在旧 bank”不等于 OTS 已经成功被 exact-key 过滤。直接按
两种格式归一化后复核，活跃 OTS 仍包含：

- 150 条 rows 命中 89 个 NM2025 seen references，对应 seen test 的 89 / 1,956 行；
- 12 条 rows 命中 11 个 NM2025 unseen references，对应 unseen test 的
  21 / 690 行。

PISTE 和全长 TCR-pMHC 没有 NM2025 exact CDR3 overlap。下一版 bank/schema 必须为
完整 CDR3 与内部 ANARCI loop 定义同一个 canonical key。

### P0-B：`neutralization` 不是有效的 neutralization supervision

CoV-AbDab 原始 12,346 条有 heavy sequence 的记录中：

- 6,029 条有 `Neutralising Vs`；
- 6,317 条没有正 neutralization annotation；
- 5,248 条有 `Not Neutralising Vs`；
- 2,844 条同时对不同 variant 有正、负 annotation；
- 3,913 条既没有正 annotation，也没有负 annotation。

当前 builder 没有按 virus/variant 展开，也没有读取正负关系，而是给所有记录统一写
`relation=neutralization`。随后 renderer 因为记录没有 antigen chain，又把 relation
完全省略。这个源现在只相当于额外抗体/VHH sequence pretraining。

在补齐 `(antibody, virus variant/antigen, label)` 之前，应二选一：

1. 从 relation mix 移除或改名为 `covabdab_receptor_sequences`；
2. 显式构建 antigen/variant mapping，按 `Neutralising Vs` /
   `Not Neutralising Vs` 展开正负 pair，再建立 group-aware split。

不能只修 renderer 让 `<neutralization>` 出现；那会把大量无正标注和负标注行错误地
训练成 neutralization。

### P0-C：resume 不恢复数据流，且当前没有源内 shuffle

checkpoint 只保存 model、optimizer、step 和 args，不保存：

- source RNG state；
- 每个 source iterator offset；
- shuffle buffer；
- 当前 task buffers。

每次 `--resume auto` 都从每个 rank 的 contiguous Arrow shard 开头重新读取。
当前 7L 在 step 101,300 后发生过一次有效 resume。按 source 权重的期望值估算，
step 389,500 时 nanobody：

- 名义读取约 8,974,080 条；
- 唯一覆盖约 6,640,128 条；
- 早期约 2,333,952 条被重复读取；
- 后部约 4,282,358 条尚未读取。

这些是按权重的期望估算，不是逐 batch replay log；但数据加载实现确定会发生 prefix
replay。下一版必须做到以下至少一项：

- checkpoint 数据流状态并精确恢复；
- 用 `(global_step, rank, pass)` 派生可 seek 的 deterministic permutation；
- 每个 resume 改 source seed 并启用足够大的 streaming shuffle，同时记录数据版本。

## 5. 其他数据质量与 split 问题

### P1-A：全长 TCR-pMHC split 不是 group-disjoint

builder 注释写“按 pair_id groups split”，实际 source 文件没有 `complex.id` 字段，
所有 59,093 条输出的 `metadata.pair_id` 都是空字符串，代码最终按记录随机切片。

train-versus-valid 虽然整记录 overlap 为 0，但仍有：

- 75 条相同 full β chains；
- 63 条相同 full α chains；
- 37 个相同 peptides。

下一版应使用 VDJdb 的 `TCR_hash`，并按 clonotype / donor / peptide / MHC 中与目标
评测一致的 group key 切分。

### P1-B：少量 role 和 duplicate 异常

- OAS：38 条 `antibody_heavy|antibody_heavy`。
- OTS：12 条 `tcr_beta|tcr_beta`；当前 renderer 只保留第一条 beta，第二条被丢失。
- PISTE：11 条 active exact duplicates。
- neutralization：157 条 active exact duplicates。

应在 semantic shard build 时做 schema assertion，而不是等到 runtime renderer
静默降级。

### P1-C：validation 过浅且不可分源解释

- neutralization 没有 valid shard；
- 每次 validation 只有 10 batches/rank；
- validation 是随机 weighted mixture；
- 只保存 aggregate token-weighted loss，没有稳定的 per-source/per-view 指标。

建议固定一份 versioned validation panel，并同时报告：

- per-source loss；
- per-view loss；
- relation prediction AUROC/AUPRC；
- chain-completion recovery；
- aggregate macro average，而不是只用 token micro average。

## 6. 当前 semantic Arrow 丢失的信息

所有活跃 shard 只有：

```text
chains, roles, task_type, source, split, relation, weight
```

OAS/OTS 上游实际存在的 FR/CDR、V/J gene、cluster id、pair id、species 等信息，在
semantic Arrow 中已经丢失。因此当前 cache 无法直接构造：

- CDR-specific infilling；
- framework-fixed design；
- cluster-aware sampling；
- donor/epitope/group-aware split audit；
- 同一 biological record 的多种 generation views。

PPI 上游 `score` 也没有保留。已核实当前 90/90 split 的 score 全部为 901–999，
所以统一映射为 `binding` 没有把二分类负样本翻成正样本，但丢失了 interaction
confidence。

下一版 canonical record 至少应保留：

```text
record_id
complex_id
chains[]:
  entity_id
  sequence
  role
  sequence_scope
  regions
  v_gene / d_gene / j_gene
task_type
relation
relation_direction
label
label_confidence
assay:
  assay_type
  value
  unit
  censor
  target_name
targets
available_views
split
split_group
source
source_version
provenance
```

semantic record 应与 grammar 解耦；grammar-v2 只是在 collator 中选择并渲染某个
view。

`sequence_scope` 必须区分 `full_chain`、`variable_domain`、`cdr3`、
`antigen_full_length`、`antigen_domain`、`epitope`、`mhc_full_length` 和
`hla_pseudosequence`。当前把 CDR3β、full TCRβ、HLA pseudosequence 都只存成普通
sequence，会让长度和生物语义混在一起。

## 7. 已落盘但未进入当前 7L 的数据

| source | train | valid | 内容 | 当前状态 |
|---|---:|---:|---|---|
| `tcr` | 145,510 | 8,617 | VDJdb/MIRA/McPAS 的 beta/α/antigen 混合记录 | 被 PISTE + 全长源替代，未进 7L |
| `mint_ppi` | 81,717,793 | 207,893 | STRING v12 physical binding | 未进当前 7L |
| `mint_actions` | 9,103,961 | 250,000 | STRING v11 action modes | 未进当前 7L |

`mint_actions` 的 active relation 分布为：

| relation | rows |
|---|---:|
| reaction | 5,056,856 |
| catalysis | 2,792,458 |
| ptmod | 555,629 |
| activation | 444,560 |
| inhibition | 176,465 |
| expression | 70,203 |
| binding | 7,790 |

它适合作为通用 relation-conditioned 预训练候选，但按当前 PPI renderer 加入后，
relation 仍然只是 fixed condition。若要学习 relation inference，仍需额外
`chains→relation` view 和可靠负样本。

此外：

- TCRdb2.0 原始 repertoire 约 18G，尚未转换为当前 semantic Arrow；
- SAbDab2 antigen-aware canonical JSONL 已落地，但尚未完成去污染训练 export；
- 当前 step389500 活跃 mix 没有真正的 antibody-antigen sequence pair。

## 8. 目录与存储现状

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/bioseq_grammar_v1`
总计约 455G，其中：

- Hugging Face generator cache 约 **300G**；
- `mint_ppi` 约 129G；
- `mint_actions` 约 19G；
- 活跃源目录同时保留 `train` 和 `train_prededup`。

不要在训练运行期间直接删除 `.cache` 或 `train_prededup`。整理时应先产出 checksum
和引用清单，再把数据分为：

```text
raw_immutable/
curated_semantic/
decontaminated_snapshots/
training_recipes/
evaluation_only/
archive_prededup/
rebuildable_cache/
```

现有目录名 `bioseq_grammar_v1` 保存的是 semantic Arrow，而 runtime 使用
grammar-v2。短期不应物理改名以免破坏 checkpoint/YAML 路径；长期应在 manifest
中把 `semantic_schema_version` 与 `runtime_grammar_version` 分开。

## 9. 下一版免疫受体 view 配方

先保留完整 biological record，再在 record 之上显式采样 view：

| biological record | 建议 views |
|---|---|
| OAS H/L | joint denoise；H→L；L→H |
| OTS α/β | joint denoise；α→β；β→α |
| PISTE pMHC-CDR3β | pMHC+relation→β；pMHC+β→relation |
| full TCR-pMHC | pMHC→α+β；pMHC+α→β；pMHC+β→α；all chains→relation |
| antibody-antigen | antigen→H+L；antigen+H→L；antigen+L→H；all chains→relation |

建议两级采样：

1. 先采 task/view；
2. 再在该 task/view 内采 source/record。

这样可以：

- 防止数据量最大的 source 控制全部训练；
- 防止 PISTE 与 full TCR 在同一 batch 中产生 17.5 versus 589.4 的 token 失衡；
- 单独控制 relation prediction、chain completion、joint denoising 的比例；
- 让 validation 与训练目标一一对应。

canonical v2 已完成数据 schema、adapter 与 split，但当前训练实现还不能执行这张表的
全部 view：

- `grammar_record_from_arrow(...)` 只恢复
  `chains/roles/task_type/source/split/relation/weight`，不恢复 `targets`；
- renderer 把 relation token 一律设为 fixed；
- antibody/TCR pair block 内不能按 view 把一条 receptor chain 设为 fixed。

所以“重建数据”本身不等于“模型已经学到这些方向”。新训练启动门槛还包括：
renderer 支持 per-chain/per-region fixed mask、relation target，以及每条 batch 记录
实际 `view_name`。

## 10. 推荐执行顺序

2026-08-04 状态：第 2--7 项的数据工作已经完成，并物化为 core build
`ir2exp_f7a60484c7e3a20db6a2`、pairing build `ir2pair_6a1a5b62752caccd2cd5`
与 candidate recipe `ir2recipe_1e3eac44551e88aedc6e`。所有输出 residual benchmark
cluster match 为 0，且 source valid/holdout 未复用。第 8--11 项仍是后续 runtime/
训练工作；在 rights、renderer/view 与 sampling gate 关闭前不能启动训练。

1. 冻结当前数据快照，命名为 `step389500_7src_legacy`，记录每个 shard checksum。
2. 已完成：用当前 antibody/TCR canonical benchmark roots 重建 downstream bank。
3. 已完成：报告 exact overlap，并完成 antibody/TCR/peptide/antigen similarity
   decontamination。
4. 已完成：candidate recipe 排除 nanobody、PPI/MINT、当前 neutralization 及无
   specificity bulk TCR。
5. 已完成：构建 SAbDab2 strict Ab–Ag 与 AbRank strict sequence-conditioned adapters。
6. 已完成：将 PISTE/full-TCR/VDJdb/MIRA/McPAS/IEDB 等合并为保留
   HLA/assay/provenance 的 canonical TCR specificity union，并重建 split/export。
7. 已完成：重建保留 regions/genes/sequence-scope/group/provenance 的 semantic schema。
8. 为四个 immune-receptor plane 增加显式 view sampler 和 relation-target renderer。
9. 分开 short-CDR3 与 full-chain token buckets，启用可恢复 shuffle/data cursor。
10. 建立固定、分源、分 view、分 seen/unseen-antigen 的 validation panel。
11. 只在新 manifest、decontamination report、runtime render audit 全部通过后启动下一轮训练。

当前只读盘点命令：

```bash
source /vepfs-mlp2/c20250601/251105016/miniforge3/etc/profile.d/conda.sh
conda activate pllm
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/dedup/inventory_integrated.py
```
