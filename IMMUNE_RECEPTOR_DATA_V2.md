# AB/TCR Canonical Data v2

审计日期：2026-08-04

项目根：`/vepfs-mlp2/c20250601/251105016/project/dllm_test`

## 1. 范围与当前边界

本管线只负责下一版免疫受体训练的四个平面：paired antibody H/L、paired
TCR alpha/beta、antibody-antigen、TCR-peptide/pMHC。已经清洗的 OAS 与 OTS
继续作为 pairing core；本管线重点补齐它们之外的 specificity、affinity、
neutralization、structural-complex 和 antibody-property evidence。

以下数据不进入本管线：MINT/STRING/general-PPI、nanobody/VHH/VNAR、无
specificity 的 bulk TCR、以及只为下游结构 benchmark 服务的训练目标。历史数据和
checkpoint 不删除，但不能与本轮 AB/TCR-only 候选数据混称。

这轮工作没有改变模型、renderer、checkpoint 或 runtime training config，也没有启动
训练；只新增了引用 immutable 数据产物的 candidate recipe manifest。
`canonicalized`、`split_ready`、`technical_data_export_ready` 和 `export_ready` 是
不同状态；技术数据 gate 通过不代表权利与 runtime gate 已经允许训练。

## 2. 权威路径

- 实现：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/immune_receptor_v2`
- 构建入口：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_immune_receptor_v2.py`
- 官方源下载入口：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/download_immune_receptor_v2_sources.py`
- focused tests：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_immune_receptor_v2.py`
- 数据与报告根：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/immune_receptor_v2`
- source registry：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/immune_receptor_v2/registries/source_registry.json`
- external download manifest：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/immune_receptor_v2/registries/external_download_manifest.json`
- benchmark quarantine：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/immune_receptor_v2/registries/benchmark_quarantine.json`
- artifact manifest：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/immune_receptor_v2/registries/artifact_manifest.json`

## 3. `bioseq.v2` 记录语义

每条 canonical record 显式保存：

| 层 | 必需语义 |
|---|---|
| sequence entity | role、原始 AA sequence、scope、observed/reconstructed、V(D)J、CDR/FR、species |
| measurement | relation、raw value/unit、normalized value/unit、censor、direction、experimental/synthetic negative type、assay |
| evidence | A/B/C/D tier、是否 direct measurement、是否 aggregate/derived |
| target mapping | raw target、canonical ID/accession、construct、exact sequence/accession/name-only confidence |
| provenance | source/version/row、raw path/SHA256、study、donor、assay、PMID、DOI、original split |
| eligibility | training/evaluation/core、互斥 pack、排除原因 |
| grouping | receptor、antibody、peptide、pMHC、antigen、study、parent 的稳定 SHA256 group ID |

规范化只去空白并转大写，不做 `J -> L` 之类不可逆替换。完全相同的 measurement
identity 可以合并 provenance；不同 assay value、censor 或正负标签必须各自保留，
禁止多数投票。源数据的 train/test 只进入 `original_split` provenance，不能直接成为
本轮 split。

## 4. Source-specific 规则

### TCR

- IEDB：把 receptor export 按 assay ID 联到 T-cell assay outcome；只接受
  alphabeta、linear peptide，区分 curated/observed 与 IEDB-calculated receptor；
  experimental negative 明确标注，MHC allele-specific 与 generic MHC 分开。
- VDJdb/McPAS：保留 paired/single-chain、MHC、study、assay 与 confidence；只有
  paired receptor + 明确 MHC 的高置信记录进入 core。
- PISTE：保留 beta、peptide、HLA pseudosequence 和 released positive/negative；
  random split 仅为 provenance，derived negative 单独成 pack，不进入 core。
- MIRA：pool response 不能伪装成单 peptide label，只进入 pool-level aux。
- full-chain TCR-pMHC：Thimble/Stitchr 结果是 VDJdb/McPAS 的 reconstructed view，
  不是新增独立 evidence，也不进入 core。

### Antibody

- AbRank：只有合法 H/L/Ag sequence 加 direct Kd/IC50 才进入 exact core；mutation
  expression/name-only target 留在 aux，长 antigen 原样保留而不截断。
- Kothiwal：保留 SPR Kd 或 cell-display EC50 的原始 nM 与归一化 M 值。
- CATNAP：保留每个独立 IC50/IC80、`<`/`>` censor 和 `ug/mL`；H/L 通过
  Immuno DB ID 联结，virus 通过 accession 联到 Env。该 Env 是 released associated
  accession，不保证等于 assay clone，因此单列 `antibody_antigen_exact_accession`
  与 B-tier，绝不阈值化为正负标签。
- SAbDab2：只读官方 antigen-aware `abag_split.csv`；要求 paired VH/VL，排除
  VHH/VNAR，且只接受 resolved protein/peptide antigen。单 antigen polymer 可进
  exact core；多 polymer complex 按 component 保留在 aux，不能把一个 component
  冒充完整复合物。官方 `ab_ag_split` 与 cluster 只作 provenance。
- FLAb properties：只保留 paired H/L 的 intrinsic property；不把 antigen-free
  property 行冒充 binding。
- CoV-AbDab：target name/variant-only 数据保留为 proxy pack，不进入 exact core。

## 5. Split 与 benchmark quarantine

所有 split 都用 seed 42 的稳定 SHA256 tie-break，不依赖 Python `hash()` 或输入行序。
group 连通分量优先于 90/5/5 比例；巨型 component 导致比例不可达时必须报告实际
deviation，不能破坏 disjointness 来凑比例。

TCR 生成 receptor-disjoint、unseen-peptide、unseen-pMHC、study-holdout 四套
manifest；antibody-antigen 生成 antibody-disjoint、antigen-disjoint、joint-hard
三套；property 另按 parent/study 分组。

quarantine 冻结实际使用的 TCR benchmark，以及 OAS pairing holdout、SAbDab/SAb23
CDR infilling、humanization mmCIF H/L 和 FLAb task 数据。审计必须分开报告：任意
exact entity、exact receptor-peptide、exact paired-TCR-peptide、exact
receptor-pMHC 与单编辑邻居；“共享 peptide”不能写成“exact interaction leakage”。

## 6. 重建命令

```bash
source /vepfs-mlp2/c20250601/251105016/miniforge3/etc/profile.d/conda.sh
conda activate pllm

python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/download_immune_receptor_v2_sources.py --sources iedb,catnap,sabdab2
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_immune_receptor_v2.py inventory
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_immune_receptor_v2.py build-tcr
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_immune_receptor_v2.py build-antibody
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_immune_receptor_v2.py finalize
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_immune_receptor_v2.py export --threads 64 --negative-ratio 1.0
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_immune_receptor_v2.py pairing-export --threads 64
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_immune_receptor_v2.py recipe
```

## 7. 2026-08-03 执行结果

### 官方 artifact 与 inventory

- 14/14 个 registry source 均为 `raw_verified`，包括 OAS/OTS 三个既有 split。
- IEDB `tcell_full_v3.zip`：44,948,923 bytes，SHA256
  `77a5404d475dd66f89f27071142ba029c77fc830d242b8dbcdc8fc4cae84bcf5`。
- IEDB `tcr_full_v3.zip`：8,990,996 bytes，SHA256
  `079b93914d282a303c5ba645a21c873f4e29a2239943ee8b996d472ae2bf1212`。
- CATNAP snapshot 固定为 `2026-08-01`，六个官方文件共 20,598,207 bytes；
  每个文件的 SHA256 见 external download manifest。
- SAbDab2 0.1.0 `splits.tar.gz`：876,381,859 bytes，官方 MD5
  `0dbb4cc499e9eb77f14008b232f2c38c`，SHA256
  `54e3b9cdae5f5ff57a4210866f87e86686203556eea6a033200ff55c0f81e2b7`。
  完整归档含 `ab_split.csv`、`abag_split.csv` 及两个 single-domain 版本；损坏旧件
  作为 `.invalid.5bc49f5d1e96` 保留。

IEDB 的旧损坏 ZIP 也作为 `.invalid.690b94b12e6e` 保留。正式源目录没有残留
`.partial`、`.aria2` 或 `.tmp` 文件。

### TCR canonical union

六个 adapter 输入为 PISTE 358,008、VDJdb 126,137、IEDB 304,662、McPAS
13,169、MIRA 160,383、full-chain derived 59,093，共 1,021,452 条。仅合并完全相同
measurement 后得到 **922,479** 条，union SHA256 为
`41b74432ec71baafd9fd1600706cac774ceab0732dbf6e41ee32c9d07da52b0e`。

| TCR pack | records | 用途 |
|---|---:|---|
| `tcr_core` | 41,489 | paired/high-confidence specificity split 候选 |
| `tcr_aux_beta` | 270,436 | single-chain/beta-only |
| `tcr_aux_low_confidence` | 71,602 | paired 但 MHC/organism/evidence 不足 |
| `tcr_aux_computational_sequence` | 33 | IEDB-calculated receptor |
| `tcr_aux_pool` | 154,518 | MIRA peptide-pool response |
| `tcr_fullchain_derived` | 59,093 | reconstructed view，非独立 evidence |
| `tcr_synthetic_negative` | 325,308 | PISTE released derived negatives |

TCR domain error 为 0，record ID 全部唯一。保留 97 个同 biological key 的正负
冲突；若忽略 MHC，receptor-peptide label conflict 为 18,984，进一步证明不能删除
HLA 后多数投票。

### Antibody canonical union

六个 adapter 输入为 AbRank 169,237、Kothiwal 709、CATNAP 191,762、SAbDab2
6,412、FLAb properties 20,001、CoV-AbDab proxy 84,801，共 472,922 条。exact
measurement merge 后得到 **470,949** 条，union SHA256 为
`3c7b5d679bdd98d7571e16b3821b1e918da7b92d3333838b136f58fed7cc42f0`。

| antibody pack | records | 用途 |
|---|---:|---|
| `antibody_antigen_exact` | 118,320 | exact released antigen sequence |
| `antibody_antigen_exact_accession` | 189,947 | CATNAP associated Env accession |
| `antibody_antigen_aux` | 58,038 | target/measurement 或多组分语义不足 |
| `antibody_antigen_proxy` | 84,657 | name/variant-only，不进 core |
| six intrinsic-property packs | 19,987 | paired H/L property measurements；其中 19,981 条满足 export-core |

interaction core 为 308,267 条；包含 19,981 条 property export-core 后总 core 为
328,248。另 6 条 polyreactivity 记录只有 released normalized fitness，保留在
canonical union，但不作为 core 监督。SAbDab2
具体为 3,363 个 single-polymer core 与 3,049 个 multi-component aux component；
official provenance 为 train 5,080 / test 1,332。CATNAP 原始 191,762 条由 IC50
131,085 与 IC80 60,677 组成，censor 为 `>` 78,055、`<` 1,139、`=` 112,568，
覆盖 692 个 paired antibodies 与 2,578 个 Env sequence。

target registry 含 14,216 个 canonical target key、5,669 个唯一 antigen entity；
record-level mapping confidence 为 exact-sequence 155,858、exact-accession 189,947、
name-only 105,157。antibody domain error 为 0，record ID 全部唯一。

### Split manifests

九套 manifest 全部 `audit.passed=true`：

| protocol | eligible | train / valid / test | largest component |
|---|---:|---:|---:|
| TCR receptor-disjoint | 41,489 | 37,340 / 2,075 / 2,074 | 0.31% |
| TCR unseen-peptide | 41,489 | 37,340 / 2,075 / 2,074 | 27.76% |
| TCR unseen-pMHC | 41,489 | 37,340 / 2,075 / 2,074 | 27.76% |
| TCR study-holdout | 41,467 | 37,320 / 2,074 / 2,073 | 38.74% |
| antibody-disjoint | 308,267 | 277,440 / 15,414 / 15,413 | 3.63% |
| antigen-disjoint | 308,267 | 277,440 / 15,414 / 15,413 | 23.72% |
| antibody-antigen joint-hard | 308,267 | 302,452 / 2,908 / 2,907 | **74.32%** |
| property parent-disjoint | 19,987 | 18,160 / 910 / 917 | 54.82% |
| property study-holdout | 19,987 | 18,160 / 910 / 917 | 54.82% |

joint-hard 的巨型连通分量使目标 90/5/5 在数学上不可达，实际约
98.11/0.94/0.94。该 manifest 可用于严格双侧 disjoint diagnostic，但不应作为唯一
主 split；不能拆 component 来美化比例。

### 修复后的 benchmark overlap

quarantine 现覆盖 16 个实际根、142 个文件（129 eval、4 all-fold、9 train），bank
含 antibody heavy 14,883、light 10,596、TCR alpha 25,629、beta 92,811、peptide
449 个唯一序列。此前 antibody overlap=0 是漏扫抗体 benchmark 的错误结果，已作废。

| audit | matched canonical records |
|---|---:|
| TCR any exact entity | 613,536 |
| TCR exact receptor entity | 427,383 |
| TCR exact peptide entity | 528,976 |
| TCR exact receptor-peptide | 187,128 |
| TCR exact paired-alpha/beta-peptide | 16,081 |
| TCR exact receptor-pMHC | 16,975 |
| TCR single-edit entity | 221,812 |
| antibody exact receptor entity | 104,538 |

这些是 canonical union 的预导出命中数，不是最终删除行数：一个 record 可以同时命中
多个类别，且 derived/source-merged provenance 必须在生成 export 时统一处理。本阶段
的 exact audit 不能单独解释为“无近邻泄漏”；完整的 receptor/peptide/antigen cluster
过滤及最终删除数见第 8 节。

artifact manifest 共冻结 23 个 canonical/split 文件、7,739,497,624 bytes，并记录
每个文件的 SHA256。最终状态为 `canonicalized=true`、`split_ready=true`、
`export_ready=false`、`training_started=false`。

## 8. 2026-08-04 cluster 去污染与 immutable technical export

exact blocklist、MMseqs2 cluster-level near-neighbor decontamination、train-only
negative 生成与 immutable artifact manifest 已完成。MMseqs2 固定为
`/vepfs-mlp2/c20250601/251105016/conda/envs/flow/bin/mmseqs` 18.8cc5c，连通分量
模式为 `--cluster-mode 1`。CDR3 阈值为 antibody 70% identity/80% coverage、TCR
80%/80%；full/variable chain 为 95%/80%；peptide 为 80%/90%；antigen 为
80%/80%。

### Recognition/property core export

core build 为 `ir2exp_f7a60484c7e3a20db6a2`，报告位于
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/immune_receptor_v2/exports/ir2exp_f7a60484c7e3a20db6a2/export_report.json`。
artifact manifest 冻结 80 个文件、2,578,231,408 bytes，逐文件记录 SHA256。

| protocol | input | benchmark blocked | kept | train / valid / test | train-only synthetic negative |
|---|---:|---:|---:|---:|---:|
| TCR receptor-cluster / seen peptide（primary） | 41,489 | 37,060 | 4,429 | 3,986 / 222 / 221 | 3,970 |
| TCR unseen peptide | 41,489 | 39,129 | 2,360 | 2,124 / 118 / 118 | 2,108 |
| TCR unseen pMHC | 41,489 | 39,129 | 2,360 | 2,124 / 118 / 118 | 2,108 |
| antibody-cluster disjoint（primary） | 308,267 | 144,936 | 163,331 | 146,998 / 8,167 / 8,166 | 0 |
| antigen-cluster disjoint | 308,267 | 144,936 | 163,331 | 146,998 / 8,167 / 8,166 | 0 |
| antibody property parent（supplemental） | 19,981 | 6,395 | 13,586 | 12,227 / 722 / 637 | 0 |

九套 export protocol 的 split audit 全部通过，导出后 residual benchmark cluster
match 均为 0；所有 TCR negatives 只由各自 train positives 做确定性 derangement，
与全量已知 positives 的 collision 为 0，validation/test synthetic negatives 为 0。
antibody+antigen joint-hard 在去污染后仍有 99.07% 的巨型连通分量，仅保留为严格
双侧 diagnostic，不作为 primary protocol。

### OAS/OTS strict pairing export

pairing build 为 `ir2pair_6a1a5b62752caccd2cd5`，报告位于
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/immune_receptor_v2/exports/ir2pair_6a1a5b62752caccd2cd5/pairing_export_report.json`。
artifact manifest 冻结 28 个文件、5,409,708,251 bytes。旧 source valid/holdout
没有复用；只从 source train 按上游 pair group 做稳定 SHA256 98/1/1 重分，并将组内
任意 exact/near benchmark 命中传播到整组。

| source | input | blocked | kept | train / valid / test | residual benchmark cluster |
|---|---:|---:|---:|---:|---:|
| OAS strict H/L | 2,486,442 | 987,593（39.72%） | 1,498,849 | 1,468,754 / 15,295 / 14,800 | 0 |
| OTS strict alpha/beta | 2,102,715 | 627,438（29.84%） | 1,475,277 | 1,445,804 / 14,732 / 14,741 | 0 |

OAS 的高删除率主要由常见 benchmark light-chain/CDR cluster 驱动；这是有意的严格
H/L-clean 策略。OTS 报告中的 261,753 条
`missing_or_invalid_required_sequence_records` 实际由 261,738 条 B-D、13 条 B-B、
1 条 A-A、1 条 D-D 组成，是 active alpha/beta scope exclusion，不是 parser 丢链；
这些记录及其上游 group 已进入 blocked 集合，不会写入三份 pairing CSV。

### Candidate recipe 与剩余 gate

候选 recipe 为 `ir2recipe_1e3eac44551e88aedc6e`，manifest 位于
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/immune_receptor_v2/exports/ir2recipe_1e3eac44551e88aedc6e/recipe_manifest.json`，
SHA256 为 `2119da4bc0f36cefd805aea4a409508399734b2d19694bd55423db6662e166be`。
primary/supplemental real records 合计 train 3,077,769、valid 39,138、test 38,565；
另有 3,970 条低置信 train-only TCR synthetic negatives，未并入 real-record 总数。

当前 gate 必须逐字区分：`technical_data_export_ready=true`，但
`rights_review_complete=false`、`runtime_views_ready=false`、
`sampling_weights_frozen=false`，所以 `export_ready=false`、
`training_ready=false`、`training_started=false`。剩余工作只有三类：完成 SAbDab2/
CATNAP training-use/redistribution rights review；实现并测试 per-chain/per-region fixed
mask 与 relation-target rendering；冻结四平面 sampling weight 和 token-budget
batching。数据技术阶段完成不等于已经允许训练，不能手工抬高 gate。
