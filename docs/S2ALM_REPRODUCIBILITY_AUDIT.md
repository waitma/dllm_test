# S²ALM：任务适配性与原代码复现审计

核查日期：2026-09-24。结论：**亲和力任务的设计相关，但当前公开产物不足以将 S²ALM 纳入可执行的官方 baseline。其沿用的 BioMap / A2binder benchmark 更值得继续核验。** 不自写 S²ALM 或替换其 checkpoint；FLAb2 仍排除，MAGE 仍保留，不测 VHH。

原论文：[Research 2025，10.34133/research.0721](https://doi.org/10.34133/research.0721)。本轮读取了正文、Notes S1–S6、两张补充表，并核查仓库历史和模型链接。没有运行 S²ALM、PALM 或 A2binder 模型。

## 1. 三类任务不能混称

| 论文任务 | 实际条件与目标 | 对我们任务的适配性 |
|---|---|---|
| Antigen binding capacity | 固定 HER2，抗体序列接分类 head；Table 1 比较 ESM-1b、MSA-1b、AbLang-H/L、AntiBERTa、EATLM | 没有变化的 antigen 输入，不进入本轮 antigen＋VH/VL 分类主任务 |
| Antigen–antibody affinity | S²ALM 提取抗体特征，ESM-2 提取 antigen，拼接后 MLP 回归；14H、14L、BioMap；按 80/10/10 划分，Pearson/Spearman；Table 2 比较 Vanilla BERT、Ens-Grad、AbMAP、AntiBERTa2、ESM-F、A2binder | **明确使用 antigen，任务设计相关**；缺少实现，H/L 的具体编码/融合、数据划分和选模无法从代码核验 |
| Computational CDR design | 掩蔽抗体 1D 与 3Di 序列的 CDR-H3；比较 ProtBERT、AbLang-H、ReprogBERT；另用 ProGen2 筛选、AF3 做结构代理评价 | CDR 补全，不是 antigen→完整配对 VH/VL；AF3 分数不是实验结合结果；混合数据库不可直接全量执行 |

以上依据[原论文正文及 Fig. 4](https://pmc.ncbi.nlm.nih.gov/articles/PMC12364524/)。结构预训练本身也不证明每项下游任务都输入结构；具体输入需作者代码确认。

## 2. 代码与权重实查

| 核查对象 | 实测结果 | 判定 |
|---|---|---|
| 论文指向的 `KDurant-123/S2ALM` | GitHub 页面/API 均返回 404；`git ls-remote` 返回 Repository not found | 当前不可获取，不能推断永久删除或未曾发布 |
| 可搜索到的 `yashasdevasurmutt/S2ALM` | 锁定 `4ffd4566acd8ad9d2afab67a49e2843150cca789`；当前 tree 只有 README、LICENSE、5 张图片 | 没有模型、训练、推理、数据或评测入口 |
| 该仓库历史及发布 | 9 个提交均是 README/图片变动；1 个 main 分支；无 tag、无 GitHub Release | 没有在这些已核查版本里找到可运行实现 |
| README 环境说明 | 要求 `requirements.txt`，但文件不在 tree；环境名称为 ProtET | 不能据此完成 S²ALM 安装 |
| README 的 checkpoint 链接 | 标签明确写 ProtET；Zenodo 15242494 的标题为 **Multi-Modal CLIP-Informed Protein Editing** | 不能当作 S²ALM 权重加载 |
| Hugging Face 名称搜索 | models/datasets 搜索 S2ALM 返回空列表 | 本轮未找到；名称搜索不能证明所有平台均无产物 |

仓库内容可直接查看：[锁定快照](https://github.com/yashasdevasurmutt/S2ALM/tree/4ffd4566acd8ad9d2afab67a49e2843150cca789)。该 README 自称 official，提交作者名均为 KDurant-123；但是当前 ownership/迁移关系没有得到进一步确认。**无论是否为原项目迁移，该快照均不具备可执行代码。** 错配模型链接的证据：[Zenodo 15242494](https://zenodo.org/records/15242494)。

缺失项包括：真实 S²ALM 模型及 tokenizer/config、预训练权重、下游回归 head/checkpoint、三套实际 split、训练与选模配置、baseline 适配器、评测脚本。不能以 ESM-2 或 ProtET 权重代替，也不能自行拼一个 MLP 后标为作者原方法。

## 3. Benchmark 可以追踪，但标签和输入要逐项限定

S²ALM 的这些数据并非全部新建；其亲和力实验沿用 [PALM/A2binder 论文](https://www.nature.com/articles/s41467-024-50903-y) 的任务。我们继续核验了该作者的[原仓库](https://github.com/TencentAILabHealthcare/PALM)，锁定 `f44090b3fbe5404ae0a9d196f5429a48ef4de5c5`。

| 数据 | 原始标签与公开代码 | 当前判断 |
|---|---|---|
| 14H / 14L | [原数据论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9606274/)说明 AlphaSeq 的 Pred_affinity 是实验校准所得 log 尺度 nM 解离常数估计；靶点固定 HR2 peptide，分别只改变 H/L | 可以研究固定靶点的局部亲和力回归；不作为未见抗原泛化。不是独立 SPR/BLI KD 集合 |
| 14H / 14L 原 A2binder 入口 | 已读 `bert_finetuning_er_common_ll_h_cdr.json` / `ll_l_cdr.json`：使用变化的单链＋其三个 CDR；模型为 `BERTBinding_4_input_cnn` | 该原入口不显式输入 antigen 或另一恒定链，不能将其描述为三路条件 baseline |
| BioMap | 公开 CSV 字段为 `pdb, antibody_seq_a, antibody_seq_b, antigen_seq, delta_g`；官方配置明确读取两链＋antigen，目标 `delta_g` | **更贴合输入要求的回归候选**，但目标是 ΔG，不能未经单位/温度/来源核验当成实验 pKD |

BioMap 上游论文称无 nanobody；本地字段审计确认两条抗体链均非空，但部分链含较长结构片段，尚未验证全都恰好是 VH/VL 边界。抗原表示亦可能为结构相关片段。裁剪链、过滤或重切分将构成新协议，不能无说明对比原表数字。

数据行数、字段完整性和原论文数字统一放在 [RESULTS S²ALM 审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#s2alm-audit)。本轮没有生成新 train/test split，没有对 BioMap 做 pKD 转换。

## 4. 原 baseline 是怎样比较的

S²ALM Table 2 的六个 baseline 行，与 PALM/A2binder Table 1 的同名行**全部均值及括号内数字逐项一致**，核对结果已保存。这只说明报告值一致，不能据此断言作者未运行，也不能证明两篇论文使用了相同的具体测试样本。

所以需要分开保留：

- S²ALM 的论文值是 `[P]`，不标本地复现 `[R]`；由于测试索引和代码不可得，也不能据这些数字宣称我们的结果与其严格同协议。
- 要运行 A2binder，应直接追踪 PALM 作者原实现，不从 S²ALM 方法描述自行重建。
- 要运行其他六个方法，仍需逐项确认 benchmark 作者的适配和权重；拿到各自基模型仓库不等于拿到本表的实验实现。

已读到的 BioMap 原 A2binder 配置：两链＋抗原，MSE 训练，训练集随机划分而非 antigen-family 留出；seed=3，80/10/10，batch=8，最多50 epoch，按 validation Pearson 选 best checkpoint，early-stop=15。`bert_finetuning_er_main.py` 随后加载 `model_best.pth` 测试。上述是当前官方代码默认值，不能自动当作 S²ALM 的训练配置。

源代码：[BioMap 配置](https://github.com/TencentAILabHealthcare/PALM/blob/f44090b3fbe5404ae0a9d196f5429a48ef4de5c5/Code/config/common/bert_finetuning_er_common_biomap.json)、[原划分实现](https://github.com/TencentAILabHealthcare/PALM/blob/f44090b3fbe5404ae0a9d196f5429a48ef4de5c5/Code/base/base_data_loader.py)、[训练/测试入口](https://github.com/TencentAILabHealthcare/PALM/blob/f44090b3fbe5404ae0a9d196f5429a48ef4de5c5/Code/bert_finetuning_er_main.py)。

PALM 作者提供 [Zenodo 7794583](https://zenodo.org/records/7794583) 的 `Model.zip`，以及[更新记录 17090473](https://zenodo.org/records/17090473)的额外 `model_best.pth`。本轮只核查元数据，**尚未下载/加载这些权重，未验证其分别对应哪个具体 split 或标签任务**；不能提前写成 A2binder 已跑通。

## 5. 本轮决定与后续门槛

| 对象 | 决定 |
|---|---|
| S²ALM 作为可执行 baseline | 暂不纳入。需作者补齐真正的实现、checkpoint 与具体评测数据/划分 |
| S²ALM 作为文献参照 | 保留 `[P]` 与不可复现原因，不能作为已跑通项目 |
| S²ALM 的固定 HER2 分类、CDR 补全 | 不用来补本轮缺少的 antigen＋VH/VL 分类和 antigen→整对生成 |
| BioMap / A2binder | 保留为输入契合度更高的原代码回归候选；先核验权重、链边界、ΔG 来源和 split，不宣称已补齐实验 pKD |
| 14H / 14L | 仅作固定靶点/单链变化的相邻任务，不优先于显式抗原条件主任务 |
| MAGE / FLAb2 / VHH | MAGE 保留；FLAb2 已排除；VHH 不测 |

## 6. 原始证据

- [审计 JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_antigen_official_20260924/s2alm_audit/audit_report.json)：请求结果、仓库 tree/commit、权重链接身份、数据字段与 SHA256。
- [正文原 XML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_antigen_official_20260924/s2alm_audit/europepmc.txt)、[补充 PDF](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_antigen_official_20260924/s2alm_audit/supplement.pdf)：经 Europe PMC 获取，未据方法描述重实现模型。
- [原表转录及跨论文核对](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_antigen_official_20260924/s2alm_audit/paper_tables.json)。
- [逐方法台账](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/ab_antigen/comparisons.tsv)：S²ALM 各任务标为 paper-only / original implementation unavailable，不进入默认 probe。

## 7. 后续 BioMap 深化核查

后续已进一步检查 PALM 原权重包目录、Source Data 与 BioMap 链标注，更新证据见 [BioMap 官方 baseline 审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/BIOMAP_BASELINE_REPRODUCIBILITY_AUDIT.md)。前文“仅查看权重元数据”为早期快照；当前仍未下载完整权重或运行模型。
