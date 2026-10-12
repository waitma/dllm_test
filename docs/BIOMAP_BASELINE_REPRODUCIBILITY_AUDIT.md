# BioMap：数据、论文与官方 baseline 代码审计

核查日期：2026-09-24。**可以作为 antigen＋双链抗体→ΔG 回归候选；优先复跑 A2binder 的作者原实现。不能称所有 baseline 均可复现，也不能把当前结果当作已经跑通。** 不测 VHH，不引入 FLAb2，不用自写模型补足缺失的官方 baseline。

## 1. 哪篇论文提供数据和实现

直接可用的锚点是 He et al., **De novo generation of SARS-CoV-2 antibody CDRH3 with a pre-trained generative large language model**, Nature Communications (2024)，[DOI](https://doi.org/10.1038/s41467-024-50903-y)、[全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11316817/)。其中 PALM-H3 是生成模型，**A2binder 是亲和力预测模型**；这里接入的是后者。

论文 Methods 将 BioMap 描述为公司组织的抗原–抗体亲和力预测竞赛数据；它不是该文新提出的独立 benchmark。该段 reference 40 指向 xTrimoPGLM（[arXiv:2401.06199](https://arxiv.org/abs/2401.06199)），不能据此认定该预训练论文提供了这里完整的竞赛协议。当前可验证的处理后数据发布者和回归比较协议来自 PALM。

| 材料 | 已验证的官方入口 |
|---|---|
| 数据 | [PALM BioMap CSV](https://github.com/TencentAILabHealthcare/PALM/blob/f44090b3fbe5404ae0a9d196f5429a48ef4de5c5/ProcessedData/BioMap/biomap.csv) |
| 模型 | [Code/model/bert_binding.py](https://github.com/TencentAILabHealthcare/PALM/blob/f44090b3fbe5404ae0a9d196f5429a48ef4de5c5/Code/model/bert_binding.py)，`BERTBinding_biomap_cnn` |
| 专用配置 | [bert_finetuning_er_common_biomap.json](https://github.com/TencentAILabHealthcare/PALM/blob/f44090b3fbe5404ae0a9d196f5429a48ef4de5c5/Code/config/common/bert_finetuning_er_common_biomap.json) |
| 训练和测试 | [bert_finetuning_er_main.py](https://github.com/TencentAILabHealthcare/PALM/blob/f44090b3fbe5404ae0a9d196f5429a48ef4de5c5/Code/bert_finetuning_er_main.py) |
| 划分与评分 | [base_data_loader.py](https://github.com/TencentAILabHealthcare/PALM/blob/f44090b3fbe5404ae0a9d196f5429a48ef4de5c5/Code/base/base_data_loader.py)、[metric.py](https://github.com/TencentAILabHealthcare/PALM/blob/f44090b3fbe5404ae0a9d196f5429a48ef4de5c5/Code/model/metric.py) |
| 发布权重 | [原始 Zenodo 7794583](https://zenodo.org/records/7794583)、[后续 Zenodo 17090473](https://zenodo.org/records/17090473)；内容边界见下文 |

官方仓库当前无 GitHub Release、无 tag，只有 main。已锁定作者公开代码 commit `f44090b3fbe5404ae0a9d196f5429a48ef4de5c5`，不用自行构建的替代实现。GitHub 没有 Release 不等于没有公开源码。

本地原样仓库：[palm_a2binder](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/ab_antigen_official/palm_a2binder)。使用 sparse checkout，但本轮检查的是完整 Git tree，不把未展开目录误报为缺失。

## 2. 论文里每个 baseline 怎么比较，代码缺什么

Table 1 是回归比较，指标 Pearson / Spearman，按 80/10/10 划分。下表区分“原方法代码”与“PALM 作者用于 BioMap 比较的适配代码”。只拿到前者，不能声称复现后者。

| 方法 | PALM 论文描述的比较方法 | 原方法产物 | BioMap 原实验复现状态 |
|---|---|---|---|
| **A2binder** | Heavy/Light RoFormer＋antigen ESM2-150M；三路 MF-CNN，拼接后回归；端到端微调 | PALM 仓库有专用模型、配置、数据和训练/测试入口 | **最可推进**；需获取预训练权重、验证环境/加载，处理第 3–5 节边界 |
| **AbMAP** | 使用作者发布的 `PropertyPredictorAttn` 提取抗体特征 | [rs239/abmap](https://github.com/rs239/abmap) 有真实实现及预训练模型；已读 `abmap/model.py`，确认该类存在 | PALM tree 未见其 BioMap adapter、antigen 融合方式、训练配置及对应 checkpoint；不能随意加 head |
| **AntiBERTa2** | 分别提取抗体重链与轻链特征 | [alchemab/antiberta2](https://huggingface.co/alchemab/antiberta2) 有官方配置、tokenizer 和预训练权重 | 缺 PALM BioMap 专用融合/回归实现、配置、实验 checkpoint；基模型不等于完整 baseline |
| **ESM-F** | 将 A2binder 的抗体编码器换为 ESM2-150M，保留 MF-CNN | [官方 ESM2-150M](https://huggingface.co/facebook/esm2_t30_150M_UR50D)；PALM 有 MF-CNN | 未找到作者发布的 ESM-F 专用配置/适配；自行替换路径及调整维度不能标为原实验复现 |
| **Ens-Grad** | one-hot 输入，两个卷积层＋全连接预测；借鉴原 CDR 设计方法 | [gifford-lab/antibody-2019](https://github.com/gifford-lab/antibody-2019) 有原 CNN、ensemble optimization 和原任务模型 | 原仓库是 CDR 设计实验；未找到 PALM BioMap 输入拼接、训练及评分适配 |
| **Vanilla BERT** | 三个随机初始化的 BERT 分别编码 antigen/H/L，拼接后两层 MLP；无预训练 | PALM 正文给 hidden=768、12 层、12 heads、intermediate=1536 | PALM 当前 tree 未见对应三路 baseline 配置/入口；预训练文件中出现 “vanilla BERT” 日志不证明该比较实现已发布 |

AbMAP 树锁定 `3d8ef30180d0bc2e2657ae38ae37c77228a0703b`；Ens-Grad 树锁定 `3388463f76b9f5ec506eea72687376bed535b311`。本轮读取原始源文件和树，没有运行模型。对缺失项的结论限定为已检查的发布材料，不推断作者未开展相应实验。

## 3. 权重包不是“下载后直接重现 Table 1”

已通过 HTTP Range 读取原始 `Model.zip` 的 ZIP 中央目录，未下载整个权重包、未反序列化或加载模型。

- `Heavy_roformer` / `Light_roformer`：包含预训练权重、config、vocab 和 tokenizer 文件，可作为按官方代码重新训练的初始化候选。
- `A2binder_affinity`：只含 `heavymodel` / `lightmodel` / `antigenmodel` 三个目录及各自 config/权重。**没有单独发布完整模型的 MF-CNN＋regression head checkpoint，也没有该目录下的 `model_best.pth`。** 不能仅凭这个包直接恢复原表的回归预测。
- `PALM_seq2seq` 是生成模型，不能替代 A2binder 回归 head。
- 后续 Zenodo 新增 `model_best.pth`；官方 README 称其为 SAbDab affinity fine-tuned 模型，未给出与原 BioMap 各测试 split 的对应关系。不能直接加载后在 BioMap 上评估，并宣称原始 held-out 成绩或无训练泄漏。

因此，**按官方入口从预训练模型重新训练**仍可推进；**直接恢复论文各次实验的完整 checkpoint**目前证据不足。这是两个不同复现目标。

## 4. 数据和链标注审计

输入字段为 `antibody_seq_a, antibody_seq_b, antigen_seq`，标签为 `delta_g`，另有 `pdb`。论文称数据无 nanobody，本地确认所有样本的两链与抗原均非空；这不等于已经完成全量链类型和 VH/VL 边界核验。标签为 ΔG，来源缺少逐条 assay/温度/单位注释，暂不转为 pKD。

**发现并核验了 H/L 列不一致的实例。** 官方 [issue #6](https://github.com/TencentAILabHealthcare/PALM/issues/6) 提到 1DQJ 链可能放反；API 显示 issue 已关闭但无评论，不能认为已经修复。对照 [RCSB 1DQJ](https://www.rcsb.org/structure/1DQJ) 原始 polymer entity 注释，LIGHT CHAIN 序列以 `DIVLTQSPAT…` 开头，HEAVY CHAIN 以 `EVQLQESGPS…` 开头，而 BioMap 同一 PDB 的样本存在两种相反的列顺序。核验行和计数见 [RESULTS 数据审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#biomap-audit)。本次只核实该实例，没有把序列前缀规则推广成全量可靠链分类器。

尚未改动原 CSV。保留原数据用于忠实复跑；若为我们的 paired VH/VL benchmark 校正链、裁剪可变区、去重或重切分，需要形成另一个显式命名的协议，且所有方法重新运行，不能直接借用 Table 1。

## 5. 原比较的 split 仍有未解决问题

代码读取 CSV 后先 `DataFrame.sample(frac=1)`，随后 `BaseDataLoader` 又按 seed 打乱索引；入口同时设置 NumPy 和 torch seed。**不能只复写一次 NumPy shuffle 就说恢复了作者 split**。当前 seed=3 配置、batch=8、最多 50 epochs、MSE、validation Pearson 选模、early-stop=15；实际 H/L token 长度上限 170、antigen 512。完整模型按原入口训练后加载该次 `model_best.pth` 再测试。

这是样本随机切分，没有按 antigen family、PDB complex 或抗体谱系隔离；不能作为未见抗原泛化的证据。论文 Table 1 注释有 `n=10` 统计检验，但已检查材料没有给出可关联到每次运行的完整 seed 列表、样本 ID 划分及模型文件。

本轮取得正式 **Source Data file**，其中 `Fig. S3` 提供六个模型各自的真实值和预测值。经六位小数归一比较，AbMAP、AntiBERTa2 和 A2binder 的真实标签多重集合不同；ESM-F、Ens-Grad、Vanilla BERT 与 A2binder 相同。**这些图的发布数据不能作为六个方法使用相同具体测试样本的证明。** 相同真实标签集合也不能证明样本 ID 一致，因为标签可能重复；不同集合不直接说明 Table 1 所有重复实验的运行方式。Source Data 没有相应 PDB/序列/原 CSV 行号映射，也不是十次实验的完整运行产物。

## 6. 其他用过 BioMap 的文章

- **S²ALM (Research 2025)**：沿用此回归 benchmark，但缺真正的实现和权重；详见 [S²ALM 审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/S2ALM_REPRODUCIBILITY_AUDIT.md)。不作为解决 PALM 缺失 baseline adapter 的途径。
- **AntiBinder (Briefings in Bioinformatics 2025)**：[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11744619/)、[作者仓库](https://github.com/fdu-wangfeilab/AntiBinder)。在 BioMap 上做五折分类，比较 TAGPPI、MARPPI、DNN-PPI、DeepPPISP、PIPR、GraphPPIS、AbAgIntPre、AttABseq、DeepAAI、DG-affinity、PESI。当前已审计入口抗体侧为 H-only，且 benchmark adapter 发布不完整；不等同于 paired H/L＋antigen 的 ΔG 回归，不混表。
- 找到 [Tsinghua-TianLab 2021 竞赛参赛仓库](https://github.com/Tsinghua-TianLab/Antibody-Affinity-Prediction-Contest)，但当前树只有数据接口、样例和测试文件，未发现完整预测模型及竞赛权重；它是参赛方材料，不能称主办方官方 benchmark 或用来补 PALM baseline。

## 7. 当前接入结论

**优先级：BioMap＋A2binder 原代码训练复跑。** 已具备真实模型/训练/评分源码及数据，预训练权重发布包目录也已核实；实际环境、权重下载校验和模型加载尚未完成，因此不标为 `[R]`。原数据 H/L 问题、完整 repeated-run split/checkpoint 缺失均需随结果披露。

其余五个 PALM baseline 仅保留论文值和缺失清单，等待 BioMap 专用作者 adapter；不自行构建模型填补。现有论文数字统一见 [RESULTS S²ALM/PALM 原表](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#s2alm-audit)，本轮不新增模型性能成绩。

证据目录：[biomap_audit](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_antigen_official_20260924/biomap_audit)。包括正文关联资料、补充 PDF、peer-review PDF、Source Data Excel、官方仓库 tree、issue/API 快照、ZIP 文件目录、RCSB 注释、CSV 逐行核查及 `audit_report.json`；获取哈希统一登记到 [manifest](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/_download_manifest.tsv)。

## 8. 后续首轮执行

用户随后授权先跑 BioMap。环境和初始化权重验证已完成，原代码作业已提交；当前状态见 [BioMap 官方运行状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#biomap-official-run)。第 7 节描述的是本次提交前的调研状态。仍保留原实验 checkpoint/repeated-run split 不完整、原始链顺序不一致的限制。
