# v7 的 264 项角色与观测范围核验

更新：2026-09-30，约 20:55–21:11 UTC。**184 项获得明确的角色或实际观测范围解释，80 项仍有注释缺口；这不等于 184 项已经完成近邻保护，也没有清除真实训练门禁。** 原 prepared 数据、benchmark、gate 和 certificate 全部未改。共享根 MD 由主会话同步。

## 产物与运行方式

- 只读程序：[audit_v7_observation_roles.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/audit_v7_observation_roles.py)。读取冻结 registry、原 264 项及逐行 source provenance；验证六份 test CSV 的 SHA、原文件/行/字段重新得到同一 sequence SHA。只写独立 evidence 目录。
- [report.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/role_resolution/report.json)：264 项逐条来源、角色、判定、已观测 loop 的位置/长度/SHA。报告不输出长序列。
- [numbering.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/role_resolution/numbering.json)、[reference_numbering.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/role_resolution/reference_numbering.json)：264 个输入和两个原始参考的逐位 IMGT→输入索引，保存 ANARCI 源码/HMM 文件 SHA。没有降低 ANARCI threshold。
- [protection_rules_proposal.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/role_resolution/protection_rules_proposal.json)：下一版 export 的可实现保护规则，状态 `proposed_not_applied`。
- [测试](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/data/test_v7_observation_roles.py)：15 项 CPU 逻辑测试通过（0.09 秒）；覆盖编号结束索引、编号不匹配、角色冲突、未编号尾巴、连续框架边界、源 gap/native 对齐、部分 loop、源构造冲突和 EMBL 元数据解析。

先激活 pllm，再运行：

```bash
source /vepfs-mlp2/c20250601/251105016/miniforge3/etc/profile.d/conda.sh
conda activate pllm
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
python scripts/data/audit_v7_observation_roles.py
python -m pytest -q scripts/tests/data/test_v7_observation_roles.py
```

默认使用已保存、与冻结输入绑定的编号结果。需要重新编号时，使用已经安装 ANARCI 的 flow 解释器运行同一脚本并加 `--refresh-numbering`。本次 264 项编号实际 8.04 秒，两个 reference 0.088 秒，4 CPU；没有下载模型。ANARCI 内部正常输出部分物种回退提示，但使用原默认阈值；不能把 no-domain 理解为已证明输入不含 H3。

已读安装源码发现：ANARCI 的函数说明称 end exclusive，但本地 `schemes.py` 实际把 `end_index` 设为最后一个输入索引。程序对每个 domain 检查去 gap 编号残基是否与原输入切片**精确相等**，自行验证 exclusive/inclusive，只有唯一匹配才给出原序列坐标。不能仅依赖文档的端点约定。

## 逐类判定

| 独立序列数 | 证据与判定 | 仍需执行的保护 |
|---:|---|---|
| 133 | 作者 label3/4 明确是 V-lambda/V-kappa；71 L、62 K 与新 ANARCI 一致，全部有可定位 L3 | heavy-H3 不适用，但必须新增 light 单链及 L3 保护，保留旧 heavy bank 并集，直到角色修正后的独立复扫通过 |
| 35 | 作者 label2 明确是 shark VNAR；通用 ANARCI 仍无 domain | 角色已确定；VNAR loop 注释仍 pending。完整实际输入及近邻保护必须保留，不能写成“没有 CDR3” |
| 48 | polyreaction 的 `CDR3_nogaps` 为空；编号精确覆盖全部输入；H3 编号范围没有残基，编号在 H3 前后的原输入坐标直接相邻 | 可说明**当前观测输入没有 H3 残基**。保持整个输入、已验证的其他 loop；不推断天然分子不存在 H3，不创建空 loop 键 |
| 1 | polyreaction 的另一个输入完全被编号，最后一个观测 IMGT 位置为 81，无未编号尾巴 | 明确是只覆盖 H3 之前区域的观测片段；按片段保护，不能填造后续残基 |
| 2 | AJ237302 的任务 native/input；官方原参考支持只观测到完整 H3 的前 2 个残基，详见下节 | 保留整个观测 native/input 和“部分 H3”元数据；不能当作完整 2-AA H3 |
| 26 | H domain 已识别但有未编号尾巴：hIL6 18、SARS-CoV-2 7、polyreaction 1 | 编号无法判定这些尾巴的区域；继续 pending，不靠长度、末端字符或框架外观猜 H3 |
| 17 | hIL6 12、SARS-CoV-2 5 无编号 domain | 作者字段提供 VHH 角色，但没有逐条完整性/region 证据；继续 pending |
| 2 | AM939711 的任务 native/input 超出所引用 germline V-segment 原参考，且 mask 位置与被称作 H3 的追加片段不同 | 源构造/标注冲突，继续 pending；原 benchmark 不修改 |

上述 184=133+48+1+2 是“角色/观测解释有证据”，80=35+26+17+2 是真实缺口。不要把它改写成“184 clean / 80 contaminated”。编号/角色判定和训练库去重是两个不同的证据阶段。

48 项两侧最后/最先编号位置并非全部都是 IMGT104/118：存在框架位置缺失；结论依赖的是**零 H3 编号残基、前后实际输入索引相邻、完整输入被精确映射及源 CDR3 为空**的联合证据，不是凭两个字母锚点猜测。

133 light / 35 VNAR 的作者类别依据：[本地 nanobody_type README](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/nanobody_raw/nbbench/hf_data/nanobody_type/README.md:25)。它是合法多分类任务，不是“轻链负例”。原目录中的重链银行收录它们，并不能替代角色正确的 light 训练键比较。

## 两个短 CDR3 记录的原始参考

本轮从官方 ENA 下载两个 accession 的原始记录，只用于核验元数据和已有 CDS translation，没有翻译/生成新序列。来源 URL、时间、完整文件 SHA 保存在 [fetch_manifest.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/role_resolution/source_references/fetch_manifest.json)。原始官方文件包含原记录内容；摘要报告只输出长度、SHA、坐标和相关元数据。这些新增参考尚未加入旧去重 registry，不能声称旧 certificate 已覆盖新增引用范围。

**AJ237302（CDRInfilling/VRClassification 零基行 537）**：官方 CDS translation 长 126；任务 `label` 长 94，是该参考从索引 0 开始的精确前缀。对 126-AA 参考编号得到 H3 长 26，任务实际 native 只含其中索引 92、93 两个残基；来源 CDR3 字段和 mask `[92,94)` 与这两个已观测残基一致。因此是**部分 H3 观测**，不是完整 H3 天生只有 2 AA。对应 task native SHA `41af006e…`，去 gap input SHA `095f891f…`。[ENA 原始记录](https://www.ebi.ac.uk/ena/browser/view/AJ237302)

**AM939711（零基行 927）**：官方记录明确标注 germline 和 `V_segment`，已有 CDS translation 长 96；任务 native 长 98，其前 96 残基与参考完全一致，额外尾部 2 残基恰等于源 CDR3 字段。但是 gap 实际位于 `[0,2)`，另两个 gap 才分别对应 CDR1/CDR2；相同 2-mer 在 native 开头和追加尾部均出现。不能将源字段、无序 substring 搜索或 `min_len=2` 视为可靠的 H3 覆盖修复。对应 native SHA `7fb52fd5…`，input SHA `4db1a24f…`。[ENA 原始记录](https://www.ebi.ac.uk/ena/browser/view/AM939711)

也核对了作者当前 hIL6/SARS-CoV-2 dataset cards：其字段说明是 VHH amino-acid sequence，没有逐记录 region 位置、原 clone reference 或完整性保证；不足以证明剩余 43 个 H/no-domain 记录的 H3 是否被观测。CDRInfilling card 的字段说明与本地冻结 CSV 实际字段也不完全一致，因此本轮实际 gap 判定以冻结 CSV 和官方任务 reader 为准，不套用 card 中另一种格式。[hIL6 card](https://huggingface.co/datasets/ZYMScott/hIL6/blob/main/README.md)、[SARS-CoV-2 card](https://huggingface.co/datasets/ZYMScott/SARS-CoV-2/blob/main/README.md)、[CDRInfilling card](https://huggingface.co/datasets/ZYMScott/CDRInfilling/blob/main/README.md)

## 下一版保护实现要求

1. **所有角色继续保护整个实际输入。** 对 heavy/light/VHH 训练键做 normalized exact 并集；继续原 H–L pair 检查。不能删除原 heavy 银行中的 light/VNAR 项后，只添加一个 N/A 标记。
2. **扩展完整链近邻规则。** 在已有 heavy identity≥95%、query/target coverage≥80% 之外，将 light/VNAR 及旧数据中可能错放角色的键一并纳入。记录比较角色、对齐坐标和覆盖；保留原规则直至联合审计通过。
3. **对已证实片段及完整性未知输入增加片段筛查。** 用整个观测片段作 query，identity≥95%、query coverage≥80%，额外检查可匹配完整受体链的局部对齐，不要求片段覆盖完整 target 的 80%。这是附加保护，原双向 coverage 检查仍保留。单纯复用双向 80% 会漏掉 72-AA 输入在 120-AA 训练链里的暴露。
4. **保护可信、角色正确的已知 loop。** L3/H3/VNAR loop 在有适用注释与坐标时，采用既有 loop identity≥80%、双向 coverage≥80% 规则；保存来源、链型、输入 SHA 和原索引。没有编号时不能制造 core；对实际为空或未观测的 H3 不创建空键。
5. **部分 2-mer 不冒充完整 H3。** AJ237302 的两个已观测 loop 残基记录为 incomplete，完整观测 input/native 必须过滤。直接全库删除含同一 2-mer 的链不代表合理的完整 H3 同源审计。若要将 ENA 全参考及其完整 H3 加入保护，必须作为显式新增 scope，绑定新引用输入并重新做对应近邻，不能回填旧证书。
6. **剩余 80 项保留声明边界。** 它们的全输入保护仍是硬要求；“完整 loop novelty”尚不能签证。VNAR 需适用的原注释/编号；26 个尾巴与 17 个无域需逐条参考或独立可靠编号；AM939711 需源构造澄清。可另立仅对实际观测输入负责的、版本化 scope，但不能把它默认为完整天然 H3 的证明。

完成规则实现后，才适合导出新的版本并对最终 manifest 做旧 AB/antigen 与 expanded TCR/epitope/support/query 双复扫。此文件、184 项的支持性判定和已有 SQLite 精确点查询均不能代替这一步。
