# V7实际grammar输入、context与监督目标核验

本轮核验实际GPU加载的冻结源码及同一sample manifest，不以当前可变工作树推测运行行为。输入是A4结构化grammar：类型/关系标记、按role组织的链块，以及role/scope/链内位置等特征；与旧legacy grammar renderer不是逐token相同。context可以被attention读取，但无直接序列重建标签；它仍可通过目标预测接收反向梯度，不应detach其表示。

## 实际子集结果

|项目|train|valid|
|---|---:|---:|
|记录数|100000|1024|
|有context记录|4648|37|
|context位置总数（包括对应结构标记）|697220|5852|
|context具有label或eligible的位置|0|0|
|具有监督资格的关系token|4648|37|
|其中binding / nonbinding|4648 / 0|37 / 0|
|target实际终止位置的监督资格数|166956|1623|
|未使用的target容量槽位|20387661|187813|
|未使用容量槽位的监督资格数|0|0|
|完全没有target监督位置的记录|0|0|

完整101024行渲染检查通过。这里eligible是监督资格，不表示每轮随机遮蔽都选中它。context包括mhc、peptide、antigen三种role；默认所有antibody heavy/light、nanobody、TCR alpha/beta role均作为target。因此双受体链都可被遮蔽，并非固定其中一条只预测另一条。没有发现额外context/target链metadata标记；已有relation_supervised字段全部为布尔类型。

普通结构前缀、context、batch padding无直接CE；target一个实际结束符被选中时计入CE。target容量尾槽attention仍为1，但label=-100，它与batch padding不同。最终CE仅取被选择的eligible位置；未被选择的目标位置也不直接计算本次CE。

关系监督是明确目标：labels.relation为binding/nonbinding且metadata.relation_supervised=True时，关系token可被遮蔽并计入总CE，单独的AA/terminal子指标不含它。用户随后明确要求保留context-target binding监督。本轮全部有监督关系记录均为binding正例；关系token预测成功不证明二分类判别能力，不能给单类验证集报告有意义的AUROC。未知/未测的配对不能自动作为nonbinding。

## 完整epoch

实际冻结take_rows源码以整数行索引执行核验：seed42/43均在3125×8×4次消费后覆盖全部100000行索引且每个恰好一次，最终epoch=1/position=0；在第1000步恢复后顺序也与不中断一致。这是索引遍历保证，不是序列内容去重证明。八项都继续完成3125步；当前中间曲线不能当作完整epoch。

## 证据与源码

- mask统计：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/context_loss_mask_audit.json
- 标签计数：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/binding_label_inventory.json
- 行索引/恢复合同：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/epoch_sampler_contract.json
- 冻结collator：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/source_snapshot_v1/dllm/pipelines/immune_llada/v3/layout.py
- 冻结loss：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/source_snapshot_v1/dllm/pipelines/immune_llada/research/model.py
- 冻结runner：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/source_snapshot_v1/scripts/research/run_v7_architecture_pilot.py

## 增加context的下一版抽样

用户已明确授权增加context记录并保留binding关系监督。计划保留当前100000 train作为原字节前缀，追加20000条来自同一已审核V7 parent train、具有context与有效明确关系监督的既有记录；原valid1024不变。达到条件时总120000，context24648（20.54%），global32下完整epoch3750步。下一版单独目录、单独manifest和provenance，不能改变运行中的p01样本。按source分层；本轮按用户确认只取原有binding正例；逐行身份/原行SHA避免与base及新增选择重叠，不声称新增生物学去重。

该版本已实际生成并通过加载及新增行A4布局检查（见文末最新记录），3750步新组尚未启动。本轮只学习已有binding正例，不做正负判别；不虚构负例或审核。正式训练仍暂缓。后续独立报告有/无context、关系loss及各标签分母；不能把增加context的数据变化收益归因于架构。

## 2026-10-02T19:56:12.027484+00:00 nonbinding缺失的配方原因已定位

parent manifest filter_names含recipe.binding_only，recipe.drop_nonbinding_sources显式列出asd_antibody/trait/tcr_native/tcr_papers。当前subset监督标签全部binding不是抽样偶然性的充分解释，而是与该配方一致；全parent统计将在追加扫描给出。用户已被告知，并收到可选目标澄清（后续可靠负例正负判别，或本轮仅binding正例）；等待该偏好不阻塞现有八卡epoch及20k已审核context正例增补。不会把未观测配对/unknown造为nonbinding。

## 2026-10-02T19:58:16.131168+00:00 用户确认本轮只学习binding正例

异步目标澄清已明确选择“本轮只学习已有binding正例，暂不做正负判别”。据此追加候选只取原V7明确binding=True监督的context记录，保留关系token监督，不引入negative、不生成负例、不请求新负例来源。缺nonbinding不再作为本轮阻断，也不把正例学习描述为正负区分能力。当前八卡样本/源码/3125步不变；独立追加20000正例context仍在实现中。

## 2026-10-02T20:17:42.129532+00:00 context正例增补入口验收通过，准备一次实际扫描

新入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/augment_v7_context_subset.py 复用原抽样器的SHA、stat和来源配额逻辑；人工fixture通过确定性、原train前缀/valid字节保留、正例与原train/valid逐行来源及原字节排重、父分片变动拒绝。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/context_augmentation_self_test.json。仅从现有父train追加20000条明确binding且relation_supervised=True的context记录，不生成标签/负例，不恢复注释分支。实际增补尚未完成；输出独立sample_context_plus20k，原八项与冻结源码保持。

## 2026-10-02T20:26:02.091441+00:00 context正例新增20000条完成，加载与loss布局检查通过

用户选择本轮只学习已有binding正例、暂不做正负判别。独立新子集 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/sample_context_plus20k 已生成：保留原100000训练行作为逐字节前缀，追加20000现有V7 parent train记录，总120000；valid1024逐字节不变。context从4648/100000（4.648%）增至24648/120000（20.54%）。新增均labels.relation=binding且metadata.relation_supervised=True，未生成标签或负例。原74个train分片的SHA/count全部核对，候选context正例297751条，排除base中4648行后可选293103条，分source比例抽样。

| 新增来源 | 行数 |
|---|---:|
| asd_antibody | 11335 |
| tcr_native | 825 |
| tcr_papers | 7685 |
| trait | 155 |

原冻结loader实际加载120000/1024并通过parent ready/两证书当前SHA绑定检查；新增20000行逐条A4布局通过，最长1299 tokens，无丢行/截断。新增context位置2972878个，label/eligible均0；binding关系监督资格20000个，target监督资格2292707个。独立复核新增原行SHA唯一、与base train/valid原行SHA无交集。这里排重指原行字节与来源身份，不声称做了新的序列相似度清理。eligible仅指可被遮蔽监督，不表示每步都纳入loss。

数据manifest SHA 1efa27ea19d10bc94143c3977c80dba8c3a29bad921c093ca7f24447635b9478；完整计数/来源证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/sample_context_plus20k/sample_manifest.json；加载与mask检查 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/context_augmentation_preflight.json。增补入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/augment_v7_context_subset.py；独立只读源码review无必须修正项，记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/context_augmentation_code_review.json。没有产生checkpoint或修改正式ready/审核文件、恢复注释分支或独立CUDA算子工作。

当前原8项记录步数1400–1475/3125，继续原100000条完整epoch，原样本/冻结源码/作业不变。新120000条尚未用于训练；global32下下一独立组应完整3750步。旧p01 runner/dispatcher/plot固定3125，不能直接拿它训练120000并称完整epoch；后续必须独立配置3750的训练调度、终态与作图并分组报告。没有平台submit/cancel、容器/worker/监听重启或重复入队。


## 2026-10-02T21:23:10+00:00 正例目标一致性核对与正式入口尾批问题

用户再次确认“本轮只学习已有 binding 正例，暂不做正负判别”。此选择与已冻结 p02 入口、关系指标及 11 份 YAML 一致，无需修改在跑实验或重新入队。Context 序列不计重建 CE，只有已有明确监督的关系 token 在被选中时计入原 CE；unknown/未观测配对不变成负例。已有新增 20000 行布局证据显示 context 的 label/eligible 均零，关系 eligible 为 20000，本次复用原证据而未重复扫描或测试。只读一致性记录：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/positive_only_reconciliation_20261002T2122.json。单类关系准确率可能由恒预测正例获得，不能证明使用了 context 或具有正负判别能力；非 binding 缺失不是本轮阻断。
