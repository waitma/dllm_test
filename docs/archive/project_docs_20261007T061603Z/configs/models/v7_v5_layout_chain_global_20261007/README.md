# V5 对象布局、固定容量的新训练版本

当前配置：/vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/research_config.json
语义数据：/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_scope_final_20261002
状态：实现与CPU验证完成，尚未提交新训练，没有本版本训练权重或下游成绩。本文覆盖此前“按真实长度布局／必须指定 generation_lengths”的描述。

## 当前训练范围

只考虑常规抗体 H/L 与 TCR；VHH／nanobody（含抗原→VHH）不纳入本版训练、专门评测或验收。通用 renderer 的兼容性测试不代表本版训练覆盖。version.json 中的 training_scope 是范围声明，训练清单构建须落实排除。

## 输入与结束位置

V5 context→关系→受体对象的顺序保留。整个受体的内容区域统一预留 **512个槽位**，包含残基、链间`.`和最终`<protd>`，不是每条链512位，也不是真实答案长度。配置 base.full_capacity 在 V5 中表示对象容量；fragment_capacity 不决定 CDR-only 的长度。上下文与头部位于这512位之外，总输入上限仍为2048。

```text
训练：<prots>antigen<protd><binding><prots><ab>H序列.L序列<protd><mask>…
生成：<prots>antigen<protd><binding><prots><ab><mask>…（共512个内容槽位）
结果：<prots>antigen<protd><binding><prots><ab>生成H.生成L<protd>…
```

- 目标头部 `<prots><ab>`／`<prots><tcr>` 是固定条件。
- 真实残基、`.`和`<protd>`参与去噪损失；合成补位及结束符后的尾部 labels=-100。
- 整个512位内容区域参与注意力，包括synthetic和尾部mask；不可在真实终点关闭attention。
- 生成不预先放置目标`.`或`<protd>`。模型同时预测链间分界及对象结束，解析仅返回第一个有效`<protd>`之前的链。
- 终点前仍有mask时继续解码；没有结束符返回capacity_exhausted；空链、多余分隔符、链数不符返回明确失败。不能将达到容量后的截断当作正常完成。
- 训练时内容超过容量或总输入超过2048直接报错，不静默裁剪。

| 数据类型 | 固定前缀及干净目标内容（随后补mask至512） |
|---|---|
| 抗体 H/L | `<prots><null><protd><unknown><prots><ab>H.L<protd>` |
| 抗原→抗体 | `<prots>antigen<protd><binding/nonbinding/unknown><prots><ab>H.L<protd>` |
| TCR 双链 | `<prots><null><protd><unknown><prots><tcr>beta.alpha<protd>` |
| TCR 单链 | `<prots><null><protd><unknown><prots><tcr>beta或alpha<protd>` |
| 肽→TCR | `<prots><pep>peptide<protd><binding/nonbinding><prots><tcr>beta[.alpha]<protd>` |
| pMHC→TCR | `<prots>MHC1[.MHC2]<protd><binding><prots><pep>peptide<protd><binding/nonbinding><prots><tcr>beta[.alpha]<protd>` |

关系标记沿用原renderer：未标注TCR recognition默认binding，MHC→肽固定binding，无条件使用null前缀。只有显式 relation_supervised 的关系参与关系监督。

## 防止真实链长通过特征泄漏

内容槽位的角色及范围编码为unknown，原始chain_ids为统一对象段，position_ids为固定1…512。首链角色仅在固定类型头部登记，区分单alpha／单beta等显式任务身份，不编码链长。

denoise在训练腐蚀之后，只根据**当前可见的`.` token**划分链段并重置位置；分隔符被mask时，该对象暂为一个尚未分开的段。偶数层在当前可见链段内部全注意力，奇数层全局注意力。不会从标签、原始CDR坐标、synthetic掩码或真实终点恢复被遮蔽的边界。

保持残差系数1.0、紧凑50词表、无ESMC和随机初始化；没有恢复旧V5大词表或旧V5权重。

## CDR-only 全长 scaffold

只有CDRβ／CDRα的记录仍保留已有prepared全长scaffold、区域位置与真实锚点。不重新采样补全长度，不缩成短CDR；synthetic_residue_mask标记的位置从存储X转换成输入mask，真实观测X保留。全缺失的配对链保留已有全mask scaffold。scaffold之后继续补尾部mask至对象容量512。

synthetic位置参与注意力但不参与残基损失或真实残基统计；scaffold的分隔及终止标签表示已有构建布局，不是缺失全长序列的实验测量。原始JSONL无需重写。

## 入口与使用限制

训练入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/run_v7_formal.py：

```text
--variant cv_segment_global --input-layout V5 --object-capacity 512
```

其余八卡初始化、冻结源码、数据审核和来源清单参数仍必须满足入口要求；新增 v5_canvas.py、v5_generation.py 也须纳入冻结源码清单。旧A4默认值与历史恢复绑定保持原样。

统一输入：/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/v3/layout.py 的 `ImmuneCollator.from_config`。
下游入口：/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/v5_layout.py 的 `inputs(mode='generation')`、`generate` 和 `cdr_inputs`。`GenerationRequest(conditions=..., targets=...)`只需声明条件链及目标角色，不需目标序列或目标长度。旧generation_lengths参数明确拒绝，避免误当成容量。

支持整个对象生成，以及已知前置链→生成后置链（H→L、beta→alpha）。已知后置链但生成前置链需要单独的条件协议，当前明确拒绝，避免偷用真实分界。已知CDR跨度填空仍是独立协议。

模型自身 `generate` 也调用同一个固定容量解码器；要求generation-mode batch，拒绝提前带有目标终点的输入。返回每个对象的status、chains、terminal_offset及canvas_ids；未经过真实训练的模型输出不代表有效下游成绩。

## 验证证据

- 83项CPU测试通过，包含固定画布、端点监督、全遮蔽训练／推理特征一致、可见分界注意力、梯度重算、保存恢复及历史A4回归。
- 7来源、train/valid首分片最多100条，共1250条抽查通过；130513个synthetic补位保持正确。抽样最大对象250 token，最大总输入1371 token。不是全量扫描。
- 保存恢复测试的临时checkpoint自动清理，没有保留测试权重。

报告：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_v5_layout_20261007/canvas_validation.json
测试日志：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_v5_layout_20261007/canvas_regression.log
复核脚本：/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/audit_v5_canvas.py
历史 validation.json、scaffold_mask_validation.json 描述之前阶段，当前布局以canvas_validation.json为准。
