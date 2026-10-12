# 扩展人工 heldout 的可识别性审计

2026-09-30 UTC，在500步实验继续运行时，以实际39文件源码、fixture manifest、collator和四档固定mask完成只读CPU审计。没有加载模型、使用GPU、训练、保存checkpoint或改动生产源码。审计使用与runner一致的按stratum最多4条collate方式；四档AA/end/selected计数与已经完成的R20真实25步summary逐项一致。

72条heldout中的36条有条件记录在全mask后仍互不相同；36条无条件记录只剩3种可区分输入，每组12条。比较的是送入denoise的全部7个特征，不依据原始标签或metadata来判断模型可见性。25/50/75% mask各自均为72种不同输入。

下表为该固定heldout集合上的输入确定性argmax上限，不是模型可学习性或真实泛化保证。

| 子集 | Full-mask可区分输入/记录 | AA-only最大正确数 | AA-only上限 | 全部目标最大正确数 | 全目标上限 | 联合argmax最优时的end正确数范围 |
| --- | --- | --- | --- | --- | --- | --- |
| 全部 | 39/72 | 14664/17864 | 82.0869% | 14756/17996 | 81.9960% | 92–112/132（69.6970%–84.8485%） |
| 有条件 | 36/36 | 9744/9744 | 100% | 9816/9816 | 100% | 72/72（100%） |
| 无条件 | 3/36 | 4920/8120 | 60.5911% | 4940/8180 | 60.3912% | 20–40/60（33.3333%–66.6667%） |

AA-only、terminal-only和全目标argmax是不同优化目标。单独优化terminal-only时三子集均可达100%，但这会牺牲部分AA预测；表中end范围限定为同时达到全目标argmax最大正确数，并计算了modal label并列时的所有end取舍。不能把无条件end最高66.7%误称为无条件end本身的普遍上限。

四档mean-AA的该集合上限分别为全部95.5217%、有条件100%、无条件90.1478%。该结果仅由实际input碰撞和标签冲突决定，不以训练结果推测上限，也不承诺模型能达到它。尤其heldout精确字符串不同并不使共享人工生成规则变成生物泛化证据。

125步出现full-end召回100%后，审计增加了逐链首次END的独立计数。相同full-mask输入只能产生同一逐链首次END位置；每个无条件输入组、每条目标链的真实结束位置分布均为`{24: 4, 127: 4, 255: 4}`。位置按链内attended target slot的零起始序号表示，数值即真实残基长度。每个输入/链取频率最高的位置，得到首次END精确命中的有限集合上限：有条件72/72=100%，无条件20/60=33.3333%，总体92/132=69.6970%。该目标与“所有真实END位置的召回率”不同：同时在三个位置输出END可以命中全部真实END，但首次输出只能是其中一个。JSON的`groups.*.fullmask_first_terminal`保存全部输入hash、fixture_id及链内位置分布；这是可识别性上限，不是当前模型实测首次停止准确率。

25步的全部方法end=0远未达到这些最优范围。有条件end也为0，不能完全用无条件信息不足来解释。500步应分开观察有条件规则迁移、无条件AA/end取舍和部分mask准确率；不能仅看总体loss，亦不能强行要求全mask所有指标同时100%。

复算程序：[audit_immune_expanded_identifiability.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/audit_immune_expanded_identifiability.py)。计数、实际fixture_id分组、信息熵下界、所有源码SHA和审计SHA：[expanded_identifiability.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/monitor/expanded_identifiability.json)。4项回归验证了冲突标签、联合最优并列取舍、零end分母、相同输入三长度的first-END上限与不一致链槽拒绝，全部通过（0.05秒）；实际生产fixture审计再次通过，39份源码和原始selected/AA/end计数均匹配。

先激活pllm，再执行：

```bash
USE_TF=0 python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/audit_immune_expanded_identifiability.py --reference-summary /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790802160338779408_r20_refined_control_3b41bffe/summary.json --output /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/monitor/expanded_identifiability.json
python -m pytest -q /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_expanded_identifiability.py
```

后续如生产源码冻结到新目录，使用程序的`--source-root`参数指向产生本次summary的对应快照，禁止拿改变后的生成协议冒充原审计。
