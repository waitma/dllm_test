# 去噪方法补充：自条件、置信度与重新遮蔽

2026-09-30。补充已有[序列架构](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_SEQUENCE_EXPANSION.md)及[视觉架构](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_VISION_EXPANSION.md)调研。本轮阅读原论文的相关方法/限制与两套作者代码，**没有实现或运行下列方法，也没有调整当前500步任务**。

## 阅读结论

| 工作与实际阅读范围 | 方法事实 | 对 scratch 主线的判断 |
|---|---|---|
| [Analog Bits，ICLR 2023](https://arxiv.org/pdf/2208.04202)，§2及附录G | 将上一次自身预测作为去噪输入；训练中混合零条件与停止梯度的预测条件。原方法仍是连续 bit diffusion，附录讨论训练/采样条件不一致。 | 自条件信号不必来自外部预训练模型；迁移到离散masked CE是新假设。不能继承论文的耗时或效果结论。 |
| [SED，2022预印本](https://arxiv.org/pdf/2211.04236)，§3、消融和限制 | 在连续embedding空间扩散，比较随机与预训练embedding；自条件采用预测反馈，论文较好结果仍受预训练embedding影响。 | 不能把整套SED称为已经验证的纯scratch方案。只保留“自己的预测反馈”这一可分离研究方向；未找到并验证作者训练代码，不以第三方复现代替。 |
| [ReMDM，v3](https://arxiv.org/html/2503.00307v3)，§3–4及作者sampler | 通过重新遮蔽允许修订已生成token，可用于既有masked diffusion权重；有概率边界及多种schedule。 | 首先属于采样器维度，不能把采样改进算成骨干训练收益；现有目标也不能自动视为其论文推导的ELBO。 |
| [RemeDi，v1](https://arxiv.org/html/2509.23653v1)，§3及附录B.1 | token与unmasking双流联合预测；原文从LLaDA-8B-Instruct初始化主流，另加4个块，并有SFT/RL阶段。 | “置信度判断与token预测分工”可研究，但双流参数、训练配方和先验都改变；不能据此直接扩展8层主干或宣称scratch同样有效。 |

上述四类工作与已有EMA teacher/JEPA不同：自条件利用自身预测作为下一次输入；teacher提供辅助目标；重新遮蔽改变状态更新；置信度流学习哪些预测值得保留。名称相近不代表训练目标、计算成本或证据可以互换。

## 实际代码核读与可复查证据

固定提交的4份文本快照、URL、SHA和阅读范围保存在[manifest](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_denoising_methods_literature_20260930/manifest.json)。没有安装、导入或执行这些外部文件。

- RemeDi固定提交 `2e1b19d34db09e345898f6c5b6ed60418659821a`：[model](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_denoising_methods_literature_20260930/remedi_model.py:1126)硬编码UPS插入位置 `[0,10,20,30]`；后续将UPS特征反馈到主流，最后独立线性层输出confidence。不能在我们的8层模型上原样照搬，或把最后一层小head说成完整双流复现。[作者源码](https://github.com/maple-research-lab/RemeDi/blob/2e1b19d34db09e345898f6c5b6ed60418659821a/remedi/modelling_remedi_bitowel.py)
- 作者[README](https://github.com/maple-research-lab/RemeDi/blob/2e1b19d34db09e345898f6c5b6ed60418659821a/README.md)明确提供的是inference代码；当前快照只有6个Python文件，不能声称已经拿到完整SFT/RL训练复现。
- ReMDM固定提交 `f830d659fb98d9b2c79ff71bfa12d578312b688a`：[diffusion](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_denoising_methods_literature_20260930/remdm_diffusion.py:658)将MDLM与cap/rescale/conf/loop明确分支；confidence记录与已生成token关联，并在重新遮蔽时更新。categorical抽样显式转float64。实际兼容性不能只看函数名，需要核对状态、缓存和数值口径。[作者源码](https://github.com/kuleshov-group/remdm/blob/f830d659fb98d9b2c79ff71bfa12d578312b688a/diffusion.py)

版本限制：SED的HTML v2、RemeDi的HTML v2链接返回404；本次分别读取实际可获得PDF（页眉v1）和RemeDi v1，未猜测新版内容。Analog Bits以PDF的ICLR2023标记为准，不沿用检索卡片中不一致的年份。这里不是全文逐项复现审计。

## 后续研究设计（本项目推断，尚未实现）

1. **自条件输入**：先用通用人工token任务检查“零反馈”和“自身预测反馈”的独立效果。反馈路径必须停止梯度，不包含真值、真实终点长度或从标签推导的特征；训练与迭代推理都需明确无反馈的初始状态。参数对照可保留同一反馈投影并始终输入零。额外前向成本须实测，除相同步数外另报相同总计算预算。
2. **置信度辅助head**：先研究小head是否能在人工错误输入中区分可保留与应修订的位置，再考虑完整双流。报告校准和错误识别，而非仅CE；比较softmax confidence、独立head和同成本控制。主损失、辅助损失、采样器分开变更。此方案不是RemeDi复现，尚无参数、显存或效果测量。
3. **重新遮蔽采样**：锁定同一checkpoint，先在人工离散任务检查状态更新、终止条件和上下文保护，再比较相同模型前向次数及wall time。不能用更多采样步造成的收益声称预训练模型更好；也不能只看参考匹配而忽略分布、多样性与长度行为。

这三项可以扩大方法覆盖，但现在优先完成现有对照、真实CUDA数值验证和数据准入。它们不会进入当前真实1000步的首轮架构对比，避免同时改变骨干、训练目标与采样。后续进入条件是独立人工CPU/GPU验证、明确预算和新的源码版本；生物语料阶段仍遵守既有数据门禁，不恢复已停止的注释分支。

所有候选都需区分表示能力、单步重建与迭代生成。当前人工partial-mask指标很快接近满分，本身提示后续不能只优化一个容易的重建平均值；这一判断来自本地诊断，并不是上述论文替本项目证明的效果。
