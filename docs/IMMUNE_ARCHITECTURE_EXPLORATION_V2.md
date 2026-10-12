# 架构扩展 v2：调研、验证与持续监督

更新：2026-09-30T20:31:23Z。资源沿用 `t-20261001033327-fmnql`：c20250601单实例8×A100-80GB、非闲时。没有新申请单卡任务。旧12项20步已全部成功；20:31Z观测新8项均到50/100步，执行失败为0、35个源码指纹匹配，尚无新的真实语料训练或下游收益结论。最新步数见自动更新的监督记录。

## 阅读范围与方法筛选

新增 [序列架构调研](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_SEQUENCE_EXPANSION.md)（14篇）和 [图像/多模态调研](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_VISION_EXPANSION.md)（10项新工作并深入DINOv3）。两份清单有Gated Attention、MBT、Perceiver IO等重叠，不能将24项直接当24篇不同论文。核读固定版本官方源文件、作者附录，链接/commit/SHA及抓取限制已各自记录。

范围包括门控/差分注意力、Value residual、HC/mHC、Mamba2/Jamba/Caduceus/GatedDeltaNet、SetTransformer/Perceiver、分层融合、局部/全局attention、LayerScale、SIGReg、EMA自监督、表示encoder与重建decoder分工、token压缩。已有SwiGLU/RMSNorm/RoPE不重复当新改动；优先落地不依赖外部预训练teacher、接口改动小、能做明确对照的机制。

## 已实现并运行的8个配置

| ID | 名称后缀 | 实际变更 | 相对R12增加参数 |
|---|---|---|---:|
| R12 | architecture_control | 同版本全局attention对照 | 无 |
| R13 | head_output_gate | 每token/head sigmoid，SDPA之后、输出投影之前 | 73,728 |
| R14 | fixed_half_attention | 固定0.5 attention输出尺度，检验R13是否仅靠减幅 | 0 |
| R15 | layerscale | attention与FFN两分支可学习逐通道scale，初值0.1 | 12,288 |
| R16 | fixed_residual_scale | 两分支固定0.1，排除R15单纯更新尺度效应 | 0 |
| R17 | chain_late_fusion | 8层前4层严格同链，后4层global；prefix独立，权重共享 | 0 |
| R18 | chain_local_global | local/global交替；同链位置半径32即最大65位置 | 0 |
| R19 | value_residual | 后层V=.5V_first+.5V_current，同forward显式传递、不detach | 0 |

机制来源：[Gated Attention](https://arxiv.org/abs/2505.06708)、[CaiT/LayerScale](https://arxiv.org/abs/2103.17239)、[MBT](https://arxiv.org/abs/2107.00135)、[Gemma2](https://arxiv.org/abs/2408.00118)、[Value Residual Learning](https://arxiv.org/abs/2410.17897)。这是本项目的受控适配，不是完整论文复现，更不是已证实的新方法。R17实际选前4/后4，与文献调研建议的前2/后6区分；后者可单独追加。R18目前用dense mask验证结构偏置，不宣称稀疏计算加速。

实现：[research_next/model.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/research_next/model.py)。仍采用A4词表与布局、原始序列C/F/W边界、角色与范围embedding、链内RoPE、单链结束标记；没有改数据输入或引入真实目标长度。基底768维、8层、12heads、MLP3072；各变体从头初始化。

## 检查与公平比较

- 新模型独立55项CPU契约在pllm和真实worker环境均通过。R12在同权重同输入下与旧R00的FP32/BF16 logits/hidden/loss逐元素一致；padding、链重排、prefix隔离、生成白名单、保存恢复、新参数梯度、两视图checkpoint重算、V_first不跨调用缓存均通过。
- 接通旧入口factory/新版runner后，合并 **115项通过**；worker环境factory **11项通过**；另有1项fixture退出后六个全局绑定恢复检查通过。原始日志与版本：[validation](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_architecture_next_local_20260930/release_integration/validation.json)。临时checkpoint已清理，仅保留证据。
- 新runner为17项测试通过；初始化与device迁移后统一重置seed42，使R13新增gate参数初始化不会改变后续遮蔽随机流。四档评估保存/恢复RNG与train/eval状态。旧R00的20步轨迹不属于同一噪声协议，v2架构比较以R12为基线。
- GPU profile按各方法独立重新测，三步每档、2048-token、FP32参数/Adam+BF16 autocast、whole-layer checkpoint。R12–R19均推荐micro16/accum16，global256；下一档32 OOM是预期上界。不能沿用旧R00–R11的micro32。
- 当前v2实现每层构造bias，监督发现相对旧实现存在显存开销。[隔离缓存原型](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_BIAS_CACHE_VALIDATION.md)已通过82项CPU等价测试，尚未部署或测GPU峰值，不能将本轮速度或显存差异全归因于新方法。改动须新版本、新profile并保留旧记录。

## 100步诊断与真实训练前的条件

每个v2方法从头100步，不从未保存的20步checkpoint续训；每5步记录梯度和25/50/75/100%固定遮蔽指标。每档只统计实际被遮蔽的AA/end，缺失子集记null。顶层5个指标仍代表全遮蔽；mask_mean是四档非空值等权平均，不等于按token总数加权。

旧toy在全mask下4条TCR的7种模型输入特征完全相同，但标签不同；AA-only上限46/82=56.10%，全部目标token联合上限51/93=54.84%，两种上限不宣称能同时达到。不能把未达到100%准确率当代码错误，也不能因总loss下降就认为迁移效果好。

工程通过要求数值/梯度有限、接口与恢复正确、有有效profile、四档评估可审计且显示学习迹象。它只支持进入真实语料初筛，不替代泛化评估。真实1000步需要去重证书与downstream_scope证书，固定验证集合/四级mask/同global256及同LR horizon。优选5000步必须记录选择依据、对照和续训路径；最终判断仍需多seed及下游测试。

当前 `immune_v7_mhc_sequence_v1` 两份证书仍缺，真实训练不启动。清理阶段需继续覆盖新增antigen-antibody任务、few-shot support/query、TCR及epitope重叠，不能通过降低审计标准来提前训练。

## 配置、版本和监督

实验清单：[experiments.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_architecture_v2/experiments.yml)。这是**现有工作池的本地配方，不是volc资源申请YAML**。入队工具：[enqueue_immune_architecture.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/enqueue_immune_architecture.py)，去重后发布；真实阶段保留数据和诊断门禁。selected5000清单暂空，等待1000步验证后选择。

旧R00–R11代码已冻结：[源码快照](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_releases/r00_r11_20260930)，32个已测源文件SHA验证一致。旧入口执行旧方法时路由快照，因此旧profile可按原版本重用；新R12–R19路由新runner。先等当前任务全部结束、再写release_map、最后切换入口，未修改执行中的实验源码。未来新版本也须同样保留源码与结果对应关系。

监督subagent负责本轮结果解释和排错；独立后台守护进程每30秒持续读取产物，每90秒查平台。跨回合持续运行的是守护进程，不承诺模型推理或聊天窗口无限在线。它按job_id保留各阶段结果，按release_map校验对应源码，区分预期OOM、执行异常、数值异常、证据错位、端点捷径、未学会结束等情形，不盲目重试或热改活跃代码。

[实时结果](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_POOL_LIVE.md) · [监督记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_SUPERVISION.md) · [判读与改进](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_SUPERVISION_FINDINGS.md)。发现实现错误时：保存失败证据→在隔离代码修复→CPU回归→干净进程GPU复测→新profile→再加入批量初筛；不把失败产物覆写成成功。维护占卡负载始终另标maintenance_only。

## 旧12项首轮结果（20步，非下游）

| 方法 | 状态 | full-mask loss | AA acc | end acc |
|---|---|---:|---:|---:|
| r00_a4_control | succeeded | 2.06152 | 0.19512 | 0.00000 |
| r01_span_mask | succeeded | 2.03641 | 0.21951 | 0.00000 |
| r02_chain_completion | succeeded | 2.17229 | 0.28049 | 0.00000 |
| r03_chain_balanced_loss | succeeded | 2.53545 | 0.18293 | 0.00000 |
| r04_qk_norm | succeeded | 2.08672 | 0.19512 | 0.00000 |
| r05_register4 | succeeded | 2.05223 | 0.19512 | 0.00000 |
| r06_local_conv7 | succeeded | 2.06022 | 0.19512 | 0.00000 |
| r07_mask_ratio_film | succeeded | 2.01476 | 0.17073 | 0.00000 |
| r08_vicreg | succeeded | 2.04524 | 0.19512 | 0.00000 |
| r09_equal_source | succeeded | 1.95289 | 0.24390 | 0.00000 |
| r10_role_pair_bias | succeeded | 2.06220 | 0.19512 | 0.00000 |
| r11_span_qknorm_register | succeeded | 1.97122 | 0.19512 | 0.00000 |

以上全部end=0，不据此敲定最终架构。R03损失训练目标不同，总loss尤其不能直接排名。


## 2026-09-30T20:24:51.052023+00:00 晋级门禁与显存优化验证补充

真实1000步入队工具新增完整证据检查，**63项测试通过**：同variant的100步COMPLETE/完整trace、profile摘要SHA/当前35文件指纹、生产配置/显存档、固定四mask统计口径、四档均值loss下降且至少一档AA恢复改善、嵌套非有限拒绝。数据证书优先，整批全部验证后才发布，增加文件锁及重复防护。当前100步尚在跑且两份数据证书缺失，未晋级真实训练；这不是生物学质量认证。[证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_architecture_next_local_20260930/enqueue_gate_validation.json)。

新registry扩至20方法后，pool init显式只初始化旧12项20步，v2八项用独立配方100步入队；pool/monitor回归 **11项通过**，重复预览8项均already_registered，没有重复任务。

监督发现的bias重复分配已做隔离优化原型：[bias_cache.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/bias_cache.py)。**82项CPU等价测试通过**，覆盖8变体、FP32/BF16、padding、每参数梯度、checkpoint双视图/跨forward状态；16项运算计数显示8次FP32 bias masked_fill降到0–2次、8个独立bias storage降到1–2。没有修改正在运行源码，没有GPU优化峰值数据，不能宣称本轮已省多少显存；待当前版本结束后另发新版本、重新GPU profile与验证。[计数证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_architecture_bias_cache_cpu_20260930/counts.json)。

当前25步早期诊断中，R18局部/全局交替的四档平均loss/AA/end表现更好，但可能受短toy对大MASK容量尾部敏感的影响。正式比较需增加长度分层与真实验证，不在100步结束前或只凭toy敲定架构；R15/R16接近也说明可学习scale的收益尚未区别于固定缩放。

20:29Z补充：监督模块增加对源码文件集合的校验，除已有文件SHA改变，也检测新文件进入runner收集范围；对应9项回归通过。仅重启本地监听，GPU实验未受影响。监督subagent继续跟踪本轮至100步，并在异常时保留证据、定位及做隔离修复；常驻daemon负责跨回合落盘，不自动修改模型或晋级真实训练。

20:31Z的50步更新：R12–R19四档mean loss依次为.3630/.4058/.3214/.3072/.2865/.4227/.3315/.3563。R18早期优势已缩小；固定尺度R16最低，R13尚未胜过固定gate R14，R15尚未胜过固定scale R16。短程排序不稳定，增加step和固定尺度对照有必要；这些仍是合成诊断，不能直接选出下游更好的方法。
