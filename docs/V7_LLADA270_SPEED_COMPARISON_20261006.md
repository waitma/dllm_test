# V7 与原纯 LLaDA270M 的速度核实

## 2026-10-06T09:41:42.093903+00:00 V5/current源码对比完成；澄清2048上限与checkpoint策略

23份关键源码对照及解释已写入 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_V5_CODE_COMPARISON_20261006.md，当前源码逐份匹配冻结manifest。V5 FSDP由保存参数确认；历史git版本可复原但未提交补丁缺失，报告明确此限制。新确认层内显式RoPE读取position_ids.max()及component float同步、稠密可训练bias反向、固定target容量与batch padding均值得受控计时；尚未实测瓶颈或切换并行方式。

2048是整条记录上限，并非统一padding。既有1024条小样本/micro32的32批实际S=516–1206，17批为516；固定target容量256/64仍在。当前每1000步及分段边界保存，保留最近3个完整恢复点（5000/5016/6000，合计约2.724GB），不同于V5最优3+最新/slim。新增证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/v5_code_review_v1/root_review.json，没有重跑模型、读原始序列、修改训练配置或清理正式恢复状态。文档f010/R24标签已纠正。

当前正式6725/25042、本地850/3750，原进程与8rank心跳正常，完整验证仍6000/750；有限观察41757已实际rc0结束，原训练继续。完整Goal仍active，本轮progress。既有安全停线、独立CUDA暂停、远端403不重试和数据/来源门禁保持。


## 2026-10-06T09:29:33.300520+00:00 ESMC V5原始实测：当前更新墙钟耗时约4倍

用户询问V5对照；直接读取两个V5 diffusion运行的原始W&B日志、实际argv和保存training_args.bin静态字段，不使用历史规划阶段1.6–2.5秒的外推。两份metadata均无resume参数，micro4/GA8/world8/global256、FSDP、可训练ESMC-300M均已核实。当前正式为DDP/micro32/GA1/world8/global256、noESMC。

| 运行 | 实际步数区间 | 累计耗时 | 平均秒/更新 | 名义条数/秒 | 末尾tqdm近期值 |
|---|---:|---:|---:|---:|---:|
| V5 c20250601，r9530a4l | 0→159383 | 121:03:02 | 2.734181 | 93.63 | 2.30 s/it |
| V5 queue012，m9aunutf | 0→79602 | 60:23:49 | 2.731451 | 93.72 | 2.48 s/it |
| 当前V7，4000→5000 | 1000次更新 | 10926.703秒 | 10.926703 | 23.43 | 未作同口径记录 |
| 当前V7，5000→6000 | 1000次更新 | 11170.567秒 | 11.170567 | 22.92 | 未作同口径记录 |

当前每次更新墙钟耗时为V5的4.00–4.09倍。两边平均均含区间内验证/保存及等待，V5为多天跨度，V7为1000步区间，后者5000→6000还含5016额外验证/保存/接续。V5末尾平滑2.30/2.48不是累计均值，不与当前区间均值混算。

两版都用单机8张A100 80GB；V5输入上限1024、workers4，当前上限2048、workers0。实际padding/token分布和注意力实现不同，同global256不等于同输入GPU纯计算比。不能单凭V5用FSDP断言它解释4倍差距；受控GPU计时仍UNRUN，原训练继续。

原始来源：

- [V5 c20250601日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_esmc_llada270m_diffusion_immune_v5_8gpu_c20250601/wandb/wandb/run-20260912_071016-r9530a4l/files/output.log:9882)；[实际argv](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_esmc_llada270m_diffusion_immune_v5_8gpu_c20250601/wandb/wandb/run-20260912_071016-r9530a4l/files/wandb-metadata.json)。
- [V5 queue012日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_esmc_llada270m_diffusion_immune_v5_8gpu/wandb/wandb/run-20260912_141528-m9aunutf/files/output.log:4921)；[实际argv](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_esmc_llada270m_diffusion_immune_v5_8gpu/wandb/wandb/run-20260912_141528-m9aunutf/files/wandb-metadata.json)。
- [结构化核实](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/v5_review/review.json)，SHA256 cf93472fde876ecb550fb8dd111038d8054d52eec69a97e19f1580b9d4ced0b6，含日志行/源SHA/静态pickle字段/计算量及当前保存事件时间。本轮没有pickle反序列化或权重payload读取。


## 2026-10-06T07:40:52.614659+00:00 正式5000→6000的新墙钟区间

原监督事件2026-10-06T04:27:00.647840+00:00→2026-10-06T07:33:11.214950+00:00，1000个更新跨11170.567秒，折合11.170567秒/更新。此区间含常规验证/保存、5016额外段尾验证/保存和接续恢复，以及观察延迟；不是GPU纯计算计时，不能据此归因DDP/FSDP、数据加载或占卡进程。当前配置仍DDP、micro32/GA1/global256。受控GPU profile仍UNRUN，不干预正在运行的原正式/本地训练。事件原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/events.jsonl；结果原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint6000_root_observation.json。

## 2026-10-06T05:43:41.648750+00:00 本地f000完整epoch验收与9/22固定末点对照

本地f000/42完整3750步/120000条epoch已验收，退出码未知如实标注；3750固定末点图已更新为9/22，13项未启动。f011平均AA CE较低，但全遮蔽AA/END没有同步优于基线，不能作理解/生成全面收益结论。正式八卡继续5400/25042，配置不变。f010/42真实接续包准备中，尚未启动；受控测速仍因非独占未执行。详情 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md，根复核 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/completion3750_root_review_20261006.json，新图 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_20261006_epoch1/step3750_architecture.png。

核实日期：2026-10-06 UTC。本轮仅读取历史日志、参数文件头、配置与当前冻结源码，没有运行模型、读取原始生物序列、占用 GPU、修改冻结源、提交/停止/重启训练。实际训练与数据审核约束保持。主进程负责同回合同步 PROJECT_PROCESS.md 和 BIOSEQ_MODEL_PLAN.md。

## 2026-10-06T04:31:36.542828+00:00 正式5000步验证与保存完成：重建CE和真实END准确率改善，首END与误报仍分化

新增保存事件区间4000→5000为10926.702977秒，即10.9267秒/步，包含验证/保存及最多一次观察间隔，不能当GPU纯计算耗时或归因DDP/loader/占卡。正式单机8卡DDP配置保持；新节点的重建与结束位置指标方向分化，详细结果见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_TRAINING_RESULTS_20261005.md。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint5000_root_observation.json（SHA4e51f7c8d2006dd5221d548aa6cdd04ea1ab2ca36b1245de6ab846a5b0e99b64）。整模型GPU计时仍未执行，独立CUDA专项继续暂停；未重启训练或重试已拒绝的远端进程查询。


## 2026-10-06T02:04:56.990109+00:00 正常读取与内存回放的配对测速入口已实现，GPU尚未执行

私有v3入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/real_batch_replay_preparation/profile_real_batch_pair_v3.py 已完成根源码复核；SHA256=b8be85365f947b2aa352f6f6c105750188b5919aaea5ca1bb6c2c9c710553a01。新4项人工CPU控制检查实际通过（suite0.094秒、脚本3.125秒、工具退出0），覆盖固定窗口/预算、配对一致与差异拒绝、CPU随机状态重置和延后噪声哈希，以及原v2父进程生命周期保持。没有重复原输入3项或旧模型测试，没有真实V7读取、模型实例/forward/backward、CUDA或远端API调用。结果/实际退出记录及运行说明见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/real_batch_replay_preparation/README.md；根复核 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/real_batch_replay_preparation/root_pair_runner_review_v3.json，SHA256=4a9b3cce0cc33ec199196709f30b24f2be25311ccda7dd6cc81969b84f2690fe。

实际实现限定f011、micro32/GA1、1024条唯一输入；normal和CPU回放分别重新创建seed42模型、空AdamW及全部RNG，每模式5预热+16分项+10整步+1profiler=32更新，共64，不超过原80更新/1800秒（默认900秒）。两阶段核对初始化及每步输入/标签/噪声/随机状态/曝光一致；同口径CPU→GPU复制，读取、拼批、验证clone、H2D及模型阶段分别记时。噪声哈希与D2H观察在本步GPU计时和profiler外；原forward同步保留。测试不写权重、序列或token缓存。

当前仅达到“代码准备与CPU控制检查完成”，还没有真实V7/GPU计时结果，不能确认GPU随机配对、显存容量、速度收益或DDP瓶颈。16个分项样本不证明统计稳定；整步wall包含额外且不对称的验证/clone成本，不能直接相减当loader收益。首次prefix索引已预热页缓存，单卡诊断不等价于正式8rank DataLoader/通信/验证/保存吞吐。未来须资源实际释放并确认独占后才可执行，当前八卡和本地训练继续，不抢占其GPU。

v1/v2及冻结正式源未改，人工fixture和测试临时权重均无残留，正式恢复状态保留。远端403不重试，序列注释停线、独立CUDA暂停与原审核范围保持。原正式25042和完整22格消融目标仍未完成。


## 2026-10-06T01:49:40.093225+00:00 批次回放输入接口已复核；配对GPU计时仍待整合

输入适配器已实现并由根端完成源码与原始证据复核。原人工CPU接口检查3项通过、实际退出0，覆盖单次捕获、同序回放、clone隔离、读取失败不重试及context序列CE0/已有binding正例。根端没有重复测试、读取真实V7记录或运行GPU。临时人工fixture已清理，无测试权重或bytecode残留。根复核原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/real_batch_replay_preparation/root_input_adapter_review_v1.json，SHA256=e15dd7305fdec3da1b8dd21a46e52a9816f818ea537266e60bf355f93624d39d；接口及限制见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/real_batch_replay_preparation/README.md。

这不是GPU测速通过：真实输入捕获尚未执行，正式数据审核没有变更，micro64只有计划边界检查。后续只准备f011的normal与CPU缓存replay配对入口，每模式32个唯一batch，5预热+16分项+10整步+1profiler，共64更新；保留原80更新/1800秒上限与默认900秒。两个模式分别重建同seed模型、optimizer和噪声RNG，核对初始化、输入/标签/噪声选择及曝光一致，H2D同口径，校验/clone成本单列。16个分项样本只用于短诊断，不能声称稳态或端到端DDP加速；此入口尚未完成或执行。

正式训练继续单机8卡DDP+NCCL，历史纯LLaDA270M为两机各4卡FSDP；并行方式差异不是已证明的慢化原因。本轮没有submit/cancel/restart或正式配置修改。原远端进程查询API403分支不重试，序列注释停线与独立CUDA暂停保持；已有完整22格消融和25042步正式目标未完成。


## 结论

**找到了真正 no-ESMC 270M 的原始速度证据，不是历史 ESMC 融合数字：两段长训练的平均速度约 2.05 和 2.12 秒/新增 optimizer step，名义 global batch 都是 256。当前 V7 正式任务近期约 10.6–11.0 秒/step，差距确实值得定位。** 历史和当前的数据、padding、每卡 microbatch、通信、mask 和验证协议不同，这个差距还不能唯一归因于新增架构、loader 或占卡进程。

**旧 270M 与当前 f011 使用相同的 8 层 / d768 / H12 / FFN3072 Transformer 主干。** 旧模型大约 194M 参数来自大词表输入 embedding 与输出 head。新模型不是将主干缩小到原来的四分之一，也没有把主干加深或加宽。f011 的固定 residual scale 与 local/global 交替不新增可训练参数；但 attention bias 的稠密展开、反向和格式转换可能带来很大的运行开销，参数量无法反映它。

**旧任务也启用了 decoder activation checkpointing。** 不能只凭当前 whole-layer checkpointing 就解释这次相对旧任务的慢化。可以后续测关闭/选择性重计算的收益，但这属于待验证优化。

结构化证据：[review.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/llada270_review/review.json)，SHA256 `b1b0e98132adfdd0a013356378d4c2e5d136b6c044441fc2a1607d9d48fe5b28`。包含原始日志行、恢复步数、元数据 SHA、pickle 静态字段、实际 checkpoint 张量形状、源码 SHA 和理论公式。未执行 pickle 反序列化，未读取模型权重 payload。

## 真实历史速度

| 原始运行 | 恢复起点 → 进度显示 | 区间实际新增步数 | 显示 elapsed | 平均秒/新增 step | 同行 tqdm 近期值 |
|---|---:|---:|---:|---:|---:|
| 2026-09-23，bjnhqrdh | 0 → 36634 | 36634 | 20:54:19 | 2.054348 | 1.89 s/it |
| 2026-09-24，kalwnovw | 36000 → 178416 | 142416 | 83:56:37 | 2.121932 | 1.94 s/it |

计算使用 `elapsed / (observed_global_step - actual_resume_step)`，不是直接用恢复后的总步数作分母。名义 global256 对应 124.61、120.64 条/秒；历史存在超长记录丢弃，因此不是实际有效训练条数或有效 token 吞吐。平均包含这些运行区间内的验证、保存与其他等待，tqdm 近期值是另一口径。当前比较区间为 1000→2000 和 2000→3000 的完整保存事件，约 10.956/10.596 秒每步，含验证、保存及少量观察延迟。两者都不是 GPU kernel 计时。

原始日志和匹配的元数据：

- [20260923 output.log](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_llada270m_noesmc_diffusion_v2_immune_v6_2node4gpu_1m/wandb/wandb/run-20260923_135822-bjnhqrdh/files/output.log)；[metadata](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_llada270m_noesmc_diffusion_v2_immune_v6_2node4gpu_1m/wandb/wandb/run-20260923_135822-bjnhqrdh/files/wandb-metadata.json)。实际 argv 无 resume。
- [20260924 output.log](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_llada270m_noesmc_diffusion_v2_immune_v6_2node4gpu_1m/wandb/wandb/run-20260924_110447-kalwnovw/files/output.log)；[metadata](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_llada270m_noesmc_diffusion_v2_immune_v6_2node4gpu_1m/wandb/wandb/run-20260924_110447-kalwnovw/files/wandb-metadata.json)。实际 argv 恢复 checkpoint-36000。

Sep28 短重启日志中出现 21–180 it/s，不能当纯 270M 的真实训练速度：进度显示已经带上约 182000 的历史总步数，只经过几分钟；没有可比稳态区间。本报告排除这些数字。旧 ESMC 融合的 2.94–2.95 s/it 也不作为这里的 no-ESMC 原件。

## 实际配置对照

| 项目 | 历史纯 LLaDA270M | 当前正式 f011 |
|---|---|---|
| Transformer | d768 / L8 / H12 / KV12 / FFN3072，SwiGLU / RMSNorm / RoPE | 相同主干尺寸和基本算子 |
| 总参数 | 269,759,232 | 75,601,176 |
| 8 层参数 | 75,509,760 | 75,509,760 |
| 词表与独立输入/输出矩阵 | V126464，合计 194,248,704 参数 | V50，合计 76,800 参数 |
| 额外特征参数 | 无 role/scope/chain bias | role 9216 + scope 4608 + chain bias 24 |
| f011 机制 | 无 | residual scale 0.1；偶数层链内 radius32、奇数层全局 |
| GPU | 2 节点 × 4 张 A100-SXM4-80GB | 1 节点 × 8 张 A100 80GB |
| 并行 | FSDP、8 进程 | DDP、8 进程；主训练 NCCL |
| 每卡 batch × GA | 4 × 8 | 32 × 1 |
| 名义 global batch | 256 | 256 |
| 序列长度 | batch 动态 padding，上限 1024 | batch 动态 padding，上限 2048 |
| target 容量 | heavy/light 167/135 | full/fragment 256/64 |
| 数据 | V6 binding-only；自然采样 | V7 审核限定范围数据；自然采样 |
| loader workers | 4 | 0 |
| activation checkpointing | actual argv `decoder_grad_ckpt=True`，保留源码 hook 为 whole-layer | 冻结 driver 显式 whole-layer |
| 精度依据 | 请求 BF16；FSDP 保存的 safetensors header 为 F32；历史运行时逐算子 trace 缺失 | FP32 参数、BF16 autocast；safetensors header 为 F32 |
| 库版本 | Torch2.8.0 / Transformers4.48.1 / Accelerate1.14.0 / NumPy2.1.3 | 当前固定训练解释器使用相同版本 |

历史 W&B metadata 的 `gpu_count=4` 是记录日志的单节点可见卡数；另以历史提交账本的 2×4 拓扑、checkpoint-182000 的 `training_args.bin` 静态 pickle 字段 `num_processes=8`、`distributed_type=FSDP` 及 `rng_state_0…7` 交叉核实。实际两长段 argv 确认 micro4、GA8、workers4、max_length1024、token/no-ESMC。

当前保存配置来自 [checkpoint-3000/research_config.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_initial_training_20261005/run/checkpoint-3000/research_config.json)。旧配置来自 [checkpoint-182000/fusion_config.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_llada270m_noesmc_diffusion_v2_immune_v6_2node4gpu_1m/checkpoint-182000/fusion_config.json)。本次两份 safetensors 只读取 header，重新求和确认为上述参数量。

当前冻结 driver 显式要求 `not trainer.is_fsdp_enabled` 且禁用 DeepSpeed，因此当前是 native DDP + NCCL；并非 FSDP。本报告不把两者差别单独当作五倍慢化的因果解释。

**不能照抄当前旧 YAML 的全部参数当历史实际值。** 该 YAML 后来加了 `source_balanced_training=True`；历史实际 argv 没有这一项，[保存的 source_sampling.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_llada270m_noesmc_diffusion_v2_immune_v6_2node4gpu_1m/checkpoint-182000/source_sampling.json) 和 `training_args.bin` 都是 false。HF `bf16=False` 字段也不能单独解释为历史没有 BF16：该任务通过 Accelerate 的 FSDP/mixed_precision 配置和模型构造控制精度；保存张量 F32 只证明保存格式，不证明训练中每个算子的 dtype。

补查 tokenization：旧数据先以 ESM grammar 逐残基编码，再由 `RemapCollator` 逐位置查表变为 `<res_*>` 的 LLaDA ID，保持张量形状。实际 tokenizer JSON 是33词表、BPE merges=0，普通token均为单字符；没有把长序列送到自然语言BPE后合并压缩的证据。旧167/135是decoder token槽位，grammar构造按 `fixed_slots - len(residue_ids)` 填EOS，不只是字符串字符预算。旧、新实际训练的每批S分布在本次历史日志里均无充分记录，仍不能由上限1024/2048认定attention实际开销固定相差4倍。

补查输出 head：保留的旧 `_denoise` 直接调用decoder，LLaDAModelLM将完整hidden传给底层，未启用 `last_logits_only`；`ff_out(x)`在全部位置计算后才屏蔽不允许的目标token。未发现旧训练只给selected位置计算head的路径。它支持大词表head理论开销，但没有历史profiler，不能把当前可读源码当作完整历史执行trace。补充证据：[supplement.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/llada270_review/supplement.json)。

## 新增计算在哪里

1. **主干矩阵乘没有加大。** 固定同一 B、padding 长度 S、d/f/n 时，Q/K/V/O、FFN 和 dense attention 的矩阵乘规模相同。新输出 head 反而从 126464 缩到 50，理论输出矩阵乘明显减少。f011 固定残差 scale 是每层两个逐元素乘法；以 B32/S1024 为例约 0.403G 次乘法，而主干矩阵乘约 5772G FLOPs，不能从这两个乘法解释数倍慢化。
2. **可训练链 bias 是很小的参数、大的中间张量。** 24 个参数按 chain identity 展开为 `[B,H,S,S]` FP32 连续张量；有 padding 的 f011 前向还会保存 local/global 两种 masked bias。RefinedModel 已按拓扑复用，不是每层重新构建完整局部拓扑。每层进入 attention 时仍将 bias 转为 Q/K 的 BF16，且调用 `ensure_finite_`；whole-layer 重计算会再次经过这一层内路径。梯度也要从展开张量回传到 24 参数。这是明确的访存/表示开销，实际耗时仍待整模型 profile。
3. **local attention 仍是 dense SDPA 加 mask。** 四个 local 层没有改成稀疏/窗口实现，不能按 radius32 理论窗口复杂度计算当前实际 FLOPs。传入 mask 使显式 `flash_attn_func` 分支不满足 `attn_mask is None` 条件；这只证明调用 PyTorch SDPA，尚不能认定底层是 math、memory-efficient 或某个 flash backend。原普通 LLaDA 在无 padding 时可以无 mask，有 padding 时一般使用 head-broadcast bias；同样不能未经 trace 就指定其实际 backend。
4. **长度和 microbatch 同时变了。** target 槽位增大及 V7 context 会改变长度分布；micro4→32 更容易把短样本 padding 到同一批中的长样本。当前上限2048不等于每批固定2048。需要实际 `B×S`、有效 token 数和 padding 分位数，不能只按 cap 估测最终吞吐。
5. **数据与通信是独立待测项。** workers4→0、FSDP→DDP、跨节点→单节点，都改变运行方式。配置能确定潜在路径，不能直接当瓶颈归因。占卡进程是否退出由父 agent 核实实际进程和 worker 日志，本报告不凭 GPU100% 或文件存在推断。

当前关键冻结源码：[RefinedModel](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3/dllm/pipelines/immune_llada/research_refined/model.py)、[ArchitectureBlock](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3/dllm/pipelines/immune_llada/research_next/model.py)、[LLaDA SDPA/精度转换](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3/dllm/pipelines/llada/models/modeling_llada.py)。旧 wrapper 的当前可读源码不是已经封存的历史运行镜像，历史实际细节优先以保存产物/argv为据。

下面仅为理论张量大小，不是 CUDA 峰值；S 是实际 padded length。三份 FP32 bias 指 base+local+global 且该批有 padding 的情形，无 padding 时 global 可复用 base。未计中间 bool/int64 mask、autograd、BF16转换副本和其他 activation，不能把表格直接相加当峰值。

| B=32、H=12 的 S | 单张 FP32 bias | 单张 BF16 bias | base+两种 masked FP32 bias |
|---:|---:|---:|---:|
| 512 | 0.375 GiB | 0.1875 GiB | 1.125 GiB |
| 1024 | 1.5 GiB | 0.75 GiB | 4.5 GiB |
| 2048 | 6 GiB | 3 GiB | 18 GiB |

因此 **每卡64可以作为下一档候选，但现有27–29GiB快照不能证明64可运行**；batch翻倍会使相同S的bias翻倍，混批S还可能上升。八卡micro64×GA1是global512，与现global256不同；不能用step数量减半就声称端到端加速，也不能偷偷改变当前固定训练日程。

理论前向矩阵 FLOPs（一次乘加计2 FLOPs）为 `n×(8BSd² + 6BSdf + 4BS²d) + 2BSdV`。不计norm/RoPE/softmax/loss/bias构造、反向、重计算、optimizer、通信；也不是实测硬件执行量。

| 相同 B32、实际 S | 当前 V50 前向矩阵 TFLOPs | 旧 V126464 前向矩阵 TFLOPs |
|---:|---:|---:|
| 512 | 2.6813 | 5.8626 |
| 1024 | 5.7750 | 12.1376 |
| 2048 | 13.1992 | 25.9244 |

这说明“当前增加了数倍主干矩阵计算”没有源码依据；实际吞吐下降更应检查长度/padding、attention实现与内存流量、loader/同步，而不是仅看总参数或固定scale。

## 最少的公平整模型对照方案

这是一份后续执行设计，本轮未运行，也不恢复暂停的独立 CUDA 算子专项。使用相同 A100、固定软件环境，在既有训练允许的边界及确认可用资源上做短整模型测量；不热改当前冻结模型。模型测试只保留计时与必要trace，不产生测试权重。

1. **先分离 pipeline 与模型。** 对同一当前 f011，比较正常 prepared loader 与相同、预先完成CPU collation的批次回放，使用完全一致的形状、mask 和更新数。记录数据等待、H2D、forward、backward、optimizer、rank等待/all-reduce及实际attention backend。CPU回放输入需与正常路径相同，不能用短人工样本替换长真实输入后声称loader是瓶颈。审核范围和context loss约束保持。
2. **固定 V7 形状比较三种整模型。** A：普通 LLaDA d768/L8/H12/3072、V50，仅padding mask；B：f000，保留role/scope/chain-local位置及chain bias；C：当前f011。固定小词表、B、S、精度、checkpointing、训练目标/选择位置、optimizer和既定批次。A→B隔离当前链特征表示路径总开销，B→C隔离local/global与固定scale增量。A是通用工程速度control，不伪称与旧v2语义相同。原V126464 270M可单加第四组区分大词表head成本，但不能拿不同布局和词表的总loss评价效果。
3. **只有已定位后再改一项。** 如模型时间占主导，先测相同f011的checkpointing开/关或选择性重算，仍用相同B/S；如loader占主导，再比较workers；如padding占主导，再测分桶；如dense bias/attention占主导，再评估保持数学语义的表示复用和backend兼容性。一次改变一个因素，不能同时改模型、batch、数据和精度。
4. **micro64独立记账。** 先记录实际长度分位数及训练峰值，再比较32/64的记录/秒、有效token/秒、峰值和OOM。在单卡整模型效率测试中，可用micro32×GA2与micro64×GA1固定global64；八卡则对应global512，不能冒充当前global256的完全同预算训练。实际晋级前须明确新的globalbatch、曝光/调度与原恢复边界。

每格建议5–10个预热更新、至少30个稳定计时更新，三个短窗口；如时序波动大再延长。保存median/p90、总有效tokens/秒、完整optimizer step时间与显存峰值，单列eval/save耗时。只选少量代表长短批次，必要时用shape相同的人工整数输入复查算子路径，但人工结果不能替代真实数据pipeline吞吐。不得占用当前8张正在训练的卡来并发profile，制造新的竞争后再将结果当原瓶颈。

完成归因的证据应是：同一配置的时间分解、实际进程清单、matched control的吞吐变化，以及数值/更新契约仍成立；历史2秒与当前11秒的差距是排查线索，单独不足以决定停容器、换任务形式或更改正式训练。

## 本轮工程记录

一次临时只读日志regex耗时超过初始工具等待；改成逐行筛选后在0.5秒内退出0，原临时reader PID4171826及父shell4171822随后实际确认均已不存在，没有发送信号。原等待句柄未保留，故不虚报该reader的退出码。它不是模型测试/训练失败，没有创建ckpt或改动训练。

现有历史[容器执行入口审阅](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_EXISTING_POOL_DIAGNOSTIC_EXECUTION.md)也未确认CLI exec/SSH：记录的是公开CLI无exec、平台WebShell可能可用但实际权限/入口未验证。该文的独立attention probe方案属于历史内容，本轮未恢复、重试或部署它。无新的远程命令、占卡任务或已确认的可用容器shell。


## 2026-10-06T04:39:59.360843+00:00 分段边界的额外工作已确认

冻结driver在5016段尾额外执行评估与保存，然后由原监督接续并读取原生恢复状态。5000是常规评估点，5016段尾结果另行记录；不能把16步间隔或接段加载时间当作稳态GPU吞吐。4000→5000的既有区间计时保持，当前仍没有GPU组件profile。8rank退出/恢复完整；5条外层清理扫描提示及无测试权重残留的证据见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/segment5016_root_review_20261006.json。未据该提示归因训练慢化或改变运行资源。
