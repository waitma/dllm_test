# V5 与当前 V7 正式训练的代码对比

## 2026-10-07T05:08:27.070594+00:00 当前正式两轮版本与V5：模型和数据差异

本节核实的是V5 ESMC融合159000对照与已完成两轮的V7 f011/50082。后文2026-10-06早期的运行速度、保存和训练阶段属于历史，不代表当前状态。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/v5_v7_model_data_comparison_20261007.json 绑定V5实际argv、参数header复核、V7原生checkpoint配置/训练来源和两版manifest；没有读取原始序列、加载权重或执行注释分支。

| 项目 | V5 | 当前V7 f011 |
|---|---|---|
| 预训练来源 | LLaDA decoder随机初始化；预训练ESMC-300M参与联合训练 | 不使用ESMC，单一模型随机初始化 |
| 主干尺寸 | 8层、768宽、12头、FFN3072 | 相同；参数减少不代表主干层数/宽度缩减 |
| 参数 | 共603,495,616：decoder269,759,232＋encoder332,997,184＋条件层739,200 | 75,601,176；去掉ESMC和大词表矩阵是主要差异 |
| 词表 | 126464行输入embedding/输出head | 50个氨基酸/grammar专用token |
| 链与位置 | decoder全局RoPE，旧grammar/链顺序和encoder对齐 | 链内重置RoPE，独立role/scope embedding，同/异链bias |
| 注意力/残差 | 全局双向attention，普通残差 | 局部radius32与全局层交替；残差固定0.1；未启用动态head gate |
| 布局 | 旧FR/CDR scaffold，缺失区域可用synthetic X补齐 | A4按metadata提取真实观测片段，保留junction C/F/W；prepared中仍可留旧X字段 |
| 长度 | batch动态padding、上限1024 | batch动态padding、上限2048；target仍有full256/fragment64容量和单个protd终止监督 |
| 损失 | 旧grammar非fixed、非synthetic目标位置，含相应结构/关系标记 | target真实残基＋单个protd＋已有正例关系；context残基和余槽不算CE |
| 并行/训练量 | FSDP，micro4×GA8×8卡，比较点159000 | 实际DDP，micro32×GA1×8卡，完成50082/两轮；均名义global256 |

两版prepared都采用immune_llada.semantic.v1 JSONL及同七类来源。V5目录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v5_receptor_completion；本次正式V7实际目录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_scope_final_20261002，不是早期中间candidate。V5 train/valid为7,626,885/102,308，正式V7为6,410,749/49,528；train减少1,216,136条（15.95%）。各来源训练行数直接从manifest分片汇总：

| 来源 | V5 train | 当前V7 train |
|---|---:|---:|
| oas | 2,485,433 | 2,472,502 |
| ots | 2,094,218 | 1,700,950 |
| asd_antibody | 276,412 | 168,767 |
| trait | 31,515 | 2,299 |
| tcr_native | 96,552 | 12,263 |
| tcr_papers | 642,755 | 114,422 |
| tcr_repertoire | 2,000,000 | 1,939,546 |

主要处理差异：V7继承binding-only配方，指定带标签来源去掉明确nonbinding；未标注序列库不因此成为已验证binder。另有语义去重、受体家族分组划分及后续下游保护范围清理，故记录数减少不能全归于某一条过滤。Foundation train/valid发生过重新分组，不代表改动benchmark作者划分。当前自然epoch采样，未按来源平衡；小样本试验增补context不等于正式数据另行加权。

MHC采用已有的sequence-first处理：保留实际提供的序列/多链信息；只有名称且能唯一解析时补pseudo，保留缺失状态和来源。具体历史转换及边界见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md 的v7构建与MHC章节。正式training_ready=true，但证书只覆盖immune_test_only_observed_pair_exact_v3，不能扩展为所有query/few-shot或相似度保护。

以上同时改变了预训练知识来源、词表、表示/监督、语料组成与训练预算，因此现有V5/V7效果对照不能单独衡量“去掉ESMC”或f011某个因子的作用。这里没有变更模型或数据，原消融继续；正式best3 45000/46000/47000＋latest50082保持完整状态。

## 2026-10-06T11:41:39.650760+00:00 补齐已有GPU逐参数梯度证据：仅where＋math路径

已读取并核对既有真实V7单卡记录，未重跑GPU。75,601,176参数f011、一个B32/S516的原生输入，原版与where版复制相同权重并恢复相同CPU/CUDA RNG，固定math attention与BF16 autocast；loss均3.8769800662994385，完全相等。78个有梯度参数张量逐元素检查通过，rtol=1e-3/atol=5e-6；只有chain_bias有非零差异，最大绝对误差2.7315691113471985e-6、相对L2误差0.00017428879800718278。其余77个有梯度张量记录为完全一致。

实际执行脚本SHA与控制器记录一致，where子进程returncode0；没有保存梯度或测试权重。本证据补充上一条“标量不足以证明完整梯度”的限制，覆盖的是一个初始化状态下的原生batch及where＋math路径，不能扩展为where＋efficient、多长度/后续优化器状态或八卡FSDP恢复验收。当前正式运行与交付FSDP绑定源码未在此轮修改；后续接入仍应保持独立版本及原数据/恢复门禁。

根核实 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/native_gradient_equivalence_root_review_20261006.json；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/local_verified_profile/native_v7_attempt1/where/gpu_gradient_equivalence.json（SHAba96064c598c8db439cd97e7052bb04faa65c5d4f442c2ded91bda9f4124c765）。测速均值和原单卡范围保持，不把该梯度检查当作新吞吐测试。



2026-10-06T11:35:49+00:00更新：本地完整模型CUDA profile和真实V7回放已确认主因是chain_bias重复索引反向。仅where展开替换，真实批次6.6288→0.5450秒/更新；工作区已修复，通过GPU梯度及64项回归。原冻结任务未切换。详见 [GPU根因报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_SPEED_ROOT_CAUSE_20261006.md)，下文未实测内容保留为早期历史。

## 2026-10-06T11:33:52.924409+00:00 真实V7单卡测速已有原件：主要热点为chain bias索引反向

复核共享工作区既有独立控制器的三个case：均returncode0、passed，原始model与where/where+efficient共用75,601,176参数、B32、BF16 autocast、whole-layer checkpoint、seed42和相同输入来源/选择/顺序。使用既有1024条前缀探针中的8个固定batch，长度516–867；8次total-only计时均值已从原始记录重算。

| 实现 | 单卡整步均值(秒) | 分阶段反向均值(秒) |
|---|---:|---:|
| 原始索引展开 | 6.628798 | 6.451850 |
| where展开 | 0.545025 | 0.384400 |
| where＋efficient SDPA | 0.310286 | 0.242226 |

原始实现实际使用math SDPA；被profile的batch有一次索引梯度kernel约8.81秒，where后该热点消失。合成B32/S516对照中，原始2.140秒、仅efficient2.015秒、仅where0.328秒，支持优先处理bias索引反向。不能仅用换FSDP解释慢化。

这是单卡完整模型重放，不含CPU组batch、稳态H2D、DDP、验证与保存；8个固定batch不是全量分布，不能直接宣称八卡正式加速12或21倍。19次GPU scalar loss最大差where约0.000193、where+efficient约0.001207；标量及梯度范数不能证明完整GPU梯度向量相等。后续接入需独立版本的数值、分布式与恢复验收。原正式冻结模型、当前任务和已交付FSDP绑定源码均未由本监督改动。

原控制器11:27:21结束并写RESUMED.json，本监督实际/proc核实PID80642/start2452750909恢复为R。暂停窗口不是训练失败，也不用于消融吞吐对比。本监督只读取既有结果，未发送信号或重复GPU测试。根复核 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/native_speed_evidence_root_review_20261006.json（SHAb38e5aca907c546f2084a2e489e2fd25ca3e15132855edaf059d7a76d866eae1）；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/local_verified_profile/native_v7_attempt1。旧“尚无GPU实测”由本条更新，独立CUDA专项和既有安全停线仍保持。


核实时间：2026-10-06T09:41:42.093903+00:00。当前正式 f011、单机八卡 DDP；V5 ESMC 融合确实采用 FSDP。此次完成训练路径的23份关键源码与原始运行配置对照，没有改动正在运行的模型、训练入口、数据或恢复状态。

## 证据范围

V5 的两个 W&B 元数据记录 commit d1d1076b0db5fd0d8dd59869b6799fe137b4c77e、d3daa3585669be0b0aebb6e5a3d474347b35cfbb；本次选取的23个路径在两版本相同。旧版已提交源码保存在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/v5_code_review_v1/v5_committed_sources。当前对照版本为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3，实际存在的每份文件均核对冻结 source_manifest_v3.json 中的 SHA。

历史 W&B 目录没有未提交工作区补丁，旧 accelerate YAML 也不在这两个 git 版本中。因此“旧源码对照”指记录 commit 可复原的版本，不能证明历史运行时没有额外未提交修改。FSDP、8进程、micro4、GA8、workers4 则由保存的 training_args.bin 静态字段独立证实；没有执行 pickle 反序列化。

[完整文件清单与哈希](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/v5_code_review_v1/inventory.json)；[文本差异](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/v5_code_review_v1/runtime_sources.diff)；[本次根复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/v5_code_review_v1/root_review.json)；[实际速度原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/v5_review/review.json)。

差异文件还包含冻结目录保留的旧 fusion 入口后续变化；**这些旧文件存在不代表当前 f011 会执行它们**。当前实际调用链见下表。没有审阅或恢复已停止的序列注释分支。

## 实际执行链路与全部关键差异

| 环节 | V5 ESMC 融合 | 当前 V7 f011 | 对比较的影响 |
|---|---|---|---|
| 正式入口 | protein_pretrain_esmc.py → FusionTrainer → LLaDAEsmcFusion | run_v7_formal.py → EightRankTrainer/FormalBaseTrainer → FactorialModel | 当前不是在旧入口上仅删掉 ESMC |
| 并行 | 保存参数确认 FSDP、world8 | 入口和评估适配器明确只支持 native DDP，主训练 NCCL | 切 FSDP 不能只改一行 YAML，还涉及评估和恢复适配 |
| 模型组成 | 可训练 ESMC、condition LayerNorm、960→768投影、LLaDA decoder | token+role+scope embedding、LLaDA backbone、同链/异链 bias | ESMC 条件网络被移除，但新增表示路径 |
| 总张量参数量 | 603,495,616：decoder269,759,232、encoder332,997,184、condition739,200 | 75,601,176 | 参数大幅减少不等于 attention 中间张量减少 |
| Decoder 尺寸 | 8层、768宽、12头、FFN3072 | 相同尺寸，SwiGLU/RMSNorm | 主干没有加深或加宽 |
| 词表 | 126464行输入 embedding 和输出 head | 50行 | 输出矩阵规模显著下降 |
| 输入编码 | grammar/ESMC token逐位置映射为LLaDA token；另建encoder输入 | A4紧凑grammar，直接供单一backbone使用 | 不再计算ESMC条件前向和反向 |
| 位置 | V5 decoder调用未传链内position_ids，使用全局RoPE；链信息用于encoder条件对齐 | 各链显式position_ids，链内重置RoPE位置 | 这是模型语义和实现路径的变化 |
| 链特征 | 没有当前独立的role/scope embedding和可训练同链bias | 12×768 role、6×768 scope、2×12 bias | 当前新增的可训练参数13,848个 |
| 注意力范围 | decoder全局双向attention+padding mask | 第0/2/4/6层同链radius32，第1/3/5/7层全局 | 局部层仍调用稠密SDPA，并未实现稀疏窗口计算 |
| attention bias | V5 fusion不向decoder传可训练chain bias | 同/异链参数展开成[B,12,S,S]，当前forward复用local/global各自的bias | 大中间张量、类型转换及反向归约是待测开销 |
| 残差 | 普通attention/FFN残差 | 两条残差分支都乘固定0.1 | 固定buffer，无新增可训练参数；不是动态门控 |
| 其他研究机制 | 无当前factorial研究开关 | f011未开启head gate、VICReg双视图、额外卷积、register、learned LayerScale | 不应把这些未启用机制算进当前正式成本 |
| 输入长度 | 原commit renderer按已有记录长度布局，batch动态padding，上限1024 | A4目标容量256/64，再按batch最长记录padding，上限2048 | V5不是后来的noESMC-v2固定167/135槽位版本，不能混用 |
| 噪声 | 吸收mask；实际joint1、其他链任务比例0；encoder同步遮住被预测位置 | 同类吸收mask sampler，只有当前eligible位置参与 | 当前无需encoder corruption镜像 |
| 监督位置 | 非fixed且非synthetic目标可参与，包含相应grammar结构/关系位置 | target真实残基+单个终止<protd>，以及已有受监督关系token；context残基和target余槽label=-100 | loss数值不属于同一完整目标空间，不能直接横比优劣 |
| CE | V5默认选中token均值，reshape为[BS,V] | 仍选中token均值，但transpose为[B,V,S]并转FP32；target位置限制为残基/终止词表 | 数学目标与布局均有变化；不能直接用loss倍数解释速度 |
| microbatch/GA | 4×8×8=global256 | 32×1×8=global256 | 每步名义记录数相同，但padding、同步次数与micro内token权重不同 |
| 精度 | 已提交源码将融合模型转BF16；保存header为F32；逐算子历史trace缺失 | FP32参数，BF16 autocast | 不能用保存F32或TrainingArguments.bf16=False推断旧版全FP32计算 |
| activation checkpointing | 实际argv decoder_grad_ckpt=True，旧whole-layer路径 | 显式whole-layer，包裹ArchitectureBlock | 两边都重计算，不能单凭此项解释新增慢化 |
| 数据读取 | prepared JSONL索引、按需seek/parse，按来源ConcatDataset，workers4 | 同类索引/读取，增加source_group；自然epoch sampler，workers0 | source_group是元数据，当前未启用source balancing |
| 日志 | 20步一次，W&B | 25步一次，无W&B；每个microstep先把5个component scalar转Python float | 当前有额外device→host同步点，耗时未实测 |
| 验证 | 随机噪声loss、按来源，原入口读取完整prepared valid | 固定1330条×4mask=5320记录评估，token/source聚合和END诊断 | 评估协议不同；当前6000点记录eval_runtime=4.594406秒 |
| 保存保留 | 最优3个+最新；非最新保留点slim为主要权重 | 最近3个完整可恢复checkpoint，不按最低loss挑选 | 当前完整恢复占用更稳定，可回退最近3个点 |
| 保存核验 | HF/FSDP保存+top-k/slim回调 | HF/DDP保存后8rank参数一致、保存参数一致、文件SHA和恢复字段核验 | 是保存阶段额外工作，不是每个step都跑 |
| 接续 | HF/FSDP旧checkpoint状态 | 固定总25042日程，分段原生恢复optimizer/scheduler/8rank RNG | 启动/接段有额外核验，不能计成纯GPU稳态速度 |

核心源码入口：

- V5 [Trainer与初始化](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/v5_code_review_v1/v5_committed_sources/examples/llada/protein_pretrain_esmc.py:213)、[融合前向](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/v5_code_review_v1/v5_committed_sources/examples/llada/protein_fusion_model.py:427)、[CE](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/v5_code_review_v1/v5_committed_sources/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py:764)。
- 当前 [正式Trainer](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3/scripts/research/run_v7_formal.py:305)、[f011选项](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3/dllm/pipelines/immune_llada/research_refined/factorial.py:26)、[前向bias复用](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3/dllm/pipelines/immune_llada/research_refined/model.py:41)、[attention/残差](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3/dllm/pipelines/immune_llada/research_next/model.py:44)、[训练loss](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3/dllm/pipelines/immune_llada/research/model.py:236)。
- [每microstep标量日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3/examples/llada/immune_pretrain_v3.py:71)、[八卡评估支持边界](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3/output/generic_eight_rank_trainer_integration_20261002/eight_rank_trainer_adapter_v1.py:101)。

## 2048并不是统一padding长度

[ImmuneCollator](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3/dllm/pipelines/immune_llada/v3/layout.py:110)先建立各条记录，再取本batch的最大长度。2048只在最终检查中作为上限；超过就报错，不会无条件补到2048，也不静默截断。

需要区分三层长度：

1. 单条原始链的真实残基长度。
2. 模型中的整条记录长度：context、多条链、grammar标记，以及每个target的容量。当前完整target容量256、片段64，均含终止位置；余槽保持mask并参与attention，但不计算CE。
3. batch张量长度S：这一组记录中最大长度。较短记录才在最外层补PAD。

例如没有context的两条完整target，当前通常为2个前缀标记+2×(1个prots+256个槽)=516；不会因为max_length=2048而再增到2048。这个516也不等于两条真实序列的长度。

复用已完成的1024条CPU诊断，不新增读取或测试：micro32共32个batch，S范围516–1206，17个为516；排除2个预热batch后的30个S的中位数539、p95=1130.45。该有限样本非全量长度分布。真实来源条件和多链拼接可能超过1024，本样本已实际出现1206，所以直接把上限改1024会拒绝部分现有记录，而不是自动消除空槽或安全提速。

同批次外层PAD占总张量位置43.58%；target剩余空槽占有效attention位置53.08%。两个比例分母不同，不能相加，也不能直接当GPU时间占比。当前更值得诊断的是固定容量和混长batch，而不是把未实际用到的2048上限当作每步计算长度。

完整变长、取消容量/终止监督属于训练语义更改；此次仅核实当前行为，没有实施。原始诊断：[result.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/data_loader_review/result.json)。

## 2026-10-06T10:01:03.673427+00:00 更正“统一padding到1024/2048”的说法

先前把最大允许长度与实际padding长度混称，造成误导。V5实际运行上限1024，其记录commit的collator取本batch最长记录再补PAD；当前上限2048，实际仍取本batch最长记录。2048不是所有输入的张量长度，也不能据此解释训练慢。当前长记录需要较高上限，但尚无证据证明2048这个具体数值是唯一或最优选择。

复用既有1024条有限CPU诊断样本，不读取原始训练序列：4条渲染后记录超过1024，长度1090/1109/1148/1206，涉及32批中的3批；17批的实际长度为516。渲染长度包含context、多链、grammar和目标容量，不能说成单链有1206残基；该样本不代表全量比例。

只比较其余29个可放入1024的原批次，当前动态padding共568224个张量位置；强制统一1024会变成950272个，即1.6724倍，稠密attention矩阵元素数量为2.6607倍。这仅为几何量计算，不能当实测时间或速度倍数。详见 [padding1024_existing_geometry.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/v5_code_review_v1/padding1024_existing_geometry.json)。

1024可作为后续短输入上限候选，但需明确长记录处理；优先评估动态padding与按长度分组，而不是把所有短批次也扩到1024。长记录可保留独立批次及较高上限；若改变目标容量/布局，需要另行验证，不能直接截断或丢弃记录。当前冻结运行、数据与证书均未改变。本轮仅更正文档与整理已有证据，无新模型测试；不恢复任何已停止分支。

## 当前checkpoint保存、验证和清理

冻结入口设置save_steps=1000、eval_steps=1000、save_only_model=False、save_total_limit=3。达到分段stop_after或总max_steps时，[on_step_end](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3/scripts/research/run_v7_formal.py:224)额外请求eval/save；因此已有5000后紧接5016的分段恢复点，后续10016等边界也采用同一逻辑。

模型及共享状态由rank0写入，各rank保存自己的RNG。保存内容为模型权重、optimizer、scheduler、8份RNG、Trainer/TrainingArguments、模型/词表/研究配置与run_provenance。随后8rank同步，在[on_save](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3/scripts/research/run_v7_formal.py:231)比较运行参数摘要、rank0保存参数摘要，并核验17项原生文件及恢复字段，成功后写formal_checkpoint.json。

保存回调确有额外CPU/I/O工作：各rank计算文件哈希，读取optimizer/scheduler/RNG，核验前后inventory；另有参数device→host摘要。这部分不是只写一个模型权重文件，也不发生在每次普通optimizer step。当前没有单独的该回调计时，不能把全部速度差归因于它。6000点记录的eval-loop 4.59秒只覆盖验证循环，不包括全部保存核验、同步及接段。

2026-10-06T09:41:42.093903+00:00实际保留：

| 点 | 字节数 | 约GB（十进制） | 恢复状态 |
|---|---:|---:|---|
| checkpoint-5000 | 908096081 | 0.908 | 完整 |
| checkpoint-5016 | 908159798 | 0.908 | 完整 |
| checkpoint-6000 | 908238233 | 0.908 | 完整 |

目录：[正式训练run](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_initial_training_20261005/run)。三份合计约2.724GB。新checkpoint正常轮换后删除最老完整目录；这里保留的是最近3个，不是验证loss最优3个。旧点必要的小型JSON证据由原监督保留在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint_evidence，不额外复制权重。测试消融仍为metrics-only，未产生训练权重，不把正式恢复状态当测试垃圾删除。

V5是另一策略：[TopKValLossCheckpointCallback](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/v5_code_review_v1/v5_committed_sources/examples/llada/protein_pretrain_esmc.py:353)保留最优3个与最新点的并集，最多4个；只有最新点保留完整恢复状态。当前未运行这个V5 callback，不能据其代码推断V7也在slim或保留best3。

## 性能判断及待验证项

实际V5两段墙钟均值2.734181/2.731451秒/更新；当前两个保存区间10.926703/11.170567，约4.00–4.09倍耗时，两者global256。差距真实；[速度报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_LLADA270_SPEED_COMPARISON_20261006.md)保留原始日志和计算。多天/千步区间、数据和评估开销不同，未完成同输入GPU控制实验。

从代码看，优先检查以下工程路径：

1. **稠密可训练bias及其反向。** 2×12参数索引展开到[B,12,S,S]，再构造local/global masked副本；梯度需归约回小参数。V5无此可训练decoder bias路径。仅参数量小不能排除大开销。
2. **新增层内host同步。** [显式RoPE分支](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3/dllm/pipelines/llada/models/modeling_llada.py:419)每层检查GPU张量并执行int(position_ids.max())，再按位置gather正余弦；whole-layer重计算还会重走。旧版按张量shape获得key_len，没有这一显式位置读取路径。每microstep的component float、forward的torch.equal/布尔检查也会引入同步。已确认同步代码位置，尚未计时其实际占比。
3. **有效长度与padding。** 2048不是固定长度；固定target槽位及micro32混长带来的padding是已观察到的现象。可考虑在未来独立实验中保持同一目标/曝光，评估批次组织或数学等价实现；不能用泄露真实结束位置的方法“省掉”训练槽位。
4. **DataLoader/通信及定期保存分开测。** 既有本地CPU getter+collate中位37.46毫秒，不是远端8rank吞吐；FSDP→DDP差异、共享GPU其他进程影响和保存核验耗时都缺独立计时，不能一概排除或认定。
5. **FSDP作为对照变量。** 当前75.6M能在DDP内训练；FSDP能分摊状态，也有通信开销。现有入口/评估/恢复契约禁止FSDP，若后续比较必须准备独立兼容入口并完成相应验收，不能用绕过检查来“切回”。本轮不更改并行方式。

已有整模型计时与输入回放入口仍处于GPU未执行状态；不重跑已通过CPU控制检查，不挤占当前八卡训练，也不恢复独立CUDA算子测试。此次完成的是代码对照与新增证据整理，不能把它称为已定位并修复4倍慢化。

## 实验标签纠正与当前状态

正式与本地冻结factorial注册表均明确：f010=R24固定残差0.1，f001=R26局部/全局radius32，f011=二者组合。旧文档将f010写成R26的6处描述已纠正（4份md）；原指标、f编号、固定图和运行模型均未改变。[纠正依据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/v5_review/factorial_label_correction.json)。不要继续把f010的500/750点改善归因于局部/全局注意力。

最新正式日志6725/25042、已核实验证/保存6000；本地f010/42训练850/3750、已核实验证750。原PID/start/argv及8rank新鲜心跳匹配，无abort；完整22格仍9格完整epoch产物（8格成功退出已核实、f000/42退出码未知），1运行、12未启动。下一正式7000、本地1000。原Goal未完成，保持active。

## 2026-10-06T10:17:16.991407+00:00 调试补充：普通训练慢化、后端限制与重复开销

复核实际冻结源码7份SHA、归档PyTorch 2.8.0源码3份SHA及正式provenance的torch=2.8.0；仅重算已有日志/长度元数据，没有运行模型或读取原始训练序列。证据和计算口径：[steady_windows_and_geometry.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/v5_code_review_v1/steady_windows_and_geometry.json)，SHA3029bd4e54cd5b1989268c56ac6649ecb81cbd27b3f27655d09c8810d503f009。

### 避开验证/保存边界后，普通训练仍然慢

从监督JSONL提取已记录step第一次被观察到的时间，选择远离1000步验证/保存和5016接段边界、所见checkpoint保持不变的区间：

| 区间 | 所见checkpoint | 更新数 | 墙钟秒/更新 |
|---|---:|---:|---:|
| 5475→5875 | 5016 | 400 | 10.1991 |
| 6125→6850 | 6000 | 725 | 10.6956 |

这是普通训练区间墙钟时间，仍含取数、CPU、计算和DDP等待；日志每25步记录、轮询造成端点误差。它说明约11秒/步不能主要用定期保存解释，不代表已经排除普通训练内的I/O或通信问题。

### 优先检查attention后端和dense bias反向

实际冻结 [构造函数](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3/dllm/pipelines/llada/models/modeling_llada.py:1131)调用`enable_mem_efficient_sdp(False)`，正式训练路径未发现后续覆盖。**这行V5也有，是继承的设置。** V7的差异是每层传入需要梯度的dense chain bias，进一步限制其他融合后端。

归档PyTorch 2.8.0选择器中，Flash拒绝非空attn_mask，cuDNN拒绝requires_grad的attn_mask；memory-efficient已被关闭。当前whole-layer采用non-reentrant checkpoint，保留bias梯度。因此记录的版本/源码条件指向**训练回退到math SDPA，属于源码推断，尚无实际CUDA kernel trace**。V5没有可训练decoder chain bias，但历史实际kernel未知，不能宣称V5一定走Flash或把全部4倍差距归因于后端。

另有两项确定的额外工作：

- [RefinedModel](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3/dllm/pipelines/immune_llada/research_refined/model.py:51)把2×12参数展开到`[B,12,S,S]`，再构造local/global副本；反向最终归约回24个参数。B32、S1024时，一份FP32张量1.5 GiB，base/local/global三份4.5 GiB。这只是张量大小，不是实测峰值，S也不是每批固定1024。radius32局部层仍调用稠密SDPA。
- mask按local/global复用后，[ArchitectureBlock](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3/dllm/pipelines/immune_llada/research_next/model.py:68)仍逐层调用`_cast_attn_bias`，重新将FP32转BF16并扫描/处理负无穷；whole-layer重计算再次执行。当前未缓存转换后的bias，不能把mask构造缓存视为消除了这一开销。

显式RoPE的GPU布尔检查和`int(position_ids.max())`位于每层内部，每microstep还有5个component float转换。这些同步点和bias的实际耗时仍需整模型分项计时，不能仅凭调用次数认定主因。

### 相同V7记录下，micro32放大混长padding

仅使用原1024条诊断记录已存的渲染长度，对比原顺序的micro32、micro4，以及按长度排序的micro32；没有重新取数、训练或更改sampler。

| 长度分组 | padding后token位置 | 单head稠密attention元素 |
|---|---:|---:|
| 原顺序micro32 | 679040 | 493794432 |
| 原顺序micro4 | 549432 | 310961632 |
| 按长度排序micro32 | 407840 | 226663328 |

同顺序micro32比micro4多23.59%的token位置、58.80%的稠密attention元素；原micro32的attention元素数为排序micro32的2.1785倍。**这是有限样本几何量，不是实测加速、V5真实长度分布或已验证的训练替换。** micro4/GA8增加调用次数并改变当前micro内token均值权重，排序改变顺序/RNG轨迹，均需单独比较。

当前target容量256/64的余槽参与attention，外层动态PAD也有开销；2048只是上限。按真实终止位置省掉目标余槽会改变结束位置学习问题，不作为数学等价的提速手段。

### 修复验证顺序和边界

先在独立整模型对照核实实际SDPA kernel、bias反向和数据/通信等待，再比较保留bias梯度的memory-efficient路径。随后验证forward内local/global bias转换复用、位置检查和缓存长度计算移出层循环、只在记录时读取日志标量；各项先验前向/loss/梯度一致性，再报告计时。长度分组和micro/GA调整独立比较。

本轮本地A100为100%利用率、64300 MiB已用，原训练继续；已有整模型profiler尚未在GPU执行。本次没有改动冻结源、重启任务、生成测试ckpt或恢复暂停的独立CUDA专项。已找到具体低效路径，尚未完成4倍慢化的算子级归因或提速验收。

## 2026-10-06T10:54:21.144337+00:00 速度归因答复：具体低效路径已找到，主因占比尚未实测

普通训练区间约10.20/10.70秒每更新，V5历史整体约2.73秒；协议和数据不同，约4倍只描述现象。保存不能解释全部慢化，2048是上限而非每批固定padding。优先检查attention：memory-efficient被关闭、可训练dense bias限制其他融合后端，按PyTorch2.8源码推断当前训练使用math路径；尚无实际kernel trace。局部层仍稠密计算，逐层bias转换/扫描、重计算以及GPU标量转Python的同步均是已发现的额外工作，耗时占比未知。

原1024条本地CPU热缓存样本micro32取数+collate均值38.75ms，不能据此完全排除生产共享存储等待。既有8份standby日志在训练开始时停止，代码先停并等待standby；实际远端进程清单缺失，不能称已逐进程排除，目前没有证据将占卡程序视为主因。长度混批与容量增加计算，几何比例不等于时间比例。

下一证据是整模型GPU trace和取数/训练/通信/保存计时，再对最高耗时项逐项验证。已交付2×4 FSDP入口包含计时、CPU工程验收通过，GPU尚未运行；本次输出目录仍不存在。FSDP不是已验证的提速修复，跨节点通信和精度路径也影响比较。当前无架构更换、冻结源码修改、任务提交或重启；独立CUDA专项保持暂停。

依据：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/v5_code_review_v1/steady_windows_and_geometry.json；/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/data_loader_review/result.json；/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/root_readonly_followup_v1.json。官方后端约束：https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp 与 https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/aten/src/ATen/native/transformers/sdp_utils_cpp.h 。本条整理现有证据，不是新增模型实验。
