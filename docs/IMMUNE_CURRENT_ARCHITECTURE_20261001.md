# 当前 no-ESMC 架构改动与验证状态

更新：2026-10-01 05:52 UTC。v3 R20完整GPU工程验收及根端强复审通过，3/24/768；T20私有准备完成，数据仍未准入，正式训练尚未启动。

## 2026-10-01 05:52 UTC v3完整GPU工程验收通过与剩余数据准入

05:44:51 UTC，v3 D01 1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959 在原GPU3终态succeeded/rc0。根端05:48以worker同一Python的全新-I进程执行原不可变verify_completion/verify_candidate，通过全部raw/summary/COMPLETES/installed source/独立CPU计数重算；禁止本地CUDA初始化，未重跑GPU。runtime manifest仍cc13036fc492109afbf4c9cf9fd11a4c87684fc9702b83ce86c961b2a08294fe，524父文件不变。

实际正式3窗/24 microsteps/768人工记录，formal1054.3747秒<1800，训练loss3.450910→3.448508→3.380331；首窗LR0符合固定warmup，后两窗LR1e-6/2e-6且参数确有更新。独立full oracle706.9962秒<1200、GA/CE/RNG比较通过；四mask评估5.1172秒<600、每mask24条/共96次，eval_batch16含实际尾批及完整2048输入，指标由raw packets独立重算。正式峰值allocated45,350,397,440字节（42.24 GiB），reserved56,555,995,136（52.67 GiB），原85%标准通过；tiny/full/formal清理弱引用释放、最终allocated0。parent1794.0503秒<3600，child rc0、正常退出观察/后代清理true。[强复审原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/terminal_review.json)、[GPU通过原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959/GPU_RELEASE_COMPLETE.json)。

两个自有空hf目录已删除，checkpoint=0；raw20,049,785字节、summary/日志/完成哈希是后续strong verifier及训练入口依赖，按必要证据保留，v1/v2失败原件不覆盖。[清理清单](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/gpu_validation_cleanup.json)。首次root提取曾把字段读到不同COMPLETE schema，读取错误已保存在review中，修正后通过，生产代码/标准未改。原8卡任务Running，0/0/39=36历史成功+1新D01成功+2D01失败，8worker维护；本地监听PID2213164继续watching，无新平台申请。

T20 v3私有两环境各48项通过且根端AST复审，source-map私有各23项通过，未publish/实际stage/apply/pointer/入队。[T20私有检查](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_pool_launcher_cpu_v3_20261001/validation.json)、[桥接复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/t20_v3_root_review.json)、[源映射复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/source_map_root_review.json)。GPU验收仅覆盖R20工程入口，不认证R21/R24/R26同版本训练，不证明架构收益。2026-10-01 05:52 UTC数据metadata仍training_ready=false、双证书缺失；审核后manifest改变须另做明确版本绑定，不能仅补文件绕过False。真实四候选1000→累计5000与最终版本训练未启动，goal仍active；独立CUDA算子保持暂停。

## 2026-10-01 05:31 UTC full梯度与清理实测通过，正式窗口运行中

v3新D01 1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959 在原GPU3已完成独立full gradient_oracle：706.9962秒<1200，1窗/8 microsteps/256人工记录，78参数比较通过，梯度倍率1.00000077、相对L2误差0.00015194，CE/RNG核对通过。11个对象弱引用全部失效；gc/empty-cache后allocated17,039,360字节，清cuBLAS workspace后allocated/reserved均0；新运行tiny与full阶段均实测支持工作区驻留候选，历史v2仍缺残留数值，不追认其根因。[完整oracle局部实测](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/full_oracle_observation.json)。

已进入train_optimizer_step/1800秒预算，正式三窗口和四mask评估仍待完成，整体gpu_accepted/raw_completed=false。524父字节、原架构、数值标准和工作量不变；独立CUDA算子暂停。监听PID2213164/原八卡任务保持；T20 v3私有rebase中、源映射未部署，数据双证书仍缺/候选training_ready=false，真实比较与最终训练未启动。验证checkpoint=0。

## 2026-10-01 05:20 UTC 当前实际v3状态

v3于05:04:13 UTC实际发布8个支持/producer文件，05:09:59构建526文件不可变包，manifest SHA cc13036fc492109afbf4c9cf9fd11a4c87684fc9702b83ce86c961b2a08294fe；524父文件保持。producer必要CPU两环境各30项，support各108项通过，实际隔离导入各8 profiles/54项目模块通过。05:12:16本地监听实际重载PID2213164并核验watching，GPU任务/worker未重启。05:14:40只在原八卡池入队D01 1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959，GPU3实际领取。固定tiny preflight耗时1.8074秒、8 microsteps/8人工记录；11个对象弱引用均释放，gc/empty-cache后allocated17,039,360字节，清原生cuBLAS workspace后allocated/reserved均0。当前完整gradient_oracle进行中，formal三窗口及四mask评估尚未通过；仍gpu_accepted=false，不能把局部tiny通过记作完整验收。[发布原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/publication.json)、[固定运行包](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r20_r27_pilot_v3_20261001/pilot_runtime_manifest.json)、[实际启动记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/d01_v3_launch.json)、[tiny阶段实测](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/lifetime_preflight_observation.json)。

T20 source-map私有两环境各23项通过，原28项映射保持；[软件源绑定记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_source_map_cpu_20261001/validation.json)。T20新私有v3正在显式rebase固定root/manifest，未发布/apply/pointer/入队；原数据准入全部保留。候选training_ready仍false、双证书未齐；真实四候选1000→累计5000与最终训练未启动，goal active。独立CUDA算子仍暂停。自有CPU临时目录已清理，人工GPU关闭checkpoint保存，小型指标/失败原件/源码哈希保留。

以下带时间记录属于当时状态。



## 主干与参数

继续从零训练，不使用 ESMC。保留项目原 LLaDA270 的 Transformer 主干：d_model=768、8 层、12 个注意力头、MLP=3072、RMSNorm、SwiGLU、RoPE theta=500000。没有把主干换成另一类模型，也没有扩大有效 Transformer 参数规模。

旧 no-ESMC 模型总参数为 269,759,232，其中大词表的输入 embedding 和输出 head 合计 194,248,704。当前采用 50-token 小词表，R20 总参数为 75,601,176；两者 8 个 Transformer blocks 都为 75,509,760 参数。这里的 75.6M 与原 270M 的差别主要来自词表，不能解释成主干缩小。

[基线与旧 checkpoint 结构核查](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_LLADA270_BASELINE_AUDIT.md)；[冻结的配置与基础模型](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_releases/r20_r27_20261001/dllm/pipelines/immune_llada/v3/model.py)。

## 输入、位置与链编码

输入 hidden state 为 token embedding + role embedding + scope embedding。role 有 12 类，scope 有 6 类；scope 表示 full、junction、core、fragment、missing、unknown 等观测范围。

位置使用显式链内 position_ids 注入 RoPE。chain_ids 用于区分链组并构造注意力关系：每个头各有可学习的同链/异链 bias，共 2×12 个参数。当前没有额外把链的绝对编号做成 chain-index embedding。R25/R26 还使用这些链组与位置决定各层可见范围。

A4 监督真实观测的残基字符串，加一个链终止符 <protd>；容量尾部不重复监督 EOS。原始观测中的 C/F/W 边界残基保留，不把它们当作可删除的结构占位符。当前仍保留 full=256、fragment=64 的目标容量和 max_length=2048，所以尚未实现完全取消固定 target canvas，也尚未取消学习终止符。

[冻结的布局实现](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_releases/r20_r27_20261001/dllm/pipelines/immune_llada/v3/layout.py)。

## 已完成 500 步的八个独立版本

以下每项均保留相同主干、词表和基础编码，分别测试机制，不是把所有模块叠加成一个最终模型。

| 版本 | 实际改动 | 500 步后的安排 |
| --- | --- | --- |
| R20 | 基础 A4 对照 | 保留为后续对照 |
| R21 | normalized 输入经 Linear(768→12)+sigmoid，为每个 token 的各注意力头输出动态赋权，再进入输出投影 | 保留候选 |
| R22 | 注意力头输出固定乘 0.5 | 与 R21 区分动态门控和单纯缩放；暂不优先 |
| R23 | attention/FFN 残差逐通道可学习 LayerScale，初值 0.1 | 与 R24 比较；暂不优先 |
| R24 | attention/FFN 残差固定乘 0.1 | 保留候选 |
| R25 | 前 4 层各链隔离、后 4 层全局交互；前缀不能在隔离层替两条链中转信息 | 此轮较慢，暂不优先 |
| R26 | 偶数索引层同链局部半径 32，奇数索引层全局交互 | 保留候选，并报告显存代价 |
| R27 | 后续层 V 与首层 V 各占 0.5；保留各层 V 投影 | 暂不优先 |

R20–R27 共同增加按注意力拓扑复用 bias：同一次 forward 内复用相同 mask/bias，避免每层重复构造；不跨 batch 缓存。R25/R26 目前用 dense mask 改变信息流，没有实现稀疏 attention 加速。

[冻结的机制实现](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_releases/r20_r27_20261001/dllm/pipelines/immune_llada/research_next/model.py)；[冻结的 bias 复用](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_releases/r20_r27_20261001/dllm/pipelines/immune_llada/research_refined/model.py)。

## 当前候选依据

固定窗口为 400、425、450、475、500 五个检查点。AA 是人工 heldout 的条件 full-mask 残基准确率；同一条曲线的五个点不是五个独立重复。

| 版本 | 最近 100 步 mean loss 均值 | 条件 full-AA 均值 | 2048 输入 profile 峰值 |
| --- | ---: | ---: | ---: |
| R20 对照 | 0.193457 | 97.01% | 41.995 GiB |
| R21 动态门控 | 0.186520 | 98.53% | 42.184 GiB |
| R24 固定残差缩放 | 0.187263 | 99.00% | 41.995 GiB |
| R26 局部/全局交替 | 0.186906 | 98.77% | 53.968 GiB |

R21/R24 的额外显存很少，R26 长输入峰值比对照增加 28.51%。三者 loss 接近，尚不能选定唯一胜出者或最终组合。

全部 8 项各完成 500 步，无 NaN 或执行失败。训练 576 条、heldout 72 条均为人工规则，seed=42；partial-mask 已接近饱和，无条件 full-mask 有输入冲突。这些结果支持后续比较方向，不证明真实语料、下游效果或生成质量提升。[完整结果、成本、曲线与逐任务证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_R20_R27_500STEP_RESULTS.md)。

## 更广方法与当前接入状态

早期原型还覆盖 span/whole-chain mask、链平衡 loss、QK norm、register tokens、局部卷积、噪声调制、角色 pair bias、VICReg 式辅助目标及来源采样。多数只有 20 步工程检查，不能据此说已有效。Perceiver/MBT、双向 SSM、Gated DeltaNet、HC/mHC、差分 attention、自条件、teacher 表征方法等仍有仅调研或设计的部分；[研究索引](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_RESEARCH_INDEX.md)区分阅读、实现和实际验证。

当前接入修正后的原生HF Trainer与更完整评估。修正梯度累积缩放属于实现正确性。v1入口两环境各29项CPU、运输/隔离34项、展示14项及producer24项通过后已实际部署；D01在GPU5执行，但oracle701.6219秒超过预设240秒，未进入正式三窗口，未通过GPU验收。正在修订预算并创建新不可变版本，全部数值标准保持。[实际整合与失败记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_PRETRAIN_INTEGRATION_20261001.md)。

来源未确认的并发写入已保存证据，producer私有实现经过独立复审和原址SHA核对后发布。实际runtime、路由和D01启动已落盘；只重载本地监听到PID2143672，原八卡GPU worker与任务未重启。固定父524/runtime526文件终态哈希均通过，模型和500步证据不受本次时限失败影响。[监督与来源记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TRAINER_GPU_SUPERVISION.md)。

按用户确认，独立 CUDA 算子测试移出主线，继续现有 attention。真实语料比较的计划是 R20/R21/R24/R26 各 1000 步，再按训练内验证与成本选择、续到累计 5000 步及第二 seed；数据覆盖审核和原双证书仍未齐，真实阶段尚未开始。[规划协议](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_architecture_next_release/protocol.yml)。
