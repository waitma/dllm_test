# V7训练速度根因：本地A100实测及修复

## 2026-10-06T13:05:08.199910+00:00 新增1000更新真实八卡墙钟窗口

8001→9001实际722.879864秒，0.722880秒/更新，包含一次常规验证/保存/源码检查/轮换；8002–9000的999个普通更新均值0.713420秒，中位数0.763121秒。9000非训练边界约9.44秒，不含段启动恢复；不可将该窗口当完整epoch或FSDP性能。未运行新的模型测试，仅读取现有逐步计时。原300步0.740862秒窗口保持；新原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/speed_after_checkpoint9000_root_review.json。正式9000保存与验证已核实并继续训练，质量指标有升降，详见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_TRAINING_RESULTS_20261005.md。

## 2026-10-06T12:42:38.746417+00:00 新代码已完成八卡接续，300步实测平均0.740862秒/步

原容器同DDP正式训练已切换到独立where-only冻结源。7000→7008与7008→7016均通过实际8rank原生optimizer/scheduler/RNG恢复、参数/完整保存和8slot退出检查；随后7016→12016于2026-10-06T12:35:30.046083+00:00单次启动，当前实际计时到7450/25042。新监督PID212174/start2455180199已通过/proc核实，8worker心跳新鲜，abort=null。平台仍为t-20261005205851-lqfjg，没有重复submit/cancel或重建容器。原监督PID3977587已按用户授权停止，不重启。

| 计时窗口 | 更新数 | 平均秒/步 |
|---|---:|---:|
| 旧版5475→5875 | 400 | 10.199121 |
| 旧版6125→6850 | 725 | 10.695606 |
| where新版7027–7326 | 300 | 0.740862 |

新版中位数0.785829秒、P95 1.057530秒，global256对应约345.54条/秒；相对上述旧窗口观察到13.77–14.44倍提速。跳过本段最初10步；计时含数据与DDP的训练循环墙钟，不含验证/保存/初始化。旧窗口使用日志首次观察端点，有轮询误差且batch不同；不能将此作为同batch严格配对或端到端epoch耗时。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/speed_300_updates_root_review.json，含300条原始计时和来源哈希；未新增模型或CUDA专项测试。

保存已实际采用主eval_loss最小3份＋最新完整恢复点取并集，当前保留7000、7008、7016（最新与best重合）。原5016、6000权重已按新规则删除，删除前的JSON/指标证据保留在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/run/checkpoint_retention_evidence；每份保留完整优化器/调度器/8rank RNG。当前7016的eval_loss=1.889094435879232；旧7000=1.8842671278774004，短接续验收不证明效果提高。原旧段最后打印7600，只有7000完整保存，约600个已打印未保存更新已从7000重算，不宣称无进度回退。

接续证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/segment7008_root_review.json 与 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/segment7016_acceptance.json；新监督 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/supervisor_v1/supervise_v7_where_20261006.py，实时状态 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/supervision_state.json；当前输出 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/run。后续每段5000更新，经实际恢复/保存/退出验收后接到总25042；遇错误停止继续准入，不重复提交。当前仍为f011、micro32/GA1/world8/global256、BF16 autocast、math attention；仅替换等价chain_bias展开表达式，未启用高效SDPA或切FSDP。原数据门禁、序列注释停线、独立CUDA暂停保持；原Goal仍active，完整正式训练和22格消融未完成。

更新：2026-10-06T11:35:49+00:00。已完成完整模型CUDA profile、受控替换、真实V7批次回放及梯度检查；原八卡冻结任务未切换版本。

**主要瓶颈是可训练chain bias的重复索引反向。** 此前优先怀疑math attention回退，实测单换后端只改善约6%，应先修复bias展开。另外确认本地同卡occupy.py抢占算力。

## 根因定位

冻结模型 [原表达式](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3/dllm/pipelines/immune_llada/research_refined/model.py:55)：

```python
base_bias = self.chain_bias[same.long()].permute(0, 3, 1, 2).contiguous()
```

chain_bias只有2×12个参数，索引却扩展到B×S×S个同/异链关系；反向将巨量重复索引的梯度归约回两行参数。实际profile捕获`IndexBackward0 → aten::_index_put_impl_ → indexing_backward_kernel_small_stride<float>`。

真实V7 B32×S867完整更新耗时9.5329秒，上述单个CUDA kernel占8.8094秒，即约92.41%。改写bias表达式后该瓶颈kernel消失，相同形状的profile更新为0.6975秒。profile用于归因，吞吐采用下述独立计时窗口。

V5代码对照没有当前可训练decoder chain_bias展开路径。V7的8层/d768/H12主干未缩小，参数减少没有消除新增反向开销。历史速度与配置见 [V5代码对照](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_V5_CODE_COMPARISON_20261006.md)。

## 完整模型对照

单卡A100-SXM4-80GB、PyTorch2.8、f011、75,601,176参数、B32/GA1、FP32参数/BF16 autocast、whole-layer checkpoint、AdamW。各组保持相同初始权重、seed、输入顺序、目标和mask，只改变指定变量。表格使用只在整步边界同步的独立计时窗口。

### 固定人工输入B32×S516

| 条件 | 平均秒/更新 |
|---|---:|
| 原版，occupy.py运行 | 3.8562 |
| 原版，占卡暂停 | 2.1404 |
| 只切efficient SDPA | 2.0150 |
| 只改where展开，仍用math SDPA | 0.3277 |
| where与efficient同时使用 | 0.1972 |

每组2次预热、3次阶段计时、3次整步计时和1次profile，共9次更新。只改where提速6.531倍；单换efficient提速1.062倍。实际事件确认原版走math SDPA、efficient组走efficient attention及其反向，不只是检查开关。

### 原生V7批次回放

使用原1024条来源前缀样本、冻结A4 collator，从中固定选8个batch共256条记录，S为516/797/629/841/701/752/785/867。各组捕获的全部32个batch内容SHA一致，选中的8个batch及顺序一致。输入只保存在RAM，随机corruption保持冻结实现及seed42。样本用于多种真实形状的工程验证，不代表全量长度分布。

| 条件 | 平均秒/更新 | 相对原版加速 |
|---|---:|---:|
| 原版math SDPA | 6.6288 | 1.00倍 |
| 只改where展开，仍用math SDPA | 0.5450 | 12.16倍 |
| where与efficient同时使用 | 0.3103 | 21.36倍 |

每组2次预热、8次阶段计时、8次整步计时和1次profile，共19次更新。表中使用8次整步计时均值；阶段计时的平均反向由6.4519秒降至0.3844秒。吞吐窗口不含CPU取数/collate、H2D、DDP、Trainer日志、验证和保存，H2D另有逐步记录。单卡结果不能直接当作八卡最终提速。

## 修复与数值验证

工作区 [RefinedModel](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/research_refined/model.py:55)已改为：

```python
base_bias = torch.where(
    same[:, None], self.chain_bias[1][None, :, None, None],
    self.chain_bias[0][None, :, None, None])
```

保留24个可训练参数、mask、target容量、END监督、局部/全局安排及梯度。AST核对确认相对冻结RefinedModel只有这个表达式发生变化，与GPU对照表达式一致；此次没有默认启用efficient，也未改batch、精度或并行方式。

- CPU FP64展开：前向完全一致，梯度最大绝对差1.92e-13。
- 小完整模型FP32/BF16及whole-layer重计算：loss完全一致，24组参数梯度通过，最大绝对差5.72e-9。
- 75.6M完整模型、原生B32×S516、CUDA BF16：首步loss均为3.8769800663，78组梯度通过；最大绝对差2.7316e-6，在chain_bias，相对L2差1.7429e-4。容差atol5e-6/rtol1e-3。归约顺序改变，不声称完整训练轨迹逐bit相同。
- 工作区既有回归64 passed：覆盖8种refined变体、FP32/BF16、padding、全部参数梯度、两个同时存活的重计算图和bias缓存生命周期。16项保存/加载测试未运行；表达式不改state_dict，本次没有创建测试权重。

## 本地占卡与恢复状态

通过NVIDIA设备fd确认PID1909/start2358659842的 [occupy.py](/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/occupy.py)与本地f010 PID80642/start2452750909使用同一物理GPU。它循环执行10000×10000矩阵乘、Linear和ReLU。只暂停f010时GPU仍100%；同时暂停二者后降至0%。同一模型开/关占卡的耗时为3.8562/2.1404秒，即约1.802倍。此证据仅限本地卡，不能据此判断远端八卡占用。

窗口使用有界控制器和独立恢复watchdog。测试后两个原PID/start匹配，均恢复并实际观察到运行态，诊断子进程已退出。没有终止/重启训练、改动冻结源或正式checkpoint，没有保存测试模型、梯度及临时token文件。

**正在运行的正式/消融任务仍使用原冻结实现。** 工作区修复已验证，后续使用新冻结版本时再验收八卡吞吐，不将未部署修复写成原任务已经加速。

## 证据

- [汇总与审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/local_verified_profile/verified_summary.json)：原件SHA、输入一致性、梯度、代码表达式和恢复状态。
- [固定输入控制记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/local_verified_profile/b32_s516_attempt2/control.json)、[真实输入控制记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/local_verified_profile/native_v7_attempt1/control.json)。
- [原版真实批次profile](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/local_verified_profile/native_v7_attempt1/baseline/measurement.json)、[where真实批次profile](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/local_verified_profile/native_v7_attempt1/where/measurement.json)、[完整GPU梯度检查](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/local_verified_profile/native_v7_attempt1/where/gpu_gradient_equivalence.json)、[回归日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/local_verified_profile/worktree_regression.log)。
