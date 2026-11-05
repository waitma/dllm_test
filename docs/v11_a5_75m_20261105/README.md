> Last updated: 2026-11-05T08:49:23+00:00。用户提交的V11已确认正式八卡训练：step9000/300000（3.00%），epoch0.705，seen4608000；最新完整checkpoint8000，8rank参数哈希一致。最新MLM验证step9000 loss=0.580162；最近80个记录点均值0.838秒/更新。8份NCCL日志SHM_DISABLE=0且实际SHM/direct，未修改或重启训练。回执 /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/v11_a5_75m_20261105/training_first_review_20261105.json。

> 2026-11-05T05:03:51+00:00 通信性能修复已应用到同一个启动脚本：`NCCL_SHM_DISABLE=0`，保留P2P/IB禁用与NCCL2.31.2；清除`CUDA_LAUNCH_BLOCKING`，只记录NCCL初始化/环境/拓扑通道日志。1B已完成八卡受控对照：8.0676→6.7087秒/步；该数值不作为V11实测速度。V11模型、数据、batch、优化器、确定性和保存逻辑保持。

# V11：A5 全局注意力 75M 可训练版本

入口：`/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_v11_a5_75m.sh`。用户已提交八卡正式训练；当前运行核实见顶部，实时进度以train/status.json为准。

```bash
bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_v11_a5_75m.sh
```

这条命令在一台八卡机器上用 DDP 共同训练一个 V11 模型。请在能访问现有共享数据的八卡作业内运行。

## 模型与训练配置

| 项目 | V11 |
|---|---|
| 参数量 | **75,741,903**，含183类 V-family head |
| 模型 | 8层，hidden768，12头，head_dim64，SwiGLU3072 |
| 注意力 | 全部8层为全局双向自注意力；保留链内RoPE、角色/scope embedding、同链/异链偏置 |
| 残差 | attention / FFN 都固定缩放0.1，与A5一致 |
| 初始化 | 从随机权重开始；不载入探索或1B训练权重 |
| 数据 | 与1B完全相同：train **6,533,037**，valid71,422，121分片；不使用探索10%子集 |
| 步数 | **300,000**，完整epoch无替换采样，最后按精确步数结束 |
| batch | **micro16 × 累积4 × 8卡 = global512** |
| 学习率 | **1e-4**，warmup **15,000**步，之后cosine |
| AdamW | betas=(0.9,0.999)，eps=1e-8，weight_decay=0.01，gradient_clip=1.0 |
| 精度/长度 | BF16，max_length2048，整层梯度重计算 |
| DataLoader | 每rank4 workers，spawn；与1B相同 |
| 日志 | 每25步，前3步/epoch结束/最后一步另记 |
| 常规验证/保存 | 每1000步；epoch结束/最后一步也验证保存；初始step0验证保存，step50额外保存 |
| 保存保留 | 最近2个checkpoint，加历史最佳probe AUROC对应checkpoint，沿用1B策略 |

除模型尺寸与A5的全局可见性外，训练配置JSON与1B正式计划逐项相同。复用1B冻结的训练器与全部依赖源码（3,193文件），通过V11独立适配器选择模型；原训练器文件不改。

## 数据和训练目标保持

原V8 prepared JSONL、链序列/顺序/角色、relation/family标签、元数据、词表、固定槽位与reader/collator保持。
MLM仍是15%选择、80%MASK/10%随机其他token/10%原样；显式关系始终MASK。binding/unlabelled按原规则做MLM，nonbinding序列保持clean，只做关系token和原V-family辅助监督；family权重0.1、183类（含unknown）相同。

原1B materialized binding同时绑定了模型尺寸，因此仅在V11新目录复制索引/绑定元数据并更换model字段，原始1B文件只读；新旧`shards`、`files`、所有样本偏移和划分逐项一致，记录没有迁移、重写或重采样。固定受体槽位仍167/135、合计304；数据manifest SHA保持`4f121884eeeb8aad83db8f245955a7924f1219ebb3308483e7c7f9d63c9152b0`。

## 目录与执行

- 配置：`/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/configs/models/v11_a5_75m_20261105/training_config.json`
- 模型与适配代码：`/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/v11_a5/model.py`
- 独立运行根目录：`/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/v11_a5_75m_300k_20261105`
- 冻结源码：`/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/v11_a5_75m_300k_20261105/bundle/source`
- 模型绑定索引：`/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/v11_a5_75m_300k_20261105/bundle/data/index.json`
- 正式结果将写入：`/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/v11_a5_75m_300k_20261105/train`
- W&B：`pressure_jian/dllm-immune`，沿用已验证的waitma账户凭据，V11使用独立随机run ID；新run不与旧smoke冲突，恢复绑定原checkpoint的run ID。
- CUDA/NCCL：独立复制NCCL2.31.2并校验文件SHA与加载路径；默认SHM启用（NCCL_SHM_DISABLE=0）、P2P/IB禁用，修复原1B入口中的Socket瓶颈。

```bash
# 查看实际命令，不训练
bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_v11_a5_75m.sh --dry-run

# 离线核对源码、数据、环境与75M参数量；不连接W&B、不使用GPU
bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_v11_a5_75m.sh --verify-only

# 使用另一个独立输出目录
bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_v11_a5_75m.sh --run-root /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/v11_a5_75m_attempt2

# 明确恢复该run目录自己的V11完整checkpoint
bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_v11_a5_75m.sh --resume /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/v11_a5_75m_300k_20261105/train/checkpoint-1000
```

默认新训拒绝已存在的train目录；resume须显式指定该run内的V11检查点。源码/配置/模型尺寸/优化器/数据/每rank RNG均由原回执与provenance校验，不能把V8/A0/1B检查点误作V11续训。

加载模型使用V11专属loader，确保读取后仍是全局注意力：

```python
from v11_a5.model import V11A5Model
model = V11A5Model.from_pretrained("/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/v11_a5_75m_300k_20261105/train/checkpoint-1000")
```

## 已完成验证与范围

1. 7项CPU回归通过：与探索A5同初始化权重/logit/梯度精确一致；全部8层跨链可见性及padding屏蔽；梯度重计算一致；V11保存/加载；旧A0 loader拒绝V11；1B训练配置/数据一致；冻结文件损坏检测。
2. 冻结源、运行环境、完整121分片与6,604,459条索引记录检查通过；训练记录6,533,037。
3. 本地单卡使用16条真实记录、micro16，包含24个family目标，做2次更新与1次恢复重放；目标、梯度范数、恢复后参数精确一致。实测峰值allocated **3.42GiB**、reserved **3.80GiB**，仅对应这些实际样本。
4. 单卡验证的global batch为16，仅验证训练器路径；正式配置仍为八卡global512。未将其当作八rank资格或maxlen2048最坏情况显存测试。交付时正式train目录尚不存在；后续开训状态见顶部。

内置probe和best保留逻辑为保持1B配方而沿用；这些探针仍受此前发现的缺链捷径/隔离范围限制，不将其作为新架构泛化提升的证据。V11是用户选择的A5版本化训练交付，不是已证明整体胜出的模型。

[离线回执](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/v11_a5_75m_20261105/offline_verification.json) · [单卡训练/恢复回执](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/v11_a5_75m_20261105/local_gpu_validation.json) · [CPU测试](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/v11_a5_75m_20261105/cpu_tests.log) · [指标审计](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/metric_audit_20261104/REPORT.md)

## 通信速度修复的依据与验收（2026-11-05）

之前1B诊断已独立完成8卡、同输入/初始化/掩码、每组5步预热＋20步计时的比较。原入口同时禁用SHM和P2P，实际走NET/Socket/0；打开SHM后走SHM/direct，四worker真实DataLoader的完整更新均值8.06757694→6.70866053秒，耗时下降16.84%、吞吐提高20.26%，逐步loss/family_loss/grad_norm等最大差为0。独立重复与缓存批次得到相同方向；这些证据支持只恢复SHM，不更改模型或训练配方。

V11外层启动器现在明确设置SHM=0、P2P_DISABLE=1、IB_DISABLE=1、CUMEM_HOST_ENABLE=0；NCCL_NET=Socket仍保留作为网络后备，不会阻止本机SHM通道。清除CUDA_LAUNCH_BLOCKING调试环境，保持efficient attention与原确定性设置。
NCCL_DEBUG_FILE写入运行目录的`nccl.%h.%p.log`，仅INIT/ENV/GRAPH；实际是否使用共享内存以日志中的`SHM/direct`为准。没有启用每次collective日志。

实际执行完整Shell入口的隔离CPU回归通过：即使继承SHM_DISABLE=1、P2P_DISABLE=0、CUDA_LAUNCH_BLOCKING=1，最终torchrun子进程也收到SHM=0/P2P_DISABLE=1及无blocking标记；8rank参数和冻结V11入口路径保持。此回归使用假解释器，仅测试进程传参，不把它称作CUDA/NCCL多卡验收。
本机只有单卡，未重跑V11八卡速度比较；1B收益不能直接外推为75M的提速比例。通信修复时V11正式train目录尚不存在；后续开训状态见顶部。未修改/重启原1B训练。外层入口修复直接作用于现有V11 bundle，不改冻结plan/模型源码，plan SHA仍为`cbf8b79eadd80982f505755fc493d97a71f41b533cd8b7cbdf9b2da2a6022118`；此次外层脚本SHA单独记录在回执中。

[1B受控诊断报告](/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/output/bert_v8_1b_comm_diagnosis_20261029/report.md) · [独立复核](/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/output/bert_v8_1b_comm_diagnosis_20261029/analysis_review.json) · [V11修复回执](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/v11_a5_75m_20261105/transport_fix.json) · [进程环境回归](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/v11_a5_75m_20261105/transport_regression.log)
