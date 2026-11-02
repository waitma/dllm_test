# A5全局注意力的归一化与输出门控探索

2026-11-02，用户要求在本地继续更多架构探索。上一轮9个最终检查点在本地重评估、5000次配对统计都完全复现，仍未达到稳定提升标准；A5的整链生成改善主要表现为输出长度覆盖率。新的候选优先检验CDR、表征及生成质量能否一起改善，保持原数据和训练目标。

## 预先固定的实验矩阵

| ID | 模型 | Q/K每头RMS归一化 | 单位初始输出门控 | 参数量 |
|---|---|---|---|---:|
| C0 | A5，全局注意力对照 | 无 | 无 | 75,741,903 |
| C1 | A5 + 归一化 | 有 | 无 | 75,742,927 |
| C2 | A5 + 门控 | 无 | 有 | 75,815,727 |
| C3 | A5 + 两者 | 有 | 有 | 75,816,751 |
| A0 | 原始架构对照 | 无 | 无 | 75,741,903 |

C0-C3构成2×2因子消融；额外从头运行A0，避免拿旧micro56或三epoch长日程中的中间checkpoint充当同配方对照。原始A0保持交替可见性，C0-C3每层全局。各组相同8层、宽768、12头、FFN3072、链内RoPE、角色/scope嵌入、同链偏置、固定残差0.1和原任务头。新增参数最多0.1%，不缩小FFN。所有原公共参数保留同种子的逐位一致初始值。

Q/K归一化复用已有冻结HeadNorm：在每头维度以FP32做RMS归一化，epsilon1e-6、学习头维度权重，再使用原RoPE。它借鉴Q/K归一化思路，并非原论文的L2归一化加单一可学习温度的完整复现。[Query-Key Normalization](https://arxiv.org/abs/2010.04245)

输出门控在SDPA之后、输出投影之前，按query位置和head调节，公式为`2*sigmoid(W_g*x_norm+b_g)`；新增W/b零初始化，使门控初始严格等于1。保留原attention残差0.1，避免普通sigmoid在初始化时额外把残差减半。该适配借鉴head-specific输出门控研究；2倍尺度和单位初始化是本实验选择，不是原论文方案的逐项复现，更不预设蛋白任务一定提升。[Gated Attention](https://arxiv.org/abs/2505.06708)

## 数据与训练

原V8 prepared JSONL、字段、链顺序/角色、metadata、relation/family标签、tokenizer/词表/特殊token、固定链输入和原reader/collator全部保持。复用字节相同的653,288条固定10%索引，SHA `4beaac6b54a1f46492533aeb77c2de2c6ac2a0bd8266fd57a6bb312679435c51`，不重抽数据。

本轮仅seed43筛选，各从随机权重开始1epoch/1276次更新，micro32、global512（16次累积，epoch末488条保留），每组消耗214,482,921个非pad token。原学习率1e-4、AdamW、5% warmup、监督/噪声、16轮生成、token验证和冻结probe均保持。已有screen或confirm权重不作为初始化。

先在当前本机单卡上对五组逐个做maxlen2048/micro32的3次真实更新资格检查；均通过原85%显存门槛才进入正式训练。每组profile结束后退出进程，新进程从头训练，不沿用profile权重。数字CUDA selector0、PCI_BUS_ID；无DDP/NCCL。后台顺序C0→C1→C2→C3→A0；不自动扩展seed或启动远程任务。

## 评估与记录

只使用screen分区评估；最终固定step1276作比较，不从中途最好checkpoint挑终点。逐步loss、每100步token验证、每250步及epoch末任务曲线/完整checkpoint沿用原逻辑。完整六文件checkpoint和参数/优化器/RNG/数据游标绑定保持，可恢复。每组完成后自动生成PNG/PDF/SVG、原始CSV、终点表，并commit/push探索分支；网络失败仅publication pending，不覆盖训练退出码。

五组齐全后报告C1−C0、C2−C0、C3−C2、C3−C1及交互C3−C2−C1+C0；同时逐项比较原始A0。重点看CDR AAR、AUROC/AUPRC，并一并核对原CE综合、整链AAR、有效率和长度误差；不凭单一loss或单种子宣告稳定提升。

原三种子稳定性门槛保持。这是探索性筛选，已查看过的旧confirm不能被当成新候选的全新未使用留出；下一轮独立确认需预先固定新的未用于设计的评估证据，不以这轮单种子或重复查看旧confirm替代。

## 路径与隔离

入口：

```bash
bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_attention.sh --gpu 0
```

新模型与适配仅在 `/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/architecture_attention`。新输出 `/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_attention_20261102`，报告 `/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/attention_20261102/REPORT.md`。每个worker从内容寻址快照导入，后续文档更新不影响训练源码绑定。

历史冻结模型/训练/评估源码保持SHA `3c9c424d816b9fda7adfba5047d9e075ac15dc036342b59461edd35031e50971`；本轮在其上单独加入新包和入口。原生产训练目录、1B模型和所有旧输出保持只读。新内容只提交 `explore/arch70m-v8-20261028`。

启动前CPU回归、shell/dry-run、快照导入及GPU资格的实际状态见同目录验证回执与PROJECT_PROCESS.md，不把尚未运行的检查算作通过。

## 实际启动核验

2026-11-02T16:05:54Z UTC：controller PID38976；首组C0 worker PID39291；实际冻结cwd和命令核验，首组step21，loss/梯度有限。CPU9项、shell/dry-run与快照核验均通过。

| 资格组 | allocated GiB | reserved GiB | 更新数 | safe |
|---|---:|---:|---:|---|
| C0 | 24.40 | 31.82 | 3 | True |
| C1 | 25.53 | 33.14 | 3 | True |
| C2 | 24.59 | 32.01 | 3 | True |
| C3 | 25.72 | 33.14 | 3 | True |
| A0 | 36.37 | 44.01 | 3 | True |

新冻结源码SHA `28609d7f776013915561eb038396d54181dcee7b1c32d071be0a9da5dc12faad`；原历史SHA保持。后续由后台队列执行，不持续人工监督。详见 [启动回执](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/attention_20261102/startup_verified.json)。
