> 当前状态复核：2026-10-29T13:24:16Z UTC。B1已完成1276步；B2/B3在正式step0因显存安全门槛停止，队列已结束。下文启动时进度是历史记录。结果与完整曲线见[本次复核](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/review_20261029/REPORT.md)。

# 本地第二轮架构探索：B1 / B2 / B3

2026-10-29，按用户“本地的架构可以开新的探索”启动独立后续实验。当前本机GPU0已启动B1正式训练，启动核验时step11/1276；B2、B3随后串行。具体实时进度以输出目录status.json为准，不持续人工轮询。

| 变体 | 相对现有模型的改动 | 参数量 | 主要比较 |
|---|---|---:|---|
| B1 | A5 的全局注意力，每层 attention / FFN 残差尺度由固定 0.1 改为可学习的通道向量 | 75,754,191 | A5：残差缩放；A4：在可学习缩放下改全局注意力 |
| B2 | 每层链内注意力 → 全局注意力 → SwiGLU FFN；FFN 宽度 3072 → 2048，残差尺度固定 0.1 | 75,748,047 | A5：近似相同参数预算下重新分配注意力 / FFN 容量 |
| B3 | B2 的三个子层均使用初始值 0.1 的可学习通道残差尺度 | 75,766,479 | B2：缩放作用；B1：有可学习缩放时的架构组合 |

B2/B3 的链内→全局结构参考 [PoET-2 encoder](https://arxiv.org/html/2508.04724v2#S3.SS2.SSS4)。只移植蛋白序列注意力思路，保留当前链内位置编码；不引入结构输入、prompt、检索或新解码器。额外 attention 由缩小 FFN 支付，不能将结果差异单独因果归于注意力顺序，也不声称复现 PoET-2 完整训练。

## 固定条件

- 每组仅 seed43，从随机权重训练 1epoch / 1276 步；本地 GPU0 顺序 B1 → B2 → B3。
- 复用八卡侧已经完成的 A0/A4/A5 seed43、同 1epoch / micro56 对照；不重跑对照，不追加种子。旧 A0 三 epoch 确认不能混入本轮终点比较。
- 原 V8 prepared JSONL、链序列/顺序/角色、元数据、relation/family 标签、词表、特殊 token、固定链输入布局、reader/collator 均保持。复用字节一致的 10% index，训练记录 653,288 条，训练及评估划分均保持。
- global batch512；micro56，累积 9×56＋8；最大长度2048；8层、d_model768、12个头；MLM / family 目标、优化器、LR1e-4、确定性设置与原冻结配方保持。
- 每个变体先在最大长度做3次真实更新，验证 reserved＋外部显存不超过85%，随后 exec 全新进程正式初始化；profile 权重不进入训练。
- 训练使用原整层梯度重计算；逐步本地 JSONL 记录 loss / 梯度 / GPU用时 / 显存。每100步评估token，250步和epoch末执行完整screen任务评估并保存checkpoint；step0同样评估及保存。
- 保存继承原六文件原子 checkpoint / SHA 回执，含优化器、scheduler、RNG及数据游标；已有 latest 可恢复，同绑定完成项跳过。最佳生成/表征指针与最新checkpoint分别保留。

## 代码、绑定与运行

原 A0–A7 冻结代码保持 SHA `053297983e0f2ece2aaad807032695fb5aebd1e0f221138a9bf0d202cc9b9ac4`。新模型和启动适配独立放在 `/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/architecture_round2`，只在新进程中替换 build / source_binding，不改旧训练器文件。第二轮源码及完整原依赖复制到内容寻址快照，worker 从该快照导入。

第二轮源码 SHA：`93518fa2a14bc029e808cefae7aef8c38d79804683bb5274af27ff5f4e67d65e`。
数据 index SHA：`4beaac6b54a1f46492533aeb77c2de2c6ac2a0bd8266fd57a6bb312679435c51`。

入口（若队列正在运行不要重复启动，controller.lock会拒绝重复控制器）：

```bash
bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_round2.sh --gpu 0
```

新输出目录：`/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_round2_20261029`。控制器 `execution.json` 和各试验 `process.json/status.json/console.log/profile.json/binding.json` 记录实际执行证据；与原八卡输出分离。训练失败原文与 publication 状态分别保存。

## 验证与结果记录

8项 CPU 检查通过：B1 初始权重/logit与A5一致、各变体有限梯度、链内/全局可见性、嵌套注意力mask下整层梯度重计算与普通反向一致、蛋白块换序等变性、safetensors往返、优化器与RNG checkpoint恢复、全尺寸参数量。微型CPU测试不代表真实GPU训练已通过；另已验证快照下导入的是新适配及相同源码绑定。GPU显存与正式训练启动回执会追加。

[验证回执](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/round2/validation.json)
[实时结果与完整曲线](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/round2/REPORT.md)
[原始曲线CSV](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/round2/all_curves.csv)

每组结束自动刷新报告、完整曲线PNG/PDF/SVG及CSV，commit/push到 `explore/arch70m-v8-20261028`。推送失败保留本地提交、记录pending，不改变训练退出码；共享发布锁避免与八卡报告相互覆盖。不改生产工作树 `/mnt/project-storage/fset-rN4kfh/251105016/dllm_test`。

所有指标分别呈现，不以较低训练loss代替生成/表征改善；只比较同seed同epoch终点，保留完整中间曲线。整链AAR受长度、保守区域影响，须并看条件CE、长度误差、CDR及表征AUROC/AUPRC；本轮单种子筛选不能给显著性或结合特异性结论。

## 实际启动回执（2026-10-29T04:57:31Z UTC）

控制器PID193460，B1 worker PID193467；实际命令已从/proc核验为冻结快照下的`--train-one B1`。最大长度2048/micro56的3次真实更新通过，峰值allocated=41.96GiB、reserved=55.78GiB，外部显存计入后仍低于85%门槛。随后全新exec已进入正式训练，首次核验step1，当前记录step11，MLM loss=2.973250且梯度有限；这些只是启动检查，尚无本轮终点改善结论。B2/B3真实GPU检查在各自启动时执行。

[启动证据](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/round2/startup_verified.json)。完成这一轮启动核验后不持续人工轮询，后台控制器按既定顺序自动执行与发布。
