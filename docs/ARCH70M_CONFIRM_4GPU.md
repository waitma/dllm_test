# 四卡独立运行：完整架构确认

用户于2026-10-31要求取消本地探索，由用户提交四卡作业。原本地控制器PID307655和worker PID307866已退出，GPU已释放。最后记录step150，最后完整checkpoint为step0；日志、原状态与进程信息已归档到`/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_confirm_20261031/local_cancelled_20261031T054604Z`，权重与恢复文件保持。本次没有提交远程作业或重新启动本地GPU任务。

## 提交作业时的命令

四卡作业容器启动命令：

```bash
bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_confirm_4gpu.sh --gpus 0,1,2,3
```

入口自动激活pllm并使用已有arch70m_env Python；代码/数据和环境沿用当前共享存储路径。若作业通过Git检出代码，使用`explore/arch70m-v8-20261028`的最新提交，确保本脚本及配套Python调度器存在。不在运行过程中自动git pull。

默认输出独立为`/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_confirm_4gpu_20261031`，九组都从随机权重开始。原本地输出`/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_confirm_20261031`保持，停止前150步中没有完整保存的更新不会被伪装成恢复进度。新四卡任务之后若中断，重复同一命令可从各组latest完整checkpoint恢复，最终完成且绑定/文件校验通过的组跳过。

可选参数：

```bash
# 只打印九组分配，不访问GPU、不创建试验目录
bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_confirm_4gpu.sh --gpus 0,1,2,3 --dry-run

# 使用可见的另外四张卡；四个ID必须不同
bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_confirm_4gpu.sh --gpus 4,5,6,7

# 网络暂时不可用时仍训练并本地提交，稍后单独发布
bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_confirm_4gpu.sh --gpus 0,1,2,3 --no-push
bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_confirm_4gpu.sh --publish-only
```

支持`--output-root /absolute/path`另指定目录；默认目录用于本次四卡运行，恢复时继续使用同一目录。`ARCH_CONDA_PREFIX`可指定已准备好的兼容conda环境，`ARCH_PYTHON`可指定对应Python。运行依赖当前共享存储中的原固定10%索引、prepared分片与筛选证据；未修改任何训练记录格式。

## 确认矩阵与分卡

四张卡各自执行独立单GPU试验，每张卡同一时刻仅一个worker；各组micro32×16=global512，独立随机初始化3epoch/3828步。A0原始架构、B0=A5全局注意力、B1=全局＋可学习残差，配对seed42/43/44共9组。

| GPU | 顺序 |
|---|---|
| 0 | A0 seed42 → B0 seed43 → B1 seed44 |
| 1 | B0 seed42 → B1 seed43 |
| 2 | B1 seed42 → A0 seed44 |
| 3 | A0 seed43 → B0 seed44 |

GPU编号按`--gpus`依次替换。NVML UUID仅用于去重审计，实际CUDA_VISIBLE_DEVICES使用数字selector并设置PCI_BUS_ID；实际CUDA初始化/内核/三次更新检查，以及四个真实CUDA UUID去重均在worker上进行。

## 显存、冻结与恢复

先四路并行完成九个架构/种子组合的最大长度2048三次真实更新。每个资格检查使用独立shadow输出/index/scratch，避免多进程同时复制或截断同一索引。九组全过原85%显存门槛并核验四张不同的真实CUDA卡、相同GPU型号后才开始正式训练。

每个正式训练worker重新启动，从内容寻址快照导入既有模型/训练器。冻结模型/配方/数据/统计代码SHA仍为`3c9c424d816b9fda7adfba5047d9e075ac15dc036342b59461edd35031e50971`；调度器和新sh独立记录自身SHA，未改变原checkpoint绑定。恢复时保留原profile文件字节和SHA，额外的当前GPU重新资格证据单独保存，避免破坏已有checkpoint的profile绑定。原训练器继续严格检查环境/模型/数据/binding，恢复需要兼容的原环境。

四路正式训练共享发布锁，但各自写自己的checkpoint/JSONL。逐组完成事件由父进程串行生成报告、commit并push；推送失败pending与训练退出码分开保存。父进程收SIGINT/SIGTERM时终止自身启动的worker进程组，避免遗留占卡进程。重复启动相同输出目录受controller/trial文件锁保护。

## 记录与验收

运行清单：`/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_confirm_4gpu_20261031/four_gpu_execution.json`。
每次执行详细清单与日志：`/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_confirm_4gpu_20261031/four_gpu/attempt_*/manifest.json`、`logs/*_train.log`、`logs/*_qualification.log`。
各试验状态：`/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_confirm_4gpu_20261031/trials/confirm_<variant>_s<seed>/status.json`，`process.json`包含GPU、命令及本次日志绝对路径。

[完整训练/评估报告](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/confirmation_20261031/REPORT.md)仍作为本轮主报告，summary.json记录实际输出目录。采用既有confirm留出、抗体辅助五折及三种子配对区间判据，完整标准见[稳定性方案](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/ARCH70M_STABLE_VALIDATION.md)。四卡并行不改变训练目标、噪声、解码、统计门槛或数据形式，未达到门槛不宣告架构稳定提升。

CPU调度/绑定测试8项、bash语法检查及真实sh的dry-run通过：九组矩阵唯一、四路并发/同卡不重叠、主线程结果发布、失败原文保留、分布式环境清理、NVML数字selector、旧profile SHA保护、真实CUDA UUID去重。开发机仅一张GPU，未运行实际四卡训练；真实资格检查由提交后的四卡worker执行。[检查回执](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/four_gpu_delivery_20261031/validation.json) · [本地取消回执](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/four_gpu_delivery_20261031/local_cancellation.json)。
