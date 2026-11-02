# 本地最终检查点复核

2026-11-02 用户授权在本机再次验证。对已完成四卡确认的 A0、B0(A5)、B1 × seed42/43/44 最终 step3828 检查点做单卡串行复核，先 seed44，再42、43。

入口：

```bash
bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_local_recheck.sh --gpu 0
```

输出 `/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_local_recheck_20261102`。

先核验原冻结源码与快照、固定10%索引和评估文件、9组完整检查点的54份文件SHA、完整连续3828步训练日志以及原抗体辅助五折输入。然后用同一个冻结快照中的模型和evaluate函数，重跑原confirm token验证、16轮解码、冻结骨干表征probe。原抗体五折head结果仅做产物与输入核验，不重新训练该head。

模型、训练数据格式、读入/切分、评估batch、解码规则和稳定提升门槛均保持。每组使用独立进程；CUDA数字卡号0对应本机唯一可用GPU，设置PCI_BUS_ID，清理继承的分布式变量。核验实际CUDA小算子、torch/CUDA/软件包版本、GPU型号及能力和注意力后端，与检查点绑定一致才加载评估。原四卡输出和生产目录只读，全部重评估产物写新目录。

逐项比较原始与重跑指标、逐样本预测，严格检查身份/顺序、离散计数；浮点同时记录完全相等与原严格阈值atol1e-8/rtol1e-6，不通过也完整记录差异，不放宽门槛。9组完成后直接调用冻结calculate函数重算5000次配对bootstrap，绕过原decision缓存；输出单独decision与差异。不会覆盖原统计或按新评估挑checkpoint。

这是相同权重的可复现性核验，不增加独立训练种子，不把原not_yet_stable变成稳定提升证据。新架构和独立泛化验证需另立实验。状态见output的execution.json、REPORT.md及逐组console.log；全部完成或出错后自动保存报告并commit/push探索分支，推送失败仅记publication.json，不覆盖验证结果。

代码在原冻结绑定之外。启动前5项CPU回归与shell语法通过；原冻结源码SHA保持 `3c9c424d816b9fda7adfba5047d9e075ac15dc036342b59461edd35031e50971`。本机实际CUDA初始化和小算子已通过。实际启动PID32781；9组54份checkpoint文件与全部评估输入/日志核验通过；首组A0 seed44已在本地GPU加载原最终权重，开始重评估。

本次GPU重评估9/9已完成，全部指标/逐样本完全相等；额外CPU独立重算生成、AUROC/AUPRC和抗体五折预测指标，并核验45个五折head权重SHA通过。诊断与图见docs/arch70m_results/local_recheck_20261102/REVIEW.md；5000次冻结统计已独立重算，全部51项数值与原判定完全相同；最终状态completed，原产物哈希保持，decision仍为not_yet_stable。
