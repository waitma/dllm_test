# V11 训练速度诊断（2026-11-05T09:16:00.576504+00:00）

当前八卡日志最近200个记录点均值 0.850887 秒/更新；step 10850/300000。记录间隔25步，只计更新，不包含验证、保存和loader等待。正式训练未改动、未中断。

已证实的可优化开销是沿用1B的整层激活重计算：本地受控开/关/开对照约0.743→0.577→0.737秒，关闭减少22.35%耗时、吞吐提高28.78%。这是一张本地GPU上的实测，不能外推为八卡已提速。

| 本地设置 | 每次64条记录的更新均值/秒 | 峰值reserved/GiB |
|---|---:|---:|
| micro16 × GA4，重计算开 | 0.747930 | 6.076 |
| micro32 × GA2，重计算开 | 0.781830 | 11.553 |
| micro64 × GA1，重计算开 | 0.788973 | 22.389 |
| micro16 × GA4，再开重计算 | 0.743236 | 6.076 |
| micro16 × GA4，重计算关 | 0.577134 | 15.211 |
| micro16 × GA4，恢复重计算 | 0.737338 | 5.879 |

相同冻结V11代码/75,741,903参数/BF16/efficient attention，原V8 reader、collator、corruption和native trainer.update；从checkpoint10000读权重，每组重建相同AdamW与原warmup scheduler。四组真实batch按正式seed42/epoch0/global512/rank0/world8/批次10000–10003取样，每组64条，长度788/1029/787/789；噪声固定并记录逐张量输入哈希。各组4次预热＋16次计时，CUDA仅在计时边界额外同步。更大micro没有提速；不能据显存空闲推断它更优。

重计算开/关/开均完成20次更新，loss逐点完全相同，全部参数SHA均为 `0b86d8f21c962ec1c458003ef6fe37feebd2a41afa502519eaa7aff3bcd34423`。这是该短窗口的等价证据，未检验八卡恢复，不应解释为任意精度/任意长度全程必然逐位一致。

另外复用冻结的synthetic_batch容量输入，在maxlen2048、每rank64条、micro16×GA4、不重计算下完成3次真实native DDP更新：allocated 35.603 GiB，reserved 40.797 GiB，总容量 94.971 GiB，额外占用 1.246 GiB，低于85%门槛。只是本地单卡容量资格，正式八卡仍待同机执行与原生恢复验证。此处激活重计算不同于磁盘checkpoint保存；未改保存规则。

实际八卡8份NCCL启动日志均已使用SHM，累计64条SHM/direct通道；旧禁用SHM的Socket问题已修复。Profiler确认efficient attention；micro16一个额外独立profile更新记录10,551次cudaLaunchKernel、198次cudaStreamSynchronize和324次aten::item。链内RoPE逐层读取position_ids.max、注意力bias逐层cast/填充及整层重计算是后续独立优化对象；同步等待与GPU计算重叠，不能把其CPU总时间直接当可删除耗时。

A100历史75M配置是micro32×GA2×8/global512，V11是micro16×GA4×8/global512，且V7/V8数据目标、A0/A5注意力调度不同。用户所述0.84秒历史逐步计时未找到原件，不能据此声称完成公平跨卡测速。硬件峰值算力也不决定整步速度：[PRO6000 Server规格](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)列内存带宽1597GB/s，[A10080规格](https://www.nvidia.com/en-eu/data-center/a100/)列1935/2039GB/s并支持NVLink；虚拟GPU的具体SKU及旧A100节点是否SXM/NVLink未独立核实。

后续候选为保留micro16×GA4/global512，取消整层激活重计算；需要独立续训包、八卡同输入性能与最大长度资格、原生optimizer/scheduler/RNG恢复核查后再切换。当前仅完成诊断，运行中的冻结bundle与超参数保持。

入口：`/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/benchmark_v11_microbatch.py --output ABS_NEW_DIRECTORY [--compare-checkpointing] [--checkpoint ABS_CHECKPOINT]`；最大长度入口：`/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/profile_v11_without_checkpointing.py --output ABS_NEW_DIRECTORY`。先激活pllm，预加载 `/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/v11_a5_75m_300k_20261105/bundle/runtime/libnccl.so.2`，使用数值CUDA_VISIBLE_DEVICES=0、CUBLAS_WORKSPACE_CONFIG=:4096:8、移除CUDA_LAUNCH_BLOCKING；只在本地独占空闲GPU运行。脚本持同一GPU lease，不记录W&B、不保存模型权重。原始输出保留 `/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/v11_speed_diagnosis_20261105`，小型审计副本在本目录。
