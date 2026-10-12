# V5 no-ESMC 270M

实际参数268,590,112；16层/d1024/head16/FFN4096，50词表。继承旧模型数据、布局、训练目标和验证口径；从头训练，不加载75.6M权重。

- 训练配置：/vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v5_noesmc_270m_20261010/training_config.json
- 当前包：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_270m_train_bs64_20261010
- 实际票据：v5_270m_scratch_bs64_20261010.json，现有八卡容器t-20261007153431-vhx2f；无需再次submit。
- 参考YAML：/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/v5_noesmc_270m_8gpu_20261010.yml
- 输出：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_270m_noesmc_300k_b512
- 300000步，micro64×GA1×8/global512，LR1e-4/warmup2000/cosine，best3＋latest完整状态。
- 八卡synthetic已通过，allocated42.79GiB/reserved48.88GiB；micro64是同global512八卡、GA至少1时的上限。真实语料门禁在启动时核对85分片全哈希，训练更新步另查实际日志。
