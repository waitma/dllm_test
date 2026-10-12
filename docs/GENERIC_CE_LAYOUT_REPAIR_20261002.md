# 通用 CE 布局修正验收记录

更新 2026-10-02T05:09:28.411744+00:00。

CE 行优先布局修正版已于 2026-10-02 05:00:05 UTC 在原 GPU1 一次通过原 tiny：26 个参数梯度全部通过，梯度倍率 1、相对 L2 误差 0、RNG replay 一致，loss 最大差 4.76837e-7。原阈值、GA8、BF16、seed43 和独立 oracle 均保留，没有新增 hooks 或独立 CUDA 算子测试。tiny 1.780 秒、总监督 20.589 秒，真实退出码 0，子进程、对象及独占临时资源清理完成。

两套环境各 4 个新增模型数学合同及 3 个新增软件来源合同均通过；软件首轮前 2 个通过，第 3 个因自有 tmp 父目录缺失在 fixture setup 失败，补目录后仅该 body 首次执行通过。没有重复已通过项目，错误原件及来源保留。新不可变包 534 个源码文件，model SHA 7e79bfb7d2327c07aa3c39e0bed89997fa627b3fa6e20cb084630b347e886592，manifest SHA 2c2c9ee3ca2f17c29da402b937902d158d82eeedb5ba7753dcf1e0be1307c148。

这只证明新源码的原 tiny 数值检查通过；该检查仅 8 条人工记录、1 个窗口，LR=0、参数变化=0，不是学习效果证据。旧源码完整工程验收仍 7/8，旧 R21 seed43 失败原件保留，不能与新 tiny 相加成 8/8。新源码完整 release 尚未执行完成/未准入，当前完整 entry 仍固定旧目录，新源码完整 profile 尚未测量；所有完整 GPU、数据与训练准入标志仍为 false。

2026-10-02T05:06:26.708214+00:00 只读快照：原八卡池 pending/running/results=0/0/56，8 个 worker 都在维护待机，心跳均小于 5 秒，监听 PID2213164 存活。此刻没有新平台状态查询，不把缓存 Running 当作新查询；未重启容器/worker/监听，未 Volc submit/cancel。checkpoint 创建/删除均为 0，自有 CPU tmp 已删除，GPU 子进程独占 tmp 清理有原报告；保留全部必要来源、原数值/失败证据及实际训练恢复状态。

用户指定数据仍为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_mhc_sequence_v1，manifest SHA e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94，train 6,849,243 / valid 52,572。manifest 的 training_ready 字段缺失，decontamination=audit_pending；audit_certificate.json、downstream_scope_certificate.json 仍缺失，父级当前范围审核未闭环且新增 manifest 未被父证书覆盖。既有清理保留，此状态不证明已发生泄漏。没有读取真实 shard、小 batch 或启动真实语料训练。

下一真实工程动作是新建不可变的完整 release 入口/运行包版本，保持既有来源验证、独立 oracle、阈值和数据门禁；不能热改已冻结的本轮 private/runtime 包。复用未变且已接受的模型数学证据，仅补必要的新入口/来源控制、当前源码完整 profile、完整梯度核对、3 个正式优化窗口、分 mask 评估及退出清理证据。当前源码学习、完整 checkpoint 恢复和原 data-selection 门禁仍需闭环。不再增加 hooks、重复已接受 tiny/CPU 项或迁移旧七项证据；不恢复此前停止的序列注释分支，独立 CUDA 算子验证继续暂停。正式数据审核未满足时不得启动真实小 batch 或八卡语料训练，用户授权无需重复确认。

[原 tiny 独立强复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/tiny_terminal_root_review.json)；[当前资源与数据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/final_state_after_CE_tiny.json)；[下一具体工程动作](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/next_engineering_after_tiny.json)；[模型 CPU 合同](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/model_CPU_contract_results.json)；[软件合同与清理](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/runtime_CPU_controls_and_cleanup.json)。

本轮模型仅修改原 forward 的 CE 张量布局：在 selected token 的 [N,V] 上计算交叉熵，再 masked_scatter 返回原 [B,L]；AA/end/balanced/aux、架构宽深、位置与链编码保持原代码。属于通用数值工程修正，不构成新生物架构优胜判断。辅助分支未额外执行，AST 保持，不能声称新辅助分支验收。

原 job：1790917182885479822_d03_r21_seed43_tiny_gpu_observation_v1_6c5bc4a4，GPU1，完成 2026-10-02 05:00:05.940501 UTC。独立只读 agent 核对报告/COMPLETE 与旧七份 full proof 来源，确认当前完整入口仍固定旧目录；本次没有运行额外模型、构建或任务。
