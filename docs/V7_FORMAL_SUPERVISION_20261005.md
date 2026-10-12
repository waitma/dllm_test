# V7 正式训练持续监督

更新时间：2026-10-06T12:21:40.644675+00:00。

收到STOP，监督器停止接续；既有训练任务未被终止。

当前训练段：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/launch_20261005/segment_10016；累计目标 10016；phase=started，terminal=None。目标步数不是已完成步数。

最新原始训练日志：`{"loss": 0.4422, "grad_norm": 0.3607656955718994, "learning_rate": 8.252974755701546e-05, "batch_aa_loss": 0.45805078744888306, "batch_terminal_loss": 0.27643823623657227, "batch_aa_count": 2287.0, "batch_terminal_count": 20.0, "batch_auxiliary_loss": 0.0, "epoch": 0.30350225629966854}`。

完整状态和实际结果：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/state.json；关键事件：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/events.jsonl。

每段全部八rank成功且原生恢复证据通过才接续；失败或STOP停止接段。不重启/重放作业，不改变训练源、审核、batch、目标或清理正式恢复文件。
