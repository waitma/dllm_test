# 正式 Trainer 原生恢复观测

## 2026-10-03T01:27:30Z 根端审阅、新v3冻结和门禁来源绑定完成

三份源码/归档、安装HF/Accelerate与stdout SHA已根端核实，原14顶层函数AST不变，未重复验收。审阅原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/resume_observation_root_review_v1.json（SHAb87aa4c0f2d48a26373c4f0cd60eaab5d44b14bae0966e8a6e06a37ed7d80ffe）。

新源码根 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3，611文件；清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_manifest_v3.json，SHA256 992bc3c5718b71c379c7a2ee4f39d8e5866f974266500d2c2ec69d5ab4db700b。driver/helper SHA仍是下表407461…/0a0ca3…，旧v1/v2不改。

原helper在新身份下一次现行assert_ready通过59.723秒、rc0，数据三份metadata SHA与v2相同；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/parent_gate_v3.json，SHA256 3bb405020788470b823fe15ee49016143582ab42ee56ecbd3102716c96835122。实际stdout /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/parent_gate_v3.stdout.json，SHA256 443990d6fb1e438d5451bc2b84dbbbb6e4df7910641da1a54247b382a4c96435。未签证/改ready/scope、模型/GPU、发布或入队。剩余是候选八卡两段真实训练的八rank恢复观测及续步原件，当前仍不称实际恢复通过。

2026-10-03T00:52:52Z：公共源码准备及新增 CPU 控制完成，尚无真实八卡恢复通过。未修改冻结 v1/v2、pool wrapper/dispatcher、公共入口、数据、证书或队列；本项不发布、不入队。主会话另行同步全局项目文档并审阅新 v3 冻结。

## 最小接入

公共入口 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/run_v7_formal.py` 的 `FormalBaseTrainer` 新增两个薄 override：`_load_optimizer_and_scheduler` 和 `_load_rng_state`。各自只调用一次原生 `super()`；`checkpoint=None` 原样返回且不建立恢复观测。实际恢复时，在原生方法返回后通过现有 collective 传播观测失败。原 optimizer/scheduler 更新、数据跳过、RNG 恢复时机和数学流程保持不变。

新 helper `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/formal_resume_observation.py` 冷导入仅 stdlib。构造发生在原 resume gate 成功后，绑定 checkpoint 路径、正式 sidecar SHA、binding SHA、rank 和原 HF 方法对象/实际源码文件 SHA/方法源码 SHA。读取相应状态文件前后检查 sidecar 所列 SHA。对 optimizer/scheduler 的完整 `state_dict()` 做带类型、容器、键、shape、dtype、所有值的递归摘要比较，允许正常 CPU/GPU device 搬移；不把 `torch.save` 序列化字节摘要误当作内存状态摘要。

各 rank 只比较自身 `rng_state_<rank>.pth` 的 Python、NumPy、CPU Torch 及可见 CUDA RNG。读取状态使用 getter，不调用 setter、不采样、不取额外 batch、不调用 step。每次观测前后 RNG 摘要必须一致。结果只保留摘要、文件与原生方法来源，不保存 RNG 原始内容、模型张量、序列或 logits。

worker 实际 Trainer 来源 `/vepfs-mlp2/c20250601/251105016/conda/envs/flow/lib/python3.12/site-packages/transformers/trainer.py`：第 2361 行在 Accelerate 包装后恢复 optimizer/scheduler；第 2458 行处理 epoch 边界 RNG 恢复；第 2504 行处理 skip/prefetch 后 RNG 恢复。观测在这两个原方法返回处，不移到 `on_train_begin`，不改变原生数据预取行为。

段结束时，必须收齐八 rank 的 optimizer/scheduler 与 RNG 两类真实返回观测，checkpoint/binding/原生方法来源一致，而且实际 final step 大于 prior step。之后 rank0 独占保留 `RESUME_<prior>_TO_<final>.json`；仍执行原生 checkpoint、八 rank 参数一致性与原 `SEGMENT/COMPLETE` 流程。该新增原件仅证明恢复时刻的内存状态匹配及实际续步，不宣称连续训练与恢复训练后续轨迹数值等价。无需新增连续训练对照分支。

## 本轮唯一 CPU 控制

激活 pllm 后，显式使用 `/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python -B /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_formal_resume_observation.py`。5/5 通过，27.250 秒，exit code 0；未重复旧 suite。

测试从实际 driver AST 提取两项 override，调用实际安装 HF 原生恢复方法各一次，使用真实 AcceleratedOptimizer 的 state_dict/load_state_dict 委派。人工 CPU AdamW 状态手工填入，没有模型构造、forward、backward 或 optimizer step。覆盖 None 透传、缺少方法观测/续步、七份或缺项回执、参数组差异、文件 SHA 漂移、方法来源漂移、错误 rank、RNG 内容差异，以及 helper/driver 冷 runpy/parse 不导入 ML。观测前后实际 CPU RNG 摘要相同。

CUDA 可用性在控制中固定为 false，相关 state/init/seed API 设置为调用即失败；CPU adapter 明确关闭 CUDA RNG 读取。八份回执仅为内存中的人工 metadata 副本，不是八进程验收。所有自有 TemporaryDirectory 人工 checkpoint 经 unittest cleanup 清理；未留下假 checkpoint/正式恢复通过件。原始 stdout 的库导入提示保留不改写。

原件目录 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_resume_observation_controls_20261003`：`receipt.json` 记录执行、来源与边界；`stdout.log` 是此次唯一执行的完整 stdout/stderr；`source` 保存本轮三个 Python 原件。

| 文件 | SHA256 |
| --- | --- |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/run_v7_formal.py` | `40746122485cbaad7acc8eee6eab9a4b72b51ef7b9a3207e89e1d93cddf0391a` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/formal_resume_observation.py` | `0a0ca3baa74ae2567953db9fdebab01f031fe0a3ebb1f4ccd28405bf78fa4c97` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_formal_resume_observation.py` | `44d52d73fff0edef60da8c815a35768bacf07ff55412aa01e0a066e2ee6a6c05` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_resume_observation_controls_20261003/stdout.log` | `31d103b944531d1288a2161ddb2291c3af9399176dc17a635dafb7c6de64a199` |

尚待主会话审阅、新源码冻结及其门禁绑定，并在以后授权的真实候选八卡分段训练中产生原生 CUDA RNG/optimizer/scheduler 恢复观测和续步原件。当前 CPU 结果不授予 GPU 验收结论。既有停止的序列注释分支、独立 CUDA 专项暂停与数据门禁保持。
