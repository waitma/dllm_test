# 隔离 pilot 生命周期执行器

日期：2026-10-01 UTC；沿用 `20260930` 实验输出目录。状态：工程原型已通过人工 CPU 验收，未部署。未启动 GPU、真实语料训练、提交或入队。

实现：[certified_pilot_execution.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/certified_pilot_execution.py)。测试：[test_certified_pilot_execution.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_certified_pilot_execution.py)。原有 [certified_pilot.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/certified_pilot.py)、冻结训练源码、数据和证书均未改动。

## 接口和执行顺序

`execute(PilotRequest(...), gpu_release_dir)` 是将来门禁满足后会进行 GPU 训练的 API。本原型没有 CLI、池注册、调度、提交或证书生成器。

1. 首先调用原有 `pilot.preflight`，其第一项仍是 `assert_ready` 双证书数据门禁。缺证、数据或选择清单变更、旧 profile 不匹配等情况在模型构造、GPU 查询和输出目录创建之前拒绝。
2. 调用现有 `verify_diagnostic(..., 500, require_learning=True)`，要求同变体完成 500 步人工诊断、完整曲线、来源和配置一致、既有 heldout 学习检查通过。这里只把人工诊断作为软件和训练过程的准入证据，不视为真实生物学效果。
3. 验证下述未来独立 GPU release。重新使用 `build_trainer`，核对其再次预检的完整 provenance，保持原目标函数、模型、采样、global batch 256、AdamW、学习率及 10000 步调度期限。
4. 保存请求和两层 provenance，并安装 checkpoint 侧车回调。新跑先评估；续训不对刚初始化且尚未恢复的模型进行“初始评估”，直接把已核实 checkpoint 路径交给 `Trainer.train(resume_from_checkpoint=...)`。
5. 要求实际累计步数精确达到 1000 或 5000，然后最终评估、保存指标、模型和状态。前后检查完整源码映射及 GPU release 文件 SHA；核实最终完整 checkpoint 后，最后写 `COMPLETE.json`。失败保留 `FAILED.json` 的阶段、异常和步数，不留下完成标志。

原始预检在调用 `build_trainer` 时再运行一次，这是有意保留的门禁复核。训练期间不引入新的 checkpoint 删除逻辑；保持 HF `save_total_limit=3` 和已有池清理器的 latest-two、实际跟踪的 best、续训引用保护约定。本原型不承诺自动选出 best。

## 未来 GPU release 的窄合同

目前没有生产者，也没有可用的通过材料。已有模型显存 profile 不能代替新增 `TokenValidationTrainer` 和生命周期的 GPU 验收。

未来独立目录需提供 `GPU_RELEASE_COMPLETE.json` 与 `gpu_release_summary.json`，schema 为 `immune.certified_pilot_gpu_release.v1`。完成文件绑定 summary 字节 SHA 和完整 source maps；summary 绑定变体、完整 config/options、原 profile 两文件 SHA、实际软件版本/GPU、指标协议、global/micro/accumulation、精度、优化器和原目标函数。

必须有 `train_optimizer_step` 与 `four_mask_evaluation` 两阶段。每阶段需记录实际 `[batch_size, config.max_length]` 输入形状、与配置相同的 full/fragment 容量、显存总量和真实峰值字节；峰值须不超过总量 85%，不能只填 `status`。训练阶段要求有限 loss、实际 optimizer step 数和有限梯度；评估阶段要求新的 Trainer 和四档 mask。两阶段都需真实 CUDA 执行证据。当前仅定义与验证合同，没有签发或伪造通过证书。

源码映射比较完整键和值，不将当前模型 release 的 39 项写死为永久数量。后续入口、注册表或模型来源发生变化时，必须取得对应新 profile，或者明确从冻结 release 核验；不能复用与 live 源码不一致的旧 profile。适配器和指标源码单独绑定。

## 续训和 checkpoint 完整性

沿用原有 1000 → 累计 5000、以及阶段内中断恢复约束。旧 pilot provenance 只允许目标累计步数字段变化；新的执行侧车同样只排除 `pilot_stop_step`。侧车绑定准入的 500 步诊断摘要、未来 GPU release、指标及完整来源映射。

每次保存侧车记录实际 checkpoint 步数，以及模型、优化器、scheduler、RNG、trainer state、配置、词表和原 provenance 九个文件的字节 SHA。恢复前必须全部一致。最终完成标志另外绑定最终模型、指标、请求、provenance、trainer state 与 checkpoint 侧车 SHA。损坏文件、缺失优化器、来源变化或另一个 GPU release 均拒绝。

这些额外文件哈希的实际 GPU 作业保存开销尚未测量。不存在对断电、SIGKILL 或文件系统故障一定写出失败日志的保证。

## CPU 验收与边界

新测试各环境均 27 项通过：本地 Python 3.10 / Transformers 4.57.6 为 14.43 秒；worker 对应 Python 3.12 / Transformers 4.48.1 为 16.49 秒。两者 Accelerate 1.14.0，Torch 2.8 系列。

测试包括原始数据门禁优先、缺 500 步或 GPU 准入时不构造 Trainer、形状/容量/峰值/NaN/评估协议不足时拒绝、保存恢复侧车、checkpoint 字节破坏、源码漂移、精确累计步数和失败无完成标志。正向准入接口明确 mock；临时“GPU”文件标有 `mock_only/not_acceptance`，真实 GPU 验证器会拒绝它们。没有生成真实或貌似真实的通过证书。

必要的真实 CPU 训练使用四条人工二维输入、三个整数类别及 tiny Linear，实际执行 HF Trainer/AdamW 保存和恢复：先 1000 步，在不同初始化下确认 checkpoint 权重及 global_step 被恢复，再完成累计 5000 步，保存完整状态和侧车。调度期限保持 10000，CPU 测试为了减少临时写盘每 1000 步保存；生产工厂仍沿用每 100 步保存。

**这项测试证明生命周期恢复完成，没有与一次连续 5000 步运行的参数、AdamW 状态、scheduler 或 RNG 做数值等价比较。** 也没有验证生产模型、新指标的 GPU 数值/显存、真实数据可训练性或生物学效果。

实际 candidate 的只读调用仍在缺少 `audit_certificate.json` 的第一门禁被拒绝，诊断/GPU/模型函数均未进入，输出目录未创建。该检查不构成认证。

验收、环境版本、源码 SHA、日志、JUnit 和清理记录：[validation.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_certified_pilot_execution_20260930/validation.json)。所有人工临时 checkpoint 及 mock 准入文件在验收后删除，保留小型测试证据。本原型至此冻结，等待独立数据认证和未来 GPU release 验收，不自动开展下一阶段。
