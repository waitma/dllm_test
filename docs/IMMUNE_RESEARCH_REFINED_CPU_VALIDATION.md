# RefinedModel 独立 CPU 等价性验证

2026-09-30 UTC：80 项全部通过，pytest 用时 14.27 秒。验证只使用人工输入和小宽度模型，没有修改生产模型实现，也没有放宽输出比较容差。

被验证实现：[research_refined/model.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/research_refined/model.py)，SHA256 `333eb67f191f0e890f93ed9806b3652cf1b15913d6ef9053e37a2d42c482c2da`。

测试文件：[test_research_refined.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_research_refined.py)。覆盖 R20–R27 全部八种变体、FP32 与 CPU BF16；d32/L4/H4/MLP64，容量48覆盖局部注意力 radius32。其目的为验证缓存优化的行为等价性，不能替代实际 GPU profile 或模型质量评测。

| 验证 | 数量 | 结论 |
| --- | --- | --- |
| 相同种子/权重，有 padding 与无 padding，logits、loss、全部参数梯度 | 32 | 通过 |
| activation checkpointing 下同时保留两个 forward 图，与未缓存父模型对照 | 16 | 通过 |
| 同一次 forward 按拓扑复用 bias；改变链 ID 和 attention mask 后跨调用隔离 | 16 | 通过 |
| save/load、严格权重与输出一致、未知 schema 与错误 parent_variant 拒绝 | 16 | 通过 |

Logits、loss 和恢复输出均使用零容差；梯度比较使用 FP32 `atol=2e-6`、BF16 `atol=3e-5`、`rtol=1e-5`。两种精度下均同时检查有限值。每次临时 checkpoint 使用 TemporaryDirectory，成功和失败路径都会删除。

初次执行曾出现16项保存恢复比较失败。定位发现既有 LLaDAModel 构造器会调用 `torch.backends.cuda.enable_flash_sdp(True)`：在外层 `sdpa_kernel(MATH)` 内创建恢复模型会重启 Flash SDPA，从而使前后两次 CPU attention 使用不同后端。原 ArchitectureModel 与新 RefinedModel 都能同样复现；权重与 RoPE 缓存逐元素一致。将两次模型构造放在共同的 backend/autocast/no_grad 上下文外，再在相同上下文内比较，FP32/BF16 差异均为零。没有通过放宽容差隐藏 loader 问题。

复算命令（先激活 pllm）：

```bash
USE_TF=0 python -m pytest -q /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_research_refined.py
```

机器记录：[validation.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_refined_validation_20260930/validation.json)；完整输出：[pytest_final.log](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_refined_validation_20260930/pytest_final.log)。1920条 warning 来自既有 CPU autocast API 弃用提示，没有执行失败。
