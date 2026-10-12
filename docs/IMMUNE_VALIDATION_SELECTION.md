# 固定验证子集的完整性清单

更新：2026-09-30。实现为隔离原型 [validation_selection.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/validation_selection.py)，不改 prepared、训练门禁、live runner 或冻结的 39 个 GPU 源文件；不读取下游评测、执行序列匹配或生物注释，不启动训练。

该选择器在每个 source 内用独立的 `random.Random(42).sample` 选择最多 256 条 valid 记录，与当前 runner 的按来源选择算法一致。小来源保留其全部记录；不拆 family，不移动 split，不通过复制或从 train 补充凑数。选择顺序及 source-local 索引均写入清单。

## 接口

```python
from scripts.architecture_prototypes.validation_selection import (
    build_validation_selection,
    write_validation_selection,
    load_verified_validation_selection,
)

selection = write_validation_selection(
    "/outside/prepared/selection.json",
    "/absolute/prepared/version",
    seed=42,
    max_records_per_source=256,
    min_source_records=100,
    min_source_families=20,
)
selection = load_verified_validation_selection(
    "/outside/prepared/selection.json",
    expected_data_dir="/absolute/prepared/version",
    expected_manifest_sha256=selection["manifest_sha256"],
    expected_selection_sha256=selection["selection_sha256"],
)
indices = selection["sources"][source]["source_local_indices"]
```

`indices` 直接对应 [PreparedImmuneDataset](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/dataset.py) 的 `split='valid', source=source` 实例：按 manifest 分片顺序拼接该来源，分片内只数非空记录。可作为 `Subset(dataset, indices)` 的索引，但选择器自身不构造训练任务或绕过 `assert_ready`。

接口允许显式不同 seed 或较小 cap 做测试；cap 限制为 1–256。真正的对照协议应固定 seed42，并在 run provenance 绑定 `selection_sha256`，不能在各变体间静默换配置或清单。`expected_*` 外部 pin 可防止误用另一份合法选择清单。

激活 `pllm` 后，也可从仓库根目录运行：

```bash
python scripts/architecture_prototypes/validation_selection.py create --data-dir /absolute/prepared/version --selection /outside/prepared/selection.json
python scripts/architecture_prototypes/validation_selection.py verify --selection /outside/prepared/selection.json --data-dir /absolute/prepared/version
```

## 清单内容与验证范围

- canonical 绝对 `data_dir`、完整 manifest SHA、实现源码 SHA、sampling/config、清单整体 `selection_sha256`。
- 所有 valid 分片的相对路径、实际解析路径、来源、记录数和 SHA；包括没有抽到记录的分片及未抽中的行。
- 每条选中记录的 source-local 索引、分片、分片内非空行号、字节偏移、原始 JSON 行 SHA、canonical JSON 内容 SHA、已有 `v7_family_group` 及目标字符串长度。
- 每来源全部 valid 与所选记录分别统计 records、known-family 数、缺失 family 元数据的记录数、没有声明目标角色的记录数。
- 目标链字符串长度、各记录目标总字符串长度及按声明 role 的长度统计：count/min/max/mean/p50/p95，分位数采用 nearest rank。

长度仅计算 `len(serialized_chain_string)`，不截掉端点、不去空格、不删 synthetic mask 位、不检查字母表、不补残基或推断观察范围。角色集合与现有 layout 的五类 receptor role 保持一致；没有把 context 长度混入目标统计。清单不保存完整链字符串。

创建和加载都会扫描整个 valid，校验 SHA、行数和非空行位置。加载重新执行确定性选择并逐字段比较，不能只改内部 digest 来伪造索引、统计、family 或位置。相同输入重复写同一份文件是幂等的，已有不同清单拒绝覆盖。JSON 原子创建在 prepared 目录以外，原数据只读。

**范围边界：** manifest 全部字节受绑定，只有 valid 分片的实际字节由本工具复核。train 分片仍由已有数据认证/训练入口核验；JSON 的 `verification_scope` 明确写出此边界。选择器不读取证书，也不签证；`training_certification=false`。子集确定、SHA 无变化不代表无评测污染或允许训练。

## 小来源与 family 警告

`min_source_records=100`、`min_source_families=20` 是显式、可配置的提示阈值，不是资格线。`available` 与 `selected` 分别报告低记录数、低 family 数、缺少 family 元数据或目标角色的情况。没有 valid 行的声明来源也会以 0 条/0 family 出现在清单与警告中，不会静默消失。缺失的 family 不被赋予新 ID；选中样本的数量与 family 数不足不会靠改 split 修复。

软件测试先使用人工临时 prepared fixtures 完成；随后在授权的真实候选物化完成后，已生成并重新核验 1,288 条 selection，详见 [候选导出实测记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_EXPANDED_CANDIDATE_EXPORT_20260930.md)。没有为未认证数据创建真实训练任务。TRAIT/native 小 valid 的问题仍需结合 [整 family 可行性报告流程](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_EXPANDED_CLEAN_EXPORT.md) 判断。

## CPU 验收

38 tests passed，2.22 秒。测试覆盖：与真实 PreparedImmuneDataset 的 source-local 索引对应、空行/多分片/交错来源/字节定位、固定抽样与全局 RNG 不变、记录和 family 统计、纯字符串长度、原文件不变、幂等、原子小 JSON、选中及未选中 valid 行篡改、manifest 重新绑定后的变更、伪造内容重算 digest、外部 pin、符号链接更换、小/空来源和缺元数据，以及 train 字节校验范围的明确边界。

证据：[测试源码](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_validation_selection.py)、[日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/validation_selection_pytest.log)、[JUnit](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/validation_selection_pytest.xml)、[验收 JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/validation_selection_validation.json)。临时 fixture 已删除；没有 checkpoint。
