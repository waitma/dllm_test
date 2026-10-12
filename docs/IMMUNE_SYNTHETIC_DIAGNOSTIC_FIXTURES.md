# 扩展人工诊断集合

日期：2026-09-30 UTC。实现：[expanded_fixtures.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/expanded_fixtures.py)。测试：[test_expanded_fixtures.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_expanded_fixtures.py)。

本集合只由程序生成的字母串组成，没有读取生物语料、下游 benchmark、test 或 few-shot 文件。它用于排查学习、条件利用、掩码和容量接口；即使执行500步，仍是**人工学习测试，不是初步真实语料训练**。人工 heldout 改善不证明生物学泛化。

## 接口与目录协议

三个接口均返回 `FixtureSplit(train_records, valid_records, manifest)`，records 为 `BioSeqRecord` 的 tuple：

```python
build_fixture_split(seed=42, train_size=128, valid_size=24,
                    full_capacity=256, fragment_capacity=64, max_length=2048)
write_fixture_manifest(directory, **generation_arguments)
load_verified_fixture_manifest(directory)
```

目录只需要 `synthetic_fixture_manifest.json`，`schema=immune.synthetic_fixture.v1`，`kind=synthetic_functional_diagnostic`，`biological_quality_claim=false`。writer不保存序列文件；loader根据manifest中的整数生成参数重新产生全部records，并逐字段核对manifest、生成器源码SHA、record/target/content hashes和manifest自身hash。未知字段、外部records参数、旧源码、改动的hash或计数均拒绝；目录内其他records文件不会被读取。

相同参数写入同一目录是幂等操作；不同参数要求新目录。临时manifest通过同目录hard link原子发布，不覆盖另一写入者已发布的manifest。生成器使用局部 `random.Random(seed)`，不改变全局Python或Torch RNG。

下一阶段的production人工配置为 **576 train / 72 valid**，每个“6组×3长度”单元恰好32/4条。默认128/24仍保证各单元存在，数量差不超过1。小CPU配置可用36/18、full64、fragment32、max768；也测试了full48、max256。

## 分组与可学习关系

固定元数据键为 `record.metadata['diagnostic_group']` 和 `record.metadata['length_bin']`，后者取 `short/medium/long`。另外记录 `synthetic_only`、`fixture_id`、`target_roles`、`condition_roles`、`target_scope` 和 `conditional_signature_visible`。

| diagnostic_group | 固定条件 | 监督目标 |
|---|---|---|
| ab_antigen | antigen | heavy + light |
| tcr_peptide | peptide | alpha + beta |
| tcr_pmhc | peptide + mhc | alpha + beta |
| unconditional_ab | 无 | heavy + light |
| unconditional_tcr | 无 | alpha + beta |
| unconditional_single | 无 | VHH |

每条record有6字符人工标识。条件组可直接从antigen/peptide前缀获取该标识；pMHC组从peptide与MHC各取3字符拼接，两个上下文共同决定目标。上下文同时暴露长度档标记A/G/S，因此目标不包含额外的隐藏随机nonce。目标内部重复“6字符标识+2字符role标记”的8字符motif，提供局部重复和条件复制关系。

不同record的6字符标识通过有限空间上的奇数仿射置换生成，保证该配置内唯一。每条目标序列都以实际字符 **C** 开始，以 **F/W** 结束；这些字符保留为监督残基，后面另有A4的`<protd>`监督位置。每条目标满足`len(sequence)+1 <= full_capacity`。train/valid按各自分层索引产生，并使用不重合标识区间；不仅整条record不重复，**任一单链目标字符串跨split也不精确重复**。共享人工motif及任务规则是有意设置，不能理解为生物序列同源隔离。

所有目标使用`full` observation scope。`fragment_capacity`记入manifest供collator配置一致性检查，本版不声称覆盖junction/core/fragment模式。

## 长度与不可识别限制

默认full256对应目标长度24/127/255，target真实最大位置间距超过32；较小full48对应12/23/47。manifest分别记录train/valid中`targets_with_residue_distance_gt32`和`cross_radius32_coverage`。这是物理长度覆盖，不是“已证明模型学会长距离推理”的结论。

若full_capacity≤34，生成器保留合法的小型CPU样例，并明确记为无法覆盖`>32`残基间距。若样例数不足18导致缺少某个分层单元，也会在manifest中显示`all_group_length_cells_present=false`。无法容纳两个目标与两个条件的max_length直接报错，不静默裁剪链。

**无条件 full-mask 不可识别是任务本身的限制。** 同类无条件record的token/role/scope/chain/position/capacity特征在全部目标槽位被mask后完全相同，但标签中的目标标识和真实长度不同。测试实际验证了这种输入碰撞，因此不能要求100%终点恢复率。部分掩码能暴露重复motif；条件任务的标识来自上下文，适合观察条件利用，但有限训练步也不保证完美恢复。

真实数据分支仍必须先执行原`assert_ready`。人工目录仅可由专门的synthetic路径接受；不能靠字段名或手填JSON绕过真实语料去污证书。

## 验证记录

2026-09-30：**32项CPU测试通过，6.30秒**；仅2条已有Torch autocast弃用warning。覆盖576/72逐单元32/4、确定性与不消耗全局RNG、任一目标跨split去重、从上下文重建全部条件目标、C/F/W边界、容量、MHC组合、跨radius32覆盖及小容量例外、无条件full-mask输入碰撞、全部分组A4 collate、tiny模型有限loss/反传、manifest roundtrip/idempotence及15类篡改拒绝。

首次运行的一个测试断言把context的`<protd>`也计入监督终止数量；修正测试为`terminal_mask & eligible_mask`后全部通过，生成器实现未因此修改。

结果文件：[expanded_fixture_tests.log](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_architecture_next_local_20260930/expanded_fixture_tests.log)、[JUnit XML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_architecture_next_local_20260930/expanded_fixture_tests.xml)。生成器冻结SHA：`2fa112b3707cad12f31946088679daf446f9b6f6d1dd69d8e5c4e2098e778231`。

本模块未修改任何live模型、runner或既有snapshot；没有提交GPU任务，也没有保存模型checkpoint。
