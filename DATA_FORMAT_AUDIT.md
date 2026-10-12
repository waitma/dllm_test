# 当前训练数据与格式

更新：2026-10-07T06:41:12.247809+00:00

## 数据选择

**本次新训练使用V5**：/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v5_receptor_completion。

| split | manifest记录数 | 分片数 |
|---|---:|---:|
| train | 7,626,885 | 78 |
| valid | 102,308 | 7 |

已逐条检查全部7,729,193条，85分片行数与manifest一致，并计算全部分片SHA256。来源为OAS、OTS、asd_antibody、TRAIT、tcr_native、tcr_papers、tcr_repertoire。沿用V5已有数据及过滤结果，不叠加V7后续清理作为默认输入。

V7历史数据仍保留于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_scope_final_20261002，只用于其已有训练／评测追溯。本次不修改或删除任何数据分片。

## 输入规则

| 类型 | 干净输入结构（受体内容随后补mask至302容量） |
|---|---|
| 无条件抗体 | `<prots><null><protd><unknown><prots><ab>H.L<protd>` |
| 抗原→抗体 | `<prots>antigen<protd><关系><prots><ab>H.L<protd>` |
| 无条件TCR | `<prots><null><protd><unknown><prots><tcr>beta[.alpha]<protd>` |
| 肽→TCR | `<prots><pep>peptide<protd><关系><prots><tcr>beta[.alpha]<protd>` |
| pMHC→TCR | `<prots>MHC<protd><binding><prots><pep>peptide<protd><关系><prots><tcr>beta[.alpha]<protd>` |

- 语义JSONL保存chains、chain_roles、regions、chain_metadata等；送入模型时统一转换，不复制一套文本数据。
- 对象容量302（166+134残基＋split＋protd），总输入上限2048。生成不需要真实目标长度。
- attention按`<prots>…<protd>`对象分段，头部和内容同段；内部`.`不分段、不重置位置。目标对象的全部预留槽位同段，不按真实／预测终点动态缩短。
- 保留CDR-only全长scaffold及真实锚点；synthetic_residue_mask标记的位置转mask，labels=-100。真实观测X保持原样。
- synthetic和尾部mask参与attention但不计损失；真实残基、`.`和`<protd>`参与监督。固定对象头部作为条件。
- VHH不纳入本版范围；本次全量扫描确认VHH为0，无需额外排除。

## 容量依据

现有固定canvas代码 /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/grammar.py 定义H/β encoder 168、decoder 167、残基最多166；L/α分别136、135、134。encoder包含cls，decoder至少留一个EOS。

新内容预算为 **166+134+split(1)+protd(1)=302**；固定头部prots与ab/tcr在预算外，整个受体块为304。只采用总预算，不限制每条链分别166/134；单链也使用同一对象预算。split位置由模型学习，attention仍按对象分段。全量检查确认V5所有对象均能放入302，无需裁剪或丢弃。

## 全量检查结果

| 对象类型 | 条数 | P99 | 最大长度 |
|---|---:|---:|---:|
| 完整双链 | 4,839,491 | 244 | 299 |
| 补全scaffold双链 | 2,764,364 | 237 | 271 |
| 单链 | 125,338 | 142 | 146 |

对象长度包含split和protd；总输入最大1218，低于2048。585,124,843个synthetic位置的长度／取值检查通过；非法记录、超容量记录、VHH均为0。

全量逐条执行schema、残基、角色、synthetic及等价布局长度检查；另以每分片3条、共255条真实collator输出核对公式。启动门禁再次校验全部分片SHA256。完整报告 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/full_audit.json；启动回执 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_v1/native_parent_preflight.json；97项代码回归测试通过。

这次是V5布局与来源绑定审核，不重新运行benchmark去重，不把V7隔离结论套用到V5。原V7抽样报告仅作历史记录。旧完整数据审计见[归档](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive/project_docs_20261007T061603Z/DATA_FORMAT_AUDIT.md)。
