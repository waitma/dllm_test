# 项目冗余文件检查 — 2026-10-01

## 已执行删除 — 2026-10-01T01:57:43.075437+00:00

用户随后明确要求删除表中六类候选，覆盖下方初次检查时的“建议归档”。已直接删除 **3,678 个文件**（逻辑大小 **197.8 MiB**），并移除 1,906 个空目录；没有保留这些文件的完整内容副本。

| 用户指定类别 | 已删除文件数 |
|---|---:|
| 根目录旧任务 YAML | 28 |
| Python／pytest 缓存 | 775／6 |
| LaTeX 编译中间文件 | 5 |
| 临时 ASD 分析文件 | 20 |
| 8 个历史测试临时目录内文件 | 2,840 |
| 旧训练看护脚本/配置、旧调试脚本 | 2／2 |

删除前逐项复核与原清单一致，本机未发现占用候选文件的进程或运行中的 pytest；保存 SHA256、原路径、大小、时间和临时分析文件行数/FASTA 记录数后删除。未找到独立 ASD 分析结论文档，因此只保留文件级统计，不补造实验结论。删除清单：[JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/project_cleanup_audit_20261001/deletion_manifest.json)、[TSV](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/project_cleanup_audit_20261001/deletion_manifest.tsv)；逐次删除记录位于 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/project_cleanup_audit_20261001/deletion_events.jsonl`。

同级历史测试报告、其余 debug 文件、论文源文件和交付物等 **658 个保留文件**在删除前后 SHA256 一致。通用 Volc 示例和 PID 便签未包含在用户此次列出的六类中，继续保留。既有数据、模型快照、历史归档与当前看护入口保留；没有 submit/cancel。已补 LaTeX 编译中间文件忽略规则，验证见 [validation](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/project_cleanup_audit_20261001/deletion_validation.json)。

验收通过：清单中 3,678 个原文件均有删除事件，658 个保留文件哈希一致，39 个必需源码文件检查通过。删除后 pytest 自动重新生成了 4 个缓存文件（修改时间晚于删除完成时间）；这些是新缓存，未再次删除，详细路径见 validation。运行测试会重新创建缓存，忽略规则不会阻止运行时写入。

下文保留初次检查的依据；其中“建议归档”等表述是删除授权前的记录，不能用于判断当前文件是否仍存在。

初次检查时间：2026-10-01T01:52:41.366620+00:00。初次检查阶段仅新增审计记录；后续删除执行结果见上节。

扫描了 981 份第一方源码、脚本、配置和文档，并核对部分本地进程、文件依赖和内容哈希。完整逐文件路径、大小、时间、Git 状态与引用见 [inventory.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/project_cleanup_audit_20261001/inventory.json)。下面大小为逻辑字节换算，不等于最终可释放磁盘空间；缓存和当前目录可能继续变化。

## 初次检查候选（历史快照）

| 类别 | 文件数 | 大小（MiB） | 结论 |
|---|---:|---:|---|
| 根目录旧任务导出 | 28 | 0.086 | 建议归档；包含 9 对 SHA256 完全相同文件，保留 task ID 映射后可只保存一份内容。 |
| 通用 Volc 示例 | 1 | 0.005 | 可移至示例区或移除；只有 hello-world 入口，无项目引用。 |
| Python 字节码 | 775 | 12.073 | 可清理的缓存；仅本轮列出的第一方目录，不含冻结 release 与第三方环境。 |
| pytest 缓存 | 6 | 0.247 | 可重新生成；仍有近期写入，应等测试结束。 |
| LaTeX 编译中间文件 | 5 | 0.089 | 可重新生成；保留 main.bbl、TeX/Bib/style 源码和 PDF/图片交付物。 |
| 早期 ASD 临时分析 | 20 | 178.737 | 建议先归档；未检出静态引用，不视为已完成所有来源/远端使用排查。 |
| 历史测试临时目录 | 2840 | 6.557 | 可清理候选；应保留同级测试日志与最终审计报告。 |
| 旧 grammar_v2 看护 | 2 | 0.007 | 建议连同配置归档；已暂停且引用入口缺失。 |
| 依赖缺失的旧调试脚本 | 2 | 0.004 | 建议归档；两个 /tmp 输入已不存在。 |
| 失效 PID 便签 | 1 | 0.000 | 可移除；只有 PID=604057，检查时该 PID 不存在。 |

## 删除前核查依据（历史快照）

1. 根目录 `t-*.yaml` 共 28 份，上轮 YAML 整理仅覆盖训练/评测作业目录，因此这些散落导出尚未归档。扫描未发现按这些文件名读取的引用；其中包括早期本项目任务和其他项目任务的导出。9 对内容完全相同，不应把重复文件误算成独立配置。样例：t-20260907044324-tcmhs.yaml（已删除，原路径 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/t-20260907044324-tcmhs.yaml`） 与 t-20260907045511-978hn.yaml（已删除，原路径 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/t-20260907045511-978hn.yaml`）。
2. 旧看护脚本（已删除，原路径 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/watch_llada_train_jobs.sh`） 和 旧看护配置（已删除，原路径 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/llada_train_watchdog_jobs.json`） 中两条任务均 `paused=true`；四个 `yaml`/`scratch_yaml` 路径及两个输出目录均不存在。本机未发现相应看护进程。现行评测看护文件提到它们是在说明“不复用”，并非调用依赖；历史任务账本仍有引用，适合归档并保留映射。
3. comp_check.py（已删除，原路径 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/debug/comp_check.py`） 和 check_composition.py（已删除，原路径 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/debug/check_composition.py`） 读取 `/tmp/vs_s2_designs_esmc.json`、`/tmp/vs_s2_designs_v3.json`；两个文件均不存在。可移出当前调试入口；不能据此整体删除 [debug 目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/debug)，其中部分图表仍由诊断脚本引用。
4. [tmp/asd_probe](/vepfs-mlp2/c20250601/251105016/project/dllm_test/tmp/asd_probe)、[tmp/asd_probe2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/tmp/asd_probe2)、[tmp/asd_sim](/vepfs-mlp2/c20250601/251105016/project/dllm_test/tmp/asd_sim)、[tmp/asd_sim2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/tmp/asd_sim2) 共 20 个中间文件，最近修改为 2026-09-17；未检出本轮范围内的目录名引用。建议先保留分析结论及来源再归档，不将“没有静态引用”升级成“已证明无用途”。
5. [历史测试目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/logs/v2_baseline_validation) 下 8 个 `*_tmp` 子树是测试临时产物；本轮仅将这些子树列为候选，不包含整个 logs，也不包含原始训练/评测结果。LaTeX 中间文件为 main.aux（已删除，原路径 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/paper/main.aux`）、main.fls（已删除，原路径 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/paper/main.fls`）、main.fdb_latexmk（已删除，原路径 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/paper/main.fdb_latexmk`）、main.blg（已删除，原路径 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/paper/main.blg`） 和 main.log（已删除，原路径 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/paper/main.log`），其中四份已被 Git 跟踪，清理时应同步忽略规则。

## 仍有作用或尚不能判断的目录

- [data/ppi_task_raw](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw) 约 191 GiB，但当前 [TCR native 输入适配](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/tcr_native/common.py) / [摄入入口](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/tcr_native/ingest_repo.py) 仍读取其中 PISTE/HLA 资料；不能按目录名视为全是已废弃 PPI 数据。
- [data/nanobody_raw](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/nanobody_raw) 中 NbBench 数据仍被 [prepare_immune_v7.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/prepare_immune_v7.py) 用于审计；旧任务退出当前评测范围，不代表其数据不再用于保护集检查。
- [qwen3_vl_arch](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch) 仍含融合模型使用的兼容代码；[refactor_baseline](/vepfs-mlp2/c20250601/251105016/project/dllm_test/refactor_baseline)（约 3.9 GiB）及 [docs/archive](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive) 保留历史证据，项目指南要求可读。
- [data/prepared](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared) 的新版本存在对父目录的链接依赖；[研究 release](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_releases) 是固定源码证据，不能作为普通重复副本清理。
- [旧 8B BERT 输出](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_esmc_llada8b_bert_immune)、[旧 8B diffusion 输出](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_esmc_llada8b_diffusion_immune) 各约 125 GiB，是后续存储审计重点，但本轮没有核实 checkpoint 的评测引用、硬链接关系、保留策略和远端任务，**未列为可删除文件**。[评测独立快照](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints) 同样保留。
- [lm-evaluation-harness](/vepfs-mlp2/c20250601/251105016/project/dllm_test/lm-evaluation-harness) 虽然当前为空，仍是 [.gitmodules](/vepfs-mlp2/c20250601/251105016/project/dllm_test/.gitmodules) 登记的 Git 子模块，不能当作无用空目录。

## 初次检查的建议与边界（历史快照）

先处理可重建缓存、编译中间文件、失效 PID 便签；再归档根目录旧导出及已失效旧入口；临时分析产物在保留来源/结论后整理。大体积数据和 checkpoint 单独核查依赖与保留用途。

项目占用粗查约 1.4 TiB，主要是 output（约 738 GiB）、data（约 464 GiB）和 downstream（约 94 GiB）；上述小文件候选合计约 197.8 MiB，主要价值是减少目录杂乱，不构成大规模空间回收方案。`du` 会处理硬链接，不能把多次目录统计直接相加当成可回收量。

本轮是有界静态检查，不是全仓库不可达性证明；没有查询所有云端任务，也没有执行所有入口。未发现引用的文件仅作为候选，不能推断无人使用。
