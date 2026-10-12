# Historical job YAML archive — 2026-09-30

本次将旧训练和评测 YAML 移出日常作业目录，完整保留原始字节、文件权限、修改时间和 SHA256。

下表统计本轮清点的 274 份文件；清点后其他任务新增的配置不纳入本次整理。

| 原目录 | 归档 YAML | 留在原位的 YAML |
|---|---:|---:|
| train_jobs | 29 | 55（含当前实验子目录） |
| eval_jobs | 110 | 80 |
| 合计 | 139 | 135 |

## 清单与筛选依据

- [manifest.tsv](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive/job_yamls_20260930/manifest.tsv)：归档文件的原绝对路径、现绝对路径、SHA256、大小、修改时间和原 Git 跟踪状态。
- [manifest.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive/job_yamls_20260930/manifest.json)：全部 274 份 YAML 的去向、保留理由、源代码/文档引用及平台查询证据摘要。`current_path` 为归档完成时的实际位置。
- [训练配置](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive/job_yamls_20260930/train_jobs) 与 [评测配置](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive/job_yamls_20260930/eval_jobs)：原内容归档。
- [本轮核验记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/job_yaml_cleanup_20260930/validation.json)：文件守恒、内容哈希、模板和源码发布检查。

候选仅来自两个作业目录的顶层，最后修改早于 2026-09-20 UTC，属于旧实验配置。日期只是初筛：源码/看护配置引用、动态配对扫描输入、已跟踪但存在未提交修改的文件、近期配置和全部当前实验子目录均保留。对候选 TaskName 所属前缀查询非终态平台任务，结果无匹配；记录的是当时可见状态，不是未来状态保证。此次没有提交、取消或重跑任何任务。

生成器模板、看护输入与六份配对扫描输入的具体用途见 [训练索引](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/README.md) 和 [评测索引](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/README.md)。原数据、权重、日志及评分产物不属于本次清理范围。

## 查询与恢复

在清单中按完整文件名查询，例如：

```bash
rg -F 'protein_esmc_llada270m_diffusion_immune_v5_8gpu.yml' /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive/job_yamls_20260930/manifest.tsv
```

需要恢复单文件时，先用 `sha256sum` 核对清单，再将对应 `archive_path` 复制到 `original_path`，使用 `cp -p -n` 保留属性且避免覆盖现有文件。恢复不会自动启动任务；历史配置中的数据、环境与 checkpoint 路径需要按当前用途复核。

历史 Markdown 文件中的 YAML 链接已指向归档；旧命令、TaskName 和原路径文字保留作当时记录，不能据此判断当前配置位置。

Git 处理沿用原状态：两份此前受跟踪的训练 YAML 在归档路径放行；其余 137 份原本被忽略的本地配置继续被忽略。本索引、两份清单和目录 README 可纳入版本控制。本地完整归档应随实验工作区备份，单独克隆源码不会获得被忽略的 137 份 YAML。
