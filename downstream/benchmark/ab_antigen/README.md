# Official antibody–antigen benchmark audit

**2026-09-24 当前决定：FLAb2 已排除；MAGE 保留。** FLAb2 已从 active source lock 和方法清单移除，不再处理其缺失代码或依赖。范围以 [README §0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/README.md) 为准。

Scope: conventional paired VH/VL only; no VHH evaluation. `official.py` verifies upstream provenance and reviewed CLI probes. The `sequence_*` modules and unified interfaces implement our derived sequence benchmark and Ours evaluation; they do not recreate external author baseline models.

The [task protocol](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md) records what each paper actually compares. The [method inventory](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/ab_antigen/comparisons.tsv) maps each method/entrypoint to its comparison protocol and unresolved requirements. Inventory entries are research candidates, not completed experiments.

- [sources.lock.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/ab_antigen/sources.lock.json): official repository URL, immutable commit, release distinction, source file SHA256, blockers, reviewed CLI probes. Current default-branch commits are not automatically paper-time versions.
- [official.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/ab_antigen/official.py): checks origin, HEAD, tracked modifications, added executable modules and pinned hashes before/after probes. AST and missing-file findings are separate from provenance verification.
- [tests](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/test_ab_antigen_official.py): rejects wrong commit/origin/hash, edited/missing files and untracked module shadowing; preserves upstream failures and distinguishes sparse checkout from absent upstream data.

```bash
source /vepfs-mlp2/c20250601/251105016/miniforge3/etc/profile.d/conda.sh
conda activate pllm
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
python -m downstream.benchmark.ab_antigen.official audit --output /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_antigen_new_audit
```

Outputs must use a new directory. Exit codes: 1 provenance failure; 2 CLI failure; 3 upstream static defects. `source_verified` means identical source, not working software; CLI success means help only, not inference. Full evaluation needs the benchmark authors' adapters, exact data/splits, weights, training and scoring code. Missing pieces stay blocked; downloading the underlying base model does not replace a benchmark-specific implementation.

GPU inference is submitted via Volc after the task gates pass. MAGE weights have been acquired and hashed; the official WT-RBD n=1 smoke job is queued; MochiBind's mixed VHH defaults must never run in this scope. Existing runtime evidence lives under [/output/downstream_generation/ab_antigen_official_20260924](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_antigen_official_20260924). Third-party code, environments, datasets and weights remain ignored by Git.

## BioMap A2binder official run

User authorized an initial BioMap run on 2026-09-24. [Acquisition wrapper](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/ab_antigen/acquire_palm.py) fetches author RoFormer initialization assets and pinned ESM2. [Run wrapper](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/ab_antigen/run_biomap.py) executes the unchanged author entry and calls original metric functions. [Volc config](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/ab_antigen/biomap_seed3.yml) uses the released seed=3 configuration, with filesystem mappings only.

The isolated Python 3.9 environment matches the published primary package versions (torch 1.12.1 CUDA 11.6, Transformers 4.24.0, NumPy 1.22.4, pandas 1.4.3, SciPy 1.7.3, sklearn 1.2.2). Dependency lock, CPU-only preflight, original-loader split membership, path mappings and runtime evidence are stored under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_antigen_official_20260924/biomap_seed3`. GPU execution must go through Volc; the run wrapper rejects CPU execution and overwriting an existing run. Status and scores are owned by [RESULTS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#biomap-official-run).

## 统一评测接口（2026-09-27）

接口将数据校验、输入导出、模型调用和离线评分分开。适用 `binding`、`spr`、`abbibench`、`chimera` 四个子集；属于项目派生协议。外部 baseline 仍调用锁定的作者发布代码，并记录输入协议差异。

- [interfaces.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/ab_antigen/interfaces.py)：CPU 校验、仅模型输入导出、按 ID 评分、Python adapter 协议、断点日志。
- [native_adapter.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/ab_antigen/native_adapter.py)：已指定 no-ESMC step150000 的 zero-shot 适配器；实际推理仅经 Volc GPU 执行。冻结探针的训练、验证选参和测试流程保留在 `sequence_eval.py`，不由此适配器重新训练。
- [接口测试](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/test_ab_antigen_interfaces.py)：覆盖 ID 对齐、缺失/重复/非有限预测、缓存版本一致性、标签隔离、生成参考隔离、WT 差分及 pKD 门控。

从仓库根目录运行，先激活 pllm。以下为 CPU 命令；`PYTHON` 指向包含现有评测依赖的环境：

```bash
source /vepfs-mlp2/c20250601/251105016/miniforge3/etc/profile.d/conda.sh
conda activate pllm
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
PYTHON=/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python
export USE_TF=0
$PYTHON -m downstream.benchmark.ab_antigen.interfaces validate --task binding --data data/downstream/ab_antigen_sequence_v1/binding.jsonl
$PYTHON -m downstream.benchmark.ab_antigen.interfaces export --task binding --data data/downstream/ab_antigen_sequence_v1/binding.jsonl --out-dir output/ab_ag_interface_inputs
$PYTHON -m downstream.benchmark.ab_antigen.interfaces evaluate --task binding --data data/downstream/ab_antigen_sequence_v1/binding.jsonl --predictions output/my_model/predictions.jsonl --out-dir output/my_model/scoring
```

最后一条需要实际模型预测文件。预测 JSONL 最小格式为 `{"id":"样本ID","score":0.5}`，分数方向统一为越高越强；zero-shot 分数无需为概率，binding frozen-probe 使用概率。AbBiBench 的 `score` 为变体条件分数减 WT 条件分数，Python `run_adapter` 自动计算该差值。SPR 排序按来源和实验分组；仅 `pkd_status=verified`、具有定量标签且提供 `pkd_prediction` 的 frozen-probe 记录可进入 pKD MAE/RMSE。

Python 预测适配器实现 `score_pair(PairInput)`，仅接收 ID、Ag、VH、VL；生成适配器实现 `generate(GenerationInput)`，仅接收 Ag 及请求元数据，不接收参考抗体、长度或接触信息。`run_adapter(..., model_identity=...)` 持久化数据、代码、模型身份和采样配置；相同配置按 ID 续跑，配置变更或截断 JSONL 报错并保留已有输出。模型身份必须由调用方提供真实权重/代码版本信息。

CHIMERA 默认取 test，按抗原生成 K20 个候选，正确/替换抗原共享 seed。返回字段为 `heavy/light`、`heavy_eos/light_eos`、`token_valid`，失败必须附 `failure_reason`，全部尝试进入分母。离线 `evaluate` 先检查请求匹配并统计 token 有效率；加 `--reference-metrics` 调用现有 ANARCI 参考恢复、多样性和新颖性评分，需要其独立环境和训练参考库。参考文件只在评分阶段读取；token 有效率不是编号有效率，也不是结合成功率。

缺失预测默认拒绝；显式 `--allow-partial` 才输出带缺失 ID 的 partial 报告。输出不能覆盖不同内容。接口校验和合成测试不代表模型已完成评测，也不替代外部 baseline 的官方协议复现。
