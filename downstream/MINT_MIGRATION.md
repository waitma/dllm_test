# MINT downstream 任务迁移计划

> 目的：把 MINT（`github.com/VarunUllanat/mint`，*Learning the language of protein-protein interactions*）官方仓库 `downstream/` 里的评测任务，迁移到本仓库 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/` 下，用于评测我们的 grammar / Ophiuchus / no-encoder 模型。
>
> **本文件只做"分析 + 具体改动清单"，先不落地大规模代码改动**，方便先审阅、再按清单逐项落地与修复。
>
> 关键约束（用户强调）：**迁移过来的代码依赖不能有问题**（不能留 `import byprot` / `import mint` 这种本仓库装不上的坏依赖）。

---

## 0. 一句话结论

- MINT `downstream/` 一共 5 大类任务（GeneralPPI / Antibody / TCR-Epitope / CovidVariants / oncoPPI）。
- **推荐优先迁移 `GeneralPPI`（6 个 PPI 子任务）+ `TCR-Epitope`**，因为我们的 grammar 模型训练数据里就有 PPI（STRING）和 TCR，最对口。
- Antibody（flab / in_silico）**本仓库已有一份从 AirGen-Dev 拷来的副本**，但它们 `import byprot`，在 `protenix_abtcr` / `pllm` 两个 env 里 **都 import 不了**（已实测 `ModuleNotFoundError: No module named 'byprot'`）→ 属于"坏依赖"，需要按本清单修。
- **好消息**：MINT 的核心模型（`ESM2` 多链版 + `Alphabet`）**已经 vendored 在本仓库** `dllm/pipelines/bioseq/ophiuchus/mint/`，所以依赖修复主要是"改 import 路径"，不需要新装 `mint` 包。

---

## 1. MINT 官方 `downstream/` 完整清单

```
downstream/
├── GeneralPPI/                      ★ 核心 PPI 评测（README 主推）
│   ├── README.md
│   ├── tasks.py                     # 数据集类 + get_task_datasets(task)
│   ├── embeddings_mint.py           # 用 MINT ESM2 抽 embedding 并缓存为 .pt
│   ├── embeddings_baselines.py      # ESM2/ProtT5/ProGen baseline 抽 embedding
│   ├── baselines.py                 # baseline 模型封装
│   ├── finetune_general.py          # 在缓存 embedding 上训练下游头(MLP/Ridge/Logistic)+出指标
│   ├── ppi/                         # Bernett gold-standard 二分类 PPI（含官方 MLP ckpt）
│   ├── human-ppi/                   # HumanPPI 二分类
│   ├── yeast-ppi/                   # YeastPPI 二分类
│   ├── pdb-bind/                    # 复合物结合亲和力回归（多链）
│   ├── mutational-ppi/              # 突变后是否相互作用
│   └── SKEMPI_v2/                   # 突变引起的结合亲和力差异回归(ΔΔG)
├── Antibody/
│   ├── flab/                        # FLAb 抗体性质回归(Kd/expression)  ← 本仓库已有副本
│   │   ├── README.md
│   │   └── finetune_flab.py
│   └── in_silico/                   # Desautels in-silico (FoldX/Rosetta DDG) ← 本仓库已有副本
│       ├── README.md
│       ├── finetune_in_silico.py
│       └── rcsb_pdb_2G75.fasta
├── TCR-Epitope/                     ★ 和我们 TCR 数据对口
│   ├── tcr-interface/               # TEIM 式：序列级 + 残基级 (train_seq.py / train_res.py / teim_utils.py)
│   ├── tcr-piste/                   # PISTE 式 (train_mhc.py)
│   └── tdc/                         # TChard (train_tchard.py + prepare_data.ipynb)
├── CovidVariants/                   # 新冠变异结合 (train.py + 两个 ipynb)
└── oncoPPI/                         # 癌症相关 PPI (train.py + prepare_data.ipynb)
```

各子任务的 `prepare_data.ipynb` 是"下数据 + 预处理成 csv"的脚本，**迁移时一般不跑 notebook**，而是直接准备好它产出的 csv（见第 6 节数据需求）。

---

## 2. MINT downstream 的运行范式（迁移要保住的结构）

GeneralPPI 是两段式（其余任务大同小异）：

1. **抽 embedding**：`embeddings_mint.py`
   - 用 `ESM2(use_multimer=True)` 加载 MINT ckpt；
   - 每个任务用对应 collator（`PPICollateFn` / `MutationalPPICollateFn` / `PDBBindCollateFn`）把多条链拼成 `chains` + `chain_ids`；
   - 取 `repr_layers=[33]` 的表征，按 chain_id 做 **mean-pool**（`sep_chains` 决定是否两条链分别 pool 再 concat）；
   - 把 train/val/test 的 embedding 存成 `embeddings/<task>/<model>/{train,val,test}.pt`。
2. **训下游头 + 出指标**：`finetune_general.py`
   - 读缓存 embedding；
   - 按 `method`：`mlp`（`SimpleMLP` 训练）/ `cv`（10-fold Ridge/Logistic GridSearch）/ `pcv`（预定义 split 的 CV）；
   - 指标：分类 = Accuracy/AUPRC/F1/AUROC，回归 = pearson/spearman/rmse。

> 迁移策略核心：**"任务/数据/下游头/指标"是模型无关的，可直接搬（依赖干净）**；**"抽 embedding 的模型封装"是 MINT 专有的，要替换成我们模型的 adapter**。

---

## 3. 推荐迁移范围（按优先级）

| 优先级 | 任务 | 理由 | 迁移难度 |
|--------|------|------|----------|
| P0 | GeneralPPI: `Bernett`/`HumanPPI`/`YeastPPI`（二分类 PPI） | 我们训了 STRING PPI，最对口；只需要序列对 csv | 低（纯序列对 + mean-pool embedding） |
| P0 | GeneralPPI: `SKEMPI_v2`、`pdb-bind`（亲和力回归） | PPI 定量能力评测 | 中（SKEMPI 是 wt/mut 双前向；pdb-bind 多链） |
| P1 | GeneralPPI: `mutational-ppi` | 突变 PPI | 中 |
| P1 | TCR-Epitope: `tdc/train_tchard.py` | 我们有 TCR 数据；TChard 是纯序列对分类，最易接 | 中（依赖 `wandb`，`mint`→ vendored） |
| P2 | TCR-Epitope: `tcr-interface`（TEIM 残基级） | 更细粒度，但依赖 `pytorch_lightning` + `teim_utils` + 残基级标签 | 高 |
| P2 | Antibody: `flab` / `in_silico` | 本仓库已有副本，但 `import byprot` 坏；修依赖即可复活 | 低-中（改 import） |
| P3 | CovidVariants / oncoPPI | 偏专题、数据准备重 | 高 |

**建议第一步只落地 P0（GeneralPPI 的 PPI + 亲和力）**，跑通后再扩 TCR。

---

## 4. 依赖问题清单（最关键）

实测结论：
- `import byprot` → ❌ 两个 conda env (`protenix_abtcr` / `pllm`) 都 `ModuleNotFoundError`。
- `import mint`（MINT 官方包名）→ ❌ 本仓库没装。
- ✅ 但 vendored MINT 在：`dllm/pipelines/bioseq/ophiuchus/mint/`
  - `from dllm.pipelines.bioseq.ophiuchus.mint.model.esm2 import ESM2`
  - `from dllm.pipelines.bioseq.ophiuchus.mint.data import Alphabet`（有 `Alphabet.from_architecture("ESM-1b")`、`encode`、`padding_idx/cls_idx/eos_idx`）
- ✅ 已有 embedding 封装可直接复用：`downstream/embeddings.py::OphiuchusEmbeddingModel`（内部就是 MINT `ESM2(use_multimer=True)` + `repr_layers=[33]` + mean-pool，`sep_chains` 可选），以及 `downstream/benchmark/common/model_api.py` 里的 `SequenceEmbedder`/`ESM2Embedder`/`OphiuchusEmbedder`/`BioSeqEmbedder` 抽象。
- ✅ 我们的 grammar 模型 forward 输出含 `hidden_states [B,S,H]`（见 `dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py`），可做 grammar 版 embedding 抽取。

### 必须替换/处理的"坏依赖"对照表

| MINT 原始 import | 问题 | 迁移后改成 |
|------------------|------|-----------|
| `import mint` / `from mint.model.esm2 import ESM2` | 无 `mint` 包 | `from dllm.pipelines.bioseq.ophiuchus.mint.model.esm2 import ESM2` |
| `mint.data.Alphabet.from_architecture("ESM-1b")` | 同上 | `from dllm.pipelines.bioseq.ophiuchus.mint.data import Alphabet` |
| `from ..data import Alphabet` / `from ..model.esm2 import ESM2`（helpers/extract.py 相对导入） | 相对路径失效 | 同上，改绝对导入 |
| `from byprot.models.lm.modules.mint.model.esm2 import ESM2`（本仓库 legacy flab/in_silico/dev/specificity） | `byprot` 装不上 | 同上，改 vendored mint |
| `from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint`（flab_align） | `deepspeed` 不一定可用，且我们 ckpt 不是 zero 格式 | 删掉 zero-checkpoint 分支，只保留普通 `torch.load` |
| `import wandb`（finetune_general / train_tchard / in_silico） | 评测不该强依赖 wandb | 包一层：`--wandb` 才 import；默认把 `wandb.log` 替换成 print/JSON 落盘 |
| `import pytorch_lightning as pl`（tcr-interface/train_seq.py） | 引入重训练框架依赖 | P2 再处理；先不迁 tcr-interface |
| `from teim_utils import *`（tcr-interface） | 同目录工具，需一起迁 | 一并迁 `teim_utils.py` 并改其内部 import |
| `from tasks import ...` / `from baselines import ...`（同目录相对裸 import） | 迁到包内后路径变 | 改成本包内显式相对/绝对导入 |
| `CONFIG_DICT_PATH = "../../data/esm2_t33_650M_UR50D.json"`（embeddings_mint） | 相对路径 + 文件可能缺 | 改成绝对路径常量，并确认该 cfg json 存在（见第 6 节） |
| 硬编码 `repr_layers=[33]` / `total_layers=33` / `1280` 维度 | 仅对 650M 成立 | 改成从 cfg 读 `encoder_layers` / `encoder_embed_dim` |
| 硬编码 ESM2 tokenizer 路径 `/vepfs-mlp2/mlp-public/zhuyiheng/hub/checkpoints/esm2_t33_650M_UR50D`（flab/in_silico） | 路径耦合 | 抽成参数/常量，确认存在 |

---

## 5. 目标目录结构（建议）

新增一个 **模型无关 + adapter 可插拔** 的子包，不污染已有 `downstream/grammar`、`downstream/comp_chain`：

```
downstream/mint_tasks/
├── README.md                 # 怎么跑（数据路径、命令）
├── __init__.py
├── tasks.py                  # 从 MINT tasks.py 迁；改数据路径为绝对路径常量
├── collators.py              # PPICollateFn / MutationalPPICollateFn / PDBBindCollateFn（改 vendored Alphabet）
├── embedders.py              # ★ adapter 层：把"链→embedding"统一成一个接口
│                             #   - MintEsm2Embedder（vendored ESM2，复用 OphiuchusEmbeddingModel）
│                             #   - GrammarEmbedder（我们 grammar 模型，用 hidden_states mean-pool）
│                             #   - Esm2HFEmbedder（HF ESM2 baseline）
├── extract_embeddings.py     # 从 embeddings_mint.py 迁；调用 embedders，缓存 .pt
├── finetune_general.py       # 从 MINT 迁；去 wandb 硬依赖，指标落 JSON
└── metrics.py                # 分类/回归指标（从 finetune_general.py 拆出来）
```

> 与现有 `downstream/benchmark/`（IRBench，自研 harness）保持独立：benchmark 是我们自己的 TCR/PPI 表征基准，`mint_tasks` 是对齐 MINT 论文口径的复现，两者互不依赖。

---

## 6. 逐文件具体改动清单

### 6.1 `downstream/mint_tasks/tasks.py`（来自 `GeneralPPI/tasks.py`）
- 改动：
  - 所有相对数据路径 `"./ppi/Intra1_seqs.csv"` 等 → 绝对常量
    `DATA_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/mint")`，
    再 `DATA_ROOT / "ppi" / "Intra1_seqs.csv"`。
  - 其余（`CSVDataset` / `MutationalCSVDataset` / `MultiCSVDataset` / `get_task_datasets`）逻辑保留。
- 依赖：仅 `pandas` / `torch` ✅ 干净。

### 6.2 `downstream/mint_tasks/collators.py`（来自 `embeddings_mint.py` 的 collator 部分）
- 改动：
  - `mint.data.Alphabet.from_architecture("ESM-1b")` → `from dllm.pipelines.bioseq.ophiuchus.mint.data import Alphabet`。
  - 保留 `PPICollateFn` / `MutationalPPICollateFn` / `PDBBindCollateFn` / `get_sequences_by_chain`。
- 依赖：`torch` + vendored mint ✅。

### 6.3 `downstream/mint_tasks/embedders.py`（替换 MINT 的 `ESMMultimerWrapper`）★ 核心
- 提供统一接口：
  ```python
  class ChainEmbedder:
      dim: int
      def embed(self, chains, chain_ids) -> torch.Tensor: ...  # [B, dim]
  ```
- `MintEsm2Embedder`：直接复用 `downstream/embeddings.py::OphiuchusEmbeddingModel`
  （已是 `ESM2(use_multimer=True)`+`repr_layers=[33]`+mean-pool，`sep_chains` 可选），
  避免重写 `ESMMultimerWrapper`。
- `GrammarEmbedder`：用 `downstream/grammar/common.py::load_grammar_checkpoint` 载入我们模型，
  把 PPI 链对渲染成 grammar token 流（`ppi_pair`），forward 取 `hidden_states`，
  按 `residue_mask` mean-pool（参考 `downstream/grammar/metrics.py::extract_chain_sequence` 的定位方式）。
  → 这是让 **我们的模型**跑 MINT PPI 任务的关键 adapter。
- `Esm2HFEmbedder`：可直接复用 `downstream/benchmark/common/model_api.py::ESM2Embedder` 作 baseline。
- 依赖：torch + 本仓库内部模块 ✅。

### 6.4 `downstream/mint_tasks/extract_embeddings.py`（来自 `embeddings_mint.py` 的 main）
- 改动：
  - 删 `import mint` / `from mint.model.esm2 import ESM2`；改用 `embedders.py`。
  - `CONFIG_DICT_PATH` → 绝对路径；优先用 `OphiuchusEmbeddingConfig` 默认值（33 层/1280），不强依赖 cfg json。
  - `repr_layers=[33]`、`total_layers=33` → 从 config 读。
  - `save_dir` → 绝对路径 `output/downstream_generation/mint_tasks/<task>/<model>/`。
- 依赖：torch + 本包 ✅。

### 6.5 `downstream/mint_tasks/finetune_general.py`（来自 `GeneralPPI/finetune_general.py`）
- 改动：
  - **`wandb` 去硬依赖**：默认不 import；`wandb.log(...)` → `_log(metrics)`（print + 追加写 JSON 到 `..._metrics.json`）。仅 `--wandb` 时才 import & log。
  - 修 **已存在的 bug**：`pre_defined_cv` 里 `Y_pred = best_model.predict_proba(X_test)` 用了未定义的 `X_test`（应为 `test_embeddings`）。迁移时一并修。
  - `from tasks import get_task_datasets` → 本包导入。
  - `model_list` 里的 baseline 名保留，但 embedding 读取路径改绝对。
- 依赖：`numpy` / `scipy` / `sklearn` / `torch`（均在 env 内）✅。

### 6.6 `downstream/mint_tasks/metrics.py`
- 把 `regression_metrics` / `classification_metrics` / `multilabel_metrics` / `calculate_mean_std` 拆出来，供 finetune 与未来任务复用。
- 依赖：`numpy` / `scipy` / `sklearn` ✅。

### 6.7 TCR-Epitope（P1：先迁 `tdc/train_tchard.py`）
- 改动：
  - `import mint` / `from mint.model.esm2 import ESM2` → vendored。
  - `import wandb` → 可选化。
  - 数据路径（TChard csv）→ 绝对常量。
  - 同样可改成"抽 embedding + 下游头"两段式，复用 `embedders.py`。
- `tcr-interface`（train_seq/res + teim_utils）= **P2**，因为引入 `pytorch_lightning` + 残基级标签，单独评估再迁。

### 6.8 已有 Antibody legacy 修复（P2，可选）
- 现有 `downstream/flab/finetune_flab_align.py`、`downstream/in_silico/finetune_in_silico.py`、
  `downstream/dev/finetune_dev_pplm.py`、`downstream/specificity/HD_Flu_Cov-paired.py`、
  `downstream/flab/finetune_flab_esm_ppi.py`、`downstream/infill/zeroshot_SAb23H2.py`：
  - 把 `from byprot.models.lm.modules.mint.model.esm2 import ESM2` → vendored mint。
  - flab_align 删 `deepspeed` zero-checkpoint 分支。
  - `wandb` 可选化。
- 注意：`downstream/flab/finetune_flab.py`（无 `_align` 后缀那个）已经是对齐我们 Ophiuchus 的干净版（用 `downstream.common` + `downstream.embeddings`），**可作为"如何接我们模型"的范例**。

---

## 7. 数据需求（迁移后才能真正跑）

放在 `data/downstream/mint/` 下（绝对路径），按 MINT `tasks.py` 期望的列名准备：

| 任务 | 文件 | 关键列 |
|------|------|--------|
| Bernett | `ppi/Intra{0,1,2}_seqs.csv` | `seq1,seq2,labels` |
| HumanPPI | `human-ppi/processed_data_{train,validation,test}.csv` | `sequence_1,sequence_2,target` |
| YeastPPI | `yeast-ppi/processed_data_{train,validation,test}.csv` | 同上 |
| SKEMPI | `SKEMPI_v2/processed_data.csv` | `seq1,seq2,seq1_mut,seq2_mut,target` + `split*` 列 |
| Pdb-bind | `pdb-bind/processed_data.csv` | `seq,chain_ids,target` |
| MutationalPPI | `mutational-ppi/processed_data.csv` | `seq1,seq2,target` |

> 这些 csv 由 MINT 各子目录的 `prepare_data.ipynb` 生成（多数来自公开数据：Bernett et al. gold-standard、SKEMPI v2、PDBBind 等）。迁移代码与准备数据可解耦：先把代码依赖修干净，数据可后续逐个补。
>
> baseline 还需 `data/esm2_t33_650M_UR50D.json`（MINT cfg）——本仓库可用 `OphiuchusEmbeddingConfig` 默认值替代，不强依赖该 json。

---

## 8. 落地后验证 checklist（保证"依赖没问题"）

1. **import 冒烟**（不碰 GPU/数据）：
   ```bash
   conda run -n protenix_abtcr python -c "import downstream.mint_tasks.tasks, downstream.mint_tasks.collators, downstream.mint_tasks.embedders, downstream.mint_tasks.finetune_general, downstream.mint_tasks.metrics; print('import OK')"
   ```
2. **`py_compile`** 所有新文件无语法错误。
3. **vendored mint 可用**：
   ```bash
   conda run -n protenix_abtcr python -c "from dllm.pipelines.bioseq.ophiuchus.mint.model.esm2 import ESM2; from dllm.pipelines.bioseq.ophiuchus.mint.data import Alphabet; print(Alphabet.from_architecture('ESM-1b').padding_idx)"
   ```
4. **无残留坏依赖**：全包 grep 不应再出现 `import byprot` / `^import mint` / `from mint.` / `import deepspeed`（wandb 仅在 `--wandb` 分支内）。
5. **小样本端到端**：用 `--test_run`（MINT 自带，`df.sample(n=20)`）跑 1 个任务的 extract→finetune，确认产出 metrics JSON。

---

## 9. 风险与注意

- **分词/词表对齐**：MINT 用 ESM-1b alphabet（vendored `Alphabet`）；grammar 模型用的是 `Esm2SequenceTokenizer` + 自己的 grammar token。两条 embedder 路径词表不同，**不能混用缓存的 .pt**，目录要按 `<model>` 分开。
- **链数 > 2**：pdb-bind 多链已由 `PDBBindCollateFn` 处理；grammar 渲染目前对 PPI 主要是 2 链（`ppi_pair`），>2 链需扩 adapter（参考 MINT issue#7 的 N 链处理）。
- **回归标签变换**：MINT 对回归用 `PowerTransformer`/`StandardScaler`（在 train 上 fit）——复现指标时必须照搬，否则数值对不上。
- **`sep_chains`**：MINT 论文推荐 `sep_chains=True`（两链分别 pool 再 concat，维度×2）。复现时与论文设置保持一致。
- **指标口径**：分类主看 AUPRC（Bernett）/Accuracy（Human/Yeast），回归看 pearson/spearman/rmse——迁移后与 MINT 论文表对齐再下结论。

---

## 附：本计划依据的源文件

- MINT 仓库：`https://github.com/VarunUllanat/mint`（`downstream/` 子树，main 分支）
- 本仓库 vendored MINT：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/ophiuchus/mint/`
- 现成 embedding 封装：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/embeddings.py`、`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/common/model_api.py`
- grammar 模型 embedding 来源：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py`（`BioSeqDiffusionOutput.hidden_states`）
- 已有干净范例：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/flab/finetune_flab.py`
