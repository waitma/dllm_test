> ⚠️ **ARCHIVE — 非现行入口，仅历史/复现用。**
> T1 的现行入口是 [`../tasks/TCR_T1_BINDING.md`](../tasks/TCR_T1_BINDING.md)；口径见 [`PROJGUIDE.md`](PROJGUIDE.md)，数字见 [`RESULTS.md`](RESULTS.md) §0.1。
> ⚠️ 本文成文于「`[R]` 本地重跑优先」的旧规约下；该规约已于 2026-09-02 **反转为论文值优先**（见 `PROJGUIDE.md` §0.2.2），
> 故本文关于「把官方权重复现进 T1 主表」的目标表述需按新规约理解：复现产出的是 `[R]` **附行**，主榜行取论文值。
> **不要**给本文填新数字。归档于 2026-09-02，内容保留未改动。

# T1 Binding —— NM2025 官方 baseline 扩充执行计划（plan-first）

> 目标：把 Nature Methods 2025 官方 benchmark（`SuoLab-GZLab/TCREpitopeBenchmark`）中**已下载官方权重 + 有官方代码 + 有 env yml** 的其余模型，全部按现有 6 个 baseline 的同一范式复现到我们的 seen/unseen 测试集上，进入 T1 主表对照。
>
> **硬性原则（不可违反）**
> - 只用**官方源码 + 官方权重**（`baselines/TCREpitopeBenchmark/Original_model/` + 各模型目录 + `environment/*.yml`）。**不复现模型数学**，wrapper 只做「读统一输入 → 调官方推理 → 写统一预测」。
> - 权重**已全部在本地** `Original_model/`，**无需下载**。
> - **禁止编造任何数字**。env 建不起来 / 权重载不了 / 推理报错 → 不 retrain、不用近似/随机顶替；主表回退论文 original-model 数值并标 **“论文值，未本地复现”**，同时保留精确 blocker。
> - 框架兼容 shim 仅允许对**已移除的框架 API**（torch `weights_only`、旧 PL 装饰器、CUDA→CPU map_location 等），**不改任何模型逻辑**。
> - 绝对路径。GPU 重活可本地跑（binding 推理都很轻，单卡秒级~分钟级）。

## 0. 现状（2026-07-21）

- **canonical original-model 集合已闭环**：15 个论文条目中，ATM-TCR / TEIM / TEINet-large / TEINet-small / NetTCR / epiTCR / ERGO 4 变体 / TPBTE 2 变体 / PanPep 共 13 个使用完整官方 checkpoint+code+`original.zip` 本地结果；SETE / TEPCAM 使用明确标注的论文 fallback。选择表：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025/summary_original_baselines.csv`。
- **数据就绪**：`data/tcr_binding_nm2025/{train,seen,unseen}/{cdr3b,others}/`。cdr3b track = CDR3β+peptide（seen/unseen × AS/PS/HS）；others track = 全长 α/β + V/J + MHC（seen/unseen × 仅 AS）。
- **external test 准备**：使用 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/scripts/prepare_nm2025_binding.py --tests-only`，只展开 `original.zip` 并生成逐 test SHA256 manifest；不得为了 external baseline 调用默认的 train+test 准备路径。
- **接入方式**：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_official_baseline.py --name <wrapper> --track cdr3b`；wrapper 提供 `predict_official(df)->DataFrame[group,label,score]` + `is_available()` + `official_checkpoint_paths()` + `NAME`；driver 要求 runtime artifact 路径与 catalog 完全一致，再统一计算 overall/macro AUPRC、AUROC、AUC0.1。该入口只加载 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ntmethod_binding/original.zip` 派生 test，不加载 `train.csv`；多 checkpoint wrapper 必须传 canonical `--tag`，tag 在 import 前绑定实际权重。
- **模板**：`baselines/wrappers/{teim,nettcr,atmtcr}.py` + 对应 `*_official_infer.py`（子进程在该模型自己的 env 里跑官方代码，`LD_LIBRARY_PATH` 隔离到该 env 的 lib）。

## 1. 待复现模型清单（按 env 难度分层）

> 每个模型的官方「Original_model_prediction」逻辑在其目录的 `.ipynb`（Module 1）或同名 `.py` 里；照抄进 `*_official_infer.py`。env 从 `environment/<Model>.yml` 建（yml 为 aarch64，x86 需按包版本手动建/兼容；能复用已有 env 则复用）。

### Tier-1（现代 torch env，cdr3b 口径，最高优先——先做这批）✅ DONE
| wrapper 名 | 模型 | 权重（Original_model/） | 官方代码目录 | env yml | 说明 |
|---|---|---|---|---|---|
| `teinet` | TEINet-large / -small | `TEINet_large.pth` / `TEINet_small.pth` | `TEINet(TEINet-large_TEINet-small)/` (`model.py`,`TEINet.ipynb`) | `TEINet.yml`（torch 2.0.1, py3.8） | pMHC→CDR3β；两权重各出一行 |
| `ergo` | ERGO AE/lstm × vdj/mc（4 变体） | `ERGO_AE_vdj.pt` `ERGO_AE_mc.pt` `ERGO_lstm_vdj.pt` `ERGO_lstm_mc.pt` | `ERGO(...)/` (`ERGO.py`,`ERGO_models.py`) | `ERGO.yml`（torch 2.0.1） | CDR3β+epitope；4 变体各出一行 |
| `tpbte` | TPBTE mc / vdj | `TPBTE_mc.pth` / `TPBTE_vdj.pth` | `TPBTE(...)/tpbte.py` | `TPBTE.yml`（torch 2.0） | transformer CDR3β+epitope |
| `sete` | SETE | `SETE/`（目录权重） | `SETE/SETE.py` | `SETE.yml`（torch 2.0） | CDR3β+epitope |

### Tier-2（现代 torch/PL，稍多依赖）
| wrapper 名 | 模型 | 权重 | env yml |
|---|---|---|---|
| `titan` | TITAN | `TITAN/` | `TITAN.yml`（torch 2.5.1） |
| `vittcr` | VitTCR | `VitTCR.pt` | `TEPCAM_VitTCR.yml`（PL 2.4） |
| `tepcam` | TEPCAM | `TEPCAM.pt` | `TEPCAM_VitTCR.yml` |
| `attntap` | AttnTAP vdj/mc | `AttnTAP_vdj.pt` `AttnTAP_mc.pt` | `AttnTAP.yml`（torch 1.11） |
| `piste` | PISTE reftcr/random/unipep | `PISTE_*.pkl` | `PISTE.yml`（cuda12.4） |
| `tcrconv` | TCRconv small/large/AB | `TCRconv_*.pt` | `tcrconv.yml`（torch 2.4） |
| `tcrh` | TCR-H | (sklearn) | `TCR-H.yml`（sklearn, py3.13） |

### Tier-3（keras/TF 旧栈，较硬）
| wrapper 名 | 模型 | 权重 | env yml |
|---|---|---|---|
| `itcep` | iTCep | `iTCep.h5` | `iTCep.yml` |
| `mcmc` | MCMC | `MCMC.h5` | `mcmc.yml`（keras 2.9） |
| `tcrgp` | TCRGP | `TCRGP/` | `TCRGP.yml`（TF 2.11） |
| `imrex` | ImRex | `ImRex.h5` | `ImRex.yml`（keras, py3.7） |
| `nettcr_ab` | NetTCR-AB（others track） | `NetTCR_AB.h5` | `NetTCR.yml` |
| `deepair` | DeepAIR（others track） | `DeepAIR/` | `DeepAIR.yml`（TF 1.2, py3.5——最难，可最后/放弃） |

### 明确跳过（README 声明不可复现接入）
DLpTCR(CNN/RESNET/FULL)、PiTE(epiSplit/tcrSplit)、vibtcr/vibtcr_AB、pMTnet、pMTnet_omni、MixTCRpred、TCRfinder（github-only / 无 bundled 权重 / 无 license）。

## 2. 每个模型的执行协议（subagent 必须逐条完成）

1. **建 env**：`conda create -n tcrbench_<name> ...` 按 `environment/<Model>.yml` 的关键包版本（python/torch/tf/sklearn）。能复用已有 env（如同为 torch2.0.1 的 TEINet/ERGO/TPBTE/SETE 可共用一个 `tcrbench_torch20`）则复用并在 wrapper 注释说明。
2. **写 `baselines/wrappers/<name>_official_infer.py`**：`os.chdir` 到官方目录 + `sys.path.insert`，照抄该模型 `.ipynb`/`.py` 里的 **Original_model_prediction** 逻辑（载 `Original_model/<权重>` + 官方 featurizer + forward），读 `--input`（列 `Epitope,CDR3B,Affinity`，或 others track 加 `CDR3A,TRBV,...`），写 `--output` probability.csv（列 `Epitope,CDR3B,y_true,y_prob`）。
3. **写 `baselines/wrappers/<name>.py`**：`predict_official(df)`（schema 映射 → 子进程调 infer → 读回 `group=Epitope,label=y_true,score=y_prob`）+ `is_available()` + `official_checkpoint_paths()` + `NAME`。`official_checkpoint_paths()` 必须返回推理实际打开的全部学习型 artifact（含 auxiliary encoder/memory），不得只返回一个未实际读取的 bundle alias。多权重变体（TEINet/ERGO/AttnTAP/PISTE/TCRconv）用参数或多个 wrapper 名区分，每变体独立 tag。
4. **跑**：`python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_official_baseline.py --name <name> --track cdr3b --neg-source AS`（多变体加 `--tag <canonical-tag>`；Tier-3 全特征模型用 `--track others`）。至少产出 **seen/AS + unseen/AS**；PS/HS 有余力再跑。失败后运行 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/scripts/summarize_nm2025.py`，由统一选择器转用论文值并标注来源。
5. **自检**：`outputs/tcr_binding_nm2025/<tag>/cdr3b/{seen,unseen}/metrics.json` 存在；`auprc_overall` ∈ (0,1)；n_scored==n_input（无静默丢样本）；seen 通常 ≥ unseen（不强制，但异常需说明）。
6. **报告**：逐模型 **DONE（seen/unseen overall AUPRC 数字）** 或 **BLOCKER（精确失败环节：env solve / weight load / featurize / forward / schema）**。

## 3. 收尾（主代理做，非 subagent）

- `scripts/summarize_nm2025.py`：把新模型加入 `METHODS` 列表 → 重跑生成 `summary_main.csv` / `summary_neg_source.csv`。
- 文档：`downstream.md` T1 主表 + `RESULTS.md` T1 主表补新 baseline 行；`baselines/README.md` T1 表补 commit/env/状态；本文件状态勾选；`PROGRESS.md` changelog。
- 数字必须与 `outputs/` 一致。blocker 如实列出。

## 4. 执行记录（滚动更新）

- 2026-07-07 立计划。Tier-1 派 composer-2.5-fast 执行中（见 PROGRESS.md）。
- 2026-07-07 Tier-1 DONE（历史记录）：8 变体 cdr3b seen/unseen overall AUPRC 已跑出；其中“SETE 无输出”的当时状态已由下方 2026-07-21 精确失败记录取代。Tier-2 / Tier-3 仍属于可选扩展，不影响当前 15 行 canonical original-model 表的完整来源选择。
- 2026-07-21 original-only 协议落地：新增 15 行 canonical catalog 和 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025/summary_original_baselines.csv`；13 行已由新 runner 重跑 AS seen/unseen，全部产物内嵌实际 checkpoint/variant/original.zip checksum 与 `retrained=false`。SETE 与 TEPCAM 自动选择论文值并标“论文值，未本地复现”。修复 TEINet/ERGO/TPBTE 的 tag→checkpoint import 时序，以及 ERGO 官方 featurizer 原地修改 peptide 后污染输出标识的问题。SETE 的官方 pickle 在 seen/AS 仍报 `X has 433 features, but ... expecting 1`，保留失败状态 JSON，不改模型数学。
