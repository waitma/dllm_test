# 多链关系学习：现状、文献与对照实验

用户确认走**实验路径**，不只留调研。本文件是设计记录 + 落地说明。
**不要改正在跑的 2M all-chains**（`protein_esmc_llada270m_diffusion_allchains_immune_v3_8gpu_2m`）。
关系实验是**新 OUTPUT_DIR、from-scratch（或本目录自续）**的 generated-only diffusion 对照臂。

Headline **不是** `eval_loss`（跨目标 / all_chains / batch 不可比），而是：

1. 生成 pairing 的 **ImmunoMatch**（现有 `scripts/downstream/run_immune_fusion_pairing.sh`）
2. 模型原生 **PLL 差** `p(L|H) − p(L)`（`scripts/downstream/score_pairing_pll.py`）

作业 YAML 已写好，**未提交**。提交前确认磁盘配额（一次 save ≈ 9 GB 余量）。

---

## 1. 我们现在的模型本质

现役入口：`protein_pretrain_esmc.py` + `protein_fusion_model.py`。

- Encoder：ESMC-300M，按链编码 `[B, C, L]`，再 gather 到 decoder 残基位。
- Decoder：LLaDA（现役 270M from-scratch）。
- 条件：`residue_cond_mode=add`。encoder 对所有待预测残基打 `<mask>`，防止泄漏。
- 目标：`diffusion`（`t ~ U(eps,1)`，Bernoulli 吸收态 mask）或 `bert`。v3 两臂数据一致。

Grammar 把多链关系写成**拼接 + 特殊 token**：`<chainsep>`、`<binding>` / `<nonbinding>`、`<ab>` / `<tcr>` / `<pep>`。
**没有显式 cognate-pair 监督**；配对只能从联合重建里隐式出现。

六种布局的 loss 预算（残基 %）：`antibody_pair` 49.6 + `tcr_pair` 40.9 + 单链 2.9 ≈ **无条件生成 93%**；抗原条件抗体 4.6%；`tcr_pmhc`+`tcr_peptide` 合计 1.98%。

一句话：我们是「多链拼接的生成式 foundation model」，还不是「多链关系 foundation model」。关系是副产品。

---

## 2. 四条通路，以前只开了一条

| 通路 | 现状 | 这次 |
|---|---|---|
| 双向联合重建（共享 `t`，`joint=1.0`） | 现役唯一在干活的 | 对照臂降到 0.5 |
| Ophiuchus 分链噪声 | 代码有，YAML 全关；且 `int(B*ratio)` 在 per_device 2/4 上会把非 joint 桶 floor 成 0 | **改为 per-row multinomial**；对照 YAML 打开 0.5 / 0.15 / 0.15 / 0.1 / 0.1 |
| ESMC 条件 | 链内进化先验，跨链几乎全靠 decoder 注意力 | 不动 |
| 表征对比 / 配对分类 | 无 | 对照臂 B：`relation_aux=cognate` in-batch InfoNCE |

`joint_loss_ratio=1.0` 时 multinomial 仍 100% 抽 joint，**2M 若因重启重新 import 这份代码，行为与改前一致**。

---

## 3. 别人怎么学多链关系（2024–2026）

按关系类型，不按论文标题。

### A. 配对链（H–L / α–β）

| 模型 | 怎么学 | 对我们 |
|---|---|---|
| AbLang2 | `VH\|VL` 拼接 + MLM | 分隔符 + 位置先验 |
| IgBert / IgT5 | 海量 unpaired + 少量 paired | 我们 repertoire 行数高但 gen% 只有 2.9% |
| p-IgGen | 先 unpaired AR，再 paired finetune | 两阶段 curriculum |
| **Ophiuchus-Ab** | 扩散 + H/L 独立 t + 整链 mask；拆 intra/inter attention | **栈最像**。简单拼接会把链内/链间注意力缠在一起 |
| **ImmunoMatch** | AntiBERTa2 上 cognate vs random 分类 | 生成 pairing 贴地板时需要这条判别监督；也是我们的官方 pairing 指标 |
| LICHEN | 条件于 heavy 生成 light | 就是 `heavy2light` |
| **SCEPTR** | 六条 CDR + MLM + autocontrastive；随机丢掉整条 α 或 β | 链丢弃 = `single_chain`；对比是表征各向同性的正道 |
| TCRBinder | 分枝 RoFormer + 后期融合 | 关系不靠拼接注意力 |

Ophiuchus：**必须在 paired 粒度建模**；独立噪声让同一模型同时会 joint / p(H) / p(L\|H)。

### B. 受体–配体（Ab–Ag / TCR–pMHC）

DecoderTCR 的定义最干净：关系 = 条件似然差
`log p(pep | TCR, pMHC) − log p(pep | pMHC)`。
我们有 `<binding>` 和固定上下文，但从未用 PLL 差来学/评。本次把**抗体配对**版 `p(L|H) − p(L)` 做成评测，不改 pretrain 权重。

### C. 三种实现范式

文献里有效的关系学习几乎都在 concat 之外再加 **conditional noise** 或 **discriminative relation**。
我们以前只有 concat + joint t。对照臂 A 加 cond，对照臂 B 再加 disc。
intra/inter 分套注意力要动 LLaDA block，本轮不做。

---

## 4. 落地了什么

### 4.1 分链 t 采样（所有 diffusion job 共享代码）

`dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py` 的 `sample_chain_conditioned_timesteps`：

- 旧：`int(B * ratio)` + 余数给 joint → per_device 2/4 时打开比例是空操作。
- 新：每行 `torch.multinomial`。`stage=='val'` 仍全部 independent。
- `joint=1.0` → 概率向量最后一维为 1 → 永远 joint。

### 4.2 关系辅助损失

`dllm/pipelines/qwen3_vl_arch/relation_aux.py` + `LLaDAEsmcFusion`：

- 配对 mask：在 `diffusion_all_chains` 扩 eligibility **之前**快照。generated 残基上最小 `chain_id` = heavy，下一个 = light。抗原不会被当成 heavy。
- `relation_aux`：`none`（默认）/ `cognate` / `chain_drop` / `both`。
- `cognate`：未腐蚀 heavy / light mean-pool，in-batch InfoNCE（ImmunoMatch 式）。
- `chain_drop`：pair-pool vs heavy-only 与 vs light-only（SCEPTR 式）。
- 只在 `model.training` 时计入；`eval_loss` 仍是重建 CE，top-k 口径不变。
- wandb：`RelationAuxLogCallback` 插在 callback 列表最前，写 `relation_aux_loss`。

CLI（已接到 `protein_pretrain_esmc.py`）：

```text
--joint_loss_ratio --heavy2light_loss_ratio --light2heavy_loss_ratio
--independent_loss_ratio --single_chain_ratio
--relation_aux none|cognate|chain_drop|both
--relation_aux_weight 0.1
--relation_aux_temperature 0.07
```

`load_fusion_for_eval` 走默认 `relation_aux=none`，旧 ckpt 可直接评。

### 4.3 对照训练 YAML（未提交）

都从 `protein_esmc_llada270m_diffusion_immune_v3_4gpu.yml` 复制：七源 v3、generated-only、4×A100、global 128、50k、from scratch。

| Job | 旋钮 | OUTPUT_DIR 后缀 |
|---|---|---|
| `protein_esmc_llada270m_diffusion_chainratio_immune_v3_4gpu` | 分链 t 0.5/0.15/0.15/0.1/0.1 | `_chainratio` |
| `protein_esmc_llada270m_diffusion_chainratio_cognate_immune_v3_4gpu` | 同上 + `--relation_aux cognate --relation_aux_weight 0.1` | `_chainratio_cognate` |

```bash
# 确认配额后再交。不要改 2M YAML，不要 --init_fusion_weights 指向 2M。
bash scripts/volc-no-proxy.sh ml_task submit --conf \
  train_jobs/protein_esmc_llada270m_diffusion_chainratio_immune_v3_4gpu.yml
bash scripts/volc-no-proxy.sh ml_task submit --conf \
  train_jobs/protein_esmc_llada270m_diffusion_chainratio_cognate_immune_v3_4gpu.yml
```

### 4.4 评测

```bash
# 1) 生成 pairing + ImmunoMatch（现有）
bash scripts/downstream/run_immune_fusion_pairing.sh <ckpt_dir> <tag>

# 2) PLL 差（新）
bash scripts/downstream/run_pairing_pll.sh <ckpt_dir> <tag>
# 或
python scripts/downstream/score_pairing_pll.py \
  --checkpoint <ckpt_dir> \
  --output-json output/downstream_generation/<tag>_pairing_pll.json
```

PLL 作业模板：`eval_jobs/eval_pairing_pll.yml`（提交前改 `PAIRING_PLL_CKPT` / `PAIRING_PLL_TAG`）。

指标：`mean_delta_true`、`mean_delta_mismatch`、`pairwise_acc`（true δ > 错配 δ）、`auroc_delta`。
真配对应高于错配（lights 整体循环移位）。

### 4.5 单测

```bash
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
source activate /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
export PYTHONPATH=$PWD
python scripts/diagnostics/test_relation_aux.py
```

覆盖：抗原不进 pairing mask、InfoNCE、`joint=1.0` 恒 joint、B=4 时非 joint 桶会真的抽到、val=independent。

---

## 5. 本轮明确没做的

- 不提交训练/评测作业（YAML 就位即可）。
- 不动 2M 的 YAML、OUTPUT_DIR、resume 逻辑。
- 不拆 intra/inter-chain attention（要改 LLaDA block）。
- 不上结构（IgGM / AAMFM）、不扩 `<tcra>`/`<tcrb>`。
- 不把 PLL 差做成 pretrain 损失（只做 eval）。
- 不加 TCR–pMHC 的 Stage-2 上采样（加行数几乎抬不动 1.98% 残基预算；若做应改 per-sample 权重）。

---

## 6. 对照规则

- 两条新臂之间、以及与 **generated-only v3 4 卡 50k**（global 128）可以比 pairing / PLL。
- **不要**和 2M all-chains、BERT、global 256 那批并排 `eval_loss`。
- 旧 270m diffusion@42000（六源、去污前）不可并排。
