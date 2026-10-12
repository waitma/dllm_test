# Baseline Verification Audit

2026-09-30 当前机器验收见 [baseline audit](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_integrity_20260930/final/baseline_audit/baseline_provenance_audit.json)。`[P]` 固定主列，`[R]` 不替代、不补齐缺失论文值；未知可比性禁止排名。T3 的 8M/150M 区分及跨协议差值保护已落实到汇总和后续转录入口。旧结果文件未重写。运行方式与证据边界见 [报告与 CPU 验证约定](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/PROJGUIDE.md#report-integrity)。

## 2026-09-24 AB-AG 原代码比较协议增量

逐方法索引：[comparisons.tsv](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/ab_antigen/comparisons.tsv)；来源锁：[sources.lock.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/ab_antigen/sources.lock.json)；具体差异与缺失项：[AB-AG 官方协议手册](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md)。此处不把已下载代码、官方 score artifact 或 CLI help 标成 `[R]`。没有 GitHub Release 的作者仓库用确切 commit，并明确 paper-time 对应关系未证明；FLAb1 tag 不冒充 FLAb2 release。原模型仓库不能替代 benchmark 作者的适配实现。当前可引用性与执行证据见 [AB-AG 执行状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-antigen)。

> Scope: external baselines only; Ours-BioSeq is intentionally not changed or
> re-run in this repair. Last updated: 2026-07-21.
>
> 📌 **评测范围（2026-08-29 收窄）**：当前只做 **AB + TCR**，其余一律不考虑。
> 下方 audit 表中的 **MINT GeneralPPI / FLAb / NbBench 三行已冻结** —— 判定结论与论文出处保留作存档，
> 但这三项不再更新、不进 headline、不写入论文表。"Light-chain pairing" 与 "Antibody CDR" 属 AB 范围，仍在推进。
> 范围定义见 [`README.md`](README.md) §0。

## 2026-09-14 TCR comparison provenance

Publisher others results and the historical-checkpoint identity checks are documented in [T1 §7.1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T1_BINDING.md). The paper/local ESM2 identity correction and T3 scoring boundaries are documented in [T3 §7.3](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T3_REPRESENTATION.md). Provenance registration is in [README §3.1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/README.md). No new native baseline rerun or paper-exact compatibility is claimed.

## 2026-09-13 T4 paired-generation candidate audit

Paired test-data provenance and the distinction between original conditioned design and the proposed reference-free receptor input are audited in [T4 §7.6](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md). This confirms source availability and documented generative use, not a new local rerun or training-overlap clearance.

The β-only / paired-αβ inventory, primary-source evidence and implementation gates are maintained in [T4 §7.5](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md); provenance status is registered in [README §3.1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/README.md). Asset existence and native paired output do not establish a validated local paired rerun. No new performance values or machine-readable compatibility flags were added; the historical audit below is not a validation of the new protocol.

## Evidence policy

Every number now carries two independent labels:

| Field | Allowed values | Meaning |
|---|---|---|
| `evidence_type` | `paper_reported`, `official_artifact_rescored`, `official_code_rerun`, `local_reimplementation`, `local_control` | where the number came from |
| `protocol_alignment` | `exact`, `compatible`, `mismatched`, `not_comparable`, `unknown` | whether it may be compared with the anchor paper |

For T1, the canonical primary row always uses the paper original-model value.
Completed official checkpoint/code reruns on `original.zip` are separate diagnostic columns.
Missing paper cells stay missing, even when a local rerun exists.
`train.csv` and retrained heads are forbidden for external T1 baseline rows.
Other benchmarks retain their benchmark-specific protocols below. Values from two
protocols are never silently merged. The immutable reference is
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/external/paper_reported_baselines.csv`; the same directory's old `baseline_registry.json` is historical. Current report policy is recorded in each new report manifest and the audit linked above.

## Audit by benchmark

| Benchmark | Primary baseline evidence | Local evidence | Verdict / action |
|---|---|---|---|
| T1 binding | Nature Methods 2025 Supplementary Table 4, original CDR3β-only AS rows (`paper_reported`) | official original checkpoint + official inference + `original.zip` (`official_code_rerun`) | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025/summary_original_baselines.csv` is a historical summary; the repaired summarizer writes a fresh paper-primary report and keeps local reruns in separate columns. Same-directory `summary_main.csv` consumes that choice; `summary_local_runs.csv` and `baseline_paper_audit.csv` remain diagnostics. **Two defects found and handled (2026-08-30, audit `audit_2026_08_29/T1_binding.md`).** (1) *AUPRC estimator mismatch:* the paper uses R `precrec::evalmod`, but the original protocol reported sklearn `average_precision_score`, which is one-sided high (17 of the 19 rows whose AUROC also agrees are positive, 0 negative; mean +0.0004 seen / +0.0018 unseen, and +0.0010 over the 54 Ours/kNN/Random rows). `precrec` reproduces the paper to a mean \|Δ\| of 8.3e-5 (seen) / 2.4e-4 (unseen), so the three main tables were re-transcribed under `precrec` and `_confusion_row` now emits `prc_auc_precrec` natively. Four rows (epiTCR seen, NetTCR seen, TEINet-large/small seen) disagree on AUROC too and are therefore *run* divergences, not estimator artifacts — their `[R]` label is optimistic by 0.003–0.045. (2) *Shipped-split contamination:* epitope `TTAATHREK` is labelled unseen yet appears in `train.csv` (94 rows), and 30 of its 66 `unseen_test` rows are verbatim training pairs (all positives). Only `train.csv`-trained methods are affected; CDR3-kNN's entire apparent unseen signal comes from it (it scores exactly 0.0 on all 624 uncontaminated rows, so removing the epitope collapses its AUROC to precisely 0.5000). |
| T2 clustering | NAR-GAB native Retention/Purity operating points (`paper_reported`) | author assignments projected to the paper's paired universe; HD/LD locally reconstructed | LD indel bug fixed and artifacts regenerated. **The paired universe now reproduces the paper exactly** (2026-08-30): 4,779 pairs / 8,395 sequences / 4,103 CDR3a / 4,292 CDR3b, all four hardcoded constants hit. The old 5,368-pair figure was not database drift as previously claimed but a single pandas-vs-dplyr bug in `curate()` — `dplyr::filter` drops NA predicates so `filter(PubMed_ID != <10x url>)` also removed rows with a missing PubMed_ID, while pandas' `!=` kept them (634 rows, exactly the 589 extra pairs, pair sets perfectly nested). Native calibration consequently went from 7/9 to **9/9** within ±0.03, with a max \|delta\| of 0.005 — HD and LD had been sitting just outside the threshold for the same root cause. The paper table stays primary because the projections and the HD/LD reconstruction are still local artifacts, but the universe is no longer a deviation. Two disclosed caveats remain: the local hybrid table uses majority-collapsed epitope labels (purity understated by <=0.0023, measured), and the embedding threshold curves have **not** been recomputed on the corrected universe. |
| T3 representation | SCEPTR Table SI, six pMHC at k=200 (`paper_reported`) | mismatched local protocol; rerun pending | 2026-09-13 primary-source audit supersedes the old compatibility verdict; see [T3 §7.2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T3_REPRESENTATION.md). Historical deltas do not establish paper reproduction. Paper values remain intact; do not treat legacy registry exact/compatible flags as current local validation. |
| T4 generation | TCRT5 authors' stored predictions rescored with the official metric port (`official_artifact_rescored`) | project generators / controls | Main paper view is the 13 sparse pMHC. `RVRAYTYSK/HLA-A*03:01` is reserved for simulation; the old 14-pMHC aggregate is retained only as `[14-pMHC control]`. **Baseline provenance is unaffected by the 2026-08-31 diagnostic, but the interpretation of our own rows changed** (`audit_2026_08_29/T4_infill_diagnostic.md`): removing the epitope entirely and re-filling a masked window of a real held-out CDR3b, our model loses to a context-blind (length, position) PWM by 7.05-12.47 pp. That holds on a holdout with zero exact overlap, on the `tcr_pair` layout that supplies 37.2% of training, and on the 616 rows the model was actually trained on (-7.47 / -11.31 pp), so it is a fitting failure, not a conditioning, layout, generalisation or contamination artifact. Switching Setting B to `--with-mhc` (the layout carrying 3.726% of training vs 0.036% for the default) made all six views worse. Our "close to OLGA" rows must therefore not be read as "learned a good unconditional prior". |
| MINT GeneralPPI | MINT Source Data, Figure 2c–h / Table S1 (`paper_reported`) | capped local data/head port | Paper values are primary. Local Bernett is not the paper Gold-standard PPI task. Sixteen legacy SKEMPI/PDB-Bind Pearson/RMSE rows are withdrawn because they were measured in fold-specific transformed target spaces. |
| FLAb | MINT Source Data Figure 3b, nested 10-fold/5-fold Ridge R² (`paper_reported`) | historical 5-fold pooled Spearman | Metrics and CV differ; historical rows are local diagnostics only. Runner is repaired for nested R², but baselines still require a fresh run. |
| Antibody CDR | Ophiuchus-Ab Table 2 AAR (`paper_reported`) | local AntiBERTy/AbLang2 masked-fill | Paper rows are primary and carry the paper's train/test-overlap caveat. Ophiuchus path resolution is fixed; its local rerun remains pending. **Our own rows are contaminated (measured 2026-08-30, `outputs/ab_cdr_leakage_report.json`).** Antibody-side decontamination was disabled for all four training runs, and against the ASD-antibody corpus (707,800 unique heavy chains) 54.4% of the 3,127 SAbDab Kong-fold test antibodies have their exact CDR-H3 present in training (whole-chain identity >=0.95 independently gives 55.3%; the H3-mismatched germline background is 0.83% at >=0.99, a 52x enrichment). The OAS corpus is clean by the same measurement -- its observed values equal the germline control at every threshold -- so ASD is the sole contamination source. SAb23H2 is far cleaner at 25% verbatim heavy chains with a median best identity already at background. Since CDR-H3 infilling is the task itself, the "Ours" rows are upper bounds, not generalisation, and must not be presented as tied with the paper's no-overlap zero-shot Ophiuchus. |
| NbBench | Table 5, frozen backbone + learned MLP, 3-seed mean (`paper_reported`) | sklearn single-seed probes | Not the same head or seed protocol. Paper Table 5 is primary; local values are diagnostic only. Paratope AUROC from the paper must never be mixed with local AUPRC. |
| Light-chain pairing | Ophiuchus-Ab Table 3, ten metrics x three methods x two prompting settings (`paper_reported`) | official-weight reruns of Ophiuchus-Ab, p-IgGen and LICHEN | **Reproduced (2026-08-30).** The earlier "no comparable paper table" verdict was stale: Table 3 exists and the local protocol (OAS holdout500, 3-residue light prompt, n=8 per heavy) matches its with-prompting block. LICHEN reproduces to within 0.015 on all nine shared metrics; p-IgGen to within 0.023 except ImmunoMatch (+0.014) and Better (+2.0 pp), consistent with its stochastic decoding. `W_property` is not implemented locally, so metric coverage is 9/10. Any comparison must state which prompting setting it belongs to — the two disagree in direction (prompting raises V-family agreement 5.3x but lowers ImmunoMatch). |
| Humanization | no comparable functional paper table in this harness | local reconstruction diagnostics | Local control only; no paper-reproduction claim. |

## Exact paper-reference checkpoints

- T1: 58 method/split/metric rows transcribed from Supplementary Table 4. Verified
  2026-08-30 against local reruns under both AUPRC estimators; the paper's
  estimator is `precrec`, confirmed to a mean \|Δ\| of 8.3e-5 / 2.4e-4 on the 19
  rows whose AUROC also agrees. Evidence:
  `outputs/tcr_binding_nm2025/auprc_convention_audit.{json,csv}`.
- T2: 18 native Retention/Purity rows for nine methods.
- T3: 36 per-pMHC k=200 AUROC rows.
- MINT GeneralPPI: MINT and ESM2-650M, six task-primary metrics each.
- FLAb: seven methods × four datasets, mean R² and standard deviation.
- Ophiuchus-Ab CDR: three methods × CDR-H1/H2/H3 AAR.
- NbBench: 11 models × 11 Table-5 tasks.

Together these form 282 labelled `paper_reported` rows. T4 is kept as an
official-artifact rescore because the released prediction CSV lets us compute
the exact metric view rather than transcribe scalars.

## Repaired defects

1. **T2 LD:** the old code placed equal-length strings in a deletion bucket and
   then required their lengths to differ by one, an impossible condition. LD
   therefore silently equalled HD. It now deletes from the longer observed
   sequence and directly looks up the shorter sequence.
2. **T4 aggregation:** the old headline averaged all 14 pMHC. The main output now
   excludes the reserved simulation pMHC and asserts `n_pmhc == 13`.
3. **MINT regression units:** predictions are inverse-transformed to raw target
   units before Pearson/RMSE. Old unlabelled regression files are not repaired in
   place; they are explicitly withdrawn pending rerun.
4. **FLAb leakage/protocol:** PowerTransformer is now fitted inside each training
   fold and Ridge selection uses nested 5-fold CV inside 10 outer folds.
5. **Ophiuchus CDR path:** the runner accepts either `<root>/fold_N` or
   `<root>/<mode>/fold_N`, avoiding duplicated `cdrh*/cdrh*` paths.
6. **T1 original-only boundary:** the official runner now loads test artifacts
   only, validates the `original.zip` checksum plus each normalized test against
   the test-only extraction manifest, records `retrained=false`, and
   maps TEINet/ERGO/TPBTE variant tags to their checkpoints before wrapper
   import. ERGO output identifiers are preserved before the official featurizer
   mutates its input lists.
7. **T1 runtime artifact mapping:** every official wrapper must disclose the
   exact model artifacts opened during inference, and the runner requires an
   exact ordered match with the catalog. TEIM therefore records both its
   sequence checkpoint and epitope autoencoder; PanPep records its model plus
   two zero-shot memory files; ERGO-AE records its classifier plus TCR
   autoencoder. Old outputs without this embedded mapping are not eligible.

## Pending baseline-only reruns

- MINT SKEMPI and PDB-Bind baselines with raw-unit regression reporting.
- FLAb baseline sweep with paper-aligned nested R².
- Ophiuchus-Ab CDR-H1/H2/H3 after the dataset-path fix.
- A true NbBench MLP/3-seed reproduction if local reproduction, rather than the
  paper Table-5 reference, is required.

Run the audit with:

```bash
/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/scripts/import_paper_baselines.py
/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/scripts/audit_baseline_provenance.py --strict
```

Current audit: all immutable paper tables, the T2 LD artifact, the T4 sparse-13
artifacts, the MINT withdrawal audit, and the T1 original-only/runtime-artifact
selection pass with zero errors and zero warnings.

## 2026-09-24T03:54:43.033669+00:00 FLAb2 已排除

FLAb2 已从本轮候选与默认探测入口移除，历史缺代码记录不再作为待办。MAGE 全量权重已取得；官方 n=1 smoke 已提交、尚无推理成功证据，仍不标 `[R]` benchmark 完成。

## 2026-09-24T04:07:06.335060+00:00 S²ALM 发布产物核查

见 [S²ALM 复现审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/S2ALM_REPRODUCIBILITY_AUDIT.md)。论文仓库404；同名仓库只有 README/LICENSE/图片，所链接权重为 ProtET，不能冒作 S²ALM。保留 `[P]`；不制造 `[R]`。PALM/A2binder 源码与数据已追踪，权重只确认公开记录，未下载或推理。

## 2026-09-24T04:48:58.155591+00:00 BioMap baseline provenance

[BioMap 官方 baseline 审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/BIOMAP_BASELINE_REPRODUCIBILITY_AUDIT.md) 区分 A2binder 原实现、基模型原仓库和缺失的 BioMap adapter；正式预测图数据只能作原始 artifact，不等同共同测试集或本地复现。

## 2026-09-24T05:06:03.673383+00:00 BioMap 从发布审计进入首轮运行

官方源码与初始化产物已校验，原模型/数据 CPU 加载验证完成；正式任务仅 Queue，不标已完成 `[R]`。证据见 [BioMap 官方运行状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#biomap-official-run)。

## 2026-09-27 AB-AG sequence v1

新 AB-AG 四子集为官方数据上的项目派生协议：CoV-UniBind evaluator 未完整公开，CHIMERA 改为 antigen-sequence-only，均不得将 Ours 入口标 [R] 作者复现。外部 baseline 保持原代码要求。版本/哈希见官方获取清单，细节见[任务手册](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md)。
