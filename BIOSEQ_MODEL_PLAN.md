# BioSeq Model Plan

> **Current architecture boundary (2026-09-11):** The only current immune data implementation is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada`; the formal training entry is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py`. The active data flow is raw → adapter → `BioSeqRecord` → offline filter → prepared semantic JSONL → training-time grammar/padding/per-chain encoder reconstruction/masking, with no model-ready token cache. The deleted `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data` alias tree, deleted `training` tree, deleted `GRAMMAR_V1.md`, and deleted legacy training/build/test/job files are historical deletion facts, not current dependencies. Retained model-layer files are `modeling_bioseq.py`, `sampling_bioseq.py`, and `relation_aux.py`; `downstream/grammar` remains active for fusion CDR/light-pairing/TCR-generation public implementations.

## Goal

Build a diffusion-model-based immune-receptor foundation model in
`/vepfs-mlp2/c20250601/251105016/project/dllm_test` for paired antibody H/L,
paired TCR α/β, antibody-antigen recognition, and TCR-epitope/pMHC recognition.

## Active biological scope（2026-07-23）

- The next foundation-training recipe has four planes only: antibody pairing,
  TCR pairing, antibody-antigen, and TCR-epitope/pMHC.
- Nanobody/VHH, MINT/STRING/general-PPI, the current antigen-free
  `neutralization` shard, and specificity-free bulk TCR are excluded from the
  active recipe. Their historical checkpoints, benchmark artifacts, and older
  roadmap notes remain provenance, not active training requirements.
- Antibody context means actual protein/peptide sequence, not only a target
  name. TCR context keeps epitope separate from MHC/B2M and records whether an
  MHC sequence is full length or a pseudosequence.
- The canonical data record must preserve sequence scope, receptor/ligand
  clusters, donor/assay, continuous value/unit/censor, and source record
  provenance. Dataset rows with the same sequences but different HLA or assay
  are not interchangeable duplicates.
- The current immune LLaDA runtime exposes the semantic record and mask contract through
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data`; target views
  are represented by record metadata plus runtime grammar/mask construction, not by the deleted
  Arrow loader. H→L/L→H, α→β/β→α, antigen-conditioned receptor generation, pMHC-conditioned
  TCR generation, and chains→relation remain task/view goals to validate on the current line.
- Historical 7L / `immune_receptor_v2` candidate counts remain in
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/TRAINING_DATA_CATALOG.md` and
  `IMMUNE_RECEPTOR_DATA_V2.md`. Current prepared v4/v5 counts are owned by
  plan §4.1 / §4.2 — do not read the 7L catalog as the live mix.

## Canonical data execution（2026-08-04）

- Historical/candidate AB/TCR-only `bioseq.v2` artifacts were built under
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/immune_receptor_v2`;
  their historical build entry was
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_immune_receptor_v2.py`.
  They are not the current runtime training input while their export/training gates remain false.
- The frozen candidate unions contain 922,479 TCR records and 470,949 antibody
  records. The strict split populations are 41,489 TCR specificity records,
  308,267 antibody-antigen interactions, and 19,987 canonical antibody-property
  records; 19,981 property records satisfy the export-core rules.
- All nine deterministic component/group split manifests pass their disjointness
  audit. A source's released train/test assignment is provenance only and is
  never silently treated as the new training split.
- Exact measurement duplicates may merge provenance; conflicting labels,
  different assay values/censors, HLA contexts, or target mapping scopes remain
  separate. No majority-vote label collapse is allowed.
- Frozen benchmark exact/near quarantine, MMseqs2 sequence-cluster
  decontamination, group-disjoint resplitting, train-only negative construction,
  and immutable SHA256 manifests are complete. Core build
  `ir2exp_f7a60484c7e3a20db6a2` and strict OAS/OTS pairing build
  `ir2pair_6a1a5b62752caccd2cd5` both pass their technical audits with zero
  residual benchmark-cluster matches.
- Candidate recipe `ir2recipe_1e3eac44551e88aedc6e` selects strict OAS H/L,
  strict OTS alpha/beta, antibody-cluster-disjoint recognition,
  receptor-cluster-disjoint TCR recognition, and parent-disjoint antibody
  properties. It references 3,077,769 real train, 39,138 valid, and 38,565 test
  records, plus a separate 3,970-row train-only synthetic TCR-negative pack.
- This execution did not change the model, renderer, checkpoint, or sampling
  weights and did not start training. `technical_data_export_ready=true`, but
  `export_ready=false` and `training_ready=false` until the SAbDab2/CATNAP rights
  review, per-chain/per-region target rendering, and four-plane sampling/token
  budget policy are completed.
- Full schema, source decisions, counts, hashes, leakage audit, and export gates
  are authoritative in
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/IMMUNE_RECEPTOR_DATA_V2.md`.

## Code Location

- Current immune data pipeline: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada`
- Current formal training entry: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py`
- Current immune data tests: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada`
- Retained model-layer compatibility code: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch`
- Historical Ophiuchus/BioSeq code and tests under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq` and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq` are not the current immune training entry.
- Weight root: `/c20250601/mj/model_weights`

## MINT downstream boundary (2026-07-21)

- MINT 重建属于 benchmark/data-harness 修复，不改变 BioSeq 模型结构、训练数据或 checkpoint。
- 当前 MINT 面板严格为 HumanPPI、YeastPPI、Gold-standard PPI、MutationalPPI、SKEMPI
  五项；canonical 数据来自
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/mint_official`。
- 普通 PPI 分类输入两条 partner 序列；MutationalPPI/SKEMPI 输入 WT 与 mutant 的两条
  partner 序列，并用 `embedding(WT PPI) - embedding(mutant PPI)` 训练下游 head。
- 只有与论文数据、fold、head、重复次数全部对齐的本地结果才可称复现。当前不修改
  BioSeq 模型，先修 baseline：HumanPPI、YeastPPI、Gold-standard PPI 的 8 模型结果
  直接引用 Source Data 并标 `[P]`；MutationalPPI、SKEMPI 在可审计本地固定 fold 上重跑
  同一组 8 模型并标 `[L]`，不跨协议排名为 paper-exact。
- 本地两项统一用 separate-chain、`embedding(WT)-embedding(mutant)`、WT/mutant 对齐
  窗口、3 次 640-hidden MLP；run cap=2048，ESM-1b 与 ProGen2-Large 按各自原生限制
  使用 1024。MutationalPPI 用 WT pair-group 10-fold；
  SKEMPI 用冻结的 notebook complex-held-out 三折。该变化只属于 benchmark harness。
- 数据构建和协议审计入口是
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/mint_tasks/OFFICIAL_REBUILD_AUDIT.md`。

## Immune LLaDA data / receptor-completion decisions（2026-09-12）

- Current implementation: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada`; formal entry: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py`.
- Current data flow is raw source reader → source adapter → canonical `BioSeqRecord` → offline deterministic filters → prepared semantic JSONL/manifest → training-time grammar rendering/collation → batch padding/tensor assembly → per-chain encoder input reconstruction → diffusion/MLM masking. No model-ready token cache is produced.
- Receptor-completion / relation-diffusion **design and risk register** are owned by `/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/PLAN_TCR_BETA_ONLY_RELATION_DIFFUSION.md`. Layout fields: `DATA_FORMAT_AUDIT.md`. How to run: `dllm/pipelines/immune_llada/README.md`.
- Method decisions that must not be “fixed” later: (1) complete epitope-conditioned single-chain rows because `GrammarTokenizer` has no `<tcra>`/`<tcrb>` and positional identity made 8,885 alpha-only rows indistinguishable from beta-only; (2) branch on actual fv/CDR3 columns, not `sequence_scope`; (3) drop all-X epitopes via a named filter, not a silent adapter `None`; (4) do not infer MHC from epitope co-occurrence; (5) accept synthetic-`X` attention cost as the price of a single α/β layout.
- Prepared versions: v4 published; v5 published. Counts: plan §4.1 / §4.2. v3-era homotypic-pair acceptance remains in `docs/IMMUNE_LLADA_DATA_ACCEPTANCE.md` and is not the current default.
- The old `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data` alias tree and old `training` tree are deleted. Do not recommend those paths. `refactor_baseline` and `docs/archive` remain historical evidence.
- The retained qwen-named directory is model-layer code only: `modeling_bioseq.py`, `sampling_bioseq.py`, and `relation_aux.py`. Active immune grammar/tokenizer/collator: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data`.

## Architecture

- The antibody training target is the exact Ophiuchus-Ab path in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/ophiuchus`.
- `BioSeqEncoderDiffusionModel` is the ESMC/ESM feature-conditioned target.
- The no-encoder antibody model uses the migrated Ophiuchus-Ab ESM2 transformer with chain-aware multimer attention.
- The ESMC/ESM feature-conditioned model runs the biological encoder on the current diffusion state `x_t`, replaces decoder residue embeddings with gathered encoder features (default; no projection), and treats ESMC freezing only as a bootstrap or ablation setting.
- ESMC should be treated as a per-protein-sequence representation model, not as a native multi-chain complex encoder.
- Multi-chain reasoning should be implemented in the BioSeq decoder/collator with explicit `chain_ids`, or by using an ESMFold2-style structure head that builds complex-level pair representations after ESMC representations.
- ESMFold2 supports biomolecular complex inputs through `StructurePredictionInput` / `ProteinInput` / `DNAInput` / `LigandInput`; its complex handling is not simply independent ESMC encoding with no downstream cross-chain module.
- The ESMFold2 preprint describes full uncropped protein sequences for each chain being passed independently to frozen ESMC 6B, followed by projection into a 2D pair representation, language-model encoder layers, recurrent pair folding layers, and an atom-level diffusion module.
- The ESMC/ESMFold2 paper does discuss multi-chain systems, but the explicit multi-chain mechanism is in ESMFold2: ESMFold2 encodes each protein chain independently with frozen ESMC 6B, then builds complex-level pair representations and folds/designs complexes downstream. ESMC itself should not be treated as a native multi-chain interaction encoder.
- For antibody, TCR-pMHC, and PPI sequence modeling, ESMC embeddings alone are insufficient as the only cross-chain mechanism; the BioSeq no-encoder path must keep Ophiuchus-Ab-style chain-aware attention, and the encoder path must pass `chain_ids` plus ESMC-derived features into a decoder or structure-style pair module.
- Ophiuchus-Ab's MINT path uses `multimer_attn` inside the ESM2 transformer layers as the cross-chain mechanism. It is not the HuggingFace-style `crossattention` / adapter-cross-attention path present in other AirGen DPLM modules.
- Ophiuchus-Ab compatibility uses the ESM2 alphabet from `/c20250601/mj/model_weights/esm2/esm2_t33_650M_UR50D`: vocabulary size 33, `<cls>` id 0, `<pad>` id 1, `<eos>` id 2, `<mask>` id 32.
- The released local ESMC tokenizer files under `/c20250601/mj/model_weights/esmc/ESMC-300M`, `/c20250601/mj/model_weights/esmc/ESMC-600M`, and `/c20250601/mj/model_weights/esmc/ESMC-6B` expose the same ids as ESM2 for standard amino-acid tokens, but id 31 differs: ESM2 uses `<null_1>`, while ESMC uses `|` as an additional special token. Use the encoder's own tokenizer when ESMC is active.
- Ophiuchus-Ab antibody collation should encode heavy and light chains independently with per-chain `<cls>/<eos>`, pad heavy to 150 tokens and light to 128 tokens, then concatenate to a fixed 278-token sequence with explicit `chain_ids`.
- The Ophiuchus-Ab architecture preset should use hidden size 1280, 33 transformer layers, 20 attention heads, FFN size 5120, ESM2 token dropout, no learned position embeddings, and chain-aware multimer attention.
- Loading `/c20250601/mj/model_weights/ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt` is implemented in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/ophiuchus/multichain.py` via `load_ophiuchus_checkpoint`, using the Ophiuchus-exact mint ESM2 block under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/ophiuchus/mint`.

## MIMIC-Inspired Design Notes

- MIMIC/LORE should be treated as a reference for the later multimodal BioSeq roadmap, not as code to import today.
- Borrow the split-track idea: aligned per-residue modalities such as amino acid sequence, chain type, CDR/region tags, ESMC features, predicted secondary structure, SASA, structure tokens, interface/surface features, and antigen/TCR/pMHC annotations should be fused within the same residue coordinate frame instead of flattened into separate long sequences.
- Keep unaligned context as separate token groups: task prompt, species, assay, tissue/cell context, target name, functional caption, and design constraints should be separate semantic tokens that can interact with sequence tracks.
- Add register tokens for global complex/entity representation; use register tokens plus track pooling for downstream antibody/TCR/PPI property heads.
- Use local group-reset RoPE or equivalent group-aware position handling so each chain or modality group preserves internal residue distances without treating all concatenated tokens as one absolute coordinate system.
- Use pathway-based training over partially observed samples: examples can have sequence only, sequence plus ESMC, sequence plus structure/surface, paired chains, antigen context, or downstream labels. Rare high-value pathways such as antibody-antigen and TCR-pMHC should be upsampled instead of drowned by sequence-only examples.
- Use asymmetric context/target budgeting for multimodal generation: large encoder context for conditioning, smaller decoder/diffusion target window for masked residues or design regions.
- Use length-bucketed dynamic batching rather than truncating protein complexes; for protein structure or interaction tasks, dropping over-budget samples is safer than cropping away global contacts.
- For protein design evaluation, borrow MIMIC's independent-oracle pattern: generate with BioSeq, then evaluate with ESMFold2/AlphaFold-style structure confidence, interface confidence, TM-score-like fold recovery, and surface/chemistry similarity instead of scoring only with the same model that generated the sequence.
- Release status as of 2026-06-11: MIMIC code, weights, and LORE release assets are not yet publicly available for direct integration.

## Qwen3-VL-Inspired Architecture Notes

- Qwen3-VL should be used as an architecture pattern for multimodal BioSeq, not as a direct code dependency.
- The useful pattern is `modality encoder -> merger/projector -> placeholder-token embedding replacement -> decoder -> task head/loss`.
- For immune receptor modeling, replace Qwen3-VL image/video tokens with typed biological placeholders such as antibody heavy/light, TCR alpha/beta, peptide, MHC, antigen, and PPI partner tokens.
- Replace Qwen3-VL 3D T/H/W multimodal RoPE with BioSeq-aware position metadata: text/control position, chain-local residue position, chain id, chain role, modality/track id, and optional interface or structure-coordinate track.
- Keep a connector layer between tunable ESMC or ESM2 encoders and the BioSeq decoder. This connector should handle length reduction, hidden-size projection, and modality/chain-type normalization, analogous to Qwen3-VL's visual merger.
- Borrow Qwen3-VL's deepstack idea cautiously: intermediate ESMC/ESM2 features can be injected into early BioSeq decoder layers at residue-token positions, but this should be behind an explicit config flag and validated against the simpler single-projection path.
- Training should expose separate learning rates for biological encoders, connector/projector modules, and decoder/diffusion modules, mirroring Qwen3-VL's separation of vision tower, merger, and LLM; the default should keep the biological encoder trainable when it is used.
- Dense decoder should be the first BioSeq foundation target; MoE-style routing can be revisited later for large mixed antibody/TCR/PPI training after the dense model and data schema are stable.

## Multi-Chain Tokenization Notes

- BioSeq should not represent multi-chain complexes by concatenating raw amino acid strings with a single generic separator only.
- ESMC's local tokenizer includes `|` at id 31 and marks it as an additional special token, which is compatible with using a chain separator for ESMC inputs. This is only tokenizer-level support; it is not proof that ESMC alone learned robust cross-chain interaction modeling.
- ESM2/MINT-compatible tokenization uses `<null_1>` at id 31 instead of `|`. Therefore `Esm2SequenceTokenizer` is correct for Ophiuchus-Ab/MINT and local ESM2 snapshots, but ESMC encoder runs should use `HuggingFaceEsmTokenizerAdapter` or an ESMC-specific tokenizer loaded from `/c20250601/mj/model_weights/esmc/<model>/tokenizer.json`.
- Keep the first pretraining token format minimal. Required tokens are only amino acid tokens plus `<pad>`, `<mask>`, `<unk>`, `<bos>`, `<eos>`, `<chain_sep>`, and a small set of complex-type header tokens such as `<type_ab>`, `<type_ab_ag>`, `<type_tcr>`, `<type_tcr_pmhc>`, and `<type_ppi>`.
- Do not add many role-specific or position-number tokens in the first version. For example, avoid `<ab_heavy>`, `<tcr_beta>`, or `<pos_53>` as default vocabulary items unless ablation shows they are needed.
- Use chain ordering conventions under each complex-type header instead of many chain-role tokens. Example: `<type_ab>` means chain order is heavy then light; `<type_tcr_pmhc>` means alpha, beta, peptide, MHC; `<type_ppi>` means partner A then partner B.
- Model hierarchy with embeddings, not vocabulary growth. Each residue should receive a chain-internal position embedding or RoPE that resets per chain, plus a chain-level index embedding for the outer multi-chain order.
- The initial position system should have two levels: `residue_position_in_chain` and `chain_index_in_complex`. Optional global absolute position can be retained only for packing/caching, not as the main biological position signal.
- Pretraining should be treated as masked generation over all eligible residues with different mask probabilities by chain/type/task. Fixed-context behavior is a mask policy choice, not a separate objective.
- The schema should support `diffusion_target_mask`, `fixed_context_mask`, and optional per-chain mask probabilities. More detailed biological relationship features or external representations can be added later through encoders/connectors rather than through more special tokens.

## Immune Receptor Modeling Considerations

- First-version pretraining should stay simple, but the data schema must preserve immune-receptor metadata even when it is not used as input tokens.
- Preserve receptor gene and region annotations where available: V/J genes for antibody light chains and TCR alpha chains, V/D/J genes for antibody heavy chains and TCR beta chains, CDR1/CDR2/CDR3 spans, framework spans, species, isotype, and source assay.
- For antibodies, paired heavy-light learning is central. Heavy-only or light-only samples can be useful for scale, but paired heavy-light data should be upsampled or given a dedicated mixture so the model learns chain pairing rather than only single-chain naturalness.
- For antibody-antigen tasks, antigen identity is often missing in large repertoire datasets. Do not treat repertoire-scale antibody samples as antigen-conditioned examples unless target antigen labels or binding assays are known.
- For TCR-pMHC, MHC allele/class and peptide identity are essential context, not optional metadata. TCR-only pretraining can learn receptor grammar, but TCR-pMHC learning requires paired alpha-beta TCR plus peptide plus MHC/HLA context whenever available.
- TCR-pMHC labels are noisy and depend on assay type. Store assay/readout metadata and avoid mixing tetramer binding, activation, expansion, tissue enrichment, and author-curated specificity as identical labels in supervised evaluations.
- Negative sampling for binding/specificity tasks must be explicit and benchmark-specific. Randomly shuffled negatives are useful for training but can overstate generalization; unseen peptide/HLA and unseen receptor splits should be maintained for evaluation.
- Multi-chain relation learning should be measured separately from single-chain grammar using pair recovery, conditional generation, binding/specificity ranking, and structure/interface oracle evaluations.
- Region-aware mask policies should be available as an ablation: uniform masked generation is the default foundation objective, while higher CDR3/paratope/peptide masking can test whether the model improves receptor-specific design.
- External structural or representation features should be added through encoder/connector paths after the minimal sequence diffusion baseline is stable, rather than expanding the first tokenizer.

## Additional Data Types

- Repertoire-scale AIRR-seq data: large unpaired or paired BCR/TCR repertoires from OAS, OTS, iReceptor/AIRR Data Commons, PIRD, and TCRdb. Use for broad masked-generation pretraining and repertoire distribution learning.
- Paired-chain receptor data: paired antibody heavy/light and paired TCR alpha/beta records. Use with higher mixture weight than single-chain data to learn receptor pairing.
- Antigen/specificity-labeled data: antibody-antigen and TCR-pMHC pairs from IEDB, VDJdb, McPAS-TCR, TBAdb/PIRD, curated antibody binding resources, and assay-specific datasets. Use for conditional mask generation, ranking, and supervised evaluation; keep assay metadata.
- Structural complex data: SAbDab/SAbDab-nano, STCRDab, TCR3d, SCEptRe, PDB-derived antibody-antigen and TCR-pMHC complexes. Use for structure/interface oracles, optional structure-conditioned training, and nonredundant benchmarks.
- Functional and biophysical data: affinity, neutralization, specificity, developability, expression, thermostability, aggregation, immunogenicity, polyreactivity, and pharmacokinetic labels. Use as downstream heads or filtering/oracle data, not as the first pretraining objective.
- MHC and epitope presentation data: peptide-HLA binding, ligand elution, immunogenicity, and MHC allele metadata. Use to make TCR-pMHC context realistic and to avoid treating peptide identity without MHC context as sufficient.
- Single-cell immune multi-omics: paired receptor sequence plus cell type, tissue, disease, clonotype expansion, gene expression, and antigen-tetramer labels. Use later for context-aware modeling and split construction, not as required input for first baseline.
- Germline/numbering/reference data: IMGT, OGRDB, ANARCI/ANARCII-style numbering, V/D/J references, CDR/FR annotations. Use for metadata normalization, region-aware masking ablations, and evaluation stratification.
- Synthetic and library screening data: display libraries, deep mutational scanning, saturation mutagenesis, designed binders, and experimentally validated generated receptors. Use for targeted fine-tuning and validation of design behavior.
- Negative and decoy data: shuffled nonbinders, hard negatives by shared peptide/HLA or similar receptor, and structure/interface decoys. Keep negative-generation protocol explicit because it strongly changes reported performance.

## TCR Data Expansion Notes

- Current local TCR resources are not empty: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ots_paired_clean/final` contains about 2.12M cleaned paired alpha/beta TCR records, and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr` already contains VDJdb, McPAS-TCR, ImmuneCODE/MIRA, IEDB export archive, and PIRD-derived assets.
- For first masked-generation pretraining, local OTS paired alpha/beta is enough to start; the immediate gap is integrating it cleanly into the unified BioSeq training mixture rather than only downloading more.
- For larger TCR grammar coverage, add bulk/unpaired TCR repertoire sources such as TCRdb2.0, iReceptor/AIRR Data Commons, and immuneACCESS/immunoSEQ public exports. These should be stored under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw` and converted into explicit single-chain or beta-chain records with sample metadata.
- TCRdb2.0 raw data is now downloaded under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw/tcrdb2_0`: 263 project zips, 263 metadata CSVs, and 1 healthy reference zip. Download validation against remote `Content-Length` passed with no missing or mismatched files; see `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw/tcrdb2_0/manifests/tcrdb2_download_validation.tsv`.
- For TCR-pMHC relation learning, prioritize specificity/context datasets such as IEDB receptor exports, latest VDJdb, McPAS-TCR, ImmuneCODE/MIRA, PIRD/TBAdb, IMMREP/Kaggle-style benchmark data, and paired peptide-HLA datasets. Store raw files under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_specificity_raw`.
- Do not blindly mix all newly downloaded bulk TCR rows into the main pretraining pool. Bulk datasets can dwarf paired OTS and erase multi-chain learning unless mixture weights cap bulk/unpaired data.
- Before any large new download, add dataset manifests and adapters that record source, license/access terms, chain availability, paired/unpaired status, peptide/MHC availability, assay type, species, tissue/disease, and split group.

## End-to-End Pipeline

- Raw data layer: keep each source immutable under absolute raw roots such as `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw/tcrdb2_0`, `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ots_paired_clean/final`, `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr`, and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi`.
- Manifest layer: every source should have a machine-readable manifest with source name, raw path, file checksum or byte validation, species, chain availability, paired/unpaired status, task labels, peptide/MHC availability, license/access terms, and split group. TCRdb2.0 already has manifests under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw/tcrdb2_0/manifests`.
- Adapter layer: current source adapters and offline preprocessing live under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data`; they normalize raw records into canonical `BioSeqRecord` rows before prepared semantic JSONL export. The former `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/adapters.py` conversion path is historical.
- Canonical data layer: current training rows use the `BioSeqRecord`/prepared semantic contract owned by `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data`. Bulk unpaired TCR rows from TCRdb2.0 should become single-chain or beta-chain records first, not multi-chain pseudo-pairs.
- Prepared-data layer: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/dataset.py` reads the immutable prepared manifest and semantic JSONL shards. Source weighting, filtering, row conversion, and rejection decisions belong to offline preprocessing, not a deleted raw-CSV runtime dataset.
- Collation/masking layer: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/grammar.py` and `collator.py` emit token tensors plus `diffusion_target_mask`, `fixed_context_mask`, chain ids, chain-internal positions, and per-chain encoder tensors. Diffusion/MLM corruption and loss operate only on the selected eligible positions.
- Retained compatibility model layer: the Ophiuchus-Ab/MINT multichain code under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/ophiuchus` remains historical/independent provenance, not the current immune training entry; the former `examples/bioseq/train_bioseq_ddp.py` entry point is retired and deleted.
- Current model layer: the ESMC/ESM2 feature-conditioned fusion path extracts features from the current diffusion state `x_t`, then performs multi-chain denoising over the concatenated token stream. Fixed context chains, such as antigen in antibody-antigen generation, remain clean and do not receive direct reconstruction loss.
- Checkpoint layer: training checkpoints should be written under absolute output roots, save `latest.pt` and `final.pt`, and preserve `backbone_state_dict`, optimizer state, step, epoch, and args for resume and downstream evaluation.
- Evaluation layer: downstream evaluation should use `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark` as the first integration point. The benchmark already accepts `--embedder bioseq:/abs/path/final.pt` through `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/common/model_api.py`.

## Fast Downstream Validation Plan

- Validation should run on every meaningful current immune-fusion checkpoint, not only the final checkpoint. Minimal cadence: initialization/smoke checkpoint, early checkpoint, mid-training checkpoint, and final checkpoint; record the actual absolute checkpoint paths in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md`.
- Tier 0, pretraining sanity: held-out diffusion loss on small OAS/OTS/nanobody/TCRdb2.0 slices, amino-acid distribution checks, valid-token rate, duplicate/near-neighbor rate against train, and chain-length distribution drift. This is the fastest failure detector for bad adapters or masks.
- Tier 1, embedding-only IRBench: run the existing benchmark with frozen embeddings and cheap heads. Priority commands are T1 TCR binding, T3 TCR representation, P1 PPI, and NbBench scalar tasks under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark`. This checks whether the trained backbone representation is useful without generation noise.
- T1 comparison boundary: Ours may continue to use the frozen-head protocol on its designated training split, but external baseline rows must not be retrained. Their canonical source is official original checkpoint + official inference + `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ntmethod_binding/original.zip`; unavailable official runs fall back to the labelled paper original-model value. This keeps the model-development protocol separate from external-baseline provenance.
- T1 retrained checkpoints are a separate artifact-reproduction audit, never a replacement for the canonical original-only table. The audit uses the released fold checkpoint + official inference + exact `retrain.zip` member and R `precrec`-compatible AUPRC. A local retrained value is selectable only when both five-fold mean AUROC and AUPRC round to the paper's four decimals; otherwise retain the local result as a diagnostic and report the marked paper retrained value. Any inference-time use of fold train data (currently only TCR-H descriptor-column reconstruction) must be declared and must not call model fitting.
- Our-model T1 retrained comparison uses the same five official `retrain.zip` train/test folds but never updates the BioSeq checkpoint. Render peptide + CDR3β as one role-explicit joint binding grammar record, read the final post-LLaDA residue states, apply one global mean across both chains, and fit only a `960 -> 256 -> 128 -> 1` ReLU/dropout MLP per fold. Internal early stopping uses a stratified 10% subset of that fold's train rows only; no test rows select epochs. This is a frozen-backbone downstream head experiment, not continued foundation-model training and not an external-baseline retrain.
- The primary rank for that experiment is against official-checkpoint baseline reruns on the same released CSV bytes (`ranking_local_release_AS.csv`). Paper means are kept in a separate `ranking_paper_reference_AS.csv` marked mixed-source, because the released seen files do not reproduce several paper means exactly. Neither ranking replaces the canonical original-only T1 table.
- Tier 2, generation/infill: run small OTS/TCR CDR infilling and antibody heavy-to-light completion. Primary metrics should be AAR for known masked regions, novelty, nearest-neighbor distance, k-mer JSD, length validity, and invalid amino-acid rate.
- Tier 3, task-specific fine-tuning: only after Tier 1/2 pass, fine-tune small heads or lightweight task adapters for IMMREP23 binding, FLAb developability, antibody specificity, and PPI. This separates representation quality from generation quality.
- The main comparison table should always include `kmer`, `esm2_150m`, `esm2_650m` when feasible, `ophiuchus`, and `bioseq:/abs/path/checkpoint.pt`. Report both seen and unseen splits for TCR binding; unseen macro-AUC0.1/AUPRC is the primary signal.
- A checkpoint is worth keeping if it improves at least one of: held-out diffusion loss, T3 representation probe, T4 infill AAR/JSD, or antibody generation validity, without degrading unseen TCR binding below the baseline noise band.

## Latest Task Roadmap

- Priority 0: keep the current immune-fusion training path working through `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py`, `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data`, and the retained fusion/model-layer integration. The exact Ophiuchus-Ab path under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq` is historical/compatibility provenance, not a replacement current entry.
- Priority 1 antibody binding task: add AbBiBench-style antibody-antigen affinity scoring and CDR design evaluation. Inputs should include heavy chain, light chain, antigen chain(s), optional complex structure, mutant region, and experimental binding score. Metrics should include Spearman/Pearson correlation for scoring, top-k enrichment for ranking, and external complex-quality oracle scores.
- Priority 1 antibody developability task: add FLAb-style property prediction heads for expression, thermostability, immunogenicity, aggregation, polyreactivity, binding affinity, and pharmacokinetics. These heads are useful for filtering generated antibodies before structure oracle evaluation.
- Priority 1 TCR-pMHC binding task: add IMMREP-style TCR specificity prediction with paired alpha-beta TCR chains, peptide, MHC allele/class, and unseen-pHLA splits. Metrics should prioritize AUPRC under strict negative sampling, with separate seen-epitope and unseen-epitope reports.
- Priority 1 TCR-pMHC model reference: track DecoderTCR as an ESM2-family baseline and architectural reference for TCR-pMHC sequence modeling. Its tasks map directly to binding prediction, interaction scoring, and TCR sequence analysis.
- Priority 1 TCR-pMHC structure task: add structure-oracle evaluation for TCR-pMHC complexes using DockQ, RMSD, TM-score, and CDR3 pLDDT-style reranking signals. SCEptRe should be considered as a frequently updated source of nonredundant immune-complex benchmark splits.
- Priority 2 PPI task: add paired-sequence PPI training/evaluation inspired by PPLM-PPI. The first target should be binary interaction prediction; the second target should be quantitative affinity prediction; the third target should be residue-level interface/contact prediction.
- Priority 2 mutation-fitness task: add ProteinGym-style single-chain and paired-chain mutation effect scoring to test whether BioSeq likelihood/diffusion scores correlate with DMS fitness and clinical variant labels.
- Priority 2 general protein foundation evaluation: use PFMBench and ProteinBench as broad external evaluation suites rather than training targets; they are useful for deciding whether BioSeq is overfit to antibody/TCR/PPI tasks or remains a general sequence foundation model.
- Priority 2 de novo binder design task: use ESMFold2, BindCraft-style filtering, and later Proteina-Complexa-style benchmarks as oracle/evaluation references for generated minibinders, antibody-derived formats, and PPI binders. Do not block the sequence foundation path on full structure generation.
- Immediate schema implication: extend BioSeq examples beyond `chains` and `task_type` to allow optional `chain_roles`, `target_chain_indices`, `mutations`, `labels`, `assay_type`, `antigen_chains`, `mhc_allele`, `peptide`, `structure_path`, and `oracle_scores`.
- Immediate code implication: add future task adapters under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data` before downloading any new large dataset; each adapter should normalize external datasets into the current prepared semantic-record contract.

## Data Schema

- Canonical BioSeq JSONL schema version is `bioseq.v1`.
- Required fields: `chains`, `task_type`, and `source`.
- Optional but preferred fields: `chain_roles`, `targets`, `generation_spec`, `split`, `labels`, `regions`, and `metadata`.
- Current `chain_roles` vocabulary should include `antibody_heavy`, `antibody_light`, `nanobody_vhh`, `tcr_alpha`, `tcr_beta`, `peptide`, `mhc`, `antigen`, and `other`.
- `targets` is only a coarse default list of chain indices that can be generated. It is not sufficient for conditional tasks by itself.
- Fine-grained generation must be represented by `generation_spec` or by a sampled training view. This view resolves to token-level `visible_mask`, `fixed_context_mask`, `diffusion_target_mask`, and `diffusion_loss_mask`.
- `generation_spec` should support chain-level completion, region-level infilling, span-level infilling, inverse region infilling, and conditional receptor generation. Examples: heavy-to-light generation, antigen-to-antibody/nanobody generation, heavy+antigen-to-light generation, alpha+beta+MHC-to-peptide design, pMHC-to-alpha+beta design, FR-conditioned CDR infilling, single-CDR infilling, and CDR-conditioned FR generation.
- `regions` should be keyed by string chain index and can store `FR1`, `CDR1`, `FR2`, `CDR2`, `FR3`, `CDR3`, and `FR4` for antibody/TCR CDR infilling.
- Current adapters, records, preparation, grammar, and collation live under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data`. The former BioSeq conversion modules are historical and are not current runtime dependencies.
- Supported source types today: `oas`, `ots`, `nanobody`, and existing `processed` JSONL.
- Full conversion of OAS/OTS/nanobody should not be run until output root, shard size, and train mixture weights are decided. Use `--limit` for small conversion checks.

## Historical PPI and Interaction Data（not a current foundation-training entry）

- Raw interaction-task downloads live under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw`.
- The rebuild script is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/build_ppi_interaction_csv.py`.
- Historical processed outputs retained for audit:
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/processed/interaction_sources_manifest.csv`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/processed/interaction_records_summary.csv`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/processed/interaction_records_unified.csv`
- The unified records CSV is a row-level integration table, not the final training format. It has `6271559` records and is useful for auditing, filtering, and writing task-specific `bioseq.v1` shards.
- Supported row-level sources in the current CSV: Figshare gold-standard PPI, HumanPPI LMDB, YeastPPI LMDB, SKEMPI, SWING MutInt, FLAb binding, TDC TCR-epitope, PISTE TCR-epitope-HLA, TEIM binding/interface metadata, oncoPPI spreadsheets, and CoV-AbDab neutralization.
- STRING-DB v12.0 was a separate large pretraining source in the historical row-level CSV plan. Its raw physical links and sequence dumps were too large for that pass; any future use requires an approved adapter under the current prepared semantic-record contract, not the deleted PPI/Arrow training path.
- PDBbind+ and the referenced bioRxiv SARS-CoV-2 binding supplement are blocked in this environment by login/subscription and HTTP 403, respectively. They should not be treated as available training sources until access is resolved.
- These CSV/builder notes are historical planning evidence only. The deleted PPI/STRING/MINT builders are not current training dependencies; future interaction work must first add an approved adapter and prepared semantic-record contract under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data`.

## Training Logic

- The foundation-model objective should stay diffusion-only by default: corrupt eligible target residues at timestep `t`, predict the clean tokens, and compute denoising cross-entropy only on target/remasked positions.
- Every training example should carry explicit masks: `diffusion_target_mask` for residues that may be noised/remasked and receive diffusion loss, and `fixed_context_mask` for residues that remain clean and visible as conditioning context.
- Training should separate biological examples from target-mask construction. The prepared semantic record contains the full clean chains plus regions/metadata; the current immune grammar fixes type markers, relation/context objects, and conditioning chains, then creates token-level masks so eligible generated tokens participate in the diffusion objective.
- The current training-time data path lives under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data`. It reads prepared semantic JSONL shards, renders the current immune grammar, and builds per-chain encoder tensors (`encoder_input_ids [batch, max_chains, chain_len]`). The deleted qwen data alias and its historical Arrow path, plus `GRAMMAR_V1.md`, are not active documentation.
- The BioSeq foundation model layer now starts at `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py`. It exposes `BioSeqNoEncoderDiffusionModel` for no-encoder training and `BioSeqEncoderDiffusionModel` for ESMC/ESM feature-conditioned training.
- The retained no-encoder model-layer design is a dense bidirectional diffusion transformer with ESM-family token embeddings, chain-local residue position embeddings (`position_ids_inner`), outer chain-index embeddings (`position_ids_chain`), timestep embeddings, RMSNorm, and SwiGLU blocks. The current immune grammar expresses entity roles through boundary/context tokens, so the decoder does not instantiate chain-role or task-type embeddings. It uses bidirectional self-attention for masked diffusion, not a causal/autoregressive language-model architecture. This model-layer design is separate from the deleted lightweight legacy backend.
- `BioSeqNoEncoderDiffusionModel.compute_loss` samples timestep noise from `diffusion_loss_mask`, replaces corrupted target residues with `<mask>`, predicts clean residue ids, and computes denoising cross-entropy only on corrupted target positions.
- `BioSeqEncoderDiffusionModel.compute_loss` uses the same diffusion objective and builds a per-chain ESMC/ESM `x_t` by applying the decoder corruption state to `encoder_input_ids`. ESMC/ESM returns token-level features for that diffusion state, and the downstream BioSeq denoiser performs joint multi-chain denoising over the concatenated decoder stream.
- Local ESMC encoder loading is implemented through `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py::load_local_esmc_encoder`. Because `transformers==4.48.1` does not recognize `model_type="esmc"`, `BioSeqEncoderDiffusionModel.from_esmc(...)` tries `AutoModel.from_pretrained(...)` first and then falls back to Biohub `esm==3.2.3`, mapping local Hugging Face-style safetensor keys into native `esm.models.esmc.ESMC` keys.
- ESM2 is also supported as the conditioning encoder via `--model-type esm2` / `BioSeqEncoderDiffusionModel.from_hf_encoder(...)`, which loads a local ESM2 snapshot under `/c20250601/mj/model_weights/esm2/*` through HF `EsmModel`. ESM2 hidden sizes: 8M=320, 35M=480, 150M=640, 650M=1280, 3B=2560.
- Current training uses the prepared-dataset loader and `GrammarBioSeqCollator` under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data`; batch assembly remains dynamic because grammar rendering, padding, encoder reconstruction, and diffusion/MLM masking depend on the current batch/state.
- The formal training entry is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py`. Deleted DDP examples, deleted Arrow sources, and deleted qwen `training` modules are not runnable alternatives.
- Foundation pretraining keeps the objective simple: the current immune grammar fixes type markers, relation tokens, and antigen / peptide / MHC-HLA context objects, and treats target structure tokens (including `<prots>`, `.`, `<protd>`) and target residues as diffusion targets. Antigen-conditioned receptor blocks include `<ab>` or `<nb>` inside `<prots>` to distinguish antibody vs nanobody design. Conditional capabilities such as chain completion, antigen-conditioned receptor generation, peptide design, and FR/CDR infilling are expressed as inference-time partial-mask prompts over the same current grammar, not as a separate deleted runtime view sampler.
- TCR grammar role resolution is role-first in the active renderer at `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/grammar.py`: an explicitly tagged single `tcr_alpha` or `tcr_beta` is never positionally paired with a peptide, antigen, or MHC context chain. The regression coverage is retained under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada`.
- To reduce multi-task imbalance, source mixture weights and optional future task schedulers should control the task distribution. The default physical batch itself is intentionally mixed rather than task-homogeneous.
- Fixed context chains, such as antigen in antibody-antigen generation, should not be remasked and should not receive direct diffusion loss. They still participate in attention or encoder conditioning, and gradients should flow through their encoder/connector parameters from the target-chain diffusion loss.
- In the BioSeq foundation loader, `full_denoise` means denoising all eligible target chains, not every chain unconditionally. Explicit `metadata["targets"]` is honored when present, but antigen, peptide, MHC, and HLA-like chains are fixed context by default and do not receive diffusion loss.
- Antibody-antigen and nanobody-antigen views should follow the receptor-design direction used by current antibody design work: fixed antigen can generate heavy/light antibody chains or VHH, fixed antigen plus one antibody chain can generate the paired antibody chain, and fixed antigen plus receptor FR regions can generate CDR regions. Antibody/nanobody-to-antigen inverse views are not part of the default task set.
- The `mhc_to_peptide_tcr` view covers the complementary TCR-pMHC case where MHC/HLA is clean fixed context and peptide plus available TCR chains are diffusion targets. This should be sampled explicitly during pretraining if peptide/TCR co-design or peptide-conditioned receptor learning is expected at inference.
- The `tcr_mhc_to_peptide` view covers peptide design from fixed TCR alpha/beta plus MHC/HLA. The `pmhc_to_tcr` view covers TCR alpha/beta design from fixed peptide plus MHC/HLA.
- FR/CDR views are region-driven, not antibody-only. They apply to TCR full-chain data whenever source adapters preserve TCR FR/CDR region annotations. For TCR-pMHC, fixed peptide/epitope plus MHC/HLA can be combined with fixed TCR FR regions to generate TCR CDR regions.
- ESMC/ESM feature-conditioned training is still diffusion training: ESMC/ESM2 features are extracted from `x_t` and condition target denoising, but the encoder is updated through the target diffusion loss rather than through a separate antigen reconstruction loss.
- The ESMC/ESM feature-conditioned path defaults to trainable encoder parameters. `freeze_encoder=True` is available only for ablations or bootstrap checks; the main foundation-model setting should fine-tune the ESMC/ESM encoder with the diffusion loss.
- Encoder conditioning **replaces** decoder residue embeddings with gathered encoder features (`use_condition_projection=False` by default). `decoder.hidden_size` is forced to match encoder latent dim for `--model-type encoder|esm2`; no-encoder ablations can use `--align-hidden-size-to-encoder`.
- The ESMC/ESM feature-conditioned path encodes each chain/sequence independently through ESMC on the current per-chain `x_t`. Implementation detail: `encoder_input_ids` has shape `[batch, max_chains, chain_len]`, is flattened to `[batch * max_chains, chain_len]` for a batched ESMC call, then reshaped to `[batch, max_chains, chain_len, hidden]`; residue features are gathered back to decoder token positions and replace residue embeddings before the multi-chain BioSeq denoiser runs.
- The current local environment keeps `transformers==4.48.1` for compatibility with Biohub `esm==3.2.3`; ESMC loading must use `BioSeqEncoderDiffusionModel.from_esmc(...)` or `load_local_esmc_encoder(...)` rather than relying on `AutoModel` alone.
- ESMC tokenizer loading for the active fusion path is implemented at `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/esm_encoding.py`; do not refer to the deleted qwen `data/esm_encoding.py` alias.
- Remask schedules, block-diffusion masks, and generation masks must operate on `diffusion_target_mask`, not on all non-pad tokens.
- The grammar renderer guarantees at least one diffusion-eligible token per record (the non-fixed residues/structure/relation tokens), so `max_sequence_length` truncation cannot produce a zero-loss microbatch the way the old view sampler could.
- The current formal foundation-training entry is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py`. Historical `qwen3_vl_bioseq_*` YAML names and grammar-v2 runs may remain in audit records, but are not current submit instructions. Any new comparable ESMC/no-encoder experiment must be defined against the current immune-fusion line and recorded in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md` before submission.
- Batch-size tuning should be driven by logged CUDA peak memory, not by guessed limits. Keep effective batch approximately fixed while changing per-GPU batch and gradient accumulation. The old 8-GPU stage-1 settings (`8 x grad_accum 2`, `2 x grad_accum 8`, and `1 x grad_accum 16`) are historical baselines only; current settings must be profiled on the immune-fusion line before submission.

## Generation Task Survey and View Priority

- Survey date: 2026-06-14.
- Antibody generation should prioritize receptor-side design, not antigen generation. DiffAb samples antibody CDRs for antibody-antigen complexes and antigen-only settings, RFdiffusion antibody design explicitly focuses sampling on CDR loops while keeping a framework close to a specified therapeutic scaffold, IgLM supports full antibody generation and variable-span/CDR infilling, and paired antibody language models show that heavy-light pairing helps cross-chain feature learning.
- Current antibody/nanobody high-priority views are therefore: `antigen_to_antibody`, `antigen_to_nanobody`, `heavy_antigen_to_light`, `light_antigen_to_heavy`, `antigen_fr_to_cdr`, `antigen_single_cdr`, `heavy_to_light`, `light_to_heavy`, `fr_to_cdr`, and `single_cdr`. `cdr_to_fr` is useful for humanization/framework-design style ablations, but should not dominate early pretraining. `antibody_to_antigen` and `nanobody_to_antigen` are removed from the default task set.
- TCR generation should prioritize pMHC/epitope-conditioned receptor design and paired-chain completion. TCR-TRANSLATE generates antigen-specific TCR sequences from unseen pMHC inputs, TCRdesign uses antigen-conditioned generation with paired-chain coherence, and recent TCR structure work argues that both alpha and beta chain information matter for specificity and structure.
- Current TCR high-priority views are therefore: `pmhc_to_tcr`, `pmhc_fr_to_cdr`, `pmhc_single_cdr`, beta-chain context to alpha-chain completion, alpha-chain context to beta-chain completion, `fr_to_cdr`, and `single_cdr` when TCR FR/CDR annotations exist. The existing names `beta_epitope_to_alpha` and `alpha_epitope_to_beta` should be treated as context-to-chain completion views; a later cleanup should either rename them to `beta_context_to_alpha` / `alpha_context_to_beta` or add strict epitope-required variants.
- TCR peptide/epitope design views are biologically useful but should be lower-weight or task-specific until evaluation data is stronger: `tcr_mhc_to_peptide` is appropriate for epitope discovery/design from fixed TCR+MHC, while `mhc_to_peptide_tcr` is a co-denoising/co-design view rather than the first downstream validation target.
- The runtime view sampler has been removed: pretraining no longer samples conditional views per record. The conditioning directions above (antibody, antibody-antigen, nanobody-antigen, TCR, TCR-epitope, TCR-pMHC, PPI, generic) are instead realized as inference-time partial-mask prompts over the grammar token stream, where the prompt fixes the given context tokens and leaves the target spans masked.
- Sources: DiffAb (`https://github.com/luost26/diffab`, `https://openreview.net/forum?id=jSorGn2Tjg`), RFdiffusion antibody design (`https://www.nature.com/articles/s41586-025-09721-5`), IgLM (`https://doi.org/10.1016/j.cels.2023.10.001`), paired antibody language models (`https://doi.org/10.1371/journal.pcbi.1012646`), TCR-TRANSLATE (`https://www.nature.com/articles/s42256-025-01096-6`), TCRdesign (`https://doi.org/10.1093/bib/bbaf691`), and paired TCR alpha/beta structure analysis (`https://www.nature.com/articles/s42003-025-07708-6`).

- Ophiuchus-Ab training uses masked diffusion over heavy/light antibody tokens with AirGen-compatible timestep corruption.
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/ophiuchus/collator.py` emits `heavy_tokens` and `light_tokens`, each with `targets`, `regions`, and `chain_ids`, plus per-example `weights`.
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/ophiuchus/training.py` computes the heavy and light losses through `MultiChainOphiuchusAbModel.compute_loss`.
- Next migration quality step: add parity checks against `/vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/src/byprot/models/lm/dplm_multichain.py` for `construct_x_t`, `compute_loss`, and generation mask transitions. The MINT module files themselves already match AirGen byte-for-byte except generated `__pycache__`.
- BioSeq training code must not import the old `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/core/trainers` implementation.
- Historical Ophiuchus-Ab compatibility code, when independently audited, must remain limited to the exact Ophiuchus-Ab path; it must not be treated as the current immune-fusion entry. The deleted/lightweight `bioseq` backend, synthetic default data, and generic `transformers.Trainer` path are not current alternatives.
- Variable-length training (no fixed `(150, 128)` padding) uses `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/ophiuchus/collator.py::MultiChainDynamicCollator`, which pads each batch to the longest chain-1/chain-2 sequence with the real `<pad>` id. This is correct because the mint ESM2 backbone derives its padding mask from `tokens.eq(<pad>)`, and `compute_loss` splits logits by per-slot length. `OphiuchusAbTrainingCollator` (fixed length) is kept for exact-length reproductions.
- Multi-dataset prepared training uses `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/dataset.py`; source adapters and offline preprocessing use `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/sources.py`. OAS heavy is oriented to slot 0, OTS beta to slot 0, and prepared records preserve canonical chain roles.
- Historical Ophiuchus-Ab DDP validation used the former `examples/bioseq/train_bioseq_ddp.py` entry point; that entry point is now retired and deleted. New immune ESMC + LLaDA training uses `examples/llada/protein_pretrain_esmc.py` and the prepared `immune_llada` loader.
- Historical validation evidence included a single-process CPU smoke run and a 2-process `torchrun` (gloo) DDP smoke run loading all three datasets at variable length. This is not a fresh validation claim for the current documentation pass; current validation remains pending the main-agent TODO in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md`.
- Initializing from generic ESM2 base weights is not part of the current antibody path. If it becomes necessary later, use the local ESM2 snapshots under `/c20250601/mj/model_weights/esm2` rather than adding a Hugging Face download dependency.

## Inference Logic

- The exact Ophiuchus-Ab inference path is retained as historical/independent compatibility provenance through `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/sample_ab.py` and aligned downstream scripts under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream`; it is not the current immune-fusion training entry.
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/sample_ab.py` loads `/c20250601/mj/model_weights/ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt` through `MultiChainOphiuchusAbModel.from_checkpoint` when that historical path is explicitly audited.
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/common.py` is the shared inference layer for those aligned historical scripts; the current fusion evaluation boundary is the retained scripts under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream` and public implementations under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar`.
- Heavy-to-light generation should fix the heavy chain and mask the light chain positions after any provided light-chain prompt. Empty light-chain input must still create masked light positions, not a fixed empty light chain.
- Antibody inference must not use `NoEncoderBioDiffusionModel`, `BioSeqDiffusionTrainer`, or any lightweight generic BioSeq backend.

## Weight Plan

- ESMC weights:
  - `/c20250601/mj/model_weights/esmc/ESMC-300M`
  - `/c20250601/mj/model_weights/esmc/ESMC-600M`
  - `/c20250601/mj/model_weights/esmc/ESMC-6B`
- ESM2 weights:
  - `/c20250601/mj/model_weights/esm2/esm2_t6_8M_UR50D`
  - `/c20250601/mj/model_weights/esm2/esm2_t12_35M_UR50D`
  - `/c20250601/mj/model_weights/esm2/esm2_t30_150M_UR50D`
  - `/c20250601/mj/model_weights/esm2/esm2_t33_650M_UR50D`
  - `/c20250601/mj/model_weights/esm2/esm2_t36_3B_UR50D`
- ESM2 load checks should cover 150M, 650M, and 3B.
- ESM2 8M/35M should be downloaded but not included in the default load checks.
- ESM2 15B is optional and should not be downloaded in the current default task.
- Downloaded files should include Hugging Face/PyTorch-compatible weights, configs, tokenizer files, README files, and remote-code files; TensorFlow `.h5` duplicates are not required.
- Ophiuchus-Ab checkpoint is stored at `/c20250601/mj/model_weights/ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt` and comes from `https://zenodo.org/records/18478480`.

## LLaDA Backbone Option (survey 2026-07-02)

- Goal: after the biological encoder (ESMC/ESM2), swap the decoder from the in-house `BioSeqDiffusionDecoder` (`/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py`) to the **LLaDA architecture** as a comparison backbone. The encoder + per-chain-encode + gather pipeline stays unchanged; only the denoiser transformer changes.
- In-repo implementation: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/llada/models/modeling_llada.py` (`LLaDAModelLM` / `LLaDAModel`, HF `PreTrainedModel`-compatible; config in `configuration_llada.py`).
- Same nature as the current decoder: LLaDA is a **bidirectional masked-diffusion transformer** (`is_causal=False`, `get_bidirectional_attention_bias`), LLaMA-style (RMSNorm / RoPE / SwiGLU). This matches the existing bidirectional masked-diffusion decoder, so it is a backbone swap, not an objective change.
- Integration point (reuses existing encoder path): `LLaDAModelLM.forward` natively accepts `inputs_embeds` (forwarded as `LLaDAModel.forward(input_embeddings=...)`, `x = wte(input_ids) if input_embeddings is None else input_embeddings`). Wiring:
  - encoder per-chain encode -> `gather_token_condition` -> `token_condition [B, S, E]` (unchanged);
  - build `inputs_embeds [B, S, d_model]`: residue positions = encoder features, non-residue (structure/relation/special) positions = `llada.wte(x_t)`;
  - `LLaDAModelLM(inputs_embeds=..., attention_mask=...)` -> logits -> `compute_masked_cross_entropy` on corrupted positions only.
  - Requires `E == d_model` for pure replacement, or add a condition projection.
- Key differences vs the in-house decoder (design decisions to lock before training):
  - **Position**: LLaDA uses **RoPE on flat sequence positions**; the in-house decoder uses **learned absolute** `position_ids_inner` (chain-local) + `position_ids_chain` (chain slot). The first LLaDA variant loses chain-aware absolute positions and relies on flat RoPE + grammar boundary tokens; chain-aware / group-reset RoPE is a later enhancement.
  - **Timestep**: LLaDA does **not** take `t` as input (RADD proves masked diffusion needs no timestep). The in-house decoder now replaces only the token-identity embedding at residue sites and **adds** inner-position / chain-slot / timestep on top; when moving to LLaDA the timestep addition can simply be dropped (RoPE supplies position).
  - **Loss**: `LLaDAModelLM.forward` does not compute loss (labels only warn); keep the external `compute_masked_cross_entropy` (masked positions only).
  - **Vocab**: LLaDA defaults to a text vocab (`vocab_size=50257`, `mask_token_id=50256`, `embedding_size=50304`). Using the architecture from scratch requires reconfiguring `vocab_size` / `mask_token_id` / `embedding_size` to the grammar vocab (ESM2=49 / ESMC=80).
  - **`input_emb_norm`**: LLaDA can scale input embeddings by `sqrt(d_model)`, which rescales the encoder condition (related to the `condition_norm` scale discussion). Keep it fixed/known across A/B runs.
- Config constraints: MDM usage asserts `rope=True`, `alibi=False`, and `use_cache=False` in `LLaDAModel.forward`.
- Weights: `/c20250601/mj/model_weights` currently has **no** LLaDA checkpoint. From-scratch training with the LLaDA architecture needs no weights. Loading pretrained LLaDA text weights is not directly reusable — the text vocab and residue embeddings do not match the biological grammar vocab.
- Open decisions: keep chain-aware positions (grouped RoPE) or accept flat RoPE; whether to re-inject timestep at residue sites; whether to add a condition projection (`use_condition_projection`) instead of pure replacement.

## 2026-07-21 Public CDR3β generation comparison architecture

- Track A is a model-output benchmark, not an architecture comparison. The common adapter contract is `generate(peptide, mhc_allele=None, mhc_pseudosequence=None, num_candidates=1000)` and the canonical row schema is defined in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/track_a.py`.
- TCRT5 conditions on peptide plus the 34-aa MHC pseudo-sequence; GRATCR and TcrDesign condition on epitope. TcrDesign Track-A ranking rows are invoked only through `tcrdesign_G.py -mode beta`; V/J prediction, TcrDesign-B, and the full binding-filter pipeline are not part of the first-stage comparison.
- After the β benchmark, a non-ranking TcrDesign addendum runs the released conditional chain `epitope -> generated β; (epitope, generated β) -> generated α`. It covers all 14×1,000 generated β conditions with 10 official α beams each. The batched adapter preserves the released architecture/checkpoint/tokenization/beam search and only avoids repeated model loads. Since the release contains no runnable alpha-reference or αβ pairing evaluator/checkpoint, the 140,000 paired rows are qualitative and no custom alpha metric is added.
- TCR-epiDiff is guarded as `reconstruction_only` because its released sampler starts from a noised real TCR. A reconstruction path must never be adapted with test references and presented as de-novo generation.
- The local BioSeq comparison is frozen to the only retained 7-layer checkpoint in the requested `11xxxx` range, `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_esmc300m_integrated_llada_7l_step117000/best.pt` (validation loss `0.4430236165889635`, SHA256 `e3d4c98ea84b4aa2280fe76ec5bf8a806de376699610e9a96960e66ceddb5c01`). It is also the retained Top-K checkpoint with the lowest recorded validation loss.
- BioSeq Track-A generation is epitope-only and does not consume MHC allele or pseudo-sequence. The fixed reproducibility protocol is seed `42`, batch size `100`, `32` denoising iterations, temperature `1.0`, and `gumbel_argmax`; rank is deterministic seeded emission order and `raw_score` remains empty because this sampler does not expose a comparable candidate score. Raw sequences receive no anchor repair.
- Under that protocol, BioSeq produced 14,000 candidates with `ValidRate=0.998786`, `UniqueRate=0.999714`, no exact or Recovery>=90% hits, one GIANA reference-cluster hit, and mean cross-epitope Jaccard `0.000044`. This indicates strong syntax/diversity but weak reference-neighborhood recovery under the current epitope-only generation protocol; it is not evidence by itself about binding or the architecture in general. The focused audit is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/bioseq_step117000_report.md`.
- All models are evaluated after generation by one external validity/exact/recovery/GIANA/Jaccard implementation. Internal model scores may be retained as `raw_score` but cannot determine the cross-model result.
- TCRT5 is the one rank-sensitive adapter: rank and `raw_score` must use cumulative generated-token log-likelihood (`compute_transition_scores(...).sum()`), matching the paper's rank-cutoff protocol, rather than Hugging Face's length-normalized beam score. The candidate set is independent of this bookkeeping correction.
- Keep the paper-native TCRT5 recovery diagnostic separate from the common Track-A metric. The paper first chooses a same-length reference and uses position identity/Hamming (with an edit-distance fallback); the common leaderboard remains the user-specified normalized Levenshtein definition. Never compare their counts without a definition label.
- The complete protocol and completed local result are frozen in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/TCR_BETA_PUBLIC_TRACK_A.md`.

## 2026-07-22 MINT headline PPI evaluation lock

- The user-selected model for the formal HumanPPI, YeastPPI, and Gold-standard PPI evaluation is the immutable BioSeq `step121000` snapshot at `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_esmc300m_integrated_llada_7l_step121000/best.pt` (SHA256 `51f0eee5fa7b127a0487a2ea8fdbc322bda6f6908ff2662087bd87d4980d4613`). Later/final checkpoints are not substituted.
- Only the three requested classification tasks are in this run: HumanPPI (primary metric Accuracy), YeastPPI (Accuracy), and registry task `Bernett`, which is the MINT paper's Gold-standard PPI task (AUPRC). MutationalPPI and SKEMPI are explicitly outside this run.
- Data come from the pinned public MINT `prepare_data.ipynb` rebuild under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/mint_official`, with no row cap. HumanPPI uses 26,319/234/180 train/validation/test rows and YeastPPI uses 4,945/95/394; both exactly match the paper counts. Gold-standard uses 163,192/59,260/52,048; its validation/test counts match, but the public notebook has 173 more training rows than the paper's 163,019, so its local result must be marked non-paper-exact.
- Each pair is rendered as one contextual two-chain `ppi` grammar record and pooled by one global mean over final decoder residue states; `--sep_chains` is not used. Each chain is deterministically left-truncated to the BioSeq native maximum of 1,024 residues. This differs from MINT baseline runs that can use a 2,048-token cap and must remain explicit in provenance.
- The downstream probe follows the paper Methods rather than the inconsistent public task defaults: two linear layers with hidden width 640, ReLU, dropout 0.2, AdamW at `1e-4`, 100 epochs for all three tasks, best validation metric selection, batch size 16, and three seeded repetitions. The public code's input-width hidden layer and Gold-standard 30-epoch setting are not used for the headline comparison because they conflict with the paper's stated 640/100 protocol.
- Before formal submission, `/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_mint_ours_step121000_three_ppi_profile.yml` profiles worst-case 1,024+1,024-residue pairs on the same `ml.pni2.3xlarge` A100 flavor. Formal extraction batch size is selected from that measured profile and recorded in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md`.
- The formal run is complete. BioSeq `step121000` `[C]` obtains HumanPPI Accuracy `0.677778±0.018144`, YeastPPI Accuracy `0.602369±0.007846`, and Gold-standard PPI AUPRC `0.592089±0.001405`. The best paper `[P]` values are MINT `0.879630±0.006929`, `0.686971±0.010634`, and `0.687157±0.003256`, giving gaps `-0.201852/-0.084602/-0.095068`. Human/Yeast use exact public fixed splits; Gold remains `paper_comparable=false` because the public notebook has +173 training rows. The complete report is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/mint_tasks/_ours_step121000_official3ppi/REPORT.md`.

## 2026-07-22 Retained-checkpoint CDR3β diagnostic policy and step189000 result

- The canonical all-14 BioSeq Track-A row remains the frozen `step117000` run. Later retained checkpoints are evaluated under an isolated `checkpoint_comparison/<run_name>` root and must not be appended to the canonical leaderboard unless a complete all-target replacement run is explicitly selected.
- The BioSeq adapter now validates the checkpoint manifest and derives `BioSeq-7L-step<manifest_step>` dynamically. Its default checkpoint and default label remain step117000, so existing reproduction commands and canonical rows are unchanged.
- A checkpoint comparison must keep the same generation controls (seed `42`, batch `100`, 32 iterations, temperature `1.0`, `gumbel_argmax`, no anchor repair) and the same external Track-A evaluator. For a one-target diagnostic, cross-epitope Jaccard is undefined; candidate-set overlap across checkpoints may be reported separately and must not be labeled cross-epitope specificity.
- Under this policy, step189000 generated 1,000 RVR candidates: ValidRate=`0.998`, UniqueRate=`1.000`, ExactHit@1000=`0`, ReferenceRecall=`0`, Recovery90Hit=`0`, median best recovery=`0.600000`, maximum recovery=`0.800000`, and GIANAHit=`0`. Relative to step117000, median recovery improved only from `0.588235`; this is not a meaningful target-specificity improvement.
- This run remains an epitope-only conditional generation test. A future MHC-aware experiment requires a training-consistent MHC conditioning representation and is a separate model/data change, not an inference-time field injection into this checkpoint.
- The complete focused artifact is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/checkpoint_comparison/bioseq_step189000_rvr_epitope_only/report.md`.

## 2026-07-23 TCRT5 full-evaluation method lock

- The paper-facing evaluator is now a separate evidence layer, implemented in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/tcrt5_full_eval.py`. It does not replace the common Track-A leaderboard: author top-20 `K=100`, sparse13/RVR `K=1000`, and current all-14 `K=1000` remain distinct protocols.
- Paper-native recovery is a two-stage operation: select the closest same-length reference by Levenshtein distance, then compute positional Hamming identity to that selected sequence. Only the no-same-length branch uses reference-length-normalized Levenshtein identity. Track-A normalized-Levenshtein recovery remains a separate metric and must keep its definition label.
- The paper's mAP is implemented literally as the pMHC macro mean of `mean(P@1,…,P@K)` after cumulative model-log-likelihood ranking. A value is paper-compatible only if the candidate block retains those scores. Current TCRT5 satisfies this condition; author main-table lists, BioSeq, GRATCR and TcrDesign do not, so their ordered AP values are diagnostics rather than calibrated cross-model mAP.
- Diversity decisions are fixed to natural-log units: positional delta entropy is `H(generated)-H(reference)`, with gap-inclusive and residue-only tables both retained; k-mer Jensen–Shannon divergence uses natural logarithms for `k=2..12`, so its upper bound is `ln(2)`.
- Evidence is three-tiered: (1) independently recomputed and checked against released sequences; (2) definition-complete but blocked from exact paper comparison by missing greedy hypotheses, likelihoods or final figure aggregation; (3) explicit extensions such as Hit@K/rank distributions. Reports must never collapse these tiers into a single “reproduced” label.
- OLGA biological plausibility requires both the positive fraction and the positive `log10 Pgen` distribution. In the current all-14 run, BioSeq-7L-step117000 has `0.979786` positive Pgen but mean positive log10 Pgen `-17.925285`, compared with TCRT5 `1.0/-6.950375`; therefore a non-zero-only validity claim is insufficient.
- The canonical comparison report is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcrt5_full_eval/TCRT5_FULL_EVAL_REPORT.md`. Its explicit non-reproducible boundaries—main-table mAP and Fig.4 Pgen moments—are part of the method contract, not missing implementation work.
