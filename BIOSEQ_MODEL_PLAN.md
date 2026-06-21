# BioSeq Model Plan

## Goal

Build a diffusion-model-based biosequence foundation model in `/vepfs-mlp2/c20250601/251105016/project/dllm_test` for antibody, antigen, TCR, TCR-pMHC, and PPI sequence tasks.

## Code Location

- Main pipeline: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq`
- Examples: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq`
- Tests: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq`
- Weight root: `/c20250601/mj/model_weights`

## Architecture

- The antibody training target is the exact Ophiuchus-Ab path in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/ophiuchus`.
- `BioSeqEncoderDiffusionModel` is the ESMC/ESM feature-conditioned target.
- The no-encoder antibody model uses the migrated Ophiuchus-Ab ESM2 transformer with chain-aware multimer attention.
- The ESMC/ESM feature-conditioned model runs the biological encoder on the current diffusion state `x_t`, projects token-level encoder states into the diffusion decoder, and treats ESMC freezing only as a bootstrap or ablation setting.
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
- Adapter layer: source-specific readers normalize raw records into `bioseq.v1` JSONL through `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/adapters.py` and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/tools/convert_data.py`.
- Canonical data layer: all training rows should use `chains`, `task_type`, `source`, optional `chain_roles`, `targets`, `regions`, `labels`, and `metadata`. Bulk unpaired TCR rows from TCRdb2.0 should become single-chain or beta-chain records first, not multi-chain pseudo-pairs.
- Mixture layer: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/datasets.py` should own sampling weights and caps. Paired immune-receptor sources, such as OAS heavy-light and OTS alpha-beta, should stay high-priority; bulk/unpaired TCR should be capped so it does not dominate multi-chain learning.
- Collation/masking layer: collators should emit token tensors plus `diffusion_target_mask`, `fixed_context_mask`, chain ids, chain-internal positions, and outer chain indices. Diffusion corruption and loss must operate only on eligible target positions.
- Model layer v1: the current no-encoder path should remain the exact Ophiuchus-Ab/MINT multichain stack under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/ophiuchus`, trained with diffusion loss through `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_bioseq_ddp.py`.
- Model layer v2: the ESMC/ESM2 feature-conditioned path should extract features from the current diffusion state `x_t`, then let the BioSeq denoiser perform multi-chain denoising over the concatenated token stream. Fixed context chains, such as antigen in antibody-antigen generation, remain clean and do not receive direct reconstruction loss.
- Checkpoint layer: training checkpoints should be written under absolute output roots, save `latest.pt` and `final.pt`, and preserve `backbone_state_dict`, optimizer state, step, epoch, and args for resume and downstream evaluation.
- Evaluation layer: downstream evaluation should use `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark` as the first integration point. The benchmark already accepts `--embedder bioseq:/abs/path/final.pt` through `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/common/model_api.py`.

## Fast Downstream Validation Plan

- Validation should run on every meaningful checkpoint, not only the final checkpoint. Minimal cadence: Ophiuchus-Ab init, early `latest.pt`, mid-training `latest.pt`, and `final.pt`.
- Tier 0, pretraining sanity: held-out diffusion loss on small OAS/OTS/nanobody/TCRdb2.0 slices, amino-acid distribution checks, valid-token rate, duplicate/near-neighbor rate against train, and chain-length distribution drift. This is the fastest failure detector for bad adapters or masks.
- Tier 1, embedding-only IRBench: run the existing benchmark with frozen embeddings and cheap heads. Priority commands are T1 TCR binding, T3 TCR representation, P1 PPI, and NbBench scalar tasks under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark`. This checks whether the trained backbone representation is useful without generation noise.
- Tier 2, generation/infill: run small OTS/TCR CDR infilling and antibody heavy-to-light completion. Primary metrics should be AAR for known masked regions, novelty, nearest-neighbor distance, k-mer JSD, length validity, and invalid amino-acid rate.
- Tier 3, task-specific fine-tuning: only after Tier 1/2 pass, fine-tune small heads or lightweight task adapters for IMMREP23 binding, FLAb developability, antibody specificity, and PPI. This separates representation quality from generation quality.
- The main comparison table should always include `kmer`, `esm2_150m`, `esm2_650m` when feasible, `ophiuchus`, and `bioseq:/abs/path/checkpoint.pt`. Report both seen and unseen splits for TCR binding; unseen macro-AUC0.1/AUPRC is the primary signal.
- A checkpoint is worth keeping if it improves at least one of: held-out diffusion loss, T3 representation probe, T4 infill AAR/JSD, or antibody generation validity, without degrading unseen TCR binding below the baseline noise band.

## Latest Task Roadmap

- Priority 0: keep the exact Ophiuchus-Ab training and inference path working through `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_ab.py`, `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/sample_ab.py`, and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq`.
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
- Immediate code implication: add task adapters under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq` before downloading any new large dataset; each adapter should normalize external datasets into the same JSONL schema.

## Data Schema

- Canonical BioSeq JSONL schema version is `bioseq.v1`.
- Required fields: `chains`, `task_type`, and `source`.
- Optional but preferred fields: `chain_roles`, `targets`, `generation_spec`, `split`, `labels`, `regions`, and `metadata`.
- Current `chain_roles` vocabulary should include `antibody_heavy`, `antibody_light`, `nanobody_vhh`, `tcr_alpha`, `tcr_beta`, `peptide`, `mhc`, `antigen`, and `other`.
- `targets` is only a coarse default list of chain indices that can be generated. It is not sufficient for conditional tasks by itself.
- Fine-grained generation must be represented by `generation_spec` or by a sampled training view. This view resolves to token-level `visible_mask`, `fixed_context_mask`, `diffusion_target_mask`, and `diffusion_loss_mask`.
- `generation_spec` should support chain-level completion, region-level infilling, span-level infilling, inverse region infilling, and conditional receptor generation. Examples: heavy-to-light generation, antigen-to-antibody/nanobody generation, heavy+antigen-to-light generation, alpha+beta+MHC-to-peptide design, pMHC-to-alpha+beta design, FR-conditioned CDR infilling, single-CDR infilling, and CDR-conditioned FR generation.
- `regions` should be keyed by string chain index and can store `FR1`, `CDR1`, `FR2`, `CDR2`, `FR3`, `CDR3`, and `FR4` for antibody/TCR CDR infilling.
- Initial adapters live at `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/adapters.py`.
- Conversion CLI lives at `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/tools/convert_data.py` and requires absolute input/output paths.
- Supported source types today: `oas`, `ots`, `nanobody`, and existing `processed` JSONL.
- Full conversion of OAS/OTS/nanobody should not be run until output root, shard size, and train mixture weights are decided. Use `--limit` for small conversion checks.

## PPI and Interaction Data

- Raw interaction-task downloads live under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw`.
- The rebuild script is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/build_ppi_interaction_csv.py`.
- Current processed outputs:
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/processed/interaction_sources_manifest.csv`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/processed/interaction_records_summary.csv`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/processed/interaction_records_unified.csv`
- The unified records CSV is a row-level integration table, not the final training format. It has `6271559` records and is useful for auditing, filtering, and writing task-specific `bioseq.v1` shards.
- Supported row-level sources in the current CSV: Figshare gold-standard PPI, HumanPPI LMDB, YeastPPI LMDB, SKEMPI, SWING MutInt, FLAb binding, TDC TCR-epitope, PISTE TCR-epitope-HLA, TEIM binding/interface metadata, oncoPPI spreadsheets, and CoV-AbDab neutralization.
- STRING-DB v12.0 remains a separate large pretraining source. Its raw physical links and sequence dumps are too large for the current row-level CSV pass and should be handled by a streaming adapter or by reusing the existing processed PPI Arrow data under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi/string_model_org_90_90_split`.
- PDBbind+ and the referenced bioRxiv SARS-CoV-2 binding supplement are blocked in this environment by login/subscription and HTTP 403, respectively. They should not be treated as available training sources until access is resolved.
- Before training, convert the large unified CSV into sharded task-specific records. Recommended first shards: `ppi_binary`, `antibody_binding`, `tcr_epitope_hla`, `tcr_epitope_binding`, `mutational_ppi`, and `ppi_mutation_affinity`.

## Training Logic

- The foundation-model objective should stay diffusion-only by default: corrupt eligible target residues at timestep `t`, predict the clean tokens, and compute denoising cross-entropy only on target/remasked positions.
- Every training example should carry explicit masks: `diffusion_target_mask` for residues that may be noised/remasked and receive diffusion loss, and `fixed_context_mask` for residues that remain clean and visible as conditioning context.
- Training should separate biological examples from target-mask construction. The stored example contains the full clean chains plus regions/metadata; the foundation collator defaults to `full_denoise` and creates token-level masks so all eligible generated residues participate in the diffusion objective.
- For the BioSeq foundation-model path, training-time data loading lives under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data`. This path should read source-specific files directly, emit canonical `BioSeqRecord` objects, sample training views, and collate ESM-family token ids plus per-chain encoder tensors. Full offline JSONL conversion is optional, not a prerequisite for training.
- The BioSeq foundation model layer now starts at `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py`. It exposes `BioSeqNoEncoderDiffusionModel` for no-encoder training and `BioSeqEncoderDiffusionModel` for ESMC/ESM feature-conditioned training.
- The first BioSeq foundation no-encoder stack is a dense bidirectional diffusion transformer with ESM-family token embeddings, chain-local residue position embeddings, outer chain-index embeddings, chain-role embeddings, task-type embeddings, timestep embeddings, RMSNorm, and SwiGLU blocks. It uses bidirectional self-attention for masked diffusion, not a causal/autoregressive language-model architecture. It is intentionally separate from the legacy `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq` lightweight backend.
- `BioSeqNoEncoderDiffusionModel.compute_loss` samples timestep noise from `diffusion_loss_mask`, replaces corrupted target residues with `<mask>`, predicts clean residue ids, and computes denoising cross-entropy only on corrupted target positions.
- `BioSeqEncoderDiffusionModel.compute_loss` uses the same diffusion objective and builds a per-chain ESMC/ESM `x_t` by applying the decoder corruption state to `encoder_input_ids`. ESMC/ESM returns token-level features for that diffusion state, and the downstream BioSeq denoiser performs joint multi-chain denoising over the concatenated decoder stream.
- Local ESMC encoder loading is implemented through `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py::load_local_esmc_encoder`. Because `transformers==4.48.1` does not recognize `model_type="esmc"`, `BioSeqEncoderDiffusionModel.from_esmc(...)` tries `AutoModel.from_pretrained(...)` first and then falls back to Biohub `esm==3.2.3`, mapping local Hugging Face-style safetensor keys into native `esm.models.esmc.ESMC` keys.
- BioSeq foundation training should use mixed physical microbatches: source stream -> `WeightedMixtureDataset` -> `DataLoader(batch_size=N)` -> `BioSeqQwenDataCollator(single_view_per_batch=False, require_homogeneous_task=False)`. A single batch may contain antibody, antibody-antigen, TCR, TCR-pMHC, PPI, and other task groups. Each record defaults to `full_denoise` and resolves to token-level diffusion masks over all eligible generated residues.
- Multi-node/multi-GPU training for the BioSeq foundation-model path should use `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py` launched by `torchrun`. It supports ordinary single-process smoke runs, single-node multi-GPU, and multi-node multi-GPU through `RANK`, `WORLD_SIZE`, and `LOCAL_RANK`.
- BioSeq foundation streaming data is DDP-sharded inside `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/mixture.py`: source iterators now use `global_shard_index = rank * num_workers + worker_id` and `num_shards = world_size * num_workers`, so DDP ranks do not read identical iterable records by default.
- The first cluster template for this path is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_16gpu_smoke.yml`. It runs the no-encoder qwen3_vl_arch model on 2 nodes x 8 GPUs as a smoke/throughput check.
- Foundation pretraining should keep the objective simple: the DDP trainer constructs `BioSeqViewSampler(allowed_views=["full_denoise"])`, so conditional views such as chain completion, antigen-conditioned receptor generation, peptide design, and FR/CDR infilling are not reachable from the main training path.
- To reduce multi-task imbalance, source mixture weights and optional future task schedulers should control the task distribution. The default physical batch itself is intentionally mixed rather than task-homogeneous.
- Fixed context chains, such as antigen in antibody-antigen generation, should not be remasked and should not receive direct diffusion loss. They still participate in attention or encoder conditioning, and gradients should flow through their encoder/connector parameters from the target-chain diffusion loss.
- In the BioSeq foundation loader, `full_denoise` means denoising all eligible target chains, not every chain unconditionally. Explicit `metadata["targets"]` is honored when present, but antigen, peptide, MHC, and HLA-like chains are fixed context by default and do not receive diffusion loss.
- Antibody-antigen and nanobody-antigen views should follow the receptor-design direction used by current antibody design work: fixed antigen can generate heavy/light antibody chains or VHH, fixed antigen plus one antibody chain can generate the paired antibody chain, and fixed antigen plus receptor FR regions can generate CDR regions. Antibody/nanobody-to-antigen inverse views are not part of the default task set.
- The `mhc_to_peptide_tcr` view covers the complementary TCR-pMHC case where MHC/HLA is clean fixed context and peptide plus available TCR chains are diffusion targets. This should be sampled explicitly during pretraining if peptide/TCR co-design or peptide-conditioned receptor learning is expected at inference.
- The `tcr_mhc_to_peptide` view covers peptide design from fixed TCR alpha/beta plus MHC/HLA. The `pmhc_to_tcr` view covers TCR alpha/beta design from fixed peptide plus MHC/HLA.
- FR/CDR views are region-driven, not antibody-only. They apply to TCR full-chain data whenever source adapters preserve TCR FR/CDR region annotations. For TCR-pMHC, fixed peptide/epitope plus MHC/HLA can be combined with fixed TCR FR regions to generate TCR CDR regions.
- ESMC/ESM feature-conditioned training is still diffusion training: ESMC/ESM2 features are extracted from `x_t` and condition target denoising, but the encoder is updated through the target diffusion loss rather than through a separate antigen reconstruction loss.
- The ESMC/ESM feature-conditioned path defaults to trainable encoder parameters. `freeze_encoder=True` is available only for ablations or bootstrap checks; the main foundation-model setting should fine-tune the ESMC/ESM encoder and connector with the diffusion loss.
- The ESMC/ESM feature-conditioned path encodes each chain/sequence independently through ESMC on the current per-chain `x_t`. Implementation detail: `encoder_input_ids` has shape `[batch, max_chains, chain_len]`, is flattened to `[batch * max_chains, chain_len]` for a batched ESMC call, then reshaped to `[batch, max_chains, chain_len, hidden]`; residue features are gathered back to decoder token positions before the multi-chain BioSeq denoiser runs.
- The current local environment keeps `transformers==4.48.1` for compatibility with Biohub `esm==3.2.3`; ESMC loading must use `BioSeqEncoderDiffusionModel.from_esmc(...)` or `load_local_esmc_encoder(...)` rather than relying on `AutoModel` alone.
- ESMC tokenizer loading must also handle the local `ESMCTokenizer` metadata without requiring a newer `transformers`: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/esm_encoding.py` falls back to `tokenizers.Tokenizer.from_file(...)` on the local `tokenizer.json`.
- Remask schedules, block-diffusion masks, and generation masks must operate on `diffusion_target_mask`, not on all non-pad tokens.
- If a sampled view loses all target residues after chain or sequence truncation, the collator should fall back to `full_denoise` for that record when eligible target tokens remain. This prevents invalid zero-loss microbatches without changing normal untruncated view sampling.
- Formal BioSeq foundation stage-1 training should run three comparable variants: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_esmc300m_stage1.yml`, `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_esmc600m_stage1.yml`, and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_no_encoder_stage1.yml`. The first two fine-tune trainable ESMC encoders. The no-encoder baseline uses the same ESMC tokenizer/vocab 64 but does not instantiate or optimize ESMC encoder parameters.
- Batch-size tuning should be driven by logged CUDA peak memory, not by guessed limits. Keep effective batch approximately fixed while changing per-GPU batch and gradient accumulation. The initial 8-GPU stage-1 settings are conservative: no-encoder `8 x grad_accum 2`, ESMC-300M `2 x grad_accum 8`, and ESMC-600M `1 x grad_accum 16`.

## Generation Task Survey and View Priority

- Survey date: 2026-06-14.
- Antibody generation should prioritize receptor-side design, not antigen generation. DiffAb samples antibody CDRs for antibody-antigen complexes and antigen-only settings, RFdiffusion antibody design explicitly focuses sampling on CDR loops while keeping a framework close to a specified therapeutic scaffold, IgLM supports full antibody generation and variable-span/CDR infilling, and paired antibody language models show that heavy-light pairing helps cross-chain feature learning.
- Current antibody/nanobody high-priority views are therefore: `antigen_to_antibody`, `antigen_to_nanobody`, `heavy_antigen_to_light`, `light_antigen_to_heavy`, `antigen_fr_to_cdr`, `antigen_single_cdr`, `heavy_to_light`, `light_to_heavy`, `fr_to_cdr`, and `single_cdr`. `cdr_to_fr` is useful for humanization/framework-design style ablations, but should not dominate early pretraining. `antibody_to_antigen` and `nanobody_to_antigen` are removed from the default task set.
- TCR generation should prioritize pMHC/epitope-conditioned receptor design and paired-chain completion. TCR-TRANSLATE generates antigen-specific TCR sequences from unseen pMHC inputs, TCRdesign uses antigen-conditioned generation with paired-chain coherence, and recent TCR structure work argues that both alpha and beta chain information matter for specificity and structure.
- Current TCR high-priority views are therefore: `pmhc_to_tcr`, `pmhc_fr_to_cdr`, `pmhc_single_cdr`, beta-chain context to alpha-chain completion, alpha-chain context to beta-chain completion, `fr_to_cdr`, and `single_cdr` when TCR FR/CDR annotations exist. The existing names `beta_epitope_to_alpha` and `alpha_epitope_to_beta` should be treated as context-to-chain completion views; a later cleanup should either rename them to `beta_context_to_alpha` / `alpha_context_to_beta` or add strict epitope-required variants.
- TCR peptide/epitope design views are biologically useful but should be lower-weight or task-specific until evaluation data is stronger: `tcr_mhc_to_peptide` is appropriate for epitope discovery/design from fixed TCR+MHC, while `mhc_to_peptide_tcr` is a co-denoising/co-design view rather than the first downstream validation target.
- Default view sampling in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/view_sampler.py` is now divided by data type: antibody, antibody-antigen, nanobody-antigen, TCR, TCR-epitope, TCR-pMHC, PPI, and generic. Explicit `allowed_views` can still override the profile for ablations or downstream-specific training.
- Sources: DiffAb (`https://github.com/luost26/diffab`, `https://openreview.net/forum?id=jSorGn2Tjg`), RFdiffusion antibody design (`https://www.nature.com/articles/s41586-025-09721-5`), IgLM (`https://doi.org/10.1016/j.cels.2023.10.001`), paired antibody language models (`https://doi.org/10.1371/journal.pcbi.1012646`), TCR-TRANSLATE (`https://www.nature.com/articles/s42256-025-01096-6`), TCRdesign (`https://doi.org/10.1093/bib/bbaf691`), and paired TCR alpha/beta structure analysis (`https://www.nature.com/articles/s42003-025-07708-6`).

- Ophiuchus-Ab training uses masked diffusion over heavy/light antibody tokens with AirGen-compatible timestep corruption.
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/ophiuchus/collator.py` emits `heavy_tokens` and `light_tokens`, each with `targets`, `regions`, and `chain_ids`, plus per-example `weights`.
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/ophiuchus/training.py` computes the heavy and light losses through `MultiChainOphiuchusAbModel.compute_loss`.
- Next migration quality step: add parity checks against `/vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/src/byprot/models/lm/dplm_multichain.py` for `construct_x_t`, `compute_loss`, and generation mask transitions. The MINT module files themselves already match AirGen byte-for-byte except generated `__pycache__`.
- BioSeq training code must not import the old `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/core/trainers` implementation.
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_ab.py` must expose only the exact Ophiuchus-Ab training path. It must not include the lightweight `bioseq` backend, synthetic default data, or the generic `transformers.Trainer` path.
- Variable-length training (no fixed `(150, 128)` padding) uses `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/ophiuchus/collator.py::MultiChainDynamicCollator`, which pads each batch to the longest chain-1/chain-2 sequence with the real `<pad>` id. This is correct because the mint ESM2 backbone derives its padding mask from `tokens.eq(<pad>)`, and `compute_loss` splits logits by per-slot length. `OphiuchusAbTrainingCollator` (fixed length) is kept for exact-length reproductions.
- Multi-dataset training mixes OAS paired antibody, OTS paired TCR, and nanobody VHH through `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/datasets.py` (`build_mixed_immune_dataset` / `default_immune_specs`). OAS heavy is oriented to slot 0, OTS beta to slot 0, and nanobody is single-chain (slot 1 uses a `[<cls>, <eos>]` placeholder).
- Multi-node/multi-GPU training uses `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_bioseq_ddp.py` with `torchrun`. It wraps only the backbone (`MultiChainOphiuchusAbModel.net`) in `DistributedDataParallel`, auto-selects `nccl`+CUDA or `gloo`+CPU, and uses `DistributedSampler`. The diffusion wrapper itself holds no parameters, so a single synced backbone forward per step keeps DDP gradient reduction correct.
- Verified end to end with both a single-process CPU smoke run and a 2-process `torchrun` (gloo) DDP smoke run loading all three datasets at variable length.
- Initializing from generic ESM2 base weights is not part of the current antibody path. If it becomes necessary later, use the local ESM2 snapshots under `/c20250601/mj/model_weights/esm2` rather than adding a Hugging Face download dependency.

## Inference Logic

- Antibody inference must use the exact Ophiuchus-Ab stack through `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/sample_ab.py` or aligned downstream scripts under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream`.
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/sample_ab.py` loads `/c20250601/mj/model_weights/ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt` through `MultiChainOphiuchusAbModel.from_checkpoint`.
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/common.py` is the shared inference layer for aligned downstream scripts; it loads `MultiChainOphiuchusAbModel`, applies AirGen-style chain padding, and calls `model.generate`.
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
