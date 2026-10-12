"""From-scratch research variants, isolated from the measured production v3 model.

Run: python -m pytest scripts/tests/immune_llada/test_research.py -q
Training/profile entry: python scripts/run_immune_research.py --help
The image-inspired adaptations below are experiments, not paper reproductions.
"""
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ..data.records import CHAIN_ROLE_TO_ID
from ..v3.model import ImmuneConfig, ImmuneModel
from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import sample_bioseq_diffusion_noise


VARIANTS = {
    'p07_v7_cdr_kong_cdrh1': {'launcher_only': True, 'generation_evaluation_only': True},
    'p07_v7_cdr_kong_cdrh2': {'launcher_only': True, 'generation_evaluation_only': True},
    'p07_v7_cdr_kong_cdrh3': {'launcher_only': True, 'generation_evaluation_only': True},
    'p07_v7_cdr_sab23_cdrl1': {'launcher_only': True, 'generation_evaluation_only': True},
    'p07_v7_cdr_sab23_cdrl2': {'launcher_only': True, 'generation_evaluation_only': True},
    'p07_v7_cdr_sab23_cdrl3': {'launcher_only': True, 'generation_evaluation_only': True},
    'p07_v7_cdr_sab23_cdrh1': {'launcher_only': True, 'generation_evaluation_only': True},
    'p07_v7_cdr_sab23_cdrh2': {'launcher_only': True, 'generation_evaluation_only': True},
    'p07_v7_cdr_sab23_cdrh3': {'launcher_only': True, 'generation_evaluation_only': True},
    'p06_v7_pool_identity': {'launcher_only': True, 'architecture_refined': True},
    'p06_v7_formal_member_0': {'launcher_only': True, 'architecture_refined': True},
    'p06_v7_formal_member_1': {'launcher_only': True, 'architecture_refined': True},
    'p06_v7_formal_member_2': {'launcher_only': True, 'architecture_refined': True},
    'p06_v7_formal_member_3': {'launcher_only': True, 'architecture_refined': True},
    'p06_v7_formal_member_4': {'launcher_only': True, 'architecture_refined': True},
    'p06_v7_formal_member_5': {'launcher_only': True, 'architecture_refined': True},
    'p06_v7_formal_member_6': {'launcher_only': True, 'architecture_refined': True},
    'p06_v7_formal_member_7': {'launcher_only': True, 'architecture_refined': True},
    'r00_a4_control': {},
    'r01_span_mask': {'span_mask': True},
    'r02_chain_completion': {'chain_mask': True},
    'r03_chain_balanced_loss': {'balanced_loss': True},
    'r04_qk_norm': {'qk_norm': True},
    'r05_register4': {'registers': 4},
    'r06_local_conv7': {'local_conv': True},
    'r07_mask_ratio_film': {'mask_film': True},
    'r08_vicreg': {'vicreg': True},
    'r09_equal_source': {'equal_source': True},
    'r10_role_pair_bias': {'role_bias': True},
    'r11_span_qknorm_register': {'span_mask': True, 'qk_norm': True, 'registers': 4},
    # Explicit registry also serves the already-running pool's AST allowlist.
    # New-family construction and checkpoint loading dispatch to its own class.
    'r12_architecture_control': {'architecture_v2': True},
    'r13_head_output_gate': {'architecture_v2': True, 'head_gate': True},
    'r14_fixed_half_attention': {'architecture_v2': True, 'fixed_head_gate': .5},
    'r15_layerscale': {'architecture_v2': True, 'residual_scale': .1, 'learn_scale': True},
    'r16_fixed_residual_scale': {'architecture_v2': True, 'residual_scale': .1},
    'r17_chain_late_fusion': {'architecture_v2': True, 'chain_late_fusion': True},
    'r18_chain_local_global': {'architecture_v2': True, 'local_radius': 32},
    'r19_value_residual': {'architecture_v2': True, 'value_residual': True},
    'r20_refined_control': {'architecture_refined': True},
    'r21_refined_head_gate': {'architecture_refined': True, 'head_gate': True},
    'r22_refined_fixed_gate': {'architecture_refined': True, 'fixed_head_gate': .5},
    'r23_refined_layerscale': {'architecture_refined': True, 'residual_scale': .1, 'learn_scale': True},
    'r24_refined_fixed_scale': {'architecture_refined': True, 'residual_scale': .1},
    'r25_refined_late_fusion': {'architecture_refined': True, 'chain_late_fusion': True},
    'r26_refined_local_global': {'architecture_refined': True, 'local_radius': 32},
    'r27_refined_value_residual': {'architecture_refined': True, 'value_residual': True},
    'd01_pilot_runtime_gpu_r20_v1': {'diagnostic_only': True},
    'd02_pilot_seed_gpu_r20_seed42_v2': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r20_refined_control', 'seed': 42},
    'd02_pilot_seed_gpu_r20_seed43_v2': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r20_refined_control', 'seed': 43},
    'd02_pilot_seed_gpu_r21_seed42_v2': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r21_refined_head_gate', 'seed': 42},
    'd02_pilot_seed_gpu_r21_seed43_v2': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r21_refined_head_gate', 'seed': 43},
    'd02_pilot_seed_gpu_r24_seed42_v2': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r24_refined_fixed_scale', 'seed': 42},
    'd02_pilot_seed_gpu_r24_seed43_v2': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r24_refined_fixed_scale', 'seed': 43},
    'd02_pilot_seed_gpu_r26_seed42_v2': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r26_refined_local_global', 'seed': 42},
    'd02_pilot_seed_gpu_r26_seed43_v2': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r26_refined_local_global', 'seed': 43},
    'd03_r21_seed43_tiny_gpu_observation_v1': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r21_refined_head_gate', 'seed': 43, 'numerical_observation_only': True},
    'd04_r21_CE_learning_v1': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r21_refined_head_gate', 'synthetic_learning_only': True, 'learning_seed': 42, 'formal_source_seed': 43},
    'd05_r20_seed42_learning_v1': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r20_refined_control', 'synthetic_learning_only': True, 'learning_seed': 42, 'formal_source_seed': 42},
    'd05_r20_seed43_learning_v1': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r20_refined_control', 'synthetic_learning_only': True, 'learning_seed': 42, 'formal_source_seed': 43},
    'd05_r21_seed42_learning_v1': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r21_refined_head_gate', 'synthetic_learning_only': True, 'learning_seed': 42, 'formal_source_seed': 42},
    'd05_r24_seed42_learning_v1': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r24_refined_fixed_scale', 'synthetic_learning_only': True, 'learning_seed': 42, 'formal_source_seed': 42},
    'd05_r24_seed43_learning_v1': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r24_refined_fixed_scale', 'synthetic_learning_only': True, 'learning_seed': 42, 'formal_source_seed': 43},
    'd05_r26_seed42_learning_v1': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r26_refined_local_global', 'synthetic_learning_only': True, 'learning_seed': 42, 'formal_source_seed': 42},
    'd05_r26_seed43_learning_v1': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r26_refined_local_global', 'synthetic_learning_only': True, 'learning_seed': 42, 'formal_source_seed': 43},
    'g01_r21_world8_member_0': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r21_refined_head_gate', 'synthetic_world8_member_only': True, 'source_profile_seed': 43, 'synthetic_learning_seed': 42},
    'g01_r21_world8_member_1': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r21_refined_head_gate', 'synthetic_world8_member_only': True, 'source_profile_seed': 43, 'synthetic_learning_seed': 42},
    'g01_r21_world8_member_2': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r21_refined_head_gate', 'synthetic_world8_member_only': True, 'source_profile_seed': 43, 'synthetic_learning_seed': 42},
    'g01_r21_world8_member_3': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r21_refined_head_gate', 'synthetic_world8_member_only': True, 'source_profile_seed': 43, 'synthetic_learning_seed': 42},
    'g01_r21_world8_member_4': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r21_refined_head_gate', 'synthetic_world8_member_only': True, 'source_profile_seed': 43, 'synthetic_learning_seed': 42},
    'g01_r21_world8_member_5': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r21_refined_head_gate', 'synthetic_world8_member_only': True, 'source_profile_seed': 43, 'synthetic_learning_seed': 42},
    'g01_r21_world8_member_6': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r21_refined_head_gate', 'synthetic_world8_member_only': True, 'source_profile_seed': 43, 'synthetic_learning_seed': 42},
    'g01_r21_world8_member_7': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r21_refined_head_gate', 'synthetic_world8_member_only': True, 'source_profile_seed': 43, 'synthetic_learning_seed': 42},
    'g02_r21_world8_member_0': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r21_refined_head_gate', 'synthetic_world8_member_only': True, 'source_profile_seed': 43, 'synthetic_learning_seed': 42},
    'g02_r21_world8_member_1': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r21_refined_head_gate', 'synthetic_world8_member_only': True, 'source_profile_seed': 43, 'synthetic_learning_seed': 42},
    'g02_r21_world8_member_2': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r21_refined_head_gate', 'synthetic_world8_member_only': True, 'source_profile_seed': 43, 'synthetic_learning_seed': 42},
    'g02_r21_world8_member_3': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r21_refined_head_gate', 'synthetic_world8_member_only': True, 'source_profile_seed': 43, 'synthetic_learning_seed': 42},
    'g02_r21_world8_member_4': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r21_refined_head_gate', 'synthetic_world8_member_only': True, 'source_profile_seed': 43, 'synthetic_learning_seed': 42},
    'g02_r21_world8_member_5': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r21_refined_head_gate', 'synthetic_world8_member_only': True, 'source_profile_seed': 43, 'synthetic_learning_seed': 42},
    'g02_r21_world8_member_6': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r21_refined_head_gate', 'synthetic_world8_member_only': True, 'source_profile_seed': 43, 'synthetic_learning_seed': 42},
    'g02_r21_world8_member_7': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r21_refined_head_gate', 'synthetic_world8_member_only': True, 'source_profile_seed': 43, 'synthetic_learning_seed': 42},
    'g03_r21_world8_member_0': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r21_refined_head_gate', 'synthetic_world8_member_only': True, 'source_profile_seed': 43, 'synthetic_learning_seed': 42},
    'g03_r21_world8_member_1': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r21_refined_head_gate', 'synthetic_world8_member_only': True, 'source_profile_seed': 43, 'synthetic_learning_seed': 42},
    'g03_r21_world8_member_2': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r21_refined_head_gate', 'synthetic_world8_member_only': True, 'source_profile_seed': 43, 'synthetic_learning_seed': 42},
    'g03_r21_world8_member_3': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r21_refined_head_gate', 'synthetic_world8_member_only': True, 'source_profile_seed': 43, 'synthetic_learning_seed': 42},
    'g03_r21_world8_member_4': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r21_refined_head_gate', 'synthetic_world8_member_only': True, 'source_profile_seed': 43, 'synthetic_learning_seed': 42},
    'g03_r21_world8_member_5': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r21_refined_head_gate', 'synthetic_world8_member_only': True, 'source_profile_seed': 43, 'synthetic_learning_seed': 42},
    'g03_r21_world8_member_6': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r21_refined_head_gate', 'synthetic_world8_member_only': True, 'source_profile_seed': 43, 'synthetic_learning_seed': 42},
    'g03_r21_world8_member_7': {'launcher_only': True, 'architecture_refined': True, 'model_variant': 'r21_refined_head_gate', 'synthetic_world8_member_only': True, 'source_profile_seed': 43, 'synthetic_learning_seed': 42},
    'p01_v7_r20_seed42': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'r20_refined_control', 'seed': 42},
    'p01_v7_r20_seed43': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'r20_refined_control', 'seed': 43},
    'p01_v7_r21_seed42': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'r21_refined_head_gate', 'seed': 42},
    'p01_v7_r21_seed43': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'r21_refined_head_gate', 'seed': 43},
    'p01_v7_r24_seed42': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'r24_refined_fixed_scale', 'seed': 42},
    'p01_v7_r24_seed43': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'r24_refined_fixed_scale', 'seed': 43},
    'p01_v7_r26_seed42': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'r26_refined_local_global', 'seed': 42},
    'p01_v7_r26_seed43': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'r26_refined_local_global', 'seed': 43},
    'p02_v7_f000_seed42': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'f000', 'seed': 42, 'study': 'v7_context_factorial_120k_v1'},
    'p02_v7_f000_seed43': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'f000', 'seed': 43, 'study': 'v7_context_factorial_120k_v1'},
    'p02_v7_f001_seed42': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'f001', 'seed': 42, 'study': 'v7_context_factorial_120k_v1'},
    'p02_v7_f001_seed43': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'f001', 'seed': 43, 'study': 'v7_context_factorial_120k_v1'},
    'p02_v7_f010_seed42': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'f010', 'seed': 42, 'study': 'v7_context_factorial_120k_v1'},
    'p02_v7_f010_seed43': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'f010', 'seed': 43, 'study': 'v7_context_factorial_120k_v1'},
    'p02_v7_f011_seed42': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'f011', 'seed': 42, 'study': 'v7_context_factorial_120k_v1'},
    'p02_v7_f011_seed43': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'f011', 'seed': 43, 'study': 'v7_context_factorial_120k_v1'},
    'p02_v7_f100_seed42': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'f100', 'seed': 42, 'study': 'v7_context_factorial_120k_v1'},
    'p02_v7_f100_seed43': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'f100', 'seed': 43, 'study': 'v7_context_factorial_120k_v1'},
    'p02_v7_f101_seed42': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'f101', 'seed': 42, 'study': 'v7_context_factorial_120k_v1'},
    'p02_v7_f101_seed43': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'f101', 'seed': 43, 'study': 'v7_context_factorial_120k_v1'},
    'p02_v7_f110_seed42': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'f110', 'seed': 42, 'study': 'v7_context_factorial_120k_v1'},
    'p02_v7_f110_seed43': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'f110', 'seed': 43, 'study': 'v7_context_factorial_120k_v1'},
    'p02_v7_f111_seed42': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'f111', 'seed': 42, 'study': 'v7_context_factorial_120k_v1'},
    'p02_v7_f111_seed43': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'f111', 'seed': 43, 'study': 'v7_context_factorial_120k_v1'},
    'p02_v7_cv_segment_global_seed42': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'cv_segment_global', 'seed': 42, 'study': 'v7_context_factorial_120k_v1'},
    'p02_v7_cv_segment_global_seed43': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'cv_segment_global', 'seed': 43, 'study': 'v7_context_factorial_120k_v1'},
    'p02_v7_cv_late_fusion_seed42': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'cv_late_fusion', 'seed': 42, 'study': 'v7_context_factorial_120k_v1'},
    'p02_v7_cv_late_fusion_seed43': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'cv_late_fusion', 'seed': 43, 'study': 'v7_context_factorial_120k_v1'},
    'p02_v7_fixed_gate_control_seed42': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'fixed_gate_control', 'seed': 42, 'study': 'v7_context_factorial_120k_v1'},
    'p02_v7_fixed_gate_control_seed43': {'launcher_only': True, 'exploratory_real_v7': True, 'model_variant': 'fixed_gate_control', 'seed': 43, 'study': 'v7_context_factorial_120k_v1'},
    'p04_v7_capacity': {'launcher_only': True, 'synthetic_capacity_profile': True, 'study': 'v7_context_factorial_120k_large_batch_v1'},
    'p05_v7_capacity_v2': {'launcher_only': True, 'synthetic_capacity_profile': True, 'study': 'v7_context_factorial_120k_large_batch_v1', 'capacity_protocol': 'v2_36_updates', 'job_id': '1790992416352344471_p05_v7_capacity_v2_a4854542', 'intent_sha256': '3429ab434375501e56860e52dd83ebe6702f436a6b18b54d26752b7ff2efc152'},
    'p04_v7_f000_seed42': {'launcher_only': True, 'model_variant': 'f000', 'seed': 42, 'study': 'v7_context_factorial_120k_large_batch_v1', 'jobs_catalog_sha256': '987394f0e0a539a0ad5e0057aa0952474eecb50f374e8ea1410f9f3168ea746d', 'batch_plan_sha256': '42be6e8e7c7ebbe578de16646cd6339c963390d2ffae5ed1226980dea46fbc04', 'source_freeze_sha256': '2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783'},
    'p04_v7_f000_seed43': {'launcher_only': True, 'model_variant': 'f000', 'seed': 43, 'study': 'v7_context_factorial_120k_large_batch_v1', 'jobs_catalog_sha256': '987394f0e0a539a0ad5e0057aa0952474eecb50f374e8ea1410f9f3168ea746d', 'batch_plan_sha256': '42be6e8e7c7ebbe578de16646cd6339c963390d2ffae5ed1226980dea46fbc04', 'source_freeze_sha256': '2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783'},
    'p04_v7_f010_seed42': {'launcher_only': True, 'model_variant': 'f010', 'seed': 42, 'study': 'v7_context_factorial_120k_large_batch_v1', 'jobs_catalog_sha256': '987394f0e0a539a0ad5e0057aa0952474eecb50f374e8ea1410f9f3168ea746d', 'batch_plan_sha256': '42be6e8e7c7ebbe578de16646cd6339c963390d2ffae5ed1226980dea46fbc04', 'source_freeze_sha256': '2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783'},
    'p04_v7_f010_seed43': {'launcher_only': True, 'model_variant': 'f010', 'seed': 43, 'study': 'v7_context_factorial_120k_large_batch_v1', 'jobs_catalog_sha256': '987394f0e0a539a0ad5e0057aa0952474eecb50f374e8ea1410f9f3168ea746d', 'batch_plan_sha256': '42be6e8e7c7ebbe578de16646cd6339c963390d2ffae5ed1226980dea46fbc04', 'source_freeze_sha256': '2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783'},
    'p04_v7_f001_seed42': {'launcher_only': True, 'model_variant': 'f001', 'seed': 42, 'study': 'v7_context_factorial_120k_large_batch_v1', 'jobs_catalog_sha256': '987394f0e0a539a0ad5e0057aa0952474eecb50f374e8ea1410f9f3168ea746d', 'batch_plan_sha256': '42be6e8e7c7ebbe578de16646cd6339c963390d2ffae5ed1226980dea46fbc04', 'source_freeze_sha256': '2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783'},
    'p04_v7_f001_seed43': {'launcher_only': True, 'model_variant': 'f001', 'seed': 43, 'study': 'v7_context_factorial_120k_large_batch_v1', 'jobs_catalog_sha256': '987394f0e0a539a0ad5e0057aa0952474eecb50f374e8ea1410f9f3168ea746d', 'batch_plan_sha256': '42be6e8e7c7ebbe578de16646cd6339c963390d2ffae5ed1226980dea46fbc04', 'source_freeze_sha256': '2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783'},
    'p04_v7_f011_seed42': {'launcher_only': True, 'model_variant': 'f011', 'seed': 42, 'study': 'v7_context_factorial_120k_large_batch_v1', 'jobs_catalog_sha256': '987394f0e0a539a0ad5e0057aa0952474eecb50f374e8ea1410f9f3168ea746d', 'batch_plan_sha256': '42be6e8e7c7ebbe578de16646cd6339c963390d2ffae5ed1226980dea46fbc04', 'source_freeze_sha256': '2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783'},
    'p04_v7_f011_seed43': {'launcher_only': True, 'model_variant': 'f011', 'seed': 43, 'study': 'v7_context_factorial_120k_large_batch_v1', 'jobs_catalog_sha256': '987394f0e0a539a0ad5e0057aa0952474eecb50f374e8ea1410f9f3168ea746d', 'batch_plan_sha256': '42be6e8e7c7ebbe578de16646cd6339c963390d2ffae5ed1226980dea46fbc04', 'source_freeze_sha256': '2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783'},
    'p04_v7_f100_seed42': {'launcher_only': True, 'model_variant': 'f100', 'seed': 42, 'study': 'v7_context_factorial_120k_large_batch_v1', 'jobs_catalog_sha256': '987394f0e0a539a0ad5e0057aa0952474eecb50f374e8ea1410f9f3168ea746d', 'batch_plan_sha256': '42be6e8e7c7ebbe578de16646cd6339c963390d2ffae5ed1226980dea46fbc04', 'source_freeze_sha256': '2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783'},
    'p04_v7_f100_seed43': {'launcher_only': True, 'model_variant': 'f100', 'seed': 43, 'study': 'v7_context_factorial_120k_large_batch_v1', 'jobs_catalog_sha256': '987394f0e0a539a0ad5e0057aa0952474eecb50f374e8ea1410f9f3168ea746d', 'batch_plan_sha256': '42be6e8e7c7ebbe578de16646cd6339c963390d2ffae5ed1226980dea46fbc04', 'source_freeze_sha256': '2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783'},
    'p04_v7_f101_seed42': {'launcher_only': True, 'model_variant': 'f101', 'seed': 42, 'study': 'v7_context_factorial_120k_large_batch_v1', 'jobs_catalog_sha256': '987394f0e0a539a0ad5e0057aa0952474eecb50f374e8ea1410f9f3168ea746d', 'batch_plan_sha256': '42be6e8e7c7ebbe578de16646cd6339c963390d2ffae5ed1226980dea46fbc04', 'source_freeze_sha256': '2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783'},
    'p04_v7_f101_seed43': {'launcher_only': True, 'model_variant': 'f101', 'seed': 43, 'study': 'v7_context_factorial_120k_large_batch_v1', 'jobs_catalog_sha256': '987394f0e0a539a0ad5e0057aa0952474eecb50f374e8ea1410f9f3168ea746d', 'batch_plan_sha256': '42be6e8e7c7ebbe578de16646cd6339c963390d2ffae5ed1226980dea46fbc04', 'source_freeze_sha256': '2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783'},
    'p04_v7_f110_seed42': {'launcher_only': True, 'model_variant': 'f110', 'seed': 42, 'study': 'v7_context_factorial_120k_large_batch_v1', 'jobs_catalog_sha256': '987394f0e0a539a0ad5e0057aa0952474eecb50f374e8ea1410f9f3168ea746d', 'batch_plan_sha256': '42be6e8e7c7ebbe578de16646cd6339c963390d2ffae5ed1226980dea46fbc04', 'source_freeze_sha256': '2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783'},
    'p04_v7_f110_seed43': {'launcher_only': True, 'model_variant': 'f110', 'seed': 43, 'study': 'v7_context_factorial_120k_large_batch_v1', 'jobs_catalog_sha256': '987394f0e0a539a0ad5e0057aa0952474eecb50f374e8ea1410f9f3168ea746d', 'batch_plan_sha256': '42be6e8e7c7ebbe578de16646cd6339c963390d2ffae5ed1226980dea46fbc04', 'source_freeze_sha256': '2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783'},
    'p04_v7_f111_seed42': {'launcher_only': True, 'model_variant': 'f111', 'seed': 42, 'study': 'v7_context_factorial_120k_large_batch_v1', 'jobs_catalog_sha256': '987394f0e0a539a0ad5e0057aa0952474eecb50f374e8ea1410f9f3168ea746d', 'batch_plan_sha256': '42be6e8e7c7ebbe578de16646cd6339c963390d2ffae5ed1226980dea46fbc04', 'source_freeze_sha256': '2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783'},
    'p04_v7_f111_seed43': {'launcher_only': True, 'model_variant': 'f111', 'seed': 43, 'study': 'v7_context_factorial_120k_large_batch_v1', 'jobs_catalog_sha256': '987394f0e0a539a0ad5e0057aa0952474eecb50f374e8ea1410f9f3168ea746d', 'batch_plan_sha256': '42be6e8e7c7ebbe578de16646cd6339c963390d2ffae5ed1226980dea46fbc04', 'source_freeze_sha256': '2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783'},
    'p04_v7_cv_segment_global_seed42': {'launcher_only': True, 'model_variant': 'cv_segment_global', 'seed': 42, 'study': 'v7_context_factorial_120k_large_batch_v1', 'jobs_catalog_sha256': '987394f0e0a539a0ad5e0057aa0952474eecb50f374e8ea1410f9f3168ea746d', 'batch_plan_sha256': '42be6e8e7c7ebbe578de16646cd6339c963390d2ffae5ed1226980dea46fbc04', 'source_freeze_sha256': '2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783'},
    'p04_v7_cv_segment_global_seed43': {'launcher_only': True, 'model_variant': 'cv_segment_global', 'seed': 43, 'study': 'v7_context_factorial_120k_large_batch_v1', 'jobs_catalog_sha256': '987394f0e0a539a0ad5e0057aa0952474eecb50f374e8ea1410f9f3168ea746d', 'batch_plan_sha256': '42be6e8e7c7ebbe578de16646cd6339c963390d2ffae5ed1226980dea46fbc04', 'source_freeze_sha256': '2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783'},
    'p04_v7_cv_late_fusion_seed42': {'launcher_only': True, 'model_variant': 'cv_late_fusion', 'seed': 42, 'study': 'v7_context_factorial_120k_large_batch_v1', 'jobs_catalog_sha256': '987394f0e0a539a0ad5e0057aa0952474eecb50f374e8ea1410f9f3168ea746d', 'batch_plan_sha256': '42be6e8e7c7ebbe578de16646cd6339c963390d2ffae5ed1226980dea46fbc04', 'source_freeze_sha256': '2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783'},
    'p04_v7_cv_late_fusion_seed43': {'launcher_only': True, 'model_variant': 'cv_late_fusion', 'seed': 43, 'study': 'v7_context_factorial_120k_large_batch_v1', 'jobs_catalog_sha256': '987394f0e0a539a0ad5e0057aa0952474eecb50f374e8ea1410f9f3168ea746d', 'batch_plan_sha256': '42be6e8e7c7ebbe578de16646cd6339c963390d2ffae5ed1226980dea46fbc04', 'source_freeze_sha256': '2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783'},
    'p04_v7_fixed_gate_control_seed42': {'launcher_only': True, 'model_variant': 'fixed_gate_control', 'seed': 42, 'study': 'v7_context_factorial_120k_large_batch_v1', 'jobs_catalog_sha256': '987394f0e0a539a0ad5e0057aa0952474eecb50f374e8ea1410f9f3168ea746d', 'batch_plan_sha256': '42be6e8e7c7ebbe578de16646cd6339c963390d2ffae5ed1226980dea46fbc04', 'source_freeze_sha256': '2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783'},
    'p04_v7_fixed_gate_control_seed43': {'launcher_only': True, 'model_variant': 'fixed_gate_control', 'seed': 43, 'study': 'v7_context_factorial_120k_large_batch_v1', 'jobs_catalog_sha256': '987394f0e0a539a0ad5e0057aa0952474eecb50f374e8ea1410f9f3168ea746d', 'batch_plan_sha256': '42be6e8e7c7ebbe578de16646cd6339c963390d2ffae5ed1226980dea46fbc04', 'source_freeze_sha256': '2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783'},
}


def vicreg_loss(x, y):
    """Negative-free global-view regularizer, using per-microbatch statistics."""
    if x.shape[0] < 2:
        raise ValueError('VICReg needs at least two examples per microbatch')
    x, y = x.float(), y.float()
    invariance = F.mse_loss(x, y)
    variance = (F.relu(1-(x.var(0, unbiased=False)+1e-4).sqrt()).mean() +
                F.relu(1-(y.var(0, unbiased=False)+1e-4).sqrt()).mean())/2
    covariance = x.new_zeros(())
    for z in (x, y):
        z = z-z.mean(0)
        cov = z.T @ z / (z.shape[0]-1)
        offdiag = cov.masked_fill(torch.eye(z.shape[1], dtype=torch.bool, device=z.device), 0)
        covariance = covariance+offdiag.square().sum()/z.shape[1]
    # Normalize canonical 25:25:1 weights to keep the auxiliary scale explicit.
    return invariance + variance + covariance/25


class ResearchModel(ImmuneModel):
    def __new__(cls, config=None, variant='r00_a4_control'):
        if VARIANTS.get(variant, {}).get('launcher_only'):
            raise ValueError('Launcher identities cannot construct a model')
        if VARIANTS.get(variant, {}).get('diagnostic_only'):
            raise ValueError('Diagnostic variants cannot construct a model')
        if cls is ResearchModel and VARIANTS.get(variant, {}).get('architecture_refined'):
            from ..research_refined.model import RefinedModel
            return super().__new__(RefinedModel)
        if cls is ResearchModel and VARIANTS.get(variant, {}).get('architecture_v2'):
            from ..research_next.model import ArchitectureModel
            return super().__new__(ArchitectureModel)
        return super().__new__(cls)

    def __init__(self, config: ImmuneConfig, variant='r00_a4_control'):
        if config.ablation not in {'A4', 'V5'} or variant not in VARIANTS:
            raise ValueError('Research suite requires A4/V5 and an explicit registered variant')
        super().__init__(config)
        self.variant = variant
        self.options = dict(VARIANTS[variant])
        dim = config.d_model
        if self.options.get('qk_norm'):
            # Existing attention() applies these before head reshaping and RoPE.
            # This is whole-projection RMSNorm, not ViT-22B's exact LayerNorm.
            for block in self.backbone.transformer.blocks:
                block.q_norm = nn.RMSNorm(dim, eps=1e-6)
                block.k_norm = nn.RMSNorm(dim, eps=1e-6)
        count = self.options.get('registers', 0)
        self.registers = nn.Parameter(torch.randn(count, dim)*1e-6) if count else None
        if count:
            self.backbone.config.max_sequence_length = config.max_length+count
        self.conv_weight = nn.Parameter(torch.zeros(7, dim)) if self.options.get('local_conv') else None
        self.mask_film = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 2*dim)) if self.options.get('mask_film') else None
        if self.mask_film is not None:
            nn.init.zeros_(self.mask_film[-1].weight)
            nn.init.zeros_(self.mask_film[-1].bias)
        roles = len(CHAIN_ROLE_TO_ID)
        self.role_bias = nn.Parameter(torch.zeros(roles, roles, config.n_heads)) if self.options.get('role_bias') else None
        self.projector = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, 128)) if self.options.get('vicreg') else None

    def local_mix(self, x, chain_ids, attention_mask, content_mask):
        """Depthwise seven-position residual; never mix different chains or padding."""
        mixed = torch.zeros_like(x)
        valid = attention_mask.bool() & content_mask & (chain_ids >= 0)
        for index, offset in enumerate(range(-3, 4)):
            shifted = x.roll(offset, 1)
            same = (chain_ids == chain_ids.roll(offset, 1)) & valid & valid.roll(offset, 1)
            if offset > 0:
                same[:, :offset] = False
            elif offset < 0:
                same[:, offset:] = False
            mixed = mixed + shifted*self.conv_weight[index]*same.unsqueeze(-1)
        return x+mixed

    def denoise(self, input_ids, attention_mask, role_ids, scope_ids, chain_ids, position_ids,
                target_slot_mask, *, output_hidden_states=False):
        length = input_ids.shape[1]
        x = self.backbone.transformer.wte(input_ids)+self.role_embeddings(role_ids)+self.scope_embeddings(scope_ids)
        if self.conv_weight is not None:
            # Use visible token identity only; never residue_mask or true endpoints.
            content = (input_ids == self.tokenizer.mask_token_id) | self.allowed_target[input_ids]
            content = content & (input_ids != self.terminal_id)
            x = self.local_mix(x, chain_ids, attention_mask, content)
        if self.mask_film is not None:
            masked = (input_ids == self.tokenizer.mask_token_id) & target_slot_mask
            ratio = masked.sum(1).float()/target_slot_mask.sum(1).clamp_min(1)
            scale, shift = self.mask_film(ratio[:, None]).chunk(2, -1)
            x = x*(1+scale[:, None])+shift[:, None]
        if self.registers is not None:
            b, count = x.shape[0], self.registers.shape[0]
            x = torch.cat((x, self.registers[None].expand(b, -1, -1)), 1)
            attention_mask = F.pad(attention_mask, (0, count), value=1)
            chain_ids = F.pad(chain_ids, (0, count), value=-2)
            role_ids = F.pad(role_ids, (0, count), value=0)
            position_ids = F.pad(position_ids, (0, count), value=0)
        same = (chain_ids[:, :, None] == chain_ids[:, None, :]) & (chain_ids[:, :, None] >= 0) & (chain_ids[:, None, :] >= 0)
        bias = self.chain_bias[same.long()].permute(0, 3, 1, 2).contiguous()
        if self.role_bias is not None:
            bias = bias+self.role_bias[role_ids[:, :, None], role_ids[:, None, :]].permute(0, 3, 1, 2)
        result = self.backbone(input_ids=None, input_embeddings=x, attention_mask=attention_mask,
                               attention_bias=bias, position_ids=position_ids, use_cache=False,
                               output_hidden_states=output_hidden_states)
        logits = result.logits[:, :length].masked_fill(target_slot_mask[:, :, None] & ~self.allowed_target[None, None], float('-inf'))
        hidden = result.hidden_states[-1][:, :length] if output_hidden_states else None
        return logits, hidden

    def corrupt(self, batch, token_t=None):
        noise_batch = dict(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                           diffusion_eligible_mask=batch['eligible_mask'], diffusion_loss_mask=batch['eligible_mask'],
                           residue_mask=batch['residue_mask'])
        noised, targets, selected, times = sample_bioseq_diffusion_noise(noise_batch, self.tokenizer.mask_token_id, token_t=token_t)
        # Common deterministic validation always bypasses training-only mask variants.
        if not self.training:
            return noised, targets, selected
        for row in range(noised.shape[0]):
            eligible = batch['eligible_mask'][row]
            ids = batch['chain_ids'][row]
            if self.options.get('chain_mask') and torch.rand((), device=noised.device) < .5:
                chains = ids[eligible & (ids >= 0)].unique()
                if len(chains):
                    chain = chains[torch.randint(len(chains), (), device=noised.device)]
                    selected[row] = eligible & (ids == chain)
            if self.options.get('span_mask') and torch.rand((), device=noised.device) < .5:
                aa = eligible & batch['residue_mask'][row]
                selected[row, aa] = False
                for chain in ids[aa].unique():
                    positions = (aa & (ids == chain)).nonzero().flatten()
                    count = max(1, min(len(positions), int(torch.ceil(times[row]*len(positions)))))
                    remaining = count
                    # Short contiguous spans, clipped at the observed chain boundary.
                    while remaining:
                        available = (~selected[row, positions]).nonzero().flatten()
                        start = int(available[torch.randint(len(available), (), device=noised.device)])
                        span = int(torch.distributions.Geometric(probs=.33).sample().item())+1
                        indices = positions[start:start+min(span, remaining)]
                        newly = (~selected[row, indices]).sum().item()
                        selected[row, indices] = True
                        remaining -= int(newly)
        noised = batch['input_ids'].masked_fill(selected, self.tokenizer.mask_token_id)
        targets = batch['input_ids'].masked_fill(~selected, -100)
        return noised, targets, selected

    def forward(self, token_t=None, **batch):
        if not self.training and token_t is None:
            token_t = torch.ones_like(batch['input_ids'], dtype=torch.float)
        noised, targets, selected = self.corrupt(batch, token_t)
        if not torch.equal(targets[selected], batch['labels'][selected]):
            raise ValueError('Input/label mismatch')
        features = self.feature_args(batch)
        use_aux = self.projector is not None and self.training
        logits, hidden = self.denoise(**{**features, 'input_ids': noised}, output_hidden_states=use_aux)
        ce = F.cross_entropy(logits.transpose(1, 2).float(), targets, ignore_index=-100, reduction='none')
        aa = selected & batch['residue_mask']
        end = selected & batch['terminal_mask']
        def average(mask):
            return (ce*mask).sum()/mask.sum().clamp_min(1)
        loss = average(selected)
        if self.options.get('balanced_loss') and self.training:
            terms = []
            for row in range(ce.shape[0]):
                for uid in batch['chain_ids'][row][selected[row]].unique():
                    chain = (batch['chain_ids'][row] == uid) & selected[row]
                    aa_chain, end_chain = chain & aa[row], chain & end[row]
                    other = chain & ~aa[row] & ~end[row]
                    terms.append((ce[row]*aa_chain).sum()/aa_chain.sum().clamp_min(1) +
                                 .1*(ce[row]*end_chain).sum()/end_chain.sum().clamp_min(1) +
                                 (ce[row]*other).sum()/other.sum().clamp_min(1))
            loss = torch.stack(terms).mean()
        aux = loss.new_zeros(())
        if use_aux:
            # Two stochastic views of the same record; no artificial negative pairs.
            second, _, _ = self.corrupt(batch, token_t)
            _, second_hidden = self.denoise(**{**features, 'input_ids': second}, output_hidden_states=True)
            # Align receptor targets, not a shared long antigen/MHC shortcut.
            pool_mask = batch['residue_mask'] & batch['attention_mask'].bool() & batch['target_slot_mask']
            def pool(h):
                return (h*pool_mask.unsqueeze(-1)).sum(1)/pool_mask.sum(1, keepdim=True).clamp_min(1)
            aux = vicreg_loss(self.projector(pool(hidden)), self.projector(pool(second_hidden)))
            loss = loss+.05*aux
        metrics = torch.stack(((ce*aa).sum(1), aa.sum(1), (ce*end).sum(1), end.sum(1)), -1).detach()
        return dict(loss=loss, aa_loss=average(aa).detach(), terminal_loss=average(end).detach(),
                    aa_count=aa.sum().detach(), terminal_count=end.sum().detach(),
                    auxiliary_loss=aux.detach(), sample_metrics=metrics)

    def save_pretrained(self, directory):
        super().save_pretrained(directory)
        self.save_research_config(directory)

    def save_research_config(self, directory):
        (Path(directory)/'research_config.json').write_text(json.dumps(dict(variant=self.variant,
            options=self.options, base=asdict(self.config), schema='immune_research.v1'), indent=2)+'\n')

    @classmethod
    def from_pretrained(cls, directory):
        from safetensors.torch import load_file
        directory = Path(directory)
        settings = json.loads((directory/'research_config.json').read_text())
        if settings.get('schema') == 'immune_architecture.refined.v1':
            from ..research_refined.model import RefinedModel
            return RefinedModel.from_pretrained(directory)
        if settings.get('schema') == 'immune_architecture.v2':
            from ..research_next.model import ArchitectureModel
            return ArchitectureModel.from_pretrained(directory)
        if settings['schema'] != 'immune_research.v1' or settings['options'] != VARIANTS[settings['variant']]:
            raise ValueError('Research checkpoint schema/method changed')
        model = cls(ImmuneConfig(**settings['base']), settings['variant'])
        if json.loads((directory/'vocabulary.json').read_text()) != [model.tokenizer.token(i) for i in range(model.tokenizer.vocab_size)]:
            raise ValueError('Vocabulary changed')
        model.load_state_dict(load_file(str(directory/'model.safetensors')), strict=True)
        return model
