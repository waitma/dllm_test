"""Declared test-only scope shared by the release verifier and training gate.

Inspect with Python in conda pllm; this module does not run annotation or training.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCOPE = 'immune_test_only_observed_pair_exact_v3'
WAIVERS = ['legacy_v7_264_unresolved_antibody_H3', 'GDPa1_dedicated_protection_evidence',
           'Specificity_dedicated_protection_evidence', 'm396_dedicated_protection_evidence']
EXCLUDED = {'tcr_representation/train', 'uptcr/few-shot/train', 'uptcr/few-shot/valid',
            'uptcr/majority/train', 'uptcr/majority/valid', 'uptcr/zero-shot/train', 'uptcr/zero-shot/valid'}
T4 = [ROOT / 'downstream/benchmark/data/tcr_generation_bench/eval_conditional.json',
      ROOT / 'downstream/benchmark/data/tcr_design/eval.json']


def required_test_paths():
    from scripts.data.audit_tcr_exposure import input_specs
    return [p.resolve() for name, p, _, _ in input_specs() if name not in EXCLUDED] + T4
