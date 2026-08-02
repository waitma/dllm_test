#!/usr/bin/env bash
# Run ImmunoMatch + ANARCI metrics on saved LLaDA light-pairing CSVs (skip CDR + generation).
#
# Usage:
#   bash scripts/downstream/run_grammar_v2_llada_pairing_metrics_only.sh esmc300m_cmp500k_llada
#   bash scripts/downstream/run_grammar_v2_llada_pairing_metrics_only.sh esmc600m_cmp500k_llada
#   bash scripts/downstream/run_grammar_v2_llada_pairing_metrics_only.sh all

set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
cd "${ROOT}"
python -u scripts/downstream/run_grammar_v2_pairing_metrics_only.py "${1:?usage: $0 <variant|all>}"
