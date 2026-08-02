#!/usr/bin/env bash
# Run volc with proxies cleared (Volc ML API must not go through lab HTTPS proxy).
# Usage: bash scripts/volc-no-proxy.sh ml_task list -n bioseq ...
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY || true
export VOLC_DEFAULT_OUTPUT="${VOLC_DEFAULT_OUTPUT:-json}"
exec volc "$@"
