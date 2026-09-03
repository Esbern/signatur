#!/usr/bin/env bash
# Run the full Phase 1 pipeline. Assumes `mamba activate signatur`.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-config.yaml}"
python -m signatur all --config "$CONFIG"
