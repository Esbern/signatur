#!/usr/bin/env bash
# Full-Denmark, unattended overnight run:
#   1. baseline (config.yaml)       - re-extract with the 80/20 split + honest holdout
#   2. climate  (config_climate.yaml) - same, then the full production fuzzy build
# Stops at the first failure (set -e) so later stages never run on bad upstream data.
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/holmes/miniforge3/etc/profile.d/conda.sh
conda activate signatur

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] === BASELINE (config.yaml): rasterize + extract + holdout ==="
python -u -m signatur rasterize --config config.yaml --force
python -u -m signatur extract   --config config.yaml --force
python -u -m signatur holdout   --config config.yaml --force

echo "[$(ts)] === CLIMATE (config_climate.yaml): full pipeline ==="
python -u -m signatur rasterize  --config config_climate.yaml --force
python -u -m signatur extract    --config config_climate.yaml --force
python -u -m signatur holdout    --config config_climate.yaml --force
python -u -m signatur model      --config config_climate.yaml --force
python -u -m signatur project    --config config_climate.yaml --force
python -u -m signatur richness   --config config_climate.yaml --force
python -u -m signatur fuzzyplots --config config_climate.yaml --force

echo "[$(ts)] === ALL DONE ==="
