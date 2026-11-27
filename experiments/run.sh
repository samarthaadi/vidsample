#!/usr/bin/env bash
# Full experiment pipeline. Usage: experiments/run.sh EXP_DIR
#   1. sample raw clips   2. normalize locally   3. features   4. evaluate -> EXP_DIR/RESULTS.md
set -euo pipefail
EXP=${1:?usage: run.sh EXP_DIR}
cd "$(dirname "$0")/.."
[ -f "$EXP/raw/manifest.csv" ] || experiments/sample.sh "$EXP/raw"
python experiments/normalize_local.py "$EXP"
python experiments/features.py "$EXP" raw norm
python experiments/evaluate.py "$EXP" "$EXP/RESULTS.md"
