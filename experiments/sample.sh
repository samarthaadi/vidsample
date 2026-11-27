#!/usr/bin/env bash
# Sample the experiment clips (raw). Usage: experiments/sample.sh OUT_DIR
set -euo pipefail
OUT=${1:?usage: sample.sh OUT_DIR}
S="python -m vidsample sample --seed 0 --workers 8 --out $OUT"

# AIGVDBench: official train split (20 open-source gens) and test split (31 gens incl. closed-source)
$S --dataset aigvdbench --split train --per-generator 15
$S --dataset aigvdbench --split test  --per-generator 8
# GenBuster-200K: train / test (same 4 gens) / benchmark (8 closed-source gens)
$S --dataset genbuster200k --split train     --per-generator 75
$S --dataset genbuster200k --split test      --per-generator 25
$S --dataset genbuster200k --split benchmark --per-generator 15
# GenVidBench: pair1 (vript real) and pair2 (HD-VG real, stream-only, clips can be huge)
$S --dataset genvidbench --split pair1 --per-generator 100
$S --dataset genvidbench --split pair2 --per-generator 25 --real 100 --max-clip-mb 40 --max-stream-gb 6
# GenVideo-100K: stream-only, no usable official split
$S --dataset genvideo100k --per-generator 40 --max-stream-gb 2
