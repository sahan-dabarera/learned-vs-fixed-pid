#!/bin/bash
# Post-hoc experiments, stage 1: evaluation only, no training.
#
#   sbatch slurm_posthoc.sh
#   sbatch --nodelist=cpunode13 slurm_posthoc.sh        # same node as the sealed run
#   sbatch --export=ALL,VENV=$HOME/venv312 slurm_posthoc.sh
#
# Runs, in one allocation so every number comes from one node:
#   0. fixed-PID parity gate
#   1. E3a  frozen scheduler gains from the VALIDATION manifest
#   2. E1   noise sweep (coarse), noise-only and combined, both modes
#   3. E2   delay sweep, both modes
#   4. E3b  frozen-gain schedulers on the full held-out manifest
#   5. E1-F PID-R with a slower derivative filter, noise-only
#   6. replay check of every sweep row identical to a sealed scenario
#
# About 65,000 episodes at ~2.5 s each, roughly 45 core-hours: about one hour
# on 64 cores. Nothing is written to artifacts/results/final or
# artifacts/models/final; posthoc_eval.py refuses to.

#SBATCH --job-name=thesis-posthoc-s1
#SBATCH --partition=base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --constraint=compute_nodes_cpu
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail

VENV="${VENV:-.venv}"
OUT="${OUT:-artifacts/results/posthoc}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none} on ${SLURM_JOB_PARTITION:-none}"
echo "cores     : ${SLURM_CPUS_PER_TASK:-unknown} allocated"
echo "commit    : $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"
echo "python    : $(python -V 2>&1)"
echo "output    : ${OUT}"
echo

stage() { echo; echo "=== $1 ($(date -u +%H:%M:%S) UTC) ==="; }

stage "parity gate"
python check_parity.py

stage "E3a frozen gains (validation manifest)"
python posthoc_frozen_gains.py --output "${OUT}/e3_frozen_gains"

stage "E1 noise sweep, coarse"
python posthoc_eval.py \
  --manifest manifests/posthoc/noise_coarse.json \
  --output "${OUT}/e1_noise_coarse"

stage "E2 delay sweep"
python posthoc_eval.py \
  --manifest manifests/posthoc/delay.json \
  --output "${OUT}/e2_delay"

stage "E3b frozen-gain schedulers on held-out"
python posthoc_eval.py \
  --manifest manifests/held_out.json \
  --no-default \
  --frozen-gains "${OUT}/e3_frozen_gains/frozen_gains.json" \
  --output "${OUT}/e3_frozen_eval"

stage "E1-F PID-R derivative filter check"
python posthoc_eval.py \
  --manifest manifests/posthoc/noise_coarse.json \
  --no-default \
  --robust-tau-d 0.02 --robust-tau-d 0.05 \
  --condition noise \
  --output "${OUT}/e1f_taud"

stage "replay check against the sealed episodes"
python posthoc_check_sealed.py "${OUT}/e1_noise_coarse" "${OUT}/e2_delay"

stage "done"
echo "Copy ${OUT} back to the laptop:"
echo "  rsync -av SERVER:thesis-experiment/${OUT} ./artifacts/results/"
