#!/bin/bash
# Plan 3, follow-up: gain occupancy and frozen gains of the chosen scheduler
# GS-D-W 2M (thesis name "GS-D"). Replays existing checkpoints only; no
# training. Pin it to an EPYC 9334 node (cpunode13, cpunode39, cpunode04):
#
#   sbatch --nodelist=cpunode13.dyn.scc.plus.ac.at slurm_plan3b.sh
#
# Steps:
#   0. parity gate
#   1. gain occupancy on validation_ranged.json and the ranged held-out set
#   2. frozen gains from validation.json (same construction as G4 / F6)
#   3. frozen variants + the dynamic arm on the ranged held-out set; the dynamic
#      rows must reproduce g4_heldout_ranged_g2 bit for bit
#
# Roughly 20-30 minutes on 64 cores.

#SBATCH --job-name=thesis-plan3b
#SBATCH --partition=base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --constraint=compute_nodes_cpu
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail

VENV="${VENV:-.venv}"
OUT="${OUT:-artifacts/results/posthoc3}"
NEW=artifacts/models/posthoc/plan3
MAN=manifests/posthoc
ARM=ppo_scheduler_ranged_h1_wide_2m

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "host      : $(hostname)"
echo "cpu       : $(lscpu | grep 'Model name' | sed 's/.*: *//')"
echo "job       : ${SLURM_JOB_ID:-none}, ${SLURM_CPUS_PER_TASK:-?} cores"
echo "commit    : $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"
stage() { echo; echo "=== $1 ($(date -u +%H:%M:%S) UTC) ==="; }
done_in() { test -f "$1/$2"; }

test -f "${NEW}/${ARM}/seed_71/best_model.zip" || { echo "missing ${NEW}/${ARM}"; exit 1; }
test -f "${OUT}/g4_heldout_ranged_g2/episodes.csv" || { echo "missing ${OUT}/g4_heldout_ranged_g2"; exit 1; }

stage "parity gate"
python check_parity.py

stage "gain occupancy, GS-D-W 2M"
done_in "${OUT}/g5_occupancy_2m" occupancy.csv || \
  python posthoc_gain_occupancy.py --models-root "${NEW}" \
    --arm "${ARM}=${MAN}/validation_ranged.json" \
    --arm "${ARM}=${MAN}/held_out_ranged.json" \
    --output "${OUT}/g5_occupancy_2m"

stage "frozen gains, GS-D-W 2M (validation.json)"
done_in "${OUT}/g5_frozen_gains_2m" frozen_gains.json || \
  python posthoc_frozen_gains.py --models-root "${NEW}" --arm "${ARM}" --output "${OUT}/g5_frozen_gains_2m"

stage "frozen variants and dynamic arm on the ranged held-out set"
if ! done_in "${OUT}/g5_frozen_eval_2m" episodes.csv; then
  python posthoc_eval.py --no-default --manifest "${MAN}/held_out_ranged.json" \
    --frozen-gains "${OUT}/g5_frozen_gains_2m/frozen_gains.json" \
    --extra-arm "${ARM}=${NEW}" \
    --output "${OUT}/g5_frozen_eval_2m"
  python posthoc_replay_check.py "${OUT}/g5_frozen_eval_2m/episodes.csv" \
    "${OUT}/g4_heldout_ranged_g2/episodes.csv" --output "${OUT}/g5_frozen_eval_2m/replay_check.json"
fi

stage "done"
echo "pack: tar czf plan3b_results.tgz ${OUT}/g5_occupancy_2m ${OUT}/g5_frozen_gains_2m ${OUT}/g5_frozen_eval_2m slurm-thesis-plan3b-${SLURM_JOB_ID:-x}.out"
