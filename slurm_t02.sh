#!/bin/bash
# T02: replays only, no training. Covers every remaining thesis session that
# needs simulation data:
#   T07  gain occupancy and frozen gains of GS-D / GS-N for the new seeds 83-149
#        (the old seeds are in posthoc3/g5_* and g4_*_wide)
#   W17  (b) trajectories and (c) gain traces on three test paths
#   W18  mean |e_ct| per episode, all main arms, all ten seeds, ranged held-out
# W17 (a) path geometry is pure geometry and was exported locally
# (posthoc_path_geometry.py).
#
# Pin it to an EPYC 9334 node (cpunode13, cpunode39, cpunode04):
#
#   sbatch --nodelist=cpunode13.dyn.scc.plus.ac.at slurm_t02.sh
#
# Every replay is checked against the earlier evaluation rows (bit-identical
# E_FA / finished / duration), so the traces belong to the reported results.
# Each stage is skipped when its output exists. Roughly 1 hour on 64 cores.

#SBATCH --job-name=thesis-t02
#SBATCH --partition=base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --constraint=compute_nodes_cpu
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail

VENV="${VENV:-.venv}"
# LOCAL=1 runs it off the cluster as a provisional run (e.g. VENV=../venv LOCAL=1 bash slurm_t02.sh):
# results go to posthoc5_local and replay-check differences are reported, not fatal.
if [ -n "${LOCAL:-}" ]; then
  OUT="${OUT:-artifacts/results/posthoc5_local}"; WARN=(--warn-only)
else
  OUT="${OUT:-artifacts/results/posthoc5}"; WARN=()
fi
P2=artifacts/models/posthoc/plan2
P3=artifacts/models/posthoc/plan3
P4=artifacts/models/posthoc/plan4
R2=artifacts/results/posthoc2
R3=artifacts/results/posthoc3
R4=artifacts/results/posthoc4
MAN=manifests/posthoc
OLD=11,23,37,53,71
NEW=83,97,109,127,149
GSD=ppo_scheduler_ranged_h1_wide_2m
GSN=ppo_scheduler_nominal_h1_wide
DRD=ppo_direct_ranged_h1
DRN=ppo_direct_nominal_h1
# Three test paths at 0.5 m/s: gentlest (10019), median (10001), hardest (10014) by curvature.
PATHS=(10019 10001 10014)

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
# shellcheck disable=SC1091
if [ -f "${VENV}/bin/activate" ]; then source "${VENV}/bin/activate"; else source "${VENV}/Scripts/activate"; fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "host      : $(hostname)"
echo "cpu       : $( (lscpu 2>/dev/null || true) | grep 'Model name' | sed 's/.*: *//')"
echo "job       : ${SLURM_JOB_ID:-none}, ${SLURM_CPUS_PER_TASK:-?} cores"
echo "commit    : $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"
stage() { echo; echo "=== $1 ($(date -u +%H:%M:%S) UTC) ==="; }
done_in() { test -f "$1/$2"; }

for A in "${P3}/${GSD}/seed_71" "${P3}/${GSN}/seed_71" "${P2}/${DRD}/seed_71" "${P2}/${DRN}/seed_71" \
         "${P4}/${GSD}/seed_149" "${P4}/${GSN}/seed_149" "${P4}/${DRD}/seed_149" "${P4}/${DRN}/seed_149"; do
  test -f "${A}/best_model.zip" || { echo "missing ${A}"; exit 1; }
done
for F in "${R4}/heldout_ranged" "${R4}/dynamic" "${R3}/g4_heldout_ranged_g1" "${R3}/g4_heldout_ranged_g2" \
         "${R3}/g4_dynamic_g1" "${R3}/g4_dynamic_g2" "${R2}/f2_heldout_ranged" "${R2}/pidrn_f2_heldout_ranged" \
         "${R2}/f4_dynamic_existing" "${R2}/f4_dynamic_new" "${R2}/pidrn_f4_dynamic"; do
  test -f "${F}/episodes.csv" || { echo "missing ${F}/episodes.csv"; exit 1; }
done
mkdir -p "${OUT}"

stage "parity gate"
python check_parity.py

stage "T07: gain occupancy, new seeds"
done_in "${OUT}/occupancy_gsd_new" occupancy.csv || \
  python posthoc_gain_occupancy.py --models-root "${P4}" --seeds "${NEW}" \
    --arm "${GSD}=${MAN}/validation_ranged.json" --arm "${GSD}=${MAN}/held_out_ranged.json" \
    --output "${OUT}/occupancy_gsd_new"
done_in "${OUT}/occupancy_gsn_new" occupancy.csv || \
  python posthoc_gain_occupancy.py --models-root "${P4}" --seeds "${NEW}" \
    --arm "${GSN}=manifests/validation.json" --arm "${GSN}=${MAN}/held_out_ranged.json" \
    --output "${OUT}/occupancy_gsn_new"

stage "T07: frozen gains, new seeds (validation.json)"
done_in "${OUT}/frozen_gains_new" frozen_gains.json || \
  python posthoc_frozen_gains.py --models-root "${P4}" --seeds "${NEW}" --arm "${GSD}" --arm "${GSN}" \
    --output "${OUT}/frozen_gains_new"
if ! done_in "${OUT}/frozen_eval_new" episodes.csv; then
  python posthoc_eval.py --no-default --seeds "${NEW}" --manifest "${MAN}/held_out_ranged.json" \
    --frozen-gains "${OUT}/frozen_gains_new/frozen_gains.json" \
    --extra-arm "${GSD}=${P4}" --extra-arm "${GSN}=${P4}" --output "${OUT}/frozen_eval_new"
  python posthoc_replay_check.py "${OUT}/frozen_eval_new/episodes.csv" "${R4}/heldout_ranged/episodes.csv" \
    --output "${OUT}/frozen_eval_new/replay_check.json" || [ -n "${LOCAL:-}" ]
fi

ARMS=(--arm "${GSD}=${P3}:${OLD}" --arm "${GSD}=${P4}:${NEW}"
      --arm "${GSN}=${P3}:${OLD}" --arm "${GSN}=${P4}:${NEW}"
      --arm "${DRD}=${P2}:${OLD}" --arm "${DRD}=${P4}:${NEW}"
      --arm "${DRN}=${P2}:${OLD}" --arm "${DRN}=${P4}:${NEW}")

stage "W17 b/c + W18: ranged held-out, all 480 scenarios, traces on three paths"
TRACE=()
for P in "${PATHS[@]}"; do
  for C in stationary-nominal stationary-combined transient-delay transient-combined; do
    TRACE+=(--trace "held_out-random_curvature_${P}-v0.5-${C}-ranged")
  done
done
done_in "${OUT}/traces_heldout" episodes.csv || \
  python posthoc_traces.py --manifest "${MAN}/held_out_ranged.json" "${ARMS[@]}" "${TRACE[@]}" \
    --reference "${R2}/f2_heldout_ranged/episodes.csv" \
    --reference "${R3}/g4_heldout_ranged_g1/episodes.csv" --reference "${R3}/g4_heldout_ranged_g2/episodes.csv" \
    --reference "${R4}/heldout_ranged/episodes.csv" --reference "${R2}/pidrn_f2_heldout_ranged/episodes.csv" \
    "${WARN[@]+"${WARN[@]}"}" --output "${OUT}/traces_heldout"

stage "W17 c: dynamic delay on the three paths, all traced"
ONLY=()
for P in "${PATHS[@]}"; do ONLY+=(--only "held_out-random_curvature_${P}-v0.5-*"); done
done_in "${OUT}/traces_dynamic" episodes.csv || \
  python posthoc_traces.py --manifest "${MAN}/dynamic.json" "${ARMS[@]}" "${ONLY[@]}" --trace '*' \
    --reference "${R2}/f4_dynamic_existing/episodes.csv" --reference "${R2}/f4_dynamic_new/episodes.csv" \
    --reference "${R3}/g4_dynamic_g1/episodes.csv" --reference "${R3}/g4_dynamic_g2/episodes.csv" \
    --reference "${R4}/dynamic/episodes.csv" --reference "${R2}/pidrn_f4_dynamic/episodes.csv" \
    "${WARN[@]+"${WARN[@]}"}" --output "${OUT}/traces_dynamic"

stage "done"
echo "pack: tar czf t02_results.tgz ${OUT} slurm-thesis-t02-${SLURM_JOB_ID:-x}.out"
