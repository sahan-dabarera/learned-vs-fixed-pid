#!/bin/bash
# Plan 4: confirmation seeds. Five more training seeds for the four learned
# arms of the thesis comparison, with every setting frozen at the value the
# five-seed runs used. No selection is redone: the arms, boxes, budgets and
# checkpoint rule are the plan-3 outcome. Pin it to an EPYC 9334 node that
# reproduced the sealed run (cpunode13, cpunode39, cpunode04):
#
#   sbatch --nodelist=cpunode13.dyn.scc.plus.ac.at slurm_plan4.sh
#
# New seeds (declared before any run): 83, 97, 109, 127, 149.
#
# Arms (same names and arguments as their five-seed runs):
#   ppo_scheduler_ranged_h1_wide_2m  GS-D-W 2M  (thesis GS-D)  as slurm_plan3.sh
#   ppo_scheduler_nominal_h1_wide    GS-N-W     (thesis GS-N)  as slurm_plan3.sh
#   ppo_direct_ranged_h1             DR-D                      as slurm_plan2.sh
#   ppo_direct_nominal_h1            DR-N                      as slurm_plan2.sh
#
# Steps:
#   0. parity gate
#   1. 20 training runs in parallel
#   2. validation scores of the kept checkpoints
#   3. the plan-3 G4 evaluation set for the new seeds
#
# Every stage is skipped when its output exists, so a resubmit continues.
# Roughly 4-5 hours on 64 cores (the 2M scheduler runs set the pace).

#SBATCH --job-name=thesis-plan4
#SBATCH --partition=base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --constraint=compute_nodes_cpu
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail

VENV="${VENV:-.venv}"
OUT="${OUT:-artifacts/results/posthoc4}"
NEW="${NEW:-artifacts/models/posthoc/plan4}"
MAN=manifests/posthoc
WIDE=artifacts/calibration/blind_ppo_wide.json
RATE=147.5,0.75,1.75
SEEDS=83,97,109,127,149
LAST=149

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "host      : $(hostname)"
echo "cpu       : $(lscpu | grep 'Model name' | sed 's/.*: *//')"
echo "job       : ${SLURM_JOB_ID:-none}, ${SLURM_CPUS_PER_TASK:-?} cores"
echo "commit    : $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"
echo "seeds     : ${SEEDS}"
stage() { echo; echo "=== $1 ($(date -u +%H:%M:%S) UTC) ==="; }
done_in() { test -f "$1/$2"; }

mkdir -p "${OUT}"
for F in held_out_ranged validation_ranged dynamic delay_wide noise_coarse noise_fine; do
  test -f "${MAN}/${F}.json" || { echo "missing ${MAN}/${F}.json"; exit 1; }
done
test -f "${WIDE}" || { echo "missing ${WIDE}"; exit 1; }

stage "parity gate"
python check_parity.py

stage "training"
SAMPLER=(--train-arg=--train-delay-range-s=0,0.3
         --train-arg=--train-noise-min-m=0.0001
         --train-arg=--train-noise-max-m=0.01
         --validation-manifest "${MAN}/validation_ranged.json")
WIDEBOX=(--scheduler-calibration "${WIDE}" --train-arg=--gain-rate-limit=${RATE})
H1=(--train-arg=--history-length=1)
PIDS=()
train() {  # arm, mode, args...
  local arm=$1 mode=$2; shift 2
  local kind=${arm#ppo_}; kind=${kind%%_*}
  if [ -f "${NEW}/${arm}/seed_${LAST}/metadata.json" ]; then
    echo "  ${arm}: already trained"; return
  fi
  python run_seeds.py --phase final --seeds "${SEEDS}" --only-mode "${mode}" \
    --arm-template "ppo_{kind}_${arm#ppo_${kind}_}" --output "${NEW}" "$@" \
    > "${OUT}/training_${arm}.log" 2>&1 &
  PIDS+=($!)
  echo "  ${arm} -> ${NEW} (log ${OUT}/training_${arm}.log)"
}
train ppo_scheduler_ranged_h1_wide_2m scheduled --only-regime disturbed "${SAMPLER[@]}" "${H1[@]}" \
  "${WIDEBOX[@]}" --scheduler-timesteps 2000000
train ppo_scheduler_nominal_h1_wide   scheduled --only-regime nominal "${H1[@]}" "${WIDEBOX[@]}"
train ppo_direct_ranged_h1            direct    --only-regime disturbed "${SAMPLER[@]}" "${H1[@]}"
train ppo_direct_nominal_h1           direct    --only-regime nominal "${H1[@]}"
for PID in ${PIDS[@]+"${PIDS[@]}"}; do wait "${PID}"; done
for LOG in "${OUT}"/training_*.log; do echo "--- ${LOG}"; tail -n 3 "${LOG}"; done

ARMS=(ppo_scheduler_ranged_h1_wide_2m ppo_scheduler_nominal_h1_wide ppo_direct_ranged_h1 ppo_direct_nominal_h1)
for A in "${ARMS[@]}"; do
  test -f "${NEW}/${A}/seed_${LAST}/best_model.zip" || { echo "missing ${NEW}/${A}/seed_${LAST}"; exit 1; }
done

stage "validation scores of the kept checkpoints"
VARGS=()
for A in "${ARMS[@]}"; do VARGS+=(--arm "${A}=${NEW}"); done
done_in "${OUT}/validation" validation_selected.csv || \
  python posthoc_validation_check.py --seeds "${SEEDS}" --output "${OUT}/validation" "${VARGS[@]}"

stage "evaluation (plan-3 G4 set)"
EARGS=(--no-default --seeds "${SEEDS}")
for A in "${ARMS[@]}"; do EARGS+=(--extra-arm "${A}=${NEW}"); done
done_in "${OUT}/heldout_ranged" episodes.csv || \
  python posthoc_eval.py "${EARGS[@]}" --manifest "${MAN}/held_out_ranged.json" --output "${OUT}/heldout_ranged"
done_in "${OUT}/heldout_fixed" episodes.csv || \
  python posthoc_eval.py "${EARGS[@]}" --manifest manifests/held_out.json --output "${OUT}/heldout_fixed"
done_in "${OUT}/delay" episodes.csv || \
  python posthoc_eval.py "${EARGS[@]}" --manifest "${MAN}/delay_wide.json" --output "${OUT}/delay"
done_in "${OUT}/dynamic" episodes.csv || \
  python posthoc_eval.py "${EARGS[@]}" --manifest "${MAN}/dynamic.json" --output "${OUT}/dynamic"
done_in "${OUT}/noise" episodes.csv || \
  python posthoc_eval.py "${EARGS[@]}" --manifest "${MAN}/noise_coarse.json" --manifest "${MAN}/noise_fine.json" \
    --output "${OUT}/noise"

stage "done"
echo "pack: tar czf plan4_results.tgz --exclude=tensorboard ${OUT} ${NEW} slurm-thesis-plan4-${SLURM_JOB_ID:-x}.out"
