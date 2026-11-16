#!/bin/bash
# Experiment plan 3: scheduler gain box (G1) and training budget (G2).
# Pin it to an EPYC 9334 node that reproduced the sealed run (cpunode13,
# cpunode39, cpunode04):
#
#   sbatch --nodelist=cpunode13.dyn.scc.plus.ac.at slurm_plan3.sh
#
# Every stage writes its own directory under ${OUT} and is skipped when that
# directory already holds its result, so a resubmit continues where it stopped.
#
# New scheduler arms (all H = 1 unless named h10; ranged = plan-2 ranged sampler):
#   ppo_scheduler_ranged_h1_wide     GS-D-W       wide box, 300k          (G1, main candidate)
#   ppo_scheduler_nominal_h1_wide    GS-N-W       wide box, 300k          (G1)
#   ppo_scheduler_ranged_h10_wide    GS-D-W H10   wide box, 300k          (G1)
#   ppo_scheduler_ranged_h1_2m       GS-D 2M      plan-2 box, 2M steps    (G2)
#   ppo_scheduler_ranged_h1_wide_2m  GS-D-W 2M    wide box, 2M steps      (G2)
# Wide box = artifacts/calibration/blind_ppo_wide.json (lower bounds Kp 1,
# Ki 0, Kd 0; upper bounds and g0 unchanged). Its rate limit is held at the
# plan-2 values (147.5, 0.75, 1.75 per s) instead of half the new range.
#
# Steps:
#   0. parity gate
#   1. all 25 training runs in the background
#   2. beside them: G0.2 gain occupancy of the plan-2 schedulers, and a replay
#      check that the updated evaluation code reproduces plan-2 rows
#   3. when G1 is done: G4 for the three G1 arms (G2 still training)
#   4. when G2 is done: G0.3/G3 validation lookup, G4 for the two G2 arms
#
# Roughly 6-7 hours on 64 cores (the 2M runs set the pace).

#SBATCH --job-name=thesis-plan3
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
OUT="${OUT:-artifacts/results/posthoc3}"
P2OUT=artifacts/results/posthoc2
P2=artifacts/models/posthoc/plan2
NEW="${NEW:-artifacts/models/posthoc/plan3}"
MAN=manifests/posthoc
WIDE=artifacts/calibration/blind_ppo_wide.json
RATE=147.5,0.75,1.75


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

mkdir -p "${OUT}"
for F in held_out_ranged validation_ranged dynamic delay_wide noise_coarse noise_fine; do
  test -f "${MAN}/${F}.json" || { echo "missing ${MAN}/${F}.json"; exit 1; }
done
test -f "${WIDE}" || { echo "missing ${WIDE}"; exit 1; }
test -f "${P2OUT}/f2_heldout_ranged/episodes.csv" || { echo "missing plan-2 results in ${P2OUT}"; exit 1; }
test -d "${P2}/ppo_scheduler_ranged_h1" || { echo "missing plan-2 models in ${P2}"; exit 1; }

stage "parity gate"
python check_parity.py

stage "training in the background"
RANGED=(--only-regime disturbed
        --train-arg=--train-delay-range-s=0,0.3
        --train-arg=--train-noise-min-m=0.0001
        --train-arg=--train-noise-max-m=0.01
        --validation-manifest "${MAN}/validation_ranged.json")
WIDEBOX=(--scheduler-calibration "${WIDE}" --train-arg=--gain-rate-limit=${RATE})
H1=(--train-arg=--history-length=1)
G1_PIDS=()
G2_PIDS=()
train() {  # name, log, args...
  local name=$1 log=$2; shift 2
  if [ -f "${NEW}/ppo_scheduler_${name}/seed_71/metadata.json" ]; then
    echo "  ${name}: already trained"; return 1
  fi
  python run_seeds.py --phase final --only-mode scheduled --arm-template "ppo_{kind}_${name}" \
    --output "${NEW}" "$@" > "${OUT}/${log}" 2>&1 &
  LAST_PID=$!
  echo "  ${name} -> ${NEW} (log ${OUT}/${log})"
}
if train ranged_h1_wide   training_ranged_h1_wide.log   "${RANGED[@]}" "${H1[@]}" "${WIDEBOX[@]}"; then G1_PIDS+=("${LAST_PID}"); fi
if train nominal_h1_wide  training_nominal_h1_wide.log  --only-regime nominal "${H1[@]}" "${WIDEBOX[@]}"; then G1_PIDS+=("${LAST_PID}"); fi
if train ranged_h10_wide  training_ranged_h10_wide.log  "${RANGED[@]}" "${WIDEBOX[@]}"; then G1_PIDS+=("${LAST_PID}"); fi
if train ranged_h1_2m     training_ranged_h1_2m.log     "${RANGED[@]}" "${H1[@]}" --scheduler-timesteps 2000000; then G2_PIDS+=("${LAST_PID}"); fi
if train ranged_h1_wide_2m training_ranged_h1_wide_2m.log "${RANGED[@]}" "${H1[@]}" "${WIDEBOX[@]}" --scheduler-timesteps 2000000; then G2_PIDS+=("${LAST_PID}"); fi

stage "G0.2 gain occupancy, plan-2 schedulers (beside the training)"
done_in "${OUT}/g0_occupancy" occupancy.csv || \
  python posthoc_gain_occupancy.py --models-root "${P2}" --workers 20 \
    --arm "ppo_scheduler_ranged_h1=${MAN}/validation_ranged.json" \
    --arm "ppo_scheduler_ranged_h1=${MAN}/held_out_ranged.json" \
    --arm "ppo_scheduler_nominal_h1=manifests/validation.json" \
    --arm "ppo_scheduler_nominal_h1=${MAN}/held_out_ranged.json" \
    --output "${OUT}/g0_occupancy"

stage "replay check: updated evaluation code vs plan-2 rows (beside the training)"
if ! done_in "${OUT}/replay_plan2" replay_check.json; then
  python posthoc_eval.py --no-default --manifest "${MAN}/held_out_ranged.json" --workers 20 \
    --extra-arm "ppo_scheduler_ranged_h1=${P2}" --extra-arm "ppo_scheduler_nominal_h1=${P2}" \
    --extra-arm "ppo_direct_ranged_h1=${P2}" \
    --output "${OUT}/replay_plan2"
  python posthoc_replay_check.py "${OUT}/replay_plan2/episodes.csv" "${P2OUT}/f2_heldout_ranged/episodes.csv" \
    --output "${OUT}/replay_plan2/replay_check.json"
fi

evaluate() {  # tag, arm names...
  local tag=$1; shift
  local arms=()
  for A in "$@"; do arms+=(--extra-arm "${A}=${NEW}"); done
  done_in "${OUT}/g4_heldout_ranged_${tag}" episodes.csv || \
    python posthoc_eval.py --no-default --manifest "${MAN}/held_out_ranged.json" "${arms[@]}" \
      --output "${OUT}/g4_heldout_ranged_${tag}"
  done_in "${OUT}/g4_heldout_fixed_${tag}" episodes.csv || \
    python posthoc_eval.py --no-default --manifest manifests/held_out.json "${arms[@]}" \
      --output "${OUT}/g4_heldout_fixed_${tag}"
  done_in "${OUT}/g4_delay_${tag}" episodes.csv || \
    python posthoc_eval.py --no-default --manifest "${MAN}/delay_wide.json" "${arms[@]}" \
      --output "${OUT}/g4_delay_${tag}"
  done_in "${OUT}/g4_dynamic_${tag}" episodes.csv || \
    python posthoc_eval.py --no-default --manifest "${MAN}/dynamic.json" "${arms[@]}" \
      --output "${OUT}/g4_dynamic_${tag}"
  done_in "${OUT}/g4_noise_${tag}" episodes.csv || \
    python posthoc_eval.py --no-default --manifest "${MAN}/noise_coarse.json" --manifest "${MAN}/noise_fine.json" \
      "${arms[@]}" --output "${OUT}/g4_noise_${tag}"
}

stage "waiting for G1 training"
for PID in ${G1_PIDS[@]+"${G1_PIDS[@]}"}; do wait "${PID}"; done
for LOG in "${OUT}"/training_*wide.log; do echo "--- ${LOG}"; tail -n 3 "${LOG}"; done

stage "G4 for the G1 arms (G2 still training)"
evaluate g1 ppo_scheduler_ranged_h1_wide ppo_scheduler_nominal_h1_wide ppo_scheduler_ranged_h10_wide

stage "G4 frozen gains and gain occupancy, wide-box schedulers"
done_in "${OUT}/g4_frozen_gains" frozen_gains.json || \
  python posthoc_frozen_gains.py --models-root "${NEW}" \
    --arm ppo_scheduler_ranged_h1_wide --arm ppo_scheduler_nominal_h1_wide \
    --output "${OUT}/g4_frozen_gains"
done_in "${OUT}/g4_frozen_eval" episodes.csv || \
  python posthoc_eval.py --no-default --manifest "${MAN}/held_out_ranged.json" \
    --frozen-gains "${OUT}/g4_frozen_gains/frozen_gains.json" \
    --extra-arm "ppo_scheduler_ranged_h1_wide=${NEW}" --extra-arm "ppo_scheduler_nominal_h1_wide=${NEW}" \
    --output "${OUT}/g4_frozen_eval"
done_in "${OUT}/g4_occupancy_wide" occupancy.csv || \
  python posthoc_gain_occupancy.py --models-root "${NEW}" \
    --arm "ppo_scheduler_ranged_h1_wide=${MAN}/validation_ranged.json" \
    --arm "ppo_scheduler_ranged_h1_wide=${MAN}/held_out_ranged.json" \
    --arm "ppo_scheduler_nominal_h1_wide=manifests/validation.json" \
    --arm "ppo_scheduler_nominal_h1_wide=${MAN}/held_out_ranged.json" \
    --output "${OUT}/g4_occupancy_wide"

stage "waiting for G2 training"
for PID in ${G2_PIDS[@]+"${G2_PIDS[@]}"}; do wait "${PID}"; done
for LOG in "${OUT}"/training_*2m.log; do echo "--- ${LOG}"; tail -n 3 "${LOG}"; done

stage "G0.3 / G3 validation scores of the selected checkpoints"
python posthoc_validation_check.py --output "${OUT}/g3_validation" \
  --arm "ppo_scheduler_ranged_h1=${P2}" --arm "ppo_scheduler_nominal_h1=${P2}" \
  --arm "ppo_direct_ranged_h1=${P2}" --arm "ppo_direct_nominal_h1=${P2}" \
  --arm "ppo_scheduler_ranged_h10=${P2}" \
  --arm "ppo_scheduler_ranged_h1_wide=${NEW}" --arm "ppo_scheduler_nominal_h1_wide=${NEW}" \
  --arm "ppo_scheduler_ranged_h10_wide=${NEW}" \
  --arm "ppo_scheduler_ranged_h1_2m=${NEW}" --arm "ppo_scheduler_ranged_h1_wide_2m=${NEW}"

stage "G4 for the G2 arms"
evaluate g2 ppo_scheduler_ranged_h1_2m ppo_scheduler_ranged_h1_wide_2m

stage "done"
echo "pack: tar czf plan3_results.tgz --exclude=tensorboard ${OUT} ${NEW} slurm-thesis-plan3-${SLURM_JOB_ID:-x}.out"
