#!/bin/bash
# Experiment plan 2: H = 1 as the main design, delay and noise drawn from
# ranges, time-varying delay. Pin it to an EPYC 9334 node that reproduced the
# sealed run (cpunode13, cpunode39, cpunode04):
#
#   sbatch --nodelist=cpunode39.dyn.scc.plus.ac.at slurm_plan2.sh
#
# Every stage writes its own directory under ${OUT} and is skipped when that
# directory already holds its result, so a resubmit continues where it stopped.
#
# Steps:
#   0. parity gate
#   1. training in the background, 30 runs:
#        ppo_{kind}_nominal_h1   nominal sampler, H = 1                  (F1)
#        ppo_{kind}_ranged_h1    delay U[0, 0.3] s, noise logU[0.1, 10] mm, H = 1  (main)
#        ppo_{kind}_ranged_h10   same sampler, H = 10                    (history ablation)
#      the ranged arms select checkpoints on manifests/posthoc/validation_ranged.json
#   2. beside it: PID-RN retune on the ranged validation subset, then the
#      existing arms on the wide delay sweep (F3) and the dynamic test (F4)
#   3. after training: F0 validation lookup, F2 held-out (ranged + fixed),
#      F3/F4 for the new arms, F5 noise sweeps, F6 frozen gains
#
# Roughly 6-7 hours on 64 cores.

#SBATCH --job-name=thesis-plan2
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
OUT="${OUT:-artifacts/results/posthoc2}"
MODELS="${MODELS:-artifacts/models/posthoc}"
NEW="${MODELS}/plan2"
MAN=manifests/posthoc

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
test -d "${MODELS}/e5_history" || { echo "missing ${MODELS}/e5_history (E5 H=1 checkpoints)"; exit 1; }

stage "parity gate"
python check_parity.py

stage "training in the background"
RANGED=(--train-arg=--history-length=1)
SAMPLER=(--train-arg=--train-delay-range-s=0,0.3
         --train-arg=--train-noise-min-m=0.0001
         --train-arg=--train-noise-max-m=0.01
         --validation-manifest "${MAN}/validation_ranged.json")
PIDS=()
train() {  # name, log, args...
  local name=$1 log=$2; shift 2
  if [ -f "${NEW}/ppo_scheduler_${name}/seed_71/metadata.json" ] && \
     [ -f "${NEW}/ppo_direct_${name}/seed_71/metadata.json" ]; then
    echo "  ${name}: already trained"; return
  fi
  python run_seeds.py --phase final --arm-template "ppo_{kind}_${name}" --output "${NEW}" "$@" \
    > "${OUT}/${log}" 2>&1 &
  PIDS+=($!)
  echo "  ${name} -> ${NEW} (log ${OUT}/${log})"
}
train nominal_h1 training_nominal_h1.log --only-regime nominal --train-arg=--history-length=1
train ranged_h1  training_ranged_h1.log  --only-regime disturbed "${RANGED[@]}" "${SAMPLER[@]}"
train ranged_h10 training_ranged_h10.log --only-regime disturbed "${SAMPLER[@]}"

stage "PID-RN retune on the ranged validation subset (beside the training)"
done_in "${OUT}/pid_retune" pid_rn.json || \
  python posthoc_pid_retune.py --manifest "${MAN}/validation_ranged.json" --workers 30 \
    --output "${OUT}/pid_retune"
PIDRN=$(python -c "import json;s=json.load(open('${OUT}/pid_retune/pid_rn.json'))['selected'];print(f\"pid_global_robust_noise={s['kp']},{s['ki']},{s['kd']},{s['tau_d']}\")")
echo "  ${PIDRN}"

OLD=(--extra-arm "ppo_scheduler_disturbed_h1=${MODELS}/e5_history"
     --extra-arm "ppo_direct_disturbed_h1=${MODELS}/e5_history"
     --pid "${PIDRN}")

stage "F3 wide delay sweep, existing arms (beside the training)"
if ! done_in "${OUT}/f3_delay_existing" episodes.csv; then
  python posthoc_eval.py --manifest "${MAN}/delay_wide.json" "${OLD[@]}" --workers 30 \
    --output "${OUT}/f3_delay_existing"
  python posthoc_check_sealed.py "${OUT}/f3_delay_existing"
fi

stage "F4 dynamic delay, existing arms (beside the training)"
done_in "${OUT}/f4_dynamic_existing" episodes.csv || \
  python posthoc_eval.py --manifest "${MAN}/dynamic.json" "${OLD[@]}" --workers 30 \
    --output "${OUT}/f4_dynamic_existing"

stage "waiting for training"
for PID in ${PIDS[@]+"${PIDS[@]}"}; do wait "${PID}"; done
for LOG in "${OUT}"/training_*.log; do echo "--- ${LOG}"; tail -n 3 "${LOG}"; done

NEWARMS=()
for NAME in nominal_h1 ranged_h1 ranged_h10; do
  NEWARMS+=(--extra-arm "ppo_scheduler_${NAME}=${NEW}" --extra-arm "ppo_direct_${NAME}=${NEW}")
done

stage "F0 validation scores of the selected checkpoints"
python posthoc_validation_check.py --output "${OUT}/f0_validation" \
  --arm "ppo_scheduler_disturbed=artifacts/models/final" \
  --arm "ppo_direct_disturbed=artifacts/models/final" \
  --arm "ppo_scheduler_nominal=artifacts/models/final" \
  --arm "ppo_direct_nominal=artifacts/models/final" \
  --arm "ppo_scheduler_disturbed_h1=${MODELS}/e5_history" \
  --arm "ppo_direct_disturbed_h1=${MODELS}/e5_history" \
  --arm "ppo_scheduler_nominal_h1=${NEW}" --arm "ppo_direct_nominal_h1=${NEW}" \
  --arm "ppo_scheduler_ranged_h1=${NEW}" --arm "ppo_direct_ranged_h1=${NEW}" \
  --arm "ppo_scheduler_ranged_h10=${NEW}" --arm "ppo_direct_ranged_h10=${NEW}"

stage "F2 held-out, ranged (main table): every arm, one job"
done_in "${OUT}/f2_heldout_ranged" episodes.csv || \
  python posthoc_eval.py --manifest "${MAN}/held_out_ranged.json" "${OLD[@]}" "${NEWARMS[@]}" \
    --output "${OUT}/f2_heldout_ranged"

stage "F2 held-out, fixed severities (comparison + sealed replay check)"
if ! done_in "${OUT}/f2_heldout_fixed" episodes.csv; then
  python posthoc_eval.py --manifest manifests/held_out.json "${OLD[@]}" "${NEWARMS[@]}" \
    --output "${OUT}/f2_heldout_fixed"
  python posthoc_check_sealed.py "${OUT}/f2_heldout_fixed"
fi

stage "F3 wide delay sweep, new arms"
done_in "${OUT}/f3_delay_new" episodes.csv || \
  python posthoc_eval.py --no-default --manifest "${MAN}/delay_wide.json" "${NEWARMS[@]}" \
    --output "${OUT}/f3_delay_new"

stage "F4 dynamic delay, new arms"
done_in "${OUT}/f4_dynamic_new" episodes.csv || \
  python posthoc_eval.py --no-default --manifest "${MAN}/dynamic.json" "${NEWARMS[@]}" \
    --output "${OUT}/f4_dynamic_new"

stage "F6 frozen gains of the H = 1 schedulers (validation manifest)"
done_in "${OUT}/f6_frozen_gains" frozen_gains.json || \
  python posthoc_frozen_gains.py --models-root "${NEW}" \
    --arm ppo_scheduler_nominal_h1 --arm ppo_scheduler_ranged_h1 \
    --output "${OUT}/f6_frozen_gains"
done_in "${OUT}/f6_frozen_eval" episodes.csv || \
  python posthoc_eval.py --no-default --manifest "${MAN}/held_out_ranged.json" \
    --frozen-gains "${OUT}/f6_frozen_gains/frozen_gains.json" \
    --extra-arm "ppo_scheduler_nominal_h1=${NEW}" --extra-arm "ppo_scheduler_ranged_h1=${NEW}" \
    --output "${OUT}/f6_frozen_eval"

stage "F5 noise sweeps (coarse + fine), new arms and PID-RN"
done_in "${OUT}/f5_noise_sweep" episodes.csv || \
  python posthoc_eval.py --no-default \
    --manifest "${MAN}/noise_coarse.json" --manifest "${MAN}/noise_fine.json" \
    "${NEWARMS[@]}" --pid "${PIDRN}" \
    --output "${OUT}/f5_noise_sweep"

stage "done"
echo "copy back: scp -r <account>@<login>:thesis-experiment/${OUT} artifacts/results/"
echo "           and ${NEW}/*/seed_*/{metadata.json,validation_history.jsonl,train.log}"
