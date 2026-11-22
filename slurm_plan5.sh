#!/bin/bash
# Plan 5: the ablation arms for the new seeds 83, 97, 109, 127, 149, so that every
# arm in the thesis has ten seeds (seeds 11-71 come from plans 2/3 and the sealed
# runs; they are not retrained). Every arm uses the exact arguments of its seed
# 11-71 run; only the seed differs.
#
#   Tier 1  GS-D-W 300k (plan-3 ranged_h1_wide): retrain (the 2M runs kept no <=300k
#           checkpoint), held-out ranged + fixed, gain occupancy, frozen gains.
#           Effort, event metrics and decision time of the main arms need no run:
#           they are columns of the plan-4 evaluation rows (posthoc4).
#   Tier 2  schedulers, 300k: narrow-range GS-D (plan-2 ranged_h1), narrow-range GS-N
#           (plan-2 nominal_h1, the one-frame twin of the clean H10 scheduler), GS-D-W H10
#           (plan-3 ranged_h10_wide), narrow H10 clean / range / point (sealed nominal,
#           plan-2 ranged_h10, sealed disturbed), point-trained one frame (posthoc2 E5
#           disturbed_h1). Held-out ranged + fixed; occupancy and frozen gains where the
#           seed 11-71 runs have them (ranged_h1, nominal_h1: G0 occupancy + F6 frozen;
#           sealed nominal/disturbed: E3 frozen on the fixed held-out set); delay sweep and
#           dynamic delay (F3/F4) for the training-range pair ranged_h1 / disturbed_h1.
#   Tier 3  direct PPO, 2M: DR-N H10 (sealed nominal), DR-D H10 (plan-2 ranged_h10),
#           DR-D H10 point (sealed disturbed), point-trained one frame (E5 disturbed_h1).
#           Held-out ranged + fixed; delay sweep and dynamic delay for disturbed_h1
#           (point-trained DR-D; its range-trained twin DR-D has them in posthoc4).
#
# Laptop (default here):  VENV=../venv TIERS="1 2 3" bash slurm_plan5.sh > artifacts/results/plan5.log 2>&1
# Cluster:                sbatch --nodelist=cpunode13.dyn.scc.plus.ac.at slurm_plan5.sh
# Each tier runs on its own (TIERS="2"); finished stages are skipped.

#SBATCH --job-name=thesis-plan5
#SBATCH --partition=base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --constraint=compute_nodes_cpu
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail

VENV="${VENV:-.venv}"
TIERS="${TIERS:-1 2 3}"
SEEDS=83,97,109,127,149
LAST=149
MAN=manifests/posthoc
NEW=artifacts/models/posthoc/plan5
OUT="${OUT:-artifacts/results/posthoc6}"
WIDE=artifacts/calibration/blind_ppo_wide.json
RATE=147.5,0.75,1.75

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
# shellcheck disable=SC1091
if [ -f "${VENV}/bin/activate" ]; then source "${VENV}/bin/activate"; else source "${VENV}/Scripts/activate"; fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "host      : $(hostname)"
echo "cpu       : $( (lscpu 2>/dev/null || true) | grep 'Model name' | sed 's/.*: *//')"
echo "job       : ${SLURM_JOB_ID:-none}, ${SLURM_CPUS_PER_TASK:-?} cores"
echo "commit    : $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"
echo "seeds     : ${SEEDS}; tiers: ${TIERS}"
stage() { echo; echo "=== $1 ($(date -u +%H:%M:%S) UTC) ==="; }
done_in() { test -f "$1/$2"; }
mkdir -p "${OUT}"

SAMPLER=(--train-arg=--train-delay-range-s=0,0.3
         --train-arg=--train-noise-min-m=0.0001
         --train-arg=--train-noise-max-m=0.01
         --validation-manifest "${MAN}/validation_ranged.json")
WIDEBOX=(--scheduler-calibration "${WIDE}" --train-arg=--gain-rate-limit=${RATE})
H1=(--train-arg=--history-length=1)

PIDS=()
train() {  # arm, mode, args...  (arm name = ppo_<kind>_<suffix>, as in the seed 11-71 run)
  local arm=$1 mode=$2; shift 2
  local kind=${arm#ppo_}; kind=${kind%%_*}
  if [ -f "${NEW}/${arm}/seed_${LAST}/metadata.json" ]; then
    echo "  ${arm}: already trained"; return
  fi
  python run_seeds.py --phase final --seeds "${SEEDS}" --only-mode "${mode}" --max-parallel 5 \
    --arm-template "ppo_{kind}_${arm#ppo_${kind}_}" --output "${NEW}" "$@" \
    > "${OUT}/training_${arm}.log" 2>&1 &
  PIDS+=($!)
  echo "  ${arm} -> ${NEW} (log ${OUT}/training_${arm}.log)"
}
wait_training() {
  for PID in ${PIDS[@]+"${PIDS[@]}"}; do wait "${PID}"; done
  PIDS=()
  for A in "$@"; do
    tail -n 2 "${OUT}/training_${A}.log" 2>/dev/null || true
    test -f "${NEW}/${A}/seed_${LAST}/best_model.zip" || { echo "missing ${NEW}/${A}/seed_${LAST}"; exit 1; }
  done
}
evaluate() {  # tag, arms...: ranged and fixed held-out sets + validation scores
  local tag=$1; shift
  local args=(--no-default --seeds "${SEEDS}") vargs=()
  for A in "$@"; do args+=(--extra-arm "${A}=${NEW}"); vargs+=(--arm "${A}=${NEW}"); done
  done_in "${OUT}/${tag}_validation" validation_selected.csv || \
    python posthoc_validation_check.py --seeds "${SEEDS}" --output "${OUT}/${tag}_validation" "${vargs[@]}"
  done_in "${OUT}/${tag}_heldout_ranged" episodes.csv || \
    python posthoc_eval.py "${args[@]}" --manifest "${MAN}/held_out_ranged.json" --output "${OUT}/${tag}_heldout_ranged"
  done_in "${OUT}/${tag}_heldout_fixed" episodes.csv || \
    python posthoc_eval.py "${args[@]}" --manifest manifests/held_out.json --output "${OUT}/${tag}_heldout_fixed"
}
stress() {  # tag, arms...: constant-delay sweep and dynamic delay (F3 / F4 manifests)
  local tag=$1; shift
  local args=(--no-default --seeds "${SEEDS}")
  for A in "$@"; do args+=(--extra-arm "${A}=${NEW}"); done
  done_in "${OUT}/${tag}_delay" episodes.csv || \
    python posthoc_eval.py "${args[@]}" --manifest "${MAN}/delay_wide.json" --output "${OUT}/${tag}_delay"
  done_in "${OUT}/${tag}_dynamic" episodes.csv || \
    python posthoc_eval.py "${args[@]}" --manifest "${MAN}/dynamic.json" --output "${OUT}/${tag}_dynamic"
}

stage "parity gate"
python check_parity.py || echo "parity gate: differences (expected off the EPYC nodes; recorded, not fatal)"

for TIER in ${TIERS}; do
case "${TIER}" in
1)
  stage "Tier 1: GS-D-W 300k (plan-3 ranged_h1_wide), training"
  train ppo_scheduler_ranged_h1_wide scheduled --only-regime disturbed "${SAMPLER[@]}" "${H1[@]}" "${WIDEBOX[@]}"
  wait_training ppo_scheduler_ranged_h1_wide
  stage "Tier 1: evaluation"
  evaluate t1 ppo_scheduler_ranged_h1_wide
  stage "Tier 1: gain occupancy (G4 construction) and frozen gains (G4 construction)"
  done_in "${OUT}/t1_occupancy" occupancy.csv || \
    python posthoc_gain_occupancy.py --models-root "${NEW}" --seeds "${SEEDS}" \
      --arm "ppo_scheduler_ranged_h1_wide=${MAN}/validation_ranged.json" \
      --arm "ppo_scheduler_ranged_h1_wide=${MAN}/held_out_ranged.json" --output "${OUT}/t1_occupancy"
  done_in "${OUT}/t1_frozen_gains" frozen_gains.json || \
    python posthoc_frozen_gains.py --models-root "${NEW}" --seeds "${SEEDS}" --arm ppo_scheduler_ranged_h1_wide \
      --output "${OUT}/t1_frozen_gains"
  done_in "${OUT}/t1_frozen_eval" episodes.csv || \
    python posthoc_eval.py --no-default --seeds "${SEEDS}" --manifest "${MAN}/held_out_ranged.json" \
      --frozen-gains "${OUT}/t1_frozen_gains/frozen_gains.json" \
      --extra-arm "ppo_scheduler_ranged_h1_wide=${NEW}" --output "${OUT}/t1_frozen_eval"
  ;;
2)
  T2=(ppo_scheduler_ranged_h1 ppo_scheduler_nominal_h1 ppo_scheduler_ranged_h10_wide ppo_scheduler_nominal ppo_scheduler_ranged_h10
      ppo_scheduler_disturbed ppo_scheduler_disturbed_h1)
  stage "Tier 2: scheduler training (300k)"
  train ppo_scheduler_ranged_h1        scheduled --only-regime disturbed "${SAMPLER[@]}" "${H1[@]}"   # plan 2
  train ppo_scheduler_nominal_h1       scheduled --only-regime nominal "${H1[@]}"   # plan 2
  train ppo_scheduler_ranged_h10_wide  scheduled --only-regime disturbed "${SAMPLER[@]}" "${WIDEBOX[@]}"  # plan 3
  train ppo_scheduler_nominal          scheduled --only-regime nominal     # sealed defaults
  train ppo_scheduler_ranged_h10       scheduled --only-regime disturbed "${SAMPLER[@]}"   # plan 2
  train ppo_scheduler_disturbed        scheduled --only-regime disturbed   # sealed defaults
  train ppo_scheduler_disturbed_h1     scheduled --only-regime disturbed "${H1[@]}"   # posthoc2 E5
  wait_training "${T2[@]}"
  stage "Tier 2: evaluation"
  evaluate t2 "${T2[@]}"
  stage "Tier 2: delay sweep and dynamic delay, narrow GS-D range / point"
  stress t2 ppo_scheduler_ranged_h1 ppo_scheduler_disturbed_h1
  stage "Tier 2: narrow GS-D / GS-N occupancy (G0 construction) and frozen gains (F6, E3 constructions)"
  done_in "${OUT}/t2_occupancy" occupancy.csv || \
    python posthoc_gain_occupancy.py --models-root "${NEW}" --seeds "${SEEDS}" \
      --arm "ppo_scheduler_ranged_h1=${MAN}/validation_ranged.json" \
      --arm "ppo_scheduler_ranged_h1=${MAN}/held_out_ranged.json" \
      --arm "ppo_scheduler_nominal_h1=manifests/validation.json" \
      --arm "ppo_scheduler_nominal_h1=${MAN}/held_out_ranged.json" --output "${OUT}/t2_occupancy"
  done_in "${OUT}/t2_frozen_gains" frozen_gains.json || \
    python posthoc_frozen_gains.py --models-root "${NEW}" --seeds "${SEEDS}" \
      --arm ppo_scheduler_ranged_h1 --arm ppo_scheduler_nominal_h1 \
      --arm ppo_scheduler_nominal --arm ppo_scheduler_disturbed --output "${OUT}/t2_frozen_gains"
  done_in "${OUT}/t2_frozen_eval_ranged" episodes.csv || \
    python posthoc_eval.py --no-default --seeds "${SEEDS}" --manifest "${MAN}/held_out_ranged.json" \
      --frozen-gains "${OUT}/t2_frozen_gains/frozen_gains.json" \
      --extra-arm "ppo_scheduler_ranged_h1=${NEW}" --extra-arm "ppo_scheduler_nominal_h1=${NEW}" \
      --output "${OUT}/t2_frozen_eval_ranged"
  done_in "${OUT}/t2_frozen_eval_fixed" episodes.csv || \
    python posthoc_eval.py --no-default --seeds "${SEEDS}" --manifest manifests/held_out.json \
      --frozen-gains "${OUT}/t2_frozen_gains/frozen_gains.json" \
      --extra-arm "ppo_scheduler_nominal=${NEW}" --extra-arm "ppo_scheduler_disturbed=${NEW}" \
      --output "${OUT}/t2_frozen_eval_fixed"
  ;;
3)
  T3=(ppo_direct_nominal ppo_direct_ranged_h10 ppo_direct_disturbed ppo_direct_disturbed_h1)
  stage "Tier 3: direct PPO training (2M)"
  train ppo_direct_nominal      direct --only-regime nominal     # sealed defaults
  train ppo_direct_ranged_h10   direct --only-regime disturbed "${SAMPLER[@]}"   # plan 2
  train ppo_direct_disturbed    direct --only-regime disturbed   # sealed defaults
  train ppo_direct_disturbed_h1 direct --only-regime disturbed "${H1[@]}"   # posthoc2 E5
  wait_training "${T3[@]}"
  stage "Tier 3: evaluation"
  evaluate t3 "${T3[@]}"
  stage "Tier 3: delay sweep and dynamic delay, point-trained DR-D (one frame)"
  stress t3 ppo_direct_disturbed_h1
  ;;
*) echo "unknown tier ${TIER}"; exit 1 ;;
esac
done

stage "done"
echo "pack: tar czf plan5_results.tgz --exclude=tensorboard ${OUT} ${NEW}"
