#!/bin/bash
# Post-hoc experiments, stage 2: E1 fine sweep, E4 noise-trained arms at each
# sigma*, E5 history ablation. Pin it to the stage-1 node:
#
#   sbatch --nodelist=cpunode39.dyn.scc.plus.ac.at slurm_posthoc2.sh
#   sbatch --nodelist=cpunode39.dyn.scc.plus.ac.at --export=ALL,SIGMAS="0.003 0.01" slurm_posthoc2.sh
#
# SIGMAS lists sigma* values in metres (E4 runs once per value), or "none".
# Decided from the stage-1 coarse sweep (DECISIONS.md, 2026-11-13): 3 mm by
# the plan's literal rule, 10 mm where GS-D and PID-R actually collapse.
#
# Steps:
#   0. parity gate
#   1. all training in the background: E5 H=1 (10 runs) + E4 per sigma* (10 each)
#   2. beside it: PID-RN retune per sigma*, then the E1 fine sweep
#   3. held-out 480: the 22 sealed instances re-run here + every new arm
#   4. new E4 arms on the coarse and fine noise sweeps
#
# Roughly 3-3.5 hours on 64 cores.

#SBATCH --job-name=thesis-posthoc-s2
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
OUT="${OUT:-artifacts/results/posthoc}"
MODELS="${MODELS:-artifacts/models/posthoc}"
SIGMAS="${SIGMAS:-0.003 0.01}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}, ${SLURM_CPUS_PER_TASK:-?} cores"
echo "commit    : $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"
echo "sigma*    : ${SIGMAS}"
stage() { echo; echo "=== $1 ($(date -u +%H:%M:%S) UTC) ==="; }
label() { python -c "from posthoc_manifests import level_label; print(level_label('noise', float('$1')))"; }

mkdir -p "${OUT}"
test -f manifests/posthoc/noise_fine.json || { echo "missing manifests/posthoc/noise_fine.json"; exit 1; }
if [ "${SIGMAS}" = "none" ]; then SIGMAS=""; fi

stage "parity gate"
python check_parity.py

stage "training in the background"
PIDS=()
python run_seeds.py --phase final --only-regime disturbed \
  --arm-template "ppo_{kind}_disturbed_h1" \
  --train-arg=--history-length=1 \
  --output "${MODELS}/e5_history" > "${OUT}/e5_training.log" 2>&1 &
PIDS+=($!)
echo "  E5 history H=1 -> ${MODELS}/e5_history"
for SIGMA in ${SIGMAS}; do
  L=$(label "${SIGMA}")
  python run_seeds.py --phase final --only-regime disturbed \
    --arm-template "ppo_{kind}_noisetrained_${L}" \
    --train-arg=--train-noise-max-m="${SIGMA}" \
    --train-arg=--checkpoint-noise-m="${SIGMA}" \
    --output "${MODELS}/e4_noise_${L}" > "${OUT}/e4_training_${L}.log" 2>&1 &
  PIDS+=($!)
  echo "  E4 sigma*=${L} -> ${MODELS}/e4_noise_${L}"
done

for SIGMA in ${SIGMAS}; do
  L=$(label "${SIGMA}")
  stage "E4 PID-RN retune at sigma*=${L} (beside the training)"
  python posthoc_pid_retune.py --sigma-m "${SIGMA}" --workers 32 \
    --output "${OUT}/e4_pid_retune_${L}"
done

stage "E1 fine sweep (beside the training)"
python posthoc_eval.py --manifest manifests/posthoc/noise_fine.json \
  --workers 32 --output "${OUT}/e1_noise_fine"

stage "waiting for training"
for PID in "${PIDS[@]}"; do wait "${PID}"; done
for LOG in "${OUT}"/e*_training*.log; do echo "--- ${LOG}"; tail -n 3 "${LOG}"; done

EXTRA=(--extra-arm "ppo_scheduler_disturbed_h1=${MODELS}/e5_history"
       --extra-arm "ppo_direct_disturbed_h1=${MODELS}/e5_history")
E4=()
for SIGMA in ${SIGMAS}; do
  L=$(label "${SIGMA}")
  PIDRN=$(python -c "import json;s=json.load(open('${OUT}/e4_pid_retune_${L}/pid_rn.json'))['selected'];print(f\"pid_global_robust_noise_${L}={s['kp']},{s['ki']},{s['kd']},{s['tau_d']}\")")
  E4+=(--extra-arm "ppo_scheduler_noisetrained_${L}=${MODELS}/e4_noise_${L}"
       --extra-arm "ppo_direct_noisetrained_${L}=${MODELS}/e4_noise_${L}"
       --pid "${PIDRN}")
done

stage "held-out 480: sealed instances on this node + every new arm"
python posthoc_eval.py --manifest manifests/held_out.json "${EXTRA[@]}" ${E4[@]+"${E4[@]}"} \
  --output "${OUT}/heldout_rerun"
python posthoc_check_sealed.py "${OUT}/heldout_rerun"

if [ ${#E4[@]} -gt 0 ]; then
  stage "E4 new arms on the coarse and fine noise sweeps"
  python posthoc_eval.py --no-default \
    --manifest manifests/posthoc/noise_coarse.json \
    --manifest manifests/posthoc/noise_fine.json \
    "${E4[@]}" \
    --output "${OUT}/e4_noise_sweep"
fi

stage "done"
echo "copy back: scp -r <account>@<login>:thesis-experiment/${OUT}/* artifacts/results/posthoc/"
