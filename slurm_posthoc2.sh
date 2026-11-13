#!/bin/bash
# Post-hoc experiments, stage 2: E1 fine sweep, E4 noise-trained arms, E5
# history ablation. Pin it to the stage-1 node so both stages share a CPU:
#
#   sbatch --nodelist=cpunode39 --export=ALL,SIGMA_STAR=0.01 slurm_posthoc2.sh
#   sbatch --nodelist=cpunode39 --export=ALL,SIGMA_STAR=none slurm_posthoc2.sh   # E4 not triggered
#
# SIGMA_STAR (metres) comes from the stage-1 E1 result and the plan's trigger
# rule. The E1 fine manifest is built on the laptop from stage-1 results and
# committed as manifests/posthoc/noise_fine.json before this job is submitted.
#
# Steps:
#   0. parity gate
#   1. E4 + E5 training, all runs concurrently (10 or 20 single-thread runs)
#   2. E4 PID-RN retune on the robust validation subset at sigma*
#   3. E1 fine sweep, all 22 sealed instances
#   4. held-out 480: sealed instances re-run on this node + new arms
#   5. new arms on the coarse and fine noise sweeps
#
# Roughly 2.5 hours on 64 cores, dominated by the 2M-decision direct runs.

#SBATCH --job-name=thesis-posthoc-s2
#SBATCH --partition=base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --constraint=compute_nodes_cpu
#SBATCH --time=10:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail

VENV="${VENV:-.venv}"
OUT="${OUT:-artifacts/results/posthoc}"
MODELS="${MODELS:-artifacts/models/posthoc}"
SIGMA_STAR="${SIGMA_STAR:?set SIGMA_STAR to sigma* in metres, or to none}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}, ${SLURM_CPUS_PER_TASK:-?} cores"
echo "commit    : $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"
echo "sigma*    : ${SIGMA_STAR}"
stage() { echo; echo "=== $1 ($(date -u +%H:%M:%S) UTC) ==="; }

mkdir -p "${OUT}"
test -f manifests/posthoc/noise_fine.json || { echo "missing manifests/posthoc/noise_fine.json"; exit 1; }

stage "parity gate"
python check_parity.py

stage "training: E5 history H=1 (and E4 if triggered), concurrently"
python run_seeds.py --phase final --only-regime disturbed \
  --arm-template "ppo_{kind}_disturbed_h1" \
  --train-arg=--history-length=1 \
  --output "${MODELS}/e5_history" > "${OUT}/e5_training.log" 2>&1 &
E5_PID=$!
if [ "${SIGMA_STAR}" != "none" ]; then
  python run_seeds.py --phase final --only-regime disturbed \
    --arm-template "ppo_{kind}_noisetrained" \
    --train-arg=--train-noise-max-m="${SIGMA_STAR}" \
    --train-arg=--checkpoint-noise-m="${SIGMA_STAR}" \
    --output "${MODELS}/e4_noise" > "${OUT}/e4_training.log" 2>&1 &
  E4_PID=$!

  stage "E4 PID-RN retune (runs beside the training)"
  python posthoc_pid_retune.py --sigma-m "${SIGMA_STAR}" --workers 20 \
    --output "${OUT}/e4_pid_retune"
fi

stage "E1 fine sweep (beside the training)"
python posthoc_eval.py --manifest manifests/posthoc/noise_fine.json \
  --workers 20 --output "${OUT}/e1_noise_fine"

stage "waiting for training"
wait "${E5_PID}"; tail -25 "${OUT}/e5_training.log"
if [ "${SIGMA_STAR}" != "none" ]; then wait "${E4_PID}"; tail -25 "${OUT}/e4_training.log"; fi

EXTRA=(--extra-arm "ppo_scheduler_disturbed_h1=${MODELS}/e5_history"
       --extra-arm "ppo_direct_disturbed_h1=${MODELS}/e5_history")
if [ "${SIGMA_STAR}" != "none" ]; then
  PIDRN=$(python -c "import json;s=json.load(open('${OUT}/e4_pid_retune/pid_rn.json'))['selected'];print(f\"pid_global_robust_noise={s['kp']},{s['ki']},{s['kd']},{s['tau_d']}\")")
  EXTRA+=(--extra-arm "ppo_scheduler_noisetrained=${MODELS}/e4_noise"
          --extra-arm "ppo_direct_noisetrained=${MODELS}/e4_noise"
          --pid "${PIDRN}")
fi

stage "held-out 480: sealed instances on this node + new arms"
python posthoc_eval.py --manifest manifests/held_out.json "${EXTRA[@]}" \
  --output "${OUT}/heldout_rerun"
python posthoc_check_sealed.py "${OUT}/heldout_rerun"

if [ "${SIGMA_STAR}" != "none" ]; then
  stage "E4 new arms on the noise sweeps"
  python posthoc_eval.py --no-default \
    --manifest manifests/posthoc/noise_coarse.json \
    --manifest manifests/posthoc/noise_fine.json \
    --extra-arm "ppo_scheduler_noisetrained=${MODELS}/e4_noise" \
    --extra-arm "ppo_direct_noisetrained=${MODELS}/e4_noise" \
    --pid "${PIDRN}" \
    --output "${OUT}/e4_noise_sweep"
fi

stage "done"
echo "rsync -av SERVER:thesis-experiment/${OUT} ./artifacts/results/"
echo "model zips: rsync -av --include='*/' --include='*.zip' --include='*.json*' --exclude='*' SERVER:thesis-experiment/${MODELS} ./artifacts/models/"
