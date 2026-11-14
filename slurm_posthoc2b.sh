#!/bin/bash
# Post-hoc stage 2, resumed after training. Job 3188678 trained all arms,
# ran both PID-RN retunes and the E1 fine sweep, then stopped on a `tail -3`
# call this cluster's coreutils rejects for several files. This job runs the
# remaining evaluation steps unchanged; it trains nothing.
#
#   sbatch --nodelist=cpunode39.dyn.scc.plus.ac.at slurm_posthoc2b.sh
#
#   3. held-out 480: the 22 sealed instances re-run here + every new arm
#   4. new E4 arms on the coarse and fine noise sweeps
#
# Roughly 1.5 hours on 64 cores.

#SBATCH --job-name=thesis-posthoc-s2b
#SBATCH --partition=base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --constraint=compute_nodes_cpu
#SBATCH --time=06:00:00
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

stage "inputs from job 3188678"
for LOG in "${OUT}"/e*_training*.log; do
  echo "--- ${LOG}"; tail -n 2 "${LOG}"
  grep -q "10/10 runs completed" "${LOG}" || { echo "training incomplete in ${LOG}"; exit 1; }
done
for SIGMA in ${SIGMAS}; do
  test -f "${OUT}/e4_pid_retune_$(label "${SIGMA}")/pid_rn.json" || { echo "missing PID-RN for ${SIGMA}"; exit 1; }
done

stage "parity gate"
python check_parity.py

EXTRA=(--extra-arm "ppo_scheduler_disturbed_h1=${MODELS}/e5_history"
       --extra-arm "ppo_direct_disturbed_h1=${MODELS}/e5_history")
E4=()
for SIGMA in ${SIGMAS}; do
  L=$(label "${SIGMA}")
  PIDRN=$(python -c "import json;s=json.load(open('${OUT}/e4_pid_retune_${L}/pid_rn.json'))['selected'];print(f\"pid_global_robust_noise_${L}={s['kp']},{s['ki']},{s['kd']},{s['tau_d']}\")")
  echo "  ${PIDRN}"
  E4+=(--extra-arm "ppo_scheduler_noisetrained_${L}=${MODELS}/e4_noise_${L}"
       --extra-arm "ppo_direct_noisetrained_${L}=${MODELS}/e4_noise_${L}"
       --pid "${PIDRN}")
done

stage "held-out 480: sealed instances on this node + every new arm"
python posthoc_eval.py --manifest manifests/held_out.json "${EXTRA[@]}" "${E4[@]}" \
  --output "${OUT}/heldout_rerun"
python posthoc_check_sealed.py "${OUT}/heldout_rerun"

stage "E4 new arms on the coarse and fine noise sweeps"
python posthoc_eval.py --no-default \
  --manifest manifests/posthoc/noise_coarse.json \
  --manifest manifests/posthoc/noise_fine.json \
  "${E4[@]}" \
  --output "${OUT}/e4_noise_sweep"

stage "done"
