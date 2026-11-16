#!/bin/bash
# Plan 2 follow-up: the PID-RN retune in job 3225586 selected Kp = 4.0, the
# lowest Kp in its grid. This job repeats the retune with the grid extended
# down to Kp = 1 (the old grid is a subset, so the old optimum stays a
# candidate) and re-runs only the selected PID on every plan-2 manifest. The
# PID has no network, so its rows are directly paired with the learned arms
# from job 3225586 on any EPYC 9334 node.
#
#   sbatch --nodelist=cpunode13.dyn.scc.plus.ac.at slurm_plan2_pidrn.sh
#
# Roughly 30-45 minutes on 64 cores.

#SBATCH --job-name=thesis-plan2-pidrn
#SBATCH --partition=base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --constraint=compute_nodes_cpu
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail

VENV="${VENV:-.venv}"
OUT="${OUT:-artifacts/results/posthoc2}"
MAN=manifests/posthoc
KP_GRID="1,1.5,2,2.5,3,4,5,6,8,10,13,18,26,40"

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

stage "parity gate"
python check_parity.py

stage "PID-RN retune, Kp grid ${KP_GRID}"
done_in "${OUT}/pid_retune_wide" pid_rn.json || \
  python posthoc_pid_retune.py --manifest "${MAN}/validation_ranged.json" --kp-grid "${KP_GRID}" \
    --output "${OUT}/pid_retune_wide"
PIDRN=$(python -c "import json;s=json.load(open('${OUT}/pid_retune_wide/pid_rn.json'))['selected'];print(f\"pid_global_robust_noise={s['kp']},{s['ki']},{s['kd']},{s['tau_d']}\")")
echo "  ${PIDRN}"

run() {  # output dir, manifests...
  local dir=$1; shift
  local args=()
  for M in "$@"; do args+=(--manifest "${M}"); done
  done_in "${OUT}/${dir}" episodes.csv || \
    python posthoc_eval.py --no-default "${args[@]}" --pid "${PIDRN}" --output "${OUT}/${dir}"
}

stage "selected PID-RN on every plan-2 manifest"
run pidrn_f2_heldout_ranged "${MAN}/held_out_ranged.json"
run pidrn_f2_heldout_fixed  manifests/held_out.json
run pidrn_f3_delay          "${MAN}/delay_wide.json"
run pidrn_f4_dynamic        "${MAN}/dynamic.json"
run pidrn_f5_noise_sweep    "${MAN}/noise_coarse.json" "${MAN}/noise_fine.json"

stage "done"
echo "pack: tar czf plan2_pidrn.tgz ${OUT}/pid_retune_wide ${OUT}/pidrn_* slurm-thesis-plan2-pidrn-*.out"
