# Data

Everything the thesis results are computed from. Nothing in this folder is written by the code
with default settings; new runs go to `runs/` (ignored by Git).

## `scenarios/`

Scenario lists ("manifests"). A scenario fixes the path, the target speed, the delay and noise
values, the event time and the noise seed. Every controller is run on the same scenarios.

| File | Scenarios | Role in the thesis |
|---|---|---|
| `train.json` | 144 | scenarios on the six training path kinds (PID calibration) |
| `validation.json` | 96 | validation set at fixed severities (parity gate, frozen gains, clean-trained checkpoints) |
| `validation_ranged.json` | 96 | validation set with drawn severities (checkpoint selection of range-trained controllers, PID-RN) |
| `held_out_ranged.json` | 480 | **main test set**: 20 unseen paths × 3 speeds × 8 cells, severities drawn from the ranges |
| `held_out.json` | 480 | fixed-severity test set (0.15 s delay, 0.3 mm noise) |
| `delay_wide.json` | 1320 | constant-delay sweep |
| `dynamic.json` | 360 | dynamic-delay test |
| `noise_coarse.json`, `noise_fine.json` | 1440, 2880 | noise sweep |
| `delay.json` | 960 | narrower delay sweep of stage 1 |

`train.json`, `validation.json` and `held_out.json` are rebuilt by
`python -m pathfollow.evaluation.make_base_scenarios`, the others by
`python -m pathfollow.evaluation.make_test_scenarios` (commands in its docstring). Both write to
`runs/scenarios/` and reproduce the stored files exactly.

The sweep, ranged and dynamic files carry a top-level block named `posthoc` with bookkeeping
(sweep level, base scenario, `sealed_identical` flag). The scenario loader ignores it.

## `calibration/`

| File | Content |
|---|---|
| `calibration.json` | gains of PID-N (`nominal`) and PID-R (`robust`) |
| `blind_ppo.json` | narrow scheduler gain range (Kp 5–300, Ki 0–1.5, Kd 2.5–6) and start gains |
| `blind_ppo_wide.json` | wide scheduler gain range (Kp 1–300, Ki 0–1.5, Kd 0–6), used by GS-N and GS-D |
| `per_path_gains.json`, `*_validation.csv` | intermediate output of the PID calibration |

The gains of PID-RN (2.0 / 0.5 / 0.5, τd 0.01 s) are the `selected` entry of
`results/stage2/pid_retune_wide_fine/pid_rn.json`. `probes/` holds the sensitivity checks made
while the severities and the fixed PIDs were set.

## `checkpoints/`

`<arm>/seed_<n>/` for every trained run:

| File | Content |
|---|---|
| `best_model.zip` | the kept checkpoint (Stable-Baselines3 PPO); this is the controller that was evaluated |
| `metadata.json` | settings, seed, budget, wall time, software versions, hashes of the code and of the gain file |
| `validation_history.jsonl` | validation score of every checkpoint during training |
| `monitor.csv` | episode returns during training |

The README in the repository root maps arm names to thesis names. Seeds are 11, 23, 37, 53, 71,
83, 97, 109, 127 and 149. `ppo_scheduler_ranged_h1_2m` was trained with seeds 11–71 only.
For `ppo_scheduler_disturbed_h1` and `ppo_direct_disturbed_h1` the checkpoints of seeds 11–71
(job 3188678) are not included; their result rows are in `results/stage1` and `results/stage2`.
The final-step models, TensorBoard files and per-run training logs are not included.

## `results/`

One folder per stage (see the stage table in the root README) and `reports/` with the generated
tables. A result folder of the evaluator contains

| File | Content |
|---|---|
| `episodes.csv.gz` | one row per run: controller, seed, scenario, completion, failure-adjusted error and the other metrics |
| `summary.csv`, `summary_by_seed.csv` | means per controller and cell, pooled and per seed |
| `metadata.json` | command, Git revision, machine, software versions, hashes of scenario list, gain files and checkpoints |

`metrics.read_csv` reads `episodes.csv.gz` when asked for `episodes.csv`.

| Folder | Content |
|---|---|
| `sealed/` | job 1760422: ten-frame controllers on the fixed-severity test set |
| `stage1/e1_*`, `e1f_taud` | noise sweeps (coarse, fine) and the PID-R derivative-filter check |
| `stage1/e2_delay` | delay sweep up to 0.4 s |
| `stage1/e3_*` | frozen gains of the ten-frame schedulers |
| `stage1/e4_*`, `heldout_rerun` | noise-trained variants and point-trained one-frame controllers (the noise-trained variants are not used in the thesis) |
| `stage2/f0_validation` | validation scores of the kept checkpoints |
| `stage2/f2_heldout_*` | main and fixed-severity test sets |
| `stage2/f3_*`, `f4_*`, `f5_*` | delay sweep, dynamic delay, noise sweep |
| `stage2/f6_*` | frozen gains of the narrow-range one-frame schedulers |
| `stage2/pid_retune*`, `pidrn_*` | PID-RN grid rounds and PID-RN on every test set; the thesis uses the `_fine` folders |
| `stage3/g0_occupancy`, `g4_occupancy_wide`, `g5_occupancy_2m` | gain records (where in its range each scheduler operates) |
| `stage3/g3_validation` | validation scores behind the scheduler selection |
| `stage3/g4_*_g1`, `g4_*_g2` | all test sets for the 300k (`g1`) and 2M (`g2`) scheduler candidates |
| `stage3/g4_frozen_*`, `g5_frozen_*` | frozen gains of GS-D (300k), GS-N and GS-D |
| `stage3/replay_plan2` | replay of stage-2 rows after the evaluator was extended |
| `stage4/` | seeds 83–149 of GS-N, GS-D, DR-N, DR-D on all test sets; `path_geometry/` holds the path coordinates |
| `stage5/` | gain records and frozen gains for seeds 83–149, replays with cross-track error, per-step traces (`traces_*/traces.csv.gz`) |
| `stage6/t1_*`, `t2_*`, `t3_*` | seeds 83–149 of the ablation controllers: wide-range 300k schedulers, narrow-range schedulers, direct PPO |
| `reports/stage<N>.md` | the generated tables; source of every number and figure in the thesis |

`stage4/paired_cache.json` and `stage6/paired_cache.json` store bootstrap results so that the
reports of stages 4 and 6 build in minutes; delete them to recompute.

## `logs/`, `parity/`

`logs/` holds the job logs (the reports read wall times from some of them); the logs of the
sealed run and of stage 1 are inside their result folders. `plan5.log` is the log of the part of
stage 6 that ran on a workstation. `parity/reference.json` is the reference of the parity gate.

## Paths recorded inside the files

`metadata.json` files and logs keep the paths of the tree the jobs ran in. They map to this
repository as follows.

| Recorded | Here |
|---|---|
| `manifests/`, `manifests/posthoc/` | `data/scenarios/` |
| `artifacts/calibration/` | `data/calibration/` |
| `artifacts/models/final/`, `artifacts/models/posthoc/plan<N>/`, `.../e5_history/` | `data/checkpoints/` |
| `artifacts/results/final/` | `data/results/sealed/` |
| `artifacts/results/posthoc/`, `posthoc2/` … `posthoc6/` | `data/results/stage1/`, `stage2/` … `stage6/` |
| `posthoc_eval.py`, `posthoc_*.py`, `slurm_*.sh` | `pathfollow/evaluation/`, `pathfollow/analysis/`, `cluster/` |

The user name in recorded paths was replaced by `user`.
