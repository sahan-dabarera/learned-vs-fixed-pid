# Learned vs. fixed PID for path following under delay and noise

Code, trained controllers and result data for the MSc thesis

> **Fixed PID, PPO Gain Scheduling, and Direct Reinforcement Learning for Path Following under
> Delay and Noise**
> Sahan Dabarera, Paris Lodron University Salzburg, 2026.

A simulated differential-drive robot (MuJoCo, 500 Hz) follows a path while its steering
command is delayed and its position measurement is noisy. Three ways of steering are compared on
the same plant, the same paths and the same disturbances:

| Family | Controllers | What is learned |
|---|---|---|
| Fixed PID | PID-N, PID-R, PID-RN | nothing: three fixed gain sets |
| PPO gain scheduler | GS-N, GS-D | a policy that sets the PID gains at 50 Hz |
| Direct PPO | DR-N, DR-D | a policy that outputs the steering command itself |

The repository holds everything behind the numbers in the thesis: the simulator and controllers,
the training and evaluation code, the scenario lists, the trained checkpoints, one result row per
evaluated run, and the scripts that turn those rows into the tables and figures.

## Contents

```
pathfollow/            Python package
  core/                robot model (MuJoCo XML), dynamics, path generators, PID
  env.py               Gymnasium environment shared by all controllers
  config.py            constants, path splits, locations of the data folders
  scenarios.py         scenario and scenario-list ("manifest") format
  metrics.py           failure-adjusted error and other per-run metrics
  tuning/              PID calibration (PID-N, PID-R), PID-RN grid search, probes
  training/            PPO training of one run (train.py) and of many seeds (run_seeds.py)
  evaluation/          test-set builders, parallel evaluator, parity and replay checks,
                       gain records, frozen gains, per-step traces
  analysis/            bootstrap statistics and the six result reports
thesis_figures/        scripts for the figures in the thesis, and their output
cluster/               the SLURM job scripts, one per stage of the study
data/
  scenarios/           scenario lists: training paths, validation set, all test sets
  calibration/         gains of the fixed PIDs and the scheduler gain ranges
  checkpoints/         trained policies: <arm>/seed_<n>/best_model.zip + training record
  results/             one row per evaluated run, per stage; reports/ holds the tables
  logs/                job logs
  parity/              reference values of the parity gate
tests/                 unit tests
```

`data/README.md` describes the data files in detail.

## Setup

Python 3.12. The versions in `requirements.txt` are the ones every job recorded.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tools/show_versions.py        # compares installed versions with the pinned ones
python -m pytest -q
```

All commands below are run from the repository root.

## Controllers and their names in the data

Checkpoint folders and result tables use internal arm names. The thesis names map to them as
follows.

| Thesis name | Arm name in `data/` | Training |
|---|---|---|
| PID-N | `pid_global_nominal` | gains 250 / 4.545 / 0.397, tuned on clean runs |
| PID-R | `pid_global_robust` | gains 8 / 0.5 / 4.4, tuned at one fixed severity |
| PID-RN | `pid_global_robust_noise` | gains 2.0 / 0.5 / 0.5, τd 0.01 s, tuned over the training ranges |
| GS-N | `ppo_scheduler_nominal_h1_wide` | clean training, one frame, wide gain range, 300k steps |
| GS-D | `ppo_scheduler_ranged_h1_wide_2m` | range training, one frame, wide gain range, 2M steps |
| DR-N | `ppo_direct_nominal_h1` | clean training, one frame, 2M steps |
| DR-D | `ppo_direct_ranged_h1` | range training, one frame, 2M steps |
| GS-D (300k) | `ppo_scheduler_ranged_h1_wide` | as GS-D with 300k steps |
| GS-D H10 | `ppo_scheduler_ranged_h10_wide` | as GS-D (300k) with a ten-frame history |
| narrow-range GS-D (300k) | `ppo_scheduler_ranged_h1` | as GS-D (300k) with the narrow gain range |
| narrow-range GS-D (2M) | `ppo_scheduler_ranged_h1_2m` | as GS-D with the narrow gain range (seeds 11–71) |
| narrow-range GS-N | `ppo_scheduler_nominal_h1` | as GS-N with the narrow gain range |
| narrow-range GS-D H10 | `ppo_scheduler_ranged_h10` | narrow gain range, ten frames, range training |
| DR-D H10 | `ppo_direct_ranged_h10` | as DR-D with a ten-frame history |
| point-trained GS-D / DR-D | `ppo_scheduler_disturbed_h1` / `ppo_direct_disturbed_h1` | one frame, trained at one severity |
| point-trained, ten frames | `ppo_scheduler_disturbed` / `ppo_direct_disturbed` | ten frames, trained at one severity |
| clean-trained, ten frames | `ppo_scheduler_nominal` / `ppo_direct_nominal` | ten frames, clean training |

`h1` / `h10` is the history length in frames, `wide` the wide gain range
(`data/calibration/blind_ppo_wide.json`; without it the narrow range `blind_ppo.json`), `2m` the
2M-step budget, `ranged` training over the delay and noise ranges, `disturbed` training at one
severity, `nominal` clean training. The reports of stages 2 and 3 use the labels of that stage
(for example `GS-D-W 2M` for the thesis GS-D); the reports of stages 4 to 6 use the thesis names.

Scenario lists: `held_out_ranged.json` is the main test set, `held_out.json` the fixed-severity
test set, `delay_wide.json` the constant-delay sweep, `dynamic.json` the dynamic-delay test,
`noise_coarse.json` and `noise_fine.json` the noise sweep, `validation_ranged.json` and
`validation.json` the validation sets, `train.json` the training paths.

## Reproducing the results

There are three levels. Each one starts from stored material, so a level can be run without the
ones below it.

### 1. Tables and figures from the stored result rows (minutes)

```bash
python -m pathfollow.analysis.report_stage4      # main results, ten seeds (Ch. 6 and 7)
python -m pathfollow.analysis.report_stage5      # gain records, frozen gains, traces (Ch. 8)
python -m pathfollow.analysis.report_stage6      # ablations and effort tables (Ch. 6 and 8)
python -m pathfollow.analysis.report_stage3      # scheduler selection (Sec. 5.5, Ch. 8)
python -m pathfollow.analysis.report_stage2      # PID-RN calibration, point-trained controllers
python -m pathfollow.analysis.report_stage1      # PID noise sweeps, derivative-filter check
```

Each command rewrites `data/results/reports/stage<N>.md`. The committed files are the output of
these commands, so `git diff data/results/reports` shows whether your run reproduces them.
Each report takes between one and ten minutes on a laptop.

The figures are drawn from those reports and from the stored traces:

```bash
python thesis_figures/ch6_main_comparison.py     # main_heatmap, main_paired_forest
python thesis_figures/ch7_stress_tests.py        # delay_sweep, dynamic_delay, noise_sweep_*
python thesis_figures/ch8_diagnosis.py           # noise_mechanism, frozen_gains, history_forest
python thesis_figures/trace_figures.py           # test_paths, trajectory_overlay, gsd_gains_over_time
python -m pathfollow.analysis.figures_scheduler_selection   # ch8_kp_histogram_gain_range, ch8_learning_curves_candidates
```

The PDFs are written to `thesis_figures/output/`.

### 2. Re-evaluating the stored checkpoints (hours)

Every job starts with the parity gate. It runs PID-N and PID-R on 24 validation scenarios and
compares the result with `data/parity/reference.json`:

```bash
python -m pathfollow.evaluation.check_parity
```

The evaluator runs any set of controllers on any scenario list, in parallel, and writes
`episodes.csv`, `summary.csv`, `summary_by_seed.csv` and `metadata.json` to a new folder. For
example, the seven main controllers on the main test set, seeds 11 to 71:

```bash
python -m pathfollow.evaluation.evaluate \
  --manifest data/scenarios/held_out_ranged.json \
  --no-default \
  --pid pid_global_nominal=250,4.54543591887462,0.39672846394679073 \
  --pid pid_global_robust=8,0.5,4.4 \
  --pid pid_global_robust_noise=2.0,0.5,0.5,0.01 \
  --extra-arm ppo_scheduler_nominal_h1_wide=data/checkpoints \
  --extra-arm ppo_scheduler_ranged_h1_wide_2m=data/checkpoints \
  --extra-arm ppo_direct_nominal_h1=data/checkpoints \
  --extra-arm ppo_direct_ranged_h1=data/checkpoints \
  --seeds 11,23,37,53,71 \
  --output runs/results/main_test
```

Add `--limit 12 --workers 4` for a quick trial. A second folder can be compared with stored rows
value by value:

```bash
python -m pathfollow.evaluation.check_replay \
  runs/results/main_test/episodes.csv data/results/stage3/g4_heldout_ranged_g2/episodes.csv \
  --output runs/results/main_test/replay_check.json
```

The check compares every row present in both files and passes only if all of them are
bit-identical. That holds on the machine type of the study. On other hardware the rows differ in
the last digits or more (a 24-scenario trial of GS-D on a laptop gave the same completion in all
24 runs and no bit-identical row), so there the check reports a failure by design.

The exact evaluation commands of every stage are in `cluster/`. The scripts skip a step when its
output folder exists; set `OUT=runs/results/stage<N>` to run a stage again into a new folder.

### 3. Retraining (hours per stage on a 64-core node)

One run:

```bash
python -m pathfollow.training.train --mode scheduled --regime disturbed --seed 11 \
  --arm ppo_scheduler_ranged_h1_wide_2m --history-length 1 \
  --train-delay-range-s 0,0.3 --train-noise-min-m 0.0001 --train-noise-max-m 0.01 \
  --calibration data/calibration/blind_ppo_wide.json --gain-rate-limit 147.5,0.75,1.75 \
  --validation-manifest data/scenarios/validation_ranged.json \
  --timesteps 2000000 --output runs/checkpoints
```

The settings of each arm, and the batches of seeds, are in the `cluster/` scripts
(`python -m pathfollow.training.run_seeds`); each script states its run time. Training refuses to
write into `data/checkpoints`, and new runs go to `runs/` by default.

## Stages of the study

The study was run in stages. Each stage is one or more cluster jobs, one folder under
`data/results/` and one report. Appendix D of the thesis lists the same jobs.

| Stage | Jobs | Script in `cluster/` | What it adds | Used in the thesis for |
|---|---|---|---|---|
| sealed | 1760422 | `stage0_sealed_run.sh` | ten-frame controllers (clean and point-trained), fixed-severity test set | history and training-range ablations |
| 1 | 3187230, 3188678, 3189024 | `stage1a_*`, `stage1b_*`, `stage1c_*` | noise and delay sweeps, derivative-filter check, point-trained one-frame controllers | noise sweep of PID-N and PID-R, filter check on PID-R |
| 2 | 3225586, 3281365, 3281375 | `stage2_*`, `stage2b_*` | one-frame design, range training, main test set, dynamic-delay test, PID-RN | PID-RN, DR-N, DR-D, test-set construction |
| 3 | 3281395, 3347265 | `stage3_*`, `stage3b_*` | wide gain range, 2M budget, scheduler selection on validation data | GS-N, GS-D, scheduler selection |
| 4 | 3358307 | `stage4_*` | seeds 83 to 149 of the four main learned controllers | all main results (ten seeds) |
| 5 | 3366201 | `stage5_*` | gain records, frozen gains, cross-track error, per-step traces | diagnosis chapter, trace figures |
| 6 | 3371665, 3374181 | `stage6_*` | seeds 83 to 149 of the ablation controllers | ablations, effort and event tables |

Terms that appear in code, data and reports:

- **Sealed run.** The first complete five-seed run (job 1760422). Its results and checkpoints
  were frozen; the evaluator refuses to write into `data/results/sealed` or `data/checkpoints`,
  and later jobs replay sealed scenarios and compare the rows (`sealed_identical`,
  `check_reference`).
- **Run labels.** Result folders keep the labels of their stage: `e1`–`e5` (stage 1), `f0`–`f6`
  (stage 2), `g0`–`g5` (stage 3), `t1`–`t3` (stage 6). `data/README.md` lists them.
- **Manifest.** A scenario list: paths, target speeds, delay and noise values, event times and
  noise seeds, fixed before any controller is run.
- **Box.** The gain range of a scheduler. **Ranged / point sampler.** Range training / training
  at one severity.
- **`posthoc`.** A key in scenario files and checkpoint records that holds the settings added
  after the sealed run (history length, training ranges, rate limit).

## What reproduces exactly

- The six reports are regenerated from the stored rows with identical numbers on any machine.
- Evaluation results are bit-identical between jobs and between nodes of the cluster used (AMD
  EPYC 9334, the versions in `requirements.txt`). On other hardware the last digits can differ,
  and with them single runs close to a failure. The parity gate checks that completion is
  identical and the error within 1 %.
- Training is deterministic on that machine type with one PyTorch thread per run. Elsewhere,
  retraining gives different weights; the stored checkpoints are the controllers of the thesis.
- Paths recorded inside `metadata.json` files and logs are those of the original working tree
  (`artifacts/...`, `manifests/...`); `data/README.md` gives the mapping.

The original development history, including the revisions named in Appendix D of the thesis, is
kept on the branch `development-history`.

## Citation

```bibtex
@mastersthesis{dabarera2026pid,
  author = {Dabarera, Sahan},
  title  = {Fixed {PID}, {PPO} Gain Scheduling, and Direct Reinforcement Learning for Path
            Following under Delay and Noise},
  school = {Paris Lodron University Salzburg},
  year   = {2026}
}
```

## License

MIT, see `LICENSE`.
