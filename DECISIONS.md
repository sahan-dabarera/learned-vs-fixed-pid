# Experiment Decision Log

This log records protocol changes after the initial plan. Parameter changes made
before the final freeze are allowed only with a reason and validation evidence.
After the freeze, correctness fixes invalidate every affected final run.

## 2026-08-19: Initial implementation

- Use one `PathFollowingEnv` for fixed PID, PPO gain scheduling, and direct PPO.
- Keep MuJoCo physics and PID updates at 500 Hz; both PPO policies act at 50 Hz.
- Restrict disturbances to actuator delay and cross-track sensor noise.
- Use a fixed training catalogue, fixed plus seeded generated validation paths,
  and separate seeded generated held-out paths.
- Tune PID gains with bounded, seeded differential evolution and confirm the
  training shortlist on validation scenarios. SIMC is not used because the
  closed-loop path-following plant does not provide the simple process model
  and step-response assumptions that make SIMC attractive.
- Use completion and failure-adjusted integrated path error as co-primary
  outcomes while retaining every raw tracking, effort, and failure metric.
- Keep the dashboard read-only over batch artifacts. It cannot write into or
  recompute an official result directory.
- PID candidates use a declared balanced subset of the training manifest:
  nominal scenarios at all speeds, all stationary conditions at 0.5 m/s, and
  the transient combined condition at 0.5 m/s. This preserves every training
  geometry and disturbance type while keeping differential evolution
  computationally feasible. Final evaluation still uses complete manifests.

## 2026-08-19: Runaway guard correction

The first integration smoke test exposed a 2 m absolute-coordinate guard that
terminated valid paths. The guard was restored to the inherited 20 m numerical
safety bound. The independent 1 m path corridor remains the tracking-failure
criterion. No training or final experiment had been run.

## 2026-08-21: Derivative-filter selection

- Set the derivative-filter time constant to `0.01 s` before PID calibration.
- Noise-only probes showed that filtering is necessary to prevent command
  chatter and saturation. Combined delay/noise checks showed that larger time
  constants add excessive phase lag when used with the selected `0.06 s`
  actuator delay.
- The selected `0.01 s` value suppressed the unfiltered noise response while
  preserving better combined-condition tracking than `0.02 s`, `0.05 s`, or
  `0.10 s`.

## 2026-08-21: Practical PID selection and Kp boundary audit

- Treat candidates within `0.5%` of the best failure-adjusted error in the
  highest-completion band as practically equivalent. Select the lowest steering
  variation, then the lowest gain magnitude, inside that band.
- Validate five distinct error/smoothness/gain trade-offs rather than numerical
  optimizer-polishing copies. Existing candidate caches can be reselected with
  `calibrate_pid.py --reselect-only`.
- Define the PPO gain box from the validated nominal and robust gains plus a
  `15%` margin. This avoids exposing PPO to the entire differential-evolution
  search box. The current box remains provisional until nominal recalibration.
- The nominal calibration subset contains no injected actuator delay. The robust
  subset includes stationary delay, noise, combined disturbance, and transient
  combined scenarios using `0.06 s` delay and `0.0003 m` noise.
- A validation-only Kp sweep found that nominal error continued improving above
  the old `Kp=50` search ceiling. Error plateaued around `Kp=250-400`; `Kp=600`
  sharply increased steering variation and wheel saturation. In the robust set,
  `Kp=40` increased error by roughly tenfold and `Kp>=50` caused completion loss.
- Therefore the robust optimum is demonstrably delay-limited, while the nominal
  PID must be jointly recalibrated with an expanded Kp range before calibration
  and PPO scheduler bounds are frozen.

## 2026-08-22: Pragmatic calibration freeze

- Do not spend another multi-hour differential-evolution run pursuing a perfect
  delay-free PID. Freeze the nominal PID at `(250.0, 4.5454, 0.3967)`, retaining
  the jointly calibrated `Ki/Kd` and refining only `Kp` with the boundary sweep.
- On validation, `Kp=250` was within `0.31%` of the lowest error at `Kp=400`
  while reducing steering variation by about `40%`. On all six training paths,
  `Kp=250` had lower error and roughly one quarter of the steering variation of
  `Kp=400`. Both completed every scenario.
- Keep the robust PID at `(29.0026, 4.4669, 2.1370)`. Its calibration explicitly
  includes the selected delay and noise conditions.
- Freeze the scheduler action box at `Kp 15-300`, `Ki 3.5-5.0`, and `Kd 0-3.0`.
  The lower Kp limit is supported by earlier Blind_PPO delay sweeps; its exact
  PID results remain prior pilot evidence because that protocol used different
  paths and disturbance schedules.
- Use absolute gain-rate limits `(20.0, 0.5, 1.5)` per second for `(Kp, Ki, Kd)`.
  A box-relative limiter would become roughly seven times faster merely because
  the Kp box was widened, contradicting the prior finding that gain jitter is
  harmful on delayed plants.
- This is a strong empirical baseline, not a claim of globally optimal PID
  gains. No final-test result was inspected or used for this decision.

## 2026-08-22: Native PPO controller ports

- Keep one shared simulator, path sampler, allocator, timing contract,
  disturbance distribution, and evaluator for both learned controllers.
- Restore the proven Blind_PPO scheduler contract: 170 observations, piecewise
  gain mapping around `(26.0, 0.5, 4.4)`, gain box `Kp 10-50`, `Ki 0-1.5`,
  `Kd 2.5-6`, box-relative rate limit `0.5/s`, unfiltered derivative behavior,
  scheduler reward, and original PPO parameters.
- Restore the proven 50 Hz EndToEnd_RL contract: 130 observations, one steering
  action, `0.2 m` cross-track scaling, `0.1 m` tracking-cost scale, steering-rate
  reward, small initial exploration, and original PPO parameters.
- Controller-specific observation encodings and rewards are intentional because
  the actions have different physical meanings. Neither controller receives
  path preview, path identity, disturbance values, or hidden plant context.
- Preserve the historical training defaults and checkpoint techniques:
  scheduler `300k` decisions with validation selection; direct PPO `1M`
  decisions with best training-rollout reward. Training processes are launched
  independently and use one Torch thread by default.
- The earlier `Kp 15-300` box and absolute rate limits remain provenance for the
  fixed-PID calibration work but are superseded for the PPO scheduler.

## 2026-08-22: Shared physical checkpoint selection and 150 ms delay

- A validation audit showed that training-rollout reward selected a materially
  worse direct-PPO checkpoint than physical validation. On the 20-scenario
  diagnostic subset, the nominal direct model improved from `42.58 mm` to
  `11.30 mm` failure-adjusted error when the final rather than reward-selected
  checkpoint was used; the disturbed model improved from `6.41 mm` to
  `5.84 mm`.
- Both learned arms now select checkpoints on the same fixed 24-scenario
  validation subset: maximize completion first, then minimize mean
  failure-adjusted error. Rewards, observations, PPO parameters, and training
  budgets remain architecture-specific.
- The fixed and scheduled PID arms now both use the already selected `0.01 s`
  derivative filter and kick-free initialization. The previous scheduler-only
  unfiltered behavior caused roughly `510` steering-variation units per second
  and `78-79%` wheel saturation in the validation noise condition.
- Set the declared delay severity and disturbed-training delay range to
  `0.15 s`. The prior `0.06 s` delay produced too little separation and did not
  reproduce the delayed regime where the earlier scheduler showed promise.
- These changes alter the control dynamics, training distribution, and model
  selection rule. Every checkpoint under `artifacts/models/native_ports` is
  therefore superseded development evidence and must not be used in the final
  comparison. Protocol-v2 training writes to `artifacts/models/protocol_v2`.
- The fixed robust PID calibrated at `0.06 s` completed only `4/12` transient
  combined checkpoint scenarios at `0.15 s`, so it is superseded. Instead of
  reopening differential evolution, adopt `(18.0, 0.5, 4.4)` from the earlier
  Blind_PPO 150 ms Kp sweep. A deliberately small confirmation at
  `Kp = 18, 22, 26` completed every transient case; `Kp=18` had the lowest
  failure-adjusted error (`17.32 mm`, versus `23.99 mm` and `34.62 mm`). It also
  completed all 12 nominal checkpoint cases at `1.59 mm` failure-adjusted
  error. This is a pragmatic prior-informed baseline, not a new optimum claim.
- The bounds in `calibration.json` now only contain the two fixed baselines.
  PPO scheduler mapping uses the separate tracked `blind_ppo.json` contract.

## 2026-08-22: Scheduler reward scale, disturbance sampler, and gain box floor

A single-seed protocol-v2 audit showed disturbance training helping the direct
arm but hurting the scheduler: transient combined failure-adjusted error moved
from `11.89 mm` (nominal-trained) to `19.86 mm` (disturbance-trained), while
nominal error moved the other way, from `1.96 mm` to `1.39 mm`. Diagnosis and
the three resulting changes are recorded here. No held-out result was opened.

### Diagnosed cause: a saturated scheduler tracking term

The tracking reward is `x^2/(1+x^2)` with `x = distance / tracking_scale_m`.
The native Blind_PPO scheduler scale is `0.005 m`. Measured gradient of that
term is `130 /m` at `3 mm` error but `1.2 /m` at `34 mm`. Under the `0.15 s`
delay the plant tracks at `22-34 mm`, so the term sat pinned at `0.97-0.98`
and the scheduler's disturbed episodes contributed almost no tracking
gradient; `PROGRESS_WEIGHT = 5.0` and the terminal bonus dominated instead.

The remaining usable gradient came from the near-nominal episodes, and a fixed
gain sweep on the 12 transient combined validation scenarios shows those pull
`Kp` the wrong way. Nominal error falls monotonically in `Kp` (`2.79 mm` at
`Kp=6` to `1.25 mm` at `Kp=50`) while transient combined error rises steeply
(`9.38 mm` at `Kp=8`, `17.32 mm` at `Kp=18`, `34.62 mm` at `Kp=26`, with
completion loss from `Kp=32`). The disturbance-trained scheduler settled at
`Kp ~21-27` and the nominal-trained one at `Kp ~11-13`.

The direct arm uses a `0.1 m` scale, so its `16-44 mm` errors stayed inside the
responsive part of the same curve, and its disturbance training worked. Same
plant, same sampler, same severities, opposite outcome. That isolates the
tracking scale rather than the disturbance distribution as the primary cause.

### Changes

- Set the scheduler `tracking_scale_m` from `0.005` to `0.02`. At `0.02` both
  the nominal operating point (`3 mm`) and the disturbed one (`25 mm`) keep a
  live gradient. This supersedes the native Blind_PPO reward scale; every other
  scheduler reward constant is unchanged.
- Replace the disturbance-training sampler. It now mirrors the four evaluation
  conditions with weights `(nominal 0.10, delay 0.30, noise 0.20, combined
  0.40)`, takes the stationary or transient form with equal probability, draws
  the declared evaluation severity outright half the time, and steps delay and
  noise together in the combined transient exactly as the evaluation manifest
  builds them. Event times are sampled as a fraction of the episode's own ideal
  duration and resolved in `reset`, spanning `1.0 s` to `0.8 x` ideal duration.
  The previous sampler drew delay uniformly and then zeroed it whenever it
  added an event, giving a mean sampled delay of `0.0366 s` against a `0.15 s`
  evaluation severity, never co-occurring delay and noise steps, and a
  hardcoded `6-12 s` event window derived from an assumed `20 s` episode when
  actual training episodes run `9-53 s`. The replacement gives a mean applied
  delay of `0.0787 s` with `34.8%` of episodes at the declared severity.
- Lower the scheduler gain box floor from `Kp 10` to `Kp 5`. The nominal-trained
  scheduler pinned itself at `10.00`, the box floor, and the fixed sweep puts
  the transient combined optimum near `Kp 8`, below the old floor. The box came
  from the Blind_PPO port under the superseded `0.06 s` delay. The box-relative
  rate limit follows the widened box, from `20.0` to `22.5` per second for `Kp`.

### Open item recorded before the changes

The robust fixed PID `(18.0, 0.5, 4.4)` was selected from a three-point
confirmation at `Kp = 18, 22, 26`, and `18` was the lowest value tested. The
sweep above reaches `9.38 mm` at `Kp=8` against its `17.32 mm`, so the robust
baseline is under-tuned and every learned advantage over it is inflated.
Re-tune it downward before the freeze. Related: the nominal-trained scheduler's
`11.89 mm` is not distinguishable from a static `Kp=13` PID at `11.80 mm`, so
no adaptation claim is supported by the current evidence.

All four protocol-v2 seed-11 checkpoints predate these changes and are
superseded development evidence.

## 2026-08-23: Robust PID Kp, scheduler box ceiling, and dynamic delay

### Robust fixed PID lowered to Kp = 8

The `Kp = 18` robust baseline was selected from a three-point confirmation at
`18, 22, 26`, with `18` the lowest value tested. A downward sweep on the
28-scenario robust calibration subset gives `6.47 mm` at `Kp=8` against
`10.49 mm` at `Kp=18`, with completion held at `1.0` throughout. By the
existing `0.5%` practical-equivalence rule `Kp=8` is the only candidate in the
band. Set `robust` to `(8.0, 0.5, 4.4)` and lower the fixed-PID search floor
from `Kp 10` to `Kp 4` so the box contains the selection. `Ki` and `Kd` are
unchanged, and the nominal PID stays at `(250.0, 4.5454, 0.3967)`.

### Scheduler gain box ceiling raised to Kp = 300

The scheduler box was `Kp 10-50`, later `5-50`, while the fixed-PID box is
`4-300` and the nominal optimum is `Kp = 250`. The scheduler could not
represent the nominal-optimal gain at all, so the strategy gain scheduling
exists to express -- stiff while undisturbed, soft once delayed -- was
unreachable by construction. The best it could do was imitate the robust PID,
which is what the audit shows: `2.59 mm` nominal against the robust PID's
`2.35 mm`.

Set the scheduler box to `Kp 5-300`, matching the fixed-PID search range. The
piecewise mapping around `Kp = 26` puts fine resolution below the anchor, where
delayed operation needs it, and coarse resolution above it across the flat
`250-400` nominal plateau. The box-relative rate limit follows the box to
`147.5 /s` for `Kp`, which traverses `250 -> 8` in `1.64 s` against a
transient event at `6.0 s`. This supersedes the `20.0 /s` absolute limit and
the earlier concern about box-relative limits scaling with box width: here the
faster limit is required for the box to be usable at all.

### Transient disturbances are no longer a single one-way step

A transient episode now takes one to three changes per channel (weights
`0.50, 0.30, 0.20`), and any change after the first returns the channel to zero
with probability `0.35`. Measured over 4000 samples, `25%` of disturbed
episodes now recover to zero delay. The previous single permanent step taught
the policies that a disturbance, once seen, never ends. Evaluation manifests
are unchanged and still use one declared step, so training covers a superset
of the scored condition.

Condition weights move from `(0.10, 0.30, 0.20, 0.40)` to
`(0.25, 0.25, 0.15, 0.35)`. At `0.10` clean episodes the direct disturbed arm
regressed on the nominal condition from `7.81 mm` to `15.81 mm`, worse than the
nominal-trained direct arm, failing the requirement that disturbance-trained
agents retain acceptable nominal behavior.

### Attribution caveat

The box ceiling, the dynamic transients, and the condition reweighting land
together, so the next run cannot attribute its result to any one of them.
If the scheduler improves materially, re-run with the box change alone before
claiming the box was the cause.

Superseded by these changes: all `protocol_v3` scheduler checkpoints, and the
`protocol_v3` direct disturbed checkpoint. The `protocol_v3` direct nominal
checkpoint is unaffected -- the nominal regime never enters the disturbance
sampler and the scheduler box does not apply to the direct arm.

## 2026-08-23: Sampler ablation and the frozen training distribution

The protocol v4 scheduler result was confounded by three simultaneous changes.
Re-running the disturbed scheduler with the widened gain box but the previous
sampler separates them, on the 12 transient combined audit scenarios:

| Gain box | Sampler | Disturbed scheduler |
|---|---|---:|
| `Kp 5-50` | single step | `11.48 mm` |
| `Kp 5-300` | single step | `8.82 mm` |
| `Kp 5-300` | dynamic | `6.86 mm` |

The box accounts for roughly `60%` of the improvement and the dynamic
transients for the rest. Neither alone explains it. The box change alone is
already enough to pass the re-tuned robust PID at `9.38 mm`, so the learned
result does not depend on the sampler change.

Adopt the `dynamic` sampler for both learned arms. A disturbance that can end
is more defensible than one that is permanent by construction, and the
experiment requires one shared training distribution across architectures.

The cost is carried by the direct arm. At `1M` decisions under the dynamic
sampler it reached `29.31 mm`; at `2M` it reached `16.54 mm`, against
`14.52 mm` under the single-step sampler. Its validation completion still
oscillates between `0.46` and `1.00` late in training, so the instability is a
property of the arm rather than of the budget. Raise the direct training budget
to `2M` decisions and report the instability as a result.

Recorded limitation: the disturbed scheduler does not implement the
stiff-when-clean, soft-when-delayed strategy that motivated widening the box.
Measured mean `Kp` is about `6.7` before the event in both clean and delayed
episodes, rising only to `8.0` afterwards, while gain total variation rises
from `4.0` to `43.7` per second. It wins by fine modulation around a low gain,
not by switching operating points, and the write-up must describe the measured
mechanism rather than the intended one.

## 2026-08-23: Declared machine and cross-machine parity tolerance

The five-seed matrix runs on the university SLURM cluster
(`cpunode*.dyn.scc.plus.ac.at`), not the laptop. `PLAN.md` already requires
final results from one declared machine, so **the cluster is that machine** and
the fixed-PID arms are re-evaluated there. No laptop number enters the final
result tables.

`check_parity.py` originally demanded bit-identical agreement between machines.
That was wrong. With every pinned package version identical, the cluster
reproduced all four completion rates exactly but the continuous metrics drifted:

| arm / condition | laptop | cpunode13 | drift |
|---|---:|---:|---:|
| `pid_global_nominal` nominal | `1.1586 mm` | `1.1601 mm` | `0.13%` |
| `pid_global_robust` nominal | `2.3542 mm` | `2.3623 mm` | `0.34%` |
| `pid_global_robust` combined | `9.3779 mm` | `9.3790 mm` | `0.01%` |
| `pid_global_nominal` combined | `151.96 mm` | `157.52 mm` | `3.66%` |

MuJoCo is not bit-identical across CPU architectures, and a last-bit difference
in one contact force amplifies over a 20-50 s closed-loop episode. The check now
requires exact agreement on episode count and completion rate, and a `1%`
relative band on the continuous metrics.

The `3.66%` outlier is `pid_global_nominal` under the transient combined
condition, which completes `0/12`. Its `J_FA` is dominated by
`corridor * (T_max - t_end)`, a discontinuous term that moves by percent when
the corridor exit shifts by a millisecond. Arms that do not complete every
episode are therefore checked on completion and failure mode only. The claim
that arm supports is that a nominal-tuned PID cannot survive a `0.15 s` delay,
and that claim rests on the completion rate, which reproduced exactly.

This is a sanity check against a wrong environment, not a reproducibility
guarantee. Exact version pinning is enforced separately by
`tools/show_versions.py`.

## 2026-08-24: Final five-seed run, and what was and was not frozen

### The run

Twenty PPO runs and the complete held-out evaluation executed as one SLURM job
on the declared machine.

| Item | Value |
|---|---|
| Code commit | `5240d11ccea0b1370ac73d83d5142cc1e02d4b6d` (branch `protocol-v2-reward-fix`) |
| Machine | `cpunode13.dyn.scc.plus.ac.at`, AMD EPYC 9334 32-Core |
| SLURM job | `1760422`, 20 cores, one node |
| Training | 20/20 runs, 95.9 min wall |
| Evaluation | 22 controllers x 480 held-out scenarios = 10560 episodes |
| Seeds | 11, 23, 37, 53, 71 |
| Sampler | `dynamic` |
| Result directory | `artifacts/results/final/` |

### Provenance actually verified, not asserted

- **The held-out set was unopened before this run.** Eight prior evaluation
  directories contain 264 distinct scenario IDs between them, and the
  intersection with the 480 held-out IDs is **empty**. Every development audit
  used `validation.json` or a subset of it.
- **The shipped models are the evaluated models.** All twenty
  `best_model.zip` files hash-match the `model_sha256` recorded by the
  evaluator on the server.
- **The shipped source is the source that trained them.** All eight files in
  `source_sha256` match once line endings are normalised. The recorded hashes
  are computed on raw bytes, so a Windows CRLF checkout differs from the LF
  bytes hashed on Linux for `train.py`, `env.py` and `controller_profiles.py`;
  the content is identical. Future hashing should normalise line endings.
- **Parity.** The fixed-PID arms reproduced all four completion rates exactly
  against the laptop reference, with `0.01%` to `0.34%` drift on the continuous
  metrics for completing arms.

### What was frozen, and what was not

Frozen before the run: controller code, plant, allocator, PID implementation,
derivative filter, both calibration artifacts, reward constants, PPO
hyperparameters, observation contracts, training budgets, the disturbance
sampler, the manifests, and the five seeds.

**Not pre-registered: the practical significance thresholds.** `PLAN.md`
requires effect thresholds to be stated before the final test results are
opened. They were not. The confidence intervals in `paired_effects.csv` are
therefore reported without a pre-declared threshold for what counts as a
practically meaningful difference, and the thesis must say so rather than
implying otherwise.

**The experiment is not strictly confirmatory.** The reward scale, the
scheduler gain box, the disturbance sampler, and the robust PID gains were all
changed during development in response to validation evidence, each recorded
above with its reason. No held-out result influenced any of them, which the
scenario-ID check above demonstrates. The correct description is a final
controlled evaluation after iterative development, not a confirmatory
experiment, and the write-up must use that wording.

### Per-path PID benchmarks are absent by design

`PLAN.md` describes privileged per-path PID benchmarks, and
`artifacts/calibration/per_path_gains.json` exists. They are not in the final
result because they are tuned per fixed named path while the held-out manifest
is entirely generated geometry; `PLAN.md` already states that generated-path
generalisation compares the global PIDs and the learned controllers only. Their
absence is intentional and should be stated in the thesis rather than left for
a reader to notice.

## 2026-08-24: The scheduler mechanism, measured on the sealed final models

An earlier note recorded that the scheduler "wins by fine modulation around a
low gain, not by switching operating points." **That was measured on the
single-seed protocol v4 model and is wrong for the final five.** The finding is
withdrawn and replaced by the following, measured on the sealed checkpoints
over 12 clean and 12 transient-combined held-out scenarios per seed.

### The disturbance-trained scheduler does switch

| seed | clean `Kp` | post-event `Kp` | clean `J_FA` | transient combined `J_FA` |
|---:|---:|---:|---:|---:|
| 11 | `9.71` | `11.51` | `2.05 mm` | `9.14 mm` |
| 23 | `225.56` | `39.02` | `1.16 mm` | `15.99 mm` |
| 37 | `38.49` | `6.23` | `1.32 mm` | `7.59 mm` |
| 53 | `35.76` | `8.76` | `1.41 mm` | `8.72 mm` |
| 71 | `27.39` | `15.70` | `1.53 mm` | `10.35 mm` |
| | | | PID nominal `1.16` | PID robust `9.10` |

Four of five seeds reduce `Kp` after the disturbance appears, and seed 23 runs
the full intended strategy: `Kp = 225` while clean, which reproduces the nominal
PID's `1.16 mm` exactly, dropping to `39` once delayed.

### Post-event gain almost perfectly predicts delayed error

`corr(post-event Kp, transient combined J_FA) = +0.998` across the five seeds.
The seeds that settle near the robust-PID optimum of `Kp = 8` achieve the
lowest delayed error; seed 23, which only reaches `39`, is the worst. Delayed
performance is therefore explained almost entirely by how close the policy gets
to the known optimal gain, not by how much it modulates.

This is post-hoc analysis on an opened test set with five points. It explains
the observed spread; it is not a pre-registered hypothesis test, and the thesis
must present it as mechanism rather than confirmation.

### Seed 37 is an existence proof, and the reliability finding

Seed 37 beats the robust PID on **both** axes: `1.32 mm` against `2.29 mm`
clean, and `7.59 mm` against `9.10 mm` under transient combined disturbance. So
gain scheduling can dominate a strongly tuned fixed PID; it is not a ceiling
effect.

What fails is reproducibility. The same training procedure, differing only in
seed, produces clean-path `Kp` between `9.7` and `225.6` -- a standard deviation
of `89` on a `5-300` box. The aggregate five-seed result is a tie with the
robust PID because the procedure sometimes finds the good policy and sometimes
does not.

The honest conclusion is therefore not that classical control wins, nor that
learned scheduling wins. It is that learned gain scheduling **can** beat a
strongly tuned PID on both nominal accuracy and delay robustness, but PPO's
seed variance makes it do so unreliably. The engineering barrier this
experiment identifies is training reproducibility, not the control concept.

## 2026-09-08: Provenance of the 200 ms observation window

Recorded while drafting Chapter 4. This is a documentation note, not a change:
no code, parameter, or result was touched.

`CONFIG.history_length = 10` was inherited unchanged from
`Path_Following_PPO/config.py` through `Blind_PPO/config.py`. Its only recorded
justification in any of the three projects is the arithmetic that ten frames at
the 50 Hz policy rate span 200 ms. There is no literature source for the value
and no history-length ablation was run, here or in the predecessor projects.

Two points worth keeping separate. `FRAME_SKIP = 10` (physics steps per policy
decision, fixed by 500/50) and `HISTORY_LENGTH = 10` (frames of memory) are
independent quantities that happen to share a value; nothing links them.

The one deliberate piece of design around the window is in `EndToEnd_RL`: when
the direct arm was run at 500 Hz, the window was held fixed **in seconds**
rather than in frames -- a 100-frame buffer subsampled every tenth frame, still
200 ms, still 130 dimensions instead of 1300. That establishes memory as a
physical timescale, not a tensor shape. It does not establish that 200 ms is
the right timescale.

Post-hoc, the value is defensible but was not verified against the disturbance
severity it now has to cover:

- A blind policy can only infer hidden actuator lag if a command and its
  delayed consequence both fall inside the same observation. The declared
  delay severity is `0.15 s`, so the 200 ms window covers it by 50 ms. That
  severity was raised from `0.06 s` to `0.15 s` on 2026-08-22, well after the
  window was fixed, and the window was never re-checked against it.
- At the `0.3-0.7 m/s` target speeds, 200 ms spans `6-14 cm` of travel: shorter
  than a path feature, long enough to resolve error rate and yaw response.
- 170 inputs into a `[64, 64]` policy is already `30471` parameters. Doubling
  the window doubles the input layer against a `300k`-decision budget.

The thesis states the window as an inherited, untuned design choice held
constant across both learned architectures, notes that it exceeds the declared
dead time, and lists the missing ablation as a limitation. A 5/10/20-frame
ablation on the disturbed scheduler arm, on development seeds and the
validation manifest only, would close it; it was deliberately not run.

## 2026-11-13: Post-hoc experiments opened, sealed run untouched

Follow-up experiments from `D:\Msc\Thesis_Final\Experiments\Experiment_Plan.md`
run on branch `posthoc-experiments`. They read the sealed checkpoints and
manifests and write only to `artifacts/results/posthoc/` and
`artifacts/models/posthoc/`; `posthoc_eval.py` refuses any output path inside
the sealed directories. No default of `evaluate.py`, `train.py`, or `env.py`
changed.

Deviations from the plan, decided before any post-hoc result was seen:

- **Transient mode included in both sweeps.** The plan swept stationary
  scenarios only. Each sweep now also runs the 60 transient records, with the
  swept value as the step size at the sealed 6 s onset, so 120 scenarios per
  level and condition instead of 60.
- **Sweeps are built from the sealed disturbed records, not the nominal ones.**
  Noise levels override the sealed `noise` and `combined` records, delay levels
  the sealed `delay` records. Noise seeds therefore match the sealed run, and
  at the sealed severity a swept scenario is identical to its sealed record.
  Those rows replay the sealed run for every controller (360 scenarios x 22
  instances), which replaces the plan's 24-scenario replay slice.
- **0.40 s delay is run for every arm**, not only for arms still completing at
  0.30 s. It is a superset and costs about 2,600 episodes.
- **E3 frozen variants keep the scheduler's rate-limited start.** A frozen
  controller runs in scheduler mode with a constant action, so its gains ramp
  from the nominal gains to the frozen target exactly as the learned
  scheduler's do. Frozen gains are time averages over the validation manifest:
  `clean` = nominal condition, `delay` = delay and combined, both modes.

A laptop smoke test against the sealed (cluster) episodes matched completion on
every row but drifted 1-18 % in learned-arm E_FA under delay, the known
cross-CPU effect. Post-hoc comparisons are therefore made within one post-hoc
run, against its own re-run of the sealed severity, never against sealed
numbers from another node.

### Stage 2 design, fixed before any stage-1 result was read

- **E4 checkpoint selection.** The 24 checkpoint scenarios keep their split
  (12 stationary-nominal, 12 transient-combined); the transient-combined noise
  step carries sigma* instead of 0.3 mm. No extra noise-only group is added.
- **E4 training noise.** Every noise severity the disturbed sampler draws,
  first change and later changes, is log-uniform in `[0.3 mm, sigma*]`
  (`env.train_noise_log_range_m`). Condition weights, delay draws, timing,
  and recovery are untouched. With the option unset the RNG stream is the
  sealed one.
- **PID-RN is a grid, not differential evolution.** The plan asks for the same
  DE setup, but the sealed PID-R `(8.0, 0.5, 4.4)` was itself chosen by a Kp
  sweep on the 28-scenario robust validation subset, not by DE (2026-08-23
  above). PID-RN repeats that procedure with sigma* in every noise-bearing
  scenario of the subset, over Kp `{4..40}` x Kd `{0.5..8}` x tau_d
  `{0.01, 0.02, 0.05, 0.1}` at the sealed Ki `0.5`, selected by the same
  `select_practical_candidate` rule. The sealed point is always in the grid.
  A laptop plumbing test at a guessed 10 mm selected Kd at the old grid floor
  of 2.0, so the Kd floor was lowered to 0.5 and tau_d 0.1 added; that test
  is not a result and used no held-out scenario.
- **E5 history** is a train-time option (`--history-length 1`); evaluation reads
  the history length off each checkpoint's input size.
- **Same node for both stages.** Stage 1 ran on `cpunode39`; stage 2 is pinned
  there and re-runs all 22 sealed instances on the 480 held-out scenarios, so
  E4 and E5 arms are compared with sealed checkpoints evaluated on the same CPU.

## 2026-11-13: Stage 1 done; stage-2 choices made after reading it

Stage 1 ran as SLURM job 3187230 on `cpunode39` (AMD EPYC 9334, the same CPU
model as the sealed run's `cpunode13`) at commit `479afe5`, clean tree. Parity
passed. All 7,920 sweep rows identical to a sealed scenario reproduced the
sealed `episodes.csv` bit for bit.

Two choices were made **after** seeing the coarse noise sweep, and are
therefore post-hoc in the strict sense:

- **E1 fine pass runs all 12 rule-generated levels**, not the plan's cap of 6.
  The breaking-point rule brackets four coarse intervals (1-3, 3-10, 10-30,
  30-100 mm), three levels each. The cap was a budget guard; the cluster was
  idle. Levels: 1.316, 1.732, 2.28, 4.054, 5.477, 7.401, 13.161, 17.321,
  22.795, 40.536, 54.772, 74.008 mm.
- **E4 runs at two sigma\* values.** The literal rule gives sigma\* = 3 mm:
  GS-D noise-only reaches 99 % completion and 4.2 mm E_FA against a 1.51 mm
  baseline there. But GS-D and PID-R collapse at 10 mm (15 % and 2 %
  completion noise-only), inside the plan's 10-30 mm realistic band, and
  training on 0.3-3 mm cannot test whether training fixes that collapse.
  E4 therefore runs at 3 mm (rule) and 10 mm (sensitivity), each with its own
  PID-RN retune. Arms carry the level in their name, e.g.
  `ppo_direct_noisetrained_10mm`, `pid_global_robust_noise_10mm`.

## 2026-11-14: Stage 2 finished on cpunode04; results written up

Stage-2 job 3188678 (cpunode39, commit `9b72c94`) finished all 30 training
runs, both PID-RN retunes and the E1 fine sweep. It then stopped under
`set -e` on `tail -3` with several files, which this cluster's coreutils
rejects. The remaining evaluation ran unchanged as `slurm_posthoc2b.sh`
(job 3189024, commit `561e425`), on cpunode04 rather than cpunode39 because
cpunode39 was fully allocated. cpunode04 is the same CPU model (AMD EPYC
9334). Its re-run of all 22 sealed instances on the 480 held-out scenarios
matched the sealed `episodes.csv` bit for bit (10,560 rows), so the node
change does not affect any comparison.

PID-RN selections: 3 mm -> (6, 0.5, 2.0, tau_d 0.02); 10 mm -> (6, 0.5, 1.0,
tau_d 0.02). Neither is on a grid edge.

`POSTHOC_RESULTS.md` is generated by `posthoc_report.py` from
`artifacts/results/posthoc/`; free-text observations live in
`posthoc_observations.md`. Paired intervals reuse `analyze._bootstrap`
(10,000 draws, generator seed 20260819). A copy goes to
`D:\Msc\Thesis_Final\Experiments\POSTHOC_RESULTS.md`.

## 2026-11-16: Plan 2 — H = 1 and ranged disturbances as the main design

`D:\Msc\Thesis_Final\Experiments\Experiment_Plan_2.md`, revised with the user:

- H = 1 (one frame, which carries the previous action) is the main design
  for every learned arm, and H = 10 is the history ablation. This is treated
  as a design fix, not a selection among H values. H = 10 is reported only
  where it differs meaningfully.
- The disturbed arms train on ranges instead of a declared point:
  - delay uniform in [0, 0.3] s, with no p = 0.5 spike at 0.15 s;
  - noise log-uniform in [0.1, 10] mm;
  - the condition grid, the change count (1-3) and the recovery probability
    are unchanged.
  - New flags: `train.py --train-delay-range-s`, `--train-noise-min-m`.
    Defaults reproduce the sealed sampler, with the same RNG call order.
- Checkpoint selection for the ranged arms uses
  `manifests/posthoc/validation_ranged.json`, `validation.json` with its
  own draws from the same ranges (seed 20261117).
- The H = 10 ablation is retrained with the same ranged sampler
  (`ppo_{kind}_ranged_h10`). Without that, an H1 − H10 difference would mix
  the history change with the sampler change. The sealed H = 10 and E5 H = 1
  arms (old sampler) remain as references.
- PID-RN is re-tuned on the ranged validation subset by the same grid and rule
  as E4 (`posthoc_pid_retune.py` without `--sigma-m`).
- Main table: `manifests/posthoc/held_out_ranged.json`, the 480 held-out
  records with each delay drawn from U[0, 0.3] s and each noise level from
  logU[0.1, 10] mm (seed 20261116). The fixed-severity held-out set runs
  alongside it as a comparison and a sealed replay check.
- F4 dynamic: `manifests/posthoc/dynamic.json`. The 60 stationary delay
  records are given a delay redrawn uniform in [0, D_max] after holds drawn
  uniform in [1, 3] s, for D_max = 0.15, 0.3 and 0.45 s, in a delay-only
  variant and a combined variant. The combined variant adds noise on its own
  schedule, log-uniform in [0.1, 10] mm, and has the same delay schedule. No
  environment change was needed: these are ordinary multi-event scenarios.
- F3 runs every level for every arm (0-0.8 s, `delay_wide.json`) instead of
  stopping early. The 960 E2 records inside it are byte-identical, so the
  existing arms double as a replay check.
- Everything runs as `slurm_plan2.sh`. Results go to `artifacts/results/posthoc2/`
  and models to `artifacts/models/posthoc/plan2/`.

## 2026-11-16: Plan 2 run finished; PID-RN settled after two grid extensions

- Job 3225586 ran on cpunode13, commit `578dda6`.
  - All 30 training runs finished.
  - Sealed replay matched bit for bit: 10,560 held-out rows and 2,640 delay-sweep rows.
- Its PID-RN retune picked Kp 4.0, the bottom of the Kp grid.
  - Round 2 (wide Kp grid, 1-40) picked (2.0, 0.5, 0.5, tau_d 0.01). Kd and tau_d were then at their lower edges.
  - Round 3 (fine grid, Kd down to 0, tau_d down to 0.005) picked the same point.
  - The selection rule applied to all 424 candidates of the three rounds also picks it.
  - Final PID-RN: (2.0, 0.5, 0.5, tau_d 0.01). Validation E_FA is 12.04 mm at 100 % completion; the Kp 4.0 pick had 15.16 mm.
  - Its rows come from `artifacts/results/posthoc2/pidrn_*`. Rounds 2 and 3 produced identical rows. The Kp 4.0 rows in the job-3225586 directories are superseded.
- `ppo_direct_ranged_h10` seed 37 is a poor run: held-out nominal completion 78 %, validation E_FA 205 mm. It was selected by the unchanged checkpoint rule and is reported as is.
