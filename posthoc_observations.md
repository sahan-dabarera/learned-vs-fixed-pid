# Observations for POSTHOC_RESULTS.md

Read by posthoc_report.py. One section per heading; numbers are copied from the
generated tables and must be updated by hand if the tables change.

## 0 Deviations

**Deviations from the plan (and why).** Items 1–8 were fixed before any post-hoc result was seen; items 9–10 were decided after reading the stage-1 coarse sweep; items 11–12 are about how the jobs ran. All are also logged in `DECISIONS.md` (2026-11-13 and 2026-11-14).

1. **Transient mode included in E1 and E2.** The plan swept stationary scenarios only. Each sweep also runs the 60 transient records, with the swept value as the step size at the sealed 6 s onset. So there are 120 scenarios per level and condition instead of 60.
2. **Sweeps are built from the sealed disturbed held-out records, not from a SWEEP-60 subset of the nominal records.**
   - Noise levels override the sealed `noise`/`combined` records; delay levels override the sealed `delay` records. Noise seeds, paths and speeds therefore match the sealed run.
   - At the sealed severity, a swept scenario is identical to its sealed record. Every such row replays the sealed run, which replaces the plan's 24-scenario replay slice.
3. **0.40 s delay was run for every arm**, not only for arms still at 100 % at 0.30 s. This is a superset of the plan.
4. **E3 frozen variants keep the scheduler's rate-limited start.**
   - A frozen controller runs in scheduler mode with a constant target action, so its gains ramp from the nominal (26, 0.5, 4.4) exactly as the learned scheduler's do.
   - F-clean and F-delay come from the full 96-scenario validation manifest: nominal condition (24 scenarios) and delay + combined (48 scenarios). The validation set has enough delay scenarios, so no delay was injected.
5. **E4 checkpoint selection.** The 24 checkpoint scenarios keep their split (12 stationary-nominal, 12 transient-combined). The transient-combined noise step carries σ* instead of 0.3 mm. No noise-only group was added.
6. **E4 training noise.** Every noise severity the disturbed sampler draws (first change and later changes) is log-uniform in [0.3 mm, σ*]. Condition weights, delay draws, timing and recovery are unchanged.
7. **PID-RN is a grid search, not differential evolution.**
   - The sealed PID-R was itself chosen by a Kp sweep on the 28-scenario robust validation subset, not by DE (DECISIONS 2026-08-23). PID-RN repeats that procedure with σ* in every noise-bearing scenario of the subset.
   - Grid: Kp {4, 6, 8, 10, 13, 18, 26, 40} × Kd {0.5, 1, 2, 3, 4.4, 6, 8} × τd {0.01, 0.02, 0.05, 0.1}, Ki 0.5, plus the sealed point.
   - Selection uses the same `select_practical_candidate` rule.
   - A laptop plumbing test at a guessed 10 mm picked Kd at the then-floor of 2.0, so the Kd floor was lowered to 0.5 and τd 0.1 was added. That test used no held-out scenario. Neither final selection is on a grid edge.
8. **E5 runs H = 1 only** (no optional H = 5).
9. **E1 fine pass ran all 12 rule-generated levels**, not the plan's cap of 6. The rule bracketed four coarse intervals (1–3, 3–10, 10–30, 30–100 mm) at three levels each. The cap was a budget guard, and the cluster was idle. The rule pooled stationary and transient rows.
10. **E4 ran at two σ\* values.**
    - The literal rule gives σ* = 3 mm: GS-D noise-only, both modes pooled, reaches 99.2 % completion at 3 mm.
    - GS-D and PID-R lose most completion at 10 mm, inside the plan's 10–30 mm "realistic" band, so 10 mm was added as a sensitivity run with its own PID-RN retune.
    - The E4 trigger (GS-D or DR-D breaks at ≤ 30 mm) was met by GS-D at 3 mm.
11. **Stage 2 was resumed on a different node.**
    - Job 3188678 (cpunode39) trained all arms and ran the PID-RN retunes and the E1 fine sweep. It then stopped on a shell error (`tail -3` with several files) before evaluation; no result was affected.
    - The remaining steps (held-out rerun, E4 sweep) ran as job 3189024 on cpunode04, because cpunode39 was fully allocated. cpunode04 has the same CPU model, and its re-run of all 22 sealed instances on the 480 held-out scenarios was bit-identical to the sealed run (10,560 rows).
    - In the E4 sweep tables, the sealed-arm columns therefore come from cpunode39 (E1) and the new-arm columns from cpunode04. Both nodes reproduced the sealed run bit for bit.
12. **E6 (recovery metric) skipped**, as decided in the plan.

## E1

- **Noise-only completion loss starts between 3 and 5.5 mm for PID-R, GS-N and GS-D.**
  - At 4.054 mm stationary: PID-R 65.0 %, GS-N 55.0 %, GS-D 77.3 %.
  - At 10 mm stationary: 0 %, 3.7 % and 6.3 %.
  - PID-N stays at 100 % up to 5.477 mm, then has 68.3 % at 10 mm and 0 % at 100 mm.
  - DR-N completes ≥ 96.7 % and DR-D ≥ 99.7 % at every level up to 100 mm.
  - In transient mode the same arms lose completion at higher σ than in stationary mode. For example, PID-R completes 100 % at 4.054 mm in transient mode and 65.0 % in stationary mode.
- **Steering and wheel saturation (noise-only, stationary), from 0.3 mm to 3 mm:**
  - steering TV/T rises from 49 to 344 /s for PID-N, 69 to 449 for PID-R, 77 to 456 for GS-N, and 73 to 426 for GS-D;
  - wheel saturation rises from 1–2 % to 72–88 % for the same four arms;
  - DR-N stays at 1–2 /s and 0 %, and DR-D at 6 /s and 2–3 %. At 30 mm DR-D is at 10 /s and 6 %.
  - At 3 mm, where PID-R, GS-N and GS-D begin to lose completion, all four PID-based arms show both high steering TV/T and high wheel saturation. PID-N shows the same at 3 mm (344 /s, 72 %) but still completes 100 %. The direct arms show neither.
- **Combined (σ + 0.15 s delay):**
  - PID-N completes 0 % at every level.
  - PID-R and GS-D keep 100 % up to 3 mm, and their E_FA falls as σ rises from 0.3 to 3 mm: PID-R 13.3 to 8.82 mm and GS-D 14.7 to 8.85 mm stationary; 9.10 to 6.71 mm and 10.4 to 6.97 mm transient. Above 3 mm, completion falls.
  - GS-N completes 64.7–73.0 % (stationary) up to 3 mm.
  - DR-D completes ≥ 99.3 % at every level, with E_FA 17.0 mm (0.3 mm) to 43.8 mm (100 mm) stationary.
- **E1-F (slower derivative filter on PID-R):**
  - τd = 0.05 s keeps PID-R at 100 % completion up to 10 mm (E_FA 4.00 mm stationary, 3.45 mm transient). At the sealed τd = 0.01 s, the same level gives 0 % and 3.3 %.
  - At 30 mm τd = 0.05 s completes 21.7 % / 46.7 %, and at 100 mm 0 %. τd = 0.02 s gives 33.3 % / 93.3 % at 10 mm.
  - At 3 mm, τd = 0.05 s lowers steering TV/T from 449 to 160 /s and wheel saturation from 86 % to 14 %.
  - E1-F was run noise-only. Its effect under delay is not measured here; PID-RN (E4) covers filter changes under delay.
- **The breaking-point rule, applied per mode on coarse + fine levels,** gives 3 mm for PID-R, GS-N and GS-D noise-only, and 4.054–5.477 mm for PID-R and GS-D combined. DR-D noise-only breaks at 13.161 mm stationary (a single failed episode, 99.7 %) and at 100 mm transient (E_FA criterion).

## E2

- **PID-N** loses completion at the first non-zero delay (33.3 % stationary, 58.3 % transient at 0.03 s) and is at 0 % from 0.10 s.
- **PID-R** completes 100 % up to 0.20 s in both modes, 33.3 % / 88.3 % at 0.30 s and 0 % at 0.40 s.
- **GS-N** completes 92.7 % (stationary) at 0.10 s, 64.7 % at 0.15 s and 0 % at 0.30 s. **GS-D** completes 100 % up to 0.15 s, 83.7 % / 86.0 % at 0.20 s, 17.7 % / 31.7 % at 0.30 s and 0.7 % / 2.0 % at 0.40 s.
- **The direct arms have the highest completion at 0.30 s and 0.40 s.**
  - At 0.30 s: DR-N 90.7 % / 94.7 %, DR-D 67.7 % / 91.7 %.
  - At 0.40 s: DR-N 29.0 % / 81.7 %, DR-D 22.7 % / 46.3 %.
  - Their E_FA at 0 s is the highest of the six arms (7.12 and 9.51 mm, against 1.16–2.29 mm for the others).
- **At the sealed stress level of 0.15 s**, PID-N completes 0 %, while PID-R, GS-D and DR-D complete 100 %, with stationary E_FA 13.0, 14.2 and 18.0 mm. By the breaking-point rule, PID-N breaks at 0.03 s; PID-R, GS-N, GS-D and DR-N at 0.10 s; and DR-D at 0.20 s.

## E3

- **Nominal and noise cells:** frozen and dynamic schedulers have equal completion (100 %), and seed-mean |ΔE_FA| is ≤ 0.22 mm for every variant.
- **GS-D F-clean** completes 46.3–53.0 % in the delay and combined cells, against 100 % for dynamic GS-D.
  - Seed-mean Δcompletion is −47.0 to −53.7 pp, with 95 % CIs excluding 0.
  - Per seed: seed 11 (F-clean gains 7.85 / 1.47 / 2.58) loses nothing; seed 23 (Kp 227, Ki 0.018) loses 100 pp.
- **GS-D F-delay** completes 86.7 % in every delay and combined cell.
  - The whole loss is seed 23 (−66.7 pp).
  - The other four seeds keep 100 %, with ΔE_FA between −3.8 and +6.9 mm.
  - For seed 23 the F-delay Kp mean is 47.3 and the median is 18.8; evaluation used the mean.
- **GS-D F-delay gains vs F-clean:** Kp is lower under F-delay for 4 of 5 seeds (all except seed 11).
- **GS-N:** frozen − dynamic completion differences vary in sign across seeds.
  - F-clean: from +93.3 pp (seed 11) to −66.7 pp (seeds 53 and 71). The seed-mean is +2.0 / +1.0 pp stationary and −12.7 / −12.3 pp transient, and the CIs include 0.
  - F-delay: seed-mean −8.3 to −12.7 pp, with CI upper bounds at 0.0 to +5.0.
- **Against PID-R** (100 % in every held-out cell), GS-D F-delay is the only frozen variant above 80 % completion in the delay cells. Paired frozen − PID-R differences are in the collapsed table.

## E4 method

- **σ\*:** 3 mm (plan rule) and 10 mm (sensitivity, chosen after the coarse sweep; see §0 item 10).
- **Sampler change:** every noise severity drawn in training is log-uniform in [0.3 mm, σ*]. Nothing else in the disturbance sampler changed. Checkpoint selection puts σ* on the 12 transient-combined checkpoint scenarios.
- **Arms:** GS-DN and DR-DN at each σ*, 5 seeds each, with the sealed PPO hyperparameters and budgets.
- **PID-RN:** a grid search on the 28-scenario robust validation subset, with σ* in its 8 noise-bearing scenarios. The subset is 12 nominal, 4 delay-only, 4 noise-only, 4 stationary-combined and 4 transient-combined; 12 of the 28 carry delay. See §0 item 7.

## E4

- **GS-DN 3 mm:**
  - Noise-only sweep: breaks at 3 mm like GS-D (stationary completion 87.0 % vs 98.3 % at 3 mm; 6.0 % vs 6.3 % at 10 mm).
  - Held-out 480: completion is identical to GS-D in every cell, and every ΔE_FA CI includes 0.
- **GS-DN 10 mm:**
  - Noise-only sweep: lower completion than GS-D at 3 mm stationary (94.7 % vs 98.3 %), and higher from 4.054 to 13.161 mm (at 10 mm: 17.3 % vs 6.3 % stationary, 44.3 % vs 23.0 % transient). At 17.3 mm and above it completes ≤ 4.7 %.
  - Held-out delay and combined cells: 73.7–74.0 % completion vs 100 % for GS-D (seed-mean −26 pp, CI excludes 0). The loss comes from seeds 11, 53 and 71.
- **DR-DN 3 mm:**
  - Held-out: seed 11 differs from DR-D seed 11 (ΔE_FA +70.7 mm in nominal, −3.3 pp completion; −36.7 pp in stationary delay). The other four seeds are within ±12.1 mm ΔE_FA and 0 to −1.7 pp completion.
  - Pooled sweep completion is 95.7–99.3 % noise-only and 87.7–95.7 % combined.
- **DR-DN 10 mm:**
  - Held-out: 100 % completion in every cell, like DR-D; seed-mean ΔE_FA −0.56 to −1.73 mm, every CI including 0.
  - Sweep: ≥ 98.3 % completion at every noise level in both conditions.
- **PID-RN 3 mm (6, 0.5, 2.0, τd 0.02):**
  - Noise-only sweep: 100 % completion to 10 mm stationary and 17.3 mm transient.
  - Held-out: 100 % in every cell. E_FA is +0.59 mm in the nominal and noise cells, and −8.77 / −9.00 mm (stationary delay / combined) and −4.11 / −4.28 mm (transient), with CIs excluding 0.
- **PID-RN 10 mm (6, 0.5, 1.0, τd 0.02):**
  - Noise-only sweep: 100 % completion to 22.8 mm stationary and 30 mm transient.
  - Combined sweep: 68.3 % at 0.3 mm, rising with σ to 100 % at 10 mm (stationary).
  - Held-out delay and combined cells: 68.3–80.0 % completion (−20 to −31.7 pp vs PID-R, CIs excluding 0).
  - On the selection subset it completed 28/28, including the 12 delay-bearing scenarios, which carried 10 mm noise where combined.

## E5

- **Completion** is 100 % for H = 1 and H = 10, for both arms, in every held-out cell.
- **GS-D H1 − GS-D (H10), seed-mean ΔE_FA:**
  - +0.72 mm in nominal and noise (CI excludes 0);
  - −4.20 / −4.18 mm in stationary delay / combined, and −2.87 / −2.83 mm in transient (CIs exclude 0);
  - all 5 seeds have lower E_FA under stationary delay.
- **DR-D H1 − DR-D (H10), seed-mean ΔE_FA:**
  - −4.44 mm nominal;
  - −9.64 / −8.68 mm in stationary delay / combined, and −6.32 / −6.23 mm in transient;
  - all CIs exclude 0. Seed 53 is the only seed with higher H1 error in any cell: +0.8 mm nominal and +4.7 / +4.4 mm in transient delay / combined.
- **Against the plan's expectation:** §6 expected H = 1 to do worse under delay, because one frame cannot show the commands still in the delay pipe. In these runs, delay-cell E_FA is lower with H = 1 for both arms, and completion is unchanged.
- **Scope:** E5 was evaluated on the held-out 480 only (delay 0.15 s), not on the E2 delay sweep.
