# Free-text sections for POSTHOC_RESULTS_2.md (read by posthoc_report2.py)

## 0 Deviations

1. **Design revised with the author (2026-11-15/16) before any run:**
   - The disturbed arms GS-D and DR-D train on a ranged sampler: delay uniform in [0, 0.3] s and noise
     log-uniform in [0.1, 10] mm, with no point mass at the declared severity.
   - Checkpoints are selected on `validation_ranged.json`.
   - The main table uses a new ranged held-out set (`held_out_ranged.json`). The fixed held-out set is kept
     as a comparison (§3.4).
   - F7 (noise-trained H = 1 arms) is therefore part of the main design (§8).
2. **H = 10 ablation of the disturbed arms was retrained** with the same ranged sampler (`GS-D H10`, `DR-D H10`)
   instead of reusing the sealed H = 10 checkpoints. Otherwise an H1 − H10 difference would also contain the
   sampler change.
   - The nominal H = 10 arms are the sealed nominal checkpoints; the nominal sampler is unchanged.
   - The sealed disturbed checkpoints and the E5 H = 1 checkpoints (both old sampler) are reported as references.
3. **PID-RN** was retuned on the ranged validation subset by the E4 grid procedure. It took three rounds:
   - round 1 selected Kp 4.0 at the lower Kp edge;
   - round 2 (Kp down to 1) selected Kd 0.5 and τd 0.01, both at their lower edges;
   - round 3 (Kd down to 0, τd down to 0.005) selected the same point. Final: (2.0, 0.5, 0.5, τd 0.01), §0.2.
   - The round-1 PID-RN rows inside the job-3225586 directories are not used.
4. **Models** for all new arms are in `artifacts/models/posthoc/plan2/`, not `f1_history1_nominal/`.
5. **F3:**
   - Every level (0–0.8 s) was run for every arm; no early stop.
   - The existing arms were re-run rather than reusing E2 rows. Their 2,640 rows at the sealed level
     (0.15 s) reproduce the sealed run bit for bit (§0).
6. **F4:**
   - Built as a scenario manifest with multiple delay events; `env.py` was not changed.
   - A combined variant (dynamic noise on its own schedule) was added at the author's request.
   - Zero-command counts are computed from the schedules (§5).
7. **F5:** PID-N, PID-R, GS-N H10 and DR-N H10 rows are the E1 rows from results 1 (same scenarios and
   controller instances), not re-runs.
8. **F6:** frozen gains come from the sealed `validation.json` (as in E3 and the plan), and are evaluated on
   the ranged held-out set only.
9. **F0:** read from the training logs; no replay. The ranged arms were validated on `validation_ranged.json`,
   all others on `validation.json`, so validation numbers are comparable only within those groups.
10. **Identical cells:** in both held-out sets the transient-nominal scenarios carry no event, so they
    reproduce the stationary-nominal cells exactly. The two nominal columns are identical by construction.

## F0

- **E_FA, H = 1 vs H = 10 on validation:** in all six pairs, validation and held-out favour the same history
  setting.
  - H = 1 has the lower E_FA for GS-D, DR-D, DR-N, GS-D old-sampler and DR-D old-sampler.
  - H = 10 has the lower E_FA for GS-N: 24.7 vs 28.6 mm on validation, 23.7 vs 27.0 mm on held-out.
- **Completion:**
  - GS-N and GS-D differ by under 1 pp on validation, in H = 10's favour, and by 0.5 / 3.0 pp on held-out,
    in H = 1's favour.
  - DR-D: 100 % (H = 1) vs 97.5 % (H = 10) on validation. The gap comes from DR-D H10 seed 37: 87.5 %,
    205 mm.
- **GS-D (ranged) has seed-mean validation completion of 84.2 %** (seeds 75.0–100 %). Its held-out completion on
  the same cell types is 94.0 %.
- **One-line answer:** validation points the same way as the held-out result for every H1/H10 pair on E_FA.
  On completion, GS-N and GS-D differ in direction by ≤ 3 pp.

## F2

- **DR-D completion:** 99.3–100 % in every ranged held-out cell.
  - Against PID-RN, all completion CIs include 0, except stationary and transient delay: +1.7 pp,
    CI [+0.3, +3.3].
  - Against PID-RN, ΔE_FA CIs include 0 in every cell except transient combined: +4.52 mm, CI [+1.85, +8.80].
- **DR-D − PID-R:**
  - completion is higher in stationary delay / noise / combined: +6.7 / +15.0 / +17.7 pp, CIs exclude 0;
  - ΔE_FA is +6.72 mm in nominal and −4.96 mm in stationary delay, CIs exclude 0.
- **GS-D completion:** 81.0–100 %.
  - GS-D − PID-RN completion: −9.7 (stationary delay), −16.0 (stationary noise), −17.3 (stationary combined)
    and −12.0 pp (transient combined), CIs exclude 0.
  - GS-D − DR-D completion: −11.3, −16.0, −18.3, −4.7 (transient noise) and −12.0 pp, CIs exclude 0.
  - In nominal, GS-D has the lower E_FA: −4.88 mm vs PID-RN, −6.34 mm vs DR-D, CIs exclude 0.
- **All 62 GS-D failures in the noise cells** have noise above 3 mm (3.1–9.2 mm), and all are time-limit
  failures.
- **D − N:**
  - GS-D − GS-N completion is +21.0 to +33.3 pp in the four delay-bearing cells, CIs exclude 0.
  - DR-D − DR-N ΔE_FA CIs include 0 in every cell.
- **H1 − H10 (§3.3):**
  - For GS-N, DR-N and GS-D, every ΔE_FA and Δcompletion CI includes 0. GS-D's seed-mean ΔE_FA is −9.9 to
    −20.1 mm in the delay-bearing cells.
  - For DR-D, ΔE_FA is −77.2 to −107.9 mm in the delay-bearing cells, with CIs excluding 0. In nominal and noise
    it is −58 mm, with CIs including 0.
  - The largest per-seed DR-D difference is seed 37 (−280 to −368 mm). Seed 23 is positive in every
    stationary cell (+1.4 to +10.0 mm).
- **Fixed held-out (§3.4: 0.15 s / 0.3 mm), not tested for significance:**
  - PID-RN, GS-D, DR-D and all four old-sampler disturbed arms complete 100 % of every cell.
  - GS-D H10 (97.7–100 %) and DR-D H10 (85.0–96.7 %) do not.
  - Stationary-delay E_FA: GS-D 11.0, DR-D 13.5, PID-RN 8.08 mm.
  - The completion differences in §3.1 appear only with the ranged severities.

## F3

- **Training delay distribution:** see the text above the tables. Ranged arms train on 0–0.3 s, so levels
  0.4–0.8 s lie outside every training range.
- **Breaking points (both modes pooled):**
  - PID-N 0.03 s;
  - GS-N 0.06 s;
  - PID-R, GS-N H10, DR-N H10 and GS-D H10 0.1 s;
  - GS-D 0.15 s (E_FA rule);
  - DR-N 0.2 s;
  - PID-RN 0.3 s (completion 67.5 %);
  - DR-D 0.3 s (completion 99.3 %).
  - DR-D H10 is below 100 % already at 0 s (95.7 %, seed 37).
- **At 0.3 s, stationary / transient completion:**
  - DR-D 98.7 / 100 %;
  - DR-N 86.7 / 90.0 %;
  - PID-RN 68.3 / 66.7 %;
  - GS-D 37.0 / 61.0 %;
  - PID-R 33.3 / 88.3 %;
  - GS-N 1.7 / 3.3 %.
- **At 0.4 s,** completion is 0 % for every PID and scheduler arm in stationary mode; GS-D reaches 0.7 % in
  transient. DR-D is at 23.3 / 62.0 % and DR-N at 16.0 / 33.3 %.
- **At 0.8 s,** every main arm completes 0 % in both modes.

## F4

- **Zero-command intervals:**
  - Over the schedules up to the time limit, a delay change starts from an empty queue (0 steps) 6, 1 and 4
    times for D_max = 0.15, 0.3 and 0.45 s. Before the median episode end: 2, 0 and 1.
  - One scenario (D_max 0.15 s) starts at 0 steps.
  - Each schedule has 11–34 changes up to the time limit (mean 20.2).
- **D_max = 0.15 s:** every main arm except PID-N and GS-N completes ≥ 99.7 % in both variants. PID-N completes
  0 % at every range.
- **D_max = 0.3 s, delay / combined completion:**
  - DR-D, DR-N and PID-RN 100 / 100 %;
  - PID-R 100 / 98.3 %;
  - GS-D 96.7 / 91.3 %;
  - GS-N 62.0 / 55.7 %.
- **D_max = 0.45 s, delay / combined completion:**
  - DR-D 100 / 100 % (E_FA 36.4 / 37.9 mm);
  - DR-N 97.0 / 96.7 %;
  - PID-R 95.0 / 73.3 %;
  - PID-RN 66.7 / 55.0 %;
  - GS-D 66.3 / 53.0 %;
  - GS-N 7.0 / 4.7 %.
  - DR-D − PID-RN ΔE_FA is −73.6 / −116.0 mm, and DR-D − PID-R is −13.5 / −31.8 mm; all CIs exclude 0.
- **D_max = 0.3 s:**
  - DR-D − PID-RN ΔE_FA is +5.1 / +4.2 mm, CIs exclude 0.
  - GS-D − PID-RN ΔE_FA is +12.9 / +28.0 mm, CIs exclude 0, with completion −3.3 / −8.7 pp.
- **H1 − H10:**
  - GS-D: every ΔE_FA and Δcompletion CI includes 0.
  - DR-D: ΔE_FA CIs exclude 0 at 0.3 and 0.45 s (−101 to −127 mm). The completion lower bounds are 0.0.

## F5

- **Breaking points, noise only, both modes pooled:**
  - PID-R 3 mm (E_FA rule), GS-N 3 mm, GS-D 3 mm (completion 93.0 %), GS-D H10 3 mm (98.2 %);
  - PID-N 4.1 mm;
  - DR-N 17.3 mm;
  - PID-RN 22.8 mm;
  - DR-D 40.5 mm.
- **GS-D trains on noise up to 10 mm,** but completes 86.0 % (stationary) at 3 mm and 2 % at 10 mm.
  - Its steering TV/T at 3 mm is 422 1/s, with 78 % wheel saturation.
  - DR-D at 3 mm: 7 1/s and 1 % saturation. PID-RN: 80 1/s and 2 %.
- **At 10 mm stationary noise:**
  - PID-RN completes 100 % (TV/T 262 1/s, saturation 36 %);
  - DR-D and DR-N complete 100 % (saturation 11 % and 6 %);
  - PID-R completes 0 %; GS-D 2 %; GS-N 4 %.
- **Noise + 0.15 s delay:**
  - GS-N is below 100 % already at 0.3 mm (pooled 76.2 %);
  - PID-N is at 0 % at every level;
  - DR-D breaks at 54.8 mm (E_FA rule).

## F6

- **GS-D (H = 1, ranged), delay-bearing cells:** mean completion is 88.2 % with dynamic gains, 89.4 % with
  F-clean and 91.0 % with F-delay.
  - Seed 71 is flagged: dynamic 95.8 % vs F-delay 82.5 % and F-clean 73.8 %.
  - Seed 53 goes the other way: dynamic 69.2 % vs frozen 96.7 / 96.2 %.
- **GS-N (H = 1):** mean 59.6 % dynamic vs 55.9 / 54.9 % frozen. Seed 23 is flagged: dynamic 44.6 % vs F-delay
  33.8 % and F-clean 15.8 %. Its F-clean Kp is 207.3, against 18–39 for the other seeds.
- **Frozen GS-D gains:** F-clean and F-delay Kp differ by 0.1–2.3 between the two operating points of a seed.
  Kp ranges 5.1–11.8 across seeds.
- **Compared with results 1 (E3, H = 10, old sampler):** there the flagged GS-D seed was 23. Here the flagged
  GS-D seed is 71, and seed 23 loses nothing when frozen.
