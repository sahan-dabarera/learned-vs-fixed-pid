# Free-text sections for stage5.md (read by report_stage5.py)

## Summary

- **The Ch 8 conclusions hold with ten seeds.** Online gain changes help GS-D but not GS-N: in the four delay cells,
  GS-D's online gains beat the better frozen variant by +13.8 pp (five seeds: +15.7 pp) and GS-N's lose by −5.4 pp
  (five seeds: −5.3 pp).
- **GS-D completion with online gains vs frozen gains:** 92.8 % against 70.4 % (F-clean) and 79.0 % (F-delay). With
  five seeds the figures were 95.7 → 72.5 / 80.0 %.
- **The help comes from half the seeds.** The online gains beat both frozen variants by more than 5 pp in 5 of 10
  GS-D seeds (11, 37, 71, 83, 149). The other five gain almost nothing from changing the gains.
- **Three new GS-D seeds barely schedule Kp.** Seeds 97, 109 and 127 hold Kp at the lower bound (1.0) in more than
  85 % of steps in every cell, so ten-seed edge occupancy is much higher than with five seeds (clean cell:
  Kp target at the lower edge 7.7 → 34.0 %).
- **Mean |e_ct| ranks the clean cell exactly as E_FA does.** PID-N < GS-N < PID-R < GS-D < PID-RN < DR-D < DR-N.
  PID-N's mean |e_ct| is 0.14 mm while its E_FA is 1.16 mm, which confirms that E_FA is limited by the sampling
  floor at the top of the ranking.
- **The trace data is complete.** Path geometry, per-step trajectories and gains for all seven controllers × ten seeds on
  three test paths, including a clear GS-D gain response to a delay step.

## Occupancy

- GS-D, ten seeds, clean cell of the ranged test set: Kp target at the lower edge 34.0 % of steps (five seeds:
  7.7 %), Kd target at the lower edge 19.9 % (unchanged), median Kp 8.1 (five seeds: 8.6).
- The rise comes from seeds 97, 109 and 127. They have Kp at the lower bound in 99 % of clean steps and 85–99 % of
  disturbed steps; seed 83 sits near the upper part of the range instead (clean mean Kp 100.6).
- The direction of the shift is the same as with five seeds: more of GS-D's Kp at the lower edge under delay and
  combined disturbance than on clean runs (seed-pooled stationary: 34.0 % clean → 54.1 % delay → 57.8 % combined).
- Seed 149 has its Kd target at the lower edge in 77–100 % of steps (applied Kd 0 in 51–87 %), as seed 37 does on
  clean and noise runs.
- GS-N, ten seeds: almost no edge occupancy (Kp lower edge 4.6–5.3 %, Kd never at an edge), median Kp rises from
  about 19 on clean runs to 28–29 under stationary delay, as with five seeds.

## Frozen

- GS-D, ten seeds, delay cells: online 92.8 %, F-clean 70.4 %, F-delay 79.0 %; online − best frozen +13.8 pp.
- Seeds with an online advantage above 5 pp: 11 (+7.9), 37 (+47.1), 71 (+21.7), 83 (+34.2), 149 (+25.8). The others
  are within ±2 pp.
- Seeds 97, 109 and 127 give the same completion with frozen gains (95–100 %), consistent with their nearly
  constant Kp.
- GS-N, ten seeds: online 60.2 %, F-clean 65.6 %, F-delay 52.5 %; online − best frozen −5.4 pp. Only seeds 23
  (+9.2) and 127 (+10.8) gain from online changes.

## Cross-track error

- Clean cell (stationary nominal, ranged test set), mean |e_ct| in mm: PID-N 0.14, GS-N 2.50, PID-R 3.74, GS-D
  8.19, PID-RN 15.76, DR-D 19.77, DR-N 26.42. The ranking is identical to the E_FA ranking.
- Seed spread of mean |e_ct| for the learned arms is wide: GS-N 0.15–9.23, GS-D 0.32–29.57, DR-D 10.3–40.6, DR-N
  6.5–139.9 mm. The best GS-N and GS-D seeds come close to PID-N.
- The precision gap at the top is larger in mean |e_ct| than in E_FA (PID-N vs GS-N: 0.14 vs 2.50 mm, against
  1.16 vs 1.91 mm in E_FA). So "PID-N is the most precise" is supported directly, not only up to the sampling
  floor.
- Every cell is now complete (cluster run, all 480 scenarios). Under delay the ranking changes: PID-RN has the lowest
  mean |e_ct| in stationary delay (20.5 mm), ahead of DR-D (29.1) and PID-R (31.1); the thesis needs only the clean cell.

## Traces

- **Gain response to a delay step (gain-over-time figure):** GS-D seed 53, path 10019, transient delay. Mean Kp is 21–24 over
  0–5 s with no delay; the delay steps to 70 ms at 6 s and Kp falls to 1.3–1.7 within about 1 s, then recovers slowly
  to 2–3 by 17 s. Kd stays at 6.0.
- Other seeds respond less or not at all: seed 83 on path 10019 Kp 117 → 34; seeds 97, 109 and 127 stay at Kp 1.0
  throughout.
- The transient delay step on the three traced paths is small (14–70 ms). For a larger change, use
  `traces_dynamic.csv.gz` (D_max up to 0.45 s, delay varies over time): GS-D seed 53 on path 10001 at D_max 0.45 s,
  delay only, has mean Kp 21.5 at its lowest delay and 2.0 at its highest.
- Representative seeds for trajectory figures: GS-D 53, DR-D 37, GS-N 109, DR-N 11.
