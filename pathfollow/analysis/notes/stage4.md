# Free-text sections for stage4.md (read by report_stage4.py)

## Summary

- **The run is clean.** 20 of 20 training runs finished, on the same node type and the same commit line as stage 3.
  The parity gate shows the same pre-existing drift notes as stage-3 job 3281395. All five evaluation sets are
  complete.
- **No conclusion reverses.** Of 1,012 paired comparisons, 882 keep their verdict. 90 become clear and 40 stop
  being clear. No comparison that is clear in both changes its sign (§3).
- **Most changes tighten an existing direction.** The ten-seed mean usually has the same sign as the five-seed
  mean, and the CI narrows (for example DR-D − PID-RN at 0.45 s dynamic delay: [+28.0, +38.7] → [+29.7, +37.2] pp).
- **GS-D is a little weaker with ten seeds.** The new seeds have a higher clean E_FA (6.0 mm against 2.7 mm on the
  ranged held-out set) and seed 149 loses half its runs under stationary delay. Its clear completion advantage over
  PID-R in stationary combined (+11.0 pp) is no longer clear (+8.0 pp, CI [−2.0, +15.3]).
- **DR-D is unchanged.** Its ten-seed numbers stay within about 1 mm and 1 pp of the five-seed numbers in every
  held-out cell, and no DR-D seed falls below 90 % completion in any disturbed cell (§8).
- **DR-N is weaker with ten seeds.** Seed 127 is imprecise even on clean runs (94 mm stationary nominal E_FA on the
  ranged set) and seed 83 loses 40 % of delay runs. Its E_FA advantage over GS-N under delay is no longer clear.
- **The nominal results hold.** GS-N and DR-N beat PID-N under delay by about as much as before, with narrower CIs.

## Validation

- The new GS-D and GS-N seeds validate lower than the first five: 93.3 % against 98.3 % (GS-D) and 87.5 % against
  92.5 % (GS-N) seed-mean completion. DR-D and DR-N reach 100 % in all ten seeds.
- GS-D seeds 109 and 149 validate at 83.3 %. Seed 149 kept its checkpoint at 1.85M steps, the latest of any GS-D
  run.
- GS-N keeps a checkpoint at 25k or 50k steps in all five new seeds; the first five kept theirs at 25k–200k.

## Changes

- **Ranged held-out:** 10 comparisons become clear, 5 stop being clear. The lost ones are GS-D − PID-R stationary
  combined completion, DR-D − PID-RN stationary delay completion (+1.7 → +1.0 pp), and three GS-N − DR-N E_FA
  cells.
- **Fixed held-out:** 13 become clear, none are lost. GS-D is now clearly less precise than PID-R on clean runs
  (+2.1 mm) and under transient delay (+12.1 mm); DR-N now clearly finishes more delay runs than GS-N.
- **Delay and dynamic delay:** 28 become clear, 12 are lost. Lost: GS-D − PID-RN completion at 0.3 s transient
  delay and at 0.45 s dynamic delay (+15.7 → +11.2 pp), GS-D − PID-R completion at 0.3 s dynamic combined, and the
  GS-N − DR-N E_FA differences at large delay.
- **Noise:** 39 become clear, 23 are lost. Lost: the small-noise E_FA advantage of GS-D over PID-RN and DR-D
  (about 2–5 mm at 1–2.3 mm noise) and GS-N − DR-N E_FA differences.

## Ranged held-out

- Seed-pooled GS-D, five → ten seeds: stationary nominal 2.68 → 4.37 mm; stationary combined 92.7 % / 28.0 mm →
  89.7 % / 38.2 mm; transient combined 96.0 % / 23.5 mm → 93.3 % / 29.3 mm.
- GS-D − PID-RN stationary nominal E_FA stays clearly negative (−4.87 → −3.18 mm, CI [−5.20, −0.54]); nine of ten
  seeds have a lower clean E_FA than PID-RN (seed 97: 14.0 mm).
- GS-D − PID-R completion: stationary noise becomes clear (+8.5 pp, CI [+3.5, +13.3]); transient noise stays
  clear (+5.5 pp); stationary combined is no longer clear (+8.0 pp, CI [−2.0, +15.3]).
- GS-D − PID-RN and GS-D − DR-D: every completion difference in the disturbed cells is negative with ten seeds, and
  the stationary combined E_FA difference to DR-D becomes clear (+17.0 mm).
- DR-D − PID-RN transient combined E_FA stays clear (+4.86 mm, CI [+2.90, +7.13]).

## Fixed held-out

- GS-D's ten-seed pooled E_FA roughly doubles in the delay and combined cells (stationary combined 14.5 → 27.9 mm),
  driven by seed 149.
- DR-D is unchanged (stationary combined 13.5 → 13.4 mm).

## Delay

- Constant delay: GS-D − PID-R completion at 0.3 s stationary becomes clear (+29.2 pp, CI [+6.8, +49.7]); GS-D −
  PID-RN E_FA at 0.4 s becomes clear (−109 mm stationary, −96 mm transient). At 0.1–0.2 s GS-D becomes clearly
  less precise than PID-R and DR-D.
- DR-D − PID-RN: clear at large delay in both modes (completion +13.7 pp at 0.6 s transient; E_FA −168 mm at 0.4 s
  stationary).
- Dynamic delay, D_max = 0.45 s: DR-D − PID-RN is unchanged (+33.3 / +45.0 pp). GS-D − PID-RN combined stays clear
  (+23.7 pp, −58.5 mm); GS-D − PID-RN delay-only completion is no longer clear (+11.2 pp, CI [−7.2, +25.2]).

## Noise

- GS-D at 10 mm noise finishes more runs with ten seeds (stationary noise 16 → 39 %), still far below DR-D (100 %)
  and PID-RN (100 %).
- GS-D − PID-RN in the combined variant becomes clearly worse in E_FA at small noise (about +11 to +19 mm, 0.3–3 mm
  noise).
- DR-D − PID-RN becomes clearly about 3 mm worse in E_FA at small noise (transient combined, 0.3–3 mm); at 22.8 mm
  transient noise DR-D becomes clearly better (−4.6 mm).

## Reliability

- Seeds with a disturbed cell below 90 % completion: GS-D 3 of the first five and 2 of the new five (37, 53, 71;
  109, 149); GS-N all ten; DR-N 1 + 1 (37; 83); DR-D none.
- The share of weak seeds is about the same in both halves, so the new seeds do not look unusual; ten seeds give a
  better estimate of how often a learned scheduler fails.

## Thesis

Interpretation, for the author. Line numbers refer to `Thesis/chapters/` in the 2026-11-20 supervisor draft.

**Method text**

- `04_controllers.tex:784,828`, `05_methodology.tex:115,296,321,607`, `03_background.tex:555,649`: "five seeds" →
  ten seeds (11, 23, 37, 53, 71, 83, 97, 109, 127, 149). Add one sentence: the arm choices of stage 3 were made with
  seeds 11–71 on validation data; seeds 83–149 were added afterwards with every setting frozen.
- `05_methodology.tex:538,591`: the small-sample limitation becomes milder. Ten seeds is still small, so keep the
  limitation but update it.
- Every caption with "pooled over five seeds (300 runs per cell)" → ten seeds, 600 runs per cell.

**Claims that change**

- `06_main_comparison.tex:108` — "GS-D is clearly better than PID-R only in completion, by 11.0 points" (stationary
  combined): no longer clear (+8.0, CI [−2.0, +15.3]). Clear with ten seeds: stationary noise (+8.5) and transient
  noise (+5.5).
- `06_main_comparison.tex:52` — GS-D − PID-RN clean E_FA −4.87 mm, "all five seeds negative": now −3.18 mm, still
  clear, nine of ten seeds negative.
- `07_stress_tests.tex:122,147` — GS-D − PID-RN at 0.45 s dynamic delay: delay-only completion +15.7 is no longer
  clear (+11.2, CI [−7.2, +25.2]); the combined +26.7 pp becomes +23.7 pp, still clear; ΔE_FA −65.7 becomes
  −58.5 mm, still clear. "four of five seeds" / "all five seeds" need recounting.
- `06_main_comparison.tex:335–391` — the per-seed paragraph names seeds 23, 37 and 71 as the outliers. With ten
  seeds the notable ones are GS-D seed 149 (48 % stationary delay completion), DR-N seeds 83 (delay failures) and
  127 (imprecise on clean runs), GS-D seed 97 (14 mm clean E_FA). Ranges in the seed-range table need recomputing.

**Claims that hold or get stronger**

- DR-D as the most robust learned controller: unchanged, and now clearly ahead of PID-RN at large delay.
- GS-N and DR-N against PID-N under delay: unchanged, narrower CIs.
- GS-D's main advantage at large delay (0.3–0.4 s) is clearer with ten seeds.

**Not re-run with ten seeds**

- Gain occupancy and frozen gains (Ch. 8, `08_diagnosis.tex`), event metrics, learning curves and cost tables
  still use five seeds. Either keep them as five-seed diagnostics and say so, or run a short follow-up job for the
  new seeds (replays only, about 30 minutes).
- All thesis figures still show five seeds and need regenerating.
