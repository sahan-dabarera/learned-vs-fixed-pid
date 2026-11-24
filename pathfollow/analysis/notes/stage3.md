# Free-text sections for stage3.md (read by report_stage3.py)

## 0 Deviations

1. **Rate limit held at the stage-2 values** (147.5, 0.75, 1.75 per s) for the wide box, through the new
   `train.py --gain-rate-limit`. The plan's "half the gain range per second" rule would have raised the Kd
   limit by 71 % (1.75 to 3.0 per s), a second change mixed into the box test. Agreed with the author
   before the run.
2. **G0.2 edge occupancy** is measured in action coordinates: a target counts as at an edge when its action
   is within 2 % of the end of that map segment. The fraction of steps exactly on a bound is reported too.
   The plan's "2 % of the gain range" puts every Kp below 10.9 at the floor on the stage-2 box.
3. **Code changes not named in the plan:**
   - `evaluate.py`, `frozen_gains.py` and the new `gain_occupancy.py` read each
     checkpoint's box and rate limit from its training metadata. Without this, the wide-box arms would have
     been evaluated on the stage-2 box.
   - The scheduler episode trace also records target gains.
   - The re-run of stage-2 GS-D, GS-N and DR-D on the ranged held-out set reproduces 7,200 of 7,200 stage-2 rows
     bit for bit. PID rows were not re-run, because they do not use the changed code path.
4. **H1 − H10 on the chosen box** compares GS-D-W (300k) with GS-D-W H10 (300k). The H10 arm was not trained
   at 2M (plan §5), so the comparison is budget-matched rather than against the chosen GS-D-W 2M.
5. **Frozen gains (§8.1)** were computed for GS-D-W and GS-N-W, as the plan lists, not for the 2M arms.
6. **Extra figure** `g2_learning_curves_2m`. The G4 figures draw the stage-2 GS-D as a grey reference.
7. **Shared training start:** each 2M run and its 300k counterpart share the first 300k steps exactly
   (same seed and settings, deterministic training). In all 10 pairs, the best checkpoint up to 300k has
   the same validation score as the 300k run's kept checkpoint (§3).

## G0

- **Stage-2 box:** Kp [5, 300], Ki [0, 1.5], Kd [2.5, 6], around g0 = (26, 0.5, 4.4).
  - PID-RN (2.0, 0.5, 0.5) is outside the stage-2 box on Kp and Kd; PID-R (8, 0.5, 4.4) is inside.
  - The wide box contains both.
- **Stage-2 GS-D, ranged held-out:** the applied Kp is on the floor (Kp = 5) in 49–53 % of policy steps in
  every condition, including nominal. The median applied Kp is 5.00 in every condition.
  - The Kd target is at its lower edge in 27–40 % of steps.
  - The Kp target is at the upper edge in ≤ 1.8 % of steps.
  - On `validation_ranged.json` the shares are similar (47–52 % on the Kp floor).
- **Stage-2 GS-N:** never at the Kp floor (0 %). Median applied Kp is 22.9–26.9; the Kd target is at its lower
  edge in 4–12 % of steps.
- **Learning curves (validation):**
  - Stage-2 GS-D's seed-mean completion moves between 69 % and 79 % over 300k steps. Its kept checkpoints are at
    100k–300k.
  - GS-N's seed-mean completion falls from 80 % at 25k to 57 % at 300k. Four of its five kept checkpoints are
    at ≤ 100k.
  - DR-D and DR-N kept checkpoints between 200k and 1.7M.

## G2

- **2M runs vs their 300k counterparts:** a longer run never selects a worse checkpoint, because it contains
  the 300k run.
  - GS-D 2M kept a later checkpoint than GS-D in 3 of 5 seeds (400k, 650k, 1.7M).
  - GS-D-W 2M did so in 3 of 5 seeds (350k, 475k, 1.35M).
  - The remaining seeds kept the same checkpoint as the 300k run.
- **Seed-mean validation completion** of the 2M curves levels off after about 300k steps: GS-D 2M at about
  80–85 %, GS-D-W 2M at about 75–85 %, falling slowly after a peak near 300–500k.
- **Validation, seed-mean completion / E_FA:**
  - GS-D 2M 91.7 % / 19.2 mm, against GS-D 84.2 % / 25.5 mm;
  - GS-D-W 2M 98.3 % / 20.7 mm, against GS-D-W 95.8 % / 18.1 mm.

## G4 held-out

- **Decision (validation only):** wide box chosen (95.8 % vs 84.2 %); 2M chosen on the wide box (+2.5 pp).
  Main scheduler: GS-D-W 2M.
- **Against stage-2 GS-D (ranged held-out, matched seeds):**
  - Every ΔE_FA CI of GS-D-W, GS-D 2M and GS-D-W 2M includes 0, except GS-D 2M transient noise
    (−0.75 mm, CI [−1.89, −0.02]).
  - Completion: GS-D-W 2M is +11.7 pp in stationary combined (CI [+2.0, +21.7]). All other version-vs-GS-D
    completion CIs include 0.
- **GS-D-W 2M vs PID-RN:**
  - completion −10.0 pp (stationary noise), −5.7 pp (stationary combined) and −4.0 pp (transient
    combined), CIs exclude 0;
  - ΔE_FA +21.4 (stationary delay), +8.7 (stationary noise), +11.2 / +11.5 mm (combined), CIs exclude 0;
  - nominal ΔE_FA −4.87 mm, CI excludes 0.
- **GS-D-W 2M vs DR-D:**
  - completion −10.0 pp (stationary noise), −6.7 pp (stationary combined) and −4.0 pp (transient
    combined), CIs exclude 0;
  - ΔE_FA +20.6 mm in stationary delay, CI excludes 0;
  - nominal ΔE_FA −6.33 mm.
- **GS-D-W 2M vs PID-R:** completion +11.0 pp (stationary combined) and +5.3 pp (transient noise), CIs exclude
  0; no ΔE_FA CI excludes 0.
- **GS-N-W − GS-N:** every CI includes 0.
- **GS-D-W − GS-D-W H10:** ΔE_FA −37.9 (stationary combined), −21.6 (transient delay) and −28.3 mm (transient
  combined), CIs exclude 0. Completion is +18.7 pp in transient combined, CI excludes 0.
  - In stage 2 (old box), every GS-D − GS-D H10 CI included 0.
- **Fixed held-out (§5.5):** see table. No significance tests.

## G4 delay

- **Constant delay, breaking points (both modes pooled):** GS-D-W 2M 0.1 s (E_FA rule), GS-D-W 0.1 s, GS-D 2M
  0.15 s, GS-D 0.15 s. The PID and DR values are unchanged from stage 2 (PID-RN 0.3 s, DR-D 0.3 s).
- **Completion at 0.3 s, stationary / transient:**
  - GS-D-W 2M 64.7 / 85.7 %;
  - GS-D-W 56.0 / 61.3 %;
  - GS-D 37.0 / 61.0 %;
  - GS-D 2M 27.3 / 70.3 %;
  - PID-RN 68.3 / 66.7 %;
  - DR-D 98.7 / 100 %.
- **At 0.4 s:** GS-D-W 2M completes 10.0 / 15.7 %, against DR-D at 23.3 / 62.0 %.
- **Dynamic delay at D_max = 0.45 s, delay / combined completion:**
  - GS-D-W 2M 82.3 / 81.7 %;
  - GS-D 66.3 / 53.0 %; GS-D-W 77.3 / 71.7 %; GS-D 2M 77.3 / 66.7 %; GS-D-W H10 46.0 / 44.7 %;
  - PID-RN 66.7 / 55.0 %; PID-R 95.0 / 73.3 %; DR-D 100 / 100 %.
- **Dynamic delay at D_max = 0.3 s:** GS-D-W 2M completes 97.7 / 100 %.

## G4 noise

- **Noise-only breaking points (both modes pooled):** GS-D-W 2M 3 mm (completion 98.3 %), GS-D-W 3 mm,
  GS-D 3 mm. PID-RN 22.8 mm and DR-D 40.5 mm, as in stage 2.
- **At 10 mm stationary noise, completion:**
  - GS-D-W 2M 15.7 %, GS-D-W 20.0 %, GS-D 1.7 %;
  - PID-RN 100 %, DR-D 100 %.
- **Mechanism for GS-D-W 2M, stationary noise:**
  - at 3 mm: steering TV/T 375 1/s, wheel saturation 66 %;
  - at 10 mm: 496 1/s and 96 %.
  - PID-RN at 3 mm: 80 1/s and 2 % (stage 2).

## G4 frozen

- **GS-D-W, delay-bearing cells:** mean completion is 94.4 % with dynamic gains, 68.7 % with F-clean and
  85.2 % with F-delay.
  - Flagged seeds (dynamic beats the best frozen by more than 5 pp): 11 (+7.9), 37 (+6.2) and 71 (+29.6 pp).
  - In stage 2 (old box), GS-D's frozen gains matched or beat dynamic on average (88.2 vs 89.4 / 91.0 %), and
    one seed (71) was flagged.
- **GS-N-W:** mean 61.3 % dynamic vs 66.7 / 52.9 % frozen. Seed 23 is flagged (+9.2 pp).
- **Wide-box gain occupancy (GS-D-W, ranged held-out):**
  - The Kp target is at the new lower edge (Kp ≤ 1.5) in 17.7–33.5 % of steps in the disturbed conditions,
    and 0.4 % in nominal.
  - The median applied Kp is 3.75–8.67 in the disturbed conditions and 12.07 in nominal; the median Kd is
    2.17–3.24.
  - The applied Kp is on the floor (Kp = 1) in 15.1–27.6 % of steps in the disturbed conditions.

## G5

- **Transient event metrics (ranged held-out):** PID-RN has the lowest post-event peak path distance in transient
  delay (40.2 mm) and combined (44.4 mm) among the main arms.
- **DR-D − PID-RN:**
  - post-event peak +14.1 mm (delay) and +20.7 mm (combined);
  - excess IAE +11.1 / +16.0 mm·s;
  - all four CIs exclude 0. The noise-cell CIs include 0.
- **GS-D-W 2M − PID-RN:**
  - post-event peak −15.2 mm in transient noise, CI [−21.6, −9.3];
  - excess IAE +8.8 mm·s (delay) and +12.9 mm·s (combined), CIs exclude 0;
  - the peak CIs for delay and combined include 0.
- **Onset and severity:** every transient record steps at 6 s from a clean start. The transient draws have a
  median delay of 0.143 s and a median noise of 1.07 mm.
- **Run time per decision (EPYC 9334, one thread):**
  - fixed PIDs 0.6–1.0 µs (constant action only);
  - learned policies 278–343 µs mean, with per-episode 99th percentiles of about 310 µs (GS-N-W: 392 µs);
  - the decision period is 20 ms.

## G12

- **Gain occupancy, GS-D-W 2M, ranged held-out:**
  - The Kp target is at the lower edge of the wide range (Kp ≤ 1.5) in 23.7–44.6 % of steps in the disturbed
    conditions, and 7.7 % in nominal.
  - The applied Kp is on the floor (Kp = 1) in 20.8–38.0 % of disturbed-condition steps, and 6.0 % in nominal.
  - At 300k (GS-D-W, §8.2) the floor shares were 15.1–27.6 % and 0.1 %.
- **Medians (ranged held-out):** applied Kp 3.75–6.74 in the disturbed conditions and 8.64 in nominal; applied
  Kd 2.63–3.18 (2.77 in nominal). The Kp target never reaches the upper edge (≤ 0.05 % of steps).
- **Frozen gains, GS-D-W 2M, delay-bearing cells:** mean completion is 95.7 % with dynamic gains, 72.5 % with
  F-clean and 80.0 % with F-delay.
  - Flagged seeds: 11 (+7.9 pp), 37 (+47.1 pp) and 71 (+21.7 pp).
  - Seeds 11 and 23 kept the same checkpoint as their 300k runs, so their frozen gains and completions equal
    §8.1.
- **Machine:** all §12 numbers come from job 3347265 on cpunode04 (AMD EPYC 9334). The replayed dynamic rows match
  the G4 rows bit for bit.
