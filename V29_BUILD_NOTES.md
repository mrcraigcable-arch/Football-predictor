# Craig's Football Predictor V29 — ACCA-FIRST

## Product goal
V29 changes the app from a general prediction dashboard with an accumulator feature into a five-team accumulator decision engine.

The primary question is now:

> Among all currently bettable outright winners, which exact five-team combination has the highest evidence-adjusted chance of all five winning while still meeting my chosen return target?

## Core methodology
1. Analyse the complete selected fixture window.
2. Keep only upcoming HOME/AWAY 90-minute winner selections with an exact matched current price.
3. For each candidate, challenge the raw model probability against:
   - de-margined bookmaker market probability;
   - independent provider prediction when available;
   - model/source disagreement;
   - draw threat;
   - existing analyst evidence score;
   - evidence completeness;
   - stress-test stability;
   - injuries/availability and lineup state;
   - existing adversarial risk count;
   - EFL Trophy/cup/U21 rotation risk before confirmed lineups;
   - large model-v-market contradiction.
4. Convert this into an **adjusted adversarial win probability**. The raw model probability is never used directly as the portfolio objective.
5. Search candidate five-folds and reject combinations below `target return / stake`.
6. Among target-reaching five-folds, maximise the product of the five adjusted win probabilities. Anchor quality and avoiding unnecessary excess odds are tie-breakers.
7. If the target is unreachable, show both the safest five and the closest-to-target five rather than silently adding extra legs or chasing weak outsiders.
8. Return the weakest leg and per-leg challenge audit so the final pre-kickoff recheck has a clear priority.

## Fixed production contract
- Exactly **5 teams**.
- **Outright 90-minute winners only**.
- Stake and return target remain adjustable.
- Default stake remains £10.
- Default target is £300.
- No 6-team escape hatch.
- No live, postponed, cancelled or abandoned fixtures.
- PROVISIONAL selections can be used for planning but are explicitly penalised; FULL VERIFIED / FINAL evidence improves their adjusted probability.
- Cup/academy matches before confirmed XIs receive an additional rotation penalty.

## Files
- `v29_acca_engine.py` — new independently testable selection/portfolio engine.
- `test_v29_acca_engine.py` — V29 unit suite.
- `apply_v29_patch.py` — fail-closed patcher for the current V28 `app.py`.

## Verification completed in this build package
- Python compilation: PASS.
- V29 unit suite: **7/7 PASS**.

The existing V28 API, fixture discovery, odds matching, provider verification, quota, settlement and historical-learning plumbing are deliberately preserved by the patch.
