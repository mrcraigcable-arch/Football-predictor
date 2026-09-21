# Craig's Football Predictor V28 — Hardened Production

V28 is the hardening release built after auditing V27 for failure modes that
could otherwise appear only after deployment.

## Main changes

- **Holdout-audited frozen scorer.** Calibration was fitted on 2023-24 + 2024-25
  and tested on untouched 2025-26 data for the six established model leagues.
  Calibrators that worsened the untouched holdout were not promoted.
- **Strict verification stages.** SCREEN VERIFIED and FULL VERIFIED are separate.
  FINAL requires FULL VERIFIED.
- **Adaptive API-Football quotas.** V28 reads daily/minute rate-limit headers,
  keeps a reserve, handles 429 responses and fails soft to PROVISIONAL.
- **Explicit pagination.** API-Football pages are followed to a safe cap.
- **Expanded-league settlement.** Provider fixture IDs allow competitions such
  as EFL Trophy to settle into the performance ledger.
- **Pre-match-only learning.** LIVE observations cannot contaminate trust or
  model/market learning.
- **Recommendation versioning.** A changed pre-match side supersedes the earlier
  signal instead of double-counting both.
- **Bayesian competition trust.** A 50-match prior dampens small samples;
  adjustment remains capped at ±3 Anchor Score points.
- **Lazy Odds API usage.** Bookmaker calls occur only when a league actually has
  fixtures in the selected window.
- **Provider fixture cross-check.** When a unique provider match is available,
  its status, kickoff and fixture IDs are attached to the dedicated row.
- **Dynamic football season year.** January-June correctly maps to the previous
  season start year.
- **LIVE market safety.** The pre-match model/market blender cannot alter LIVE
  rows that are intentionally using the current market.
- **Safer unreachable-target output.** The app shows both Safest achievable and
  Closest to target rather than silently chasing odds.
- **Pandas hardening.** First-row ledger/history writes and settlement dtypes are
  handled without relying on deprecated pandas coercion behaviour.

## Historical holdout summary

The untouched 2025-26 evaluation contained:
- Premier League: 353 matches
- Championship: 521
- Bundesliga: 294
- La Liga: 365
- Serie A: 344
- Ligue 1: 282
- pooled: 2,159

La Liga and Serie A stayed RAW because their fitted calibration worsened holdout
log loss. The pooled calibrator also stayed unpromoted.

## Secrets

No additional key is needed:

```toml
ODDS_API_KEY = "..."
API_FOOTBALL_KEY = "..."
```

No football model guarantees an outcome.

## Verification

Run the repeatable release contract locally with:

```bash
python -m unittest -v test_v28_production.py
```

The final release report is in `V28_FINAL_VERIFICATION.md`.
