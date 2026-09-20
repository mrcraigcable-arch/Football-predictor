# Craig's Football Predictor V23 — Production Candidate

V23 is the production-oriented continuation of V21/V22. It keeps full-fixture,
probability-first ranking and upgrades both statistical validation and data plumbing.

## Active core improvements

- Four-model league-specific ensemble
- 70/15/15 chronological TRAIN / TUNE / FINAL TEST architecture
- Final test is untouched by model-weight, time-decay and calibration selection
- Per-league recency half-life selection
- Empirical-Bayes style small-sample shrinkage
- Opponent-adjusted performance-versus-expectation
- Venue form, rest and 14-day congestion
- Temperature probability calibration
- Automatic fallback to the previous conservative engine when V23 does not improve final-test log loss
- Historical Top-10 rank strike-rate audit
- Historical Safest-Five product audit on matchdays with 5+ fixtures
- Full fixture ranking: missing odds never removes a future fixture
- No EV or +4pp edge ranking gates

## Bookmaker consensus as a predictive input

V23 does not hard-code a market weight. The existing validation ledger is used
to learn a binary selected-team blender from model probability + de-margined
market probability. It requires at least 50 settled observations and is promoted
only if a chronological holdout beats model-only log loss.

## Optional external enrichment

Set `API_FOOTBALL_KEY` in Streamlit Community Cloud Secrets to enable the
API-Football connector. For the strongest current candidates V23 can:

- verify fixture identity
- retrieve verified injuries
- retrieve confirmed line-ups when published
- calculate rolling **provider expected goals (xG/xGA)** from completed fixture statistics
- apply a bounded xG/context second-stage blend

If the provider is missing, ambiguous or does not supply xG, the base model is
left untouched. The app never substitutes a goals-rate proxy and calls it true xG.

`ODDS_API_KEY` remains the current-price source. Current odds are used for
stake/target return calculations; they do not gate future fixtures from Top 10.

## Supported domestic competitions

- Premier League
- Championship
- Bundesliga
- La Liga
- Serie A
- Ligue 1

## Secrets

```toml
ODDS_API_KEY = "..."
API_FOOTBALL_KEY = "..."   # optional premium enrichment
```

No model can guarantee a football result. V23 is designed to make probability
estimation and validation more rigorous, not to claim certainty.
