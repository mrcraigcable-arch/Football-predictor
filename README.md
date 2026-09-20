# Craig's Football Predictor V21.1 — Full Fixture Ranking

Probability-first Streamlit football predictor.

## What V21.1 changes

- The selected date window controls the full ranking pool.
- Every unfinished supported fixture in that window can be ranked even when current bookmaker odds have not yet been verified.
- The Top 10 is ordered by the model's strongest HOME/AWAY win probability.
- Positive EV and model-vs-market edge are diagnostics only; they do not remove teams.
- The strongest five are simply ranks 1–5, so the user can decide which legs to keep or swap.
- A personalised 5/6-team acca can still use stake + target return, but only fixtures with verified current prices can enter the return calculation.
- LIVE fixtures are included only while a current market is still available and are clearly marked time-sensitive.
- Finished/stale fixtures are silently excluded.
- The embedded V18 evidence audit remains advisory and visible for deeper checking.

## Supported competitions in the current code

- Premier League
- Championship
- Bundesliga
- La Liga
- Serie A
- Ligue 1

## Odds key

Store `ODDS_API_KEY` in Streamlit Community Cloud Secrets. The app loads it automatically; do not hard-code the key into this public repository.
