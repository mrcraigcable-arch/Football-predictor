# Craig's Football Predictor V24 — Fast Production

V24 fixes the V23 performance bottleneck without throwing away the deeper
validation and premium-data capability.

## Normal app use — fast path

On an ordinary Today / Saturday / Next 7 Days run, V24:

1. fetches current fixtures and odds
2. loads/fits one cached recency-weighted live model per active league
3. ranks every unfinished fixture
4. shows the Top 10 and Safest Five
5. calculates the target acca only where current prices exist

It does **not** run the expensive multi-model TRAIN/TUNE/FINAL TEST process on
every normal screen refresh.

## Deep validation — on demand

Open **Engine & data connections** and press:

`RUN / REFRESH DEEP VALIDATION`

This runs the full production audit for the active leagues:
- four candidate models
- 70/15/15 chronological TRAIN / TUNE / untouched FINAL TEST
- recency half-life selection
- learned ensemble weights
- probability calibration
- legacy-model comparison
- Top-10 rank historical audit
- Safest-Five historical audit

Successful results are cached in the current Streamlit process/session and then
replace the fast model for subsequent rankings.

## API-Football — second data service

The second service is **API-Football**. It supplies the richer context that the
basic fixture/odds feeds cannot reliably provide:

- provider expected goals (when exposed in fixture statistics)
- verified injuries
- confirmed line-ups
- fixture/team identity

It is deliberately **on-demand** in V24. Connect it in either of two ways:

### Permanent
Add to Streamlit Community Cloud → App → Settings → Secrets:

```toml
ODDS_API_KEY = "your Odds API key"
API_FOOTBALL_KEY = "your API-Football key"
```

### Temporary session
Open **Engine & data connections** inside the app, paste the API-Football key and
press **CONNECT API-FOOTBALL**.

Then press **REFRESH XG / INJURIES / LINE-UPS** whenever you want the premium
context pass. It never blocks the initial Top 10.

## Existing product rules preserved

- no positive-EV ranking gate
- no +4pp edge ranking gate
- future fixtures can rank before current odds are published
- finished/stale matches remain hidden
- current prices are required only for live inclusion and target-return maths
- bookmaker/model blending is learned from settled history, never given an arbitrary weight

No probability model guarantees a football result.
