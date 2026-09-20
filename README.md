# Craig's Football Predictor V25 — Analyst Engine

V25 is built around the reasoning process used to construct a five-team football
acca instead of treating a single machine-learning probability as the whole answer.

## Outputs

**Top 10 strongest teams** stays probability-first across every unfinished fixture
in the selected date range.

**Five-Team Anchors** is the second-stage analytical shortlist. It challenges the
probability with:
- recent and home/away form
- opponent-adjusted performance
- scoring/conceding profile
- provider xG when genuinely available
- injuries and suspensions
- importance of the missing player
- confirmed starting XI / bench
- important normal starters omitted or rotated
- all-competition rest and congestion
- API-Football's independent forecast
- bookmaker market agreement/disagreement
- draw threat
- an explicit adversarial "what could make this lose?" check

`Anchor score` is not presented as a win probability. The model's calibrated win
probability remains visible separately.

## Availability logic

V25 does not count injuries equally.

For unavailable players with usable provider statistics, current-season importance
is estimated from minutes share, starts, rating, and position-adjusted goal/assist
contribution. Players are labelled KEY / IMPORTANT / ROTATION / DEPTH.

The two teams' burdens are compared. This allows several important absences for
one team to outweigh one key absence for the other team.

If confirmed lineups are available:
- a player reported injured but actually starting is removed from the injury burden
- V25 builds a current core-player list
- important normal starters who are benched or omitted are identified even when
  the reason is rotation rather than injury
- bench omissions receive a smaller analyst penalty than a complete matchday-XI omission

If API-Football reports that player-stat coverage is not available, V25 fails
closed rather than inventing a player-importance score.

## Verified analyst pass

Use **RUN FULL VERIFIED ANALYST PASS** to enrich the ten strongest current cases.
The final five then receive a second pass so injury-driven reordering cannot leave
a newly promoted anchor without the confirmed-XI/core-player check. Provider xG
is also attempted for final anchors when the competition actually exposes it.

Provider results are cached in the running Streamlit process.

## Speed

Normal rankings still use the fast three-season model. Full historical
TRAIN -> TUNE -> untouched FINAL TEST validation remains separate and on-demand.

## Secrets

```toml
ODDS_API_KEY = "..."
API_FOOTBALL_KEY = "..."
```

## Stress testing

The included `V25_STRESS_TEST_REPORT.md` documents the final automated tests.
The final build passed 33/33 structural and synthetic decision checks plus Python
compilation.

Known deliberate limitations:
- player importance is a transparent current-season heuristic, not a learned
  historical player plus/minus coefficient
- process-level caches reset on a full Streamlit container restart
- accumulator joint probability is an independence approximation
- API-Football coverage, lineups and xG vary by league and fixture

No football model can guarantee an outcome.
