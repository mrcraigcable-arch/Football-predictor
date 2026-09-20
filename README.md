# Craig's Football Predictor V26 — Expanded Universe

V26 removes the biggest structural limitation in V25: the app no longer sees only
the small hard-coded fixture set.

## Fixture discovery

When `API_FOOTBALL_KEY` is connected and **ALL ANALYSABLE FIXTURES** is selected,
API-Football becomes the broader discovery layer for the selected date range.

Default **UK + Major Europe** coverage includes:
- Premier League, Championship, League One, League Two
- National League
- FA Cup
- EFL Cup / League Cup
- **EFL Trophy**
- Scottish professional leagues/cups
- UEFA Champions League
- UEFA Europa League
- UEFA Conference League
- selected major/second-tier leagues and cups across Germany, Spain, Italy,
  France, Netherlands, Portugal, Belgium, Turkey, Greece, Austria, Switzerland,
  Denmark, Norway and Sweden

Youth/reserve competitions and friendlies are excluded by default. U21 teams are
still allowed when they participate in the senior EFL Trophy.

## Probability-source hierarchy

1. Dedicated in-house OpenFootball model where a supported league model exists.
2. API-Football prediction for expanded competitions.
3. API-Football de-margined 1X2 market as a clearly labelled fallback when a
   provider prediction is unavailable.
4. If neither source exists, the fixture is discovered but not promoted into the
   probability ranking. Nothing is invented.

Provider-only selections show their source explicitly.

## Dedicated model set

V26 expands the in-house set to:
- Premier League
- Championship
- League One
- League Two
- Bundesliga
- 2. Bundesliga
- 3. Liga
- La Liga
- La Liga 2
- Serie A
- Serie B
- Ligue 1
- Ligue 2

The Odds API sport keys used for these competitions are current documented keys.

## Quota protection

Expanded discovery uses one date-range fixture request when supported and is limited to a 14-day window. Competition-season coverage is checked before prediction/odds calls. Provider prediction calls are capped at 24 per run and total provider market calls at 12.
High-priority UK competitions, especially the EFL Trophy, are processed first.

The full verified analyst pass remains separate/on-demand because injury,
player-importance, lineup and xG analysis can use additional requests.

## Five-Team Anchors

The V25 analyst architecture remains:
- model probability
- recent/venue form
- opponent-adjusted performance
- scoring/xG
- injuries/suspensions weighted by player importance
- confirmed-XI rotation/omissions
- all-competition rest/congestion
- market agreement
- draw threat
- adversarial failure check

Anchor Score is not presented as a calibrated win probability.

## Secrets

```toml
ODDS_API_KEY = "..."
API_FOOTBALL_KEY = "..."
```

No football model can guarantee an outcome.


## Duplicate protection and evidence quality

Dedicated-model fixtures take priority over provider-discovered duplicates. V26
uses date + team-name matching rather than only exact strings, reducing duplicate
rows caused by suffixes such as FC/AFC.

Expanded provider rows with little corroborating evidence receive an Analyst
Evidence Quality penalty. A high provider probability on its own therefore does
not automatically jump above a well-supported dedicated-model anchor.
