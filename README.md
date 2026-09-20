# Craig's Football Predictor V19

Multi-league Streamlit research model for 1X2 probabilities and market-edge screening.

Supported:
- Premier League
- Championship
- League One
- League Two
- La Liga
- Bundesliga
- Serie A
- Ligue 1

V19 turns the predictor into an automatic, picks-first dashboard. It runs when
the app opens, shows one ranked shortlist and an automatic acca result, and
moves API controls, thresholds and model laboratories into a collapsed Settings
sidebar. The redundant comparison lists and decorative bottom navigation have
been removed so the same selection is not repeated in several places.

The default shortlist is always the ten highest model win probabilities for the
chosen fixture dates. Date controls support Today, This Saturday and a custom
range. A separate personalised acca planner accepts a stake, target return and
five/six-team preference, then selects the verified combination with the highest
estimated joint success probability that reaches the requested odds. If the
target is impossible for the available fixtures, it reports the closest honest
alternative rather than weakening data-verification rules or claiming certainty.

The V18 safety engine remains underneath the new interface. It grades every
outright pick as ELITE, STRONG, WATCHLIST or REJECT from
model probability, scoring profile, venue-adjusted form, verified bookmaker
value, availability evidence, opponent strength and supporting indicators.

The automatic acca builder considers four to seven legs and searches for the
eligible combination closest to the requested return. It is deliberately
fail-closed: only outright-win selections rated ELITE with a verified BET
decision can enter. Missing market odds, fixture identity, lineup/injury data or
true xG are never silently treated as passed checks.
