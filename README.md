# Craig's Football Predictor V18

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

V18 adds a transparent manual-selection layer on top of the calibrated 1X2
model. It grades every outright pick as ELITE, STRONG, WATCHLIST or REJECT from
model probability, scoring profile, venue-adjusted form, verified bookmaker
value, availability evidence, opponent strength and supporting indicators.

The automatic acca builder considers four to seven legs and searches for the
eligible combination closest to the requested return. It is deliberately
fail-closed: only outright-win selections rated ELITE with a verified BET
decision can enter. Missing market odds, fixture identity, lineup/injury data or
true xG are never silently treated as passed checks.
