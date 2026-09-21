# V28 Final Production Verification

Verification date: 2026-09-21

## Release result

V28 Hardened Production passed the pre-deployment release gate.

- Python compilation: PASS
- Clean Streamlit environment install from `requirements.txt`: PASS
- Streamlit server health endpoint and HTML shell: PASS
- Streamlit AppTest full script execution: PASS (0 uncaught exceptions)
- V28 production contract and pure-function suite: PASS (25/25)
- Carried-forward V27 regression/stress suite: PASS (48/48)

## Release-gate coverage

The verified release retains and tests:

- default immediate fixture view plus Today, Saturday, next-seven-days and custom date range controls
- probability-ranked Top 10 before personalised acca controls
- user-entered stake, target return and five/six-team optimisation
- FINAL-only target accas and strict FULL VERIFIED readiness
- separate SCREEN VERIFIED and FULL VERIFIED provider stages
- holdout-audited frozen scorer with no live-path model fitting
- quota-header accounting, reserve protection, 429 handling and safe pagination caps
- EFL Trophy discovery priority and provider-fixture settlement
- pre-match-only ledger learning and superseded-signal protection
- confirmed-XI overrides, injury importance and late-information states
- Bayesian competition trust requiring 20 settled selections and capped at ±3 Anchor Score
- LIVE market/blender isolation
- lazy bookmaker requests after the selected league has fixtures
- safer unreachable-target output showing Safest achievable and Closest to target
- first-row pandas ledger/history hardening

## Environment observations

The keyless AppTest correctly failed soft when current 2026-27 OpenFootball files were unavailable for some leagues. It returned user-visible feed warnings and no uncaught exception. Live provider verification must therefore be completed on the deployed Streamlit app, where the existing `ODDS_API_KEY` and `API_FOOTBALL_KEY` secrets are available.

## Post-deployment smoke gate

After deployment, confirm:

1. the page identifies itself as V28 Hardened Production;
2. System health changes API-Football from CONNECTED / UNTESTED to VERIFIED after a provider request;
3. daily/minute quota values appear when returned by the provider;
4. ALL ANALYSABLE FIXTURES can return an EFL Trophy payload when fixtures exist in the selected range;
5. only FULL VERIFIED rows become FINAL;
6. the Streamlit deployment reports no uncaught runtime exception.

No additional API key or paid plan change is required.
