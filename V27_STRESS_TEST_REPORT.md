# V27 Production Consolidation — Stress Test Report

**48/48 automated checks passed.** Python compilation also passed.

## Automated checks

- PASS — **Frozen probabilities sum to 1**: {'neutral': [0.384861011272219, 0.2820534404596988, 0.3330855482680823], 'strong': [0.6464043659322308, 0.20801265362323562, 0.14558298044453352]}
- PASS — **Frozen probabilities bounded**: {'neutral': [0.384861011272219, 0.2820534404596988, 0.3330855482680823], 'strong': [0.6464043659322308, 0.20801265362323562, 0.14558298044453352]}
- PASS — **Stronger home evidence raises home probability**: {'neutral_home': 0.384861011272219, 'strong_home': 0.6464043659322308}
- PASS — **Neutral draw probability plausible**: 0.2820534404596988
- PASS — **Normal live path contains no model fit**
- PASS — **Normal live path uses V27 frozen scorer**
- PASS — **Analyst score finite**: 77.0
- PASS — **Stability grade valid**: {'grade': 'HIGH', 'floor': 77.0, 'range': 0.0, 'std': 0.0}
- PASS — **Fragile draw-heavy case not HIGH**: {'grade': 'MEDIUM', 'floor': 57.5, 'range': 4.5, 'std': 1.35}
- PASS — **Three important injuries outweigh one key injury**: {'one': 7.6, 'three': 16.9}
- PASS — **Favourable availability raises case**
- PASS — **Adverse availability lowers case**
- PASS — **Confirmed XI overrides stale injury**: {'burden': 0.0, 'key_absences': 0, 'important_absences': 0, 'missing': [], 'importance_complete': True, 'stale_reported_but_starting': ['P1'], 'confirmed_lineup': True}
- PASS — **Benched core player detected**: {'available': True, 'burden': 3.6, 'missing_core': [{'player': 'Star', 'player_id': 10, 'importance': 88, 'lineup_status': 'BENCH', 'weighted_impact': 39.6}]}
- PASS — **Bench impact discounted**: {'available': True, 'burden': 3.6, 'missing_core': [{'player': 'Star', 'player_id': 10, 'importance': 88, 'lineup_status': 'BENCH', 'weighted_impact': 39.6}]}
- PASS — **Inside 2h without XI -> REVALIDATE NOW**
- PASS — **Confirmed XI -> XI VERIFIED**
- PASS — **3h window -> RECHECK NEAR KICKOFF**
- PASS — **FINAL anchors outrank provisional**: [{'Pick': 'HOME', 'Game state': 'UPCOMING', 'Anchor readiness': 'FINAL', 'Anchor score': 75, 'Ranking %': 72, 'team': 'Final1'}, {'Pick': 'HOME', 'Game state': 'UPCOMING', 'Anchor readiness': 'FINAL', 'Anchor score': 74, 'Ranking %': 70, 'team': 'Final2'}, {'Pick': 'HOME', 'Game state': 'UPCOMING', 'Anchor readiness': 'FINAL', 'Anchor score': 73, 'Ranking %': 69, 'team': 'Final3'}, {'Pick': 'HOME', 'Game state': 'UPCOMING', 'Anchor readiness': 'FINAL', 'Anchor score': 72, 'Ranking %': 68, 'team': 'Final4'}, {'Pick': 'HOME', 'Game state': 'UPCOMING', 'Anchor readiness': 'FINAL', 'Anchor score': 71, 'Ranking %': 67, 'team': 'Final5'}]
- PASS — **Target acca uses FINAL anchors only**: {'status': 'READY', 'ids': [0, 1, 2, 3, 4]}
- PASS — **RANKED priced row enters ledger**
- PASS — **Prediction-only row excluded from price ledger**
- PASS — **Trust adjusts mature competition only**: [{'League': 'Premier League', 'Anchor score': 73.0, 'Competition trust adjustment': 3.0, 'Competition trust': '40 settled • actual 70.0% vs stated 50.0%'}, {'League': 'Unknown', 'Anchor score': 70.0, 'Competition trust adjustment': 0.0, 'Competition trust': 'NEUTRAL — insufficient settled history'}]
- PASS — **Trust adjustment capped ±3 in source**
- PASS — **Two-season live state**: Two-season live state
- PASS — **Automatic analyst pass**: Automatic analyst pass
- PASS — **Provisional screen capped at 8**: Provisional screen capped at 8
- PASS — **First pass lightweight**: First pass lightweight
- PASS — **Final five full context**: Final five full context
- PASS — **Automatic network cap 80**: Automatic network cap 80
- PASS — **API accounting present**: API accounting present
- PASS — **Near-kickoff cache freshness 15m**: Near-kickoff cache freshness 15m
- PASS — **Stability engine present**: Stability engine present
- PASS — **Late revalidation present**: Late revalidation present
- PASS — **Anchor history snapshot present**: Anchor history snapshot present
- PASS — **Anchor export present**: Anchor export present
- PASS — **Anchor restore present**: Anchor restore present
- PASS — **Automatic settlement present**: Automatic settlement present
- PASS — **Post-match diagnostics present**: Post-match diagnostics present
- PASS — **Competition trust needs 20**: Competition trust needs 20
- PASS — **Trust modifies Anchor score only**: Trust modifies Anchor score only
- PASS — **FINAL/PROVISIONAL readiness present**: FINAL/PROVISIONAL readiness present
- PASS — **Expanded EFL Trophy retained**: Expanded EFL Trophy retained
- PASS — **League One retained**: League One retained
- PASS — **League Two retained**: League Two retained
- PASS — **Admin lab toggle exists**: Admin lab toggle exists
- PASS — **No stale Prediction field**: No stale Prediction field
- PASS — **No duplicate top-level defs**: No duplicate top-level defs


## Stress-test conclusions

### Streamlit CPU throttling
The normal live path no longer fits a scikit-learn model. It uses a frozen V27
scorer and recent two-season team state. Historical ML fitting remains in the
Admin research tools only.

### Automatic analyst workflow
The app automatically investigates the strongest provisional candidates when
API-Football is connected. It first runs a lighter availability/lineup pass on
eight candidates, then gives the eventual five the fuller
schedule/provider-prediction/xG/core-XI pass.

### API runaway protection
Automatic analyst work has an 80-real-network-call budget. Cached calls do not
consume the network-miss counter. If the budget is hit, the row becomes
`PROVISIONAL — API budget`.

### Injury / lineup logic
The suite verifies:
- several important opponent absences can outweigh one key selected-team absence
- adverse availability weakens the selection
- confirmed starters override stale injury reports
- important benched normal starters are detected
- bench omissions are discounted versus complete matchday omissions

### Stability and late information
The Stability engine perturbs form, draw risk, market probability, availability
and scoring assumptions. Inside two hours of kickoff an unconfirmed XI becomes
`REVALIDATE NOW`; a confirmed XI becomes `XI VERIFIED`.

### Final vs provisional
FINAL anchors rank ahead of provisional rows, and the target-return acca builder
uses FINAL anchors only.

### Learning and competition trust
Final anchors are snapshotted before kickoff. Settled selections feed diagnostic
tags based on warnings that existed pre-match. Competition trust remains neutral
until 20 settled selections exist and is capped at ±3 Anchor Score points.

## Remaining deliberate limitations

1. The frozen V27 live scorer solves the normal-startup CPU problem but is not a
   newly trained league-specific ML artifact. Heavy historical ML validation
   remains available separately.
2. API-Football can still be request-heavy around injury-dense matches and
   confirmed lineups. The hard budget fails soft to PROVISIONAL instead of
   pretending verification completed.
3. Community Cloud process state is not durable across a full container
   replacement. V27 supports Anchor History export and restore; a database would
   be required for fully automatic cross-reboot learning.
4. Post-match diagnostics identify warnings present before kickoff; they cannot
   prove the true cause of a football result.
5. Live API behavior with the user's actual API-Football account, quota and the
   exact EFL Trophy payload still requires one deployment smoke test.

## Deployment conclusion

V27 is ready for the next deployment smoke test. The CPU-heavy normal fit is
gone, automatic analyst work is bounded, incomplete verification remains clearly
PROVISIONAL, and the app now tracks stability, late information, competition
trust and frozen anchor history.
