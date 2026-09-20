from v18_selection import build_acca, build_best_chance_acca, score_selection


def _row(team="A", price=2.2):
    return {
        "Match": f"{team} v B", "Pick": "HOME", "Confidence %": 72,
        "Edge pp": 7, "EV %": 12, "Best market odds": price,
        "Decision": "BET", "Validation": "APPROVED", "Market integrity": "OK",
        "Context": {
            "available": True, "xg_like_home": 1.9, "xg_like_away": 0.9,
            "injuries": "Verified: no material absences",
            "home": {"ppg": 2.2, "venue_ppg": 2.4, "position": 2, "played": 8,
                     "clean_sheet": 50, "gf": 1.9},
            "away": {"ppg": 0.8, "venue_ppg": 0.6, "position": 15, "played": 8,
                     "clean_sheet": 12, "gf": 0.8},
        },
    }


def test_elite_requires_verified_bet():
    audit = score_selection(_row())
    assert audit["tier"] == "ELITE"
    assert audit["acca_eligible"] is True


def test_missing_market_cannot_enter_acca():
    row = _row()
    row.update({"Decision": "PREDICTION ONLY", "Edge pp": None, "EV %": None})
    audit = score_selection(row)
    assert audit["acca_eligible"] is False
    assert audit["tier"] in {"WATCHLIST", "REJECT"}


def test_acca_builder_never_forces_unverified_legs():
    rows = []
    for i in range(4):
        row = _row(chr(65 + i), 2.6)
        row["V18 audit"] = score_selection(row)
        rows.append(row)
    result = build_acca(rows, target_odds=50, min_legs=4, max_legs=7)
    assert result["status"] == "READY"
    assert len(result["legs"]) == 4
    assert result["combined_odds"] > 1

    rows[0]["V18 audit"]["acca_eligible"] = False
    stopped = build_acca(rows, target_odds=50, min_legs=4, max_legs=7)
    assert stopped["status"] == "INSUFFICIENT"


def test_best_chance_builder_maximises_probability_at_target():
    rows=[]
    # Six safer 1.95 legs just clear a 50.0 target; five longer-priced legs are
    # available, but their joint model probability is lower.
    for i in range(6):
        row=_row(f"Safe {i}",1.95)
        row.update({"Confidence %":68,"Bookmakers":10,"Kickoff ISO":"2026-09-26T14:00:00Z",
                    "Market diagnostic":{"stage":"accepted"},"League":"Test"})
        row["V18 audit"]=score_selection(row)
        rows.append(row)
    for i in range(5):
        row=_row(f"Long {i}",2.25)
        row.update({"Confidence %":52,"Bookmakers":10,"Kickoff ISO":"2026-09-26T15:00:00Z",
                    "Market diagnostic":{"stage":"accepted"},"League":"Test"})
        row["V18 audit"]=score_selection(row)
        rows.append(row)
    result=build_best_chance_acca(rows,target_odds=50,leg_counts=(5,6))
    assert result["status"]=="READY"
    assert len(result["legs"])==6
    assert result["combined_odds"]>=50
    assert all(leg["Match"].startswith("Safe") for leg in result["legs"])


def test_best_chance_builder_reports_closest_below_target():
    rows=[]
    for i in range(5):
        row=_row(f"Short {i}",1.5)
        row.update({"Confidence %":75,"Bookmakers":10,"Kickoff ISO":"2026-09-26T14:00:00Z",
                    "Market diagnostic":{"stage":"accepted"},"League":"Test"})
        row["V18 audit"]=score_selection(row)
        rows.append(row)
    result=build_best_chance_acca(rows,target_odds=50,leg_counts=(5,))
    assert result["status"]=="BELOW_TARGET"
    assert len(result["legs"])==5
    assert result["combined_odds"]<50
