from v18_selection import build_acca, score_selection


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
