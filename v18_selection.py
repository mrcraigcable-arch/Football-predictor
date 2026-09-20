"""V18 decision layer for Craig's Football Predictor.

This module deliberately contains no Streamlit code so the scoring and acca
construction rules can be tested independently from the interface.
"""

from __future__ import annotations

from itertools import combinations
from math import isfinite, log


WEIGHTS = {
    "model_probability": 30,
    "goals_profile": 20,
    "recent_venue_form": 15,
    "market_value": 15,
    "availability_evidence": 10,
    "opponent_strength": 5,
    "supporting_indicators": 5,
}


def _number(value, default=None):
    try:
        value = float(value)
        return value if isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _clip(value, low=0.0, high=100.0):
    return max(low, min(high, float(value)))


def _side(context, pick):
    if not isinstance(context, dict):
        return {}, {}
    if pick == "HOME":
        return context.get("home", {}) or {}, context.get("away", {}) or {}
    if pick == "AWAY":
        return context.get("away", {}) or {}, context.get("home", {}) or {}
    return {}, {}


def score_selection(row):
    """Return a transparent 0-100 manual-selection score and its audit trail.

    The score uses only fields already verified by the app. The current feed
    has no true xG or player-availability provider, so goals_profile is labelled
    as a scoring-rate proxy and missing availability is penalised.
    """
    pick = str(row.get("Pick", "")).upper()
    context = row.get("Context") if isinstance(row.get("Context"), dict) else {}
    selected, opponent = _side(context, pick)
    concerns = []
    positives = []

    confidence = _number(row.get("Confidence %"), 0.0)
    model_score = _clip((confidence - 45.0) / 35.0 * 100.0)
    if confidence >= 68:
        positives.append(f"Model win probability {confidence:.1f}%")
    elif confidence < 60:
        concerns.append(f"Model probability only {confidence:.1f}%")

    home_proxy = _number(context.get("xg_like_home"))
    away_proxy = _number(context.get("xg_like_away"))
    if pick == "HOME" and home_proxy is not None and away_proxy is not None:
        goals_advantage = home_proxy - away_proxy
    elif pick == "AWAY" and home_proxy is not None and away_proxy is not None:
        goals_advantage = away_proxy - home_proxy
    else:
        goals_advantage = None
    goals_score = _clip(50 + 32 * goals_advantage) if goals_advantage is not None else 0.0
    if goals_advantage is None:
        concerns.append("Scoring-rate profile unavailable")
    elif goals_advantage >= 0.35:
        positives.append(f"Scoring-rate advantage {goals_advantage:+.2f}")
    elif goals_advantage < 0:
        concerns.append(f"Scoring-rate profile opposes pick ({goals_advantage:+.2f})")

    selected_ppg = _number(selected.get("ppg"))
    opponent_ppg = _number(opponent.get("ppg"))
    selected_venue = _number(selected.get("venue_ppg"), selected_ppg)
    opponent_venue = _number(opponent.get("venue_ppg"), opponent_ppg)
    if None not in (selected_ppg, opponent_ppg, selected_venue, opponent_venue):
        form_advantage = 0.65 * (selected_ppg - opponent_ppg) + 0.35 * (selected_venue - opponent_venue)
        form_score = _clip(50 + 24 * form_advantage)
        if form_advantage >= 0.5:
            positives.append(f"Recent/venue PPG advantage {form_advantage:+.2f}")
        elif form_advantage <= -0.35:
            concerns.append(f"Recent/venue form opposes pick ({form_advantage:+.2f})")
    else:
        form_advantage = None
        form_score = 0.0
        concerns.append("Recent home/away split unavailable")

    edge = _number(row.get("Edge pp"))
    ev = _number(row.get("EV %"))
    if edge is None or ev is None:
        value_score = 0.0
        concerns.append("Current bookmaker value not verified")
    else:
        value_score = _clip(35 + 5.0 * edge + 1.2 * ev)
        if edge >= 4 and ev > 0:
            positives.append(f"Verified edge {edge:+.1f}pp; EV {ev:+.1f}%")
        else:
            concerns.append(f"Price fails preferred value margin ({edge:+.1f}pp edge)")

    injuries = str(context.get("injuries", "")).upper()
    availability_verified = bool(injuries and "UNAVAILABLE" not in injuries)
    availability_score = 100.0 if availability_verified else 20.0
    if not availability_verified:
        concerns.append("Lineups/injuries not independently verified")

    selected_pos = _number(selected.get("position"))
    opponent_pos = _number(opponent.get("position"))
    if selected_pos is not None and opponent_pos is not None:
        rank_advantage = opponent_pos - selected_pos
        opponent_score = _clip(50 + 6 * rank_advantage)
    else:
        rank_advantage = None
        opponent_score = 35.0

    clean_sheet = _number(selected.get("clean_sheet"))
    opponent_gf = _number(opponent.get("gf"))
    if clean_sheet is not None and opponent_gf is not None:
        indicators_score = _clip(0.75 * clean_sheet + 25 * (1.6 - opponent_gf))
    else:
        indicators_score = 35.0

    components = {
        "model_probability": round(model_score, 1),
        "goals_profile": round(goals_score, 1),
        "recent_venue_form": round(form_score, 1),
        "market_value": round(value_score, 1),
        "availability_evidence": round(availability_score, 1),
        "opponent_strength": round(opponent_score, 1),
        "supporting_indicators": round(indicators_score, 1),
    }
    base_score = sum(components[key] * weight / 100.0 for key, weight in WEIGHTS.items())

    penalties = []
    validation = str(row.get("Validation", ""))
    decision = str(row.get("Decision", ""))
    market_integrity = str(row.get("Market integrity", ""))
    played = min(_number(selected.get("played"), 0), _number(opponent.get("played"), 0))
    if played < 5:
        penalties.append((6, "Current-season sample below five matches"))
    if "FALLBACK" in validation:
        penalties.append((8, "League calibration uses a safety fallback"))
    elif "APPROVED RAW" in validation:
        penalties.append((3, "League evidence favours an uncalibrated model"))
    if not availability_verified:
        penalties.append((5, "Availability evidence missing"))
    if market_integrity and market_integrity != "OK":
        penalties.append((8, "Bookmaker prices disagree"))
    if decision == "PASS":
        penalties.append((15, "Existing confidence/value gate failed"))
    if decision == "PREDICTION ONLY":
        penalties.append((10, "No matched current market"))
    if pick == "DRAW":
        penalties.append((20, "Draws are excluded from the acca shortlist"))
    if form_advantage is not None and form_advantage <= -0.35:
        penalties.append((7, "Recent form conflicts with the model pick"))
    if goals_advantage is not None and goals_advantage < 0:
        penalties.append((6, "Scoring profile conflicts with the model pick"))

    penalty_total = sum(amount for amount, _ in penalties)
    final_score = round(_clip(base_score - penalty_total), 1)
    outright = pick in ("HOME", "AWAY")
    if outright and decision == "BET" and final_score >= 70:
        tier = "ELITE"
    elif outright and decision in ("BET", "VERIFY") and final_score >= 60:
        tier = "STRONG"
    elif outright and final_score >= 48:
        tier = "WATCHLIST"
    else:
        tier = "REJECT"

    return {
        "score": final_score,
        "base_score": round(base_score, 1),
        "tier": tier,
        "components": components,
        "penalties": [{"points": p, "reason": r} for p, r in penalties],
        "penalty_total": penalty_total,
        "positives": positives,
        "concerns": concerns,
        "goals_profile_label": "scoring-rate proxy (not provider xG)",
        "acca_eligible": tier == "ELITE" and decision == "BET" and outright,
    }


def build_acca(rows, target_odds=50.0, min_legs=4, max_legs=7):
    """Choose the eligible combination closest to target odds without forcing legs."""
    candidates = []
    for row in rows:
        audit = row.get("V18 audit") or score_selection(row)
        price = _number(row.get("Best market odds"))
        if audit.get("acca_eligible") and price is not None and price > 1.01:
            candidates.append((row, audit, price))
    candidates.sort(key=lambda item: item[1]["score"], reverse=True)
    candidates = candidates[:14]  # bounded exhaustive search for interactive use
    if len(candidates) < min_legs:
        return {"status": "INSUFFICIENT", "legs": [], "combined_odds": None,
                "reason": f"Only {len(candidates)} fully verified elite selection(s); {min_legs} required."}

    best = None
    target_log = log(max(target_odds, 1.01))
    for count in range(min_legs, min(max_legs, len(candidates)) + 1):
        for combo in combinations(candidates, count):
            combined = 1.0
            scores = []
            for _, audit, price in combo:
                combined *= price
                scores.append(audit["score"])
            # Primary objective: odds proximity. Small tie-breaker rewards stronger legs.
            objective = abs(log(combined) - target_log) + 0.002 * (100 - sum(scores) / len(scores))
            if best is None or objective < best[0]:
                best = (objective, combo, combined)
    _, combo, combined = best
    return {"status": "READY", "legs": [item[0] for item in combo],
            "combined_odds": round(combined, 2), "reason": "Closest eligible combination to the target price."}


def build_best_chance_acca(rows, target_odds=50.0, leg_counts=(5, 6), min_books=3):
    """Find the highest-model-probability acca that can reach the target.

    This planner is deliberately different from ``build_acca``. It does not
    require an ELITE/value-bet label because a short-priced, high-probability
    winner can be useful in an acca even when its price is not positive EV.
    It still requires a uniquely matched fixture, current price, bookmaker
    agreement and minimum market depth. The joint probability is an
    independence estimate, not a guarantee.
    """
    counts = sorted({int(x) for x in leg_counts if int(x) > 0})
    candidates = []
    for row in rows:
        audit = row.get("V18 audit") or score_selection(row)
        pick = str(row.get("Pick", "")).upper()
        price = _number(row.get("Best market odds"))
        probability = _number(row.get("Confidence %"))
        books = _number(row.get("Bookmakers"), 0)
        diagnostic = row.get("Market diagnostic") or {}
        accepted_match = isinstance(diagnostic, dict) and diagnostic.get("stage") == "accepted"
        if (
            pick in ("HOME", "AWAY")
            and price is not None and price > 1.01
            and probability is not None and 0 < probability < 100
            and str(row.get("Market integrity", "")) == "OK"
            and books >= min_books
            and bool(row.get("Kickoff ISO"))
            and accepted_match
        ):
            candidates.append({
                "row": row,
                "audit": audit,
                "price": price,
                "probability": probability / 100.0,
            })

    # Keep exhaustive search responsive across a multi-day range while
    # retaining both the strongest favourites and useful target-price legs.
    by_probability = sorted(candidates, key=lambda x: x["probability"], reverse=True)[:20]
    by_efficiency = sorted(
        candidates,
        key=lambda x: x["probability"] * (x["price"] ** 0.5),
        reverse=True,
    )[:12]
    pool = []
    seen = set()
    for item in by_probability + by_efficiency:
        identity = (item["row"].get("League"), item["row"].get("Match"))
        if identity not in seen:
            seen.add(identity)
            pool.append(item)

    valid_counts = [count for count in counts if count <= len(pool)]
    if not valid_counts:
        needed = min(counts) if counts else 5
        return {
            "status": "INSUFFICIENT", "legs": [], "combined_odds": None,
            "joint_probability": None,
            "reason": f"Only {len(pool)} selections have fully matched current prices; {needed} required.",
        }

    target = max(float(target_odds), 1.01)
    best_ready = None
    best_below = None
    for count in valid_counts:
        for combo in combinations(pool, count):
            combined = 1.0
            joint = 1.0
            scores = []
            for item in combo:
                combined *= item["price"]
                joint *= item["probability"]
                scores.append(_number(item["audit"].get("score"), 0))
            mean_score = sum(scores) / len(scores)
            if combined >= target:
                # Maximise estimated success chance; audit quality and avoiding
                # unnecessary excess odds are tie-breakers only.
                rank = (joint, mean_score, -combined)
                if best_ready is None or rank > best_ready[0]:
                    best_ready = (rank, combo, combined, joint)
            else:
                # If the target cannot be reached, show the closest achievable
                # return, then prefer the safer combination at that level.
                rank = (combined, joint, mean_score)
                if best_below is None or rank > best_below[0]:
                    best_below = (rank, combo, combined, joint)

    chosen = best_ready or best_below
    if chosen is None:
        return {"status": "INSUFFICIENT", "legs": [], "combined_odds": None,
                "joint_probability": None, "reason": "No valid combination could be formed."}
    _, combo, combined, joint = chosen
    ready = best_ready is not None
    return {
        "status": "READY" if ready else "BELOW_TARGET",
        "legs": [item["row"] for item in combo],
        "combined_odds": round(combined, 2),
        "joint_probability": round(joint * 100, 2),
        "target_odds": round(target, 2),
        "reason": (
            "Highest estimated joint success probability among combinations reaching the target."
            if ready else
            "No five/six-team combination reaches the target with the verified prices; this is the closest available return."
        ),
    }
