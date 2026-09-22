"""V29 ACCA-FIRST engine for Craig's Football Predictor.

Primary product objective:
    Choose exactly five outright 90-minute match winners that maximise the
    estimated probability of the whole accumulator landing, subject to a
    user-defined minimum combined-odds target.

The engine deliberately separates raw prediction probability from selection
probability. Each candidate is challenged against market consensus, an
independent provider prediction when available, draw threat, analyst evidence,
line-up/availability state, stability and competition/rotation risk before it
can enter the optimiser.

No football model guarantees an outcome. Joint probabilities are an
independence approximation used to compare candidate five-folds.
"""

from __future__ import annotations

from itertools import combinations
from math import isfinite
from typing import Any, Dict, Iterable, List, Optional


def _num(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        value = float(value)
        return value if isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _selected_provider_probability(row: Dict[str, Any]) -> Optional[float]:
    pick = str(row.get("Pick", "")).upper()
    ctx = row.get("Context") if isinstance(row.get("Context"), dict) else {}
    provider = ctx.get("provider_prediction") if isinstance(ctx.get("provider_prediction"), dict) else {}
    if not provider or not provider.get("available"):
        return None
    raw = provider.get("home") if pick == "HOME" else provider.get("away") if pick == "AWAY" else None
    value = _num(raw)
    if value is None:
        return None
    # Provider fields in the app are percentages, but tolerate probability form.
    return _clip(value / 100.0 if value > 1.0 else value, 0.01, 0.99)


def _is_unavailable(row: Dict[str, Any]) -> bool:
    provider_state = str(row.get("Provider fixture state", "")).upper()
    status_text = " ".join(
        str(row.get(k, "")) for k in ("Fixture status", "External data status", "Timing label")
    ).upper()
    return provider_state == "UNAVAILABLE" or any(
        token in status_text for token in ("POSTPONED", "CANCELLED", "CANCELED", "ABANDONED")
    )


def challenge_candidate(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return an adversarially-adjusted outright-win candidate or ``None``.

    Hard gates are factual/betability gates only. Soft evidence never removes a
    candidate merely because one source is missing; instead the uncertainty is
    explicitly penalised so a stronger fully-evidenced team can replace it.
    """

    pick = str(row.get("Pick", "")).upper()
    game_state = str(row.get("Game state", "")).upper()
    price = _num(row.get("Best market odds"))
    raw_pct = _num(row.get("Ranking %"), _num(row.get("Confidence %")))
    anchor_pct = _num(row.get("Anchor score"), 50.0)
    diagnostic = row.get("Market diagnostic") if isinstance(row.get("Market diagnostic"), dict) else {}

    if (
        pick not in ("HOME", "AWAY")
        or game_state != "UPCOMING"
        or _is_unavailable(row)
        or price is None
        or price <= 1.01
        or raw_pct is None
        or not 0.0 < raw_pct < 100.0
        or diagnostic.get("stage") != "accepted"
    ):
        return None

    raw = raw_pct / 100.0
    market_pct = _num(row.get("Market fair %"))
    market = market_pct / 100.0 if market_pct is not None and 0 < market_pct < 100 else None
    provider = _selected_provider_probability(row)

    # Consensus probability: model remains the main signal; live bookmaker
    # consensus and the independent provider challenge it rather than replace it.
    sources = [("model", raw, 0.62)]
    if market is not None:
        sources.append(("market", market, 0.23))
    if provider is not None:
        sources.append(("provider", provider, 0.15))
    weight_sum = sum(w for _, _, w in sources)
    consensus = sum(p * w for _, p, w in sources) / weight_sum

    adjustments: List[Dict[str, Any]] = []
    def penalise(label: str, pp: float, reason: str) -> None:
        nonlocal consensus
        consensus -= pp / 100.0
        adjustments.append({"label": label, "pp": -round(pp, 1), "reason": reason})

    def reward(label: str, pp: float, reason: str) -> None:
        nonlocal consensus
        consensus += pp / 100.0
        adjustments.append({"label": label, "pp": round(pp, 1), "reason": reason})

    # The manual method rejected selections when independent views diverged.
    probs = [p for _, p, _ in sources]
    spread_pp = (max(probs) - min(probs)) * 100 if len(probs) > 1 else 0.0
    if spread_pp >= 20:
        penalise("model disagreement", 8.0, f"Probability sources disagree by {spread_pp:.1f}pp.")
    elif spread_pp >= 14:
        penalise("model disagreement", 5.0, f"Probability sources disagree by {spread_pp:.1f}pp.")
    elif spread_pp >= 9:
        penalise("model disagreement", 2.5, f"Probability sources disagree by {spread_pp:.1f}pp.")
    elif len(probs) >= 2:
        reward("consensus", 1.0, "Available probability sources broadly agree.")

    # Analyst score is contextual corroboration, not another raw probability.
    if anchor_pct >= 78:
        reward("analyst evidence", 3.0, f"Anchor score is strong at {anchor_pct:.0f}/100.")
    elif anchor_pct >= 70:
        reward("analyst evidence", 1.5, f"Anchor score supports the pick at {anchor_pct:.0f}/100.")
    elif anchor_pct < 58:
        penalise("analyst evidence", 5.0, f"Anchor score is only {anchor_pct:.0f}/100.")
    elif anchor_pct < 64:
        penalise("analyst evidence", 2.5, f"Anchor score is moderate at {anchor_pct:.0f}/100.")

    draw_pct = _num(row.get("Draw %"))
    if draw_pct is not None:
        if draw_pct >= raw_pct:
            penalise("draw threat", 9.0, "The draw is modelled at least as likely as the selected-team win.")
        elif draw_pct >= raw_pct - 5:
            penalise("draw threat", 5.0, "The draw probability is within 5pp of the selected-team win.")
        elif draw_pct >= raw_pct - 10:
            penalise("draw threat", 2.0, "The draw remains a material competing outcome.")

    evidence_pct = _num(row.get("Evidence completeness %"), 0.0) or 0.0
    if evidence_pct < 35:
        penalise("evidence coverage", 7.0, f"Only {evidence_pct:.0f}% of corroborating evidence is available.")
    elif evidence_pct < 50:
        penalise("evidence coverage", 4.0, f"Only {evidence_pct:.0f}% of corroborating evidence is available.")
    elif evidence_pct >= 75:
        reward("evidence coverage", 1.0, f"Evidence coverage is {evidence_pct:.0f}%.")

    stability = str(row.get("Stability", "")).upper()
    if stability == "LOW":
        penalise("stability", 7.0, "The pick is unstable under stress tests.")
    elif stability == "MEDIUM":
        penalise("stability", 2.0, "The pick has medium stress-test stability.")
    elif stability == "HIGH":
        reward("stability", 1.5, "The pick remains strong under stress tests.")

    readiness = str(row.get("Anchor readiness", "PROVISIONAL")).upper()
    late = str(row.get("Late info", "")).upper()
    ctx = row.get("Context") if isinstance(row.get("Context"), dict) else {}
    lineups_confirmed = bool(ctx.get("lineups_confirmed"))
    if readiness != "FINAL":
        penalise("verification", 3.0, "The selection is not yet FULL VERIFIED.")
    if "REVALIDATE NOW" in late and not lineups_confirmed:
        penalise("late lineup", 4.0, "Kick-off is close and confirmed XIs are not yet verified.")

    # Existing analyst engine already records adversarial risks. Treat the count
    # as a compact way of carrying injuries, schedule, scoring and opposition
    # concerns into the portfolio optimiser without double-counting them heavily.
    risk_count = int(_num(row.get("Analyst risk count"), 0) or 0)
    if risk_count >= 4:
        penalise("adversarial risks", 4.0, f"Analyst layer found {risk_count} material concerns.")
    elif risk_count >= 2:
        penalise("adversarial risks", 2.0, f"Analyst layer found {risk_count} material concerns.")

    # Cup/academy rotation is structurally less predictable before line-ups.
    league = str(row.get("League", ""))
    match = str(row.get("Match", ""))
    academy_fixture = any(x in match.upper() for x in (" U21", " U23", " ACADEMY"))
    cup_fixture = any(x in league.casefold() for x in ("trophy", "cup"))
    if (academy_fixture or cup_fixture) and not lineups_confirmed:
        penalise("rotation risk", 2.5, "Cup/academy fixture before confirmed line-ups carries extra rotation variance.")

    # Penalise extreme model-v-market optimism independently of general spread.
    if market is not None:
        gap_pp = (raw - market) * 100
        if gap_pp >= 16:
            penalise("market contradiction", 4.0, f"Model is {gap_pp:.1f}pp above the de-margined market.")
        elif gap_pp >= 10:
            penalise("market contradiction", 2.0, f"Model is {gap_pp:.1f}pp above the de-margined market.")

    adjusted = _clip(consensus, 0.05, 0.95)
    implied = 1.0 / price
    edge = adjusted - implied

    return {
        "row": row,
        "price": price,
        "raw_probability": raw,
        "adjusted_probability": adjusted,
        "market_probability": market,
        "provider_probability": provider,
        "anchor": _clip((anchor_pct or 0.0) / 100.0, 0.0, 1.0),
        "spread_pp": spread_pp,
        "adjusted_edge": edge,
        "adjustments": adjustments,
        "readiness": str(row.get("Anchor readiness", "PROVISIONAL")),
    }


def _candidate_pool(candidates: List[Dict[str, Any]], cap: int = 34) -> List[Dict[str, Any]]:
    """Keep a diverse bounded pool without trimming only on favourite probability.

    The old app took the top analyst anchors first, which could remove the exact
    medium-priced team needed for a high-probability target-reaching five-fold.
    V29 retains candidates from three lenses before deduping: safest winners,
    probability/price efficiency and positive adjusted edge.
    """
    if len(candidates) <= cap:
        return list(candidates)

    safest = sorted(candidates, key=lambda x: x["adjusted_probability"], reverse=True)[:18]
    efficient = sorted(
        candidates,
        key=lambda x: x["adjusted_probability"] * (x["price"] ** 0.42),
        reverse=True,
    )[:18]
    value = sorted(candidates, key=lambda x: x["adjusted_edge"], reverse=True)[:12]

    result: List[Dict[str, Any]] = []
    seen = set()
    for item in safest + efficient + value:
        r = item["row"]
        identity = (r.get("League"), r.get("Match"))
        if identity not in seen:
            seen.add(identity)
            result.append(item)
        if len(result) >= cap:
            break
    return result


def _pack_combo(combo: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    combo = tuple(combo)
    odds = 1.0
    joint = 1.0
    raw_joint = 1.0
    anchor_sum = 0.0
    for item in combo:
        odds *= item["price"]
        joint *= item["adjusted_probability"]
        raw_joint *= item["raw_probability"]
        anchor_sum += item["anchor"]
    weakest = min(combo, key=lambda x: x["adjusted_probability"])
    enriched_legs = []
    for x in combo:
        enriched = dict(x["row"])
        enriched["V29 raw %"] = round(x["raw_probability"] * 100, 2)
        enriched["V29 adjusted %"] = round(x["adjusted_probability"] * 100, 2)
        enriched["V29 source spread pp"] = round(x["spread_pp"], 2)
        enriched["V29 adjusted edge pp"] = round(x["adjusted_edge"] * 100, 2)
        enriched["V29 challenge"] = [a for a in x["adjustments"] if a.get("pp", 0) < 0]
        enriched_legs.append(enriched)
    return {
        "legs": enriched_legs,
        "candidate_audits": [
            {
                "Match": x["row"].get("Match"),
                "Pick": x["row"].get("Pick"),
                "raw_probability": round(x["raw_probability"] * 100, 2),
                "adjusted_probability": round(x["adjusted_probability"] * 100, 2),
                "market_probability": round(x["market_probability"] * 100, 2) if x["market_probability"] is not None else None,
                "provider_probability": round(x["provider_probability"] * 100, 2) if x["provider_probability"] is not None else None,
                "price": round(x["price"], 4),
                "adjusted_edge_pp": round(x["adjusted_edge"] * 100, 2),
                "spread_pp": round(x["spread_pp"], 2),
                "readiness": x["readiness"],
                "adjustments": x["adjustments"],
            }
            for x in combo
        ],
        "combined_odds": round(odds, 2),
        "joint_probability": round(joint * 100, 2),
        "raw_joint_probability": round(raw_joint * 100, 2),
        "mean_anchor_score": round(anchor_sum / len(combo) * 100, 1),
        "weakest_match": weakest["row"].get("Match"),
        "weakest_adjusted_probability": round(weakest["adjusted_probability"] * 100, 2),
    }


def build_acca_first(
    rows: Iterable[Dict[str, Any]],
    target_odds: float = 30.0,
    legs: int = 5,
    pool_cap: int = 34,
) -> Dict[str, Any]:
    """Build the safest target-reaching outright-winner accumulator.

    Objective hierarchy:
      1. Exactly five 90-minute outright winners (``legs`` kept as a test hook;
         production UI fixes it at five).
      2. Combined decimal odds >= target_odds.
      3. Maximise adversarially-adjusted joint win probability.
      4. Prefer higher evidence/anchor quality.
      5. Prefer less unnecessary excess price once probability/quality tie.

    If the target is unreachable, return both the safest five and the closest
    five so the UI never silently inflates risk merely to chase the payout.
    """
    legs = int(legs)
    if legs != 5:
        raise ValueError("V29 production methodology requires exactly five teams.")
    target = max(float(target_odds), 1.01)

    challenged = [c for c in (challenge_candidate(r) for r in rows) if c is not None]
    pool = _candidate_pool(challenged, cap=pool_cap)
    if len(pool) < legs:
        return {
            "status": "INSUFFICIENT",
            "legs": [],
            "candidate_count": len(challenged),
            "optimiser_pool_count": len(pool),
            "reason": f"Only {len(pool)} upcoming outright winners have a securely matched current price; five are required.",
        }

    reaching = []
    below = []
    for combo in combinations(pool, legs):
        packed = _pack_combo(combo)
        # Whole-slip quality. The primary term is adjusted joint probability;
        # mean anchor and lower excess odds are tie-breakers only.
        rank = (
            packed["joint_probability"],
            packed["mean_anchor_score"],
            -packed["combined_odds"],
        )
        record = (rank, packed)
        if packed["combined_odds"] >= target:
            reaching.append(record)
        else:
            below.append(record)

    if reaching:
        reaching.sort(key=lambda x: x[0], reverse=True)
        best = reaching[0][1]
        alternatives = [x[1] for x in reaching[1:4]]
        best.update(
            {
                "status": "READY",
                "target_odds": round(target, 2),
                "candidate_count": len(challenged),
                "optimiser_pool_count": len(pool),
                "alternatives": alternatives,
                "method": "ACCA-FIRST adversarial probability optimisation",
                "reason": "Highest adversarially-adjusted joint win probability among five-team outright-winner combinations that reach the target.",
            }
        )
        return best

    # Target cannot be reached: safest is probability-first; closest is payout-first.
    safest_record = max(
        below,
        key=lambda x: (x[1]["joint_probability"], x[1]["mean_anchor_score"], x[1]["combined_odds"]),
    )[1]
    closest_record = max(
        below,
        key=lambda x: (x[1]["combined_odds"], x[1]["joint_probability"], x[1]["mean_anchor_score"]),
    )[1]
    return {
        "status": "BELOW_TARGET",
        "target_odds": round(target, 2),
        "candidate_count": len(challenged),
        "optimiser_pool_count": len(pool),
        "safest": safest_record,
        "closest": closest_record,
        "method": "ACCA-FIRST adversarial probability optimisation",
        "reason": "No five-team combination reaches the requested target without leaving the current eligible outright-winner pool.",
    }
