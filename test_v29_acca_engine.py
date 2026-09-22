import unittest
from v29_acca_engine import build_acca_first, challenge_candidate


def row(name, price, prob, *, anchor=72, market=None, provider=None, draw=20,
        stability="HIGH", readiness="FINAL", evidence=75, risks=0,
        game_state="UPCOMING", status="", league="League One", match=None,
        lineups=True):
    ctx={"lineups_confirmed": lineups}
    if provider is not None:
        ctx["provider_prediction"]={"available":True,"home":provider,"away":100-provider}
    return {
        "Match": match or name,
        "League": league,
        "Pick":"HOME",
        "Game state":game_state,
        "Best market odds":price,
        "Ranking %":prob,
        "Confidence %":prob,
        "Anchor score":anchor,
        "Market fair %":market,
        "Draw %":draw,
        "Stability":stability,
        "Anchor readiness":readiness,
        "Evidence completeness %":evidence,
        "Analyst risk count":risks,
        "Market diagnostic":{"stage":"accepted"},
        "Fixture status":status,
        "Context":ctx,
    }


class V29Tests(unittest.TestCase):
    def test_rejects_non_upcoming_and_postponed(self):
        self.assertIsNone(challenge_candidate(row("live",2,60,game_state="LIVE")))
        self.assertIsNone(challenge_candidate(row("pst",2,60,status="POSTPONED")))

    def test_disagreement_reduces_probability(self):
        good=challenge_candidate(row("good",2,66,market=63,provider=65))
        bad=challenge_candidate(row("bad",2,66,market=40,provider=42))
        self.assertGreater(good["adjusted_probability"],bad["adjusted_probability"])

    def test_provisional_and_low_stability_are_penalised(self):
        final=challenge_candidate(row("f",2,65,market=60,provider=64))
        risky=challenge_candidate(row("p",2,65,market=60,provider=64,readiness="PROVISIONAL",stability="LOW"))
        self.assertGreater(final["adjusted_probability"],risky["adjusted_probability"])

    def test_exactly_five_only(self):
        with self.assertRaises(ValueError):
            build_acca_first([],target_odds=10,legs=6)

    def test_optimizer_maximises_probability_subject_to_target(self):
        # Five very safe short favourites cannot reach target 20 on their own;
        # optimizer must bring in enough price, but should choose the safer of
        # two similarly priced target legs.
        rows=[
            row("A",1.35,82,market=80,provider=81),
            row("B",1.40,80,market=78,provider=79),
            row("C",1.45,78,market=75,provider=77),
            row("D",1.50,76,market=73,provider=75),
            row("E",1.55,74,market=71,provider=73),
            row("F",3.4,58,market=46,provider=55,anchor=72),
            row("G",3.4,51,market=45,provider=49,anchor=62,stability="MEDIUM"),
            row("H",2.8,61,market=50,provider=58,anchor=70),
        ]
        result=build_acca_first(rows,target_odds=20)
        self.assertEqual(result["status"],"READY")
        self.assertEqual(len(result["legs"]),5)
        self.assertGreaterEqual(result["combined_odds"],20)
        names={r["Match"] for r in result["legs"]}
        self.assertNotIn("G",names)

    def test_unreachable_returns_safest_and_closest(self):
        rows=[row(chr(65+i),1.2+i*.02,75-i,market=70-i,provider=73-i) for i in range(7)]
        result=build_acca_first(rows,target_odds=100)
        self.assertEqual(result["status"],"BELOW_TARGET")
        self.assertEqual(len(result["safest"]["legs"]),5)
        self.assertEqual(len(result["closest"]["legs"]),5)

    def test_cup_u21_without_lineup_gets_rotation_penalty(self):
        senior=challenge_candidate(row("senior",2,65,market=60,provider=64,lineups=False))
        cup=challenge_candidate(row("cup",2,65,market=60,provider=64,lineups=False,league="EFL Trophy",match="Burton v Forest U21"))
        self.assertGreater(senior["adjusted_probability"],cup["adjusted_probability"])


if __name__=="__main__":
    unittest.main(verbosity=2)
