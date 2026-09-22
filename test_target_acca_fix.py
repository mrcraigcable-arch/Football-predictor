import ast
import unittest
from itertools import combinations
from pathlib import Path

import numpy as np


APP = Path(__file__).with_name("app.py")
TREE = ast.parse(APP.read_text(encoding="utf-8"))


def load_builder():
    node = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "build_analyst_target_acca")
    scope = {"np": np, "combinations": combinations}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(APP), "exec"), scope)
    return scope["build_analyst_target_acca"]


def row(name, price=1.8, readiness="PROVISIONAL — provider season not covered", state="UPCOMING", fixture_status=""):
    return {
        "Match": name,
        "Pick": "HOME",
        "Game state": state,
        "Best market odds": price,
        "Ranking %": 60,
        "Anchor score": 70,
        "Anchor readiness": readiness,
        "Market diagnostic": {"stage": "accepted"},
        "Fixture status": fixture_status,
    }


class TargetAccaFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = staticmethod(load_builder())

    def test_provisional_fallback_builds_five(self):
        result = self.build([row(str(i)) for i in range(6)], target_odds=10, leg_counts=(5,))
        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["pool"], "PROVISIONAL")
        self.assertEqual(len(result["legs"]), 5)

    def test_five_finals_exclude_provisionals(self):
        rows = [row(f"F{i}", readiness="FINAL") for i in range(5)] + [row("P", price=5)]
        result = self.build(rows, target_odds=10, leg_counts=(5,))
        self.assertEqual(result["pool"], "FINAL")
        self.assertTrue(all(x["Anchor readiness"] == "FINAL" for x in result["legs"]))

    def test_unreachable_target_returns_alternatives(self):
        result = self.build([row(str(i), price=1.2) for i in range(6)], target_odds=100, leg_counts=(5,))
        self.assertEqual(result["status"], "BELOW_TARGET")
        self.assertEqual(len(result["safest"]["legs"]), 5)
        self.assertEqual(len(result["closest"]["legs"]), 5)

    def test_postponed_fixture_is_hard_excluded(self):
        rows = [row(str(i)) for i in range(5)] + [row("postponed", price=10, fixture_status="POSTPONED")]
        result = self.build(rows, target_odds=10, leg_counts=(5,))
        self.assertEqual(result["status"], "READY")
        self.assertNotIn("postponed", [x["Match"] for x in result["legs"]])

    def test_insufficient_explains_both_pools(self):
        result = self.build([row("F", readiness="FINAL"), row("P")], target_odds=10, leg_counts=(5,))
        self.assertEqual(result["status"], "INSUFFICIENT")
        self.assertIn("1 FINAL and 1 price-verified PROVISIONAL", result["reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
