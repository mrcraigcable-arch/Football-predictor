import ast
import py_compile
import unittest
from datetime import date, datetime, timezone
from itertools import combinations
from math import isfinite
from pathlib import Path

import numpy as np
import pandas as pd


APP = Path(__file__).with_name("app.py")
SOURCE = APP.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def load_functions(*names, extra=None):
    wanted = set(names)
    nodes = [n for n in TREE.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    module = ast.Module(body=nodes, type_ignores=[])
    scope = {
        "np": np,
        "pd": pd,
        "date": date,
        "datetime": datetime,
        "timezone": timezone,
        "combinations": combinations,
        "isfinite": isfinite,
    }
    if extra:
        scope.update(extra)
    exec(compile(module, str(APP), "exec"), scope)
    return scope


class ProductionContractTests(unittest.TestCase):
    def test_compiles(self):
        py_compile.compile(str(APP), doraise=True)

    def test_no_duplicate_top_level_functions(self):
        names = [n.name for n in TREE.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        self.assertEqual(len(names), len(set(names)))

    def test_v28_brand(self):
        self.assertIn("V28 HARDENED", SOURCE)
        self.assertIn("V28 Hardened Production", SOURCE)

    def test_date_range_controls_retained(self):
        self.assertIn('"Custom date range"', SOURCE)
        self.assertIn('"SATURDAY // RUN"', SOURCE)
        self.assertIn('"NEXT 7 DAYS // RUN"', SOURCE)

    def test_stake_and_target_controls_retained(self):
        self.assertIn('"Stake (£)"', SOURCE)
        self.assertIn('"Target return (£)"', SOURCE)
        self.assertIn('"Best of 5 or 6"', SOURCE)

    def test_probability_ranked_top_ten_retained(self):
        self.assertIn("Top 10 strongest teams", SOURCE)
        self.assertIn('.sort_values("Ranking %",ascending=False).head(10)', SOURCE)

    def test_strict_verification_contract(self):
        self.assertIn('ext=="FULL VERIFIED"', SOURCE)
        self.assertIn('readiness=="FINAL"', SOURCE)
        self.assertIn("SCREEN VERIFIED", SOURCE)

    def test_live_rows_do_not_enter_prematch_ledger(self):
        self.assertIn('state=="UPCOMING"', SOURCE)
        self.assertIn('"Snapshot type":"PREMATCH"', SOURCE)

    def test_live_market_blender_guard(self):
        self.assertIn('if str(r.get("Game state","")).upper()=="LIVE":', SOURCE)

    def test_recommendation_versioning(self):
        self.assertIn('"Lifecycle"]="SUPERSEDED"', SOURCE)

    def test_competition_trust_is_bayesian_and_capped(self):
        self.assertIn("prior_strength=50.0", SOURCE)
        self.assertIn("np.clip(gap_pp*.10,-3,3)", SOURCE)
        self.assertIn("if len(g)<20: continue", SOURCE)

    def test_api_pagination_and_safe_cap_present(self):
        self.assertIn("def _api_football_get_all", SOURCE)
        self.assertIn("safe cap is {max_pages}", SOURCE)

    def test_api_quota_guard_present(self):
        self.assertIn("def _api_budget_guard", SOURCE)
        self.assertIn("daily reserve reached", SOURCE)
        self.assertIn("minute reserve reached", SOURCE)

    def test_api_429_handling_present(self):
        self.assertIn("r.status_code==429", SOURCE)
        self.assertIn("Retry-After", SOURCE)

    def test_efl_trophy_retained(self):
        self.assertIn('"EFL Trophy":46', SOURCE)

    def test_provider_fixture_id_settlement_retained(self):
        self.assertIn('Provider fixture ID', SOURCE)
        self.assertIn('"fixtures",{"id":int(float(pfid))}', SOURCE)

    def test_lazy_odds_api_contract(self):
        self.assertIn("if not games: continue", SOURCE)
        self.assertLess(SOURCE.index("if not games: continue"), SOURCE.index("odds_fetch(odds_key", SOURCE.index("if not games: continue")))

    def test_pandas_empty_frame_hardening(self):
        self.assertIn("if led.empty:", SOURCE)
        self.assertIn("if hist.empty:", SOURCE)


class PureFunctionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope = load_functions(
            "_v18_number",
            "football_season_year_for_date",
            "_provider_fixture_state",
            "_api_football_get_all",
        )

    def test_dynamic_season_january(self):
        self.assertEqual(self.scope["football_season_year_for_date"]("2027-01-10"), 2026)

    def test_dynamic_season_september(self):
        self.assertEqual(self.scope["football_season_year_for_date"]("2026-09-10"), 2026)

    def test_provider_completed_state(self):
        fx = {"fixture": {"status": {"short": "FT"}}}
        self.assertEqual(self.scope["_provider_fixture_state"](fx), "FINAL")

    def test_provider_postponed_state(self):
        fx = {"fixture": {"status": {"short": "PST"}}}
        self.assertEqual(self.scope["_provider_fixture_state"](fx), "UNAVAILABLE")

    def test_provider_live_state(self):
        fx = {"fixture": {"status": {"short": "2H"}}}
        self.assertEqual(self.scope["_provider_fixture_state"](fx), "LIVE")

    def test_pagination_collects_all_pages(self):
        calls = []

        def fake_meta(path, params, api_key):
            page = params["page"]
            calls.append(page)
            return {"response": [page], "paging": {"current": page, "total": 3}}

        self.scope["_api_football_get_meta"] = fake_meta
        got = self.scope["_api_football_get_all"]("fixtures", {"league": 46}, "key", max_pages=4)
        self.assertEqual(got, [1, 2, 3])
        self.assertEqual(calls, [1, 2, 3])

    def test_pagination_rejects_oversized_result(self):
        def fake_meta(path, params, api_key):
            return {"response": [1], "paging": {"current": 1, "total": 13}}

        self.scope["_api_football_get_meta"] = fake_meta
        with self.assertRaisesRegex(RuntimeError, "safe cap"):
            self.scope["_api_football_get_all"]("fixtures", {}, "key", max_pages=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
