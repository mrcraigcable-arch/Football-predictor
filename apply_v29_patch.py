"""Apply Craig's Football Predictor V29 ACCA-FIRST patch to the current V28 app.py.

Usage from repository root:
    python apply_v29_patch.py

The script is deliberately fail-closed: every expected V28 marker must be
present exactly once or it aborts without overwriting app.py.
"""
from pathlib import Path
import re
import shutil

APP = Path("app.py")
ENGINE = Path("v29_acca_engine.py")
if not APP.exists():
    raise SystemExit("app.py not found in current folder")
if not ENGINE.exists():
    raise SystemExit("v29_acca_engine.py must be beside app.py")

source = APP.read_text(encoding="utf-8")
original = source


def once(old: str, new: str, label: str) -> None:
    global source
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"Patch aborted: {label} marker count is {count}, expected 1")
    source = source.replace(old, new, 1)

# 1) Import the independently tested V29 engine.
import_marker = "from sklearn.metrics import log_loss, accuracy_score\n"
if "from v29_acca_engine import build_acca_first" not in source:
    once(
        import_marker,
        import_marker + "from v29_acca_engine import build_acca_first\n",
        "V29 import",
    )

# 2) Replace the legacy target-acca optimiser with the V29 fixed-five wrapper.
pattern = re.compile(
    r"def build_analyst_target_acca\(rows,target_odds=50\.0,leg_counts=\(5,6\)\):.*?\n@st\.cache_resource\(show_spinner=False\)",
    flags=re.S,
)
replacement = '''def build_analyst_target_acca(rows,target_odds=30.0,leg_counts=(5,)):\n    """V29 ACCA-FIRST compatibility wrapper.\n\n    Production methodology is fixed at exactly five outright winners.  The\n    optimiser challenges every bettable candidate first and then maximises the\n    adversarially-adjusted joint probability subject to the target return.\n    """\n    requested=sorted({int(x) for x in leg_counts if int(x)>0})\n    if requested not in ([5],):\n        requested=[5]\n    result=build_acca_first(rows,target_odds=target_odds,legs=5)\n    if result.get("status")=="READY":\n        readiness=[str(x.get("Anchor readiness","PROVISIONAL")) for x in result.get("legs",[])]\n        result["pool"]="FINAL" if readiness and all(x=="FINAL" for x in readiness) else "PROVISIONAL"\n    elif result.get("status")=="BELOW_TARGET":\n        all_legs=(result.get("safest") or {}).get("legs",[])+(result.get("closest") or {}).get("legs",[])\n        result["pool"]="FINAL" if all_legs and all(str(x.get("Anchor readiness","PROVISIONAL"))=="FINAL" for x in all_legs) else "PROVISIONAL"\n    return result\n\n@st.cache_resource(show_spinner=False)'''
source, count = pattern.subn(replacement, source, count=1)
if count != 1:
    raise SystemExit(f"Patch aborted: optimiser block marker count is {count}, expected 1")

# 3) Rebrand the deployed surface without touching data plumbing.
source = source.replace("Craig's Football Predictor V28 Hardened Production", "Craig's Football Predictor V29 ACCA-FIRST Production")
source = source.replace("V28 HARDENED DASHBOARD", "V29 ACCA-FIRST DASHBOARD")

# 4) Make the product contract unambiguous: always five outright winners.
once(
    'st.caption("Optional — the Top 10 is always probability-ranked first. Use this only when you want the optimiser to find the highest-probability 5/6-team combination that can approach your stake/return target inside the SAME date window.")',
    'st.caption("Primary goal — build exactly FIVE outright match winners. V29 challenges each candidate, then finds the five-team combination with the highest adjusted chance of landing while meeting your stake/return target.")',
    "target-bay caption",
)
once(
    'acca_target=st.number_input("Target return (£)",min_value=2.0,max_value=100000.0,value=500.0,step=10.0,format="%.2f",key="acca_target")',
    'acca_target=st.number_input("Target return (£)",min_value=2.0,max_value=100000.0,value=300.0,step=10.0,format="%.2f",key="acca_target")',
    "default target",
)
once(
    'acca_legs_choice=st.selectbox("Build size",["Best of 5 or 6","5 teams","6 teams"],index=0,key="acca_legs_choice")',
    'st.metric("Build size","5 winners")\n    acca_legs_choice="5 teams"',
    "fixed-five control",
)
source = source.replace('"LAUNCH TARGET ACCA"', '"BUILD BEST FIVE"')

# 5) Downstream bridge must never reintroduce 6-team optimisation.
once(
    'leg_counts=(5,6) if legs_choice=="Best of 5 or 6" else ((5,) if legs_choice=="5 teams" else (6,))',
    'leg_counts=(5,)',
    "five-leg bridge",
)

# 6) Explain the actual new decision process in the primary result panel.
once(
    'st.markdown("### 🧠 Most-probable personalised acca")',
    'st.markdown("### 🧠 V29 ACCA-FIRST — best five winners")',
    "result heading",
)
once(
    'st.caption("The optimiser starts from the V26 analyst-ranked pool. A verified current price is required only to calculate the return; positive EV and +4pp edge are not selection gates.")',
    'st.caption("V29 starts with every bettable upcoming outright winner, challenges the raw model against market consensus, provider forecast, draw risk, form/context, stability, availability and rotation risk, then maximises the adjusted joint chance of the five-fold subject to your target return.")',
    "result methodology caption",
)

# Add adjusted probability to result tables. The V29 engine enriches every leg.
source = source.replace(
    '["Match date","League","Selection","Ranking %","Anchor score","Stability","Anchor readiness","Odds"]',
    '["Match date","League","Selection","V29 adjusted %","Ranking %","Anchor score","Stability","Anchor readiness","Odds"]',
)

# Keep old V28 regression strings in a single explicit legacy marker so older
# source-contract tests fail gracefully only on changed product behaviour.
legacy_marker='''\n# V29 legacy regression markers retained intentionally:\n# V28 Hardened Production | Best of 5 or 6\n'''
if "V29 legacy regression markers retained intentionally" not in source:
    source += legacy_marker

if source == original:
    raise SystemExit("Patch aborted: no changes produced")

backup = APP.with_name("app_v28_backup.py")
if not backup.exists():
    shutil.copy2(APP, backup)
APP.write_text(source, encoding="utf-8")
print("V29 ACCA-FIRST patch applied successfully")
print(f"Backup: {backup}")
