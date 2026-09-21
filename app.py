import streamlit as st
import pandas as pd
import numpy as np
import requests
import html
from collections import defaultdict, deque
from fractions import Fraction
from itertools import combinations
from math import isfinite, log
import math
import copy
import json
import time
from datetime import date, datetime, timezone, timedelta
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score



# --- V18 DECISION / SAFETY ENGINE (embedded, preserved in V22) ---------------------------
# Kept inside app.py so Streamlit cannot deploy the UI and selection engine from
# different commits. This is the V18 scoring/acca logic preserved in V22.
WEIGHTS = {
    "model_probability": 35,
    "goals_profile": 20,
    "recent_venue_form": 20,
    "market_value": 0,
    "availability_evidence": 10,
    "opponent_strength": 10,
    "supporting_indicators": 5,
}

def _v18_number(value, default=None):
    try:
        value=float(value)
        return value if isfinite(value) else default
    except (TypeError, ValueError):
        return default

def _v18_clip(value, low=0.0, high=100.0):
    return max(low,min(high,float(value)))

LIVE_ACTION_WINDOW_MINUTES = 150

def _kickoff_state(kickoff_iso, live_window_minutes=LIVE_ACTION_WINDOW_MINUTES):
    """Return UPCOMING, LIVE, STALE or UNKNOWN from the verified kickoff.

    LIVE means the scheduled kickoff has passed but the match is still inside a
    conservative action window. The completed-results feed remains the primary
    source for FINAL status; STALE is a safety backstop for feed lag so a match
    cannot remain recommendable for hours after it should have finished.
    """
    if not kickoff_iso:
        return {"state":"UNKNOWN","minutes":None,"label":"Kickoff time unavailable"}
    try:
        kickoff=pd.to_datetime(kickoff_iso,utc=True,errors="coerce")
        if pd.isna(kickoff):
            return {"state":"UNKNOWN","minutes":None,"label":"Kickoff time unavailable"}
        now=pd.Timestamp.now(tz="UTC")
        delta=(now-kickoff).total_seconds()/60.0
        if delta < 0:
            mins_to=max(0,int(round(-delta)))
            return {"state":"UPCOMING","minutes":mins_to,"label":f"Starts in ~{mins_to} min"}
        mins_since=max(0,int(delta))
        if mins_since <= int(live_window_minutes):
            return {"state":"LIVE","minutes":mins_since,"label":f"LIVE • started ~{mins_since} min ago"}
        return {"state":"STALE","minutes":mins_since,"label":""}
    except Exception:
        return {"state":"UNKNOWN","minutes":None,"label":"Kickoff time unavailable"}

def _kickoff_is_future(kickoff_iso):
    return _kickoff_state(kickoff_iso)["state"] == "UPCOMING"

def _kickoff_is_actionable(kickoff_iso):
    return _kickoff_state(kickoff_iso)["state"] in ("UPCOMING","LIVE")

def _v18_side(context,pick):
    if not isinstance(context,dict): return {},{}
    if pick=="HOME": return context.get("home",{}) or {}, context.get("away",{}) or {}
    if pick=="AWAY": return context.get("away",{}) or {}, context.get("home",{}) or {}
    return {},{}

def score_selection(row):
    """Transparent V18 evidence score with fail-closed uncertainty penalties."""
    pick=str(row.get("Pick","")).upper()
    context=row.get("Context") if isinstance(row.get("Context"),dict) else {}
    selected,opponent=_v18_side(context,pick)
    concerns=[]; positives=[]

    confidence=_v18_number(row.get("Confidence %"),0.0)
    model_score=_v18_clip((confidence-45.0)/35.0*100.0)
    if confidence>=68: positives.append(f"Model win probability {confidence:.1f}%")
    elif confidence<60: concerns.append(f"Model probability only {confidence:.1f}%")

    hp=_v18_number(context.get("xg_like_home")); ap=_v18_number(context.get("xg_like_away"))
    if pick=="HOME" and hp is not None and ap is not None: goals_advantage=hp-ap
    elif pick=="AWAY" and hp is not None and ap is not None: goals_advantage=ap-hp
    else: goals_advantage=None
    goals_score=_v18_clip(50+32*goals_advantage) if goals_advantage is not None else 0.0
    if goals_advantage is None: concerns.append("Scoring-rate profile unavailable")
    elif goals_advantage>=0.35: positives.append(f"Scoring-rate advantage {goals_advantage:+.2f}")
    elif goals_advantage<0: concerns.append(f"Scoring-rate profile opposes pick ({goals_advantage:+.2f})")

    selected_ppg=_v18_number(selected.get("ppg")); opponent_ppg=_v18_number(opponent.get("ppg"))
    selected_venue=_v18_number(selected.get("venue_ppg"),selected_ppg)
    opponent_venue=_v18_number(opponent.get("venue_ppg"),opponent_ppg)
    if None not in (selected_ppg,opponent_ppg,selected_venue,opponent_venue):
        form_advantage=.65*(selected_ppg-opponent_ppg)+.35*(selected_venue-opponent_venue)
        form_score=_v18_clip(50+24*form_advantage)
        if form_advantage>=.5: positives.append(f"Recent/venue PPG advantage {form_advantage:+.2f}")
        elif form_advantage<=-.35: concerns.append(f"Recent/venue form opposes pick ({form_advantage:+.2f})")
    else:
        form_advantage=None; form_score=0.0; concerns.append("Recent home/away split unavailable")

    edge=_v18_number(row.get("Edge pp")); ev=_v18_number(row.get("EV %"))
    if edge is None or ev is None:
        value_score=0.0; concerns.append("Current bookmaker value not verified")
    else:
        value_score=_v18_clip(35+5.0*edge+1.2*ev)
        if edge>=4 and ev>0: positives.append(f"Verified edge {edge:+.1f}pp; EV {ev:+.1f}%")
        else: concerns.append(f"Price fails preferred value margin ({edge:+.1f}pp edge)")

    injuries=str(context.get("injuries","")).upper()
    availability_verified=bool(injuries and "UNAVAILABLE" not in injuries)
    availability_score=100.0 if availability_verified else 20.0
    if not availability_verified: concerns.append("Lineups/injuries not independently verified")

    selected_pos=_v18_number(selected.get("position")); opponent_pos=_v18_number(opponent.get("position"))
    opponent_score=_v18_clip(50+6*(opponent_pos-selected_pos)) if selected_pos is not None and opponent_pos is not None else 35.0
    clean_sheet=_v18_number(selected.get("clean_sheet")); opponent_gf=_v18_number(opponent.get("gf"))
    indicators_score=_v18_clip(.75*clean_sheet+25*(1.6-opponent_gf)) if clean_sheet is not None and opponent_gf is not None else 35.0

    components={
        "model_probability":round(model_score,1),"goals_profile":round(goals_score,1),
        "recent_venue_form":round(form_score,1),"market_value":round(value_score,1),
        "availability_evidence":round(availability_score,1),"opponent_strength":round(opponent_score,1),
        "supporting_indicators":round(indicators_score,1),
    }
    base_score=sum(components[k]*w/100.0 for k,w in WEIGHTS.items())
    penalties=[]
    validation=str(row.get("Validation","")); decision=str(row.get("Decision","")); integrity=str(row.get("Market integrity",""))
    played=min(_v18_number(selected.get("played"),0),_v18_number(opponent.get("played"),0))
    if played<5: penalties.append((6,"Current-season sample below five matches"))
    if "FALLBACK" in validation: penalties.append((8,"League calibration uses a safety fallback"))
    elif "APPROVED RAW" in validation: penalties.append((3,"League evidence favours an uncalibrated model"))
    if not availability_verified: penalties.append((5,"Availability evidence missing"))
    if integrity and integrity!="OK": penalties.append((8,"Bookmaker prices disagree"))
    # V21: value/EV is no longer an eligibility gate. Keep market-quality
    # evidence visible, but do not punish a strong win prediction simply because
    # the bookmaker price offers little model edge.
    if decision=="PREDICTION ONLY": penalties.append((10,"No matched current market"))
    if pick=="DRAW": penalties.append((20,"Draws are excluded from the acca shortlist"))
    if form_advantage is not None and form_advantage<=-.35: penalties.append((7,"Recent form conflicts with the model pick"))
    if goals_advantage is not None and goals_advantage<0: penalties.append((6,"Scoring profile conflicts with the model pick"))
    if str(row.get("Game state","")).upper()=="LIVE":
        concerns.append("Match has started; the model evidence is pre-match and the current score is not ingested")
    penalty_total=sum(p for p,_ in penalties)
    final_score=round(_v18_clip(base_score-penalty_total),1)
    outright=pick in ("HOME","AWAY")
    if outright and final_score>=70: tier="ELITE"
    elif outright and final_score>=60: tier="STRONG"
    elif outright and final_score>=48: tier="WATCHLIST"
    else: tier="REJECT"
    return {"score":final_score,"base_score":round(base_score,1),"tier":tier,"components":components,
            "penalties":[{"points":p,"reason":r} for p,r in penalties],"penalty_total":penalty_total,
            "positives":positives,"concerns":concerns,"goals_profile_label":"scoring-rate proxy (not provider xG)",
            "acca_eligible":tier=="ELITE" and decision=="BET" and outright}

def build_acca(rows,target_odds=50.0,min_legs=4,max_legs=7):
    candidates=[]
    for row in rows:
        audit=row.get("V18 audit") or score_selection(row)
        price=_v18_number(row.get("Best market odds"))
        if audit.get("acca_eligible") and price is not None and price>1.01:
            candidates.append((row,audit,price))
    candidates.sort(key=lambda x:x[1]["score"],reverse=True); candidates=candidates[:14]
    if len(candidates)<min_legs:
        return {"status":"INSUFFICIENT","legs":[],"combined_odds":None,"reason":f"Only {len(candidates)} fully verified elite selection(s); {min_legs} required."}
    best=None; target_log=log(max(target_odds,1.01))
    for count in range(min_legs,min(max_legs,len(candidates))+1):
        for combo in combinations(candidates,count):
            combined=1.0; scores=[]
            for _,audit,price in combo: combined*=price; scores.append(audit["score"])
            objective=abs(log(combined)-target_log)+.002*(100-sum(scores)/len(scores))
            if best is None or objective<best[0]: best=(objective,combo,combined)
    _,combo,combined=best
    return {"status":"READY","legs":[x[0] for x in combo],"combined_odds":round(combined,2),"reason":"Closest eligible combination to the target price."}

def build_best_chance_acca(rows,target_odds=50.0,leg_counts=(5,6),min_books=3):
    counts=sorted({int(x) for x in leg_counts if int(x)>0}); candidates=[]
    for row in rows:
        audit=row.get("V18 audit") or score_selection(row); pick=str(row.get("Pick","")).upper()
        price=_v18_number(row.get("Best market odds")); prob=_v18_number(row.get("Confidence %")); books=_v18_number(row.get("Bookmakers"),0)
        diag=row.get("Market diagnostic") or {}; accepted=isinstance(diag,dict) and diag.get("stage")=="accepted"
        game_state=str(row.get("Game state") or _kickoff_state(row.get("Kickoff ISO"))["state"]).upper()
        live_market_prob=_v18_number(row.get("Market fair %"))
        ranking_prob=(live_market_prob/100.0) if game_state=="LIVE" and live_market_prob is not None and 0<live_market_prob<100 else (prob/100.0 if prob is not None else None)
        # V24 probability-first rule: do not exclude a bettable favourite because
        # it lacks positive EV, a +4pp edge, eight bookmakers, or perfect market
        # agreement. Those are advisory diagnostics only. The hard gates are:
        # exact fixture match, a valid current 1X2 price, and an actionable game.
        if pick in ("HOME","AWAY") and price is not None and price>1.01 and ranking_prob is not None and 0<ranking_prob<1 and books>=1 and game_state in ("UPCOMING","LIVE") and accepted:
            candidates.append({"row":row,"audit":audit,"price":price,"probability":ranking_prob,
                               "probability_source":"live market fair" if game_state=="LIVE" and live_market_prob is not None else "pre-match model"})
    by_prob=sorted(candidates,key=lambda x:x["probability"],reverse=True)[:20]
    by_eff=sorted(candidates,key=lambda x:x["probability"]*(x["price"]**.5),reverse=True)[:12]
    pool=[]; seen=set()
    for item in by_prob+by_eff:
        ident=(item["row"].get("League"),item["row"].get("Match"))
        if ident not in seen: seen.add(ident); pool.append(item)
    valid=[c for c in counts if c<=len(pool)]
    if not valid:
        need=min(counts) if counts else 5
        return {"status":"INSUFFICIENT","legs":[],"combined_odds":None,"joint_probability":None,"reason":f"Only {len(pool)} selections have fully matched current prices; {need} required."}
    target=max(float(target_odds),1.01); best_ready=None; best_below=None
    for count in valid:
        for combo in combinations(pool,count):
            combined=1.0; joint=1.0; scores=[]
            for item in combo: combined*=item["price"]; joint*=item["probability"]; scores.append(_v18_number(item["audit"].get("score"),0))
            mean=sum(scores)/len(scores)
            if combined>=target:
                rank=(joint,mean,-combined)
                if best_ready is None or rank>best_ready[0]: best_ready=(rank,combo,combined,joint)
            else:
                rank=(combined,joint,mean)
                if best_below is None or rank>best_below[0]: best_below=(rank,combo,combined,joint)
    chosen=best_ready or best_below
    if chosen is None: return {"status":"INSUFFICIENT","legs":[],"combined_odds":None,"joint_probability":None,"reason":"No valid combination could be formed."}
    _,combo,combined,joint=chosen; ready=best_ready is not None
    return {"status":"READY" if ready else "BELOW_TARGET","legs":[x["row"] for x in combo],"combined_odds":round(combined,2),"joint_probability":round(joint*100,2),"target_odds":round(target,2),"reason":"Highest model-estimated joint probability among combinations reaching the target. The multiplication assumes match independence; V23 separately audits Safest-Five historical hit rates." if ready else "No requested-size combination reaches the target with verified prices; this is the closest available return."}
# --- END V18 ENGINE -------------------------------------------------------------

st.set_page_config(page_title="Craig's Football Predictor V28 Hardened Production", page_icon="📈", layout="wide")



# --- V14 VISUAL SYSTEM: mobile-first neon dashboard ---
st.markdown("""
<style>
:root{
 --bg:#03111d; --panel:#071b2b; --panel2:#0a2235; --line:#173e58;
 --cyan:#18a9ff; --green:#00ef83; --amber:#ffab00; --red:#ff334f;
 --text:#f4f8fc; --muted:#9db0c4;
}
.stApp{
 background:
 radial-gradient(900px 380px at 55% -120px, rgba(0,239,131,.17), transparent 55%),
 linear-gradient(180deg,#041522 0%,#020b13 100%);
 color:var(--text);
}
.block-container{max-width:980px;padding-top:.65rem;padding-bottom:6rem}
h1,h2,h3{letter-spacing:-.025em}
h1{font-weight:850}
h2,h3{color:#f8fbff}
[data-testid="stHeader"]{background:rgba(2,11,19,.78)}
[data-testid="stToolbar"]{right:.5rem}

/* Main section containers */
div[data-testid="stVerticalBlockBorderWrapper"]{
 border:1px solid var(--line)!important;border-radius:20px!important;
 background:linear-gradient(145deg,rgba(9,31,48,.96),rgba(4,17,29,.96))!important;
 box-shadow:0 8px 28px rgba(0,0,0,.22);
}
div[data-testid="stExpander"]{
 border:1px solid #23455e!important;border-radius:17px!important;
 background:linear-gradient(145deg,#091a29,#06121e)!important;
 overflow:hidden;
}
div[data-testid="stExpander"] details summary{
 min-height:64px;padding:.45rem .7rem;font-weight:700
}

/* Inputs: touch-safe and compact */
div[data-testid="stNumberInput"] input,
div[data-testid="stDateInput"] input,
div[data-baseweb="select"]>div{
 background:#101f30!important;border:1px solid #244c67!important;
 border-radius:12px!important;color:#fff!important;
}
div[data-testid="stNumberInput"] button{
 background:#12334b!important;color:#fff!important;border-color:#285b7a!important;
 min-width:44px!important;min-height:44px!important
}
div[data-testid="stNumberInput"] button:hover{border-color:var(--green)!important}
label[data-testid="stWidgetLabel"] p{font-weight:650;color:#dce8f3}

/* Buttons */
.stButton>button{
 width:100%;min-height:52px;border-radius:13px;border:1px solid #0af08b;
 background:linear-gradient(90deg,#00d97c,#00f29a);
 color:#00170d;font-weight:850;font-size:1.05rem;
 box-shadow:0 0 22px rgba(0,239,131,.20);
}
.stButton>button:hover{border-color:#64ffc0;color:#00170d;filter:brightness(1.05)}

/* Metrics */
div[data-testid="stMetric"]{
 background:linear-gradient(145deg,#092239,#061726);
 border:1px solid #17679a;border-radius:16px;padding:14px 15px;
 min-height:112px;
}
div[data-testid="stMetricLabel"]{color:var(--muted)}
div[data-testid="stMetricValue"]{font-weight:850}

/* Status messages */
div[data-testid="stAlert"]{border-radius:14px;border-left-width:5px}

/* Tables */
[data-testid="stDataFrame"]{border-radius:16px;overflow:hidden;border:1px solid var(--line)}

/* visual badges reusable from markdown */
.v14-brand{
 border:1px solid #0b8057;border-radius:18px;padding:16px 18px;margin:4px 0 16px;
 background:linear-gradient(135deg,rgba(0,239,131,.12),rgba(5,29,46,.92) 45%,rgba(10,67,95,.35));
 box-shadow:0 8px 30px rgba(0,0,0,.20)
}
.v14-brandline{display:flex;align-items:center;gap:12px}
.v14-logo{font-size:2rem;filter:drop-shadow(0 0 8px rgba(0,239,131,.45))}
.v14-title{font-size:1.55rem;font-weight:900;line-height:1.05}
.v14-title b{color:var(--green)}
.v14-sub{color:#b3c5d6;margin-top:6px;font-size:.9rem}
.v14-chip{display:inline-block;float:right;border:1px solid #00c86e;border-radius:10px;
 padding:5px 11px;color:#00f18a;font-weight:900;background:#06251b}
.v14-section{
 margin:18px 0 10px;padding:10px 13px;border-left:4px solid var(--cyan);
 background:linear-gradient(90deg,rgba(20,158,255,.13),transparent);
 border-radius:10px;font-weight:850;font-size:1.22rem
}
.v14-key{padding:10px 13px;border-radius:13px;background:#071a29;border:1px solid #173e58;
 margin:8px 0 16px;color:#d9e7f3}
.green{color:var(--green)} .amber{color:var(--amber)} .red{color:var(--red)} .blue{color:#2b9cff}

/* Desktop parameter row; mobile stays comfortable */
@media(min-width:700px){
 div[data-testid="stHorizontalBlock"]{gap:.75rem}
}
@media(max-width:699px){
 .block-container{padding-left:.85rem;padding-right:.85rem}
 .v14-title{font-size:1.35rem}
 h1{font-size:2rem}
 h2{font-size:1.65rem}
 div[data-testid="stMetric"]{min-height:96px}
}

/* bottom visual nav */
.v14-nav{
 position:fixed;left:0;right:0;bottom:0;z-index:999;
 display:flex;justify-content:space-around;align-items:center;
 padding:10px 8px calc(10px + env(safe-area-inset-bottom));
 background:rgba(3,17,29,.96);border-top:1px solid #17425e;
 backdrop-filter:blur(14px);box-shadow:0 -8px 25px rgba(0,0,0,.30)
}
.v14-nav span{color:#9db5c9;font-size:.78rem;text-align:center;min-width:22%}
.v14-nav .active{color:var(--green);font-weight:800}
</style>
<div class="v14-brand">
 <span class="v14-chip">V28 HARDENED</span>
 <div class="v14-brandline"><span class="v14-logo">📈</span>
 <div><div class="v14-title">Craig's Football <b>Predictor</b></div>
 <div class="v14-sub">Nitrous-inspired dashboard styling with evidence-backed football decisions. • Real market comparison</div></div></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
:root { --green:#22e682; --blue:#35a7ff; --amber:#ffb000; --red:#ff4057; --panel:#111b28; --muted:#94a3b8; }
.stApp { background: radial-gradient(circle at 75% 0%, #10243a 0%, #07111c 32%, #050b12 72%); color:#f8fafc; }
.block-container { max-width:920px; padding-top:1.1rem; padding-bottom:4rem; }
h1,h2,h3 { letter-spacing:-.02em; }
[data-testid="stMetric"] { background:linear-gradient(145deg,rgba(17,27,40,.98),rgba(8,17,28,.98)); border:1px solid #24364a; border-radius:18px; padding:14px 16px; box-shadow:0 10px 30px rgba(0,0,0,.18); }
[data-testid="stMetricLabel"] { color:#aab8c8; }
[data-testid="stMetricValue"] { color:#f8fafc; }
[data-testid="stExpander"] { background:linear-gradient(145deg,rgba(17,27,40,.96),rgba(8,17,28,.96)); border:1px solid #26384b; border-radius:17px; overflow:hidden; margin-bottom:10px; }
div.stButton > button { border-radius:14px; min-height:48px; border:1px solid #2b435b; font-weight:750; }
div.stButton > button[kind="primary"] { background:linear-gradient(90deg,#18d977,#24ec92); color:#04120b; border:0; }
[data-testid="stAlert"] { border-radius:16px; }
.hero { padding:20px 22px; border-radius:22px; border:1px solid rgba(34,230,130,.7); background:linear-gradient(110deg,rgba(8,76,60,.7),rgba(7,17,28,.92)); box-shadow:0 0 35px rgba(34,230,130,.08); margin:8px 0 18px; }
.hero-title { font-size:1.55rem; font-weight:850; }
.hero-sub { color:#b8c6d6; margin-top:4px; }
.brand { display:flex; align-items:center; gap:14px; margin-bottom:8px; }
.brand-icon { font-size:2.3rem; background:#071827; border-radius:16px; padding:8px 12px; }
.brand-name { font-size:1.65rem; font-weight:900; line-height:1.0; }
.brand-sub { color:#93a4b8; margin-top:5px; }
.vbadge { display:inline-block; color:#22e682; border:1px solid #22e682; border-radius:12px; padding:4px 10px; font-weight:850; margin-left:8px; }
.section-note { color:#94a3b8; margin-top:-8px; margin-bottom:14px; }
@media (max-width:640px){
 .block-container { padding-left:1rem; padding-right:1rem; }
 .brand-name { font-size:1.45rem; }
 [data-testid="stMetricValue"] { font-size:1.75rem; }
}
</style>
<div class="brand">
  <div class="brand-icon">📈</div>
  <div><div class="brand-name">Craig's Football Predictor <span class="vbadge">V28 HARDENED</span></div>
  <div class="brand-sub">Data. Discipline. Evidence-backed decisions.</div></div>
</div>
<div class="hero"><div class="hero-title">🏆 Smarter football predictions</div>
<div class="hero-sub">Historical modelling + current bookmaker consensus + fail-closed verification.</div></div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* V17.1 FINISH — presentation-only layer. Prediction and validation logic unchanged. */
:root{--vbg:#030912;--vpanel:#081522;--vpanel2:#0b1d2d;--vline:#16354b;--vgreen:#20e884;--vblue:#22a8ff;--vmuted:#8ea4b8}
.stApp{background:radial-gradient(700px 330px at 50% -100px,rgba(32,232,132,.16),transparent 58%),linear-gradient(180deg,#06131f 0%,#020811 72%)!important}
.block-container{max-width:860px!important;padding-top:.55rem!important;padding-bottom:6.5rem!important}
[data-testid="stHeader"]{background:rgba(3,9,18,.70)!important;backdrop-filter:blur(16px)}
.v14-brand{position:relative;overflow:hidden;border:1px solid rgba(32,232,132,.38)!important;border-radius:24px!important;padding:18px 19px!important;background:linear-gradient(135deg,rgba(11,40,47,.96),rgba(5,18,30,.98))!important;box-shadow:0 16px 45px rgba(0,0,0,.34),inset 0 1px 0 rgba(255,255,255,.04)!important}
.v14-brand:after{content:"";position:absolute;width:170px;height:170px;border-radius:50%;right:-70px;top:-95px;background:rgba(32,232,132,.10);filter:blur(4px)}
.v14-chip,.vbadge{border:1px solid rgba(32,232,132,.55)!important;background:rgba(9,52,39,.72)!important;color:#50f3a5!important}
.hero{display:none!important}
.v14-title,.brand-name{font-weight:900!important}.v14-sub,.brand-sub{color:#91a8bc!important}
/* compact polished controls */
div[data-baseweb="select"]>div,[data-testid="stDateInput"] input,[data-testid="stNumberInput"] input{background:#0a1826!important;border-color:#1c3b52!important}
.stButton>button{min-height:58px!important;border-radius:18px!important;font-weight:900!important;letter-spacing:.01em!important;box-shadow:0 10px 30px rgba(32,232,132,.16)!important}
/* metric tiles become compact dashboard cells */
div[data-testid="stMetric"]{min-height:92px!important;padding:12px 14px!important;border:1px solid #153d59!important;border-radius:18px!important;background:linear-gradient(145deg,#0a2134,#061521)!important;box-shadow:none!important}
div[data-testid="stMetricLabel"] p{font-size:.82rem!important;color:#8fa8bd!important}
div[data-testid="stMetricValue"]{font-size:1.65rem!important;line-height:1.1!important}
/* match cards */
div[data-testid="stExpander"]{border:1px solid #1b3c53!important;border-radius:20px!important;background:linear-gradient(145deg,#0a1927,#06111c)!important;box-shadow:0 12px 30px rgba(0,0,0,.18)!important;margin-bottom:12px!important}
div[data-testid="stExpander"] details summary{min-height:72px!important;padding:.6rem .8rem!important}
div[data-testid="stExpander"] details summary p{font-weight:800!important;line-height:1.45!important}
[data-testid="stAlert"]{border-radius:18px!important;padding:14px 16px!important}
[data-testid="stDataFrame"]{border:1px solid #17384f!important;border-radius:18px!important;background:#071521!important}
/* section headings */
h2{font-size:1.55rem!important;margin-top:1.25rem!important;margin-bottom:.65rem!important}h3{font-size:1.18rem!important}
hr{border-color:#173247!important}
/* download */
[data-testid="stDownloadButton"] button{min-height:52px!important;border-radius:16px!important;background:#0a1927!important;border:1px solid #27475d!important;color:#e9f4fc!important}
/* mobile: two-column dashboard instead of giant one-column tiles */
@media(max-width:699px){
 .block-container{padding-left:.72rem!important;padding-right:.72rem!important}
 .v14-brand{margin-top:0!important}.v14-title{font-size:1.28rem!important}.v14-logo{font-size:1.65rem!important}
 div[data-testid="stHorizontalBlock"]{gap:.48rem!important;flex-wrap:wrap!important}
 div[data-testid="stHorizontalBlock"]>div[data-testid="stColumn"]{min-width:calc(50% - .3rem)!important;flex:1 1 calc(50% - .3rem)!important}
 div[data-testid="stMetric"]{min-height:84px!important;padding:10px 12px!important}
 div[data-testid="stMetricValue"]{font-size:1.48rem!important}
 div[data-testid="stMetricLabel"] p{font-size:.76rem!important}
 div[data-testid="stExpander"] details summary{min-height:68px!important}
 h2{font-size:1.42rem!important}
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* --- V28 HARDENED EDITION VISUAL LAYER --------------------------------------- */
:root{
  --street-bg:#04070d; --street-panel:#0a1119; --street-panel2:#101924;
  --street-line:#243446; --street-text:#f4f8ff; --street-muted:#9db0c2;
  --nitro:#2be7ff; --turbo:#55ff9e; --flame:#ff8a1d; --mag:#ff4fd8;
}
.stApp{
  background:
    radial-gradient(900px 380px at 10% -10%, rgba(255,138,29,.18), transparent 55%),
    radial-gradient(760px 320px at 100% 0%, rgba(43,231,255,.16), transparent 54%),
    linear-gradient(180deg, #080c12 0%, #03060a 100%) !important;
  color:var(--street-text) !important;
}
.block-container{max-width:900px!important;padding-top:.5rem!important;padding-bottom:6.6rem!important}
.brand,.hero{display:none!important}
.v14-brand{
  position:relative; overflow:hidden;
  border:1px solid rgba(255,255,255,.10)!important;
  border-radius:26px!important;
  padding:20px 20px 18px!important;
  background:
    linear-gradient(135deg, rgba(13,18,27,.97), rgba(7,11,18,.98))!important;
  box-shadow:0 18px 48px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.05)!important;
}
.v14-brand:before{
  content:""; position:absolute; inset:0;
  background:
    linear-gradient(115deg, transparent 0 14%, rgba(255,138,29,.10) 14.5% 17%, transparent 17.5% 100%),
    linear-gradient(115deg, transparent 0 18%, rgba(43,231,255,.11) 18.5% 21%, transparent 21.5% 100%),
    repeating-linear-gradient(135deg, rgba(255,255,255,.018) 0 9px, rgba(0,0,0,0) 9px 18px);
  pointer-events:none;
}
.v14-brand:after{
  content:""; position:absolute; right:-80px; top:-76px; width:220px; height:220px; border-radius:50%;
  background:radial-gradient(circle, rgba(43,231,255,.20) 0%, rgba(43,231,255,.08) 36%, transparent 68%);
  filter:blur(6px);
}
.v14-chip,.vbadge{
  border:1px solid rgba(255,138,29,.6)!important;
  background:linear-gradient(90deg, rgba(255,138,29,.18), rgba(255,79,216,.16))!important;
  color:#ffd4aa!important; font-weight:900!important; letter-spacing:.08em!important;
}
.v14-logo{filter:drop-shadow(0 0 11px rgba(43,231,255,.55))!important}
.v14-title{font-size:1.58rem!important; letter-spacing:.02em!important; text-transform:uppercase!important}
.v14-title b{color:#ffffff!important}
.v14-sub{color:#c0cddd!important; max-width:640px}
.v14-sub:after{
  content:"Probability first • ranked strongest to weakest • verified current markets";
  display:block; margin-top:6px; color:#8fa7bc; font-size:.82rem; letter-spacing:.02em;
}

/* Section blocks / inputs */
div[data-testid="stVerticalBlockBorderWrapper"]{
  border:1px solid rgba(255,255,255,.07)!important;
  border-radius:22px!important;
  background:
    linear-gradient(180deg, rgba(15,21,30,.96), rgba(8,12,18,.97))!important;
  box-shadow:0 12px 30px rgba(0,0,0,.24)!important;
}
div[data-testid="stExpander"]{
  border:1px solid rgba(61,89,115,.58)!important;
  background:linear-gradient(180deg, rgba(13,19,28,.98), rgba(8,12,18,.98))!important;
  border-radius:20px!important;
  position:relative;
}
div[data-testid="stExpander"]:before{
  content:""; position:absolute; left:0; top:0; bottom:0; width:5px;
  background:linear-gradient(180deg, var(--flame), var(--nitro));
}
div[data-testid="stExpander"] details summary{min-height:74px!important}

h2,h3{letter-spacing:.01em!important}
h2{color:#f7fbff!important; text-transform:uppercase!important; font-size:1.45rem!important}
label[data-testid="stWidgetLabel"] p{color:#dbe8f6!important; text-transform:uppercase; letter-spacing:.04em; font-size:.78rem}

/* Inputs */
div[data-baseweb="select"]>div,
[data-testid="stDateInput"] input,
[data-testid="stNumberInput"] input{
  background:linear-gradient(180deg,#121b26,#0c141d)!important;
  border:1px solid #314457!important; color:#fff!important;
  border-radius:14px!important;
}

/* Buttons */
.stButton>button,
[data-testid="stFormSubmitButton"] button,
[data-testid="stDownloadButton"] button{
  min-height:56px!important; border-radius:16px!important; font-weight:900!important;
  letter-spacing:.03em!important; text-transform:uppercase!important;
  color:#fff!important; border:1px solid rgba(255,255,255,.10)!important;
  background:
    linear-gradient(100deg, rgba(255,138,29,.95), rgba(255,79,216,.86) 52%, rgba(43,231,255,.90))!important;
  box-shadow:0 10px 28px rgba(255,138,29,.18), 0 0 0 1px rgba(255,255,255,.04) inset!important;
}
.stButton>button:hover,
[data-testid="stFormSubmitButton"] button:hover,
[data-testid="stDownloadButton"] button:hover{filter:brightness(1.05)!important; transform:translateY(-1px)}

/* Metrics */
div[data-testid="stMetric"]{
  position:relative; overflow:hidden;
  min-height:98px!important;
  border-radius:18px!important; border:1px solid rgba(255,255,255,.08)!important;
  background:linear-gradient(180deg, rgba(15,22,31,.96), rgba(8,13,19,.98))!important;
}
div[data-testid="stMetric"]:before{
  content:""; position:absolute; left:0; top:0; right:0; height:4px;
  background:linear-gradient(90deg, var(--flame), var(--mag), var(--nitro));
}
div[data-testid="stMetricLabel"]{color:#9fb0c2!important}
div[data-testid="stMetricValue"]{color:#fff!important}

/* Alerts / tables */
[data-testid="stAlert"]{border-radius:18px!important}
[data-testid="stDataFrame"]{border:1px solid rgba(70,95,122,.65)!important; border-radius:18px!important}

/* Bottom nav */
.v14-nav{background:rgba(6,10,16,.95)!important;border-top:1px solid rgba(255,255,255,.10)!important}
.v14-nav .active{color:var(--flame)!important}

/* Mobile */
@media(max-width:699px){
  .block-container{padding-left:.72rem!important;padding-right:.72rem!important}
  .v14-title{font-size:1.24rem!important}
  .v14-sub{font-size:.86rem!important}
  div[data-testid="stMetricValue"]{font-size:1.44rem!important}
}
</style>
<div style="margin:-2px 0 12px; padding:10px 14px; border-radius:16px; border:1px solid rgba(255,138,29,.22); background:linear-gradient(90deg, rgba(255,138,29,.08), rgba(43,231,255,.06)); color:#c8d7e7; font-size:.88rem;">
  <b style="color:#fff; letter-spacing:.04em;">STREET EDITION</b> · V26 expanded fixture universe + analyst ranking: unfinished fixtures stay in the probability table even when current odds are not yet verified. Price verification is required only for return calculations and live betting.
</div>
""", unsafe_allow_html=True)

RAW="https://raw.githubusercontent.com/openfootball/football.json/master"
LEAGUES={
    "Premier League":{"of":"en.1","odds":"soccer_epl"},
    "Championship":{"of":"en.2","odds":"soccer_efl_champ"},
    "League One":{"of":"en.3","odds":"soccer_england_league1"},
    "League Two":{"of":"en.4","odds":"soccer_england_league2"},
    "Bundesliga":{"of":"de.1","odds":"soccer_germany_bundesliga"},
    "2. Bundesliga":{"of":"de.2","odds":"soccer_germany_bundesliga2"},
    "3. Liga":{"of":"de.3","odds":"soccer_germany_liga3"},
    "La Liga":{"of":"es.1","odds":"soccer_spain_la_liga"},
    "La Liga 2":{"of":"es.2","odds":"soccer_spain_segunda_division"},
    "Serie A":{"of":"it.1","odds":"soccer_italy_serie_a"},
    "Serie B":{"of":"it.2","odds":"soccer_italy_serie_b"},
    "Ligue 1":{"of":"fr.1","odds":"soccer_france_ligue_one"},
    "Ligue 2":{"of":"fr.2","odds":"soccer_france_ligue_two"},
}

# V15 league-aware promotion policy. These choices are based on the V14
# chronological unseen-data calibration tests at the 62% audit threshold.
# No league is allowed to inherit another league's calibration result.
V15_POLICY={
    "Premier League":{"engine":"calibrated","status":"APPROVED","evidence":"Legacy V14 gap +6.1pp → +2.1pp"},
    "Championship":{"engine":"raw","status":"RAW FALLBACK","evidence":"Legacy calibrated validation unavailable"},
    "League One":{"engine":"raw","status":"V26 NEW","evidence":"Dedicated V26 model; run Deep Validation for league-specific promotion evidence"},
    "League Two":{"engine":"raw","status":"V26 NEW","evidence":"Dedicated V26 model; run Deep Validation for league-specific promotion evidence"},
    "Bundesliga":{"engine":"calibrated","status":"APPROVED","evidence":"Legacy V14 gap +3.4pp → -1.5pp"},
    "2. Bundesliga":{"engine":"raw","status":"V26 NEW","evidence":"Dedicated V26 model; run Deep Validation for league-specific promotion evidence"},
    "3. Liga":{"engine":"raw","status":"V26 NEW","evidence":"Dedicated V26 model; run Deep Validation for league-specific promotion evidence"},
    "La Liga":{"engine":"raw","status":"APPROVED RAW","evidence":"Legacy raw model retained"},
    "La Liga 2":{"engine":"raw","status":"V26 NEW","evidence":"Dedicated V26 model; run Deep Validation for league-specific promotion evidence"},
    "Serie A":{"engine":"raw","status":"APPROVED RAW","evidence":"Legacy raw model retained"},
    "Serie B":{"engine":"raw","status":"V26 NEW","evidence":"Dedicated V26 model; run Deep Validation for league-specific promotion evidence"},
    "Ligue 1":{"engine":"raw","status":"APPROVED RAW","evidence":"Legacy raw model retained"},
    "Ligue 2":{"engine":"raw","status":"V26 NEW","evidence":"Dedicated V26 model; run Deep Validation for league-specific promotion evidence"},
}
# Only leagues actually present in OpenFootball's 2026/27 JSON repository are exposed.
SEASONS=["2018-19","2019-20","2020-21","2021-22","2022-23","2023-24","2024-25","2025-26","2026-27"]
FEATURES=[
    "h_pts","a_pts","h_gf","a_gf","h_ga","a_ga",
    "elo_diff","elo_home",
    "h_venue_ppg","a_venue_ppg",
    "h_opp_adj_ppg","a_opp_adj_ppg",
    "h_perf_vs_expect","a_perf_vs_expect",
    "h_rest_days","a_rest_days",
    "h_games14","a_games14",
]

HEADERS={"User-Agent":"Mozilla/5.0 FootballPredictorV28/1.0","Accept":"application/json"}

def get_json(url):
    r=requests.get(url,headers=HEADERS,timeout=25)
    r.raise_for_status()
    data=r.json()
    if not isinstance(data,dict) or "matches" not in data:
        raise ValueError("Unexpected OpenFootball JSON format.")
    return data

@st.cache_data(ttl=1800,show_spinner=False)
def season_json(season,code):
    return get_json(f"{RAW}/{season}/{code}.json")

def team_name(x):
    if isinstance(x,str): return x
    if isinstance(x,dict):
        return str(x.get("name") or x.get("title") or x.get("code") or "")
    return str(x)

def score_ft(m):
    s=m.get("score")
    if isinstance(s,dict):
        ft=s.get("ft")
        if isinstance(ft,(list,tuple)) and len(ft)>=2:
            try: return int(ft[0]),int(ft[1])
            except: pass
    return None

def elo_p(d): return 1/(1+10**(-d/400))

def _mean_or(rows,key,default=0.0):
    vals=[]
    for x in rows:
        try:
            v=float(x.get(key))
            if np.isfinite(v): vals.append(v)
        except Exception:
            pass
    return float(np.mean(vals)) if vals else float(default)

def _empirical_bayes(recent_value, recent_n, prior_value, prior_strength=5.0):
    """Shrink noisy small samples toward a longer-run team prior."""
    n=max(float(recent_n),0.0)
    k=max(float(prior_strength),0.01)
    return float((n*float(recent_value)+k*float(prior_value))/(n+k))

def _team_live_features(hist, team, venue, fixture_date):
    rows=list(hist[team])
    last=rows[-8:]
    long_run=rows[-20:]
    venue_rows=[x for x in rows if x.get("venue")==venue][-8:]

    # Long-run team priors stabilize early-season / small-sample form.
    prior_pts=_mean_or(long_run,"pts",1.35)
    prior_gf=_mean_or(long_run,"gf",1.25)
    prior_ga=_mean_or(long_run,"ga",1.25)
    recent_pts=_mean_or(last,"pts",prior_pts)
    recent_gf=_mean_or(last,"gf",prior_gf)
    recent_ga=_mean_or(last,"ga",prior_ga)
    pts=_empirical_bayes(recent_pts,len(last),prior_pts,5.0)
    gf=_empirical_bayes(recent_gf,len(last),prior_gf,5.0)
    ga=_empirical_bayes(recent_ga,len(last),prior_ga,5.0)

    venue_prior=prior_pts
    venue_raw=_mean_or(venue_rows,"pts",venue_prior)
    venue_ppg=_empirical_bayes(venue_raw,len(venue_rows),venue_prior,4.0)

    # Stronger opponent adjustment: evaluate each result against what Elo said
    # was expected BEFORE that match. Positive residual = team performed better
    # than the strength of its opponent/context implied.
    residuals=[]
    adj_points=[]
    for x in last:
        try:
            outcome=float(x.get("result_score", float(x.get("pts",0.0))/3.0))
            expected=float(x.get("expected_score",0.5))
            residuals.append(outcome-expected)
            opp_elo=float(x.get("opp_elo",1500.0))
            # descriptive opponent-adjusted points retained for diagnostics/features
            adj_points.append(float(x.get("pts",0.0))*float(np.clip(opp_elo/1500.0,.80,1.20)))
        except Exception:
            pass
    perf_vs_expect=float(np.mean(residuals)) if residuals else 0.0
    opp_adj_raw=float(np.mean(adj_points)) if adj_points else pts
    opp_adj=_empirical_bayes(opp_adj_raw,len(adj_points),prior_pts,4.0)

    fd=pd.to_datetime(fixture_date,errors="coerce")
    rest_days=7.0
    games14=0.0
    if pd.notna(fd):
        dated=[]
        for x in rows:
            dt=pd.to_datetime(x.get("date"),errors="coerce")
            if pd.notna(dt) and dt < fd:
                dated.append(dt)
                if 0 <= (fd-dt).days <= 14:
                    games14 += 1.0
        if dated:
            rest_days=float(np.clip((fd-max(dated)).days,1,21))

    return {
        "pts":pts,"gf":gf,"ga":ga,"venue_ppg":venue_ppg,
        "opp_adj_ppg":opp_adj,"perf_vs_expect":perf_vs_expect,
        "rest_days":rest_days,"games14":games14,
        "sample_n":len(last),
    }

def build_feature_values(hist, elo, home, away, fixture_date):
    hs=_team_live_features(hist,home,"H",fixture_date)
    aa=_team_live_features(hist,away,"A",fixture_date)
    eh,ea=float(elo[home]),float(elo[away])
    return {
        "h_pts":hs["pts"],"a_pts":aa["pts"],
        "h_gf":hs["gf"],"a_gf":aa["gf"],
        "h_ga":hs["ga"],"a_ga":aa["ga"],
        "elo_diff":eh-ea,"elo_home":elo_p(eh+55-ea),
        "h_venue_ppg":hs["venue_ppg"],"a_venue_ppg":aa["venue_ppg"],
        "h_opp_adj_ppg":hs["opp_adj_ppg"],"a_opp_adj_ppg":aa["opp_adj_ppg"],
        "h_perf_vs_expect":hs["perf_vs_expect"],"a_perf_vs_expect":aa["perf_vs_expect"],
        "h_rest_days":hs["rest_days"],"a_rest_days":aa["rest_days"],
        "h_games14":hs["games14"],"a_games14":aa["games14"],
    }

def make_training(code,w=8):
    """Chronological training frame with no future leakage.

    Each row is built using only matches already completed before that fixture.
    The richer V22 features add venue form, opponent-adjusted form, recovery time
    and recent fixture congestion while retaining the proven Elo/core-form inputs.
    """
    hist=defaultdict(lambda:deque(maxlen=20))
    elo=defaultdict(lambda:1500.0)
    rows=[]
    used=0
    for season in SEASONS:
        try: matches=season_json(season,code)["matches"]
        except Exception: continue
        matches=sorted(matches,key=lambda m:str(m.get("date","")))
        for m in matches:
            sc=score_ft(m)
            if sc is None: continue
            h=team_name(m.get("team1","")).strip()
            a=team_name(m.get("team2","")).strip()
            if not h or not a: continue
            dt=pd.to_datetime(m.get("date"),errors="coerce")
            if pd.isna(dt): continue
            hg,ag=sc
            eh,ea=float(elo[h]),float(elo[a])

            vals=build_feature_values(hist,elo,h,a,dt)
            if hg>ag: y=0; hp,ap,res=3,0,1.
            elif hg<ag: y=2; hp,ap,res=0,3,0.
            else: y=1; hp,ap,res=1,1,.5
            rows.append({**vals,"y":y,"_date":dt,"_home":h,"_away":a})

            # Store pre-match expectation for future opponent-adjusted residuals.
            ex_home=elo_p(eh+55-ea)
            ex_away=1.0-ex_home
            hist[h].append({"pts":hp,"gf":hg,"ga":ag,"venue":"H","opp_elo":ea,"date":dt,
                            "result_score":res,"expected_score":ex_home})
            hist[a].append({"pts":ap,"gf":ag,"ga":hg,"venue":"A","opp_elo":eh,"date":dt,
                            "result_score":1.0-res,"expected_score":ex_away})
            ex=elo_p(eh-ea)
            elo[h]+=24*(res-ex); elo[a]+=24*((1-res)-(1-ex))
            used+=1
    frame=pd.DataFrame(rows)
    if not frame.empty:
        frame=frame.sort_values("_date").reset_index(drop=True)
    return frame,hist,elo,used


FAST_SEASONS=["2024-25","2025-26","2026-27"]

@st.cache_data(ttl=21600,show_spinner=False)
def make_training_fast(code,w=8):
    """Low-latency live training frame.

    Uses the latest three seasons only. Deep validation still uses the full
    historical archive, but ordinary rankings do not download/process nine
    seasons before showing anything.
    """
    hist=defaultdict(lambda:deque(maxlen=20))
    elo=defaultdict(lambda:1500.0)
    rows=[]; used=0
    for season in FAST_SEASONS:
        try:
            matches=season_json(season,code)["matches"]
        except Exception:
            continue
        for m in sorted(matches,key=lambda x:str(x.get("date",""))):
            sc=score_ft(m)
            if sc is None: continue
            h=team_name(m.get("team1","")).strip()
            a=team_name(m.get("team2","")).strip()
            if not h or not a: continue
            dt=pd.to_datetime(m.get("date"),errors="coerce")
            if pd.isna(dt): continue
            hg,ag=sc; eh,ea=float(elo[h]),float(elo[a])
            vals=build_feature_values(hist,elo,h,a,dt)
            if hg>ag: y=0; hp,ap,res=3,0,1.0
            elif hg<ag: y=2; hp,ap,res=0,3,0.0
            else: y=1; hp,ap,res=1,1,0.5
            rows.append({**vals,"y":y,"_date":dt,"_home":h,"_away":a})
            ex_home=elo_p(eh+55-ea); ex_away=1.0-ex_home
            hist[h].append({"pts":hp,"gf":hg,"ga":ag,"venue":"H","opp_elo":ea,"date":dt,
                            "result_score":res,"expected_score":ex_home})
            hist[a].append({"pts":ap,"gf":ag,"ga":hg,"venue":"A","opp_elo":eh,"date":dt,
                            "result_score":1.0-res,"expected_score":ex_away})
            ex=elo_p(eh-ea)
            elo[h]+=24*(res-ex); elo[a]+=24*((1-res)-(1-ex))
            used+=1
    f=pd.DataFrame(rows)
    if not f.empty:
        f=f.sort_values("_date").reset_index(drop=True)
    return f,hist,elo,used

def _time_decay_weights(frame, half_life_days=540.0):
    if "_date" not in frame.columns or frame.empty:
        return np.ones(len(frame),dtype=float)
    d=pd.to_datetime(frame["_date"],errors="coerce")
    ref=d.max()
    age=(ref-d).dt.days.fillna(0).clip(lower=0).to_numpy(dtype=float)
    return np.power(0.5, age/max(float(half_life_days),1.0))

class V22ProbabilityEnsemble:
    def __init__(self, members, weights, temperature=1.0):
        self.members=members
        self.weights=np.asarray(weights,dtype=float)
        self.weights=self.weights/self.weights.sum()
        self.temperature=float(temperature)

    def predict_proba(self, X):
        total=None
        for weight,(model,cols) in zip(self.weights,self.members):
            p=np.asarray(model.predict_proba(X[cols].fillna(0)),dtype=float)
            total=weight*p if total is None else total+weight*p
        p=np.clip(total,1e-8,1.0)
        # Temperature scaling on probabilities. T>1 softens overconfidence;
        # T<1 sharpens only when chronological validation supports it.
        p=np.power(p,1.0/max(self.temperature,1e-6))
        p=p/p.sum(axis=1,keepdims=True)
        return p

def _temperature_search(prob, y):
    best=(float("inf"),1.0)
    for t in np.arange(0.70,1.51,0.05):
        p=np.clip(prob,1e-8,1.0)
        p=np.power(p,1.0/t); p=p/p.sum(axis=1,keepdims=True)
        try: loss=log_loss(y,p,labels=[0,1,2])
        except Exception: continue
        if loss<best[0]: best=(float(loss),float(t))
    return best

def _fit_candidate_models(train_frame, sample_weight):
    all_cols=list(FEATURES)
    elo_cols=["elo_diff","elo_home"]
    form_cols=[
        "h_pts","a_pts","h_gf","a_gf","h_ga","a_ga",
        "h_venue_ppg","a_venue_ppg","h_opp_adj_ppg","a_opp_adj_ppg",
        "h_perf_vs_expect","a_perf_vs_expect",
        "h_rest_days","a_rest_days","h_games14","a_games14",
    ]
    specs=[
        ("Gradient boost",HistGradientBoostingClassifier(
            max_iter=140,max_leaf_nodes=9,learning_rate=.04,
            min_samples_leaf=32,l2_regularization=8,random_state=42),all_cols),
        ("Full logistic",LogisticRegression(
            max_iter=1600,C=.55,solver="lbfgs"),all_cols),
        ("Elo specialist",LogisticRegression(
            max_iter=1200,C=.7,solver="lbfgs"),elo_cols),
        ("Form/context specialist",LogisticRegression(
            max_iter=1500,C=.55,solver="lbfgs"),form_cols),
    ]
    fitted=[]
    for name,model,cols in specs:
        model.fit(train_frame[cols].fillna(0),train_frame["y"],sample_weight=sample_weight)
        fitted.append((name,model,cols))
    return fitted

def _candidate_probabilities(fitted, frame):
    result=[]
    for name,model,cols in fitted:
        p=np.asarray(model.predict_proba(frame[cols].fillna(0)),dtype=float)
        result.append((name,p))
    return result




@st.cache_data(ttl=1800,show_spinner=False)
def fixture_context(code, fixture_date, home, away):
    """Transparent context layer from OpenFootball completed matches only. No injury claims are invented."""
    season = SEASONS[-1]
    try:
        matches=season_json(season,code)["matches"]
    except Exception:
        return {"available":False,"reason":"Current-season context unavailable"}
    cutoff=pd.to_datetime(fixture_date,errors="coerce")
    rec=defaultdict(list); table=defaultdict(lambda:{"p":0,"pts":0,"gf":0,"ga":0,"w":0,"d":0,"l":0})
    for m in sorted(matches,key=lambda z:str(z.get("date",""))):
        dt=pd.to_datetime(m.get("date"),errors="coerce"); sc=score_ft(m)
        if sc is None or pd.isna(dt) or (not pd.isna(cutoff) and dt>=cutoff): continue
        h=team_name(m.get("team1","")).strip(); a=team_name(m.get("team2","")).strip(); hg,ag=sc
        if not h or not a: continue
        hp,ap=(3,0) if hg>ag else ((0,3) if hg<ag else (1,1))
        for t,gf,ga,pts,venue in [(h,hg,ag,hp,"H"),(a,ag,hg,ap,"A")]:
            x=table[t]; x["p"]+=1; x["pts"]+=pts; x["gf"]+=gf; x["ga"]+=ga
            x["w"]+=pts==3; x["d"]+=pts==1; x["l"]+=pts==0
            rec[t].append({"date":dt,"gf":gf,"ga":ga,"pts":pts,"venue":venue,"cs":ga==0,"btts":gf>0 and ga>0})
    order=sorted(table,key=lambda t:(table[t]["pts"],table[t]["gf"]-table[t]["ga"],table[t]["gf"]),reverse=True)
    pos={t:i+1 for i,t in enumerate(order)}
    def snap(t,venue):
        rr=rec.get(t,[]); last=rr[-8:]; va=[x for x in rr if x["venue"]==venue][-5:]
        def form(xs): return "".join("W" if x["pts"]==3 else "D" if x["pts"]==1 else "L" for x in xs) or "—"
        return {"position":pos.get(t),"played":table[t]["p"],"points":table[t]["pts"],"form":form(last),
                "ppg":round(sum(x["pts"] for x in last)/len(last),2) if last else None,
                "gf":round(sum(x["gf"] for x in last)/len(last),2) if last else None,
                "ga":round(sum(x["ga"] for x in last)/len(last),2) if last else None,
                "clean_sheet":round(100*sum(x["cs"] for x in last)/len(last),1) if last else None,
                "btts":round(100*sum(x["btts"] for x in last)/len(last),1) if last else None,
                "venue_form":form(va),"venue_ppg":round(sum(x["pts"] for x in va)/len(va),2) if va else None}
    hs,as_=snap(home,"H"),snap(away,"A")
    # Simple transparent scoreline context, not a replacement for the 1X2 model.
    import math
    lh=max(.15, ((hs.get("gf") or 1.2)+(as_.get("ga") or 1.2))/2)
    la=max(.15, ((as_.get("gf") or 1.0)+(hs.get("ga") or 1.0))/2)
    scores=[]
    for i in range(6):
        for j in range(6):
            pr=math.exp(-lh)*lh**i/math.factorial(i)*math.exp(-la)*la**j/math.factorial(j)
            scores.append((pr,f"{i}-{j}"))
    scores=sorted(scores,reverse=True)[:3]
    return {"available":True,"home":hs,"away":as_,"xg_like_home":round(lh,2),"xg_like_away":round(la,2),
            "scorelines":[{"score":x[1],"prob":round(x[0]*100,1)} for x in scores],
            "injuries":"UNAVAILABLE — no verified injury/suspension provider connected"}


def _calibrate_temperature(prob, y, temperature):
    p=np.clip(np.asarray(prob,dtype=float),1e-8,1.0)
    p=np.power(p,1.0/max(float(temperature),1e-6))
    return p/p.sum(axis=1,keepdims=True)

def _rank_product_backtest(frame, prob):
    """Backtest the actual product: strongest-team ranks and safest-five days."""
    if frame.empty or len(prob)!=len(frame):
        return {"rank_rows":[],"safest_five":{}}
    records=[]
    y=frame["y"].to_numpy()
    dates=pd.to_datetime(frame["_date"],errors="coerce").dt.date
    for j in range(len(frame)):
        hp=float(prob[j,0]); ap=float(prob[j,2])
        pick=0 if hp>=ap else 2
        records.append({
            "date":dates.iloc[j],
            "prob":max(hp,ap),
            "won":bool(y[j]==pick),
            "pick":"HOME" if pick==0 else "AWAY",
        })
    rr=pd.DataFrame(records).dropna(subset=["date"])
    rank_stats={k:{"n":0,"wins":0} for k in range(1,11)}
    combo_days=0; combo_wins=0; expected_joint=[]
    leg_wins=0; leg_total=0
    for _,g in rr.groupby("date"):
        g=g.sort_values("prob",ascending=False).reset_index(drop=True)
        for idx,row in g.head(10).iterrows():
            rank=idx+1
            rank_stats[rank]["n"]+=1
            rank_stats[rank]["wins"]+=int(row["won"])
        if len(g)>=5:
            five=g.head(5)
            combo_days+=1
            combo_wins+=int(bool(five["won"].all()))
            expected_joint.append(float(np.prod(five["prob"].to_numpy(dtype=float))))
            leg_wins+=int(five["won"].sum())
            leg_total+=5
    rank_rows=[]
    for rank,stt in rank_stats.items():
        if stt["n"]:
            rank_rows.append({
                "Rank":rank,"Selections":stt["n"],"Wins":stt["wins"],
                "Strike %":round(100*stt["wins"]/stt["n"],1)
            })
    safest={
        "Days with 5+ fixtures":combo_days,
        "All-five wins":combo_wins,
        "Five-fold hit rate %":round(100*combo_wins/combo_days,1) if combo_days else None,
        "Leg strike %":round(100*leg_wins/leg_total,1) if leg_total else None,
        "Mean estimated joint %":round(100*float(np.mean(expected_joint)),1) if expected_joint else None,
    }
    return {"rank_rows":rank_rows,"safest_five":safest}

@st.cache_resource(show_spinner=False)
def deep_validate_model(lname,code):
    """V25 deep validation: TRAIN -> TUNE -> untouched FINAL TEST. Runs only on demand."""
    f,hist,elo,used=make_training(code)
    if len(f)<240:
        raise RuntimeError(f"Only {len(f)} completed historical matches available; V25 needs at least 240 for a clean three-way split.")

    n=len(f)
    train_end=max(160,int(n*.70))
    tune_end=max(train_end+40,int(n*.85))
    if n-tune_end < 35:
        tune_end=n-35
    if tune_end-train_end < 35:
        train_end=tune_end-35

    train_set=f.iloc[:train_end].copy()
    tune_set=f.iloc[train_end:tune_end].copy()
    test_set=f.iloc[tune_end:].copy()

    for name,part in [("train",train_set),("tune",tune_set),("final test",test_set)]:
        if len(part)<30 or part["y"].nunique()<3:
            raise RuntimeError(f"{lname}: {name} split is too small or lacks all three 1X2 outcomes.")

    # TUNE PHASE: choose recency horizon using tune data only.
    half_lives=[270.0,365.0,540.0,730.0,1095.0]
    hl_trials=[]
    for hl in half_lives:
        w=_time_decay_weights(train_set,hl)
        probe=HistGradientBoostingClassifier(
            max_iter=120,max_leaf_nodes=9,learning_rate=.04,
            min_samples_leaf=32,l2_regularization=8,random_state=42)
        probe.fit(train_set[FEATURES].fillna(0),train_set["y"],sample_weight=w)
        pp=probe.predict_proba(tune_set[FEATURES].fillna(0))
        hl_trials.append((float(log_loss(tune_set["y"],pp,labels=[0,1,2])),hl))
    _,best_half_life=min(hl_trials,key=lambda x:x[0])

    train_w=_time_decay_weights(train_set,best_half_life)
    fitted=_fit_candidate_models(train_set,train_w)
    tune_probs=_candidate_probabilities(fitted,tune_set)
    tune_losses=[(name,float(log_loss(tune_set["y"],p,labels=[0,1,2]))) for name,p in tune_probs]
    min_tune=min(x[1] for x in tune_losses)
    raw_weights=np.array([np.exp(-8.0*(loss-min_tune)) for _,loss in tune_losses],dtype=float)
    raw_weights=raw_weights/raw_weights.sum()
    tune_ensemble=sum(w*p for w,(_,p) in zip(raw_weights,tune_probs))
    _,temperature=_temperature_search(tune_ensemble,tune_set["y"].to_numpy())

    # FINAL TEST PHASE: freeze all choices above, refit only on TRAIN+TUNE,
    # and judge the chosen configuration once on the untouched newest block.
    fit_set=f.iloc[:tune_end].copy()
    fit_w=_time_decay_weights(fit_set,best_half_life)
    final_members=_fit_candidate_models(fit_set,fit_w)
    final_probs=_candidate_probabilities(final_members,test_set)
    final_ensemble=sum(w*p for w,(_,p) in zip(raw_weights,final_probs))
    final_ensemble=_calibrate_temperature(final_ensemble,test_set["y"].to_numpy(),temperature)
    final_loss=float(log_loss(test_set["y"],final_ensemble,labels=[0,1,2]))

    legacy=HistGradientBoostingClassifier(
        max_iter=100,max_leaf_nodes=7,learning_rate=.045,
        min_samples_leaf=38,l2_regularization=8,random_state=42)
    legacy.fit(fit_set[FEATURES].fillna(0),fit_set["y"],sample_weight=fit_w)
    legacy_p=legacy.predict_proba(test_set[FEATURES].fillna(0))
    legacy_loss=float(log_loss(test_set["y"],legacy_p,labels=[0,1,2]))

    promote=(final_loss + 0.001) < legacy_loss
    chosen_test_p=final_ensemble if promote else legacy_p
    chosen_pred=np.argmax(chosen_test_p,axis=1)
    acc=float(accuracy_score(test_set["y"],chosen_pred))
    avg_conf=float(np.max(chosen_test_p,axis=1).mean())
    calibration_gap=(avg_conf-acc)*100.0

    # Product-level stress tests on the same untouched final block.
    product_audit=_rank_product_backtest(test_set,chosen_test_p)

    # Refit the promoted architecture on all history only AFTER the untouched
    # test has approved it. Hyperparameters remain frozen from the tune block.
    full_w=_time_decay_weights(f,best_half_life)
    if promote:
        full_fitted=_fit_candidate_models(f,full_w)
        members=[(model,cols) for _,model,cols in full_fitted]
        model=V22ProbabilityEnsemble(members,raw_weights,temperature)
        engine_label="V25 VALIDATED ENSEMBLE"
        chosen_loss=final_loss
    else:
        legacy_full=HistGradientBoostingClassifier(
            max_iter=100,max_leaf_nodes=7,learning_rate=.045,
            min_samples_leaf=38,l2_regularization=8,random_state=42)
        legacy_full.fit(f[FEATURES].fillna(0),f["y"],sample_weight=full_w)
        model=legacy_full
        engine_label="V25 LEGACY SAFETY FALLBACK"
        chosen_loss=legacy_loss

    weights_map={name:round(float(w),3) for (name,_),w in zip(tune_losses,raw_weights)}
    meta={
        "League":lname,
        "Engine":engine_label,
        "Train matches":int(len(train_set)),
        "Tune matches":int(len(tune_set)),
        "Final test matches":int(len(test_set)),
        "Legacy log loss":round(legacy_loss,4),
        "V25 log loss":round(float(chosen_loss),4),
        "Accuracy %":round(acc*100,1),
        "Avg confidence %":round(avg_conf*100,1),
        "Calibration gap pp":round(calibration_gap,1),
        "Time-decay half-life days":int(best_half_life),
        "Temperature":round(float(temperature if promote else 1.0),2),
        "Ensemble weights":weights_map if promote else {"Legacy conservative":1.0},
        "Promoted":bool(promote),
        "Rank audit":product_audit["rank_rows"],
        "Safest-five audit":product_audit["safest_five"],
    }
    validation_status="V25 PROMOTED" if promote else "V25 FALLBACK"
    validation_evidence=(
        f"Untouched final test: legacy log loss {legacy_loss:.4f}; "
        f"candidate {final_loss:.4f}; {len(test_set)} newest unseen matches. "
        f"All decay/weight/calibration choices were frozen on the earlier tune block."
    )
    return model,hist,elo,used,engine_label,validation_status,validation_evidence,meta


V28_FROZEN_CALIBRATION={
    "Premier League":{"temperature":0.80,"draw_multiplier":1.00,"promoted":True,"holdout_n":353,"raw_logloss":1.00678,"cal_logloss":1.00753,"raw_selected_gap_pp":-2.805,"cal_selected_gap_pp":1.105,"reason":"Promoted for substantially better selected-team calibration with negligible log-loss change."},
    "Championship":{"temperature":1.10,"draw_multiplier":0.95,"promoted":True,"holdout_n":521,"raw_logloss":1.04215,"cal_logloss":1.04057,"raw_selected_gap_pp":-1.496,"cal_selected_gap_pp":-2.118,"reason":"Promoted on improved untouched-holdout log loss."},
    "Bundesliga":{"temperature":0.85,"draw_multiplier":1.10,"promoted":True,"holdout_n":294,"raw_logloss":0.96362,"cal_logloss":0.96134,"raw_selected_gap_pp":-6.085,"cal_selected_gap_pp":-4.314,"reason":"Promoted on better log loss and selected-team calibration."},
    "La Liga":{"temperature":1.00,"draw_multiplier":1.00,"promoted":False,"holdout_n":365,"raw_logloss":0.98660,"cal_logloss":0.99025,"raw_selected_gap_pp":-2.378,"cal_selected_gap_pp":-0.742,"reason":"Raw retained because fitted calibration worsened untouched-holdout log loss."},
    "Serie A":{"temperature":1.00,"draw_multiplier":1.00,"promoted":False,"holdout_n":344,"raw_logloss":0.96388,"cal_logloss":0.97623,"raw_selected_gap_pp":-3.384,"cal_selected_gap_pp":-2.404,"reason":"Raw retained because fitted calibration materially worsened untouched-holdout log loss."},
    "Ligue 1":{"temperature":0.95,"draw_multiplier":0.90,"promoted":True,"holdout_n":282,"raw_logloss":0.97434,"cal_logloss":0.96707,"raw_selected_gap_pp":-2.069,"cal_selected_gap_pp":0.076,"reason":"Promoted on better log loss and near-zero selected-team calibration gap."},
}
V28_POOLED_AUDIT={"holdout_n":2159,"raw_logloss":0.99495,"candidate_cal_logloss":0.99664,"raw_selected_gap_pp":-2.860,"candidate_selected_gap_pp":-1.258,"decision":"RAW FALLBACK","reason":"Pooled calibration improved selected-team gap but worsened holdout log loss; unvalidated expanded leagues stay raw."}

def _apply_v28_frozen_calibration(probs,lname):
    p=np.asarray(probs,dtype=float); cfg=V28_FROZEN_CALIBRATION.get(str(lname))
    if not cfg or not cfg.get("promoted"):
        return p/p.sum()
    T=float(cfg["temperature"]); dm=float(cfg["draw_multiplier"])
    q=np.power(np.clip(p,1e-9,1.0),1.0/T); q[1]*=dm
    return q/q.sum()

V27_STATE_SEASONS=["2025-26","2026-27"]

@st.cache_data(ttl=21600,show_spinner=False)
def build_live_state_snapshot(code):
    """Build recent team state only. No sklearn model is fitted in normal use."""
    hist=defaultdict(lambda:deque(maxlen=20))
    elo=defaultdict(lambda:1500.0)
    used=0
    for season in V27_STATE_SEASONS:
        try:
            matches=season_json(season,code)["matches"]
        except Exception:
            continue
        for m in sorted(matches,key=lambda x:str(x.get("date",""))):
            sc=score_ft(m)
            if sc is None: continue
            h=team_name(m.get("team1","")).strip(); a=team_name(m.get("team2","")).strip()
            if not h or not a: continue
            dt=pd.to_datetime(m.get("date"),errors="coerce")
            if pd.isna(dt): continue
            hg,ag=sc; eh,ea=float(elo[h]),float(elo[a])
            if hg>ag: hp,ap,res=3,0,1.0
            elif hg<ag: hp,ap,res=0,3,0.0
            else: hp,ap,res=1,1,0.5
            ex_home=elo_p(eh+55-ea); ex_away=1.0-ex_home
            hist[h].append({"pts":hp,"gf":hg,"ga":ag,"venue":"H","opp_elo":ea,"date":dt,
                            "result_score":res,"expected_score":ex_home})
            hist[a].append({"pts":ap,"gf":ag,"ga":hg,"venue":"A","opp_elo":eh,"date":dt,
                            "result_score":1.0-res,"expected_score":ex_away})
            ex=elo_p(eh-ea)
            elo[h]+=24*(res-ex); elo[a]+=24*((1-res)-(1-ex))
            used+=1
    return hist,elo,used

class V28FrozenProbabilityModel:
    """Frozen production scorer with embedded holdout-tested calibration."""
    def __init__(self,lname):
        self.lname=str(lname)
    def predict_proba(self,X):
        x=X.fillna(0).copy()
        ep=np.clip(pd.to_numeric(x["elo_home"],errors="coerce").fillna(.5).to_numpy(float),.08,.92)
        elo_logit=np.log(ep/(1-ep))
        form=(x["h_pts"].to_numpy(float)-x["a_pts"].to_numpy(float))/1.5
        venue=(x["h_venue_ppg"].to_numpy(float)-x["a_venue_ppg"].to_numpy(float))/1.5
        opp=(x["h_opp_adj_ppg"].to_numpy(float)-x["a_opp_adj_ppg"].to_numpy(float))/1.5
        perf=(x["h_perf_vs_expect"].to_numpy(float)-x["a_perf_vs_expect"].to_numpy(float))*2.0
        scoring=((x["h_gf"].to_numpy(float)-x["a_ga"].to_numpy(float)) -
                 (x["a_gf"].to_numpy(float)-x["h_ga"].to_numpy(float)))/2.0
        rest=np.clip((x["h_rest_days"].to_numpy(float)-x["a_rest_days"].to_numpy(float))/7.0,-1,1)
        congest=np.clip((x["a_games14"].to_numpy(float)-x["h_games14"].to_numpy(float))/4.0,-1,1)
        z=.72*elo_logit+.34*form+.18*venue+.16*opp+.10*perf+.12*scoring+.06*rest+.05*congest
        home_cond=1/(1+np.exp(-np.clip(z,-6,6)))
        draw=np.clip(.29-.055*np.abs(z),.16,.31)
        home=(1-draw)*home_cond
        away=(1-draw)*(1-home_cond)
        raw=np.column_stack([home,draw,away])
        return np.asarray([_apply_v28_frozen_calibration(row,self.lname) for row in raw],dtype=float)

@st.cache_resource(show_spinner=False)
def train_fast_live(lname, code):
    hist,elo,used=build_live_state_snapshot(code)
    if used<80:
        raise RuntimeError(f"Only {used} recent completed matches available for live state.")
    model=V28FrozenProbabilityModel(lname)
    meta={"League":lname,"Engine":"V28 FROZEN HOLDOUT-AUDITED","Training matches":0,
          "State matches":int(used),"Mode":"No runtime ML training"}
    return model,hist,elo,used,"V28 FROZEN HOLDOUT-AUDITED","FROZEN LIVE",\
        f"Frozen scorer using recent team state; calibration policy was selected on an untouched 2025-26 holdout and no sklearn fit runs in Streamlit.",meta

@st.cache_resource(show_spinner=False)
def _validated_model_registry():
    # Survives ordinary Streamlit reruns and user sessions in the process.
    # A full container restart safely falls back to the fast live model.
    return {}

def _validated_session_models():
    return _validated_model_registry()

def get_live_model(lname,code):
    """Use a validated deep model when one has been run in this process;
    otherwise use the fast cached production model immediately."""
    pool=_validated_session_models()
    if lname in pool:
        return pool[lname]
    return train_fast_live(lname,code)

def fast_fixture_context(hist, home, away, fixture_date):
    """Cheap context derived from the already-built training state.

    Avoids rescanning the entire current season for every fixture. Full/deeper
    context is refreshed only for the displayed Top 10.
    """
    hs=_team_live_features(hist,home,"H",fixture_date)
    aa=_team_live_features(hist,away,"A",fixture_date)
    def snap(x):
        return {
            "position":None,
            "played":x.get("sample_n"),
            "points":None,
            "form":"—",
            "ppg":round(float(x.get("pts",0)),2),
            "gf":round(float(x.get("gf",0)),2),
            "ga":round(float(x.get("ga",0)),2),
            "clean_sheet":None,
            "btts":None,
            "venue_form":"—",
            "venue_ppg":round(float(x.get("venue_ppg",0)),2),
            "opp_adj_ppg":round(float(x.get("opp_adj_ppg",0)),2),
            "perf_vs_expect":round(float(x.get("perf_vs_expect",0)),3),
            "rest_days":round(float(x.get("rest_days",7)),1),
            "games14":int(x.get("games14",0)),
        }
    hsn,asn=snap(hs),snap(aa)
    lh=max(.15,(hsn["gf"]+asn["ga"])/2)
    la=max(.15,(asn["gf"]+hsn["ga"])/2)
    return {
        "available":True,"home":hsn,"away":asn,
        "xg_like_home":round(lh,2),"xg_like_away":round(la,2),
        "scorelines":[],
        "injuries":"UNAVAILABLE — deep provider refresh not run",
        "context_mode":"fast",
    }


@st.cache_data(ttl=1800,show_spinner=False)
def fixtures_for(code):
    return season_json("2026-27",code)["matches"]


# Optional premium football-context provider.
# Add API_FOOTBALL_KEY to Streamlit Secrets to activate. The app remains fully
# operational without it and never fabricates xG, injuries or line-ups.
API_FOOTBALL_BASE="https://v3.football.api-sports.io"
def football_season_year_for_date(value=None):
    dt=pd.to_datetime(value if value is not None else date.today(),errors="coerce")
    if pd.isna(dt): return date.today().year if date.today().month>=7 else date.today().year-1
    return int(dt.year if dt.month>=7 else dt.year-1)

API_FOOTBALL_LEAGUES={
    "Premier League":39,"Championship":40,"League One":41,"League Two":42,
    "Bundesliga":78,"2. Bundesliga":79,"3. Liga":80,
    "La Liga":140,"La Liga 2":141,
    "Serie A":135,"Serie B":136,
    "Ligue 1":61,"Ligue 2":62,
}

def _streamlit_api_football_secret():
    override=str(st.session_state.get("api_football_key_override","")).strip()
    if override:
        return override
    try:
        return str(st.secrets.get("API_FOOTBALL_KEY","")).strip()
    except Exception:
        return ""

@st.cache_resource(show_spinner=False)
def _v28_api_state():
    return {
        "logical_calls":0,"network_calls":0,"errors":0,"rate_limits":0,
        "daily_limit":None,"daily_remaining":None,
        "minute_limit":None,"minute_remaining":None,
        "last_status":None,"last_ok_utc":None,"last_error":"",
        "last_endpoint":"","auth_ok":None
    }

class ApiBudgetDeferred(RuntimeError):
    pass

def _quota_int(v):
    try: return int(float(v))
    except Exception: return None

def _update_api_state_from_headers(headers,status,path,error=""):
    stx=_v28_api_state()
    stx["last_status"]=int(status) if status is not None else None
    stx["last_endpoint"]=str(path)
    dl=_quota_int(headers.get("x-ratelimit-requests-limit"))
    dr=_quota_int(headers.get("x-ratelimit-requests-remaining"))
    ml=_quota_int(headers.get("X-RateLimit-Limit")) or _quota_int(headers.get("x-ratelimit-limit"))
    mr=_quota_int(headers.get("X-RateLimit-Remaining"))
    if mr is None: mr=_quota_int(headers.get("x-ratelimit-remaining"))
    if dl is not None: stx["daily_limit"]=dl
    if dr is not None: stx["daily_remaining"]=dr
    if ml is not None: stx["minute_limit"]=ml
    if mr is not None: stx["minute_remaining"]=mr
    if status is not None and 200 <= int(status) < 300:
        stx["last_ok_utc"]=datetime.now(timezone.utc).isoformat()
        stx["last_error"]=""
        stx["auth_ok"]=True
    elif status in (401,403):
        stx["auth_ok"]=False
    if error:
        stx["last_error"]=str(error)[:250]

def _api_budget_guard(reserve_daily=12,reserve_minute=1):
    stx=_v28_api_state()
    dr=stx.get("daily_remaining"); mr=stx.get("minute_remaining")
    if dr is not None and dr <= int(reserve_daily):
        raise ApiBudgetDeferred(f"API-Football daily reserve reached ({dr} remaining).")
    if mr is not None and mr <= int(reserve_minute):
        raise ApiBudgetDeferred(f"API-Football minute reserve reached ({mr} remaining).")

@st.cache_data(ttl=900,show_spinner=False)
def _api_football_fetch_cached(path, params, api_key):
    _api_budget_guard()
    stx=_v28_api_state(); stx["network_calls"]+=1
    if not api_key:
        return {"response":[],"paging":{},"headers":{},"status":0}
    url=f"{API_FOOTBALL_BASE}/{path}"
    try:
        r=requests.get(url,headers={"x-apisports-key":api_key},params=params,timeout=25)
        hdr=dict(r.headers); _update_api_state_from_headers(hdr,r.status_code,path)
        if r.status_code==429:
            stx["rate_limits"]+=1
            retry=_quota_int(r.headers.get("Retry-After"))
            if retry is not None and 0 < retry <= 3:
                time.sleep(retry)
                r=requests.get(url,headers={"x-apisports-key":api_key},params=params,timeout=25)
                hdr=dict(r.headers); _update_api_state_from_headers(hdr,r.status_code,path)
            if r.status_code==429:
                raise ApiBudgetDeferred("API-Football per-minute rate limit reached; deferred until the next refresh.")
        if r.status_code in (401,403):
            raise RuntimeError(f"API-Football key rejected (HTTP {r.status_code}).")
        r.raise_for_status()
        payload=r.json()
        errs=payload.get("errors") if isinstance(payload,dict) else None
        if errs: raise RuntimeError(f"API-Football error: {errs}")
        return {"response":payload.get("response",[]) if isinstance(payload,dict) else [],
                "paging":payload.get("paging",{}) if isinstance(payload,dict) else {},
                "headers":hdr,"status":r.status_code}
    except ApiBudgetDeferred:
        raise
    except Exception as e:
        stx["errors"]+=1
        _update_api_state_from_headers({},getattr(getattr(e,"response",None),"status_code",None),path,str(e))
        raise

def _api_football_get_meta(path, params, api_key):
    stx=_v28_api_state(); stx["logical_calls"]+=1
    data=_api_football_fetch_cached(path,params,api_key)
    _update_api_state_from_headers(data.get("headers",{}),data.get("status"),path)
    return data

def _api_football_get(path, params, api_key):
    return _api_football_get_meta(path,params,api_key).get("response",[])

def _api_football_get_all(path, params, api_key, max_pages=12):
    first=dict(params or {}); first["page"]=int(first.get("page",1) or 1)
    meta=_api_football_get_meta(path,first,api_key)
    out=list(meta.get("response",[]) or [])
    paging=meta.get("paging") or {}
    current=int(paging.get("current") or 1); total=int(paging.get("total") or 1)
    if total > int(max_pages):
        raise RuntimeError(f"{path} returned {total} pages; safe cap is {max_pages}. Narrow the query.")
    while current < total:
        current+=1
        nxt=dict(params or {}); nxt["page"]=current
        m=_api_football_get_meta(path,nxt,api_key)
        out.extend(m.get("response",[]) or [])
    return out

def _api_health_label(api_key):
    if not api_key: return "⚠️ NOT CONNECTED"
    stx=_v28_api_state()
    if stx.get("auth_ok") is False: return "❌ KEY REJECTED"
    if stx.get("last_ok_utc"): return "✅ VERIFIED"
    return "🟡 CONNECTED / UNTESTED"

@st.cache_data(ttl=86400,show_spinner=False)
def api_football_coverage(lname, api_key, league_id=None, season_year=None):
    """Read provider coverage before trusting empty injury/lineup/stat feeds."""
    if not api_key:
        return {"available":False}
    lid=league_id if league_id is not None else API_FOOTBALL_LEAGUES.get(lname)
    season_year=int(season_year or football_season_year_for_date())
    if not lid:
        return {"available":False}
    resp=_api_football_get("leagues",{"id":int(lid),"season":season_year},api_key)
    if not resp:
        return {"available":False}
    item=resp[0] if isinstance(resp[0],dict) else {}
    seasons=item.get("seasons") or []
    season=next((x for x in seasons if int(x.get("year",0) or 0)==season_year), seasons[-1] if seasons else {})
    cov=season.get("coverage") or {}
    fx=cov.get("fixtures") or {}
    return {
        "available":True,
        "injuries":bool(cov.get("injuries")),
        "predictions":bool(cov.get("predictions")),
        "odds":bool(cov.get("odds")),
        "players":bool(cov.get("players")),
        "lineups":bool(fx.get("lineups")),
        "fixture_statistics":bool(fx.get("statistics_fixtures")),
        "player_statistics":bool(fx.get("statistics_players")),
    }

def _extract_stat(stats, name):
    target=str(name).casefold().replace(" ","_")
    for item in stats or []:
        t=str(item.get("type","")).casefold().replace(" ","_")
        if t==target:
            try:
                v=item.get("value")
                if isinstance(v,str): v=v.replace("%","")
                return float(v)
            except Exception:
                return None
    return None

@st.cache_data(ttl=1800,show_spinner=False)
def api_football_fixture_context(lname, match_day, home, away, api_key, league_id=None, season_year=None):
    """Best-effort true provider context for ONE upcoming fixture.

    Returns only values actually supplied by API-Football. It is intentionally
    conservative: if fixture identity is ambiguous, no enrichment is applied.
    """
    if not api_key:
        return {"available":False,"reason":"API_FOOTBALL_KEY not connected"}
    league_id=league_id if league_id is not None else API_FOOTBALL_LEAGUES.get(lname)
    season_year=int(season_year or football_season_year_for_date())
    if not league_id:
        return {"available":False,"reason":"Provider league ID unavailable"}
    coverage=api_football_coverage(lname,api_key,league_id,season_year)
    date_s=pd.to_datetime(match_day).strftime("%Y-%m-%d")
    fixtures=_api_football_get("fixtures",{"league":int(league_id),"season":season_year,"date":date_s},api_key)

    matches=[]
    for f in fixtures:
        teams=f.get("teams",{})
        hn=str((teams.get("home") or {}).get("name",""))
        an=str((teams.get("away") or {}).get("name",""))
        if team_match(home,hn) and team_match(away,an):
            matches.append(f)
    if len(matches)!=1:
        return {"available":False,"reason":f"Expected one provider fixture match; found {len(matches)}"}

    fx=matches[0]
    fixture_id=(fx.get("fixture") or {}).get("id")
    ht=(fx.get("teams") or {}).get("home") or {}
    at=(fx.get("teams") or {}).get("away") or {}
    if not fixture_id:
        return {"available":False,"reason":"Provider fixture has no ID"}

    injuries=_api_football_get("injuries",{"fixture":fixture_id},api_key) if coverage.get("injuries") else []
    lineups=_api_football_get("fixtures/lineups",{"fixture":fixture_id},api_key) if coverage.get("lineups") else []

    home_inj=[]; away_inj=[]
    for x in injuries:
        team=(x.get("team") or {}).get("name","")
        pobj=x.get("player") or {}
        player=pobj.get("name","")
        reason=pobj.get("reason") or x.get("reason") or ""
        rec={"player":player,"player_id":pobj.get("id"),"reason":reason,
             "type":x.get("type") or pobj.get("type") or "Unavailable"}
        if team_match(team,home): home_inj.append(rec)
        elif team_match(team,away): away_inj.append(rec)

    confirmed={"home":None,"away":None}; confirmed_bench={"home":None,"away":None}
    for lu in lineups:
        tn=(lu.get("team") or {}).get("name","")
        starters=[]; bench=[]
        for p in (lu.get("startXI") or []):
            pobj=p.get("player") or {}
            if pobj.get("name"):
                starters.append({"id":pobj.get("id"),"name":pobj.get("name"),"pos":pobj.get("pos")})
        for p in (lu.get("substitutes") or []):
            pobj=p.get("player") or {}
            if pobj.get("name"):
                bench.append({"id":pobj.get("id"),"name":pobj.get("name"),"pos":pobj.get("pos")})
        if team_match(tn,home):
            confirmed["home"]=starters or None; confirmed_bench["home"]=bench or None
        elif team_match(tn,away):
            confirmed["away"]=starters or None; confirmed_bench["away"]=bench or None

    # Current fixture stats usually appear only once a match starts; for a
    # pre-match fixture we explicitly do NOT call them "xG".
    return {
        "available":True,"fixture_id":fixture_id,
        "home_team_id":ht.get("id"),"away_team_id":at.get("id"),
        "injuries":{"home":home_inj,"away":away_inj},
        "confirmed_lineup":confirmed,
        "confirmed_bench":confirmed_bench,
        "lineups_confirmed":bool(confirmed["home"] and confirmed["away"]),
        "coverage":coverage,
        "reason":"Verified provider fixture matched",
    }

@st.cache_data(ttl=21600,show_spinner=False)
def api_football_recent_xg(team_id, before_date, api_key, sample=5):
    """Rolling TRUE xG/xGA from provider fixture statistics when available."""
    if not api_key or not team_id:
        return {"available":False}
    # Pull recent completed fixtures for the team, then inspect provider stats.
    fixtures=_api_football_get("fixtures",{"team":int(team_id),"last":int(sample+3),"status":"FT"},api_key)
    cutoff=pd.to_datetime(before_date,utc=True,errors="coerce")
    vals=[]
    for f in fixtures:
        fd=pd.to_datetime((f.get("fixture") or {}).get("date"),utc=True,errors="coerce")
        if pd.isna(fd) or (pd.notna(cutoff) and fd>=cutoff): continue
        fid=(f.get("fixture") or {}).get("id")
        if not fid: continue
        stats_resp=_api_football_get("fixtures/statistics",{"fixture":fid},api_key)
        if len(stats_resp)!=2: continue
        own=None; opp=None
        for block in stats_resp:
            tid=(block.get("team") or {}).get("id")
            xg=_extract_stat(block.get("statistics"),"expected_goals")
            if tid==int(team_id): own=xg
            else: opp=xg
        if own is not None and opp is not None:
            vals.append((float(own),float(opp)))
        if len(vals)>=sample: break
    if len(vals)<3:
        return {"available":False,"matches":len(vals)}
    return {
        "available":True,"matches":len(vals),
        "xg_for":float(np.mean([x[0] for x in vals])),
        "xg_against":float(np.mean([x[1] for x in vals])),
    }

def _external_context_adjustment(base_prob, home_xg, away_xg, pick, injuries, confirmed_lineup):
    """Bounded second-stage context adjustment.

    It never creates a probability from missing data. True xG is translated into
    a Poisson directional probability and blended modestly according to sample
    depth. Injury/lineup data modifies uncertainty only, not player 'value',
    because player-impact coefficients have not yet been learned historically.
    """
    p=float(base_prob)
    details=[]
    if home_xg.get("available") and away_xg.get("available"):
        # Expected scoring rates combine own creation and opponent prevention.
        lh=max(.15,(home_xg["xg_for"]+away_xg["xg_against"])/2)
        la=max(.15,(away_xg["xg_for"]+home_xg["xg_against"])/2)
        ph=pdra=pa=0.0
        for i in range(8):
            for j in range(8):
                pr=math.exp(-lh)*lh**i/math.factorial(i)*math.exp(-la)*la**j/math.factorial(j)
                if i>j: ph+=pr
                elif i<j: pa+=pr
                else: pdra+=pr
        xgp=ph if pick=="HOME" else pa
        sample=min(home_xg.get("matches",0),away_xg.get("matches",0))
        weight=min(.15,.03*sample)  # capped; model remains dominant
        p=(1-weight)*p+weight*xgp
        details.append(f"true xG blend {weight*100:.0f}% from {sample} recent provider matches")

    h_inj=len((injuries or {}).get("home",[]))
    a_inj=len((injuries or {}).get("away",[]))
    if h_inj or a_inj:
        details.append(f"verified absences: home {h_inj}, away {a_inj}")
    if confirmed_lineup:
        details.append("confirmed line-ups available")
    return float(np.clip(p,.01,.99)),details


def _pct_string(v):
    try:
        if isinstance(v,str): v=v.replace("%","").strip()
        x=float(v)
        return x if np.isfinite(x) else None
    except Exception:
        return None

@st.cache_data(ttl=21600,show_spinner=False)
def api_football_prediction(fixture_id, api_key):
    if not api_key or not fixture_id: return {"available":False}
    resp=_api_football_get("predictions",{"fixture":int(fixture_id)},api_key)
    if not resp: return {"available":False}
    item=resp[0] if isinstance(resp[0],dict) else {}
    pred=item.get("predictions") or {}; pct=pred.get("percent") or {}
    return {"available":True,"home":_pct_string(pct.get("home")),"draw":_pct_string(pct.get("draw")),
            "away":_pct_string(pct.get("away")),"winner":(pred.get("winner") or {}).get("name"),
            "advice":pred.get("advice"),"comparison":item.get("comparison") or {}}

@st.cache_data(ttl=21600,show_spinner=False)
def api_football_team_schedule_context(team_id, before_date, league_id, api_key):
    if not api_key or not team_id: return {"available":False}
    fixtures=_api_football_get("fixtures",{"team":int(team_id),"last":8,"status":"FT"},api_key)
    cutoff=pd.to_datetime(before_date,utc=True,errors="coerce")
    dates=[]; games14=0; other_comp14=0
    for f in fixtures:
        dt=pd.to_datetime((f.get("fixture") or {}).get("date"),utc=True,errors="coerce")
        if pd.isna(dt) or (pd.notna(cutoff) and dt>=cutoff): continue
        dates.append(dt)
        if pd.notna(cutoff) and 0 <= (cutoff-dt).days <= 14:
            games14+=1
            try:
                if int((f.get("league") or {}).get("id"))!=int(league_id): other_comp14+=1
            except Exception: pass
    if not dates: return {"available":False}
    rest=(cutoff-max(dates)).total_seconds()/86400 if pd.notna(cutoff) else None
    return {"available":True,"rest_days":round(float(rest),1) if rest is not None else None,
            "games14":int(games14),"other_comp_games14":int(other_comp14)}

def _player_importance_from_response(resp, team_id, league_id, team_played):
    if not resp: return {"available":False}
    item=resp[0] if isinstance(resp[0],dict) else {}; pobj=item.get("player") or {}; blocks=item.get("statistics") or []
    chosen=None
    for stx in blocks:
        try:
            if int((stx.get("team") or {}).get("id"))==int(team_id) and int((stx.get("league") or {}).get("id"))==int(league_id):
                chosen=stx; break
        except Exception: pass
    if chosen is None:
        for stx in blocks:
            try:
                if int((stx.get("team") or {}).get("id"))==int(team_id): chosen=stx; break
            except Exception: pass
    if chosen is None: return {"available":False,"player":pobj.get("name"),"player_id":pobj.get("id")}

    games=chosen.get("games") or {}; goals=chosen.get("goals") or {}
    appearances=float(games.get("appearences") or games.get("appearances") or 0)
    starts=float(games.get("lineups") or 0); minutes=float(games.get("minutes") or 0); pos=str(games.get("position") or "")
    try: rating=float(games.get("rating") or 0)
    except Exception: rating=0.0
    g=float(goals.get("total") or 0); a=float(goals.get("assists") or 0)
    denom=max(float(team_played or 0),appearances,starts,1.0)
    minute_share=float(np.clip(minutes/(denom*90.0),0,1)); start_share=float(np.clip(starts/denom,0,1))
    rating_score=float(np.clip((rating-6.0)/1.5,0,1)) if rating>0 else 0.35
    per90=(g+a)*90.0/max(minutes,90.0); p=pos.casefold()
    if "goal" in p:
        importance=65*minute_share+25*start_share+10*rating_score
    elif "def" in p:
        importance=55*minute_share+25*start_share+15*rating_score+5*float(np.clip(per90/.20,0,1))
    elif "mid" in p:
        importance=45*minute_share+25*start_share+15*rating_score+15*float(np.clip(per90/.50,0,1))
    else:
        importance=40*minute_share+25*start_share+15*rating_score+20*float(np.clip(per90/.75,0,1))
    importance=float(np.clip(importance,0,100))
    label="KEY" if importance>=70 else "IMPORTANT" if importance>=52 else "ROTATION" if importance>=30 else "DEPTH"
    return {"available":True,"player":pobj.get("name"),"player_id":pobj.get("id"),"position":pos or "Unknown",
            "importance":round(importance,1),"importance_label":label,"minutes":int(minutes),"starts":int(starts),
            "appearances":int(appearances),"goals":int(g),"assists":int(a),"rating":round(rating,2) if rating else None}

@st.cache_data(ttl=21600,show_spinner=False)
def api_football_player_importance(player_id, team_id, league_id, team_played, api_key, season_year=None):
    if not api_key or not player_id: return {"available":False}
    season_year=int(season_year or football_season_year_for_date())
    return _player_importance_from_response(_api_football_get("players",{"id":int(player_id),"season":int(season_year)},api_key),
                                            team_id,league_id,team_played)

@st.cache_data(ttl=21600,show_spinner=False)
def api_football_team_core_players(team_id, league_id, team_played, api_key, season_year=None):
    if not api_key or not team_id: return []
    season_year=int(season_year or football_season_year_for_date())
    out=[]
    for page in (1,2,3):
        batch=_api_football_get("players",{"team":int(team_id),"season":int(season_year),"page":page},api_key)
        if not batch: break
        for item in batch:
            imp=_player_importance_from_response([item],team_id,league_id,team_played)
            if imp.get("available"): out.append(imp)
        if len(batch)<20: break
    unique={}
    for x in out: unique[x.get("player_id")]=x
    return sorted(unique.values(),key=lambda x:float(x.get("importance",0)),reverse=True)

def _confirmed_lineup_rotation_summary(core_players, starters, bench, injury_ids):
    """Detect important normal starters who are benched/omitted for any reason."""
    if not starters or not core_players:
        return {"available":False,"burden":None,"missing_core":[]}
    starter_ids={x.get("id") for x in starters if x.get("id")}
    bench_ids={x.get("id") for x in (bench or []) if x.get("id")}
    injury_ids={x for x in (injury_ids or []) if x}
    missing=[]
    core=[x for x in core_players if float(x.get("importance",0))>=52][:14]
    for x in core:
        pid=x.get("player_id")
        if not pid or pid in starter_ids or pid in injury_ids: continue
        status="BENCH" if pid in bench_ids else "NOT IN MATCHDAY XI"
        weight=.45 if status=="BENCH" else 1.0
        missing.append({**x,"lineup_status":status,
                        "weighted_impact":round(float(x.get("importance",0))*weight,1)})
    burden=min(100.0,sum(float(x.get("weighted_impact",0)) for x in missing)/11.0)
    return {"available":True,"burden":round(burden,1),"missing_core":missing}

def _availability_team_summary(injuries, starter_rows, team_id, league_id, team_played, api_key, player_stats_supported=True, season_year=None, importance_lookup=None):
    season_year=int(season_year or football_season_year_for_date())
    starter_ids={x.get("id") for x in (starter_rows or []) if x.get("id")}; missing=[]; stale=[]; all_known=True
    for rec in injuries or []:
        pid=rec.get("player_id")
        if pid and pid in starter_ids:
            stale.append(rec.get("player")); continue
        imp=(importance_lookup or {}).get(pid) if pid else None
        if not isinstance(imp,dict):
            imp=api_football_player_importance(pid,team_id,league_id,team_played,api_key,season_year) if (pid and player_stats_supported) else {"available":False}
        if imp.get("available"):
            missing.append({**rec,**imp,"importance":round(float(imp.get("importance")),1)})
        else:
            all_known=False
            missing.append({**rec,"importance":None,"importance_label":"UNKNOWN"})
    missing=sorted(missing,key=lambda x:float(x.get("importance") or -1),reverse=True)
    burden=min(100.0,sum(float(x.get("importance") or 0) for x in missing)/11.0) if all_known else None
    return {"burden":round(burden,1) if burden is not None else None,
            "key_absences":sum(float(x.get("importance") or 0)>=70 for x in missing),
            "important_absences":sum(52<=float(x.get("importance") or 0)<70 for x in missing),
            "missing":missing,"importance_complete":all_known,
            "stale_reported_but_starting":[x for x in stale if x],"confirmed_lineup":bool(starter_rows)}

def _selected_values(row, home_value, away_value):
    p=str(row.get("Pick","")).upper()
    return (home_value,away_value) if p=="HOME" else (away_value,home_value) if p=="AWAY" else (None,None)

def _analyst_case(row):
    """Analyst score ranks the complete case; it is deliberately not a probability."""
    def num(v,default=None):
        try:
            x=float(v); return x if np.isfinite(x) else default
        except Exception: return default
    base=num(row.get("Ranking %"),num(row.get("Confidence %"),50.0))
    score=float(base); supports=[]; risks=[]; adjustments=[]; evidence=1; possible=8
    ctx=row.get("Context") if isinstance(row.get("Context"),dict) else {}; pick=str(row.get("Pick","")).upper()
    def add(label,delta,reason,positive=None):
        nonlocal score
        score+=delta; adjustments.append({"factor":label,"delta":round(delta,1),"reason":reason})
        if positive is True: supports.append(reason)
        elif positive is False: risks.append(reason)

    h=ctx.get("home") or {}; a=ctx.get("away") or {}
    hp=num(h.get("ppg")); ap=num(a.get("ppg")); hv=num(h.get("venue_ppg"),hp); av=num(a.get("venue_ppg"),ap)
    if None not in (hp,ap,hv,av):
        evidence+=1; d=(.65*(hp-ap)+.35*(hv-av))*(1 if pick=="HOME" else -1)
        if d>=.65: add("Recent/venue form",4,f"Recent and venue form strongly supports the pick ({d:+.2f} PPG edge).",True)
        elif d>=.25: add("Recent/venue form",2,f"Recent and venue form supports the pick ({d:+.2f} PPG edge).",True)
        elif d<=-.65: add("Recent/venue form",-5,f"Recent and venue form materially opposes the pick ({d:+.2f} PPG edge).",False)
        elif d<=-.25: add("Recent/venue form",-2.5,f"Recent and venue form slightly opposes the pick ({d:+.2f} PPG edge).",False)

    hpe=num(h.get("perf_vs_expect")); ape=num(a.get("perf_vs_expect"))
    if hpe is not None and ape is not None:
        evidence+=1; d=(hpe-ape)*(1 if pick=="HOME" else -1)
        if d>=.10: add("Opponent-adjusted performance",2.5,"Selected team has been outperforming pre-match expectation against recent opposition.",True)
        elif d<=-.10: add("Opponent-adjusted performance",-2.5,"Opponent-adjusted recent performance favours the other side.",False)

    hx=ctx.get("true_xg_home"); ax=ctx.get("true_xg_away")
    if isinstance(hx,dict) and hx.get("available") and isinstance(ax,dict) and ax.get("available"):
        evidence+=1
        hr=(num(hx.get("xg_for"),1.2)+num(ax.get("xg_against"),1.2))/2
        ar=(num(ax.get("xg_for"),1.0)+num(hx.get("xg_against"),1.0))/2
        d=(hr-ar)*(1 if pick=="HOME" else -1)
        if d>=.40: add("Provider xG",3.5,f"Provider xG supports the pick ({d:+.2f} expected-goal-rate edge).",True)
        elif d<=-.30: add("Provider xG",-4,f"Provider xG conflicts with the pick ({d:+.2f} expected-goal-rate edge).",False)
    else:
        gh=num(ctx.get("xg_like_home")); ga=num(ctx.get("xg_like_away"))
        if gh is not None and ga is not None:
            evidence+=1
            d=(gh-ga)*(1 if pick=="HOME" else -1)
            if d>=.45: add("Scoring profile",2,"Recent scoring/conceding profile supports the selection.",True)
            elif d<=-.35: add("Scoring profile",-2.5,"Recent scoring/conceding profile conflicts with the selection.",False)

    avx=ctx.get("availability") or {}; hav=avx.get("home") or {}; aav=avx.get("away") or {}
    hb=num(hav.get("total_burden"),num(hav.get("burden"))); ab=num(aav.get("total_burden"),num(aav.get("burden")))
    if hb is not None and ab is not None:
        evidence+=1; sb,ob=_selected_values(row,hb,ab); adv=ob-sb
        if adv>=12: add("Availability",7,f"Availability strongly favours the pick: opponent burden {ob:.1f} vs selected {sb:.1f}.",True)
        elif adv>=6: add("Availability",4,f"Availability favours the pick: opponent burden {ob:.1f} vs selected {sb:.1f}.",True)
        elif adv>=2.5: add("Availability",2,"Opponent carries the larger player-availability burden.",True)
        elif adv<=-12: add("Availability",-8,f"Major availability disadvantage: selected burden {sb:.1f} vs opponent {ob:.1f}.",False)
        elif adv<=-6: add("Availability",-5,f"Availability materially weakens the pick: selected burden {sb:.1f} vs opponent {ob:.1f}.",False)
        elif adv<=-2.5: add("Availability",-2.5,"Selected side carries the larger player-availability burden.",False)
        selected_av=hav if pick=="HOME" else aav
        key=[x for x in (selected_av.get("missing") or []) if float(x.get("importance",0) or 0)>=70]
        if key: risks.append("Key absence: "+", ".join(str(x.get("player","?")) for x in key[:3]))
        rot=(selected_av.get("rotation") or {}).get("missing_core") or []
        major_rot=[x for x in rot if float(x.get("importance",0) or 0)>=70]
        if major_rot:
            risks.append("Key normal starter not in confirmed XI: "+", ".join(f'{x.get("player","?")} ({x.get("lineup_status","")})' for x in major_rot[:3]))

    sch=ctx.get("schedule") or {}; hs=sch.get("home") or {}; aas=sch.get("away") or {}
    hr=num(hs.get("rest_days")); ar=num(aas.get("rest_days")); hg=num(hs.get("games14")); ag=num(aas.get("games14"))
    if None not in (hr,ar,hg,ag):
        evidence+=1; sr,orr=_selected_values(row,hr,ar); sg,og=_selected_values(row,hg,ag); total=0
        if sr-orr>=2.5: total+=2
        elif sr-orr<=-2.5: total-=2
        if og-sg>=2: total+=2
        elif og-sg<=-2: total-=2
        if total>0: add("Schedule",total,"Rest/fixture congestion is more favourable for selected team.",True)
        elif total<0: add("Schedule",total,"Rest/fixture congestion is more favourable for opponent.",False)
        selected_s=hs if pick=="HOME" else aas
        if num(selected_s.get("other_comp_games14"),0)>=2:
            risks.append("Selected team has had multiple non-league/cup/European matches in the last 14 days.")

    pp=ctx.get("provider_prediction") or {}
    if pp.get("available"):
        evidence+=1; psel=num(pp.get("home") if pick=="HOME" else pp.get("away"))
        if psel is not None:
            if psel>=62: add("Independent provider model",2.5,f"Independent provider forecast also supports selected side strongly ({psel:.0f}%).",True)
            elif psel<=42: add("Independent provider model",-3.5,f"Independent provider forecast is materially cooler on selected side ({psel:.0f}%).",False)

    market=num(row.get("Market fair %"))
    if market is not None:
        evidence+=1; gap=float(base)-market
        if abs(gap)<=5: add("Market confirmation",1.5,"Model and de-margined bookmaker market are broadly aligned.",True)
        elif gap>=12: add("Market disagreement",-3.5,f"Model is {gap:.1f}pp more confident than current market.",False)
        elif gap<=-8: add("Market confirmation",2,"Current market is even more confident in selected side than model.",True)

    draw=num(row.get("Draw %"),0)
    if draw>=base: add("Adversarial draw check",-7,"Draw is modelled at least as likely as selected-team win.",False)
    elif draw>=base-5: add("Adversarial draw check",-4,"Draw probability sits close to selected-team win probability.",False)
    if base<55: add("Base probability",-3,"Underlying win probability is below 55%; not a naturally strong anchor.",False)
    if str(row.get("Game state","")).upper()=="LIVE": add("Live state",-12,"Match already started; pre-match analyst logic is no longer clean.",False)

    completeness=round(100*min(evidence,possible)/possible,0)
    engine=str(row.get("Model engine",""))
    if completeness<35:
        add("Evidence quality",-8,"Too little corroborating evidence is currently available for anchor-level confidence.",False)
    elif completeness<50:
        add("Evidence quality",-4,"The probability has limited independent corroboration; treat it as a lower-confidence anchor.",False)
    if "market fallback" in engine.casefold():
        add("Probability source",-4,"This expanded fixture is ranked from bookmaker consensus because a dedicated/provider prediction was unavailable.",False)

    score=float(np.clip(score,0,100))
    severe=sum(float(x.get("delta",0))<=-4 for x in adjustments)
    verdict="ANCHOR" if score>=70 and severe==0 else "STRONG" if score>=63 and severe<=1 else "WATCH" if score>=56 else "CAUTION"
    return {"score":round(score,1),"verdict":verdict,"supports":supports[:6],"risks":risks[:6],
            "adjustments":adjustments,"evidence_completeness":completeness,"availability":avx}

def _anchor_stability(row):
    variants=[copy.deepcopy(row)]
    for kind in ("form","draw","market","availability","scoring"):
        r=copy.deepcopy(row); pick=str(r.get("Pick","")).upper()
        ctx=r.get("Context") if isinstance(r.get("Context"),dict) else {}
        if kind=="form":
            h=ctx.get("home") or {}; a=ctx.get("away") or {}
            if pick=="HOME":
                if h.get("ppg") is not None: h["ppg"]=max(0,float(h["ppg"])-.20)
                if a.get("ppg") is not None: a["ppg"]=float(a["ppg"])+.20
            elif pick=="AWAY":
                if a.get("ppg") is not None: a["ppg"]=max(0,float(a["ppg"])-.20)
                if h.get("ppg") is not None: h["ppg"]=float(h["ppg"])+.20
        elif kind=="draw":
            r["Draw %"]=float(r.get("Draw %") or 0)+5
        elif kind=="market" and r.get("Market fair %") is not None:
            r["Market fair %"]=float(r["Market fair %"])+5
        elif kind=="availability":
            av=ctx.get("availability") or {}; sel=av.get("home" if pick=="HOME" else "away") or {}; opp=av.get("away" if pick=="HOME" else "home") or {}
            if sel.get("burden") is not None: sel["burden"]=float(sel["burden"])+3
            if opp.get("burden") is not None: opp["burden"]=max(0,float(opp["burden"])-3)
        elif kind=="scoring":
            k="xg_like_home" if pick=="HOME" else "xg_like_away"
            if ctx.get(k) is not None: ctx[k]=max(.1,float(ctx[k])-.20)
        r["Context"]=ctx; variants.append(r)
    scores=[_analyst_case(v)["score"] for v in variants]
    base_case=_analyst_case(row); floor=min(scores); spread=max(scores)-min(scores); std=float(np.std(scores))
    evidence=float(base_case.get("evidence_completeness",0))
    grade="LOW" if evidence<50 or floor<55 or spread>10 else ("HIGH" if floor>=62 and std<=2.6 and spread<=7 else "MEDIUM")
    return {"grade":grade,"floor":round(floor,1),"range":round(spread,1),"std":round(std,2)}

def _late_info_status(row):
    if str(row.get("Game state","")).upper()!="UPCOMING": return "LIVE / CLOSED"
    ko=pd.to_datetime(row.get("Kickoff ISO"),utc=True,errors="coerce")
    if pd.isna(ko): return "KICKOFF UNKNOWN"
    mins=(ko-pd.Timestamp.now(tz="UTC")).total_seconds()/60
    ctx=row.get("Context") if isinstance(row.get("Context"),dict) else {}
    if ctx.get("lineups_confirmed"): return "XI VERIFIED"
    if mins<=0: return "STARTED"
    if mins<=120: return "REVALIDATE NOW"
    if mins<=240: return "RECHECK NEAR KICKOFF"
    return "PRE-MATCH CURRENT"

def _apply_analyst_engine(frame):
    out=frame.copy(); cases=[_analyst_case(r) for r in out.to_dict("records")]
    out["Analyst case"]=cases; out["Anchor score"]=[x["score"] for x in cases]
    out["Analyst verdict"]=[x["verdict"] for x in cases]; out["Evidence completeness %"]=[x["evidence_completeness"] for x in cases]
    out["Analyst risk count"]=[len(x["risks"]) for x in cases]
    stabs=[_anchor_stability(r) for r in out.to_dict("records")]
    out["Stability"]=[x["grade"] for x in stabs]; out["Stability floor"]=[x["floor"] for x in stabs]; out["Stability range"]=[x["range"] for x in stabs]
    out["Late info"]=[_late_info_status(r) for r in out.to_dict("records")]
    return out

def _anchor_five(frame):
    q=frame[frame["Pick"].isin(["HOME","AWAY"]) & frame["Game state"].eq("UPCOMING")].copy()
    if q.empty: return q
    if "Anchor readiness" in q.columns:
        q["_ready_rank"]=q["Anchor readiness"].astype(str).eq("FINAL").astype(int)
        q=q.sort_values(["_ready_rank","Anchor score","Ranking %"],ascending=[False,False,False]).drop(columns=["_ready_rank"])
    else:
        q=q.sort_values(["Anchor score","Ranking %"],ascending=[False,False])
    return q.head(5)

def build_analyst_target_acca(rows,target_odds=50.0,leg_counts=(5,6)):
    counts=sorted({int(x) for x in leg_counts if int(x)>0}); candidates=[]
    for row in rows:
        try: price=float(row.get("Best market odds")); prob=float(row.get("Ranking %"))/100; anchor=float(row.get("Anchor score"))/100
        except Exception: continue
        diag=row.get("Market diagnostic") if isinstance(row.get("Market diagnostic"),dict) else {}
        readiness=str(row.get("Anchor readiness","FINAL"))
        if (str(row.get("Pick","")).upper() in ("HOME","AWAY") and str(row.get("Game state","")).upper()=="UPCOMING"
            and readiness=="FINAL"
            and np.isfinite(price) and price>1.01 and 0<prob<1 and 0<anchor<=1 and diag.get("stage")=="accepted"):
            candidates.append({"row":row,"price":price,"prob":prob,"anchor":anchor})
    candidates=sorted(candidates,key=lambda x:(x["anchor"],x["prob"]),reverse=True)[:14]
    valid=[n for n in counts if n<=len(candidates)]
    if not valid:
        need=min(counts) if counts else 5
        return {"status":"INSUFFICIENT","legs":[],"reason":f"Only {len(candidates)} FINAL analyst selections have verified current prices; {need} required."}
    target=max(float(target_odds),1.01); reaching=[]; below=[]
    for n in valid:
        for combo in combinations(candidates,n):
            odds=float(np.prod([x["price"] for x in combo]))
            joint=float(np.prod([x["prob"] for x in combo]))
            mean_anchor=float(np.mean([x["anchor"] for x in combo]))
            quality=joint*(mean_anchor**2)
            rec=(combo,odds,joint,mean_anchor,quality)
            if odds>=target: reaching.append(rec)
            else: below.append(rec)
    if reaching:
        combo,odds,joint,mean_anchor,quality=max(reaching,key=lambda z:(z[4],z[2],z[3],-z[1]))
        return {"status":"READY","legs":[x["row"] for x in combo],"combined_odds":round(odds,2),
                "joint_probability":round(joint*100,2),"mean_anchor_score":round(mean_anchor*100,1),"target_odds":round(target,2)}
    if not below:
        return {"status":"INSUFFICIENT","legs":[],"reason":"No valid FINAL analyst combination could be formed."}
    safest=max(below,key=lambda z:(z[4],z[2],z[3],-z[1]))
    closest=max(below,key=lambda z:(z[1],z[4],z[2]))
    def pack(rec):
        combo,odds,joint,mean_anchor,quality=rec
        return {"legs":[x["row"] for x in combo],"combined_odds":round(odds,2),
                "joint_probability":round(joint*100,2),"mean_anchor_score":round(mean_anchor*100,1)}
    return {"status":"BELOW_TARGET","target_odds":round(target,2),
            "safest":pack(safest),"closest":pack(closest),
            "reason":"Target is not reachable with the requested FINAL leg count and verified prices."}

@st.cache_resource(show_spinner=False)
def _external_analysis_registry():
    return {}

def _row_kickoff_or_date(row):
    v=row.get("Kickoff ISO")
    if v is not None and not pd.isna(v) and str(v).strip() and str(v).lower()!="nan":
        return v
    return row.get("Match date")

def _deep_fixture_analysis(row, api_key, include_xg=False, include_lineup_core=False, full_context=True):
    lname=str(row.get("League",""))
    lid=row.get("Provider league ID") or API_FOOTBALL_LEAGUES.get(lname)
    season_year=int(row.get("Provider season") or football_season_year_for_date(row.get("Match date")))
    provider_fixture_id=row.get("Provider fixture ID")
    if provider_fixture_id and lid:
        cov=api_football_coverage(lname,api_key,lid,season_year)
        injuries=_api_football_get("injuries",{"fixture":int(provider_fixture_id)},api_key) if cov.get("injuries") else []
        lineups=_api_football_get("fixtures/lineups",{"fixture":int(provider_fixture_id)},api_key) if cov.get("lineups") else []
        home=str(row.get("Home team","")); away=str(row.get("Away team",""))
        home_id=row.get("Provider home team ID"); away_id=row.get("Provider away team ID")
        home_inj=[]; away_inj=[]
        for x in injuries:
            team=(x.get("team") or {}).get("name",""); pobj=x.get("player") or {}
            rec={"player":pobj.get("name",""),"player_id":pobj.get("id"),
                 "reason":pobj.get("reason") or x.get("reason") or "",
                 "type":x.get("type") or pobj.get("type") or "Unavailable"}
            if team_match(team,home): home_inj.append(rec)
            elif team_match(team,away): away_inj.append(rec)
        confirmed={"home":None,"away":None}; benches={"home":None,"away":None}
        for lu in lineups:
            tn=(lu.get("team") or {}).get("name",""); starters=[]; bench=[]
            for p in lu.get("startXI") or []:
                po=p.get("player") or {}
                if po.get("name"): starters.append({"id":po.get("id"),"name":po.get("name"),"pos":po.get("pos")})
            for p in lu.get("substitutes") or []:
                po=p.get("player") or {}
                if po.get("name"): bench.append({"id":po.get("id"),"name":po.get("name"),"pos":po.get("pos")})
            if team_match(tn,home): confirmed["home"]=starters or None; benches["home"]=bench or None
            elif team_match(tn,away): confirmed["away"]=starters or None; benches["away"]=bench or None
        ext={"available":True,"fixture_id":provider_fixture_id,"home_team_id":home_id,"away_team_id":away_id,
             "injuries":{"home":home_inj,"away":away_inj},"confirmed_lineup":confirmed,"confirmed_bench":benches,
             "lineups_confirmed":bool(confirmed["home"] and confirmed["away"]),"coverage":cov}
    else:
        ext=api_football_fixture_context(lname,row.get("Match date"),row.get("Home team"),row.get("Away team"),api_key,
                                         league_id=lid,season_year=season_year)
    if not ext.get("available"): return {"available":False,"reason":ext.get("reason","Provider fixture unavailable")}
    ctx=row.get("Context") if isinstance(row.get("Context"),dict) else {}
    hp=((ctx.get("home") or {}).get("played")) or 0; ap=((ctx.get("away") or {}).get("played")) or 0
    lines=ext.get("confirmed_lineup") or {}; inj=ext.get("injuries") or {}; cov=ext.get("coverage") or {}
    hcore=[]; acore=[]
    if cov.get("injuries") and (inj.get("home") or inj.get("away")):
        hcore=api_football_team_core_players(ext.get("home_team_id"),lid,hp,api_key,season_year)
        acore=api_football_team_core_players(ext.get("away_team_id"),lid,ap,api_key,season_year)
    hlookup={x.get("player_id"):x for x in hcore if x.get("player_id")}
    alookup={x.get("player_id"):x for x in acore if x.get("player_id")}
    if cov.get("injuries"):
        hav=_availability_team_summary(inj.get("home"),lines.get("home"),ext.get("home_team_id"),lid,hp,api_key,player_stats_supported=True,season_year=season_year,importance_lookup=hlookup)
        aav=_availability_team_summary(inj.get("away"),lines.get("away"),ext.get("away_team_id"),lid,ap,api_key,player_stats_supported=True,season_year=season_year,importance_lookup=alookup)
        hav["coverage"]=True; aav["coverage"]=True
    else:
        hav={"burden":None,"missing":[],"coverage":False}
        aav={"burden":None,"missing":[],"coverage":False}
    pred=api_football_prediction(ext.get("fixture_id"),api_key) if (full_context and cov.get("predictions")) else {"available":False}
    hs=api_football_team_schedule_context(ext.get("home_team_id"),_row_kickoff_or_date(row),lid,api_key) if full_context else {"available":False}
    aas=api_football_team_schedule_context(ext.get("away_team_id"),_row_kickoff_or_date(row),lid,api_key) if full_context else {"available":False}
    hx=ax={"available":False}
    if include_xg and cov.get("fixture_statistics"):
        hx=api_football_recent_xg(ext.get("home_team_id"),_row_kickoff_or_date(row),api_key,3)
        ax=api_football_recent_xg(ext.get("away_team_id"),_row_kickoff_or_date(row),api_key,3)

    if include_lineup_core and ext.get("lineups_confirmed"):
        benches=ext.get("confirmed_bench") or {}
        if not hcore: hcore=api_football_team_core_players(ext.get("home_team_id"),lid,hp,api_key,season_year)
        if not acore: acore=api_football_team_core_players(ext.get("away_team_id"),lid,ap,api_key,season_year)
        hinj={x.get("player_id") for x in (inj.get("home") or [])}
        ainj={x.get("player_id") for x in (inj.get("away") or [])}
        hrot=_confirmed_lineup_rotation_summary(hcore,lines.get("home"),benches.get("home"),hinj)
        arot=_confirmed_lineup_rotation_summary(acore,lines.get("away"),benches.get("away"),ainj)
        hav["rotation"]=hrot; aav["rotation"]=arot
        if hav.get("burden") is not None and hrot.get("burden") is not None:
            hav["total_burden"]=round(float(hav["burden"])+float(hrot["burden"]),1)
        if aav.get("burden") is not None and arot.get("burden") is not None:
            aav["total_burden"]=round(float(aav["burden"])+float(arot["burden"]),1)

    return {"available":True,"verification_stage":"FULL" if full_context else "SCREEN","verified_at_utc":datetime.now(timezone.utc).isoformat(),"availability":{"home":hav,"away":aav},"confirmed_lineup":lines,
            "confirmed_bench":ext.get("confirmed_bench"),
            "lineups_confirmed":ext.get("lineups_confirmed",False),"coverage":cov,"provider_prediction":pred,
            "schedule":{"home":hs,"away":aas},"true_xg_home":hx if hx.get("available") else None,
            "true_xg_away":ax if ax.get("available") else None}

ODDS_BASE="https://api.the-odds-api.com/v4/sports"


# ---------------------------------------------------------------------
# V26 EXPANDED FIXTURE UNIVERSE
# API-Football is the discovery layer; dedicated OpenFootball models remain
# the preferred probability source where available.
# ---------------------------------------------------------------------
PROVIDER_PREDICTION_BUDGET=24
PROVIDER_MARKET_BUDGET=12
V28_ANALYST_NETWORK_BUDGET=80

def _competition_text(fx):
    lg=fx.get("league") or {}
    return str(lg.get("name","")).strip(),str(lg.get("country","")).strip()

def _is_youth_or_womens_competition(name):
    n=str(name).casefold()
    blocked=("women","womens","femin","u17","u18","u19","u20","u21","u23",
             "premier league 2","professional development","youth","reserve")
    return any(x in n for x in blocked)

def provider_competition_allowed(fx, universe="UK + Major Europe"):
    """Filter the global fixture feed to competitions useful for this product."""
    name,country=_competition_text(fx)
    n=name.casefold(); c=country.casefold()
    if not name:
        return False
    # Youth sides inside the senior EFL Trophy are valid; youth competitions are not.
    if _is_youth_or_womens_competition(name) and "efl trophy" not in n:
        return False
    if "friendly" in n:
        return False

    england_terms=(
        "premier league","championship","league one","league two","national league",
        "fa cup","league cup","efl cup","efl trophy","community shield"
    )
    scotland_terms=("premiership","championship","league one","league two","fa cup","league cup")
    wales_terms=("cymru premier","welsh cup")
    ni_terms=("premiership","irish cup","league cup")
    ireland_terms=("premier division","fai cup")
    uefa_terms=("uefa champions league","uefa europa league","uefa conference league")
    major={
        "germany":("bundesliga","2. bundesliga","3. liga","dfb pokal"),
        "spain":("la liga","segunda","copa del rey"),
        "italy":("serie a","serie b","coppa italia"),
        "france":("ligue 1","ligue 2","coupe de france"),
        "netherlands":("eredivisie","knvb beker"),
        "portugal":("primeira liga","taça de portugal","taca de portugal"),
        "belgium":("jupiler pro league","pro league","cup"),
        "turkey":("süper lig","super lig","cup"),
        "greece":("super league","cup"),
        "austria":("bundesliga","cup"),
        "switzerland":("super league","cup"),
        "denmark":("superliga","cup"),
        "norway":("eliteserien","cup"),
        "sweden":("allsvenskan","superettan","cup"),
    }

    if c=="england":
        return any(x in n for x in england_terms)
    if c=="scotland":
        return any(x in n for x in scotland_terms)
    if c=="wales":
        return any(x in n for x in wales_terms)
    if c in ("northern-ireland","northern ireland"):
        return any(x in n for x in ni_terms)
    if c=="ireland" and universe!="UK only":
        return any(x in n for x in ireland_terms)
    if any(x in n for x in uefa_terms):
        return True
    if universe=="UK only":
        return False
    if universe=="Core model leagues only":
        return False
    terms=major.get(c)
    return bool(terms and any(x in n for x in terms))

def provider_competition_priority(fx):
    name,country=_competition_text(fx)
    n=name.casefold(); c=country.casefold()
    if "efl trophy" in n: return 100
    if c=="england" and any(x in n for x in ("fa cup","league cup","efl cup","national league")): return 95
    if any(x in n for x in ("uefa champions league","uefa europa league","uefa conference league")): return 92
    if c in ("england","scotland"): return 88
    if any(x in n for x in ("cup","copa","coppa","pokal","coupe","beker")): return 82
    return 75

V28_PRIORITY_EXPANDED_COMPETITIONS={
    "UEFA Champions League":2,
    "UEFA Europa League":3,
    "UEFA Conference League":848,
    "National League":43,
    "FA Cup":45,
    "EFL Trophy":46,
    "EFL Cup":48,
}

@st.cache_data(ttl=900,show_spinner=False)
def api_football_fixture_range(start_iso, end_iso, api_key):
    """Discover the broad window, but fail over to priority competitions if global paging is too large."""
    if not api_key: return []
    params={"from":str(start_iso),"to":str(end_iso),"timezone":"Europe/London"}
    try:
        return _api_football_get_all("fixtures",params,api_key,max_pages=12)
    except ApiBudgetDeferred:
        raise
    except Exception:
        # A global world-football range can be too heavily paginated. In that
        # case preserve the product's most valuable expanded competitions rather
        # than silently returning only page 1.
        season=football_season_year_for_date(start_iso)
        out=[]
        for name,lid in V28_PRIORITY_EXPANDED_COMPETITIONS.items():
            try:
                _api_budget_guard(reserve_daily=12,reserve_minute=1)
                q={"league":int(lid),"season":int(season),"from":str(start_iso),"to":str(end_iso),"timezone":"Europe/London"}
                out.extend(_api_football_get_all("fixtures",q,api_key,max_pages=4))
            except ApiBudgetDeferred:
                break
            except Exception:
                continue
        return out

def _provider_fixture_state(fx):
    stx=((fx.get("fixture") or {}).get("status") or {})
    short=str(stx.get("short","")).upper()
    if short in ("FT","AET","PEN","CANC","ABD","AWD","WO","PST"):
        return "FINAL" if short in ("FT","AET","PEN") else "UNAVAILABLE"
    if short in ("NS","TBD"):
        return "UPCOMING"
    return "LIVE"

def _provider_fixture_key(fx):
    teams=fx.get("teams") or {}; h=(teams.get("home") or {}).get("name",""); a=(teams.get("away") or {}).get("name","")
    dt=pd.to_datetime((fx.get("fixture") or {}).get("date"),errors="coerce")
    day=dt.date().isoformat() if pd.notna(dt) else ""
    return (norm(h),norm(a),day)

def _existing_row_key(row):
    return (norm(row.get("Home team","")),norm(row.get("Away team","")),str(row.get("Match date",""))[:10])

def provider_match_for_existing(home,away,match_day,fixtures):
    candidates=[]
    for fx in fixtures or []:
        teams=fx.get("teams") or {}
        h=(teams.get("home") or {}).get("name",""); a=(teams.get("away") or {}).get("name","")
        dt=pd.to_datetime((fx.get("fixture") or {}).get("date"),errors="coerce")
        if pd.isna(dt) or dt.date().isoformat()!=str(match_day)[:10]: continue
        if team_match(home,h) and team_match(away,a):
            candidates.append(fx)
    return candidates[0] if len(candidates)==1 else None

def provider_fixture_duplicates_existing(fx, rows):
    teams=fx.get("teams") or {}
    h=(teams.get("home") or {}).get("name",""); a=(teams.get("away") or {}).get("name","")
    dt=pd.to_datetime((fx.get("fixture") or {}).get("date"),errors="coerce")
    if pd.isna(dt): return False
    day=dt.date().isoformat()
    for r in rows:
        if str(r.get("Match date",""))[:10]!=day: continue
        if team_match(h,r.get("Home team","")) and team_match(a,r.get("Away team","")):
            return True
    return False

@st.cache_data(ttl=900,show_spinner=False)
def api_football_fixture_odds(fixture_id, home, away, api_key):
    """Parse API-Football pre-match 1X2 odds for competitions The Odds API lacks."""
    if not api_key or not fixture_id:
        return None
    resp=_api_football_get_all("odds",{"fixture":int(fixture_id)},api_key,max_pages=6)
    if not resp:
        return None
    root=resp[0] if isinstance(resp[0],dict) else {}
    books=[]
    for bk in root.get("bookmakers") or []:
        chosen=None
        for bet in bk.get("bets") or []:
            nm=str(bet.get("name","")).casefold()
            if nm in ("match winner","1x2","fulltime result","full time result") or ("match" in nm and "winner" in nm):
                chosen=bet; break
        if not chosen: continue
        mapped={"HOME":None,"DRAW":None,"AWAY":None}
        for v in chosen.get("values") or []:
            label=str(v.get("value","")).strip()
            try: odd=float(v.get("odd"))
            except Exception: continue
            if not np.isfinite(odd) or odd<=1.01 or odd>100: continue
            low=label.casefold()
            if low in ("home","1") or team_match(label,home): mapped["HOME"]=odd
            elif low in ("draw","x","tie"): mapped["DRAW"]=odd
            elif low in ("away","2") or team_match(label,away): mapped["AWAY"]=odd
        if all(mapped.values()):
            inv=np.array([1/mapped["HOME"],1/mapped["DRAW"],1/mapped["AWAY"]],dtype=float)
            over=float(inv.sum())
            if .95<=over<=1.25:
                fair=inv/over
                books.append({"bookmaker":bk.get("name","Unknown"),
                              "odds":[mapped["HOME"],mapped["DRAW"],mapped["AWAY"]],
                              "fair":fair.tolist()})
    if not books:
        return None
    arr=np.array([x["odds"] for x in books],dtype=float)
    med=np.median(arr,axis=0); best=np.max(arr,axis=0)
    inv=1/med; fair=inv/inv.sum()
    return {"median":med.tolist(),"best":best.tolist(),"fair":fair.tolist(),"books":len(books),
            "integrity":"OK","detail":books,"source":"API-Football"}

def _provider_prediction_to_row(fx, pred, market=None):
    fixture=fx.get("fixture") or {}; teams=fx.get("teams") or {}; lg=fx.get("league") or {}
    home=(teams.get("home") or {}).get("name",""); away=(teams.get("away") or {}).get("name","")
    hp=_pct_string(pred.get("home")); dp=_pct_string(pred.get("draw")); ap=_pct_string(pred.get("away"))
    source="API-Football prediction"
    if None in (hp,dp,ap):
        if market is None:
            return None
        hp,dp,ap=[float(x)*100 for x in market["fair"]]
        source="API-Football market fallback"
    probs=np.array([hp,dp,ap],dtype=float)
    if not np.isfinite(probs).all() or probs.sum()<=0:
        return None
    probs=probs/probs.sum()
    i=0 if probs[0]>=probs[2] else 2
    conf=float(probs[i])
    if conf<=0 or not np.isfinite(conf):
        return None
    dt=pd.to_datetime(fixture.get("date"),errors="coerce")
    if pd.isna(dt): return None
    match_day=dt.date().isoformat()
    kickoff_iso=str(fixture.get("date") or "")
    _,kickoff_label=kickoff_uk_from_iso(kickoff_iso)
    labels=["HOME","DRAW","AWAY"]
    odd=best_odd=mprob=edge=ev=None; books=None; integrity=None; detail=[]; diag={"stage":"no-events","reason":"Provider prediction available; no verified market loaded yet","trace":[]}
    if market:
        odd=float(market["median"][i]); best_odd=float(market["best"][i]); mprob=float(market["fair"][i]); edge=conf-mprob; ev=conf*best_odd-1
        books=int(market["books"]); integrity=market.get("integrity"); detail=market.get("detail",[])
        diag={"stage":"accepted","reason":"Verified API-Football 1X2 market","trace":[],"provider":"API-Football"}
    return {
        "League":str(lg.get("name") or "Provider competition"),
        "Competition country":str(lg.get("country") or ""),
        "Match":f"{home} v {away}","Match date":match_day,"Home team":home,"Away team":away,"Pick":labels[i],
        "Home %":round(probs[0]*100,1),"Draw %":round(probs[1]*100,1),"Away %":round(probs[2]*100,1),
        "Confidence %":round(conf*100,1),"Fair odds":round(1/conf,2),
        "Market odds":round(odd,2) if odd else None,"Market fair %":round(mprob*100,1) if mprob else None,
        "Edge pp":round(edge*100,1) if edge is not None else None,"EV %":round(ev*100,1) if ev is not None else None,
        "Bookmakers":books,"Best market odds":round(best_odd,2) if best_odd else None,
        "Market integrity":integrity,"Bookmaker detail":detail,"Market diagnostic":diag,
        "Kickoff ISO":kickoff_iso,"Kickoff UK":kickoff_label,"Game state":"UPCOMING",
        "Timing label":"Upcoming","Minutes since kickoff":None,
        "Decision":"RANKED" if market else "PREDICTION ONLY",
        "Decision reason":f"Expanded-universe fixture ranked from {source}. "+("A verified current 1X2 market is attached." if market else "No current 1X2 market has been verified yet."),
        "Training matches":None,"Model engine":source,"Validation":"PROVIDER FALLBACK",
        "Validation evidence":"Used because this competition is outside the dedicated OpenFootball model set.",
        "Secondary checks":[],"Context":{"available":True,"home":{},"away":{},"injuries":"UNAVAILABLE — verified analyst pass not run","provider_source":source},
        "Provider fixture ID":fixture.get("id"),"Provider league ID":lg.get("id"),"Provider season":lg.get("season"),
        "Provider home team ID":(teams.get("home") or {}).get("id"),"Provider away team ID":(teams.get("away") or {}).get("id"),
        "Fixture source":"API-Football expanded universe",
    }

def norm(s):
    import re, unicodedata
    s=unicodedata.normalize("NFKD",str(s)).encode("ascii","ignore").decode().lower()
    s=s.replace("&"," and ")
    s=re.sub(r"\b(fc|cf|afc|ac|calcio|club|and)\b"," ",s)
    return re.sub(r"[^a-z0-9]","",s)

# Explicit aliases are safer than increasingly loose fuzzy matching for money data.
# Keys and values are normalized forms. Add only verified club-name variants here.
TEAM_ALIASES = {
    "rayovallecanodemadrid":"rayovallecano",
    "rcdespanyoldebarcelona":"espanyol",
    "reialclubdeportiuespanyoldebarcelona":"espanyol",
    "rcdespanyol":"espanyol",
    "deportivoalaves":"alaves",
    "deportivoalaves":"alaves",
    "athleticclubbilbao":"athleticclub",
    "athleticbilbao":"athleticclub",
    "realbetisbalompie":"realbetis",
    "rcdmalorca":"mallorca",
    "realclubdeportivomallorca":"mallorca",
    "rceltadevigo":"celtavigo",
    "realclubceltadevigo":"celtavigo",
    "realoviedo":"oviedo",
}

def canonical_team(s):
    x=norm(s)
    return TEAM_ALIASES.get(x,x)

def team_match(a,b):
    """Fail-closed team match for market data: canonical equality or safe long prefix."""
    x,y=canonical_team(a),canonical_team(b)
    if not x or not y: return False
    if x==y: return True
    return min(len(x),len(y))>=8 and (x.startswith(y) or y.startswith(x))


@st.cache_data(ttl=300,show_spinner=False)
def odds_fetch(api_key,sport_key):
    # V15.4: preserve the actual HTTP/API response diagnostics. Never expose apiKey.
    url=f"{ODDS_BASE}/{sport_key}/odds/"
    safe_params={"regions":"uk","markets":"h2h","oddsFormat":"decimal","dateFormat":"iso"}
    params={"apiKey":api_key,**safe_params}
    r=requests.get(url,params=params,timeout=25)
    meta={
        "endpoint":f"/v4/sports/{sport_key}/odds/",
        "sport_key":sport_key,
        "region":"uk",
        "market":"h2h",
        "http_status":r.status_code,
        "remaining":r.headers.get("x-requests-remaining"),
        "used":r.headers.get("x-requests-used"),
        "last":r.headers.get("x-requests-last"),
        "events":0,
        "api_message":"",
    }
    try:
        payload=r.json()
    except Exception:
        payload=None
        meta["api_message"]=(r.text or "")[:300]
    if isinstance(payload,dict):
        meta["api_message"]=str(payload.get("message") or payload.get("error") or payload.get("code") or "")[:300]
    if r.status_code in (401,403):
        raise RuntimeError(f"Odds API key rejected (HTTP {r.status_code}).")
    if r.status_code==429:
        raise RuntimeError("Odds API usage allowance reached (HTTP 429).")
    if r.status_code>=400:
        raise RuntimeError(f"Odds API HTTP {r.status_code}: {meta['api_message'] or 'request failed'}")
    data=payload if isinstance(payload,list) else []
    meta["events"]=len(data)
    if not isinstance(payload,list) and not meta["api_message"]:
        meta["api_message"]="Unexpected non-list response from odds endpoint"
    return data,meta

def _event_date_utc(e):
    try:
        return pd.to_datetime(e.get("commence_time"),utc=True).date()
    except Exception:
        return None

def kickoff_uk_from_iso(value):
    """Convert The Odds API UTC kickoff to Europe/London, including BST/GMT."""
    if not value:
        return None, None
    try:
        from zoneinfo import ZoneInfo
        dt=pd.to_datetime(value,utc=True).to_pydatetime().astimezone(ZoneInfo("Europe/London"))
        return dt, dt.strftime("%a %d %b • %H:%M UK")
    except Exception:
        return None, None

def matched_kickoff_from_diag(matchdiag):
    """Recover kickoff from the unique event trace even if bookmaker validation later fails."""
    if not isinstance(matchdiag,dict): return None
    hits=[t for t in matchdiag.get("trace",[]) if t.get("Home match") and t.get("Away match") and t.get("Date match")]
    if len(hits)==1:
        return hits[0].get("API time") or None
    return None

def fixture_kickoff_iso(match):
    """Best-effort scheduled kickoff from the fixture feed.

    OpenFootball commonly supplies date + local time. We convert Europe/London
    local fixture time to UTC so the existing live/upcoming/stale logic can be
    reused even when no bookmaker market is currently matched.
    """
    try:
        day=str(match.get("date",""))[:10]
        tm=str(match.get("time","")).strip()
        if not day or not tm:
            return None
        from zoneinfo import ZoneInfo
        local=pd.Timestamp(f"{day} {tm}").to_pydatetime().replace(tzinfo=ZoneInfo("Europe/London"))
        return local.astimezone(timezone.utc).isoformat()
    except Exception:
        return None

def consensus_for(events,home,away,fixture_date=None):
    """Fail-closed UK soccer 1X2 consensus plus rejection diagnostics."""
    trace=[]; candidates=[]
    for e in events:
        eh=e.get("home_team",""); ea=e.get("away_team",""); ed=_event_date_utc(e)
        hm=team_match(home,eh); am=team_match(away,ea)
        date_ok=(fixture_date is None) or (ed is not None and abs((ed-fixture_date).days)<=1)
        trace.append({"API event":f"{eh} v {ea}","API time":e.get("commence_time",""),
                      "Home match":hm,"Away match":am,"Date match":date_ok,
                      "OpenFootball":f"{home} v {away}"})
        if hm and am and date_ok: candidates.append(e)
    if len(candidates)!=1:
        return None,{"stage":"event","reason":f"Expected exactly 1 matching event; found {len(candidates)}",
                    "trace":trace,"rejected_books":[]}
    event=candidates[0]

    valid=[]; rejected=[]
    for b in event.get("bookmakers",[]):
        title=b.get("title",b.get("key","Unknown"))
        h2h=[m for m in b.get("markets",[]) if m.get("key")=="h2h"]
        if len(h2h)!=1:
            rejected.append({"Bookmaker":title,"Reason":f"h2h count {len(h2h)}"}); continue
        vals={"H":[],"D":[],"A":[]}; seen=[]
        for o in h2h[0].get("outcomes",[]):
            n=str(o.get("name","")).strip(); p=o.get("price"); seen.append(f"{n}={p}")
            if not isinstance(p,(int,float)) or not np.isfinite(p) or p<=1.01 or p>100: continue
            if n.casefold()=="draw": vals["D"].append(float(p))
            elif team_match(home,n): vals["H"].append(float(p))
            elif team_match(away,n): vals["A"].append(float(p))
        if any(len(vals[k])!=1 for k in ("H","D","A")):
            rejected.append({"Bookmaker":title,"Reason":"H/D/A mapping failed","Outcomes":" | ".join(seen)}); continue
        prices=np.array([vals["H"][0],vals["D"][0],vals["A"][0]],dtype=float)
        overround=float((1/prices).sum())
        if not (0.95 <= overround <= 1.20):
            rejected.append({"Bookmaker":title,"Reason":f"overround {overround:.3f}","Outcomes":" | ".join(seen)}); continue
        fair=(1/prices)/overround
        valid.append({"book":title,"prices":prices,"fair":fair,"overround":overround})

    if not valid:
        return None,{"stage":"bookmaker","reason":"Event matched, but no bookmaker passed 1X2 integrity checks",
                    "trace":trace,"rejected_books":rejected}
    arr=np.vstack([v["prices"] for v in valid]); fairs=np.vstack([v["fair"] for v in valid])
    median_prices=np.median(arr,axis=0)
    fair=np.median(fairs,axis=0); fair=fair/fair.sum(); best_prices=np.max(arr,axis=0)
    q25=np.percentile(arr,25,axis=0); q75=np.percentile(arr,75,axis=0)
    dispersion=np.max((q75-q25)/np.maximum(median_prices,1e-9))
    integrity="OK" if dispersion<=0.35 else "VERIFY"
    detail=[{"Bookmaker":v["book"],"Home":round(v["prices"][0],3),"Draw":round(v["prices"][1],3),
             "Away":round(v["prices"][2],3),"Overround %":round(v["overround"]*100,1)} for v in valid]
    market={"median":median_prices,"fair":fair,"best":best_prices,"books":len(valid),
            "integrity":integrity,"detail":detail,"event":event,"rejected":rejected}
    return market,{"stage":"accepted","reason":f"Matched event; {len(valid)} valid bookmaker(s), {len(rejected)} rejected",
                   "trace":trace,"rejected_books":rejected}


def secondary_auto_verify(market, pick_idx, conf, min_edge_pp, validation_status, matchdiag, kickoff_iso):
    """Fail-closed secondary audit for candidates that would previously need manual verification.
    Uses only independently observed bookmaker rows already accepted by the 1X2 parser.
    It never invents team-news/injury evidence.
    """
    checks=[]
    if validation_status == "RAW FALLBACK":
        return "PASS", "Secondary audit stopped: this league uses RAW FALLBACK because calibrated validation is unavailable.", checks
    if not market or matchdiag.get("stage") != "accepted" or not kickoff_iso:
        return "PASS", "Secondary audit failed fixture/market/kickoff identity checks.", checks
    detail=market.get("detail",[])
    if len(detail) < 8:
        return "PASS", f"Secondary audit failed bookmaker depth ({len(detail)} valid books).", checks
    col=("Home","Draw","Away")[pick_idx]
    prices=np.array([float(x[col]) for x in detail if isinstance(x.get(col),(int,float))],dtype=float)
    if len(prices)<8:
        return "PASS", "Secondary audit could not reconstruct enough independent prices.", checks
    # Robust price agreement: compare the middle 50% and remove dependence on the single best quote.
    q25,q50,q75=np.percentile(prices,[25,50,75])
    rel_iqr=(q75-q25)/max(q50,1e-9)
    checks.append(f"{len(prices)} accepted bookmakers; selected-price IQR {rel_iqr*100:.1f}%")
    if rel_iqr > .20:
        return "PASS", "Secondary audit rejected the signal because bookmaker prices are too dispersed.", checks
    # Rebuild each bookmaker's de-margined probability for the selected outcome.
    fairs=[]
    for x in detail:
        ps=np.array([x.get("Home"),x.get("Draw"),x.get("Away")],dtype=float)
        if np.all(np.isfinite(ps)) and np.all(ps>1.01):
            inv=1/ps; fairs.append(float((inv/inv.sum())[pick_idx]))
    if len(fairs)<8:
        return "PASS", "Secondary audit could not reconstruct bookmaker fair probabilities.", checks
    # Use the 75th percentile market probability: a deliberately tougher market comparison than the median.
    tough_market=float(np.percentile(fairs,75))
    robust_edge=conf-tough_market
    median_ev=conf*q50-1
    checks.append(f"Robust edge vs 75th-percentile market: {robust_edge*100:.1f}pp")
    checks.append(f"EV at median bookmaker price (not best price): {median_ev*100:.1f}%")
    rejected=len(market.get("rejected",[])); total=len(detail)+rejected
    reject_rate=(rejected/total) if total else 1
    checks.append(f"Bookmaker rejection rate: {reject_rate*100:.1f}%")
    if reject_rate > .35:
        return "PASS", "Secondary audit rejected the signal because too many bookmaker markets failed integrity checks.", checks
    if robust_edge < min_edge_pp/100 or median_ev <= 0:
        return "PASS", "Secondary audit removed the apparent value when tested against tougher consensus assumptions.", checks
    return "BET", "SECONDARY AUTO VERIFIED — the value survives a tougher bookmaker-consensus audit without relying on the best quote.", checks


@st.cache_data(ttl=3600,show_spinner=False)
def chronological_backtest(code):
    f,_,_,_=make_training(code)
    if len(f)<500: raise RuntimeError("Not enough historical matches for a meaningful holdout.")
    f=f.reset_index(drop=True); cut=int(len(f)*.80)
    tr,te=f.iloc[:cut],f.iloc[cut:]
    model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,learning_rate=.045,min_samples_leaf=38,l2_regularization=8,random_state=42)
    model.fit(tr[FEATURES].fillna(0),tr["y"])
    proba=model.predict_proba(te[FEATURES].fillna(0))
    actual=te["y"].to_numpy(); pred=np.argmax(proba,axis=1); conf=np.max(proba,axis=1)
    correct=pred==actual; rows=[]
    for lo,hi in [(0,.50),(.50,.55),(.55,.60),(.60,.65),(.65,.70),(.70,.75),(.75,1.01)]:
        m=(conf>=lo)&(conf<hi)
        if m.sum():
            rows.append({"Confidence band":f"{int(lo*100)}–{int(min(hi,1)*100)}%",
                         "Predictions":int(m.sum()),
                         "Average stated %":round(float(conf[m].mean()*100),1),
                         "Actual win %":round(float(correct[m].mean()*100),1),
                         "Calibration gap pp":round(float((conf[m].mean()-correct[m].mean())*100),1)})
    onehot=np.eye(3)[actual]
    brier=float(np.mean(np.sum((proba-onehot)**2,axis=1)))
    return {"train_n":len(tr),"test_n":len(te),"accuracy":float(correct.mean()),
            "logloss":float(log_loss(actual,proba,labels=[0,1,2])),
            "brier":brier,"bands":pd.DataFrame(rows)}

def calibration_grade(gap):
    a=abs(float(gap))
    return "GOOD" if a<=3 else ("WATCH" if a<=6 else "POOR")

show_admin_lab=st.sidebar.toggle(
    "Show Admin / Model Lab",
    value=False,
    help="Historical backtests and model-research tools. Leave this off for normal betting analysis."
)
if show_admin_lab:
    st.markdown('<div class="v14-section">🧰 Admin / historical validation</div>', unsafe_allow_html=True)
    with st.expander("Run chronological backtest",expanded=False):
        st.caption("Train on the earlier 80% of historical matches and test only on the later unseen 20%.")
        bt_league=st.selectbox("Backtest competition",list(LEAGUES),key="bt_league")
        if st.button("RUN BACKTEST",use_container_width=True,key="run_bt"):
            with st.spinner("Running chronological holdout test..."):
                try:
                    bt=chronological_backtest(LEAGUES[bt_league]["of"])
                    a,b,c=st.columns(3)
                    a.metric("Holdout accuracy",f'{bt["accuracy"]*100:.1f}%')
                    b.metric("Log loss",f'{bt["logloss"]:.3f}')
                    c.metric("Brier score",f'{bt["brier"]:.3f}')
                    st.caption(f'Trained on {bt["train_n"]:,} earlier matches • Tested on {bt["test_n"]:,} later unseen matches')
                    bands=bt["bands"].copy()
                    bands["Grade"]=bands["Calibration gap pp"].apply(calibration_grade)
                    st.dataframe(bands,hide_index=True,use_container_width=True)
                    strong=bands[(bands["Predictions"]>=25)&(bands["Calibration gap pp"].abs()<=5)]
                    if len(strong): st.success("At least one confidence band has a usable sample and calibration within ±5pp.")
                    else: st.warning("No confidence band yet combines a strong sample with tight calibration. Do not loosen BET rules from this result.")
                except Exception as e:
                    st.error(f"Backtest could not complete: {e}")



# =========================
# V14 MODEL LAB
# =========================
def _fit_candidate(train, model_name):
    X=train[FEATURES].fillna(0)
    y=train["y"]
    if model_name=="Legacy V11":
        m=HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,l2_regularization=2,random_state=42)
    elif model_name=="Regularised":
        m=HistGradientBoostingClassifier(max_iter=130,max_leaf_nodes=9,learning_rate=.055,
                                         min_samples_leaf=28,l2_regularization=5,random_state=42)
    else:  # V14 Live Conservative
        m=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,learning_rate=.045,
                                         min_samples_leaf=38,l2_regularization=8,random_state=42)
    m.fit(X,y)
    return m

def _score_block(model,test):
    proba=model.predict_proba(test[FEATURES].fillna(0))
    actual=test["y"].to_numpy()
    pred=np.argmax(proba,axis=1)
    conf=np.max(proba,axis=1)
    correct=(pred==actual)
    onehot=np.eye(3)[actual]
    brier=float(np.mean(np.sum((proba-onehot)**2,axis=1)))
    return {
        "n":len(test),
        "accuracy":float(correct.mean()),
        "logloss":float(log_loss(actual,proba,labels=[0,1,2])),
        "brier":brier,
        "confidence":float(conf.mean()),
        "cal_gap":float((conf.mean()-correct.mean())*100),
    }

@st.cache_data(ttl=3600,show_spinner=False)
def walk_forward_model_lab(code):
    """Expanding-window validation: every test block is later than its training data."""
    f,_,_,_=make_training(code)
    f=f.reset_index(drop=True)
    if len(f)<900:
        raise RuntimeError("Not enough historical matches for walk-forward testing.")

    # Keep the earliest 55% as the initial training window; evaluate the rest
    # in five chronological blocks. This prevents future matches leaking backward.
    start=int(len(f)*.55)
    remaining=len(f)-start
    block=max(80,remaining//5)
    models=["Legacy V11","Regularised","V14 Live Conservative"]
    agg={m:[] for m in models}
    fold_rows=[]

    fold=0
    test_start=start
    while test_start < len(f):
        test_end=min(len(f),test_start+block)
        if test_end-test_start < 35:
            break
        train=f.iloc[:test_start].copy()
        test=f.iloc[test_start:test_end].copy()
        fold+=1
        for name in models:
            model=_fit_candidate(train,name)
            sc=_score_block(model,test)
            agg[name].append(sc)
            fold_rows.append({
                "Fold":fold,"Model":name,"Train matches":len(train),"Test matches":sc["n"],
                "Accuracy %":round(sc["accuracy"]*100,1),
                "Log loss":round(sc["logloss"],3),
                "Brier":round(sc["brier"],3),
                "Avg confidence %":round(sc["confidence"]*100,1),
                "Overconfidence pp":round(sc["cal_gap"],1),
            })
        test_start=test_end

    summary=[]
    for name,vals in agg.items():
        if not vals: continue
        total=sum(x["n"] for x in vals)
        def wavg(k): return sum(x[k]*x["n"] for x in vals)/total
        summary.append({
            "Model":name,
            "Test matches":total,
            "Accuracy %":round(wavg("accuracy")*100,1),
            "Log loss":round(wavg("logloss"),3),
            "Brier":round(wavg("brier"),3),
            "Avg confidence %":round(wavg("confidence")*100,1),
            "Overconfidence pp":round(wavg("cal_gap"),1),
        })
    summary=pd.DataFrame(summary).sort_values(["Log loss","Brier"],ascending=True).reset_index(drop=True)
    folds=pd.DataFrame(fold_rows)
    return summary,folds


if show_admin_lab:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#063b31,#08253d);border:1px solid #00e59b;
    border-radius:20px;padding:18px;margin:12px 0 18px 0;">
    <div style="font-size:13px;color:#77f7c7;font-weight:800;letter-spacing:.08em;">LIVE ENGINE</div>
    <div style="font-size:26px;font-weight:900;color:white;margin-top:4px;">🧰 Historical engine research</div>
    <div style="color:#b9c7d5;margin-top:8px;line-height:1.5;">
    V15 promotes only the league-specific probability treatment supported by V14 unseen-data tests: calibrated Conservative for Premier League and Bundesliga; raw Conservative for La Liga, Serie A and Ligue 1; Championship remains a raw safety fallback until calibration is validated.
    </div>
    </div>
    """,unsafe_allow_html=True)

    st.markdown('<div class="v14-section">🧠 Admin Model Lab</div>',unsafe_allow_html=True)
    with st.expander("Walk-forward model comparison",expanded=False):
        st.caption("V14 repeatedly trains only on the past and predicts the next chronological block. Three model configurations compete on exactly the same unseen matches.")
        lab_league=st.selectbox("Model Lab competition",list(LEAGUES),key="v14_lab_league")
        if st.button("RUN V15 MODEL LAB",use_container_width=True,key="run_v14_lab"):
            with st.spinner("Running expanding-window model comparison..."):
                try:
                    summary,folds=walk_forward_model_lab(LEAGUES[lab_league]["of"])
                    winner=summary.iloc[0]
                    st.markdown("#### 🏆 Unseen-data leaderboard")
                    st.dataframe(summary,hide_index=True,use_container_width=True)
                    c1,c2,c3=st.columns(3)
                    c1.metric("Best model",winner["Model"])
                    c2.metric("Best log loss",f'{winner["Log loss"]:.3f}')
                    c3.metric("Best Brier",f'{winner["Brier"]:.3f}')
                    current=summary[summary["Model"]=="Legacy V11"].iloc[0]
                    if winner["Model"]!="Legacy V11" and winner["Log loss"] < current["Log loss"]:
                        improvement=(current["Log loss"]-winner["Log loss"])/current["Log loss"]*100
                        st.success(f'{winner["Model"]} beats the V11 configuration on unseen log loss by {improvement:.1f}%. This is evidence for promotion, not an automatic live-model switch.')
                    else:
                        st.warning("The legacy V11 configuration was not convincingly beaten. V14 will not promote extra complexity just because it is newer.")
                    with st.expander("See every chronological fold"):
                        st.dataframe(folds,hide_index=True,use_container_width=True)
                except Exception as e:
                    st.error(f"V14 Model Lab could not complete: {e}")



@st.cache_data(ttl=3600,show_spinner=False)
def v14_historical_strategy_test(code, min_conf_pct):
    """
    Walk-forward prediction audit. This never invents historical bookmaker prices.
    If the underlying historical frame does not contain verified H/D/A odds,
    ROI/edge are reported as unavailable rather than fabricated.
    """
    f,_,_,_=make_training(code)
    f=f.reset_index(drop=True)
    if len(f)<900:
        raise RuntimeError("Not enough history for strategy validation.")
    start=int(len(f)*.55)
    remaining=len(f)-start
    block=max(80,remaining//5)
    rows=[]
    test_start=start
    while test_start<len(f):
        test_end=min(len(f),test_start+block)
        if test_end-test_start<35: break
        train=f.iloc[:test_start]
        test=f.iloc[test_start:test_end].copy()
        model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,learning_rate=.045,
                                             min_samples_leaf=38,l2_regularization=8,random_state=42)
        model.fit(train[FEATURES].fillna(0),train["y"])
        proba=model.predict_proba(test[FEATURES].fillna(0))
        actual=test["y"].to_numpy()
        pred=np.argmax(proba,axis=1)
        conf=np.max(proba,axis=1)
        for i in range(len(test)):
            rows.append({"confidence":float(conf[i]),"correct":bool(pred[i]==actual[i])})
        test_start=test_end

    r=pd.DataFrame(rows)
    threshold=min_conf_pct/100.0
    q=r[r["confidence"]>=threshold].copy()
    if q.empty:
        return {"n":0,"win_rate":None,"avg_conf":None,"gap":None,"roi_available":False}
    return {
        "n":len(q),
        "win_rate":float(q["correct"].mean()),
        "avg_conf":float(q["confidence"].mean()),
        "gap":float((q["confidence"].mean()-q["correct"].mean())*100),
        "roi_available":False
    }


def _fit_v14_calibrated(train):
    """Fit Conservative base model, then sigmoid-calibrate on a later calibration slice."""
    if len(train)<350:
        m=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,learning_rate=.045,
                                         min_samples_leaf=38,l2_regularization=8,random_state=42)
        m.fit(train[FEATURES].fillna(0),train["y"])
        return m
    cut=max(250,int(len(train)*.82))
    proper=train.iloc[:cut]
    cal=train.iloc[cut:]
    base=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,learning_rate=.045,
                                        min_samples_leaf=38,l2_regularization=8,random_state=42)
    base.fit(proper[FEATURES].fillna(0),proper["y"])
    if len(cal)>=75 and cal["y"].nunique()==3:
        try:
            calibrated=CalibratedClassifierCV(base,method="sigmoid",cv="prefit")
            calibrated.fit(cal[FEATURES].fillna(0),cal["y"])
            return calibrated
        except Exception:
            pass
    return base

@st.cache_data(ttl=3600,show_spinner=False)
def v14_calibration_lab(code,min_conf_pct):
    f,_,_,_=make_training(code)
    f=f.reset_index(drop=True)
    if len(f)<900: raise RuntimeError("Not enough history for calibration validation.")
    start=int(len(f)*.55); block=max(80,(len(f)-start)//5)
    raw_rows=[]; cal_rows=[]; pos=start
    while pos<len(f):
        end=min(len(f),pos+block)
        if end-pos<35: break
        train=f.iloc[:pos]; test=f.iloc[pos:end]
        raw=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,learning_rate=.045,
                                           min_samples_leaf=38,l2_regularization=8,random_state=42)
        raw.fit(train[FEATURES].fillna(0),train["y"])
        calibrated=_fit_v14_calibrated(train)
        actual=test["y"].to_numpy()
        for label,model,bucket in [("Raw",raw,raw_rows),("Calibrated",calibrated,cal_rows)]:
            proba=model.predict_proba(test[FEATURES].fillna(0))
            pred=np.argmax(proba,axis=1); conf=np.max(proba,axis=1)
            for i in range(len(test)):
                bucket.append({"confidence":float(conf[i]),"correct":bool(pred[i]==actual[i])})
        pos=end

    out=[]
    threshold=min_conf_pct/100
    for name,rows in [("Raw Conservative",raw_rows),("V14 Calibrated",cal_rows)]:
        r=pd.DataFrame(rows); q=r[r["confidence"]>=threshold]
        if len(q):
            out.append({"Model":name,"Qualifying picks":len(q),
                        "Strike rate %":round(q["correct"].mean()*100,1),
                        "Avg confidence %":round(q["confidence"].mean()*100,1),
                        "Calibration gap pp":round((q["confidence"].mean()-q["correct"].mean())*100,1)})
    return pd.DataFrame(out)

if show_admin_lab:
    st.markdown("### 🎯 V15 Validation Lab — V14 Evidence")
    st.caption("V14 learns its probability correction only from earlier matches, then tests the corrected probabilities on later unseen matches.")
    with st.expander("Run calibration comparison",expanded=False):
        cal_league=st.selectbox("Calibration competition",list(LEAGUES),key="v14_cal_league")
        cal_conf=st.number_input("Audit threshold (%)",35,90,62,1,key="v14_cal_conf")
        if st.button("RUN CALIBRATION TEST",use_container_width=True,key="run_v14_cal"):
            with st.spinner("Testing raw vs calibrated probabilities..."):
                try:
                    ct=v14_calibration_lab(LEAGUES[cal_league]["of"],cal_conf)
                    st.dataframe(ct,hide_index=True,use_container_width=True)
                    if len(ct)==2:
                        raw=float(ct.iloc[0]["Calibration gap pp"]); new=float(ct.iloc[1]["Calibration gap pp"])
                        if abs(new)<abs(raw):
                            st.success(f"Calibration improved: {raw:+.1f}pp → {new:+.1f}pp on later unseen selections.")
                        else:
                            st.warning(f"Calibration did not improve: {raw:+.1f}pp → {new:+.1f}pp. V14 will not claim an improvement.")
                except Exception as e:
                    st.error(f"Calibration test could not complete: {e}")


if show_admin_lab:
    st.markdown("### 💷 V15 Strategy Audit")
    st.caption("Tests the promoted live model on later unseen matches. V14 will not fabricate historical odds: ROI stays disabled until verified historical prices are available in the dataset.")
    with st.expander("Run confidence strategy audit",expanded=False):
        audit_league=st.selectbox("Strategy competition",list(LEAGUES),key="v14_audit_league")
        audit_conf=st.number_input("Minimum model confidence (%)",min_value=35,max_value=90,value=62,step=1,key="v14_audit_conf")
        if st.button("RUN STRATEGY AUDIT",use_container_width=True,key="run_v14_audit"):
            with st.spinner("Replaying unseen predictions..."):
                try:
                    ar=v14_historical_strategy_test(LEAGUES[audit_league]["of"],audit_conf)
                    if ar["n"]==0:
                        st.warning("No unseen predictions met that confidence threshold.")
                    else:
                        a,b,c=st.columns(3)
                        a.metric("Qualifying picks",f'{ar["n"]:,}')
                        b.metric("Strike rate",f'{ar["win_rate"]*100:.1f}%')
                        c.metric("Avg confidence",f'{ar["avg_conf"]*100:.1f}%')
                        if abs(ar["gap"])<=3:
                            st.success(f'Calibration gap: {ar["gap"]:+.1f}pp — tight at this threshold.')
                        elif abs(ar["gap"])<=6:
                            st.warning(f'Calibration gap: {ar["gap"]:+.1f}pp — monitor before loosening rules.')
                        else:
                            st.error(f'Calibration gap: {ar["gap"]:+.1f}pp — probabilities remain materially miscalibrated at this threshold.')
                        st.info("Historical ROI / profit / edge: NOT YET VERIFIED. V14 refuses to calculate these without real historical bookmaker odds.")
                except Exception as e:
                    st.error(f"Strategy audit could not complete: {e}")


if show_admin_lab:
    st.markdown("### 🧭 V15.2 Market-Integrity Fix")
    st.caption("The live predictor now chooses the probability engine per competition from V14 unseen-data evidence. It never applies calibration globally.")
    with st.expander("View league engine policy",expanded=False):
        policy_rows=[]
        for _league,_p in V15_POLICY.items():
            policy_rows.append({"Competition":_league,"Live engine":"Calibrated Conservative" if _p["engine"]=="calibrated" else "Raw Conservative","Status":_p["status"],"Evidence":_p["evidence"]})
        st.dataframe(pd.DataFrame(policy_rows),hide_index=True,use_container_width=True)

if show_admin_lab:
    with st.expander("🧪 V28 frozen-scorer holdout audit",expanded=False):
        st.caption("Calibration fitted on 2023-24 + 2024-25 only; 2025-26 was kept untouched until final evaluation.")
        st.dataframe(pd.DataFrame([{"League":k,**v} for k,v in V28_FROZEN_CALIBRATION.items()]),hide_index=True,use_container_width=True)
        st.json(V28_POOLED_AUDIT)

st.subheader("🔐 Data connections")

# V22: load the Odds API key automatically from Streamlit Secrets.
# The secret stays outside GitHub/source code. A temporary session override is
# still available for diagnostics, but normal use requires no re-entry.
def _streamlit_odds_secret():
    try:
        return str(st.secrets.get("ODDS_API_KEY", "")).strip()
    except Exception:
        return ""

stored_odds_key = _streamlit_odds_secret()
session_override = st.session_state.get("odds_key_override", "").strip()

if stored_odds_key:
    st.success("Odds API connected automatically from Streamlit Secrets.")
    with st.expander("Change connection for this session", expanded=False):
        entered=st.text_input(
            "Temporary Odds API key",
            value=session_override,
            type="password",
            placeholder="Optional temporary override",
            help="This override lasts only for the current Streamlit session. Your saved Streamlit Secret is unchanged."
        )
        c1,c2=st.columns(2)
        with c1:
            if st.button("Use temporary key",use_container_width=True):
                if entered.strip():
                    st.session_state["odds_key_override"]=entered.strip()
                    st.success("Temporary key loaded for this session.")
                    st.rerun()
                else:
                    st.warning("Paste a key first.")
        with c2:
            if st.button("Use saved key",use_container_width=True):
                st.session_state.pop("odds_key_override",None)
                st.rerun()
else:
    st.warning("No saved ODDS_API_KEY was found in Streamlit Secrets. Add it once in your Streamlit app settings and the predictor will connect automatically thereafter.")
    entered=st.text_input(
        "The Odds API key",
        value=session_override,
        type="password",
        placeholder="Paste The Odds API key",
        help="For permanent automatic connection, save this as ODDS_API_KEY in Streamlit Secrets rather than GitHub."
    )
    c1,c2=st.columns(2)
    with c1:
        if st.button("Connect for this session",use_container_width=True):
            if entered.strip():
                st.session_state["odds_key_override"]=entered.strip()
                st.success("Odds key loaded for this session.")
                st.rerun()
            else:
                st.warning("Paste the key first.")
    with c2:
        if st.button("Clear session key",use_container_width=True):
            st.session_state.pop("odds_key_override",None)
            st.rerun()


with st.expander("⚡ Engine & data connections",expanded=False):
    st.caption("Normal use uses only the latest three seasons and cached live models. Full-history validation and premium xG/injury/line-up refresh are separate, so they never block the first Top 10.")
    _c1,_c2=st.columns(2)
    with _c1:
        st.markdown("**Deep engine validation**")
        st.caption("Runs the expensive TRAIN → TUNE → FINAL TEST process only when you choose. Successful results are cached for this session and then used by the live ranking.")
        if st.button("RUN / REFRESH DEEP VALIDATION",use_container_width=True,key="v24_run_deep"):
            st.session_state["v24_deep_requested"]=True
        st.caption("Deep validation is intentionally separate from normal rankings; it can take time on a cold cache.")
    with _c2:
        st.markdown("**API-Football enrichment**")
        _af_now=_streamlit_api_football_secret()
        st.caption("Used for the verified analyst pass: injuries/suspensions, player importance, confirmed line-ups, all-competition workload, provider forecast and provider xG when available. It never blocks the instant base ranking.")
        _af_entry=st.text_input(
            "API-Football key",
            value="",
            type="password",
            placeholder="Paste API-Football key",
            key="v24_api_football_entry"
        )
        b1,b2=st.columns(2)
        with b1:
            if st.button("CONNECT API-FOOTBALL",use_container_width=True,key="v24_af_connect"):
                if _af_entry.strip():
                    st.session_state["api_football_key_override"]=_af_entry.strip()
                    st.success("API-Football connected for this session.")
                    st.rerun()
                else:
                    st.warning("Paste the API-Football key first.")
        with b2:
            if st.button("CLEAR API-FOOTBALL",use_container_width=True,key="v24_af_clear"):
                st.session_state.pop("api_football_key_override",None)
                st.rerun()
        st.write("Status: "+("✅ Connected" if _af_now else "⚪ Not connected"))
        if _af_now and st.button("RUN FULL VERIFIED ANALYST PASS",use_container_width=True,key="v24_provider_refresh"):
            st.session_state["v24_provider_refresh_requested"]=True

st.markdown("### 🏁 Probability-first mode")
st.caption("Every unfinished fixture in your chosen dates enters the ranking pool. No positive-EV or +4pp edge gate. V28 uses a no-fit scorer with embedded untouched-holdout calibration policy, adaptive API quotas, strict FULL verification and provider-aware learning. Historical fitting remains outside normal startup.")
with st.expander("How V26 builds the five-team anchors",expanded=False):
    st.markdown("""
V28 keeps model probability separate from analyst confidence. Dedicated leagues use the in-house model; extra cups/leagues discovered by API-Football use a clearly labelled provider prediction or market fallback. It then tries to disprove each favourite using recent/venue form, opponent-adjusted performance, scoring profile, schedule, market disagreement and draw risk. When API-Football is connected, the verified pass adds comparative injuries/suspensions, player importance, confirmed line-ups, all-competition workload, an independent provider forecast and provider xG where supplied.

**Anchor score is not a win probability.** It is the transparent ranking score for the full analytical case. Five-Team Anchors are the five cases that survive that broader investigation best.
""")
# Compatibility values retained for older diagnostics only; they no longer gate the Top 10 or acca pool.
min_conf=0
min_edge=0
min_books=1
max_edge=100
topn=10

# V22 fixture picker: date changes stay local until GO is pressed.
# This prevents the expensive football analysis from rerunning while the user
# is still tapping around the calendar on a phone/tablet.
_today=date.today()
_days_ahead=(5-_today.weekday())%7
_this_saturday=_today+timedelta(days=_days_ahead)

if "fixture_start_day" not in st.session_state:
    st.session_state["fixture_start_day"]=_today
if "fixture_end_day" not in st.session_state:
    st.session_state["fixture_end_day"]=_today
if "fixture_range_picker" not in st.session_state:
    st.session_state["fixture_range_picker"]=(
        st.session_state["fixture_start_day"],
        st.session_state["fixture_end_day"],
    )

st.markdown("### 📅 Race window")
st.caption("Choose the dates you want analysed, then press RUN. This same window feeds both the Top 10 ranking and the personalised acca builder.")
_q1,_q2,_q3=st.columns(3)
with _q1:
    if st.button("TODAY // RUN",use_container_width=True,key="fixtures_today_go"):
        st.session_state["fixture_start_day"]=_today
        st.session_state["fixture_end_day"]=_today
        st.session_state["fixture_range_picker"]=(_today,_today)
        st.rerun()
with _q2:
    if st.button("SATURDAY // RUN",use_container_width=True,key="fixtures_saturday_go"):
        st.session_state["fixture_start_day"]=_this_saturday
        st.session_state["fixture_end_day"]=_this_saturday
        st.session_state["fixture_range_picker"]=(_this_saturday,_this_saturday)
        st.rerun()
with _q3:
    if st.button("NEXT 7 DAYS // RUN",use_container_width=True,key="fixtures_week_go"):
        _week_end=_today+timedelta(days=6)
        st.session_state["fixture_start_day"]=_today
        st.session_state["fixture_end_day"]=_week_end
        st.session_state["fixture_range_picker"]=(_today,_week_end)
        st.rerun()

with st.form("fixture_range_form",clear_on_submit=False):
    _range=st.date_input(
        "Custom date range",
        key="fixture_range_picker",
        min_value=_today,
        max_value=_today+timedelta(days=30),
        help="Choose the first and last date, then press GO. Nothing recalculates while you are choosing dates.",
    )
    _apply_dates=st.form_submit_button("RUN THIS DATE RANGE",use_container_width=True,type="primary")

if _apply_dates:
    if isinstance(_range,(tuple,list)) and len(_range)>=2:
        _new_start,_new_end=_range[0],_range[1]
    elif isinstance(_range,(tuple,list)) and len(_range)==1:
        _new_start=_new_end=_range[0]
    else:
        _new_start=_new_end=_range
    if _new_end < _new_start:
        _new_start,_new_end=_new_end,_new_start
    st.session_state["fixture_start_day"]=_new_start
    st.session_state["fixture_end_day"]=_new_end

start_day=st.session_state["fixture_start_day"]
end_day=st.session_state["fixture_end_day"]
if start_day==end_day:
    st.success(f"Active fixtures: {start_day.strftime('%A %d %B %Y')}")
else:
    st.success(f"Active fixtures: {start_day.strftime('%a %d %b')} → {end_day.strftime('%a %d %b %Y')}")

st.markdown("### 🎯 Target build bay")
st.caption("Optional — the Top 10 is always probability-ranked first. Use this only when you want the optimiser to find the highest-probability 5/6-team combination that can approach your stake/return target inside the SAME date window.")
ac1,ac2,ac3=st.columns(3)
with ac1:
    acca_stake=st.number_input("Stake (£)",min_value=1.0,max_value=10000.0,value=10.0,step=1.0,format="%.2f",key="acca_stake")
with ac2:
    acca_target=st.number_input("Target return (£)",min_value=2.0,max_value=100000.0,value=500.0,step=10.0,format="%.2f",key="acca_target")
with ac3:
    acca_legs_choice=st.selectbox("Build size",["Best of 5 or 6","5 teams","6 teams"],index=0,key="acca_legs_choice")
if st.button("LAUNCH TARGET ACCA",use_container_width=True,key="build_personal_acca"):
    st.session_state["build_personal_acca_requested"]=True

build_acca_requested=bool(st.session_state.get("build_personal_acca_requested",False))

# --- V16 live validation ledger -------------------------------------------------
LEDGER_COLUMNS=[
    "Signal ID","Fixture Key","Lifecycle","Snapshot type","Recorded UTC","League","Home team","Away team","Kickoff ISO","Kickoff UK","Pick",
    "Provider fixture ID","Model probability %","Market fair %","Consensus odds","Entry best odds","Latest best odds",
    "Bookmakers","Edge pp","Expected value %","Engine","Validation","Result","Won","Profit units",
    "Latest snapshot UTC","Minutes to kickoff","CLV %","CLV status"
]

def _empty_ledger():
    return pd.DataFrame(columns=LEDGER_COLUMNS)

def _ensure_ledger():
    if "v16_ledger" not in st.session_state or not isinstance(st.session_state.v16_ledger,pd.DataFrame):
        st.session_state.v16_ledger=_empty_ledger()
    for c in LEDGER_COLUMNS:
        if c not in st.session_state.v16_ledger.columns: st.session_state.v16_ledger[c]=np.nan
    st.session_state.v16_ledger=st.session_state.v16_ledger[LEDGER_COLUMNS]

def _fixture_key(r):
    return "|".join([str(r.get("League","")),str(r.get("Home team","")),str(r.get("Away team","")),str(r.get("Kickoff ISO",""))])

def _signal_id(r):
    return _fixture_key(r)+"|"+str(r.get("Pick",""))

def _is_ledger_trackable(r):
    decision=str(r.get("Decision","")).upper()
    try:
        price=float(r.get("Best market odds"))
    except Exception:
        price=np.nan
    state=str(r.get("Game state","")).upper()
    diag=r.get("Market diagnostic") if isinstance(r.get("Market diagnostic"),dict) else {}
    return (decision in ("RANKED","BET") and np.isfinite(price) and price>1.01
            and state=="UPCOMING" and diag.get("stage")=="accepted")

def _record_live_bets(df):
    _ensure_ledger(); led=st.session_state.v16_ledger.copy(); now=datetime.now(timezone.utc)
    for _,r in df.iterrows():
        if not _is_ledger_trackable(r):
            continue
        sid=_signal_id(r); fkey=_fixture_key(r); best=float(r["Best market odds"]); kickoff=pd.to_datetime(r.get("Kickoff ISO"),utc=True,errors="coerce")
        mins=(kickoff.to_pydatetime()-now).total_seconds()/60 if pd.notna(kickoff) else np.nan
        supersede=led.index[(led["Fixture Key"].astype(str)==str(fkey)) & (led["Signal ID"].astype(str)!=str(sid)) & (led["Lifecycle"].astype(str)!="SUPERSEDED")].tolist()
        for jx in supersede:
            led.at[jx,"Lifecycle"]="SUPERSEDED"
        existing=led.index[led["Signal ID"]==sid].tolist()
        if existing:
            j=existing[0]; led.at[j,"Latest best odds"]=best; led.at[j,"Latest snapshot UTC"]=now.isoformat(); led.at[j,"Minutes to kickoff"]=round(mins,1) if np.isfinite(mins) else np.nan
            entry=float(led.at[j,"Entry best odds"]) if pd.notna(led.at[j,"Entry best odds"]) else np.nan
            if np.isfinite(entry) and best>0:
                # Decimal-odds CLV: entry/closing - 1. Positive means the signal beat the later price.
                led.at[j,"CLV %"]=round((entry/best-1)*100,2)
                led.at[j,"CLV status"]="VERIFIED near-kickoff snapshot" if np.isfinite(mins) and 0<=mins<=60 else "PROVISIONAL — latest pre-kickoff snapshot"
        else:
            row={c:np.nan for c in LEDGER_COLUMNS}
            row.update({"Signal ID":sid,"Fixture Key":fkey,"Lifecycle":"ACTIVE","Snapshot type":"PREMATCH","Recorded UTC":now.isoformat(),"League":r["League"],"Home team":r["Home team"],"Away team":r["Away team"],
                "Kickoff ISO":r.get("Kickoff ISO"),"Kickoff UK":r.get("Kickoff UK"),"Pick":r["Pick"],"Provider fixture ID":r.get("Provider fixture ID"),"Model probability %":r["Confidence %"],
                "Market fair %":r.get("Market fair %"),"Consensus odds":r.get("Market odds"),"Entry best odds":best,"Latest best odds":best,
                "Bookmakers":r.get("Bookmakers"),"Edge pp":r.get("Edge pp"),"Expected value %":r.get("EV %"),"Engine":r.get("Model engine"),
                "Validation":r.get("Validation"),"Result":"PENDING","Won":np.nan,"Profit units":np.nan,"Latest snapshot UTC":now.isoformat(),
                "Minutes to kickoff":round(mins,1) if np.isfinite(mins) else np.nan,"CLV %":0.0,"CLV status":"ENTRY SNAPSHOT"})
            if led.empty:
                led=pd.DataFrame([row],columns=LEDGER_COLUMNS)
            else:
                led=pd.concat([led,pd.DataFrame([row])],ignore_index=True)
    st.session_state.v16_ledger=led[LEDGER_COLUMNS]


ANCHOR_HISTORY_COLUMNS=[
    "Signal ID","Fixture Key","Lifecycle","Recorded UTC","League","Home team","Away team","Kickoff ISO","Pick","Model win %",
    "Anchor score","Stability","Evidence %","Market odds","External status","Supports","Risks",
    "Result","Won","Diagnostic tag"
]
def _ensure_anchor_history():
    if "v27_anchor_history" not in st.session_state or not isinstance(st.session_state.v27_anchor_history,pd.DataFrame):
        st.session_state.v27_anchor_history=pd.DataFrame(columns=ANCHOR_HISTORY_COLUMNS)
    for c in ANCHOR_HISTORY_COLUMNS:
        if c not in st.session_state.v27_anchor_history.columns: st.session_state.v27_anchor_history[c]=np.nan
    st.session_state.v27_anchor_history=st.session_state.v27_anchor_history[ANCHOR_HISTORY_COLUMNS]

def _competition_trust_map():
    _ensure_ledger(); led=st.session_state.v16_ledger.copy()
    if "Lifecycle" in led.columns:
        led=led[led["Lifecycle"].astype(str)!="SUPERSEDED"]
    if "Snapshot type" in led.columns:
        led=led[led["Snapshot type"].astype(str)=="PREMATCH"]
    led=led[pd.to_numeric(led["Won"],errors="coerce").notna()].copy()
    out={}
    if led.empty: return out
    prior_strength=50.0
    for league,g in led.groupby("League"):
        if len(g)<20: continue
        wins=float(pd.to_numeric(g["Won"],errors="coerce").sum()); n=float(len(g))
        expected=float(pd.to_numeric(g["Model probability %"],errors="coerce").mean()/100.0)
        observed=wins/n
        posterior=(wins+prior_strength*expected)/(n+prior_strength)
        gap_pp=(posterior-expected)*100.0
        adj=float(np.clip(gap_pp*.10,-3,3))
        out[str(league)]={"n":int(n),"observed":round(observed*100,1),"expected":round(expected*100,1),
                          "posterior":round(posterior*100,1),"adjustment":round(adj,1),
                          "method":"Bayesian shrinkage (50-match prior)"}
    return out

def _apply_competition_trust(frame):
    out=frame.copy(); trust=_competition_trust_map(); adjs=[]; labels=[]
    for _,r in out.iterrows():
        t=trust.get(str(r.get("League","")))
        if t:
            adjs.append(t["adjustment"]); labels.append(f'{t["n"]} settled • actual {t["observed"]:.1f}% vs stated {t["expected"]:.1f}%')
        else:
            adjs.append(0.0); labels.append("NEUTRAL — insufficient settled history")
    out["Competition trust adjustment"]=adjs; out["Competition trust"]=labels
    out["Anchor score"]=np.clip(pd.to_numeric(out["Anchor score"],errors="coerce").fillna(0)+pd.Series(adjs,index=out.index),0,100)
    return out

def _record_anchor_snapshots(frame):
    _ensure_anchor_history(); hist=st.session_state.v27_anchor_history.copy(); existing=set(hist["Signal ID"].astype(str)); rows=[]
    now=datetime.now(timezone.utc).isoformat()
    for _,r in frame.iterrows():
        sid=_signal_id(r); fkey=_fixture_key(r)
        if sid in existing: continue
        mask=(hist["Fixture Key"].astype(str)==str(fkey)) & (hist["Signal ID"].astype(str)!=str(sid)) & (hist["Lifecycle"].astype(str)!="SUPERSEDED")
        hist.loc[mask,"Lifecycle"]="SUPERSEDED"
        case=r.get("Analyst case") if isinstance(r.get("Analyst case"),dict) else {}
        rows.append({"Signal ID":sid,"Fixture Key":fkey,"Lifecycle":"ACTIVE","Recorded UTC":now,"League":r.get("League"),"Home team":r.get("Home team"),"Away team":r.get("Away team"),
                     "Kickoff ISO":r.get("Kickoff ISO"),"Pick":r.get("Pick"),"Model win %":r.get("Ranking %"),"Anchor score":r.get("Anchor score"),
                     "Stability":r.get("Stability"),"Evidence %":r.get("Evidence completeness %"),"Market odds":r.get("Best market odds"),
                     "External status":r.get("External data status"),"Supports":json.dumps(case.get("supports",[])),"Risks":json.dumps(case.get("risks",[])),
                     "Result":"PENDING","Won":np.nan,"Diagnostic tag":""})
    if rows:
        if hist.empty:
            hist=pd.DataFrame(rows,columns=ANCHOR_HISTORY_COLUMNS)
        else:
            hist=pd.concat([hist,pd.DataFrame(rows)],ignore_index=True)
    st.session_state.v27_anchor_history=hist[ANCHOR_HISTORY_COLUMNS]

def _settle_anchor_history_from_ledger():
    _ensure_anchor_history(); _ensure_ledger(); hist=st.session_state.v27_anchor_history.copy(); led=st.session_state.v16_ledger.copy()
    if hist.empty or led.empty: return
    lookup={str(r["Signal ID"]):r for _,r in led.iterrows() if pd.notna(r.get("Won")) and str(r.get("Lifecycle","ACTIVE"))!="SUPERSEDED"}
    for i,r in hist.iterrows():
        if str(r.get("Lifecycle","ACTIVE"))=="SUPERSEDED": continue
        rr=lookup.get(str(r["Signal ID"]))
        if rr is None: continue
        won=bool(rr.get("Won")); hist.at[i,"Won"]=won; hist.at[i,"Result"]=rr.get("Result")
        if won: tag="Won"
        else:
            risks=str(r.get("Risks","")).lower()
            if "draw" in risks: tag="Pre-match draw warning present"
            elif "availability" in risks or "absence" in risks: tag="Pre-match availability warning present"
            elif "market" in risks: tag="Pre-match market disagreement present"
            elif "confirmed xi" in risks or "lineup" in risks: tag="Pre-match lineup warning present"
            else: tag="No major pre-match contradiction captured"
        hist.at[i,"Diagnostic tag"]=tag
    st.session_state.v27_anchor_history=hist[ANCHOR_HISTORY_COLUMNS]

def _logit01(p):
    p=float(np.clip(float(p),1e-5,1-1e-5))
    return math.log(p/(1-p))

def _learn_market_blender():
    """Learn model-vs-market weighting from settled signals, never by hand.

    Requires enough settled observations in the app ledger. Until then, the
    main model probability remains untouched and market probability is shown
    only as evidence.
    """
    try:
        _ensure_ledger()
        led=st.session_state.v16_ledger.copy()
    except Exception:
        return None,{"status":"unavailable","n":0}
    if led.empty:
        return None,{"status":"collecting","n":0}
    q=led.copy()
    q["Won_num"]=q["Won"].map({True:1,False:0})
    q["mp"]=pd.to_numeric(q["Model probability %"],errors="coerce")/100.0
    q["mk"]=pd.to_numeric(q["Market fair %"],errors="coerce")/100.0
    q=q.dropna(subset=["Won_num","mp","mk"])
    q=q[(q["mp"]>0)&(q["mp"]<1)&(q["mk"]>0)&(q["mk"]<1)]
    if len(q)<50 or q["Won_num"].nunique()<2:
        return None,{"status":"collecting","n":int(len(q))}
    X=np.column_stack([q["mp"].map(_logit01),q["mk"].map(_logit01)])
    y=q["Won_num"].astype(int).to_numpy()
    # Time-ordered holdout so the market blender must prove itself as well.
    cut=max(35,int(len(q)*.75))
    trX,teX=X[:cut],X[cut:]
    try_=y[:cut]; tey=y[cut:]
    if len(np.unique(try_))<2 or len(tey)<10:
        return None,{"status":"collecting","n":int(len(q))}
    blend=LogisticRegression(C=.5,max_iter=1000).fit(trX,try_)
    blend_p=blend.predict_proba(teX)[:,1]
    base_p=q["mp"].to_numpy()[cut:]
    blend_loss=float(log_loss(tey,blend_p,labels=[0,1]))
    base_loss=float(log_loss(tey,base_p,labels=[0,1]))
    if blend_loss >= base_loss:
        return None,{"status":"not_promoted","n":int(len(q)),"base_loss":base_loss,"blend_loss":blend_loss}
    final=LogisticRegression(C=.5,max_iter=1000).fit(X,y)
    return final,{
        "status":"promoted","n":int(len(q)),
        "base_loss":base_loss,"blend_loss":blend_loss,
        "model_coef":float(final.coef_[0][0]),
        "market_coef":float(final.coef_[0][1]),
    }

def _apply_market_blender(df):
    blender,meta=_learn_market_blender()
    out=df.copy()
    out["Probability source"]="V25 football model"
    if blender is None:
        return out,meta
    for i,r in out.iterrows():
        try:
            if str(r.get("Game state","")).upper()=="LIVE":
                continue
            mp=float(r.get("Confidence %"))/100.0
            mk=float(r.get("Market fair %"))/100.0
            if not (0<mp<1 and 0<mk<1): continue
            X=np.array([[_logit01(mp),_logit01(mk)]])
            p=float(blender.predict_proba(X)[0,1])*100.0
            # Blend is for the selected team win event; keep it bounded.
            out.at[i,"Ranking %"]=float(np.clip(p,1.0,99.0))
            out.at[i,"Probability source"]="Validated model + market blender"
        except Exception:
            pass
    return out,meta

def _settle_ledger():
    _ensure_ledger(); led=st.session_state.v16_ledger.copy()
    for _c in ("Result","Won","Lifecycle","Snapshot type","CLV status"):
        if _c in led.columns:
            led[_c]=led[_c].astype(object)
    api_key=_streamlit_api_football_secret(); provider_settled=0
    for j,r in led.iterrows():
        if str(r.get("Lifecycle","ACTIVE"))=="SUPERSEDED": continue
        if str(r.get("Snapshot type","PREMATCH"))!="PREMATCH": continue
        if str(r.get("Result","PENDING")) not in ("PENDING","nan",""): continue
        kickoff=pd.to_datetime(r.get("Kickoff ISO"),utc=True,errors="coerce")
        if pd.notna(kickoff) and pd.Timestamp.now(tz="UTC") < kickoff+pd.Timedelta(minutes=105):
            continue
        target=None
        pfid=r.get("Provider fixture ID")
        if api_key and pd.notna(pfid) and provider_settled<10:
            try:
                fx=_api_football_get("fixtures",{"id":int(float(pfid))},api_key); provider_settled+=1
                if fx:
                    f=fx[0]; status=str(((f.get("fixture") or {}).get("status") or {}).get("short","")).upper()
                    goals=f.get("goals") or {}; hg,ag=goals.get("home"),goals.get("away")
                    if status in ("FT","AET","PEN") and hg is not None and ag is not None:
                        target=(int(hg),int(ag))
                    else:
                        continue
            except ApiBudgetDeferred:
                break
            except Exception:
                pass
        if target is None:
            lname=r.get("League"); meta=LEAGUES.get(lname)
            if meta:
                try: matches=fixtures_for(meta["of"])
                except Exception: matches=[]
                for m in matches:
                    h=team_name(m.get("team1","")).strip(); a=team_name(m.get("team2","")).strip()
                    if team_match(r.get("Home team",""),h) and team_match(r.get("Away team",""),a):
                        sc=score_ft(m)
                        if sc is not None: target=sc; break
        if target is None: continue
        hg,ag=target; actual="HOME" if hg>ag else "AWAY" if ag>hg else "DRAW"; won=(actual==r.get("Pick"))
        price=float(r.get("Entry best odds")) if pd.notna(r.get("Entry best odds")) else np.nan
        led.at[j,"Result"]=f"{hg}-{ag} ({actual})"; led.at[j,"Won"]=bool(won)
        led.at[j,"Profit units"]=round(price-1,3) if won and np.isfinite(price) else -1.0
    st.session_state.v16_ledger=led[LEDGER_COLUMNS]

def _tracker_panel():
    _ensure_ledger(); _settle_ledger(); led=st.session_state.v16_ledger
    st.subheader("📈 Live performance")
    st.caption("Verified priced selections are frozen at first capture. Re-running before kickoff updates only the latest-price snapshot. The ledger lives in this Streamlit session, so export it to keep a durable copy.")
    uploaded=st.file_uploader("Restore validation history",type=["csv"],key="v16_ledger_upload")
    if uploaded is not None and st.button("RESTORE LEDGER",use_container_width=True,key="restore_v16"):
        try:
            imp=pd.read_csv(uploaded); missing=[c for c in LEDGER_COLUMNS if c not in imp.columns]
            if missing: st.error("That file is not a V16 ledger: missing "+", ".join(missing[:5]))
            else: st.session_state.v16_ledger=imp[LEDGER_COLUMNS]; st.success(f"Restored {len(imp)} signal(s)."); st.rerun()
        except Exception as e: st.error(f"Could not restore ledger: {e}")
    led=st.session_state.v16_ledger
    if led.empty:
        st.info("No V16 BET signals have been recorded in this session yet."); return
    settled=led[led["Profit units"].notna()]; wins=int((settled["Won"]==True).sum()) if len(settled) else 0
    pnl=float(pd.to_numeric(settled["Profit units"],errors="coerce").sum()) if len(settled) else 0.0
    roi=(pnl/len(settled)*100) if len(settled) else np.nan
    clv=pd.to_numeric(led.loc[led["CLV status"]=="VERIFIED near-kickoff snapshot","CLV %"],errors="coerce").dropna()
    a,b,c,d=st.columns(4); a.metric("Signals",len(led)); b.metric("Settled",len(settled)); c.metric("Strike rate",f"{wins/len(settled)*100:.1f}%" if len(settled) else "—"); d.metric("ROI",f"{roi:.1f}%" if np.isfinite(roi) else "—")
    # Evidence metrics are descriptive only; no threshold is auto-tuned from this live sample.
    max_dd=np.nan; cal_gap=np.nan; brier=np.nan
    if len(settled):
        ordered=settled.sort_values("Recorded UTC"); profits=pd.to_numeric(ordered["Profit units"],errors="coerce").fillna(0).to_numpy(); curve=np.cumsum(profits); peaks=np.maximum.accumulate(np.r_[0.0,curve]); draw=np.r_[0.0,curve]-peaks; max_dd=float(draw.min())
        probs=pd.to_numeric(settled["Model probability %"],errors="coerce").to_numpy()/100; yy=(settled["Won"]==True).astype(float).to_numpy(); ok=np.isfinite(probs)
        if ok.any(): brier=float(np.mean((probs[ok]-yy[ok])**2)); cal_gap=float(np.mean(probs[ok])*100-np.mean(yy[ok])*100)
    e1,e2,e3=st.columns(3); e1.metric("Max drawdown",f"{max_dd:.2f}u" if np.isfinite(max_dd) else "—"); e2.metric("Calibration gap",f"{cal_gap:+.1f}pp" if np.isfinite(cal_gap) else "—"); e3.metric("Pick Brier",f"{brier:.3f}" if np.isfinite(brier) else "—")
    st.caption(f"Profit/loss: {pnl:+.2f} units at recorded best price" + (f" • Mean verified CLV: {clv.mean():+.2f}%" if len(clv) else " • Verified CLV: not enough near-kickoff snapshots yet"))
    show=led[["Kickoff UK","League","Home team","Away team","Pick","Model probability %","Entry best odds","Latest best odds","Edge pp","Expected value %","Result","Profit units","CLV %","CLV status"]].copy()
    st.dataframe(show,use_container_width=True,hide_index=True)
    if len(settled):
        with st.expander("Performance by league"):
            perf=[]
            for lg,g in settled.groupby("League"):
                gp=pd.to_numeric(g["Profit units"],errors="coerce").sum(); n=len(g); w=int((g["Won"]==True).sum())
                perf.append({"League":lg,"Bets":n,"Wins":w,"Strike %":round(w/n*100,1),"Profit units":round(gp,2),"ROI %":round(gp/n*100,1)})
            st.dataframe(pd.DataFrame(perf).sort_values("Bets",ascending=False),use_container_width=True,hide_index=True)
    st.download_button("⬇️ EXPORT VALIDATION LEDGER",data=led.to_csv(index=False).encode("utf-8"),file_name="football_predictor_v16_live_ledger.csv",mime="text/csv",use_container_width=True)

scope=st.selectbox("Competition",["ALL ANALYSABLE FIXTURES"]+list(LEAGUES))
universe_mode=st.selectbox("Expanded fixture universe",["UK + Major Europe","UK only","Core model leagues only"],index=0,help="API-Football expands fixture discovery beyond the dedicated model leagues. Core-model fixtures still use the in-house model first.")

st.info("V28 • HARDENED PRODUCTION — dedicated models plus API-Football fixture discovery for UK cups, EFL Trophy and major European competitions.")

if True:
    selected=LEAGUES if scope=="ALL ANALYSABLE FIXTURES" else {scope:LEAGUES[scope]}
    odds_key=(st.session_state.get("odds_key_override", "").strip() or _streamlit_odds_secret())
    api_football_key=_streamlit_api_football_secret()
    _stats=_v28_api_state()
    _api_label=_api_health_label(api_football_key)
    _quota_bits=[]
    if _stats.get("daily_remaining") is not None: _quota_bits.append(f'daily {_stats.get("daily_remaining")}/{_stats.get("daily_limit") or "?"}')
    if _stats.get("minute_remaining") is not None: _quota_bits.append(f'minute {_stats.get("minute_remaining")}/{_stats.get("minute_limit") or "?"}')
    _quota_text=" · ".join(_quota_bits) if _quota_bits else "quota unknown until first live response"
    st.markdown(
        f"**System health** · Models ✅ HOLDOUT-AUDITED NO-FIT · Odds {'✅ KEY PRESENT' if odds_key else '⚠️ NOT CONNECTED'} · "
        f"API-Football {_api_label} · {_quota_text}"
    )
    quota_remaining=None
    diagnostics=[]
    out=[]; warnings=[]
    skipped_completed=0
    skipped_stale=0
    engine_diagnostics=[]
    _load_status=st.empty()
    _partial_status=st.empty()
    provider_window_fixtures=[]
    if api_football_key and scope=="ALL ANALYSABLE FIXTURES" and universe_mode!="Core model leagues only" and (end_day-start_day).days<=13:
        try:
            _load_status.info("Checking authoritative fixture statuses…")
            provider_window_fixtures=api_football_fixture_range(start_day.isoformat(),end_day.isoformat(),api_football_key)
        except ApiBudgetDeferred as e:
            warnings.append(f"API-Football discovery deferred ({e})")
        except Exception as e:
            warnings.append(f"API-Football discovery unavailable ({e})")
    with st.spinner("Loading fixtures and cached live models..."):
        for _league_no,(lname,meta) in enumerate(selected.items(),1):
            _load_status.info(f"Loading {lname} ({_league_no}/{len(selected)})…")
            code=meta["of"]
            odds_events=[]
            api_diag={}  # reset per league; never reuse diagnostics from a previous league
            try:
                matches=fixtures_for(code)
            except Exception as e:
                warnings.append(f"{lname}: fixture feed unavailable ({e})")
                continue
            games=[]
            for _m in matches:
                try:
                    _match_day=pd.to_datetime(str(_m.get("date",""))[:10]).date()
                except Exception:
                    continue
                if start_day <= _match_day <= end_day:
                    games.append((_m,_match_day))
            if not games: continue

            if odds_key:
                try:
                    odds_events,api_diag=odds_fetch(odds_key,meta["odds"])
                    quota_remaining=api_diag.get("remaining")
                    diagnostics.append({"League":lname,"Sport key":meta["odds"],
                                        "API events returned":api_diag.get("events",0),
                                        "Credits used":api_diag.get("used"),
                                        "Credits remaining":api_diag.get("remaining"),
                                        "Status":"OK"})
                    for ev in odds_events[:20]:
                        diagnostics.append({"League":lname,"Sport key":meta["odds"],
                                            "API events returned":"","Credits used":"","Credits remaining":"",
                                            "Status":f'ODDS EVENT: {ev.get("home_team","?")} v {ev.get("away_team","?")} @ {ev.get("commence_time","?")}'})
                except Exception as e:
                    warnings.append(f"{lname}: current odds unavailable ({e})")
                    diagnostics.append({"League":lname,"Sport key":meta["odds"],
                                        "API events returned":0,"Credits used":"","Credits remaining":"",
                                        "Status":f"ERROR: {e}"})

            try:
                if st.session_state.get("v24_deep_requested",False):
                    with st.spinner(f"Deep-validating {lname} once and caching the result..."):
                        _deep=deep_validate_model(lname,code)
                    _validated_session_models()[lname]=_deep
                model,hist,elo,ntrain,engine_label,validation_status,validation_evidence,engine_meta=get_live_model(lname,code)
                engine_diagnostics.append(engine_meta)
            except Exception as e:
                warnings.append(f"{lname}: model unavailable ({e})")
                continue
            for g,match_day in games:
                # Never analyse/recommend a fixture that the results feed already marks complete.
                if score_ft(g) is not None:
                    skipped_completed += 1
                    continue
                h=team_name(g.get("team1","")).strip(); a=team_name(g.get("team2","")).strip()
                if not h or not a: continue
                _pfx=provider_match_for_existing(h,a,match_day,provider_window_fixtures) if provider_window_fixtures else None
                if _pfx is not None:
                    _pstate=_provider_fixture_state(_pfx)
                    if _pstate in ("FINAL","UNAVAILABLE"):
                        if _pstate=="FINAL": skipped_completed+=1
                        else: skipped_stale+=1
                        continue
                vals=build_feature_values(hist,elo,h,a,pd.to_datetime(match_day))
                x=pd.DataFrame([[vals[k] for k in FEATURES]],columns=FEATURES)
                pr=model.predict_proba(x)[0]
                # V21 ranks TEAMS most likely to win, not the most likely 1X2
                # outcome. Every fixture therefore contributes its stronger team
                # (home or away) even when DRAW is the single highest 1X2 outcome.
                labels=["HOME","DRAW","AWAY"]
                i=0 if float(pr[0]) >= float(pr[2]) else 2
                conf=float(pr[i])
                market,matchdiag=consensus_for(odds_events,h,a,match_day) if odds_events else (None,{"stage":"no-events","reason":"Odds endpoint returned zero current/live events for this league","trace":[],"rejected_books":[],"api": api_diag})
                if isinstance(matchdiag,dict) and odds_key:
                    matchdiag.setdefault("api", api_diag)
                if odds_key:
                    diagnostics.append({"League":lname,"Sport key":meta["odds"],
                                        "API events returned":"","Credits used":"",
                                        "Credits remaining":"",
                                        "Status":f'FIXTURE MATCH {"YES" if market else "NO"}: {h} v {a} — {matchdiag.get("reason","")}'})
                    for tr in matchdiag.get("trace",[]):
                        diagnostics.append({"League":lname,"Sport key":meta["odds"],"API events returned":"",
                                            "Credits used":"","Credits remaining":"",
                                            "Status":f'MATCH TRACE: {tr.get("API event")} | home={tr.get("Home match")} away={tr.get("Away match")} date={tr.get("Date match")} | {tr.get("API time")}'})
                    for rb in matchdiag.get("rejected_books",[])[:20]:
                        diagnostics.append({"League":lname,"Sport key":meta["odds"],"API events returned":"",
                                            "Credits used":"","Credits remaining":"",
                                            "Status":f'BOOK REJECT: {rb.get("Bookmaker")} — {rb.get("Reason")} — {rb.get("Outcomes","")}'})
                # Kick-off source priority:
                # 1) uniquely matched current odds event
                # 2) uniquely matched event trace
                # 3) OpenFootball scheduled date/time
                _provider_kickoff=((_pfx.get("fixture") or {}).get("date") if _pfx is not None else None)
                kickoff_iso=_provider_kickoff or (market.get("event",{}).get("commence_time") if market else None) or matched_kickoff_from_diag(matchdiag) or fixture_kickoff_iso(g)
                kickoff_dt,kickoff_label=kickoff_uk_from_iso(kickoff_iso)
                timing=_kickoff_state(kickoff_iso)

                # If the fixture feed has no time, future calendar dates are still
                # valid ranking candidates. For today's date with no reliable time,
                # keep the fixture out because we cannot tell whether it has started.
                if timing["state"]=="UNKNOWN":
                    if match_day > date.today():
                        timing={"state":"UPCOMING","minutes":None,"label":f"{match_day.strftime('%a %d %b')} • kickoff time unverified"}
                    else:
                        continue

                if timing["state"]=="STALE":
                    skipped_stale += 1
                    continue

                market_verified=bool(market is not None and matchdiag.get("stage")=="accepted")

                # A LIVE match is useful only while a current market is actually
                # available. Pre-match/future fixtures stay in the ranking even when
                # their bookmaker market is not yet matched.
                if timing["state"]=="LIVE" and not market_verified:
                    continue

                if not kickoff_label:
                    kickoff_label=timing["label"]

                odd=np.nan; best_odd=np.nan; mprob=np.nan; edge=np.nan; ev=np.nan; books=0
                if market_verified:
                    med,mfair,books=market["median"],market["fair"],market["books"]
                    odd=float(med[i]); best_odd=float(market["best"][i]); mprob=float(mfair[i])
                    edge=conf-mprob
                    ev=conf*best_odd-1

                context = fast_fixture_context(hist, h, a, pd.to_datetime(match_day))
                secondary_checks=[]

                # V22: probability ranking and market verification are separate.
                # Missing current odds no longer removes an unfinished future fixture.
                if not market_verified:
                    decision="PREDICTION ONLY"
                    decision_reason="Ranked by model win probability. A current bookmaker market has not been verified yet, so no price/return calculation is shown."
                else:
                    decision="RANKED"
                    if market.get("integrity")!="OK":
                        decision_reason="Ranked by win probability. Current bookmaker prices are more dispersed than usual, so compare the live price before using it."
                        secondary_checks.append("Market-price dispersion warning")
                    elif books < 3:
                        decision_reason=f"Ranked by win probability. Current price is verified, but the market sample is limited ({books} bookmaker{'s' if books!=1 else ''})."
                        secondary_checks.append("Limited bookmaker sample")
                    else:
                        decision_reason=f"Ranked by win probability with a verified current market from {books} bookmaker{'s' if books!=1 else ''}."

                if timing["state"]=="LIVE":
                    decision_reason=("LIVE / TIME-SENSITIVE — "+decision_reason+" The displayed price is the current odds snapshot, "
                                     "but the model evidence is pre-match and the current score is not ingested. Check the live score and price immediately before placing anything.")
                    secondary_checks.append(timing["label"])
                out.append({"League":lname,"Match":f"{h} v {a}","Match date":match_day.isoformat(),"Home team":h,"Away team":a,"Pick":labels[i],
                            "Home %":round(pr[0]*100,1),"Draw %":round(pr[1]*100,1),
                            "Away %":round(pr[2]*100,1),"Confidence %":round(conf*100,1),
                            "Fair odds":round(1/conf,2),
                            "Market odds":round(odd,2) if np.isfinite(odd) else None,
                            "Market fair %":round(mprob*100,1) if np.isfinite(mprob) else None,
                            "Edge pp":round(edge*100,1) if np.isfinite(edge) else None,
                            "EV %":round(ev*100,1) if np.isfinite(ev) else None,
                            "Bookmakers":books if books else None,
                            "Best market odds":round(best_odd,2) if market and np.isfinite(best_odd) else None,
                            "Market integrity":market.get("integrity") if market else None,
                            "Bookmaker detail":market.get("detail",[]) if market else [],
                            "Market diagnostic":matchdiag,
                            "Kickoff ISO":kickoff_iso,"Kickoff UK":kickoff_label,
                            "Game state":timing["state"],"Timing label":timing["label"],"Minutes since kickoff":timing["minutes"] if timing["state"]=="LIVE" else None,
                            "Decision":decision,"Decision reason":decision_reason,"Training matches":ntrain,
                            "Model engine":engine_label,"Validation":validation_status,
                            "Validation evidence":validation_evidence,"Secondary checks":secondary_checks,"Context":context,
                            "Provider fixture ID":((_pfx.get("fixture") or {}).get("id") if _pfx is not None else None),
                            "Provider league ID":((_pfx.get("league") or {}).get("id") if _pfx is not None else API_FOOTBALL_LEAGUES.get(lname)),
                            "Provider season":((_pfx.get("league") or {}).get("season") if _pfx is not None else football_season_year_for_date(match_day)),
                            "Provider home team ID":(((_pfx.get("teams") or {}).get("home") or {}).get("id") if _pfx is not None else None),
                            "Provider away team ID":(((_pfx.get("teams") or {}).get("away") or {}).get("id") if _pfx is not None else None),
                            "Fixture source":"OpenFootball + API-Football status cross-check" if _pfx is not None else "OpenFootball"})
            _partial_status.caption(f"Found {len(out)} ranked fixture candidate(s) so far.")

    # V26: API-Football becomes the discovery layer for competitions outside the
    # dedicated OpenFootball model set. Dedicated rows always win duplicate checks.
    provider_discovered=0; provider_ranked=0; provider_unranked=0; provider_markets=0
    if scope=="ALL ANALYSABLE FIXTURES" and universe_mode!="Core model leagues only" and api_football_key:
        try:
            if (end_day-start_day).days>13:
                raise RuntimeError("Expanded fixture discovery is limited to a 14-day window to protect API quota and response time.")
            _load_status.info("Discovering expanded UK / European fixtures…")
            pfixtures=provider_window_fixtures or api_football_fixture_range(start_day.isoformat(),end_day.isoformat(),api_football_key)
            candidates=[]
            for fx in pfixtures:
                if not provider_competition_allowed(fx,universe_mode): continue
                if _provider_fixture_state(fx)!="UPCOMING": continue
                if provider_fixture_duplicates_existing(fx,out): continue
                provider_discovered+=1
                candidates.append((provider_competition_priority(fx),fx))
            candidates.sort(key=lambda z:z[0],reverse=True)

            # Prediction and odds calls are quota-bounded. Coverage is checked
            # before either endpoint is called, as recommended by API-Football.
            provider_rows=[]; market_calls=0
            for _,fx in candidates[:PROVIDER_PREDICTION_BUDGET]:
                fid=(fx.get("fixture") or {}).get("id"); lg=fx.get("league") or {}
                lid=lg.get("id"); season_year=lg.get("season")
                cov=api_football_coverage(str(lg.get("name","")),api_football_key,lid,season_year)
                pred={"available":False}
                if cov.get("predictions"):
                    pred=api_football_prediction(fid,api_football_key)
                row=_provider_prediction_to_row(fx,pred,None)
                if row is None and cov.get("odds") and market_calls<PROVIDER_MARKET_BUDGET:
                    teams=fx.get("teams") or {}
                    h=(teams.get("home") or {}).get("name",""); a=(teams.get("away") or {}).get("name","")
                    market=api_football_fixture_odds(fid,h,a,api_football_key); market_calls+=1
                    row=_provider_prediction_to_row(fx,{"available":False},market)
                if row is not None:
                    provider_rows.append(row); provider_ranked+=1
                else:
                    provider_unranked+=1

            # Spend only the remaining market-call budget on pricing the strongest
            # expanded selections. Total market calls never exceed the advertised cap.
            provider_rows.sort(key=lambda r:float(r.get("Confidence %") or 0),reverse=True)
            for row in provider_rows:
                if market_calls>=PROVIDER_MARKET_BUDGET: break
                if row.get("Best market odds"): continue
                market=api_football_fixture_odds(row.get("Provider fixture ID"),row.get("Home team"),row.get("Away team"),api_football_key)
                market_calls+=1
                if not market: continue
                labels=["HOME","DRAW","AWAY"]; i=labels.index(row["Pick"])
                conf=float(row["Confidence %"])/100.0; odd=float(market["median"][i]); best=float(market["best"][i]); mf=float(market["fair"][i])
                row["Market odds"]=round(odd,2); row["Best market odds"]=round(best,2); row["Market fair %"]=round(mf*100,1)
                row["Edge pp"]=round((conf-mf)*100,1); row["EV %"]=round((conf*best-1)*100,1)
                row["Bookmakers"]=int(market["books"]); row["Market integrity"]=market.get("integrity"); row["Bookmaker detail"]=market.get("detail",[])
                row["Market diagnostic"]={"stage":"accepted","reason":"Verified API-Football 1X2 market","trace":[],"provider":"API-Football"}
                row["Decision"]="RANKED"; row["Decision reason"]="Expanded-universe selection with verified API-Football 1X2 pricing."
                provider_markets+=1

            out.extend(provider_rows)
            _partial_status.caption(f"Expanded discovery: {provider_discovered} extra fixture(s) found • {provider_ranked} rankable • {provider_markets} with verified provider prices.")
        except Exception as e:
            warnings.append(f"Expanded API-Football discovery unavailable ({e})")

    _load_status.empty()
    _partial_status.empty()

    if st.session_state.get("v24_deep_requested",False):
        st.session_state["v24_deep_requested"]=False

    if api_football_key and scope=="ALL ANALYSABLE FIXTURES" and universe_mode!="Core model leagues only":
        st.caption(f"Expanded universe: {provider_discovered} additional fixture(s) discovered • {provider_ranked} rankable • {provider_markets} with provider prices. Prediction budget {PROVIDER_PREDICTION_BUDGET}; price budget {PROVIDER_MARKET_BUDGET}.")
    elif scope=="ALL ANALYSABLE FIXTURES" and universe_mode!="Core model leagues only":
        st.info("Expanded fixture discovery is ready, but API_FOOTBALL_KEY is not connected. The app is currently showing the dedicated model leagues only.")

    if warnings:
        with st.expander("Data/model warnings"):
            for w in warnings: st.warning(w)
    # Completed and stale fixtures are removed silently. Future unfinished fixtures remain
    # rankable even if their current bookmaker market is not yet verified.
    if not out:
        st.info("No unfinished supported fixtures are available in the selected window.")
        st.stop()
    d=pd.DataFrame(out).sort_values("Confidence %",ascending=False)
    _extreg=_external_analysis_registry()
    for _i,_r in d.iterrows():
        _deep=_extreg.get(_signal_id(_r))
        if isinstance(_deep,dict) and _deep.get("available"):
            _ctx=d.at[_i,"Context"] if isinstance(d.at[_i,"Context"],dict) else {}
            for _k in ("availability","confirmed_lineup","confirmed_bench","lineups_confirmed","coverage","provider_prediction","schedule","true_xg_home","true_xg_away"):
                _ctx[_k]=_deep.get(_k)
            _ctx["injuries"]="VERIFIED PLAYER-IMPACT ANALYSIS"
            d.at[_i,"Context"]=_ctx
            d.at[_i,"External data status"]="Verified analyst pass (cached)"
    d["V18 audit"]=[score_selection(r) for r in d.to_dict("records")]
    d["Selection score"]=[audit["score"] for audit in d["V18 audit"]]
    d["Tier"]=[audit["tier"] for audit in d["V18 audit"]]
    d["Ranking %"]=pd.to_numeric(d["Confidence %"],errors="coerce")

    if engine_diagnostics:
        with st.expander("🧪 V26 engine stress test — chronological unseen matches",expanded=False):
            st.caption("Fast mode returns predictions immediately. When you run Deep Validation, the selected leagues use TRAIN → TUNE/CALIBRATE → untouched FINAL TEST and cache the approved result for the rest of the session.")
            _eng=pd.DataFrame(engine_diagnostics)
            _cols=["League","Engine","Training matches","Train matches","Tune matches","Final test matches","Legacy log loss","V25 log loss","Accuracy %","Avg confidence %","Calibration gap pp","Recency half-life days","Time-decay half-life days","Temperature","Promoted"]
            st.dataframe(_eng[[c for c in _cols if c in _eng.columns]],hide_index=True,use_container_width=True)
            for meta in engine_diagnostics:
                with st.expander(f'{meta["League"]} engine details',expanded=False):
                    if meta.get("Ensemble weights"):
                        st.json(meta.get("Ensemble weights",{}))
                    else:
                        st.caption("Fast live model active. Run Deep Validation to generate ensemble weights and untouched final-test product audits.")
                    _rr=meta.get("Rank audit") or []
                    if _rr:
                        st.caption("Top-10 rank strike rates on the untouched final test")
                        st.dataframe(pd.DataFrame(_rr),hide_index=True,use_container_width=True)
                    _sf=meta.get("Safest-five audit") or {}
                    if _sf:
                        st.caption("Historical Safest-Five product audit on final-test matchdays with at least five fixtures")
                        st.json(_sf)
            st.caption("Market-consensus blending is learned separately from settled live-ledger results and is promoted only after beating model-only probabilities on a chronological holdout.")
            st.json(market_blend_meta)
    _live_mask=d["Game state"].eq("LIVE") & pd.to_numeric(d["Market fair %"],errors="coerce").notna()
    d.loc[_live_mask,"Ranking %"]=pd.to_numeric(d.loc[_live_mask,"Market fair %"],errors="coerce")
    # If enough settled history exists, learn the model/market relationship from
    # our own results and use it only after it beats model-only probabilities on
    # a chronological holdout. No arbitrary 70/30 market weighting.
    d,market_blend_meta=_apply_market_blender(d)
    _settle_ledger()
    d=_apply_competition_trust(_apply_analyst_engine(d))

    if "External data status" not in d.columns:
        d["External data status"]="Not connected" if not api_football_key else "Connected — full analyst pass available"
    def _deep_cache_is_fresh(row,deep):
        if not isinstance(deep,dict) or not deep.get("available"): return False
        ts=pd.to_datetime(deep.get("verified_at_utc"),utc=True,errors="coerce")
        if pd.isna(ts): return False
        age=(pd.Timestamp.now(tz="UTC")-ts).total_seconds()/60
        ko=pd.to_datetime(row.get("Kickoff ISO"),utc=True,errors="coerce")
        mins=(ko-pd.Timestamp.now(tz="UTC")).total_seconds()/60 if pd.notna(ko) else 9999
        return age <= (15 if mins<=180 else 360)

    _provider_refresh=bool(api_football_key)
    if _provider_refresh:
        candidate_idx=d[d["Game state"].eq("UPCOMING")].sort_values(["Anchor score","Ranking %"],ascending=[False,False]).head(8).index.tolist()
        xg_idx=set()
        _analyst_budget_start=_v28_api_state().get("network_calls",0)
        _progress=st.progress(0,text="Running verified analyst pass…")
        for _n,ridx in enumerate(candidate_idx,1):
            r=d.loc[ridx]
            try:
                if _v28_api_state().get("network_calls",0)-_analyst_budget_start >= V28_ANALYST_NETWORK_BUDGET:
                    d.at[ridx,"External data status"]="DEFERRED — analyst API budget reached"
                    continue
                _cached=_external_analysis_registry().get(_signal_id(r))
                deep=_cached if _deep_cache_is_fresh(r,_cached) else _deep_fixture_analysis(
                    r,api_football_key,include_xg=False,include_lineup_core=False,full_context=False
                )
                _external_analysis_registry()[_signal_id(r)]=deep
                if deep.get("available"):
                    ctx=d.at[ridx,"Context"] if isinstance(d.at[ridx,"Context"],dict) else {}
                    for k in ("availability","confirmed_lineup","confirmed_bench","lineups_confirmed","coverage","provider_prediction","schedule","true_xg_home","true_xg_away"):
                        ctx[k]=deep.get(k)
                    ctx["injuries"]="VERIFIED PLAYER-IMPACT ANALYSIS"
                    d.at[ridx,"Context"]=ctx
                    d.at[ridx,"External data status"]="SCREEN VERIFIED" if deep.get("verification_stage")=="SCREEN" else "FULL VERIFIED"
                else:
                    d.at[ridx,"External data status"]=deep.get("reason","Provider unavailable")
            except ApiBudgetDeferred as e:
                d.at[ridx,"External data status"]=f"DEFERRED — {e}"
            except Exception as e:
                d.at[ridx,"External data status"]=f"SCREEN VERIFY ERROR — {e}"
            _progress.progress(_n/max(len(candidate_idx),1),text=f"Verified analyst pass {_n}/{len(candidate_idx)}")
        _progress.empty()
        d["V18 audit"]=[score_selection(r) for r in d.to_dict("records")]
        d=_apply_competition_trust(_apply_analyst_engine(d))

        # A verified injury/schedule pass can reorder the five. Make one small
        # second pass so every FINAL anchor attempts provider xG too, not only
        # the preliminary five chosen before availability was known.
        _final_five_idx=_anchor_five(d).index.tolist()
        for ridx in _final_five_idx:
            try:
                r=d.loc[ridx]
                _ctx0=d.at[ridx,"Context"] if isinstance(d.at[ridx,"Context"],dict) else {}
                _need_xg=not (
                    isinstance(_ctx0.get("true_xg_home"),dict)
                    and isinstance(_ctx0.get("true_xg_away"),dict)
                )
                # Always perform the final confirmed-XI/core-player check for the
                # final five. xG is only re-requested when it was not already cached.
                _cached=_external_analysis_registry().get(_signal_id(r))
                _ctx_cached=(r.get("Context") if isinstance(r.get("Context"),dict) else {})
                if _deep_cache_is_fresh(r,_cached) and _ctx_cached.get("lineups_confirmed") and not _need_xg:
                    deep=_cached
                elif _v28_api_state().get("network_calls",0)-_analyst_budget_start >= V28_ANALYST_NETWORK_BUDGET:
                    d.at[ridx,"External data status"]="DEFERRED — analyst API budget reached"
                    continue
                else:
                    deep=_deep_fixture_analysis(
                        r,api_football_key,include_xg=_need_xg,include_lineup_core=True,full_context=True
                    )
                _external_analysis_registry()[_signal_id(r)]=deep
                if deep.get("available"):
                    ctx=d.at[ridx,"Context"] if isinstance(d.at[ridx,"Context"],dict) else {}
                    for k in ("availability","confirmed_lineup","confirmed_bench","lineups_confirmed","coverage","provider_prediction","schedule"):
                        ctx[k]=deep.get(k)
                    if deep.get("true_xg_home") is not None: ctx["true_xg_home"]=deep.get("true_xg_home")
                    if deep.get("true_xg_away") is not None: ctx["true_xg_away"]=deep.get("true_xg_away")
                    ctx["injuries"]="VERIFIED PLAYER-IMPACT ANALYSIS"
                    d.at[ridx,"Context"]=ctx
                    if deep.get("verification_stage")=="FULL":
                        d.at[ridx,"External data status"]="FULL VERIFIED"
                    else:
                        d.at[ridx,"External data status"]="PROVISIONAL — full verification incomplete"
            except ApiBudgetDeferred as e:
                d.at[ridx,"External data status"]=f"DEFERRED — {e}"
            except Exception as e:
                d.at[ridx,"External data status"]=f"FULL VERIFY ERROR — {e}"
        if _final_five_idx:
            d["V18 audit"]=[score_selection(r) for r in d.to_dict("records")]
            d=_apply_competition_trust(_apply_analyst_engine(d))
        st.session_state["v24_provider_refresh_requested"]=False

    def _anchor_readiness(r):
        ext=str(r.get("External data status",""))
        late=str(r.get("Late info",""))
        stability=str(r.get("Stability",""))
        if not api_football_key:
            return "PROVISIONAL — API-Football not connected"
        if ext.startswith("DEFERRED"):
            return "PROVISIONAL — API budget/rate limit"
        if ext.startswith("FULL VERIFY ERROR"):
            return "PROVISIONAL — full verification error"
        if late=="REVALIDATE NOW" and not ((r.get("Context") or {}).get("lineups_confirmed") if isinstance(r.get("Context"),dict) else False):
            return "PROVISIONAL — lineup recheck due"
        if stability=="LOW":
            return "PROVISIONAL — low stability"
        if ext=="FULL VERIFIED":
            return "FINAL"
        return "PROVISIONAL — full verification incomplete"

    d["Anchor readiness"]=[_anchor_readiness(r) for r in d.to_dict("records")]

    if d["Game state"].eq("LIVE").any():
        live_names=d.loc[d["Game state"].eq("LIVE"),["Match","Timing label"]].head(5)
        live_text=" • ".join(f'{r["Match"]}: {r["Timing label"]}' for _,r in live_names.iterrows())
        st.warning("🔴 LIVE MATCHES INCLUDED — "+live_text+". Live odds can move or suspend immediately. For live rows, ranking uses current market fair probability when available; the deeper model evidence remains pre-match and does not know the current score.")
    _record_live_bets(d)

    # V17 CLEAN PICKS DASHBOARD — answer first, detail on demand.
    bet_count=int(d.apply(_is_ledger_trackable,axis=1).sum())
    strong_mask=(d["Confidence %"] >= 68.0) & (~d.apply(_is_ledger_trackable,axis=1))
    strong_count=int(strong_mask.sum())

    def decimal_to_fractional(v, max_denominator=100):
        try: dec=float(v)
        except (TypeError, ValueError): return "—"
        if not np.isfinite(dec) or dec <= 1.0: return "—"
        frac=Fraction(dec-1.0).limit_denominator(max_denominator)
        if frac.numerator == frac.denominator: return "Evens"
        return f"{frac.numerator}/{frac.denominator}"

    def _num(row,key,default=np.nan):
        try:
            v=float(row.get(key,default)); return v if np.isfinite(v) else default
        except Exception: return default

    def _short_team(name):
        x=str(name or "")
        for suffix in [" Football Club"," County FC"," City FC"," United FC"," FC"]:
            if x.endswith(suffix): x=x[:-len(suffix)]
        return x.strip()

    def _context_signal(r):
        ctx=r.get("Context")
        if not isinstance(ctx,dict) or not ctx.get("available"): return "➖"
        hh,aa=ctx.get("home",{}),ctx.get("away",{})
        hp,ap=hh.get("ppg"),aa.get("ppg"); hv,av=hh.get("venue_ppg"),aa.get("venue_ppg")
        if hp is None or ap is None: return "➖"
        pred=str(r.get("Pick","")).upper()
        if pred=="HOME": support=(hp-ap)+.5*((hv or hp)-(av or ap))
        elif pred=="AWAY": support=(ap-hp)+.5*((av or ap)-(hv or hp))
        else: return "➖"
        return "✅" if support>=.5 else "⚠️" if support<=-.35 else "➖"

    def _market_signal(r):
        if pd.isna(r.get("Market fair %")): return "➖"
        if str(r.get("Market integrity",""))=="OK": return "✅"
        return "⚠️"

    def _pick_reason(r):
        conf=_num(r,"Confidence %",0)
        ctx=_context_signal(r); market=_market_signal(r)
        if conf>=68 and market=="➖": return "Strong model prediction; current bookmaker odds are not verified."
        if conf>=68 and ctx=="⚠️": return "Strong model prediction, but recent context adds caution."
        if conf>=68: return "One of the strongest model win predictions in this fixture window."
        return "Model selection — open details for the full evidence."

    def _detail_panel(r):
        st.markdown(f'**{r["Match"]}**  ·  {r.get("Kickoff UK","time unavailable")}')
        c1,c2,c3=st.columns(3)
        c1.metric("Home",f'{r["Home %"]:.0f}%'); c2.metric("Draw",f'{r["Draw %"]:.0f}%'); c3.metric("Away",f'{r["Away %"]:.0f}%')
        st.markdown(f'**Model:** {r.get("Model engine","—")}  ·  **Validation:** {r.get("Validation","—")}')
        if pd.notna(r.get("Market fair %")):
            st.write(f'Bookmaker market probability: {r.get("Market fair %"):.0f}% • {int(r.get("Bookmakers",0))} bookmaker(s) in the current sample')
            st.write(f'Consensus odds: **{decimal_to_fractional(r.get("Market odds"))}** • Best verified: **{decimal_to_fractional(r.get("Best market odds"))}**')
        else:
            st.write("Current bookmaker odds were not securely matched, so value/EV is not claimed.")
        ctx=r.get("Context")
        if isinstance(ctx,dict) and ctx.get("available"):
            hh,aa=ctx.get("home",{}),ctx.get("away",{})
            hn,an=_short_team(r.get("Home team")),_short_team(r.get("Away team"))
            st.write(f'Form PPG: {hn} {hh.get("ppg","—")} • {an} {aa.get("ppg","—")}')
            st.write(f'Goals for/against: {hh.get("gf","—")}/{hh.get("ga","—")} • {aa.get("gf","—")}/{aa.get("ga","—")}')
            sc=ctx.get("scorelines",[])
            if sc: st.write("Likely scores: "+" • ".join(x["score"] for x in sc[:3]))
        audit=r.get("V18 audit") if isinstance(r.get("V18 audit"),dict) else score_selection(r)
        st.markdown("**V18 evidence audit**")
        a1,a2,a3=st.columns(3)
        a1.metric("Evidence score",f'{audit["score"]:.0f}/100')
        a2.metric("Evidence tier*",audit["tier"])
        a3.metric("Risk penalties",f'{audit["penalty_total"]:.0f}')
        component_labels={"model_probability":"Model probability","goals_profile":"Scoring profile*","recent_venue_form":"Recent + venue form","market_value":"Verified market value","availability_evidence":"Availability evidence","opponent_strength":"Opponent strength","supporting_indicators":"Supporting indicators"}
        component_rows=[{"Evidence component":component_labels[k],"Weight":f"{w}%","Component score":audit["components"][k]} for k,w in WEIGHTS.items()]
        st.dataframe(pd.DataFrame(component_rows),hide_index=True,use_container_width=True)
        st.caption("*Evidence tier is advisory and never blocks a team from the probability ranking. Scoring profile is a goals-rate proxy, not provider-supplied xG.")
        if audit.get("positives"): st.success("Supports pick: "+" • ".join(audit["positives"]))
        if audit.get("concerns"): st.warning("Concerns: "+" • ".join(audit["concerns"]))
        if audit.get("penalties"): st.caption("Risk penalties: "+" • ".join(f'-{p["points"]} {p["reason"]}' for p in audit["penalties"]))
        st.caption(r.get("Decision reason", ""))
        if r.get("Fixture source"):
            st.caption(f'Data source: {r.get("Fixture source")} • Probability engine: {r.get("Model engine")}')
        case=r.get("Analyst case") if isinstance(r.get("Analyst case"),dict) else _analyst_case(r)
        st.markdown("**V26 analyst case**")
        x1,x2,x3=st.columns(3)
        x1.metric("Anchor score",f'{r.get("Anchor score",case["score"]):.0f}/100'); x2.metric("Stability",str(r.get("Stability","—"))); x3.metric("Evidence coverage",f'{case["evidence_completeness"]:.0f}%')
        st.caption(f'Late-info status: {r.get("Late info","—")} • Competition trust: {r.get("Competition trust","NEUTRAL")}')
        if case.get("supports"): st.success("Supports: "+" • ".join(case["supports"][:5]))
        if case.get("risks"): st.warning("Adversarial check: "+" • ".join(case["risks"][:5]))
        av=case.get("availability") or {}
        if av:
            _hav=av.get("home") or {}; _aav=av.get("away") or {}
            st.write(f'Availability burden: {_short_team(r.get("Home team"))} **{_hav.get("burden","—")}** • {_short_team(r.get("Away team"))} **{_aav.get("burden","—")}**')
            _miss=[]
            for _side,_lab in [(_hav,"Home"),(_aav,"Away")]:
                for _p in (_side.get("missing") or [])[:3]:
                    _miss.append(f'{_lab}: {_p.get("player","?")} ({_p.get("importance_label","?")} {_p.get("importance","?")}/100)')
            if _miss: st.caption("Important absences: "+" • ".join(_miss))
            _rot=[]
            for _side,_lab in [(_hav,"Home"),(_aav,"Away")]:
                for _p in ((_side.get("rotation") or {}).get("missing_core") or [])[:3]:
                    _rot.append(f'{_lab}: {_p.get("player","?")} {_p.get("lineup_status","")}')
            if _rot: st.caption("Confirmed-XI changes: "+" • ".join(_rot))

    # V28 HARDENED DASHBOARD — one clear question: who is most likely to win?
    def _context_signal_v171(r):
        ctx=r.get("Context")
        if not isinstance(ctx,dict) or not ctx.get("available"): return "➖"
        hh,aa=ctx.get("home",{}),ctx.get("away",{})
        hp,ap=hh.get("ppg"),aa.get("ppg"); hv,av=hh.get("venue_ppg"),aa.get("venue_ppg")
        if hp is None or ap is None: return "➖"
        pred=str(r.get("Pick","")).upper()
        if pred=="HOME": support=(hp-ap)+.5*((hv if hv is not None else hp)-(av if av is not None else ap))
        elif pred=="AWAY": support=(ap-hp)+.5*((av if av is not None else ap)-(hv if hv is not None else hp))
        else: return "➖"
        return "✅" if support>=.5 else "⚠️" if support<=-.35 else "➖"

    def _validation_word(r):
        v=str(r.get("Validation",""))
        if "FALLBACK" in v: return "RAW"
        if "APPROVED" in v: return "VALIDATED"
        return "CHECK"

    def _team_for_pick(r):
        if r["Pick"]=="HOME": return _short_team(r.get("Home team"))
        if r["Pick"]=="AWAY": return _short_team(r.get("Away team"))
        return "Draw"

    def _price_for(r):
        v=r.get("Best market odds") if pd.notna(r.get("Best market odds")) else r.get("Market odds")
        return decimal_to_fractional(v)

    def _clean_reason(r,lens):
        ctx=_context_signal_v171(r); market=_market_signal(r)
        if _num(r,"Draw %",0) > _num(r,"Confidence %",0):
            return "Highest team win probability in this fixture, although the draw is modelled as the single most likely 1X2 outcome."
        if market=="➖": return "One of the model's strongest win probabilities; bookmaker odds are not verified."
        if ctx=="⚠️": return "Strong win probability, although recent match context adds caution."
        if ctx=="✅": return "Strong win probability and recent match context supports the selection."
        return "One of the model's strongest win probabilities in this fixture window."

    def _goal_price(row):
        for key in ("Best market odds","Market odds"):
            try:
                v=float(row.get(key))
                if np.isfinite(v) and v>1.0:
                    return v
            except Exception:
                pass
        return np.nan

    def _build_goal_acca(frame,legs_choice,stake,target):
        target_odds=float(target)/float(stake) if float(stake)>0 else np.inf
        leg_counts=(5,6) if legs_choice=="Best of 5 or 6" else ((5,) if legs_choice=="5 teams" else (6,))
        result=build_analyst_target_acca(frame.to_dict("records"),target_odds=target_odds,leg_counts=leg_counts)
        if result["status"]=="INSUFFICIENT":
            return None,{"reason":result["reason"]}
        if result["status"]=="READY":
            part=pd.DataFrame(result["legs"]).copy(); part["_goal_price"]=pd.to_numeric(part["Best market odds"],errors="coerce")
            return {"legs":part,"total_odds":float(result["combined_odds"]),"joint_prob":float(result["joint_probability"])/100.0,
                    "mean_anchor":float(result.get("mean_anchor_score",0)),"return":float(stake)*float(result["combined_odds"]),
                    "target_odds":target_odds,"status":"TARGET REACHED","alternatives":None},None
        bundles={}
        for key in ("safest","closest"):
            b=result[key]; part=pd.DataFrame(b["legs"]).copy(); part["_goal_price"]=pd.to_numeric(part["Best market odds"],errors="coerce")
            bundles[key]={"legs":part,"total_odds":float(b["combined_odds"]),"joint_prob":float(b["joint_probability"])/100.0,
                          "mean_anchor":float(b.get("mean_anchor_score",0)),"return":float(stake)*float(b["combined_odds"])}
        return {"status":"TARGET NOT REACHABLE","target_odds":target_odds,"alternatives":bundles},None

    # Ranking pool: all unfinished upcoming fixtures plus LIVE fixtures that still
    # have a current market. The acca optimiser itself will require verified prices.
    _bettable_now=d[d["Game state"].isin(["UPCOMING","LIVE"])].copy()

    if build_acca_requested:
        st.markdown("### 🧠 Most-probable personalised acca")
        st.caption("The optimiser starts from the V26 analyst-ranked pool. A verified current price is required only to calculate the return; positive EV and +4pp edge are not selection gates.")
        _acca,_acca_err=_build_goal_acca(_bettable_now,acca_legs_choice,float(acca_stake),float(acca_target))
        if _acca_err:
            st.warning(_acca_err["reason"])
        else:
            if _acca["status"]=="TARGET REACHED":
                actual_legs=len(_acca["legs"])
                m1,m2,m3,m4=st.columns(4)
                m1.metric("Stake",f"£{acca_stake:,.2f}"); m2.metric("Target",f"£{acca_target:,.2f}")
                m3.metric("Combined odds",decimal_to_fractional(_acca["total_odds"])); m4.metric("Mean anchor score",f'{_acca.get("mean_anchor",0):.0f}/100')
                st.caption(f'Model joint chance (independence approximation): {_acca["joint_prob"]*100:.1f}%')
                st.success(f'Highest-probability {actual_legs}-team FINAL combination found that reaches the target: estimated return £{_acca["return"]:,.2f}.')
                _show=_acca["legs"].copy(); _show["Selection"]=_show.apply(_team_for_pick,axis=1); _show["Odds"]=_show["_goal_price"].apply(decimal_to_fractional)
                st.dataframe(_show[["Match date","League","Selection","Ranking %","Anchor score","Stability","Odds"]],hide_index=True,use_container_width=True)
            else:
                st.warning(f'The requested £{acca_target:,.2f} return cannot be reached with the requested FINAL 5/6-team setting.')
                for _label,_key in [("🛡️ Safest achievable","safest"),("🎯 Closest to target","closest")]:
                    _b=_acca["alternatives"][_key]
                    st.markdown(f"**{_label}** — est. return £{_b['return']:,.2f} · joint chance {_b['joint_prob']*100:.1f}% · mean anchor {_b['mean_anchor']:.0f}/100")
                    _show=_b["legs"].copy(); _show["Selection"]=_show.apply(_team_for_pick,axis=1); _show["Odds"]=_show["_goal_price"].apply(decimal_to_fractional)
                    st.dataframe(_show[["Match date","League","Selection","Ranking %","Anchor score","Stability","Odds"]],hide_index=True,use_container_width=True)
            st.caption("Joint chance is an independence approximation and is used only as a comparison aid.")

    st.markdown("""
    <style>
    .v171-head{margin:.25rem 0 1rem}.v171-head h2{margin:0!important;font-size:1.65rem!important}.v171-head p{margin:.25rem 0 0;color:#8fa8bd;font-size:.92rem}
    .v171-lens{margin:1.25rem 0 .6rem;font-size:1.15rem;font-weight:900}.v171-card{border:1px solid #1b3c53;border-radius:18px;background:linear-gradient(145deg,#091a29,#06121e);padding:15px 16px;margin:0 0 8px}
    .v171-top{display:flex;justify-content:space-between;gap:12px;align-items:flex-start}.v171-team{font-size:1.2rem;font-weight:950;line-height:1.15}.v171-pick{color:#9db5c9;font-size:.83rem;font-weight:800;margin-top:3px}.v171-prob{font-size:1.55rem;font-weight:950;white-space:nowrap}.v171-prob small{display:block;font-size:.65rem;color:#8fa8bd;text-align:right;font-weight:700}
    .v171-badges{display:flex;gap:6px;flex-wrap:wrap;margin:11px 0 8px}.v171-badge{border:1px solid #28465d;border-radius:999px;padding:4px 8px;font-size:.72rem;font-weight:850;color:#cbd9e5}.v171-good{border-color:#1f7d55;color:#65efaa}.v171-hot{border-color:#8b6d18;color:#ffd66d}.v171-reason{font-size:.86rem;color:#c5d3df;line-height:1.4}.v171-empty{border:1px dashed #28465d;border-radius:16px;padding:14px;color:#91a8bc;margin-bottom:8px}
    @media(max-width:520px){.v171-card{padding:13px 14px}.v171-team{font-size:1.12rem}.v171-prob{font-size:1.42rem}}
    </style>
    <div class="v171-head"><h2>Top 10 strongest teams</h2><p>Every unfinished fixture in the selected dates can enter this ranking. Odds verification does not decide whether a team is shown. No EV or edge filter. You choose what goes into your bet.</p></div>
    """,unsafe_allow_html=True)

    # Recommendations include UPCOMING plus clearly marked recent LIVE games.
    # Live rows use current market fair probability for ordering when available.
    _active=_bettable_now.copy()
    likely=_active[_active["Pick"].isin(["HOME","AWAY"])].sort_values("Ranking %",ascending=False).head(10)
    anchor_five=_anchor_five(d)
    _record_anchor_snapshots(anchor_five)
    _settle_ledger()
    _settle_anchor_history_from_ledger()

    # Probability Top 10 and complete Five-Team Anchors answer different questions.

    def _render_clean_rows(frame,lens):
        if frame.empty:
            msg="No unfinished win selections are available in this fixture window."
            st.markdown('<div class="v171-empty">'+html.escape(msg)+'</div>',unsafe_allow_html=True); return
        for pos,(_,r) in enumerate(frame.iterrows(),1):
            team=_team_for_pick(r); conf=_num(r,"Confidence %",0); price=_price_for(r); ctx=_context_signal_v171(r); market=_market_signal(r); val=_validation_word(r)
            audit=r.get("V18 audit") if isinstance(r.get("V18 audit"),dict) else score_selection(r)
            game_state=str(r.get("Game state","")); is_live=game_state=="LIVE"
            display_prob=_num(r,"Ranking %",conf); prob_label="live market fair" if is_live and pd.notna(r.get("Market fair %")) else "model chance"
            live_badge=('<span class="v171-badge v171-hot">🔴 '+html.escape(str(r.get("Timing label","LIVE")))+'</span>') if is_live else ''
            odds_badge='<span class="v171-badge">Odds '+html.escape(price)+'</span>' if price!="—" else '<span class="v171-badge v171-hot">MARKET NOT VERIFIED</span>'
            main_badge='<span class="v171-badge v171-good">RANKED WIN PICK</span>'
            if str(r.get("Decision",""))=="PREDICTION ONLY":
                main_badge='<span class="v171-badge">PREDICTION ONLY</span>'+main_badge
            draw_badge='<span class="v171-badge v171-hot">DRAW THREAT</span>' if _num(r,"Draw %",0) > _num(r,"Confidence %",0) else ''
            _case=r.get("Analyst case") if isinstance(r.get("Analyst case"),dict) else _analyst_case(r)
            main_badge=live_badge+draw_badge+f'<span class="v171-badge">Anchor {_case["score"]:.0f}/100</span>'+f'<span class="v171-badge">Evidence {audit["score"]:.0f}/100</span>'+main_badge
            reason=_clean_reason(r,lens)
            if is_live: reason="TIME-SENSITIVE: match already started. Check the live score and current price immediately. "+reason
            card=('<div class="v171-card"><div class="v171-top"><div><div class="v171-team">'+str(pos)+'. '+html.escape(team)+'</div><div class="v171-pick">'+html.escape(str(r["Pick"]))+' • '+html.escape(str(r.get("League","")))+'</div></div><div class="v171-prob">'+f'{display_prob:.0f}'+'%<small>'+html.escape(prob_label)+'</small></div></div><div class="v171-badges">'+main_badge+odds_badge+'<span class="v171-badge">Model '+html.escape(val)+'</span><span class="v171-badge">Market '+market+'</span><span class="v171-badge">Context '+ctx+'</span></div><div class="v171-reason">'+html.escape(reason)+'</div></div>')
            st.markdown(card,unsafe_allow_html=True)
            with st.expander("See full analysis",expanded=False): _detail_panel(r)

    st.markdown('<div class="v171-lens">🧠 Five-Team Anchors</div>',unsafe_allow_html=True)
    st.caption("The closest app equivalent to the full pre-bet conversation: probability first, then form, opponent quality, scoring profile, availability, schedule, market cross-check and an adversarial attempt to find the reason the pick could fail.")
    if not anchor_five.empty:
        _afive=anchor_five[["Match date","League","Match","Pick","Ranking %","Anchor score","Stability","Late info","Anchor readiness","Analyst verdict","Evidence completeness %","Best market odds","External data status"]].copy()
        _afive["Selection"]=anchor_five.apply(_team_for_pick,axis=1); _afive["Current odds"]=_afive["Best market odds"].apply(decimal_to_fractional)
        _afive=_afive[["Match date","League","Selection","Ranking %","Anchor score","Stability","Late info","Anchor readiness","Analyst verdict","Evidence completeness %","Current odds","External data status"]]
        _afive=_afive.rename(columns={"Ranking %":"Model win %","Evidence completeness %":"Evidence %"})
        st.dataframe(_afive,hide_index=True,use_container_width=True)
        for _pos,(_, _r) in enumerate(anchor_five.iterrows(),1):
            _case=_r.get("Analyst case") if isinstance(_r.get("Analyst case"),dict) else _analyst_case(_r); _team=_team_for_pick(_r)
            st.markdown(f"**{_pos}. {_team} — Anchor {_r.get('Anchor score',_case['score']):.0f}/100 · Model {_r['Ranking %']:.0f}% · Stability {_r.get('Stability','—')} · {_r.get('Anchor readiness','PROVISIONAL')}**")
            st.caption("Why it made the five: "+(" • ".join(_case.get("supports",[])[:2]) or "Strong overall analytical ranking."))
            st.caption("What could beat it: "+(" • ".join(_case.get("risks",[])[:2]) or "No major contradiction found in the available evidence.")); st.caption("Late-info status: "+str(_r.get("Late info","—")))
            with st.expander(f"Full anchor analysis — {_team}",expanded=False): _detail_panel(_r)
    else:
        st.info("No upcoming outright-win candidates are available in this window.")

    st.markdown('<div class="v171-lens">🏆 Full Top 10 ranking</div>',unsafe_allow_html=True)
    st.caption("Ranked strongest to weakest across every unfinished fixture in the selected dates. Upcoming matches use the model whether or not odds are already verified; LIVE matches are included only while a current market remains available.")
    _render_clean_rows(likely,"likely")

    with st.expander(f"All ranked unfinished fixtures ({len(d)})",expanded=False):
        for _,r in d.sort_values("Confidence %",ascending=False).iterrows():
            team=_team_for_pick(r)
            _state_prefix="🔴 LIVE • " if str(r.get("Game state",""))=="LIVE" else ""
            with st.expander(f'{_state_prefix}{team} • {r["Pick"]} • {r["Ranking %"]:.0f}% win probability',expanded=False): _detail_panel(r)
    with st.expander("📈 Performance, learning & validation",expanded=False):
        _tracker_panel()
        _ensure_anchor_history(); _ah=st.session_state.v27_anchor_history.copy()
        if not _ah.empty:
            settled=_ah[pd.to_numeric(_ah["Won"],errors="coerce").notna()].copy()
            st.caption(f"Frozen anchor snapshots: {len(_ah)} • settled: {len(settled)}")
            if not settled.empty:
                st.dataframe(settled[["League","Home team","Away team","Pick","Anchor score","Stability","Won","Diagnostic tag"]].tail(30),hide_index=True,use_container_width=True)
            st.download_button("⬇️ EXPORT ANCHOR HISTORY",data=_ah.to_csv(index=False).encode("utf-8"),
                               file_name="v27_anchor_history.csv",mime="text/csv",use_container_width=True)
        _restore=st.file_uploader("Restore Anchor History CSV",type=["csv"],key="restore_v27_anchor_history")
        if _restore is not None and st.button("RESTORE ANCHOR HISTORY",use_container_width=True,key="restore_anchor_history_btn"):
            try:
                _restored=pd.read_csv(_restore)
                missing=[c for c in ANCHOR_HISTORY_COLUMNS if c not in _restored.columns]
                if missing:
                    st.error("Restore file is missing required columns: "+", ".join(missing))
                else:
                    st.session_state.v27_anchor_history=_restored[ANCHOR_HISTORY_COLUMNS].copy()
                    st.success(f"Restored {len(_restored)} anchor snapshots.")
                    st.rerun()
            except Exception as e:
                st.error(f"Anchor-history restore failed: {e}")
    _stats=_v28_api_state()
    _q=f"daily remaining {_stats.get('daily_remaining')}" if _stats.get("daily_remaining") is not None else "daily quota unknown"
    _m=f"minute remaining {_stats.get('minute_remaining')}" if _stats.get("minute_remaining") is not None else "minute quota unknown"
    st.caption(f"API-Football: {_stats.get('network_calls',0)} real network call(s) from {_stats.get('logical_calls',0)} logical request(s); cache avoided approximately {max(0,_stats.get('logical_calls',0)-_stats.get('network_calls',0))} repeated call(s) • {_q} • {_m}.")
    if quota_remaining is not None: st.caption(f"Odds API credits remaining: {quota_remaining}")
    if not odds_key: st.warning("No current-odds key is connected, so current bettable-market verification is unavailable.")

with st.expander("🔌 Advanced data providers",expanded=False):
    _af=_streamlit_api_football_secret()
    st.write("**API-Football enrichment:** "+("✅ Connected" if _af else "⚪ Not connected"))
    st.caption("API-Football is on-demand. RUN FULL VERIFIED ANALYST PASS adds match-specific injuries/suspensions, player-importance estimates, confirmed line-ups, all-competition workload, an independent provider forecast and provider xG where supplied. Missing data is never invented.")
    st.write("**The Odds API:** "+("✅ Connected" if (_streamlit_odds_secret() or st.session_state.get("odds_key_override")) else "⚪ Not connected"))
    st.caption("Current prices feed return calculations immediately. Predictive market blending is never assigned a hand-picked weight: V26 learns it from settled ledger history and activates it only after chronological holdout improvement.")

st.divider()
st.caption("V28 Hardened Production: holdout-audited no-fit scoring, strict FULL verification, adaptive rate/quota handling, provider settlement, Bayesian trust and superseded-recommendation control. No model can guarantee a result.")

