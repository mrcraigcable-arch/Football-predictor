import streamlit as st
import pandas as pd
import numpy as np
import requests
import html
from collections import defaultdict, deque
from fractions import Fraction
from itertools import combinations
from math import isfinite, log
from datetime import date, datetime, timezone, timedelta
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss



# --- V18 DECISION / SAFETY ENGINE (embedded in V19.5) ---------------------------
# Kept inside app.py so Streamlit cannot deploy the UI and selection engine from
# different commits. This is the V18 scoring/acca logic preserved in V19.5.
WEIGHTS = {
    "model_probability": 30,
    "goals_profile": 20,
    "recent_venue_form": 15,
    "market_value": 15,
    "availability_evidence": 10,
    "opponent_strength": 5,
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
    if decision=="PASS": penalties.append((15,"Existing confidence/value gate failed"))
    if decision=="PREDICTION ONLY": penalties.append((10,"No matched current market"))
    if pick=="DRAW": penalties.append((20,"Draws are excluded from the acca shortlist"))
    if form_advantage is not None and form_advantage<=-.35: penalties.append((7,"Recent form conflicts with the model pick"))
    if goals_advantage is not None and goals_advantage<0: penalties.append((6,"Scoring profile conflicts with the model pick"))
    penalty_total=sum(p for p,_ in penalties)
    final_score=round(_v18_clip(base_score-penalty_total),1)
    outright=pick in ("HOME","AWAY")
    if outright and decision=="BET" and final_score>=70: tier="ELITE"
    elif outright and decision in ("BET","VERIFY") and final_score>=60: tier="STRONG"
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
        if pick in ("HOME","AWAY") and price is not None and price>1.01 and prob is not None and 0<prob<100 and str(row.get("Market integrity",""))=="OK" and books>=min_books and bool(row.get("Kickoff ISO")) and accepted:
            candidates.append({"row":row,"audit":audit,"price":price,"probability":prob/100.0})
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
    return {"status":"READY" if ready else "BELOW_TARGET","legs":[x["row"] for x in combo],"combined_odds":round(combined,2),"joint_probability":round(joint*100,2),"target_odds":round(target,2),"reason":"Highest estimated joint success probability among combinations reaching the target." if ready else "No requested-size combination reaches the target with verified prices; this is the closest available return."}
# --- END V18 ENGINE -------------------------------------------------------------

st.set_page_config(page_title="Craig's Football Predictor V19.5", page_icon="📈", layout="wide")



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
 <span class="v14-chip">V19.5</span>
 <div class="v14-brandline"><span class="v14-logo">📈</span>
 <div><div class="v14-title">Craig's Football <b>Predictor</b></div>
 <div class="v14-sub">Data. Discipline. Evidence-backed decisions. • Real market comparison</div></div></div>
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
  <div><div class="brand-name">Craig's Football Predictor <span class="vbadge">V19.5</span></div>
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

RAW="https://raw.githubusercontent.com/openfootball/football.json/master"
LEAGUES={
    "Premier League":{"of":"en.1","odds":"soccer_epl"},
    "Championship":{"of":"en.2","odds":"soccer_efl_champ"},
    "Bundesliga":{"of":"de.1","odds":"soccer_germany_bundesliga"},
    "La Liga":{"of":"es.1","odds":"soccer_spain_la_liga"},
    "Serie A":{"of":"it.1","odds":"soccer_italy_serie_a"},
    "Ligue 1":{"of":"fr.1","odds":"soccer_france_ligue_one"},
}

# V15 league-aware promotion policy. These choices are based on the V14
# chronological unseen-data calibration tests at the 62% audit threshold.
# No league is allowed to inherit another league's calibration result.
V15_POLICY={
    "Premier League":{"engine":"calibrated","status":"APPROVED","evidence":"V14 gap +6.1pp → +2.1pp"},
    "Championship":{"engine":"raw","status":"RAW FALLBACK","evidence":"V14 calibrated validation unavailable"},
    "Bundesliga":{"engine":"calibrated","status":"APPROVED","evidence":"V14 gap +3.4pp → -1.5pp"},
    "La Liga":{"engine":"raw","status":"APPROVED RAW","evidence":"Raw gap -0.7pp; calibration worsened to -11.7pp"},
    "Serie A":{"engine":"raw","status":"APPROVED RAW","evidence":"Calibration improvement too small for 56% fewer selections"},
    "Ligue 1":{"engine":"raw","status":"APPROVED RAW","evidence":"Raw gap -2.1pp; calibration worsened to -9.9pp"},
}
# Only leagues actually present in OpenFootball's 2026/27 JSON repository are exposed.
SEASONS=["2018-19","2019-20","2020-21","2021-22","2022-23","2023-24","2024-25","2025-26","2026-27"]
FEATURES=["h_pts","a_pts","h_gf","a_gf","h_ga","a_ga","elo_diff","elo_home"]

HEADERS={"User-Agent":"Mozilla/5.0 FootballPredictorV15/1.0","Accept":"application/json"}

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

def make_training(code,w=8):
    hist=defaultdict(lambda:deque(maxlen=w))
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
            hg,ag=sc; eh,ea=elo[h],elo[a]
            def av(t,k):
                return float(np.mean([x[k] for x in hist[t]])) if hist[t] else 0.
            vals={"h_pts":av(h,"pts"),"a_pts":av(a,"pts"),
                  "h_gf":av(h,"gf"),"a_gf":av(a,"gf"),
                  "h_ga":av(h,"ga"),"a_ga":av(a,"ga"),
                  "elo_diff":eh-ea,"elo_home":elo_p(eh+55-ea)}
            if hg>ag: y=0; hp,ap,s=3,0,1.
            elif hg<ag: y=2; hp,ap,s=0,3,0.
            else: y=1; hp,ap,s=1,1,.5
            rows.append({**vals,"y":y})
            hist[h].append({"pts":hp,"gf":hg,"ga":ag})
            hist[a].append({"pts":ap,"gf":ag,"ga":hg})
            ex=elo_p(eh-ea)
            elo[h]+=24*(s-ex); elo[a]+=24*((1-s)-(1-ex))
            used+=1
    return pd.DataFrame(rows),hist,elo,used



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


@st.cache_resource(show_spinner=False)
def train(lname,code):
    """V15 live engine: conservative model with league-specific calibration policy."""
    f,hist,elo,used=make_training(code)
    if len(f)<120:
        raise RuntimeError(f"Only {len(f)} completed historical matches available.")
    X=f[FEATURES].fillna(0); y=f["y"]
    policy=V15_POLICY[lname]

    def conservative():
        return HistGradientBoostingClassifier(
            max_iter=100,max_leaf_nodes=7,learning_rate=.045,
            min_samples_leaf=38,l2_regularization=8,random_state=42)

    if policy["engine"]=="calibrated":
        # Strict chronology: base model sees the earlier 82%; sigmoid calibrator
        # sees only the later 18%. No future fixture/result enters either stage.
        cut=max(250,int(len(f)*.82))
        proper=f.iloc[:cut]; cal=f.iloc[cut:]
        base=conservative()
        base.fit(proper[FEATURES].fillna(0),proper["y"])
        model=base
        if len(cal)>=50 and cal["y"].nunique()==3:
            try:
                calibrated=CalibratedClassifierCV(base,method="sigmoid",cv="prefit")
                calibrated.fit(cal[FEATURES].fillna(0),cal["y"])
                model=calibrated
            except Exception:
                # Fail closed to raw rather than pretending calibration succeeded.
                model=base
        engine_label="V15 CALIBRATED CONSERVATIVE" if model is not base else "V15 RAW SAFETY FALLBACK"
    else:
        model=conservative()
        model.fit(X,y)
        engine_label="V15 RAW CONSERVATIVE"

    return model,hist,elo,used,engine_label,policy["status"],policy["evidence"]

@st.cache_data(ttl=1800,show_spinner=False)
def fixtures_for(code):
    return season_json("2026-27",code)["matches"]


ODDS_BASE="https://api.the-odds-api.com/v4/sports"

def norm(s):
    import re, unicodedata
    s=unicodedata.normalize("NFKD",str(s)).encode("ascii","ignore").decode().lower()
    s=re.sub(r"\b(fc|cf|afc|ac|calcio|club)\b"," ",s)
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

st.markdown('<div class="v14-section">🧪 Model validation</div>', unsafe_allow_html=True)
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


st.markdown("""
<div style="background:linear-gradient(135deg,#063b31,#08253d);border:1px solid #00e59b;
border-radius:20px;padding:18px;margin:12px 0 18px 0;">
<div style="font-size:13px;color:#77f7c7;font-weight:800;letter-spacing:.08em;">LIVE ENGINE</div>
<div style="font-size:26px;font-weight:900;color:white;margin-top:4px;">🎯 Calibrated Conservative live</div>
<div style="color:#b9c7d5;margin-top:8px;line-height:1.5;">
V15 promotes only the league-specific probability treatment supported by V14 unseen-data tests: calibrated Conservative for Premier League and Bundesliga; raw Conservative for La Liga, Serie A and Ligue 1; Championship remains a raw safety fallback until calibration is validated.
</div>
</div>
""",unsafe_allow_html=True)

st.markdown('<div class="v14-section">🧠 V14 Model Lab</div>',unsafe_allow_html=True)
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


st.markdown("### 🧭 V15.2 Market-Integrity Fix")
st.caption("The live predictor now chooses the probability engine per competition from V14 unseen-data evidence. It never applies calibration globally.")
with st.expander("View league engine policy",expanded=False):
    policy_rows=[]
    for _league,_p in V15_POLICY.items():
        policy_rows.append({"Competition":_league,"Live engine":"Calibrated Conservative" if _p["engine"]=="calibrated" else "Raw Conservative","Status":_p["status"],"Evidence":_p["evidence"]})
    st.dataframe(pd.DataFrame(policy_rows),hide_index=True,use_container_width=True)

st.subheader("🔐 Live data connection")

# V19.5: load the Odds API key automatically from Streamlit Secrets.
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

st.subheader("⚙️ Model parameters")
st.caption("The default view analyses Today automatically and shows the 10 most likely winners. Use the fixture window only when you want to narrow or extend the dates.")
pc1,pc2=st.columns(2)
with pc1:
    min_conf=st.number_input("Minimum confidence (%)",min_value=45,max_value=90,value=62,step=1)
    min_edge=st.number_input("Minimum market edge (pp)",min_value=0,max_value=20,value=4,step=1)
    min_books=st.number_input("Minimum bookmakers",min_value=3,max_value=25,value=8,step=1)
with pc2:
    max_edge=st.number_input("Manual verification above edge (pp)",min_value=8,max_value=30,value=15,step=1)
    topn=st.number_input("Show top predictions",min_value=3,max_value=30,value=10,step=1)

# V19.5 fixture picker: date changes stay local until GO is pressed.
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

st.markdown("### 📅 Fixtures to analyse")
st.caption("Quick-pick a common window, or choose your own range. The app only refreshes the football analysis when you press a GO button.")
_q1,_q2,_q3=st.columns(3)
with _q1:
    if st.button("TODAY — GO",use_container_width=True,key="fixtures_today_go"):
        st.session_state["fixture_start_day"]=_today
        st.session_state["fixture_end_day"]=_today
        st.session_state["fixture_range_picker"]=(_today,_today)
        st.rerun()
with _q2:
    if st.button("SATURDAY — GO",use_container_width=True,key="fixtures_saturday_go"):
        st.session_state["fixture_start_day"]=_this_saturday
        st.session_state["fixture_end_day"]=_this_saturday
        st.session_state["fixture_range_picker"]=(_this_saturday,_this_saturday)
        st.rerun()
with _q3:
    if st.button("NEXT 7 DAYS — GO",use_container_width=True,key="fixtures_week_go"):
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
    _apply_dates=st.form_submit_button("GO — ANALYSE THIS DATE RANGE",use_container_width=True,type="primary")

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

st.markdown("### 🎯 Personalise acca")
st.caption("Optional — the Top 10 still loads automatically. Enter a stake and target only when you want a tailored 5/6-team acca.")
ac1,ac2,ac3=st.columns(3)
with ac1:
    acca_stake=st.number_input("Amount to bet (£)",min_value=1.0,max_value=10000.0,value=10.0,step=1.0,format="%.2f",key="acca_stake")
with ac2:
    acca_target=st.number_input("Amount to achieve (£)",min_value=2.0,max_value=100000.0,value=500.0,step=10.0,format="%.2f",key="acca_target")
with ac3:
    acca_legs_choice=st.selectbox("Acca size",["Best of 5 or 6","5 teams","6 teams"],index=0,key="acca_legs_choice")
if st.button("BUILD MOST-PROBABLE ACCA",use_container_width=True,key="build_personal_acca"):
    st.session_state["build_personal_acca_requested"]=True

build_acca_requested=bool(st.session_state.get("build_personal_acca_requested",False))

# --- V16 live validation ledger -------------------------------------------------
LEDGER_COLUMNS=[
    "Signal ID","Recorded UTC","League","Home team","Away team","Kickoff ISO","Kickoff UK","Pick",
    "Model probability %","Market fair %","Consensus odds","Entry best odds","Latest best odds",
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

def _signal_id(r):
    return "|".join([str(r.get("League","")),str(r.get("Home team","")),str(r.get("Away team","")),str(r.get("Kickoff ISO","")),str(r.get("Pick",""))])

def _record_live_bets(df):
    _ensure_ledger(); led=st.session_state.v16_ledger.copy(); now=datetime.now(timezone.utc)
    for _,r in df[df.Decision=="BET"].iterrows():
        sid=_signal_id(r); best=float(r["Best market odds"]); kickoff=pd.to_datetime(r.get("Kickoff ISO"),utc=True,errors="coerce")
        mins=(kickoff.to_pydatetime()-now).total_seconds()/60 if pd.notna(kickoff) else np.nan
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
            row.update({"Signal ID":sid,"Recorded UTC":now.isoformat(),"League":r["League"],"Home team":r["Home team"],"Away team":r["Away team"],
                "Kickoff ISO":r.get("Kickoff ISO"),"Kickoff UK":r.get("Kickoff UK"),"Pick":r["Pick"],"Model probability %":r["Confidence %"],
                "Market fair %":r.get("Market fair %"),"Consensus odds":r.get("Market odds"),"Entry best odds":best,"Latest best odds":best,
                "Bookmakers":r.get("Bookmakers"),"Edge pp":r.get("Edge pp"),"Expected value %":r.get("EV %"),"Engine":r.get("Model engine"),
                "Validation":r.get("Validation"),"Result":"PENDING","Won":np.nan,"Profit units":np.nan,"Latest snapshot UTC":now.isoformat(),
                "Minutes to kickoff":round(mins,1) if np.isfinite(mins) else np.nan,"CLV %":0.0,"CLV status":"ENTRY SNAPSHOT"})
            led=pd.concat([led,pd.DataFrame([row])],ignore_index=True)
    st.session_state.v16_ledger=led[LEDGER_COLUMNS]

def _settle_ledger():
    _ensure_ledger(); led=st.session_state.v16_ledger.copy()
    for j,r in led.iterrows():
        if str(r.get("Result","PENDING")) not in ("PENDING","nan",""): continue
        lname=r.get("League"); meta=LEAGUES.get(lname)
        if not meta: continue
        try: matches=fixtures_for(meta["of"])
        except Exception: continue
        target=None
        for m in matches:
            h=team_name(m.get("team1","")).strip(); a=team_name(m.get("team2","")).strip()
            if team_match(r.get("Home team",""),h) and team_match(r.get("Away team",""),a):
                sc=score_ft(m)
                if sc is not None: target=sc; break
        if target is None: continue
        hg,ag=target; actual="HOME" if hg>ag else "AWAY" if ag>hg else "DRAW"; won=(actual==r.get("Pick"))
        price=float(r.get("Entry best odds")) if pd.notna(r.get("Entry best odds")) else np.nan
        led.at[j,"Result"]=f"{hg}-{ag} ({actual})"; led.at[j,"Won"]=bool(won); led.at[j,"Profit units"]=round(price-1,3) if won and np.isfinite(price) else -1.0
    st.session_state.v16_ledger=led[LEDGER_COLUMNS]

def _tracker_panel():
    _ensure_ledger(); _settle_ledger(); led=st.session_state.v16_ledger
    st.subheader("📈 Live performance")
    st.caption("Signals are frozen at first BET classification. Re-running before kickoff updates only the latest-price snapshot. The ledger lives in this Streamlit session, so export it to keep a durable copy.")
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

scope=st.selectbox("Competition",["ALL SUPPORTED LEAGUES"]+list(LEAGUES))

st.info("V19.5 • CONSOLIDATED BUILD — V18 evidence engine + visible personalised acca + automatic secret-based odds connection.")

if True:
    selected=LEAGUES if scope=="ALL SUPPORTED LEAGUES" else {scope:LEAGUES[scope]}
    odds_key=(st.session_state.get("odds_key_override", "").strip() or _streamlit_odds_secret())
    quota_remaining=None
    diagnostics=[]
    out=[]; warnings=[]
    with st.spinner("Loading verified fixtures and training league models..."):
        for lname,meta in selected.items():
            code=meta["of"]
            odds_events=[]
            api_diag={}  # reset per league; never reuse diagnostics from a previous league
            if odds_key:
                try:
                    odds_events,api_diag=odds_fetch(odds_key,meta["odds"])
                    quota_remaining=api_diag.get("remaining")
                    diagnostics.append({"League":lname,"Sport key":meta["odds"],
                                        "API events returned":api_diag.get("events",0),
                                        "Credits used":api_diag.get("used"),
                                        "Credits remaining":api_diag.get("remaining"),
                                        "Status":"OK"})
                    # Record the actual event names/times returned so matching failures are visible.
                    for ev in odds_events[:20]:
                        diagnostics.append({"League":lname,"Sport key":meta["odds"],
                                            "API events returned":"",
                                            "Credits used":"","Credits remaining":"",
                                            "Status":f'ODDS EVENT: {ev.get("home_team","?")} v {ev.get("away_team","?")} @ {ev.get("commence_time","?")}'})
                except Exception as e:
                    warnings.append(f"{lname}: current odds unavailable ({e})")
                    diagnostics.append({"League":lname,"Sport key":meta["odds"],
                                        "API events returned":0,"Credits used":"",
                                        "Credits remaining":"","Status":f"ERROR: {e}"})
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
            try:
                model,hist,elo,ntrain,engine_label,validation_status,validation_evidence=train(lname,code)
            except Exception as e:
                warnings.append(f"{lname}: model unavailable ({e})")
                continue
            def av(t,k): return float(np.mean([x[k] for x in hist[t]])) if hist[t] else 0.
            for g,match_day in games:
                h=team_name(g.get("team1","")).strip(); a=team_name(g.get("team2","")).strip()
                if not h or not a: continue
                vals={"h_pts":av(h,"pts"),"a_pts":av(a,"pts"),
                      "h_gf":av(h,"gf"),"a_gf":av(a,"gf"),
                      "h_ga":av(h,"ga"),"a_ga":av(a,"ga"),
                      "elo_diff":elo[h]-elo[a],"elo_home":elo_p(elo[h]+55-elo[a])}
                x=pd.DataFrame([[vals[k] for k in FEATURES]],columns=FEATURES)
                pr=model.predict_proba(x)[0]
                i=int(np.argmax(pr)); labels=["HOME","DRAW","AWAY"]; conf=float(pr[i])
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
                # Kick-off source priority: uniquely matched Odds API event (UTC -> Europe/London).
                # This is also used as a final event-identity check before a green BET is allowed.
                kickoff_iso=(market.get("event",{}).get("commence_time") if market else None) or matched_kickoff_from_diag(matchdiag)
                kickoff_dt,kickoff_label=kickoff_uk_from_iso(kickoff_iso)
                if not kickoff_label:
                    kickoff_label=f"{match_day.strftime('%a %d %b')} • time unavailable"
                odd=np.nan; best_odd=np.nan; mprob=np.nan; edge=np.nan; ev=np.nan; books=0
                if market:
                    med,mfair,books=market["median"],market["fair"],market["books"]
                    # Display consensus price; calculate EV at the best verified UK price.
                    odd=float(med[i]); best_odd=float(market["best"][i]); mprob=float(mfair[i])
                    edge=conf-mprob
                    ev=conf*best_odd-1
                context = fixture_context(code, match_day, h, a)
                secondary_checks=[]
                # V16.5 decision hierarchy: establish whether this is a positive betting candidate,
                # then automatically investigate unusually large edges instead of handing work to the user. VERIFY is reserved for otherwise-qualifying
                # candidates whose market evidence needs manual checking.
                if not market:
                    decision="PREDICTION ONLY"
                    decision_reason="No matched current market."
                elif conf < min_conf/100:
                    decision="PASS"
                    decision_reason=f"Model confidence {conf*100:.1f}% is below the {min_conf:.0f}% threshold."
                elif edge < min_edge/100:
                    decision="PASS"
                    decision_reason=f"Market edge {edge*100:.1f}pp is below the +{min_edge:.0f}pp threshold."
                elif ev <= 0:
                    decision="PASS"
                    decision_reason=f"Model EV is not positive ({ev*100:.1f}%)."
                elif market.get("integrity")!="OK":
                    decision="VERIFY"
                    decision_reason="Positive candidate, but bookmaker-price dispersion failed the market-integrity guard."
                elif books < min_books:
                    decision="VERIFY"
                    decision_reason=f"Positive candidate, but only {books} bookmakers passed validation (minimum {min_books})."
                elif edge > max_edge/100:
                    decision, decision_reason, secondary_checks = secondary_auto_verify(
                        market, i, conf, min_edge, validation_status, matchdiag, kickoff_iso
                    )
                elif not kickoff_iso:
                    decision="VERIFY"
                    decision_reason="Positive candidate, but the matched market has no verified kickoff timestamp."
                elif matchdiag.get("stage")!="accepted":
                    decision="VERIFY"
                    decision_reason="Positive candidate, but the fixture/market audit did not finish in the accepted state."
                else:
                    decision="BET"
                    decision_reason="AUTO VERIFIED — unique fixture, named 1X2 market, kickoff, confidence, edge, positive EV, bookmaker depth and market integrity all passed."
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
                            "Decision":decision,"Decision reason":decision_reason,"Training matches":ntrain,
                            "Model engine":engine_label,"Validation":validation_status,
                            "Validation evidence":validation_evidence,"Secondary checks":secondary_checks,"Context":context})
    if warnings:
        with st.expander("Data/model warnings"):
            for w in warnings: st.warning(w)
    if not out:
        st.info("No supported OpenFootball fixtures were found in that fixture window.")
        st.stop()
    d=pd.DataFrame(out).sort_values("Confidence %",ascending=False)
    d["V18 audit"]=[score_selection(r) for r in d.to_dict("records")]
    d["Selection score"]=[audit["score"] for audit in d["V18 audit"]]
    d["Tier"]=[audit["tier"] for audit in d["V18 audit"]]
    _record_live_bets(d)

    # V17 CLEAN PICKS DASHBOARD — answer first, detail on demand.
    bet_count=int((d.Decision=="BET").sum())
    strong_mask=(d["Confidence %"] >= 68.0) & (d["Decision"] != "BET")
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
        pred=str(r.get("Prediction","")).upper()
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
        if r.get("Decision")=="BET": return "Model and verified price clear the value rules."
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
            st.write(f'Bookmaker market: {r.get("Market fair %"):.0f}% • Edge: {r.get("Edge pp"):+.1f}pp • EV: {r.get("EV %"):+.1f}% • {int(r.get("Bookmakers",0))} bookmakers')
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
        a2.metric("Tier",audit["tier"])
        a3.metric("Risk penalties",f'{audit["penalty_total"]:.0f}')
        component_labels={"model_probability":"Model probability","goals_profile":"Scoring profile*","recent_venue_form":"Recent + venue form","market_value":"Verified market value","availability_evidence":"Availability evidence","opponent_strength":"Opponent strength","supporting_indicators":"Supporting indicators"}
        component_rows=[{"Evidence component":component_labels[k],"Weight":f"{w}%","Component score":audit["components"][k]} for k,w in WEIGHTS.items()]
        st.dataframe(pd.DataFrame(component_rows),hide_index=True,use_container_width=True)
        st.caption("*Scoring-rate proxy from completed-match goals; it is not provider-supplied xG.")
        if audit.get("positives"): st.success("Supports pick: "+" • ".join(audit["positives"]))
        if audit.get("concerns"): st.warning("Concerns: "+" • ".join(audit["concerns"]))
        if audit.get("penalties"): st.caption("Risk penalties: "+" • ".join(f'-{p["points"]} {p["reason"]}' for p in audit["penalties"]))
        st.caption(r.get("Decision reason", ""))

    # V17.1 THREE-LENS DASHBOARD — three simple questions, detail only on demand.
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
        if lens=="both": return "High win probability and the verified price also passes the value rules."
        if lens=="value": return "Verified bookmaker price clears the model's value, edge and EV rules."
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
        result=build_best_chance_acca(frame.to_dict("records"),target_odds=target_odds,leg_counts=leg_counts,min_books=int(min_books))
        if result["status"]=="INSUFFICIENT":
            return None,{"reason":result["reason"]}
        part=pd.DataFrame(result["legs"]).copy()
        part["_goal_price"]=pd.to_numeric(part["Best market odds"],errors="coerce")
        return {"legs":part,"total_odds":float(result["combined_odds"]),"joint_prob":float(result["joint_probability"])/100.0,"return":float(stake)*float(result["combined_odds"]),"target_odds":target_odds,"status":"TARGET REACHED" if result["status"]=="READY" else "TARGET NOT REACHABLE"},None

    if build_acca_requested:
        st.markdown("### 🧠 Most-probable personalised acca")
        st.caption("The optimiser searches verified 5/6-team combinations and chooses the one with the highest modelled joint win probability that can reach your target.")
        _acca,_acca_err=_build_goal_acca(d,acca_legs_choice,float(acca_stake),float(acca_target))
        if _acca_err:
            st.warning(_acca_err["reason"])
        else:
            actual_legs=len(_acca["legs"])
            m1,m2,m3,m4=st.columns(4)
            m1.metric("Stake",f"£{acca_stake:,.2f}")
            m2.metric("Target",f"£{acca_target:,.2f}")
            m3.metric("Combined odds",decimal_to_fractional(_acca["total_odds"]))
            m4.metric("Model joint chance",f'{_acca["joint_prob"]*100:.1f}%')
            if _acca["status"]=="TARGET REACHED":
                st.success(f'Highest-probability {actual_legs}-team combination found that reaches the target: estimated return £{_acca["return"]:,.2f}.')
            else:
                st.warning(f'The requested £{acca_target:,.2f} return cannot be reached with the selected 5/6-team setting using currently verified prices. Closest available {actual_legs}-team combination returns about £{_acca["return"]:,.2f}.')
            _show=_acca["legs"][["Match date","League","Match","Pick","Confidence %","_goal_price"]].copy()
            _show["Selection"]=_acca["legs"].apply(_team_for_pick,axis=1)
            _show["Odds"]=_acca["legs"]["_goal_price"].apply(decimal_to_fractional)
            _show=_show[["Match date","League","Selection","Match","Pick","Confidence %","Odds"]]
            st.dataframe(_show,hide_index=True,use_container_width=True)
            st.caption("Joint chance multiplies the model probabilities as an independence approximation; it is a ranking aid, not a guaranteed success probability.")

    st.markdown("""
    <style>
    .v171-head{margin:.25rem 0 1rem}.v171-head h2{margin:0!important;font-size:1.65rem!important}.v171-head p{margin:.25rem 0 0;color:#8fa8bd;font-size:.92rem}
    .v171-lens{margin:1.25rem 0 .6rem;font-size:1.15rem;font-weight:900}.v171-card{border:1px solid #1b3c53;border-radius:18px;background:linear-gradient(145deg,#091a29,#06121e);padding:15px 16px;margin:0 0 8px}
    .v171-top{display:flex;justify-content:space-between;gap:12px;align-items:flex-start}.v171-team{font-size:1.2rem;font-weight:950;line-height:1.15}.v171-pick{color:#9db5c9;font-size:.83rem;font-weight:800;margin-top:3px}.v171-prob{font-size:1.55rem;font-weight:950;white-space:nowrap}.v171-prob small{display:block;font-size:.65rem;color:#8fa8bd;text-align:right;font-weight:700}
    .v171-badges{display:flex;gap:6px;flex-wrap:wrap;margin:11px 0 8px}.v171-badge{border:1px solid #28465d;border-radius:999px;padding:4px 8px;font-size:.72rem;font-weight:850;color:#cbd9e5}.v171-good{border-color:#1f7d55;color:#65efaa}.v171-hot{border-color:#8b6d18;color:#ffd66d}.v171-reason{font-size:.86rem;color:#c5d3df;line-height:1.4}.v171-empty{border:1px dashed #28465d;border-radius:16px;padding:14px;color:#91a8bc;margin-bottom:8px}
    @media(max-width:520px){.v171-card{padding:13px 14px}.v171-team{font-size:1.12rem}.v171-prob{font-size:1.42rem}}
    </style>
    <div class="v171-head"><h2>Top selections</h2><p>Probability-ranked picks for the selected fixture window. Tap a match only when you want the full analysis.</p></div>
    """,unsafe_allow_html=True)

    likely=d[d["Pick"].isin(["HOME","AWAY"])].sort_values("Confidence %",ascending=False).head(int(topn))
    value=d[d["Decision"].eq("BET")].copy()
    if not value.empty:
        value["_vscore"]=pd.to_numeric(value["Edge pp"],errors="coerce").fillna(0)+0.15*pd.to_numeric(value["EV %"],errors="coerce").fillna(0)
        value=value.sort_values(["_vscore","Confidence %"],ascending=False).head(5)
    both=d[(d["Decision"].eq("BET")) & (d["Confidence %"].ge(68.0)) & d["Pick"].isin(["HOME","AWAY"])].sort_values("Confidence %",ascending=False).head(5)

    def _render_clean_rows(frame,lens):
        if frame.empty:
            msg={"likely":"No win selections available.","value":"No verified value bets in this fixture window — the app has not lowered the rules to create activity.","both":"No selection currently has both 68%+ win probability and verified value."}[lens]
            st.markdown('<div class="v171-empty">'+html.escape(msg)+'</div>',unsafe_allow_html=True); return
        for pos,(_,r) in enumerate(frame.iterrows(),1):
            team=_team_for_pick(r); conf=_num(r,"Confidence %",0); price=_price_for(r); ctx=_context_signal_v171(r); market=_market_signal(r); val=_validation_word(r)
            audit=r.get("V18 audit") if isinstance(r.get("V18 audit"),dict) else score_selection(r)
            odds_badge='<span class="v171-badge">Odds '+html.escape(price)+'</span>' if price!="—" else '<span class="v171-badge">Odds not verified</span>'
            main_badge='<span class="v171-badge v171-good">VALUE</span>' if r.get("Decision")=="BET" else '<span class="v171-badge">WIN PICK</span>'
            if lens=="both": main_badge='<span class="v171-badge v171-hot">WIN + VALUE</span>'
            tier_class='v171-good' if audit["tier"]=="ELITE" else 'v171-hot' if audit["tier"]=="STRONG" else ''
            main_badge=f'<span class="v171-badge {tier_class}">{html.escape(audit["tier"])}</span><span class="v171-badge">Evidence {audit["score"]:.0f}/100</span>'+main_badge
            card=('<div class="v171-card"><div class="v171-top"><div><div class="v171-team">'+str(pos)+'. '+html.escape(team)+'</div><div class="v171-pick">'+html.escape(str(r["Pick"]))+' • '+html.escape(str(r.get("League","")))+'</div></div><div class="v171-prob">'+f'{conf:.0f}'+'%<small>model chance</small></div></div><div class="v171-badges">'+main_badge+odds_badge+'<span class="v171-badge">Model '+html.escape(val)+'</span><span class="v171-badge">Market '+market+'</span><span class="v171-badge">Context '+ctx+'</span></div><div class="v171-reason">'+html.escape(_clean_reason(r,lens))+'</div></div>')
            st.markdown(card,unsafe_allow_html=True)
            with st.expander("See full analysis",expanded=False): _detail_panel(r)

    st.markdown('<div class="v171-lens">🏆 Most likely winners</div>',unsafe_allow_html=True)
    st.caption(f"Top {int(topn)} highest model win probabilities. Price does not decide this ranking."); _render_clean_rows(likely,"likely")
    st.markdown('<div class="v171-lens">💰 Best value bets</div>',unsafe_allow_html=True)
    st.caption("Only selections that pass the existing bookmaker, edge, EV and verification rules."); _render_clean_rows(value,"value")
    st.markdown('<div class="v171-lens">🔥 Win chance + value</div>',unsafe_allow_html=True)
    st.caption("The overlap: 68%+ predicted win probability and a verified value price."); _render_clean_rows(both,"both")

    with st.expander(f"All other matches ({len(d)})",expanded=False):
        for _,r in d.sort_values("Confidence %",ascending=False).iterrows():
            team=_team_for_pick(r)
            with st.expander(f'{team} • {r["Pick"]} • {r["Confidence %"]:.0f}% • {r["Decision"]}',expanded=False): _detail_panel(r)
    if odds_key:
        with st.expander("Technical / data diagnostics",expanded=False):
            if diagnostics: st.dataframe(pd.DataFrame(diagnostics),hide_index=True,use_container_width=True)
            else: st.info("No odds diagnostics were produced.")
    with st.expander("📈 Model performance & validation",expanded=False): _tracker_panel()
    if quota_remaining is not None: st.caption(f"Odds API credits remaining: {quota_remaining}")
    if not odds_key: st.warning("No current-odds key is connected, so value classifications remain disabled.")
st.divider()
st.caption("V19.5 rule: BET requires matched current UK 1X2 prices, verified fixture/kickoff, bookmaker depth, confidence, minimum edge and positive EV. Live validation records evidence; it does not loosen betting rules.")

