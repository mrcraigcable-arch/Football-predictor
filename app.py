import streamlit as st
import pandas as pd
import numpy as np
import requests
from collections import defaultdict, deque
from datetime import date
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

st.set_page_config(page_title="Craig's Football Predictor V15", page_icon="📈", layout="wide")



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
 <span class="v14-chip">V15</span>
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
  <div><div class="brand-name">Craig's Football Predictor <span class="vbadge">V15</span></div>
  <div class="brand-sub">Data. Discipline. Evidence-backed decisions.</div></div>
</div>
<div class="hero"><div class="hero-title">🏆 Smarter football predictions</div>
<div class="hero-sub">Historical modelling + current bookmaker consensus + fail-closed verification.</div></div>
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
    url=f"{ODDS_BASE}/{sport_key}/odds/"
    r=requests.get(url,params={"apiKey":api_key,"regions":"uk","markets":"h2h",
                              "oddsFormat":"decimal","dateFormat":"iso"},timeout=25)
    if r.status_code in (401,403):
        raise RuntimeError("Odds API key was rejected.")
    if r.status_code==429:
        raise RuntimeError("Odds API monthly request allowance has been reached.")
    r.raise_for_status()
    data=r.json()
    return data, {
        "remaining":r.headers.get("x-requests-remaining"),
        "used":r.headers.get("x-requests-used"),
        "last":r.headers.get("x-requests-last"),
        "events":len(data) if isinstance(data,list) else 0
    }

def _event_date_utc(e):
    try:
        return pd.to_datetime(e.get("commence_time"),utc=True).date()
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
entered=st.text_input("The Odds API key",value=st.session_state.get("odds_key",""),
                      type="password",placeholder="Paste free The Odds API key",
                      help="Held in this Streamlit session; it is not written to your GitHub repository.")
c1,c2=st.columns(2)
with c1:
    if st.button("Connect odds",use_container_width=True):
        if entered.strip():
            st.session_state["odds_key"]=entered.strip()
            st.success("Odds key loaded for this session.")
        else: st.warning("Paste the key first.")
with c2:
    if st.button("Clear odds key",use_container_width=True):
        st.session_state.pop("odds_key",None); st.rerun()

st.subheader("⚙️ Model parameters")
st.caption("Tap the number boxes to change them. No sliders, so scrolling cannot accidentally alter your rules.")
pc1,pc2=st.columns(2)
with pc1:
    min_conf=st.number_input("Minimum confidence (%)",min_value=45,max_value=90,value=62,step=1)
    min_edge=st.number_input("Minimum market edge (pp)",min_value=0,max_value=20,value=4,step=1)
    min_books=st.number_input("Minimum bookmakers",min_value=3,max_value=25,value=8,step=1)
with pc2:
    max_edge=st.number_input("Manual verification above edge (pp)",min_value=8,max_value=30,value=15,step=1)
    topn=st.number_input("Show top predictions",min_value=3,max_value=20,value=10,step=1)
    day=st.date_input("Match date",date.today())
scope=st.selectbox("Competition",["ALL SUPPORTED LEAGUES"]+list(LEAGUES))

st.info("V15.2 diagnostic market-integrity engine: BET requires matched current odds, bookmaker depth, confidence, edge and positive EV. Large disagreements are isolated for verification.")

if st.button("🔎 ANALYZE MATCHES",use_container_width=True,type="primary"):
    selected=LEAGUES if scope=="ALL SUPPORTED LEAGUES" else {scope:LEAGUES[scope]}
    odds_key=st.session_state.get("odds_key","").strip()
    quota_remaining=None
    diagnostics=[]
    out=[]; warnings=[]
    with st.spinner("Loading verified fixtures and training league models..."):
        for lname,meta in selected.items():
            code=meta["of"]
            odds_events=[]
            if odds_key:
                try:
                    odds_events,oddiag=odds_fetch(odds_key,meta["odds"])
                    quota_remaining=oddiag.get("remaining")
                    diagnostics.append({"League":lname,"Sport key":meta["odds"],
                                        "API events returned":oddiag.get("events",0),
                                        "Credits used":oddiag.get("used"),
                                        "Credits remaining":oddiag.get("remaining"),
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
            games=[m for m in matches if str(m.get("date",""))[:10]==day.isoformat()]
            if not games: continue
            try:
                model,hist,elo,ntrain,engine_label,validation_status,validation_evidence=train(lname,code)
            except Exception as e:
                warnings.append(f"{lname}: model unavailable ({e})")
                continue
            def av(t,k): return float(np.mean([x[k] for x in hist[t]])) if hist[t] else 0.
            for g in games:
                h=team_name(g.get("team1","")).strip(); a=team_name(g.get("team2","")).strip()
                if not h or not a: continue
                vals={"h_pts":av(h,"pts"),"a_pts":av(a,"pts"),
                      "h_gf":av(h,"gf"),"a_gf":av(a,"gf"),
                      "h_ga":av(h,"ga"),"a_ga":av(a,"ga"),
                      "elo_diff":elo[h]-elo[a],"elo_home":elo_p(elo[h]+55-elo[a])}
                x=pd.DataFrame([[vals[k] for k in FEATURES]],columns=FEATURES)
                pr=model.predict_proba(x)[0]
                i=int(np.argmax(pr)); labels=["HOME","DRAW","AWAY"]; conf=float(pr[i])
                market,matchdiag=consensus_for(odds_events,h,a,day) if odds_events else (None,{"stage":"no-events","reason":"No odds events returned","trace":[],"rejected_books":[]})
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
                odd=np.nan; best_odd=np.nan; mprob=np.nan; edge=np.nan; ev=np.nan; books=0
                if market:
                    med,mfair,books=market["median"],market["fair"],market["books"]
                    # Display consensus price; calculate EV at the best verified UK price.
                    odd=float(med[i]); best_odd=float(market["best"][i]); mprob=float(mfair[i])
                    edge=conf-mprob
                    ev=conf*best_odd-1
                if not market:
                    # No market data means there is not enough evidence to make a betting
                    # decision at all. Keep this blue regardless of model confidence.
                    decision="PREDICTION ONLY"
                elif market.get("integrity")!="OK":
                    decision="VERIFY"
                elif books < min_books:
                    decision="VERIFY"
                elif edge > max_edge/100:
                    decision="VERIFY"
                elif conf>=min_conf/100 and edge>=min_edge/100 and ev>0:
                    decision="BET"
                else:
                    decision="PASS"
                out.append({"League":lname,"Match":f"{h} v {a}","Pick":labels[i],
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
                            "Decision":decision,"Training matches":ntrain,
                            "Model engine":engine_label,"Validation":validation_status,
                            "Validation evidence":validation_evidence})
    if warnings:
        with st.expander("Data/model warnings"):
            for w in warnings: st.warning(w)
    if not out:
        st.info("No supported OpenFootball fixtures were found for that date.")
        st.stop()
    d=pd.DataFrame(out).sort_values("Confidence %",ascending=False)

    # V15 dashboard summary
    bet_count=int((d.Decision=="BET").sum())
    verify_count=int((d.Decision=="VERIFY").sum())
    pass_count=int((d.Decision=="PASS").sum())
    pred_count=int((d.Decision=="PREDICTION ONLY").sum())
    st.subheader("📊 Analysis overview")
    m1,m2,m3,m4,m5=st.columns(5)
    m1.metric("🎯 Analysed",len(d))
    m2.metric("🟢 BET",bet_count)
    m3.metric("🟠 Verify",verify_count)
    m4.metric("🔵 Prediction",pred_count)
    m5.metric("🔴 Pass",pass_count)

    if odds_key:
        with st.expander("🧪 Data diagnostics"):
            st.caption("Technical feed details. The API key is never displayed.")
            if diagnostics:
                st.dataframe(pd.DataFrame(diagnostics),hide_index=True,use_container_width=True)
            else:
                st.warning("No odds diagnostics were produced.")

    st.subheader("💚 Strongest qualifying selections")
    bets=d[d.Decision=="BET"].sort_values(["Edge pp","Confidence %"],ascending=False).head(topn)
    if len(bets):
        st.success(f"{len(bets)} selection(s) clear every automatic rule.")
    elif st.session_state.get("odds_key","").strip():
        st.info("No fixture clears every automatic BET rule. PASS is the correct output.")
    else:
        st.info("Prediction-only mode: connect current odds before any BET decision.")

    st.markdown("**Decision key:** 🟢 BET &nbsp;&nbsp; 🟠 VERIFY &nbsp;&nbsp; 🔴 PASS &nbsp;&nbsp; 🔵 PREDICTION ONLY", unsafe_allow_html=True)

    def fmt(v,suffix=""):
        if pd.isna(v): return "—"
        return f"{v}{suffix}"

    def match_card(r, expanded=False):
        icon={"BET":"🟢","VERIFY":"🟠","PASS":"🔴","PREDICTION ONLY":"🔵"}.get(r["Decision"],"⚪")
        with st.expander(f'{icon} {r["Match"]} — {r["Pick"]} {r["Confidence %"]:.1f}%', expanded=expanded):
            st.caption(f'{r["League"]}  •  Decision: {r["Decision"]}')
            st.markdown(f'**🧠 {r["Model engine"]}**  ·  Validation: **{r["Validation"]}**')
            st.caption(r["Validation evidence"])
            c1,c2,c3=st.columns(3)
            c1.metric("Home",f'{r["Home %"]:.1f}%')
            c2.metric("Draw",f'{r["Draw %"]:.1f}%')
            c3.metric("Away",f'{r["Away %"]:.1f}%')
            c1,c2=st.columns(2)
            c1.metric("Model fair odds",fmt(r["Fair odds"]))
            c2.metric("Market odds",fmt(r["Market odds"]))
            if pd.notna(r.get("Best market odds")):
                st.caption(f'Best verified UK price for EV: {r["Best market odds"]:.2f} • Market integrity: {r.get("Market integrity","—")}')
            if r.get("Bookmaker detail"):
                with st.expander("Bookmaker 1X2 audit"):
                    st.dataframe(pd.DataFrame(r["Bookmaker detail"]),use_container_width=True,hide_index=True)
            c1,c2=st.columns(2)
            c1.metric("Market fair %",fmt(r["Market fair %"],"%"))
            c2.metric("Bookmakers",fmt(r["Bookmakers"]))
            c1,c2=st.columns(2)
            c1.metric("Edge",fmt(r["Edge pp"]," pp"))
            c2.metric("Model EV",fmt(r["EV %"],"%"))
            if r["Decision"]=="VERIFY":
                st.warning("Manual verification required: the market disagreement is unusually large or bookmaker coverage is too thin. This is deliberately NOT labelled BET.")
            elif r["Decision"]=="BET":
                st.success("Clears the current confidence, market-edge, bookmaker-count and positive-EV rules.")
            elif pd.isna(r["Market odds"]):
                st.info("🔵 PREDICTION ONLY — no matched current odds, so this is not a PASS and not a BET.")

    if len(bets):
        for _,r in bets.iterrows():
            match_card(r, expanded=True)

    verifies=d[d.Decision=="VERIFY"].sort_values("Edge pp",ascending=False)
    if len(verifies):
        st.subheader("🟠 Manual verification queue")
        st.caption("Large model/market disagreements are isolated here instead of being treated as automatic value.")
        for _,r in verifies.head(topn).iterrows():
            match_card(r)

    st.subheader("📋 All analysed matches")
    for _,r in d.head(max(topn,20)).iterrows():
        match_card(r)

    if quota_remaining is not None:
        st.caption(f"Odds API credits remaining (provider header): {quota_remaining}")
    if not st.session_state.get("odds_key","").strip():
        st.warning("No current-odds key is connected, so BET labels, market edge and EV remain disabled.")

st.divider()
st.caption("V15.2 fail-closed rule: BET requires matched current UK 1X2 bookmaker prices, de-margined market probability, sufficient model confidence, minimum edge and positive EV.")

st.markdown("""
<div class="v14-nav">
 <span class="active">🏠<br>Matches</span>
 <span>📊<br>Analysis</span>
 <span>🎯<br>Calibrate</span>
 <span>⚙️<br>Settings</span>
</div>
""", unsafe_allow_html=True)
