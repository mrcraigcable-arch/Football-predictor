import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from collections import defaultdict, deque
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

st.set_page_config(page_title="Football Predictor", page_icon="⚽", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:1.5rem;max-width:1100px}
[data-testid="stMetricValue"]{font-size:1.7rem}
</style>
""", unsafe_allow_html=True)

st.title("⚽ Football Predictor")
st.caption("Probability + market-value screening. Predictions are probabilities, not guaranteed winners.")

BASE="https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
FIX="https://www.football-data.co.uk/matches/resources/fixtures.csv"
FEATURES=["home_pts","away_pts","home_gf","away_gf","home_ga","away_ga",
          "home_shots","away_shots","home_sot","away_sot","elo_diff",
          "elo_home_prob","market_h","market_d","market_a"]

def getcsv(url):
    r=requests.get(url,timeout=30); r.raise_for_status()
    return pd.read_csv(BytesIO(r.content))

def devig(h,d,a):
    try:
        v=np.array([float(h),float(d),float(a)])
        if np.any(v<=1): return [np.nan]*3
        q=1/v; return list(q/q.sum())
    except: return [np.nan]*3

def oddscols(df):
    for x in [("AvgCH","AvgCD","AvgCA"),("AvgH","AvgD","AvgA"),
              ("B365CH","B365CD","B365CA"),("B365H","B365D","B365A")]:
        if all(c in df.columns for c in x): return x
    return None

@st.cache_data(ttl=21600, show_spinner=False)
def history():
    fs=[]
    for y in range(2014,2027):
        ss=f"{y%100:02d}{(y+1)%100:02d}"
        try:
            d=getcsv(BASE.format(season=ss,league="E0"))
            if {"HomeTeam","AwayTeam","FTR"}.issubset(d.columns):
                d["Date"]=pd.to_datetime(d["Date"],dayfirst=True,errors="coerce")
                fs.append(d)
        except: pass
    if not fs: raise RuntimeError("Historical data source unavailable.")
    return pd.concat(fs,ignore_index=True).sort_values("Date").reset_index(drop=True)

def features(df, window=8, return_state=False):
    hist=defaultdict(lambda:deque(maxlen=window)); elo=defaultdict(lambda:1500.0)
    rows=[]; oc=oddscols(df)
    for _,r in df.sort_values("Date").iterrows():
        h,a=r.HomeTeam,r.AwayTeam; hh=list(hist[h]); aa=list(hist[a])
        av=lambda z,k,d: float(np.mean([x[k] for x in z])) if z else d
        ep=1/(1+10**((elo[a]-(elo[h]+60))/400))
        mh=md=ma=np.nan
        if oc: mh,md,ma=devig(r.get(oc[0]),r.get(oc[1]),r.get(oc[2]))
        rows.append(dict(home=h,away=a,date=r.Date,target=r.get("FTR"),
            home_pts=av(hh,"p",1.35),away_pts=av(aa,"p",1.15),
            home_gf=av(hh,"gf",1.35),away_gf=av(aa,"gf",1.15),
            home_ga=av(hh,"ga",1.15),away_ga=av(aa,"ga",1.35),
            home_shots=av(hh,"s",12),away_shots=av(aa,"s",10),
            home_sot=av(hh,"t",4),away_sot=av(aa,"t",3.5),
            elo_diff=elo[h]-elo[a],elo_home_prob=ep,
            market_h=mh,market_d=md,market_a=ma))
        if r.get("FTR") in ["H","D","A"] and pd.notna(r.get("FTHG")):
            hg,ag=float(r.FTHG),float(r.FTAG)
            hp=3 if r.FTR=="H" else 1 if r.FTR=="D" else 0
            ap=3 if r.FTR=="A" else 1 if r.FTR=="D" else 0
            hist[h].append({"p":hp,"gf":hg,"ga":ag,"s":float(r.get("HS",12) if pd.notna(r.get("HS",np.nan)) else 12),"t":float(r.get("HST",4) if pd.notna(r.get("HST",np.nan)) else 4)})
            hist[a].append({"p":ap,"gf":ag,"ga":hg,"s":float(r.get("AS",10) if pd.notna(r.get("AS",np.nan)) else 10),"t":float(r.get("AST",3.5) if pd.notna(r.get("AST",np.nan)) else 3.5)})
            actual=1 if r.FTR=="H" else .5 if r.FTR=="D" else 0
            delta=20*(actual-ep); elo[h]+=delta; elo[a]-=delta
    return (pd.DataFrame(rows),hist,elo) if return_state else pd.DataFrame(rows)

@st.cache_resource(show_spinner=False)
def model():
    raw=history(); f=features(raw)
    f=f[f.target.isin(["H","D","A"])].copy()
    f[["market_h","market_d","market_a"]]=f[["market_h","market_d","market_a"]].fillna(1/3)
    cut=int(len(f)*.82)
    base=HistGradientBoostingClassifier(max_iter=220,learning_rate=.05,max_leaf_nodes=15,l2_regularization=2,random_state=42)
    base.fit(f.iloc[:cut][FEATURES],f.iloc[:cut].target)
    cal=CalibratedClassifierCV(base,method="sigmoid",cv="prefit")
    cal.fit(f.iloc[cut:][FEATURES],f.iloc[cut:].target)
    return cal,raw

def current_features(raw, fx):
    _,hist,elo=features(raw,return_state=True); oc=oddscols(fx); out=[]
    for _,r in fx.iterrows():
        h,a=r.HomeTeam,r.AwayTeam; hh=list(hist[h]); aa=list(hist[a])
        av=lambda z,k,d: float(np.mean([x[k] for x in z])) if z else d
        ep=1/(1+10**((elo[a]-(elo[h]+60))/400)); mh=md=ma=np.nan
        if oc: mh,md,ma=devig(r.get(oc[0]),r.get(oc[1]),r.get(oc[2]))
        out.append(dict(Date=r.get("Date",""),HomeTeam=h,AwayTeam=a,
          home_pts=av(hh,"p",1.35),away_pts=av(aa,"p",1.15),home_gf=av(hh,"gf",1.35),away_gf=av(aa,"gf",1.15),
          home_ga=av(hh,"ga",1.15),away_ga=av(aa,"ga",1.35),home_shots=av(hh,"s",12),away_shots=av(aa,"s",10),
          home_sot=av(hh,"t",4),away_sot=av(aa,"t",3.5),elo_diff=elo[h]-elo[a],elo_home_prob=ep,
          market_h=mh,market_d=md,market_a=ma))
    return pd.DataFrame(out)

c1,c2=st.columns(2)
min_conf=c1.slider("Minimum confidence",50,90,62,1)/100
min_edge=c2.slider("Minimum market edge",0,15,4,1)/100

if st.button("🔎 ANALYZE CURRENT PREMIER LEAGUE MATCHES",type="primary",use_container_width=True):
    try:
        with st.spinner("Training/checking model and loading current fixtures…"):
            m,raw=model(); fx=getcsv(FIX)
            if "Div" in fx.columns: fx=fx[fx.Div=="E0"].copy()
            if fx.empty: st.warning("No Premier League fixtures are present in the current feed."); st.stop()
            f=current_features(raw,fx)
            for c in ["market_h","market_d","market_a"]: f[c]=f[c].fillna(1/3)
            p=m.predict_proba(f[FEATURES]); ix={c:i for i,c in enumerate(m.classes_)}
            f["HOME"]=p[:,ix["H"]]; f["DRAW"]=p[:,ix["D"]]; f["AWAY"]=p[:,ix["A"]]
            probs=f[["HOME","DRAW","AWAY"]].to_numpy(); mk=f[["market_h","market_d","market_a"]].to_numpy()
            picki=np.argmax(probs,axis=1); names=np.array(["HOME","DRAW","AWAY"])
            f["PICK"]=names[picki]; f["CONFIDENCE"]=probs[np.arange(len(f)),picki]
            f["EDGE"]=f.CONFIDENCE-mk[np.arange(len(f)),picki]
            f["FAIR_ODDS"]=1/f.CONFIDENCE
            f["DECISION"]=np.where((f.CONFIDENCE>=min_conf)&(f.EDGE>=min_edge),"BET","PASS")
            f=f.sort_values(["DECISION","EDGE","CONFIDENCE"],ascending=False)
        bets=(f.DECISION=="BET").sum()
        st.success(f"Analysis complete — {bets} match(es) passed both filters.")
        for _,r in f.iterrows():
            icon="🟢" if r.DECISION=="BET" else "⚪"
            with st.expander(f"{icon} {r.HomeTeam} vs {r.AwayTeam} — {r.DECISION}",expanded=r.DECISION=="BET"):
                a,b,c,d=st.columns(4)
                a.metric("Pick",r.PICK); b.metric("Confidence",f"{r.CONFIDENCE:.1%}")
                c.metric("Market edge",f"{r.EDGE:+.1%}"); d.metric("Model fair odds",f"{r.FAIR_ODDS:.2f}")
                st.write(f"Home **{r.HOME:.1%}** · Draw **{r.DRAW:.1%}** · Away **{r.AWAY:.1%}**")
                if r.DECISION=="BET":
                    st.info("BET means the selection passed the probability and market-edge filters. It does not mean the outcome is guaranteed.")
    except Exception as e:
        st.error("Could not complete analysis. Try again shortly.")
        st.code(str(e))

st.divider()
st.caption("V1 research model. Before staking real money, validate calibration and performance on unseen matches. Late lineups/injuries are not yet incorporated.")
