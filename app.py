import streamlit as st
import pandas as pd
import numpy as np
import requests
from collections import defaultdict, deque
from datetime import date
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

st.set_page_config(page_title="Football Predictor V5", page_icon="⚽", layout="wide")
st.title("⚽ Football Predictor V5")
st.caption("Reliable raw-GitHub fixtures + historical results. No API key. Prediction mode remains available even when bookmaker odds are unavailable.")

RAW="https://raw.githubusercontent.com/openfootball/football.json/master"
LEAGUES={
    "Premier League":"en.1",
    "Championship":"en.2",
    "Bundesliga":"de.1",
    "La Liga":"es.1",
    "Serie A":"it.1",
    "Ligue 1":"fr.1",
}
# Only leagues actually present in OpenFootball's 2026/27 JSON repository are exposed.
SEASONS=["2018-19","2019-20","2020-21","2021-22","2022-23","2023-24","2024-25","2025-26","2026-27"]
FEATURES=["h_pts","a_pts","h_gf","a_gf","h_ga","a_ga","elo_diff","elo_home"]

HEADERS={"User-Agent":"Mozilla/5.0 FootballPredictorV5/1.0","Accept":"application/json"}

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
def train(code):
    f,hist,elo,used=make_training(code)
    if len(f)<120: raise RuntimeError(f"Only {len(f)} completed historical matches available.")
    X=f[FEATURES].fillna(0); y=f["y"]
    cut=int(len(X)*.8)
    base=HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,l2_regularization=2,random_state=42)
    base.fit(X.iloc[:cut],y.iloc[:cut])
    model=base
    if len(X)-cut>=50 and y.iloc[cut:].nunique()==3:
        try:
            cal=CalibratedClassifierCV(base,method="sigmoid",cv="prefit")
            cal.fit(X.iloc[cut:],y.iloc[cut:])
            model=cal
        except Exception: pass
    return model,hist,elo,used

@st.cache_data(ttl=1800,show_spinner=False)
def fixtures_for(code):
    return season_json("2026-27",code)["matches"]

scope=st.selectbox("Competition",["ALL SUPPORTED LEAGUES"]+list(LEAGUES))
day=st.date_input("Match date",date.today())
min_conf=st.slider("Minimum prediction confidence (%)",45,90,62)
topn=st.slider("Show top predictions",3,20,10)

st.info("V5 uses OpenFootball raw JSON for fixtures and historical results. It does NOT invent bookmaker odds. Until a verified current-odds feed is connected, selections are PREDICTION/PASS only — never BET.")

if st.button("🔎 ANALYZE MATCHES",use_container_width=True):
    selected=LEAGUES if scope=="ALL SUPPORTED LEAGUES" else {scope:LEAGUES[scope]}
    out=[]; warnings=[]
    with st.spinner("Loading verified fixtures and training league models..."):
        for lname,code in selected.items():
            try:
                matches=fixtures_for(code)
            except Exception as e:
                warnings.append(f"{lname}: fixture feed unavailable ({e})")
                continue
            games=[m for m in matches if str(m.get("date",""))[:10]==day.isoformat()]
            if not games: continue
            try:
                model,hist,elo,ntrain=train(code)
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
                out.append({"League":lname,"Match":f"{h} v {a}","Pick":labels[i],
                            "Home %":round(pr[0]*100,1),"Draw %":round(pr[1]*100,1),
                            "Away %":round(pr[2]*100,1),"Confidence %":round(conf*100,1),
                            "Fair odds":round(1/conf,2),
                            "Market odds":None,"Edge pp":None,"EV %":None,
                            "Decision":"PREDICTION" if conf>=min_conf/100 else "PASS",
                            "Training matches":ntrain})
    if warnings:
        with st.expander("Data/model warnings"):
            for w in warnings: st.warning(w)
    if not out:
        st.info("No supported OpenFootball fixtures were found for that date.")
        st.stop()
    d=pd.DataFrame(out).sort_values("Confidence %",ascending=False)
    st.subheader("🏆 Strongest model predictions")
    q=d[d.Decision=="PREDICTION"].head(topn)
    if q.empty: st.info("No fixture clears the confidence threshold.")
    else: st.dataframe(q,hide_index=True,use_container_width=True)
    st.subheader("All analysed matches")
    st.dataframe(d,hide_index=True,use_container_width=True)
    st.warning("BET labels are disabled in V5 because no verified current bookmaker-odds feed is connected. This prevents false market edges or EV figures.")

st.divider()
st.caption("V5 fail-closed rule: predictions may run from verified fixture/history data; BET requires a separate verified current-odds source.")
