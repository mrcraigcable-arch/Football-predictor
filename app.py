import streamlit as st
import pandas as pd
import numpy as np
import requests
from collections import defaultdict, deque
from datetime import date
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

st.set_page_config(page_title="Football Predictor V6", page_icon="⚽", layout="wide")
st.title("⚽ Football Predictor V6")
st.caption("Raw-GitHub fixtures + historical model + current UK bookmaker consensus odds. Market edge and EV are only calculated from verified 1X2 prices.")

RAW="https://raw.githubusercontent.com/openfootball/football.json/master"
LEAGUES={
    "Premier League":{"of":"en.1","odds":"soccer_epl"},
    "Championship":{"of":"en.2","odds":"soccer_efl_champ"},
    "Bundesliga":{"of":"de.1","odds":"soccer_germany_bundesliga"},
    "La Liga":{"of":"es.1","odds":"soccer_spain_la_liga"},
    "Serie A":{"of":"it.1","odds":"soccer_italy_serie_a"},
    "Ligue 1":{"of":"fr.1","odds":"soccer_france_ligue_one"},
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


ODDS_BASE="https://api.the-odds-api.com/v4/sports"

def norm(s):
    import re, unicodedata
    s=unicodedata.normalize("NFKD",str(s)).encode("ascii","ignore").decode().lower()
    s=re.sub(r"\b(fc|cf|afc|ac|calcio|club|de|madrid)\b"," ",s)
    return re.sub(r"[^a-z0-9]","",s)

def team_match(a,b):
    x,y=norm(a),norm(b)
    return x==y or (len(x)>=5 and x in y) or (len(y)>=5 and y in x)

def odds_fetch(api_key,sport_key):
    url=f"{ODDS_BASE}/{sport_key}/odds/"
    r=requests.get(url,params={"apiKey":api_key,"regions":"uk","markets":"h2h",
                              "oddsFormat":"decimal","dateFormat":"iso"},timeout=25)
    if r.status_code in (401,403):
        raise RuntimeError("Odds API key was rejected.")
    if r.status_code==429:
        raise RuntimeError("Odds API monthly request allowance has been reached.")
    r.raise_for_status()
    return r.json(), r.headers.get("x-requests-remaining"), r.headers.get("x-requests-used")

def consensus_for(events,home,away):
    event=None
    for e in events:
        if team_match(home,e.get("home_team","")) and team_match(away,e.get("away_team","")):
            event=e; break
    if not event: return None
    books=[]
    for b in event.get("bookmakers",[]):
        for m in b.get("markets",[]):
            if m.get("key")!="h2h": continue
            vals={}
            for o in m.get("outcomes",[]):
                n=o.get("name",""); p=o.get("price")
                if not isinstance(p,(int,float)) or p<=1: continue
                if n=="Draw": vals["D"]=float(p)
                elif team_match(home,n): vals["H"]=float(p)
                elif team_match(away,n): vals["A"]=float(p)
            if all(k in vals for k in ("H","D","A")):
                books.append((vals["H"],vals["D"],vals["A"],b.get("title","")))
    if not books: return None
    arr=np.array([x[:3] for x in books],dtype=float)
    med=np.median(arr,axis=0)
    inv=1/med; fair=inv/inv.sum()
    return med,fair,len(books)

st.subheader("🔐 Current odds connection")
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

scope=st.selectbox("Competition",["ALL SUPPORTED LEAGUES"]+list(LEAGUES))
day=st.date_input("Match date",date.today())
min_conf=st.slider("Minimum prediction confidence (%)",45,90,62)
min_edge=st.slider("Minimum market edge (percentage points)",0,20,4)
topn=st.slider("Show top predictions",3,20,10)

st.info("V6 uses OpenFootball for fixtures/history and The Odds API for current UK 1X2 prices. Without a connected odds key, V6 stays in prediction-only mode and cannot issue BET.")

if st.button("🔎 ANALYZE MATCHES",use_container_width=True):
    selected=LEAGUES if scope=="ALL SUPPORTED LEAGUES" else {scope:LEAGUES[scope]}
    odds_key=st.session_state.get("odds_key","").strip()
    quota_remaining=None
    out=[]; warnings=[]
    with st.spinner("Loading verified fixtures and training league models..."):
        for lname,meta in selected.items():
            code=meta["of"]
            odds_events=[]
            if odds_key:
                try:
                    odds_events,rem,used=odds_fetch(odds_key,meta["odds"])
                    if rem is not None: quota_remaining=rem
                except Exception as e:
                    warnings.append(f"{lname}: current odds unavailable ({e})")
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
                market=consensus_for(odds_events,h,a) if odds_events else None
                odd=np.nan; mprob=np.nan; edge=np.nan; ev=np.nan; books=0
                if market:
                    med,mfair,books=market
                    odd=float(med[i]); mprob=float(mfair[i])
                    edge=conf-mprob
                    ev=conf*odd-1
                if market and conf>=min_conf/100 and edge>=min_edge/100 and ev>0:
                    decision="BET"
                elif not market:
                    decision="PREDICTION" if conf>=min_conf/100 else "PASS"
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
                            "Decision":decision,"Training matches":ntrain})
    if warnings:
        with st.expander("Data/model warnings"):
            for w in warnings: st.warning(w)
    if not out:
        st.info("No supported OpenFootball fixtures were found for that date.")
        st.stop()
    d=pd.DataFrame(out).sort_values("Confidence %",ascending=False)
    st.subheader("🏆 Strongest qualifying selections")
    bets=d[d.Decision=="BET"].sort_values(["Edge pp","Confidence %"],ascending=False).head(topn)
    if len(bets):
        st.dataframe(bets,hide_index=True,use_container_width=True)
    elif st.session_state.get("odds_key","").strip():
        st.info("No fixture clears confidence + market-edge + positive-EV rules. PASS is the correct output.")
    else:
        q=d[d.Decision=="PREDICTION"].head(topn)
        if q.empty: st.info("No fixture clears the confidence threshold.")
        else: st.dataframe(q,hide_index=True,use_container_width=True)
    st.subheader("All analysed matches")
    st.dataframe(d,hide_index=True,use_container_width=True)
    if quota_remaining is not None:
        st.caption(f"Odds API credits remaining (provider header): {quota_remaining}")
    if not st.session_state.get("odds_key","").strip():
        st.warning("No current-odds key is connected, so BET labels, market edge and EV remain disabled.")

st.divider()
st.caption("V6 fail-closed rule: BET requires matched current UK 1X2 bookmaker prices, de-margined market probability, sufficient model confidence, minimum edge and positive EV.")
