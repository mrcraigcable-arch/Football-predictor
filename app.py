
import streamlit as st
import pandas as pd
import numpy as np
import requests, io
from collections import defaultdict, deque
from datetime import date
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

st.set_page_config(page_title="Football Predictor V3 Mobile", page_icon="⚽", layout="wide")
st.title("⚽ Football Predictor V3 Mobile")
st.caption("Multi-league 1X2 probability + live fixture/odds screening. Probabilities are not guarantees.")

API_BASE="https://v3.football.api-sports.io"
LEAGUES={
 "Premier League":{"api":39,"fd":"E0"},
 "Championship":{"api":40,"fd":"E1"},
 "League One":{"api":41,"fd":"E2"},
 "League Two":{"api":42,"fd":"E3"},
 "La Liga":{"api":140,"fd":"SP1"},
 "Bundesliga":{"api":78,"fd":"D1"},
 "Serie A":{"api":135,"fd":"I1"},
 "Ligue 1":{"api":61,"fd":"F1"},
 "Champions League":{"api":2,"fd":None},
}
SEASONS=["1819","1920","2021","2122","2223","2324","2425","2526","2627"]
FEATURES=["h_pts","a_pts","h_gf","a_gf","h_ga","a_ga","h_sh","a_sh","h_sot","a_sot","elo_diff","elo_home","mkt_h","mkt_d","mkt_a"]

def api_key():
    # Prefer the phone-entered session key. Fall back to Streamlit Secrets if later configured.
    session_key = st.session_state.get("api_football_key", "")
    if session_key:
        return session_key
    try:
        return st.secrets["API_FOOTBALL_KEY"]
    except Exception:
        return ""

def api_get(path,params):
    k=api_key()
    if not k: raise RuntimeError("API key not configured. Add API_FOOTBALL_KEY to Streamlit Secrets.")
    r=requests.get(API_BASE+path,headers={"x-apisports-key":k},params=params,timeout=30)
    r.raise_for_status()
    j=r.json()
    if j.get("errors"): raise RuntimeError(str(j["errors"]))
    return j.get("response",[])

@st.cache_data(ttl=1800,show_spinner=False)
def fixtures_for_day(day, league_id):
    return api_get("/fixtures",{"date":str(day),"league":league_id,"season":day.year if day.month>=7 else day.year-1})

@st.cache_data(ttl=1800,show_spinner=False)
def odds_for_fixture(fid):
    return api_get("/odds",{"fixture":fid})

def get_1x2(resp):
    # Aggregate available bookmaker Match Winner odds using median prices.
    H=[];D=[];A=[]
    for block in resp:
        for bm in block.get("bookmakers",[]):
            for bet in bm.get("bets",[]):
                if bet.get("name") in ("Match Winner","1x2","Winner"):
                    for v in bet.get("values",[]):
                        name=str(v.get("value","")).lower()
                        try: price=float(v.get("odd"))
                        except: continue
                        if name in ("home","1"): H.append(price)
                        elif name in ("draw","x"): D.append(price)
                        elif name in ("away","2"): A.append(price)
    if not (H and D and A): return None
    odds=np.array([np.median(H),np.median(D),np.median(A)],float)
    inv=1/odds; probs=inv/inv.sum()
    return odds,probs

def fd_csv(season,div):
    u=f"https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
    r=requests.get(u,headers={"User-Agent":"Mozilla/5.0"},timeout=30); r.raise_for_status()
    if b"<html" in r.content[:800].lower(): raise ValueError("HTML response")
    return pd.read_csv(io.BytesIO(r.content),engine="python",on_bad_lines="skip",encoding_errors="replace")

def elo_p(d): return 1/(1+10**(-d/400))

def history_features(df,w=8):
    df=df.copy(); df["Date"]=pd.to_datetime(df["Date"],dayfirst=True,errors="coerce")
    df=df.dropna(subset=["Date","HomeTeam","AwayTeam","FTR"]).sort_values("Date")
    hist=defaultdict(lambda:deque(maxlen=w)); elo=defaultdict(lambda:1500.0); rows=[]
    def av(t,k): return float(np.mean([x[k] for x in hist[t]])) if hist[t] else 0.
    for _,r in df.iterrows():
        h,a=r.HomeTeam,r.AwayTeam; eh,ea=elo[h],elo[a]
        vals={"h_pts":av(h,"pts"),"a_pts":av(a,"pts"),"h_gf":av(h,"gf"),"a_gf":av(a,"gf"),
              "h_ga":av(h,"ga"),"a_ga":av(a,"ga"),"h_sh":av(h,"sh"),"a_sh":av(a,"sh"),
              "h_sot":av(h,"sot"),"a_sot":av(a,"sot"),"elo_diff":eh-ea,"elo_home":elo_p(eh+55-ea)}
        oc=next((s for s in [("AvgH","AvgD","AvgA"),("B365H","B365D","B365A")] if all(c in df.columns for c in s)),None)
        mp=[np.nan]*3
        if oc:
            try:
                o=np.array([float(r[c]) for c in oc]); inv=1/o; mp=inv/inv.sum()
            except: pass
        vals.update({"mkt_h":mp[0],"mkt_d":mp[1],"mkt_a":mp[2]})
        vals["FTR"]=r.FTR; rows.append(vals)
        hg=float(r.get("FTHG",0) or 0); ag=float(r.get("FTAG",0) or 0)
        hs=float(r.get("HS",0) or 0); ass=float(r.get("AS",0) or 0)
        hst=float(r.get("HST",0) or 0); ast=float(r.get("AST",0) or 0)
        if r.FTR=="H": hp,ap,s=3,0,1
        elif r.FTR=="A": hp,ap,s=0,3,0
        else: hp,ap,s=1,1,.5
        hist[h].append({"pts":hp,"gf":hg,"ga":ag,"sh":hs,"sot":hst})
        hist[a].append({"pts":ap,"gf":ag,"ga":hg,"sh":ass,"sot":ast})
        ex=elo_p(eh-ea); elo[h]+=24*(s-ex); elo[a]+=24*((1-s)-(1-ex))
    return pd.DataFrame(rows),hist,elo

@st.cache_resource(show_spinner=False)
def train(div):
    parts=[]
    for s in SEASONS:
        try: parts.append(fd_csv(s,div))
        except: pass
    if not parts: raise RuntimeError("Historical training feed unavailable")
    f,hist,elo=history_features(pd.concat(parts,ignore_index=True))
    f=f[f.FTR.isin(["H","D","A"])]
    X=f[FEATURES].replace([np.inf,-np.inf],np.nan)
    med=X.median(numeric_only=True); X=X.fillna(med).fillna(0)
    y=f.FTR.map({"H":0,"D":1,"A":2})
    cut=int(len(X)*.8)
    base=HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,l2_regularization=2,random_state=42)
    base.fit(X.iloc[:cut],y.iloc[:cut])
    if len(X)-cut>=50 and y.iloc[cut:].nunique()==3:
        m=CalibratedClassifierCV(base,method="sigmoid",cv="prefit"); m.fit(X.iloc[cut:],y.iloc[cut:])
    else: m=base
    return m,hist,elo,med.to_dict()

def norm(s): return ''.join(c.lower() for c in s if c.isalnum())
def find_team(name, pool):
    n=norm(name)
    exact=[p for p in pool if norm(p)==n]
    if exact:return exact[0]
    # common API vs Football-Data naming tolerance
    toks=set(name.lower().replace("fc","").split())
    cand=sorted(pool,key=lambda p:len(toks & set(p.lower().replace("fc","").split())),reverse=True)
    return cand[0] if cand and len(toks & set(cand[0].lower().replace("fc","").split())) else None

st.subheader("🔐 API connection")
entered_key = st.text_input(
    "API-Football key",
    value=st.session_state.get("api_football_key", ""),
    type="password",
    placeholder="Paste your API-Football key here",
    help="The key is used for this app session and is not written into the public GitHub repository."
)
c1, c2 = st.columns(2)
with c1:
    if st.button("Connect API", use_container_width=True):
        if entered_key.strip():
            st.session_state["api_football_key"] = entered_key.strip()
            st.success("API key loaded for this session.")
        else:
            st.warning("Paste your API key first.")
with c2:
    if st.button("Clear key", use_container_width=True):
        st.session_state.pop("api_football_key", None)
        st.rerun()

scope=st.selectbox("Competition",["ALL SUPPORTED LEAGUES"]+list(LEAGUES))
day=st.date_input("Match date",date.today())
min_conf=st.slider("Minimum confidence (%)",45,90,62)
min_edge=st.slider("Minimum market edge (percentage points)",0,20,4)
topn=st.slider("Show top picks",3,20,10)

if not api_key():
    st.warning("Paste your API-Football key above and tap Connect API before analysing matches.")
else:
    st.success("API connection key is loaded.")

if st.button("🔎 ANALYZE MATCHES",use_container_width=True):
    if not api_key():
        st.error("Connect your API-Football key first.")
        st.stop()
    selected=list(LEAGUES.items()) if scope=="ALL SUPPORTED LEAGUES" else [(scope,LEAGUES[scope])]
    out=[]
    with st.spinner("Loading fixtures, models and bookmaker odds..."):
      for lname,cfg in selected:
        try: games=fixtures_for_day(day,cfg["api"])
        except Exception as e:
            st.warning(f"{lname}: fixtures unavailable ({e})"); continue
        if not games: continue
        if not cfg["fd"]:
            # No matching Football-Data training division in this build.
            continue
        try: model,hist,elo,med=train(cfg["fd"])
        except Exception as e:
            st.warning(f"{lname}: model unavailable ({e})"); continue
        pool=list(set(list(hist.keys())+list(elo.keys())))
        for g in games:
            home=g["teams"]["home"]["name"]; away=g["teams"]["away"]["name"]; fid=g["fixture"]["id"]
            h=find_team(home,pool); a=find_team(away,pool)
            if not h or not a: continue
            try: oraw=odds_for_fixture(fid); op=get_1x2(oraw)
            except: op=None
            if op: odds,mp=op
            else: odds=np.array([np.nan]*3); mp=np.array([np.nan]*3)
            def av(t,k): return float(np.mean([x[k] for x in hist[t]])) if hist[t] else 0.
            vals={"h_pts":av(h,"pts"),"a_pts":av(a,"pts"),"h_gf":av(h,"gf"),"a_gf":av(a,"gf"),
                  "h_ga":av(h,"ga"),"a_ga":av(a,"ga"),"h_sh":av(h,"sh"),"a_sh":av(a,"sh"),
                  "h_sot":av(h,"sot"),"a_sot":av(a,"sot"),"elo_diff":elo[h]-elo[a],
                  "elo_home":elo_p(elo[h]+55-elo[a]),"mkt_h":mp[0],"mkt_d":mp[1],"mkt_a":mp[2]}
            x=pd.DataFrame([[vals[f] if np.isfinite(vals[f]) else med.get(f,0) for f in FEATURES]],columns=FEATURES)
            pr=model.predict_proba(x)[0]; i=int(np.argmax(pr)); labels=["HOME","DRAW","AWAY"]
            conf=float(pr[i]); market=float(mp[i]) if np.isfinite(mp[i]) else np.nan
            edge=conf-market if np.isfinite(market) else np.nan
            odd=float(odds[i]) if np.isfinite(odds[i]) else np.nan
            ev=conf*odd-1 if np.isfinite(odd) else np.nan
            bet=np.isfinite(edge) and np.isfinite(ev) and conf>=min_conf/100 and edge>=min_edge/100 and ev>0
            out.append({"League":lname,"Match":f"{home} v {away}","Pick":labels[i],
                        "Home %":round(pr[0]*100,1),"Draw %":round(pr[1]*100,1),"Away %":round(pr[2]*100,1),
                        "Confidence %":round(conf*100,1),"Market %":round(market*100,1) if np.isfinite(market) else None,
                        "Odds":round(odd,2) if np.isfinite(odd) else None,"Fair odds":round(1/conf,2),
                        "Edge pp":round(edge*100,1) if np.isfinite(edge) else None,
                        "EV %":round(ev*100,1) if np.isfinite(ev) else None,
                        "Decision":"BET" if bet else ("NO ODDS" if not np.isfinite(edge) else "PASS")})
    if not out: st.info("No analysable matches found for the selected date/leagues.")
    else:
        d=pd.DataFrame(out)
        bets=d[d.Decision=="BET"].sort_values(["Edge pp","Confidence %"],ascending=False)
        st.subheader("🏆 Strongest qualifying picks")
        if bets.empty: st.info("No match clears the thresholds with genuine bookmaker odds.")
        else: st.dataframe(bets.head(topn),hide_index=True,use_container_width=True)
        st.subheader("All analysed matches")
        st.dataframe(d.sort_values("Confidence %",ascending=False),hide_index=True,use_container_width=True)

st.divider()
st.caption("V3 Mobile uses API-Football for current fixtures/odds and Football-Data historical league data for model training. Missing odds never create a BET. Your manually entered API key is kept in the current Streamlit session and is not written to GitHub.")
