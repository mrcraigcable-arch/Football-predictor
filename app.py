import streamlit as st
import pandas as pd
import numpy as np
import requests, io, time
from collections import defaultdict, deque
from datetime import date
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

st.set_page_config(page_title="Football Predictor V4", page_icon="⚽", layout="wide")
st.title("⚽ Football Predictor V4")
st.caption("2026/27 multi-league 1X2 screening. Historical model + current Football-Data fixtures/market odds. Probabilities are not guarantees.")

FIXTURES_URL = "https://www.football-data.co.uk/matches/resources/fixtures.csv"
LEAGUES = {
    "Premier League":"E0",
    "Championship":"E1",
    "League One":"E2",
    "League Two":"E3",
    "La Liga":"SP1",
    "Bundesliga":"D1",
    "Serie A":"I1",
    "Ligue 1":"F1",
}
SEASONS=["1819","1920","2021","2122","2223","2324","2425","2526","2627"]
FEATURES=["h_pts","a_pts","h_gf","a_gf","h_ga","a_ga","h_sh","a_sh","h_sot","a_sot",
          "elo_diff","elo_home","mkt_h","mkt_d","mkt_a"]

HEADERS = {
    "User-Agent":"Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Safari/604.1",
    "Accept":"text/csv,text/plain,*/*",
    "Cache-Control":"no-cache",
}

def fetch_csv(url, attempts=3):
    last=None
    for n in range(attempts):
        try:
            r=requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 429:
                last=RuntimeError("Football-Data rate-limited the request (HTTP 429).")
                time.sleep(1.5*(n+1)); continue
            r.raise_for_status()
            head=r.content[:1000].lower()
            ctype=(r.headers.get("content-type") or "").lower()
            if b"<html" in head or b"<!doctype" in head:
                raise ValueError("Source returned HTML instead of CSV.")
            df=pd.read_csv(io.BytesIO(r.content), engine="python", on_bad_lines="skip",
                           encoding_errors="replace")
            if len(df.columns)<4:
                raise ValueError("Downloaded file is not a valid fixture CSV.")
            return df
        except Exception as e:
            last=e
            if n < attempts-1: time.sleep(1.2*(n+1))
    raise last

@st.cache_data(ttl=900, show_spinner=False)
def current_fixtures():
    df=fetch_csv(FIXTURES_URL)
    needed={"Div","Date","HomeTeam","AwayTeam"}
    if not needed.issubset(df.columns):
        raise ValueError("Fixture feed is missing required columns: " + ", ".join(sorted(needed-set(df.columns))))
    df["DateParsed"]=pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.date
    return df

@st.cache_data(ttl=86400, show_spinner=False)
def fd_history(season, div):
    return fetch_csv(f"https://www.football-data.co.uk/mmz4281/{season}/{div}.csv", attempts=2)

def elo_p(d): return 1/(1+10**(-d/400))

def market_columns(columns):
    # Prefer market averages. Pinnacle is intentionally not used as a preferred source.
    for trio in [("AvgH","AvgD","AvgA"),("B365H","B365D","B365A"),
                 ("MaxH","MaxD","MaxA"),("WHH","WHD","WHA")]:
        if all(c in columns for c in trio):
            return trio
    return None

def odds_and_market(row, columns):
    trio=market_columns(columns)
    if not trio: return None
    try:
        odds=np.array([float(row[c]) for c in trio], dtype=float)
        if not np.all(np.isfinite(odds)) or np.any(odds<=1):
            return None
        inv=1/odds
        probs=inv/inv.sum()
        return odds, probs, trio
    except Exception:
        return None

def history_features(df,w=8):
    df=df.copy()
    df["Date"]=pd.to_datetime(df["Date"],dayfirst=True,errors="coerce")
    df=df.dropna(subset=["Date","HomeTeam","AwayTeam","FTR"]).sort_values("Date")
    hist=defaultdict(lambda:deque(maxlen=w))
    elo=defaultdict(lambda:1500.0)
    rows=[]
    oc=market_columns(df.columns)
    def av(t,k): return float(np.mean([x[k] for x in hist[t]])) if hist[t] else 0.
    for _,r in df.iterrows():
        h,a=r.HomeTeam,r.AwayTeam; eh,ea=elo[h],elo[a]
        vals={"h_pts":av(h,"pts"),"a_pts":av(a,"pts"),"h_gf":av(h,"gf"),"a_gf":av(a,"gf"),
              "h_ga":av(h,"ga"),"a_ga":av(a,"ga"),"h_sh":av(h,"sh"),"a_sh":av(a,"sh"),
              "h_sot":av(h,"sot"),"a_sot":av(a,"sot"),"elo_diff":eh-ea,"elo_home":elo_p(eh+55-ea)}
        mp=[np.nan]*3
        if oc:
            try:
                o=np.array([float(r[c]) for c in oc],dtype=float)
                if np.all(np.isfinite(o)) and np.all(o>1):
                    inv=1/o; mp=inv/inv.sum()
            except Exception: pass
        vals.update({"mkt_h":mp[0],"mkt_d":mp[1],"mkt_a":mp[2],"FTR":r.FTR})
        rows.append(vals)
        def num(k):
            try:
                v=float(r.get(k,0))
                return v if np.isfinite(v) else 0.
            except Exception: return 0.
        hg,ag,hs,ass,hst,ast=[num(k) for k in ("FTHG","FTAG","HS","AS","HST","AST")]
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
        try:
            d=fd_history(s,div)
            if {"Date","HomeTeam","AwayTeam","FTR"}.issubset(d.columns):
                parts.append(d)
        except Exception:
            pass
    if not parts:
        raise RuntimeError("Historical Football-Data training files could not be downloaded.")
    f,hist,elo=history_features(pd.concat(parts,ignore_index=True))
    f=f[f.FTR.isin(["H","D","A"])]
    if len(f)<150:
        raise RuntimeError(f"Only {len(f)} usable historical matches were available.")
    X=f[FEATURES].replace([np.inf,-np.inf],np.nan)
    med=X.median(numeric_only=True)
    X=X.fillna(med).fillna(0)
    y=f.FTR.map({"H":0,"D":1,"A":2})
    cut=max(1,int(len(X)*.8))
    base=HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,
                                        l2_regularization=2,random_state=42)
    base.fit(X.iloc[:cut],y.iloc[:cut])
    if len(X)-cut>=50 and y.iloc[cut:].nunique()==3:
        try:
            m=CalibratedClassifierCV(base,method="sigmoid",cv="prefit")
            m.fit(X.iloc[cut:],y.iloc[cut:])
        except Exception:
            m=base
    else:
        m=base
    return m,hist,elo,med.to_dict(),len(f)

scope=st.selectbox("Competition",["ALL SUPPORTED LEAGUES"]+list(LEAGUES))
day=st.date_input("Match date",date.today())
min_conf=st.slider("Minimum confidence (%)",45,90,62)
min_edge=st.slider("Minimum market edge (percentage points)",0,20,4)
topn=st.slider("Show top picks",3,20,10)

st.info("V4 does not require an API key. A BET can only appear when the fixture feed contains genuine 1X2 market odds.")

if st.button("🔎 ANALYZE MATCHES",use_container_width=True):
    try:
        with st.spinner("Downloading the current fixture/odds file..."):
            fx=current_fixtures()
    except Exception as e:
        st.error("Current fixture feed could not be verified, so V4 stopped rather than guessing.")
        st.code(str(e))
        st.stop()

    selected=LEAGUES if scope=="ALL SUPPORTED LEAGUES" else {scope:LEAGUES[scope]}
    fx=fx[(fx["DateParsed"]==day) & (fx["Div"].astype(str).isin(selected.values()))].copy()

    if fx.empty:
        st.info("The verified Football-Data feed contains no supported fixtures for that date.")
        st.stop()

    out=[]
    model_errors=[]
    with st.spinner("Training league models and analysing verified fixtures..."):
        for lname,div in selected.items():
            games=fx[fx["Div"].astype(str)==div]
            if games.empty: continue
            try:
                model,hist,elo,med,ntrain=train(div)
            except Exception as e:
                model_errors.append(f"{lname}: {e}")
                continue

            def av(t,k): return float(np.mean([x[k] for x in hist[t]])) if hist[t] else 0.
            for _,g in games.iterrows():
                home=str(g["HomeTeam"]); away=str(g["AwayTeam"])
                op=odds_and_market(g, games.columns)
                if op:
                    odds,mp,trio=op
                else:
                    odds=np.array([np.nan]*3); mp=np.array([np.nan]*3); trio=None

                vals={"h_pts":av(home,"pts"),"a_pts":av(away,"pts"),
                      "h_gf":av(home,"gf"),"a_gf":av(away,"gf"),
                      "h_ga":av(home,"ga"),"a_ga":av(away,"ga"),
                      "h_sh":av(home,"sh"),"a_sh":av(away,"sh"),
                      "h_sot":av(home,"sot"),"a_sot":av(away,"sot"),
                      "elo_diff":elo[home]-elo[away],
                      "elo_home":elo_p(elo[home]+55-elo[away]),
                      "mkt_h":mp[0],"mkt_d":mp[1],"mkt_a":mp[2]}
                x=pd.DataFrame([[vals[f] if np.isfinite(vals[f]) else med.get(f,0)
                                 for f in FEATURES]],columns=FEATURES)
                pr=model.predict_proba(x)[0]
                i=int(np.argmax(pr)); labels=["HOME","DRAW","AWAY"]
                conf=float(pr[i])
                market=float(mp[i]) if np.isfinite(mp[i]) else np.nan
                edge=conf-market if np.isfinite(market) else np.nan
                odd=float(odds[i]) if np.isfinite(odds[i]) else np.nan
                ev=conf*odd-1 if np.isfinite(odd) else np.nan
                bet=(np.isfinite(edge) and np.isfinite(ev) and
                     conf>=min_conf/100 and edge>=min_edge/100 and ev>0)
                decision="BET" if bet else ("NO MARKET DATA" if not np.isfinite(edge) else "PASS")
                out.append({
                    "League":lname,"Match":f"{home} v {away}","Pick":labels[i],
                    "Home %":round(pr[0]*100,1),"Draw %":round(pr[1]*100,1),
                    "Away %":round(pr[2]*100,1),"Confidence %":round(conf*100,1),
                    "Market %":round(market*100,1) if np.isfinite(market) else None,
                    "Odds":round(odd,2) if np.isfinite(odd) else None,
                    "Fair odds":round(1/conf,2),
                    "Edge pp":round(edge*100,1) if np.isfinite(edge) else None,
                    "EV %":round(ev*100,1) if np.isfinite(ev) else None,
                    "Decision":decision
                })

    if model_errors:
        with st.expander("League model warnings"):
            for x in model_errors: st.warning(x)

    if not out:
        st.error("Fixtures were found, but none could be analysed with a verified league model.")
        st.stop()

    d=pd.DataFrame(out)
    bets=d[d.Decision=="BET"].sort_values(["Edge pp","Confidence %"],ascending=False)
    st.subheader("🏆 Strongest qualifying picks")
    if bets.empty:
        st.info("No match clears both thresholds with genuine market odds. That is a valid PASS result.")
    else:
        st.dataframe(bets.head(topn),hide_index=True,use_container_width=True)

    st.subheader("All analysed matches")
    st.dataframe(d.sort_values("Confidence %",ascending=False),hide_index=True,use_container_width=True)

st.divider()
st.caption("V4 fail-closed rule: no verified fixture feed or no genuine 1X2 odds = no BET. Football-Data market averages are preferred over Pinnacle.")
