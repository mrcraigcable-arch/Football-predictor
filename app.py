
import streamlit as st
import pandas as pd
import numpy as np
import requests, io
from collections import defaultdict, deque
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Football Predictor V2", page_icon="⚽", layout="wide")
st.title("⚽ Football Predictor V2")
st.caption("Multi-league 1X2 probability + market-value screening. Probabilities, not guaranteed winners.")

LEAGUES = {
    "Premier League": "E0",
    "Championship": "E1",
    "League One": "E2",
    "League Two": "E3",
    "La Liga": "SP1",
    "Bundesliga": "D1",
    "Serie A": "I1",
    "Ligue 1": "F1",
}
SEASONS = ["1617","1718","1819","1920","2021","2122","2223","2324","2425","2526","2627"]
BASE = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
FIXTURE_URL = "https://www.football-data.co.uk/matches/resources/fixtures.csv"

def read_csv_bytes(raw):
    if b"<html" in raw[:1000].lower() or b"<!doctype" in raw[:1000].lower():
        raise ValueError("Source returned HTML instead of CSV")
    return pd.read_csv(io.BytesIO(raw), engine="python", on_bad_lines="skip", encoding_errors="replace")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_csv(url):
    r = requests.get(url, timeout=30, headers={"User-Agent":"Mozilla/5.0 FootballPredictorV2/2.0"})
    r.raise_for_status()
    return read_csv_bytes(r.content)

def odds_cols(df):
    # Prefer Football-Data market-average 1X2 prices. Never fabricate missing odds.
    sets = [("AvgH","AvgD","AvgA"), ("B365H","B365D","B365A"), ("MaxH","MaxD","MaxA")]
    for s in sets:
        if all(c in df.columns for c in s):
            return s
    return None

def devig(h,d,a):
    vals=np.array([h,d,a],dtype=float)
    if np.any(~np.isfinite(vals)) or np.any(vals<=1):
        return [np.nan]*3
    inv=1/vals
    return inv/inv.sum()

def elo_prob(diff):
    return 1/(1+10**(-diff/400))

def rolling_features(df, window=8):
    df=df.copy()
    df["Date"]=pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df=df.dropna(subset=["Date","HomeTeam","AwayTeam","FTR"]).sort_values("Date")
    hist=defaultdict(lambda: deque(maxlen=window))
    elo=defaultdict(lambda:1500.0)
    rows=[]
    def avg(team,key,default=0.0):
        z=hist[team]
        return float(np.mean([x[key] for x in z])) if z else default
    for _,r in df.iterrows():
        h,a=r.HomeTeam,r.AwayTeam
        eh,ea=elo[h],elo[a]
        rec={
            "Date":r.Date,"HomeTeam":h,"AwayTeam":a,"FTR":r.FTR,
            "h_pts":avg(h,"pts"),"a_pts":avg(a,"pts"),
            "h_gf":avg(h,"gf"),"a_gf":avg(a,"gf"),
            "h_ga":avg(h,"ga"),"a_ga":avg(a,"ga"),
            "h_sh":avg(h,"sh"),"a_sh":avg(a,"sh"),
            "h_sot":avg(h,"sot"),"a_sot":avg(a,"sot"),
            "elo_diff":eh-ea,"elo_home":elo_prob((eh+55)-ea),
        }
        oc=odds_cols(df)
        if oc:
            p=devig(r.get(oc[0],np.nan),r.get(oc[1],np.nan),r.get(oc[2],np.nan))
        else: p=[np.nan]*3
        rec.update({"mkt_h":p[0],"mkt_d":p[1],"mkt_a":p[2]})
        rows.append(rec)
        hg,ag=float(r.get("FTHG",0)),float(r.get("FTAG",0))
        hs,ass=float(r.get("HS",0) or 0),float(r.get("AS",0) or 0)
        hst,ast=float(r.get("HST",0) or 0),float(r.get("AST",0) or 0)
        if r.FTR=="H": hp,ap,score=3,0,1
        elif r.FTR=="A": hp,ap,score=0,3,0
        else: hp,ap,score=1,1,.5
        hist[h].append({"pts":hp,"gf":hg,"ga":ag,"sh":hs,"sot":hst})
        hist[a].append({"pts":ap,"gf":ag,"ga":hg,"sh":ass,"sot":ast})
        exp=elo_prob(eh-ea); k=24
        elo[h]+=k*(score-exp); elo[a]+=k*((1-score)-(1-exp))
    return pd.DataFrame(rows), hist, elo

FEATURES=["h_pts","a_pts","h_gf","a_gf","h_ga","a_ga","h_sh","a_sh","h_sot","a_sot","elo_diff","elo_home","mkt_h","mkt_d","mkt_a"]

@st.cache_resource(show_spinner=False)
def train_league(div):
    parts=[]
    for s in SEASONS:
        try:
            d=fetch_csv(BASE.format(season=s,div=div))
            if {"Date","HomeTeam","AwayTeam","FTR"}.issubset(d.columns):
                parts.append(d)
        except Exception:
            pass
    if not parts:
        raise RuntimeError(f"No historical data available for {div}")
    raw=pd.concat(parts,ignore_index=True)
    feat,hist,elo=rolling_features(raw)
    feat=feat[feat.FTR.isin(["H","D","A"])].copy()
    # Market columns are useful but optional in training. Median imputation prevents fake current edge.
    X=feat[FEATURES].replace([np.inf,-np.inf],np.nan)
    X=X.fillna(X.median(numeric_only=True)).fillna(0)
    y=feat.FTR.map({"H":0,"D":1,"A":2})
    # Chronological holdout calibration.
    cut=max(100,int(len(X)*0.8))
    Xtr,Xcal=X.iloc[:cut],X.iloc[cut:]
    ytr,ycal=y.iloc[:cut],y.iloc[cut:]
    base=HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,l2_regularization=2.0,random_state=42)
    base.fit(Xtr,ytr)
    if len(Xcal)>=50 and ycal.nunique()==3:
        cal=CalibratedClassifierCV(base,method="sigmoid",cv="prefit")
        cal.fit(Xcal,ycal)
        model=cal
    else:
        model=base
    return model,hist,elo,X.median(numeric_only=True).to_dict(),len(feat)

def team_avg(hist,team,key):
    z=hist[team]
    return float(np.mean([x[key] for x in z])) if z else 0.0

def fixture_vector(row,hist,elo,med):
    h,a=row.HomeTeam,row.AwayTeam
    oc=None
    for s in [("AvgH","AvgD","AvgA"),("B365H","B365D","B365A"),("MaxH","MaxD","MaxA")]:
        if all(c in row.index for c in s):
            oc=s; break
    p=[np.nan]*3
    actual=[np.nan]*3
    if oc:
        actual=[pd.to_numeric(row.get(c),errors="coerce") for c in oc]
        p=devig(*actual)
    vals={
        "h_pts":team_avg(hist,h,"pts"),"a_pts":team_avg(hist,a,"pts"),
        "h_gf":team_avg(hist,h,"gf"),"a_gf":team_avg(hist,a,"gf"),
        "h_ga":team_avg(hist,h,"ga"),"a_ga":team_avg(hist,a,"ga"),
        "h_sh":team_avg(hist,h,"sh"),"a_sh":team_avg(hist,a,"sh"),
        "h_sot":team_avg(hist,h,"sot"),"a_sot":team_avg(hist,a,"sot"),
        "elo_diff":elo[h]-elo[a],"elo_home":elo_prob((elo[h]+55)-elo[a]),
        "mkt_h":p[0],"mkt_d":p[1],"mkt_a":p[2],
    }
    # Model can still produce a probability without current odds, but VALUE/BET is prohibited.
    x=pd.DataFrame([[vals[f] if np.isfinite(vals[f]) else med.get(f,0) for f in FEATURES]],columns=FEATURES)
    return x,p,actual

@st.cache_data(ttl=900,show_spinner=False)
def current_fixture_feed():
    df=fetch_csv(FIXTURE_URL)
    if not {"Div","HomeTeam","AwayTeam"}.issubset(df.columns):
        raise RuntimeError("Fixture feed does not contain expected Div/HomeTeam/AwayTeam columns.")
    return df

scope=st.selectbox("Competition",["ALL SUPPORTED LEAGUES"]+list(LEAGUES.keys()))
min_conf=st.slider("Minimum confidence (%)",45,90,62)
min_edge=st.slider("Minimum market edge (percentage points)",0,20,4)
top_n=st.slider("Show top picks",3,20,10)

if st.button("🔎 ANALYZE CURRENT MATCHES",use_container_width=True):
    try:
        fx=current_fixture_feed()
        wanted=list(LEAGUES.items()) if scope=="ALL SUPPORTED LEAGUES" else [(scope,LEAGUES[scope])]
        results=[]
        with st.spinner("Training/checking league models and loading current fixtures..."):
            for lname,div in wanted:
                f=fx[fx.Div.astype(str)==div].copy()
                if f.empty: continue
                try:
                    model,hist,elo,med,n=train_league(div)
                except Exception as e:
                    st.warning(f"{lname}: model unavailable ({e})")
                    continue
                for _,r in f.iterrows():
                    x,mkt,actual=fixture_vector(r,hist,elo,med)
                    pr=model.predict_proba(x)[0]
                    pick_i=int(np.argmax(pr)); labels=["HOME","DRAW","AWAY"]
                    pick=labels[pick_i]; conf=float(pr[pick_i])
                    market=float(mkt[pick_i]) if np.isfinite(mkt[pick_i]) else np.nan
                    edge=(conf-market) if np.isfinite(market) else np.nan
                    odds=float(actual[pick_i]) if np.isfinite(actual[pick_i]) else np.nan
                    fair=1/conf if conf>0 else np.nan
                    ev=(conf*odds-1) if np.isfinite(odds) else np.nan
                    bet=bool(np.isfinite(edge) and np.isfinite(odds) and conf>=min_conf/100 and edge>=min_edge/100 and ev>0)
                    results.append({
                        "League":lname,"Match":f"{r.HomeTeam} v {r.AwayTeam}",
                        "Home %":round(pr[0]*100,1),"Draw %":round(pr[1]*100,1),"Away %":round(pr[2]*100,1),
                        "Pick":pick,"Confidence %":round(conf*100,1),
                        "Market %":round(market*100,1) if np.isfinite(market) else None,
                        "Odds":round(odds,2) if np.isfinite(odds) else None,
                        "Fair odds":round(fair,2),
                        "Edge pp":round(edge*100,1) if np.isfinite(edge) else None,
                        "EV %":round(ev*100,1) if np.isfinite(ev) else None,
                        "Decision":"BET" if bet else ("NO MARKET ODDS" if not np.isfinite(edge) else "PASS"),
                        "_rank": (conf + max(edge,0) if np.isfinite(edge) else conf*.5)
                    })
        if not results:
            st.warning("No fixtures from the selected supported leagues are present in Football-Data's current fixture feed.")
        else:
            out=pd.DataFrame(results).sort_values(["Decision","_rank"],ascending=[True,False]).drop(columns="_rank")
            bets=out[out.Decision=="BET"].sort_values(["Edge pp","Confidence %"],ascending=False)
            st.subheader("🏆 Today's strongest qualifying picks")
            if bets.empty:
                st.info("No match currently clears both thresholds with genuine market odds. PASS is a valid result.")
            else:
                st.dataframe(bets.head(top_n),use_container_width=True,hide_index=True)
            st.subheader("All analysed fixtures")
            st.dataframe(out.head(max(top_n,50)),use_container_width=True,hide_index=True)
            st.caption("BET requires genuine current market odds, confidence threshold, edge threshold and positive model EV. Missing odds can never create a BET.")
    except Exception as e:
        st.error("Could not complete analysis.")
        st.code(str(e))

st.divider()
st.caption("V2 research model. Supported: EPL, Championship, League One, League Two, La Liga, Bundesliga, Serie A and Ligue 1. Validate on unseen matches before staking real money. Late lineups, injuries and weather are not yet incorporated.")
