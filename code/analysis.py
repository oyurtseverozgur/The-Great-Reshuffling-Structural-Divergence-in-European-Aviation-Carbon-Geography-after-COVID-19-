#!/usr/bin/env python3
"""Re-analysis pipeline for 'The Great Reshuffling' (revision, Sep 2026).
Keys on STATE_NAME (STATE_CODE collisions LY, LS)."""
import glob, json, numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm

PRICED = {"AUSTRIA","BELGIUM","BULGARIA","CROATIA","CYPRUS","CZECHIA","DENMARK","ESTONIA",
 "FINLAND","FRANCE","GERMANY","GREECE","HUNGARY","IRELAND","ITALY","LATVIA","LITHUANIA",
 "LUXEMBOURG","MALTA","NETHERLANDS","POLAND","PORTUGAL","ROMANIA","SLOVAKIA","SLOVENIA",
 "SPAIN","SWEDEN","CANARY ISLANDS","NORWAY","ICELAND","SWITZERLAND","UNITED KINGDOM"}
NONPRICED = {"TURKIYE","SERBIA","BOSNIA AND HERZEGOVINA","MONTENEGRO","ALBANIA",
 "NORTH MACEDONIA","KOSOVO","MOLDOVA","GEORGIA","ARMENIA"}
MICRO = {"MONACO","LIECHTENSTEIN"}

def load():
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("data/co2_*.csv"))], ignore_index=True)
    df["STATE_NAME"] = (df.STATE_NAME.str.strip().str.upper()
        .str.replace(r"T.RKIYE|T.RK.YE","TURKIYE",regex=True)
        .str.replace(r"MOLDOVA.*","MOLDOVA",regex=True))
    assert not df.duplicated(["YEAR","MONTH","STATE_NAME"]).any()
    df = df[~df.STATE_NAME.isin(MICRO)].copy()
    assert set(df.STATE_NAME) == PRICED | NONPRICED, set(df.STATE_NAME) ^ (PRICED|NONPRICED)
    df["GROUP"] = np.where(df.STATE_NAME.isin(NONPRICED),"Periphery","Core")
    return df

def panel(df, months, years=None):
    d = df[df.MONTH.isin(months)]
    if years is not None: d = d[d.YEAR.isin(years)]
    p = d.groupby(["YEAR","STATE_NAME","GROUP"],as_index=False).agg(
        CO2=("CO2_QTY_TONNES","sum"),TF=("TF","sum"),NM=("MONTH","nunique"))
    p = p[p.NM == len(months)]          # require every month in window
    return p

def balance(p):
    ny = p.YEAR.nunique(); k = p.groupby("STATE_NAME").YEAR.nunique()
    return p[p.STATE_NAME.isin(k[k==ny].index)].copy()

def sigma(p): return p.groupby("YEAR").CO2.apply(lambda x: np.std(np.log(x),ddof=1))

def beta(p,y0,y1,cond=False,drop=None):
    a=p[p.YEAR==y0].set_index("STATE_NAME"); b=p[p.YEAR==y1].set_index("STATE_NAME")
    idx=a.index.intersection(b.index)
    if drop: idx=idx.drop([s for s in drop if s in idx])
    a,b=a.loc[idx],b.loc[idx]
    g=(np.log(b.CO2)-np.log(a.CO2))/(y1-y0)
    X=pd.DataFrame({"lnE0":np.log(a.CO2)})
    if cond:
        X["D_periph"]=(a.GROUP=="Periphery").astype(float)
        X["ln_int0"]=np.log(a.CO2/a.TF)
    m=sm.OLS(g,sm.add_constant(X)).fit()
    mr=sm.OLS(g,sm.add_constant(X)).fit(cov_type="HC3")
    out=dict(window=f"{y0}-{y1}",n=len(g),beta=m.params["lnE0"],se=m.bse["lnE0"],p=m.pvalues["lnE0"],
             se_hc3=mr.bse["lnE0"],p_hc3=mr.pvalues["lnE0"],r2=m.rsquared,adjr2=m.rsquared_adj)
    if cond:
        for v in ["D_periph","ln_int0"]:
            out[v]=m.params[v]; out[v+"_se"]=m.bse[v]; out[v+"_p"]=m.pvalues[v]; out[v+"_p_hc3"]=mr.pvalues[v]
    # implied half-life (years) if beta<0: b = -(1-exp(-lambda T))/T
    T=y1-y0; bb=out["beta"]
    if bb<0 and 1+bb*T>0:
        lam=-np.log(1+bb*T)/T; out["lambda"]=lam; out["half_life"]=np.log(2)/lam
    return out

def theil(sub):
    x=sub.CO2.values; mu=x.mean(); T=np.mean(x/mu*np.log(x/mu)); Tb=Tw=0
    for _,g in sub.groupby("GROUP"):
        xg=g.CO2.values; mg=xg.mean(); sg=xg.sum()/x.sum()
        Tb+=sg*np.log(mg/mu); Tw+=sg*np.mean(xg/mg*np.log(xg/mg))
    return T,Tb,Tw

def chow(p,grp,post_start,drop=(2020,2021)):
    """Level+slope break. Obs in `drop` excluded. post dummy = t>=post_start."""
    s=p[p.GROUP==grp].groupby("YEAR").CO2.sum(); s=s[~s.index.isin(drop)]
    t=s.index.values.astype(float); y=np.log(s.values)
    r=sm.OLS(y,sm.add_constant(t)).fit()
    d=(t>=post_start).astype(float)
    u=sm.OLS(y,sm.add_constant(np.column_stack([t,d,d*(t-post_start)]))).fit()
    k=len(t)-4; F=((r.ssr-u.ssr)/2)/(u.ssr/k)
    return dict(group=grp,n=len(t),df=(2,k),F=F,p=1-stats.f.cdf(F,2,k),
                level_shift=u.params[2],slope_pre=u.params[1],slope_shift=u.params[3])

def growth(p,grp,y0,y1):
    s=p[p.GROUP==grp].groupby("YEAR").CO2.sum(); return (s[y1]/s[y0])**(1/(y1-y0))-1

def summary(p,tag,yend,res):
    g=p.groupby(["YEAR","GROUP"]).CO2.sum().unstack()
    sh=100*g.Periphery/g.sum(axis=1)
    tur=(p[p.STATE_NAME=="TURKIYE"].set_index("YEAR").CO2/g.sum(axis=1)*100).fillna(0)
    r=dict(n_states=p.STATE_NAME.nunique(),
      core_Mt={y:g.loc[y,"Core"]/1e6 for y in [2010,2019,2021,yend]},
      per_Mt={y:g.loc[y,"Periphery"]/1e6 for y in [2010,2019,2021,yend]},
      core_idx2021=100*g.loc[2021,"Core"]/g.loc[2019,"Core"],
      per_idx2021=100*g.loc[2021,"Periphery"]/g.loc[2019,"Periphery"],
      core_idx_end=100*g.loc[yend,"Core"]/g.loc[2019,"Core"],
      per_idx_end=100*g.loc[yend,"Periphery"]/g.loc[2019,"Periphery"],
      share={int(y):round(v,2) for y,v in sh.items()},
      tur_share={y:tur[y] for y in [2010,2019,yend]},
      g_core_pre=growth(p,"Core",2010,2019),g_per_pre=growth(p,"Periphery",2010,2019),
      g_core_post=growth(p,"Core",2019,yend),g_per_post=growth(p,"Periphery",2019,yend),
      g_core_22=growth(p,"Core",2022,yend),g_per_22=growth(p,"Periphery",2022,yend))
    # rank of Turkiye
    rk=p[p.YEAR==yend].sort_values("CO2",ascending=False).STATE_NAME.tolist()
    rk19=p[p.YEAR==2019].sort_values("CO2",ascending=False).STATE_NAME.tolist()
    r["top5_end"]=rk[:5]; r["top5_2019"]=rk19[:5]
    if "TURKIYE" in rk: r["tur_rank_end"]=rk.index("TURKIYE")+1; r["tur_rank_2019"]=rk19.index("TURKIYE")+1
    gi=p.groupby(["YEAR","GROUP"]).apply(lambda x:x.CO2.sum()/x.TF.sum(),include_groups=False).unstack()
    r["intensity"]={int(y):(round(gi.loc[y,"Core"],2),round(gi.loc[y,"Periphery"],2)) for y in gi.index}
    res[tag+"_summary"]=r

def full(p,tag,yend,res,drop=None):
    s=sigma(p); res[tag+"_sigma"]={int(k):round(v,4) for k,v in s.items()}
    res[tag+"_beta"]=[beta(p,2010,2019,drop=drop),beta(p,2019,yend,drop=drop),beta(p,2022,yend,drop=drop)]
    res[tag+"_beta_cond"]=[beta(p,2010,2019,True,drop),beta(p,2019,yend,True,drop),beta(p,2022,yend,True,drop)]
    res[tag+"_theil"]={int(y):dict(zip(["T","Tb","Tw"],theil(p[p.YEAR==y]))) for y in [2010,2015,2019,2021,2022,yend]}
    for k in res[tag+"_theil"]: res[tag+"_theil"][k]["between_pct"]=100*res[tag+"_theil"][k]["Tb"]/res[tag+"_theil"][k]["T"]
    res[tag+"_chow_post2022_drop2020_21"]=[chow(p,g,2022,(2020,2021)) for g in ["Core","Periphery"]]
    res[tag+"_chow_post2022_keep2020"]=[chow(p,g,2022,(2021,)) for g in ["Core","Periphery"]]
    res[tag+"_chow_original_t>=2020"]=[chow(p,g,2020,(2021,)) for g in ["Core","Periphery"]]
    a=p[p.YEAR==2019].set_index("STATE_NAME").CO2; b=p[p.YEAR==yend].set_index("STATE_NAME").CO2
    i=a.index.intersection(b.index); rho,pv=stats.spearmanr(a[i],b[i]); res[tag+"_spearman"]=(rho,pv)
    rho2,_=stats.spearmanr(p[p.YEAR==2010].set_index("STATE_NAME").CO2[i],b[i]); res[tag+"_spearman_2010_end"]=rho2
    ch=((b[i]/a[i]-1)*100).sort_values(); res[tag+"_state_change"]={k:round(v,1) for k,v in ch.items()}

if __name__=="__main__":
    df=load(); res={}
    years=list(range(2010,2027))
    jan=panel(df,[1]); janb=balance(jan)
    res["jan_states_by_year"]=jan.groupby("YEAR").STATE_NAME.nunique().to_dict()
    res["balanced_excluded"]=sorted(set(jan.STATE_NAME)-set(janb.STATE_NAME))
    summary(jan,"jan42",2026,res); summary(janb,"jan39",2026,res)
    full(janb,"jan39",2026,res)
    # robustness: Jan-May window (2026 has 5 months)
    jm=balance(panel(df,[1,2,3,4,5])); summary(jm,"janmay",2026,res); full(jm,"janmay",2026,res)
    # annual full-year 2010-2025
    ann=balance(panel(df,list(range(1,13)),range(2010,2026))); summary(ann,"annual",2025,res); full(ann,"annual",2025,res)
    # excluding Turkiye
    full(janb[janb.STATE_NAME!="TURKIYE"],"jan39_noTUR",2026,res)
    summary(jan[jan.STATE_NAME!="TURKIYE"],"jan42_noTUR",2026,res)
    # unbalanced: beta on all states available at both endpoints
    res["jan42_beta_unbal"]=[beta(jan,2010,2019),beta(jan,2019,2026),beta(jan,2022,2026)]
    # NOTE share by group
    n=df.groupby(["YEAR","GROUP"]).NOTE.apply(lambda s:(s.astype(str).str.upper()=="TRUE").mean()*100).unstack()
    res["note_share"]=n.round(1).to_dict()
    jan.to_csv("out/jan_panel_42.csv",index=False)
    json.dump(res,open("out/results.json","w"),indent=1,default=lambda o: float(o) if isinstance(o,(np.floating,np.integer)) else str(o))
    print("done")
