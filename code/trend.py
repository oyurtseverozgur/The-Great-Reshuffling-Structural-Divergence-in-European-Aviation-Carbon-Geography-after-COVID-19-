import numpy as np, pandas as pd, statsmodels.api as sm, json
from analysis import load, panel, balance
jan = balance(panel(load(),[1])); res={}
g = jan.groupby(["YEAR","GROUP"]).CO2.sum().unstack()
for grp in ["Core","Periphery"]:
    s=np.log(g[grp]); pre=s.loc[2010:2020]; t=pre.index.values.astype(float)
    m=sm.OLS(pre.values,sm.add_constant(t)).fit()
    out={}
    for y in [2022,2023,2024,2025,2026]:
        pr=m.get_prediction(np.array([[1.0,float(y)]])); f=pr.summary_frame(alpha=0.05)
        out[y]=dict(actual=float(np.exp(s[y])/1e6), trend=float(np.exp(f["mean"].iloc[0])/1e6),
                    gap_pct=float(100*(np.exp(s[y]-f["mean"].iloc[0])-1)),
                    pi_lo=float(100*(np.exp(s[y]-f["obs_ci_upper"].iloc[0])-1)), pi_hi=float(100*(np.exp(s[y]-f["obs_ci_lower"].iloc[0])-1)))
    res[grp]=dict(slope=m.params[1], slope_se=m.bse[1], r2=m.rsquared, proj=out)
# relative log gap ln(P/C)
r=np.log(g.Periphery/g.Core); r=r[r.index!=2021]
t=r.index.values.astype(float); d=(t>=2022).astype(float)
u=sm.OLS(r.values,sm.add_constant(np.column_stack([t,d,d*(t-2022)]))).fit()
rr=sm.OLS(r.values,sm.add_constant(t)).fit()
F=((rr.ssr-u.ssr)/2)/(u.ssr/(len(t)-4))
from scipy import stats
res["relgap"]=dict(slope_pre=u.params[1],se_pre=u.bse[1],level_shift=u.params[2],p_level=u.pvalues[2],slope_shift=u.params[3],p_slope=u.pvalues[3],F=F,p=1-stats.f.cdf(F,2,len(t)-4),df=(2,len(t)-4))
json.dump(res,open("out/trend_results.json","w"),indent=1,default=float)
print(json.dumps(res,indent=1,default=float))
