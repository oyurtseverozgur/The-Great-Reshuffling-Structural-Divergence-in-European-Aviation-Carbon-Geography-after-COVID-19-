"""Table 4 / A2 with HC3 standard errors and t(n-k) reference distribution."""
import json, numpy as np, pandas as pd, statsmodels.api as sm
import macro
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
jan, wb, ISO = macro.jan, macro.wb, macro.ISO
out = {}
def build(y0, y1, wy0, wy1):
    a = jan[jan.YEAR==y0].set_index("STATE_NAME"); a = a[a.index.isin(ISO)]
    b = jan[jan.YEAR==y1].set_index("STATE_NAME").loc[a.index]; iso = a.index.map(ISO)
    d = pd.DataFrame(index=a.index)
    d["g"] = (np.log(b.CO2)-np.log(a.CO2))/(y1-y0); d["lnE0"] = np.log(a.CO2)
    d["D_periph"] = (a.GROUP=="Periphery").astype(float)
    d["gGDPpc"] = (np.log(wb.loc[iso,f"gdppc{wy1}"].values)-np.log(wb.loc[iso,f"gdppc{wy0}"].values))/(wy1-wy0)
    d["gPop"] = (np.log(wb.loc[iso,f"pop{wy1}"].values)-np.log(wb.loc[iso,f"pop{wy0}"].values))/(wy1-wy0)
    d["lnGDPpc0"] = np.log(wb.loc[iso,f"gdppc{wy0}"].values); d["ln_int0"] = np.log(a.CO2/a.TF)
    return d
for win in [(2010,2019,2010,2019),(2019,2026,2019,2025)]:
    d = build(*win)
    for k,s in macro.specs.items():
        X = sm.add_constant(d[s]); m = sm.OLS(d.g, X).fit(cov_type="HC3", use_t=True)
        out[f"{k}_{win[0]}_{win[1]}"] = dict(n=int(m.nobs), r2=m.rsquared, adjr2=m.rsquared_adj,
            coef={c:(m.params[c], m.bse[c], m.pvalues[c]) for c in s},
            vif=max(VIF(X.values,i) for i in range(1,X.shape[1])) if X.shape[1]>2 else 1.0)
json.dump(out, open("out/table4_t.json","w"), indent=1, default=float)
for k,v in out.items():
    print(k, "n=%d R2=%.2f adj=%.2f VIF=%.1f"%(v["n"],v["r2"],v["adjr2"],v["vif"]))
    for c,(b,se,p) in v["coef"].items(): print("   %-9s %+.4f (%.4f) p=%.3f"%(c,b,se,p))
