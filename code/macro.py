"""Macro-conditioned beta and per-capita robustness (World Bank WDI, accessed 26 Sep 2026)."""
import numpy as np, pandas as pd, statsmodels.api as sm, json
from analysis import load, panel, balance
ISO = {"ALBANIA":"ALB","ARMENIA":"ARM","AUSTRIA":"AUT","BELGIUM":"BEL","BOSNIA AND HERZEGOVINA":"BIH",
"BULGARIA":"BGR","SWITZERLAND":"CHE","CYPRUS":"CYP","CZECHIA":"CZE","GERMANY":"DEU","DENMARK":"DNK",
"SPAIN":"ESP","ESTONIA":"EST","FINLAND":"FIN","FRANCE":"FRA","UNITED KINGDOM":"GBR","GEORGIA":"GEO",
"GREECE":"GRC","CROATIA":"HRV","HUNGARY":"HUN","IRELAND":"IRL","ICELAND":"ISL","ITALY":"ITA",
"LITHUANIA":"LTU","LUXEMBOURG":"LUX","LATVIA":"LVA","MOLDOVA":"MDA","NORTH MACEDONIA":"MKD","MALTA":"MLT",
"MONTENEGRO":"MNE","NETHERLANDS":"NLD","NORWAY":"NOR","POLAND":"POL","PORTUGAL":"PRT","ROMANIA":"ROU",
"SERBIA":"SRB","SLOVAKIA":"SVK","SLOVENIA":"SVN","SWEDEN":"SWE","TURKIYE":"TUR","KOSOVO":"XKX"}
wb = pd.read_csv("data/wb_macro.csv").set_index("iso3")
df = load(); jan = balance(panel(df,[1]))
res = {}

def reg(y0, y1, wy0, wy1, spec):
    a = jan[jan.YEAR==y0].set_index("STATE_NAME"); b = jan[jan.YEAR==y1].set_index("STATE_NAME")
    a = a[a.index.isin(ISO)]; b = b.loc[a.index]
    iso = a.index.map(ISO)
    T = y1-y0; Tw = wy1-wy0
    d = pd.DataFrame(index=a.index)
    d["g"] = (np.log(b.CO2)-np.log(a.CO2))/T
    d["lnE0"] = np.log(a.CO2)
    d["D_periph"] = (a.GROUP=="Periphery").astype(float)
    d["gGDPpc"] = (np.log(wb.loc[iso,f"gdppc{wy1}"].values)-np.log(wb.loc[iso,f"gdppc{wy0}"].values))/Tw
    d["gPop"] = (np.log(wb.loc[iso,f"pop{wy1}"].values)-np.log(wb.loc[iso,f"pop{wy0}"].values))/Tw
    d["lnGDPpc0"] = np.log(wb.loc[iso,f"gdppc{wy0}"].values)
    d["ln_int0"] = np.log(a.CO2/a.TF)
    X = sm.add_constant(d[spec]); m = sm.OLS(d.g, X).fit(cov_type="HC3")
    return dict(window=f"{y0}-{y1}", n=int(m.nobs), r2=m.rsquared, adjr2=m.rsquared_adj,
                coef={k:(m.params[k], m.bse[k], m.pvalues[k]) for k in X.columns if k!="const"},
                vif_max=max(VIF(X.values,i) for i in range(1,X.shape[1])) if X.shape[1]>2 else 1.0)
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
specs = {"M1_abs":["lnE0"],
         "M2_group":["lnE0","D_periph"],
         "M3_macro":["lnE0","gGDPpc","gPop","lnGDPpc0"],
         "M4_full":["lnE0","D_periph","gGDPpc","gPop","lnGDPpc0","ln_int0"]}
for win in [(2010,2019,2010,2019),(2019,2026,2019,2025)]:
    for k,s in specs.items():
        res[f"{k}_{win[0]}_{win[1]}"] = reg(*win, s)

# per-capita dispersion (kg CO2 per capita, January), 2026 uses 2025 population
pc = {}
for y,py in [(2010,2010),(2019,2019),(2022,2022),(2026,2025)]:
    s = jan[(jan.YEAR==y)&(jan.STATE_NAME.isin(ISO))].set_index("STATE_NAME")
    v = s.CO2/wb.loc[s.index.map(ISO),f"pop{py}"].values
    pc[y] = dict(sd_log=float(np.std(np.log(v),ddof=1)), n=len(v),
                 core_pc=float(s[s.GROUP=="Core"].CO2.sum()/wb.loc[s[s.GROUP=="Core"].index.map(ISO),f"pop{py}"].sum()*1000),
                 per_pc=float(s[s.GROUP=="Periphery"].CO2.sum()/wb.loc[s[s.GROUP=="Periphery"].index.map(ISO),f"pop{py}"].sum()*1000))
res["per_capita"] = pc
# per-capita beta 2019-2026
a = jan[(jan.YEAR==2019)&(jan.STATE_NAME.isin(ISO))].set_index("STATE_NAME"); b = jan[jan.YEAR==2026].set_index("STATE_NAME").loc[a.index]
pa = a.CO2/wb.loc[a.index.map(ISO),"pop2019"].values; pb = b.CO2/wb.loc[a.index.map(ISO),"pop2025"].values
g = (np.log(pb)-np.log(pa))/7; m = sm.OLS(g, sm.add_constant(np.log(pa))).fit(cov_type="HC3")
res["pc_beta_2019_2026"] = (m.params.iloc[1], m.bse.iloc[1], m.pvalues.iloc[1], m.rsquared, int(m.nobs))
a = jan[(jan.YEAR==2010)&(jan.STATE_NAME.isin(ISO))].set_index("STATE_NAME"); b = jan[jan.YEAR==2019].set_index("STATE_NAME").loc[a.index]
pa = a.CO2/wb.loc[a.index.map(ISO),"pop2010"].values; pb = b.CO2/wb.loc[a.index.map(ISO),"pop2019"].values
g = (np.log(pb)-np.log(pa))/9; m = sm.OLS(g, sm.add_constant(np.log(pa))).fit(cov_type="HC3")
res["pc_beta_2010_2019"] = (m.params.iloc[1], m.bse.iloc[1], m.pvalues.iloc[1], m.rsquared, int(m.nobs))
json.dump(res, open("out/macro_results.json","w"), indent=1, default=float)
for k,v in res.items():
    if k.startswith("M"):
        print(k, "n=",v["n"], "R2=%.3f"%v["r2"], "VIFmax=%.1f"%v["vif_max"])
        for c,(b_,se,p) in v["coef"].items(): print("   %-10s %+.4f (%.4f) p=%.3f"%(c,b_,se,p))
print(json.dumps(res["per_capita"],indent=0)); print(res["pc_beta_2010_2019"], res["pc_beta_2019_2026"])
