#!/usr/bin/env python3
"""Additional analyses for the JATE major revision (R1, Oct 2026).

Covers: HC3 confidence intervals and influence diagnostics for absolute beta,
alternative break definitions and an interrupted-trend model, a yardstick for the
ETS-group return to its 2019 level, persistence of rankings across aggregation
windows, leave-one-out analysis of the non-ETS group, an income-based grouping,
and a market-maturity control.
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from analysis import load, panel, balance, theil
from macro import ISO, wb

df = load()
jan = balance(panel(df, [1]))
jm = balance(panel(df, [1, 2, 3, 4, 5]))
ann = balance(panel(df, list(range(1, 13)), range(2010, 2026)))
res = {}


def growth_frame(p, y0, y1):
    a = p[p.YEAR == y0].set_index("STATE_NAME")
    b = p[p.YEAR == y1].set_index("STATE_NAME").loc[a.index]
    d = pd.DataFrame(index=a.index)
    d["g"] = (np.log(b.CO2) - np.log(a.CO2)) / (y1 - y0)
    d["lnE0"] = np.log(a.CO2)
    d["D"] = (a.GROUP == "Periphery").astype(float)
    d["ln_int0"] = np.log(a.CO2 / a.TF)
    d["F0"] = a.TF
    return d


# 1. absolute beta: HC3 CI, influence, robust M-estimator
def beta_diag(p, y0, y1):
    d = growth_frame(p, y0, y1)
    X = sm.add_constant(d[["lnE0"]])
    m = sm.OLS(d.g, X).fit()
    mr = m.get_robustcov_results(cov_type="HC3", use_t=True)
    ci = mr.conf_int()[1]
    infl = m.get_influence()
    cook = pd.Series(infl.cooks_distance[0], index=d.index)
    dfb = pd.Series(infl.dfbetas[:, 1], index=d.index)
    thr = 4 / len(d)
    loo = pd.Series({s: sm.OLS(d.g.drop(s), X.drop(s)).fit().params["lnE0"] for s in d.index})
    keep = cook[cook <= thr].index
    mk = sm.OLS(d.g.loc[keep], X.loc[keep]).fit(cov_type="HC3", use_t=True)
    rlm = sm.RLM(d.g, X, M=sm.robust.norms.HuberT()).fit()
    return dict(window=f"{y0}-{y1}", n=len(d), beta=m.params["lnE0"], se=m.bse["lnE0"],
                p=m.pvalues["lnE0"], se_hc3=mr.bse[1], p_hc3=mr.pvalues[1],
                ci_hc3=(ci[0], ci[1]), r2=m.rsquared,
                cook_thr=thr, n_influential=int((cook > thr).sum()),
                influential=cook[cook > thr].sort_values(ascending=False).round(3).to_dict(),
                max_abs_dfbeta=(dfb.abs().idxmax(), float(dfb.abs().max())),
                loo_min=(loo.idxmax(), float(loo.max())), loo_max_neg=(loo.idxmin(), float(loo.min())),
                beta_drop_infl=mk.params["lnE0"], p_drop_infl=mk.pvalues["lnE0"], n_drop_infl=int(mk.nobs),
                beta_huber=rlm.params["lnE0"], p_huber=rlm.pvalues["lnE0"])


res["beta_diag"] = [beta_diag(jan, 2010, 2019), beta_diag(jan, 2019, 2026), beta_diag(jan, 2022, 2026)]


# 2. break sensitivity on group log totals and on the log ratio
def series(p, grp=None, ratio=False, drop_states=()):
    q = p[~p.STATE_NAME.isin(drop_states)]
    g = q.groupby(["YEAR", "GROUP"]).CO2.sum().unstack()
    return np.log(g.Periphery / g.Core) if ratio else np.log(g[grp])


def seg_test(s, drop, post_start, pulses=()):
    s = s[~s.index.isin(drop)]
    t = s.index.values.astype(float)
    y = s.values
    base = [t] + [(t == pyr).astype(float) for pyr in pulses]
    Xr = sm.add_constant(np.column_stack(base))
    d = (t >= post_start).astype(float)
    Xu = sm.add_constant(np.column_stack(base + [d, d * (t - post_start)]))
    r = sm.OLS(y, Xr).fit()
    u = sm.OLS(y, Xu).fit()
    k = len(t) - Xu.shape[1]
    F = ((r.ssr - u.ssr) / 2) / (u.ssr / k)
    return dict(n=len(t), df=(2, k), F=F, p=1 - stats.f.cdf(F, 2, k),
                level=u.params[-2], p_level=u.pvalues[-2], slope=u.params[-1], p_slope=u.pvalues[-1],
                slope_pre=u.params[1])


specs = {
    "A_baseline_drop2021_post2022": dict(drop=(2021,), post_start=2022),
    "B_keep2021_pulse_post2022": dict(drop=(), post_start=2022, pulses=(2021,)),
    "C_drop2020_22_post2023": dict(drop=(2020, 2021, 2022), post_start=2023),
    "D_drop2021_22_post2023": dict(drop=(2021, 2022), post_start=2023),
    
}
brk = {}
for name, sp in specs.items():
    brk[name] = {
        "ETS": seg_test(series(jan, "Core"), **sp),
        "nonETS": seg_test(series(jan, "Periphery"), **sp),
        "ratio": seg_test(series(jan, ratio=True), **sp),
        "ratio_noTUR": seg_test(series(jan, ratio=True, drop_states=("TURKIYE",)), **sp),
    }
# full-year series: 2020 and 2021 are both pandemic years
brk["F_annual_drop2020_21_post2022"] = {
    "ratio": seg_test(series(ann, ratio=True), drop=(2020, 2021), post_start=2022),
    "ratio_noTUR": seg_test(series(ann, ratio=True, drop_states=("TURKIYE",)), drop=(2020, 2021), post_start=2022)}
brk["G_janmay_drop2021_post2022"] = {
    "ratio": seg_test(series(jm, ratio=True), drop=(2021,), post_start=2022)}
res["break_sensitivity"] = brk

# 3. yardstick for "ETS group back at its 2019 level"
ets = series(jan, "Core")
pre = ets.loc[2010:2020]
dl = pre.diff().dropna()
res["ets_level_yardstick"] = dict(
    change_jan_2019_2026=float(np.exp(ets[2026] - ets[2019]) - 1),
    change_janmay=float(np.exp(series(jm, "Core")[2026] - series(jm, "Core")[2019]) - 1),
    change_annual_2019_2025=float(np.exp(series(ann, "Core")[2025] - series(ann, "Core")[2019]) - 1),
    pre_yoy_mean=float(dl.mean()), pre_yoy_sd=float(dl.std(ddof=1)),
    pre_yoy_min=float(dl.min()), pre_yoy_max=float(dl.max()))

# 4. persistence across aggregation windows
def ranks(p, y):
    return p[p.YEAR == y].set_index("STATE_NAME").CO2


pers = {}
for y in [2019, 2025]:
    a, b = ranks(jan, y), ranks(ann, y)
    i = a.index.intersection(b.index)
    pers[f"spearman_level_jan_vs_annual_{y}"] = stats.spearmanr(a[i], b[i])[0]
gj = (ranks(jan, 2026) / ranks(jan, 2019) - 1)
ga = (ranks(ann, 2025) / ranks(ann, 2019) - 1)
gm = (ranks(jm, 2026) / ranks(jm, 2019) - 1)
i = gj.index.intersection(ga.index)
pers["spearman_change_jan_vs_annual"] = stats.spearmanr(gj[i], ga[i])[0]
pers["spearman_change_jan_vs_janmay"] = stats.spearmanr(gj[i], gm[i])[0]
pers["annual_change_nonETS"] = ga[[s for s in i if s in set(jan[jan.GROUP == "Periphery"].STATE_NAME)]].round(3).to_dict()
pers["annual_top_increases"] = ga.sort_values(ascending=False).head(8).round(3).to_dict()
pers["annual_top_decreases"] = ga.sort_values().head(6).round(3).to_dict()
pers["n_nonETS_growing_annual"] = int((ga[[s for s in i if s in set(jan[jan.GROUP == "Periphery"].STATE_NAME)]] > 0).sum())
# seasonality: July-August CO2 relative to January
sm_ = df[df.YEAR == 2019].pivot_table(index="STATE_NAME", columns="MONTH", values="CO2_QTY_TONNES", aggfunc="sum")
seas = ((sm_[7] + sm_[8]) / 2 / sm_[1]).loc[i]
pers["seasonality_jul_aug_over_jan_2019_top"] = seas.sort_values(ascending=False).head(8).round(2).to_dict()
diffgap = (gj[i] - ga[i])
pers["corr_seasonality_vs_jan_minus_annual_change"] = stats.spearmanr(seas, diffgap)[0]
res["persistence"] = pers

# 5. leave-one-out across non-ETS states
non = sorted(jan[jan.GROUP == "Periphery"].STATE_NAME.unique())
lo = {}
for s in non + [None]:
    q = jan if s is None else jan[jan.STATE_NAME != s]
    g = q.groupby(["YEAR", "GROUP"]).CO2.sum().unstack()
    sh = 100 * g.Periphery / g.sum(axis=1)
    d = growth_frame(q, 2019, 2026)
    m = sm.OLS(d.g, sm.add_constant(d[["lnE0", "D"]])).fit(cov_type="HC3", use_t=True)
    rt = seg_test(np.log(g.Periphery / g.Core), drop=(2021,), post_start=2022)
    lo[s or "none"] = dict(share2010=sh[2010], share2019=sh[2019], share2026=sh[2026],
                           chg_nonETS=g.Periphery[2026] / g.Periphery[2019] - 1,
                           cagr_pre=(g.Periphery[2019] / g.Periphery[2010]) ** (1 / 9) - 1,
                           cagr_post=(g.Periphery[2026] / g.Periphery[2019]) ** (1 / 7) - 1,
                           premium=m.params["D"], p_premium=m.pvalues["D"],
                           ratio_F=rt["F"], ratio_p=rt["p"], ratio_slope_pre=rt["slope_pre"])
res["leave_one_out_nonETS"] = lo
# non-ETS excluding Turkiye: balanced 8-state group
q = jan[jan.STATE_NAME != "TURKIYE"]
g = q.groupby(["YEAR", "GROUP"]).CO2.sum().unstack()
res["nonETS_exTUR_balanced"] = dict(Mt={int(y): g.Periphery[y] / 1e6 for y in [2010, 2019, 2021, 2026]},
                                    idx2026=100 * g.Periphery[2026] / g.Periphery[2019],
                                    share={int(y): 100 * g.Periphery[y] / g.sum(axis=1)[y] for y in [2010, 2019, 2026]})

# 6. income-based grouping (Reviewer 1)
inc = {}
gdp10 = {s: wb.loc[ISO[s], "gdppc2010"] for s in jan.STATE_NAME.unique() if s in ISO}
gdp10["CANARY ISLANDS"] = wb.loc["ESP", "gdppc2010"]  # national value used for grouping only
gdp10 = pd.Series(gdp10)
for label, cut in [("below_median", gdp10.median()), ("lowest_third", gdp10.quantile(1 / 3))]:
    low = set(gdp10[gdp10 < cut].index)
    q = jan.copy()
    q["GROUP"] = np.where(q.STATE_NAME.isin(low), "Periphery", "Core")
    gg = q.groupby(["YEAR", "GROUP"]).CO2.sum().unstack()
    sh = 100 * gg.Periphery / gg.sum(axis=1)
    th = {y: theil(q[q.YEAR == y]) for y in [2010, 2019, 2026]}
    d = growth_frame(q, 2019, 2026)
    m = sm.OLS(d.g, sm.add_constant(d[["lnE0", "D"]])).fit(cov_type="HC3", use_t=True)
    rt = seg_test(np.log(gg.Periphery / gg.Core), drop=(2021,), post_start=2022)
    inc[label] = dict(cut=float(cut), n_low=len(low), low_states=sorted(low),
                      n_low_ETS=len(low & set(jan[jan.GROUP == "Core"].STATE_NAME)),
                      share={int(y): sh[y] for y in [2010, 2019, 2026]},
                      chg_low=gg.Periphery[2026] / gg.Periphery[2019] - 1,
                      chg_high=gg.Core[2026] / gg.Core[2019] - 1,
                      theil_between_pct={y: 100 * th[y][1] / th[y][0] for y in th},
                      premium=m.params["D"], p_premium=m.pvalues["D"], r2=m.rsquared,
                      ratio_F=rt["F"], ratio_p=rt["p"], ratio_slope_pre=rt["slope_pre"])
# horse race: ETS dummy and low-income dummy together
low = set(gdp10[gdp10 < gdp10.median()].index)
d = growth_frame(jan, 2019, 2026)
d["LOW"] = d.index.isin(low).astype(float)
hr = sm.OLS(d.g, sm.add_constant(d[["lnE0", "D", "LOW"]])).fit(cov_type="HC3", use_t=True)
inc["horse_race"] = {k: (hr.params[k], hr.bse[k], hr.pvalues[k]) for k in ["lnE0", "D", "LOW"]}
inc["horse_race_r2"] = hr.rsquared
# 2x2 cells: mean state growth 2019-2026 and aggregate change
cells = {}
for e in [0, 1]:
    for l_ in [0, 1]:
        idx = d[(d.D == e) & (d.LOW == l_)].index
        if len(idx) == 0:
            continue
        a = jan[(jan.YEAR == 2019) & jan.STATE_NAME.isin(idx)].CO2.sum()
        b = jan[(jan.YEAR == 2026) & jan.STATE_NAME.isin(idx)].CO2.sum()
        cells[f"nonETS={e},low={l_}"] = dict(n=len(idx), mean_g=float(d.loc[idx, "g"].mean()),
                                             agg_chg=float(b / a - 1), states=sorted(idx))
inc["cells"] = cells
res["income_grouping"] = inc

# 7. market maturity: departures per capita (January 2019, per 1,000 inhabitants)
a = jan[jan.YEAR == 2019].set_index("STATE_NAME")
a = a[a.index.isin(ISO)]
iso = a.index.map(ISO)
b = jan[jan.YEAR == 2026].set_index("STATE_NAME").loc[a.index]
dd = pd.DataFrame(index=a.index)
dd["g"] = (np.log(b.CO2) - np.log(a.CO2)) / 7
dd["lnE0"] = np.log(a.CO2)
dd["D"] = (a.GROUP == "Periphery").astype(float)
dd["gGDPpc"] = (np.log(wb.loc[iso, "gdppc2025"].values) - np.log(wb.loc[iso, "gdppc2019"].values)) / 6
dd["gPop"] = (np.log(wb.loc[iso, "pop2025"].values) - np.log(wb.loc[iso, "pop2019"].values)) / 6
dd["lnGDPpc0"] = np.log(wb.loc[iso, "gdppc2019"].values)
dd["ln_int0"] = np.log(a.CO2 / a.TF)
dd["ln_fpc0"] = np.log(a.TF / wb.loc[iso, "pop2019"].values * 1000)
mat = {}
for name, cols in {"M5_group_maturity": ["lnE0", "D", "ln_fpc0"],
                   "M6_full_maturity": ["lnE0", "D", "gGDPpc", "gPop", "lnGDPpc0", "ln_int0", "ln_fpc0"]}.items():
    m = sm.OLS(dd.g, sm.add_constant(dd[cols])).fit(cov_type="HC3", use_t=True)
    mat[name] = dict(n=int(m.nobs), r2=m.rsquared, adjr2=m.rsquared_adj,
                     coef={k: (m.params[k], m.bse[k], m.pvalues[k]) for k in cols})
mat["corr_fpc_gdppc"] = float(np.corrcoef(dd.ln_fpc0, dd.lnGDPpc0)[0, 1])
res["maturity"] = mat

json.dump(res, open("out/revision_results.json", "w"), indent=1,
          default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
print(json.dumps(res, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
