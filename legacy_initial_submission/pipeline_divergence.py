#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 THE GREAT RESHUFFLING: Structural Divergence in European Aviation
 Carbon Geography after COVID-19 (2010-2026)
 
 Full reproducible pipeline - Google Colab compatible
 Author: Ozgur Yurtsever (Istanbul Nisantasi University)
 
 INPUT : co2_emmissions_by_state_2010.csv ... co2_emmissions_by_state_2026.csv
         (EUROCONTROL Aviation Intelligence Unit, monthly state-level data;
          columns: YEAR, MONTH, STATE_NAME, STATE_CODE, CO2_QTY_TONNES, TF, NOTE)
 OUTPUT: ./output/  -> figures (300 DPI, Excel-style), tables (CSV + LaTeX),
                       full results log
 USAGE : place CSVs in ./data/ (or Colab: upload / mount Drive) and run.
================================================================================
"""

import glob, os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
import statsmodels.api as sm

warnings.filterwarnings("ignore")

DATA_DIR   = "./data"
OUT_DIR    = "./output"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------ constants
# Carbon-pricing coverage classification (EU ETS aviation since 2012; Swiss ETS
# linked 2020; UK ETS from 2021; EEA members Norway & Iceland in EU ETS).
# Source: European Commission, DG CLIMA aviation pages (accessed 2026).
PRICED = {
    "AUSTRIA","BELGIUM","BULGARIA","CROATIA","CYPRUS","CZECHIA","DENMARK",
    "ESTONIA","FINLAND","FRANCE","GERMANY","GREECE","HUNGARY","IRELAND",
    "ITALY","LATVIA","LITHUANIA","LUXEMBOURG","MALTA","NETHERLANDS","POLAND",
    "PORTUGAL","ROMANIA","SLOVAKIA","SLOVENIA","SPAIN","SWEDEN",
    "CANARY ISLANDS","NORWAY","ICELAND","SWITZERLAND","UNITED KINGDOM",
}
NONPRICED = {
    "TURKIYE","SERBIA","BOSNIA AND HERZEGOVINA","MONTENEGRO","ALBANIA",
    "NORTH MACEDONIA","KOSOVO","MOLDOVA","GEORGIA","ARMENIA",
}
EXCLUDE_MICRO = {"MONACO","LIECHTENSTEIN"}          # <2 flights/day, sporadic
BREAK_YEAR    = 2020
DROP_YEAR     = 2021          # deepest-lockdown year, dropped from trend tests

XL = dict(blue="#4472C4", orange="#ED7D31", gray="#A5A5A5",
          gold="#FFC000", dblue="#2F528F")

plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":9,
    "axes.edgecolor":"#BFBFBF","axes.linewidth":0.8,
    "axes.grid":True,"grid.color":"#D9D9D9","grid.linewidth":0.7,
    "axes.axisbelow":True,"axes.titlesize":10,"axes.titleweight":"bold",
    "legend.frameon":True,"legend.edgecolor":"#BFBFBF","legend.fontsize":8.5,
    "figure.dpi":300,"savefig.dpi":300})

# =============================================================== 1. LOAD & QC
def load_panel(data_dir=DATA_DIR):
    """Load all yearly CSVs. Keys on STATE_NAME (never STATE_CODE: the codes
    LY and LS each map to two different states in the raw files)."""
    files = sorted(glob.glob(os.path.join(data_dir, "co2_emmissions_by_state_*.csv")))
    if not files:
        sys.exit(f"No input CSVs found under {data_dir}/")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d.columns = [c.strip().upper() for c in d.columns]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    # -- normalise state names (Unicode-safe: TUERKIYE variants, Moldova) ----
    df["STATE_NAME"] = (df["STATE_NAME"].str.strip().str.upper()
                        .str.replace(r"T.RKIYE|T.RK.YE", "TURKIYE", regex=True)
                        .str.replace(r"MOLDOVA.*", "MOLDOVA", regex=True))

    # -- data-quality report --------------------------------------------------
    print("=" * 70, "\nDATA QUALITY REPORT")
    dup = df.groupby("STATE_CODE")["STATE_NAME"].nunique()
    for code in dup[dup > 1].index:
        names = sorted(df.loc[df.STATE_CODE == code, "STATE_NAME"].unique())
        print(f"  [!] STATE_CODE collision {code}: {names} -> keyed on STATE_NAME")
    assert not df.duplicated(["YEAR","MONTH","STATE_NAME"]).any(), \
        "Duplicate YEAR-MONTH-STATE rows found - inspect raw files."
    roster = df.groupby("YEAR")["STATE_NAME"].nunique()
    print("  States per year:", dict(roster))
    note = df.groupby("YEAR")["NOTE"].apply(lambda s: (s.astype(str)
            .str.upper().eq("TRUE")).mean() * 100).round(1)
    print("  Share of NOTE=TRUE rows by year (%):", dict(note))
    return df

# ====================================================== 2. PANEL CONSTRUCTION
def build_panels(df):
    df = df[~df.STATE_NAME.isin(EXCLUDE_MICRO)].copy()
    df["GROUP"] = np.where(df.STATE_NAME.isin(NONPRICED), "Non-priced", "Priced")

    # annual totals; the last calendar year may be partial -> flag it
    months_per = df.groupby(["YEAR"])["MONTH"].nunique()
    partial = months_per[months_per < 12].index.tolist()
    if partial:
        print(f"  [i] Partial year(s) detected (annualise or truncate): {partial}")

    ann = (df.groupby(["YEAR","STATE_NAME","GROUP"], as_index=False)
             .agg(CO2=("CO2_QTY_TONNES","sum"), TF=("TF","sum"),
                  N_MONTHS=("MONTH","nunique")))
    jan = df[df.MONTH == 1].rename(columns={"CO2_QTY_TONNES":"CO2"})[
              ["YEAR","STATE_NAME","GROUP","CO2","TF"]]

    def balance(p):
        n_years = p.YEAR.nunique()
        keep = p.groupby("STATE_NAME")["YEAR"].nunique()
        core = keep[keep == n_years].index
        out  = p[p.STATE_NAME.isin(core)].copy()
        out["INT"] = out.CO2 / out.TF
        return out

    return balance(ann), balance(jan)

# ===================================================== 3. CONVERGENCE METRICS
def sigma_convergence(p):
    return p.groupby("YEAR").apply(
        lambda x: np.std(np.log(x.CO2), ddof=1), include_groups=False)

def beta_convergence(p, y0, y1):
    a = p[p.YEAR == y0].set_index("STATE_NAME").CO2
    b = p[p.YEAR == y1].set_index("STATE_NAME").CO2
    x  = np.log(a)
    g  = (np.log(b) - np.log(a)) / (y1 - y0)
    sl, ic, r, pv, se = stats.linregress(x, g)
    return dict(window=f"{y0}-{y1}", beta=sl, se=se, p=pv, r2=r**2, n=len(x))

def theil_decomposition(sub):
    x  = sub.CO2.values; mu = x.mean()
    T  = np.mean(x/mu * np.log(x/mu))
    Tb = Tw = 0.0
    for _, g in sub.groupby("GROUP"):
        xg, mg, sg = g.CO2.values, g.CO2.mean(), g.CO2.sum()/x.sum()
        Tb += sg*np.log(mg/mu)
        Tw += sg*np.mean(xg/mg*np.log(xg/mg))
    return T, Tb, Tw

def chow_2020(p, group):
    s = p[p.GROUP == group].groupby("YEAR").CO2.sum()
    s = s[s.index != DROP_YEAR]
    t, y = s.index.values.astype(float), np.log(s.values)
    full  = sm.OLS(y, sm.add_constant(t)).fit()
    d     = (t >= BREAK_YEAR).astype(float)
    X     = sm.add_constant(np.column_stack([t, d, d*(t-BREAK_YEAR)]))
    br    = sm.OLS(y, X).fit()
    F     = ((full.ssr-br.ssr)/2) / (br.ssr/(len(t)-4))
    return F, 1-stats.f.cdf(F, 2, len(t)-4), br.params[3]

# ================================================================= 4. FIGURES
def make_figures(p, tag):
    g   = p.groupby(["YEAR","GROUP"]).CO2.sum().unstack()/1e6
    yrs = g.index.values
    sh  = 100*g["Non-priced"]/g.sum(axis=1)

    # Figure 1 - trajectories & share
    fig,(a1,a2) = plt.subplots(1,2, figsize=(9.0,3.4))
    a1.plot(yrs,100*g["Priced"]/g.loc[2019,"Priced"], color=XL["blue"],
            lw=1.8, marker="o", ms=3.5, label="Carbon-priced core")
    a1.plot(yrs,100*g["Non-priced"]/g.loc[2019,"Non-priced"], color=XL["orange"],
            lw=1.8, marker="s", ms=3.5, label="Non-priced periphery")
    a1.axhline(100, color="#7F7F7F", lw=.9, ls="--")
    a1.axvline(BREAK_YEAR, color="#C00000", lw=.9, ls=":")
    a1.set_title("(a) CO$_2$ emissions, index 2019 = 100"); a1.legend()
    a1.grid(axis="x", visible=False)
    a2.bar(yrs, sh, color=XL["blue"], edgecolor=XL["dblue"], lw=.5, width=.65)
    for x,v in zip(yrs,sh): a2.text(x, v+.15, f"{v:.1f}", ha="center", fontsize=6.6)
    a2.set_title("(b) Non-priced periphery share of total CO$_2$ (%)")
    a2.grid(axis="x", visible=False)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig1_{tag}.png", bbox_inches="tight"); plt.close()

    # Figure 2 - sigma & beta
    sig = sigma_convergence(p)
    fig,(a1,a2) = plt.subplots(1,2, figsize=(9.0,3.4))
    a1.plot(sig.index, sig.values, color=XL["blue"], lw=1.8, marker="o", ms=3.5)
    a1.axvline(BREAK_YEAR, color="#C00000", lw=.9, ls=":")
    a1.set_title("(a) $\\sigma$-convergence: SD of log CO$_2$")
    a1.grid(axis="x", visible=False)
    y0, y1 = 2019, int(p.YEAR.max())
    a = p[p.YEAR==y0].set_index("STATE_NAME"); b = p[p.YEAR==y1].set_index("STATE_NAME")
    x = np.log(a.CO2); gr = (np.log(b.CO2)-np.log(a.CO2))/(y1-y0)*100
    for gname,col,mk in [("Priced",XL["blue"],"o"),("Non-priced",XL["orange"],"s")]:
        m = a.GROUP==gname
        a2.scatter(x[m], gr[m], s=22, color=col, marker=mk,
                   edgecolor="white", lw=.4, zorder=3, label=gname)
    sl,ic,r,pv,se = stats.linregress(x,gr)
    xx = np.linspace(x.min(),x.max(),50)
    a2.plot(xx, ic+sl*xx, color=XL["gray"], lw=1.4, ls="--",
            label=f"OLS ($\\beta$={sl:.2f})")
    a2.set_title(f"(b) $\\beta$-convergence, {y0}$-${y1}"); a2.legend()
    a2.grid(axis="x", visible=False)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig2_{tag}.png", bbox_inches="tight"); plt.close()

    # Figure 3 - state-level change bars
    ch  = (b.CO2/a.CO2-1)*100; ch = ch.sort_values()
    grp = a.GROUP
    fig,ax = plt.subplots(figsize=(7.2, .2*len(ch)+1.4))
    cols = [XL["orange"] if grp[s]=="Non-priced" else XL["blue"] for s in ch.index]
    ax.barh(range(len(ch)), ch.values, color=cols, edgecolor="#7F7F7F", lw=.3, height=.68)
    ax.set_yticks(range(len(ch)))
    ax.set_yticklabels([s.title() for s in ch.index], fontsize=7.2)
    ax.axvline(0, color="#595959", lw=.9)
    for i,v in enumerate(ch.values):
        ax.text(v+(1.5 if v>=0 else -1.5), i, f"{v:+.0f}%", va="center",
                ha="left" if v>=0 else "right", fontsize=6.4)
    ax.set_title(f"Change in CO$_2$ emissions, {y0} $\\rightarrow$ {y1} (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(fc=XL["blue"],label="Carbon-priced core"),
                       Patch(fc=XL["orange"],label="Non-priced periphery")],
              loc="lower right")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig3_{tag}.png", bbox_inches="tight"); plt.close()

    # Figure 4 - intensity crossover
    gi = p.groupby(["YEAR","GROUP"]).apply(
        lambda x: x.CO2.sum()/x.TF.sum(), include_groups=False).unstack()
    fig,ax = plt.subplots(figsize=(5.6,3.3))
    ax.plot(gi.index, gi["Priced"], color=XL["blue"], lw=1.8, marker="o",
            ms=3.5, label="Carbon-priced core")
    ax.plot(gi.index, gi["Non-priced"], color=XL["orange"], lw=1.8, marker="s",
            ms=3.5, label="Non-priced periphery")
    ax.axvline(BREAK_YEAR, color="#C00000", lw=.9, ls=":")
    ax.set_title("Average CO$_2$ per flight (tonnes)")
    ax.legend(); ax.grid(axis="x", visible=False)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig4_{tag}.png", bbox_inches="tight"); plt.close()

# ==================================================================== 5. MAIN
def run(panel, tag):
    print("\n" + "="*70 + f"\nRESULTS - {tag} panel "
          f"({panel.STATE_NAME.nunique()} states, {panel.YEAR.nunique()} years)")
    yend = int(panel.YEAR.max())
    g = panel.groupby(["YEAR","GROUP"]).CO2.sum().unstack()
    sh = 100*g["Non-priced"]/g.sum(axis=1)
    print(f"Periphery share: 2010={sh[2010]:.1f}%  2019={sh[2019]:.1f}%  {yend}={sh[yend]:.1f}%")
    print(f"Core growth 2019->{yend}: {100*(g.loc[yend,'Priced']/g.loc[2019,'Priced']-1):+.1f}% | "
          f"Periphery: {100*(g.loc[yend,'Non-priced']/g.loc[2019,'Non-priced']-1):+.1f}%")

    sig = sigma_convergence(panel)
    print(f"sigma(logCO2): 2010={sig[2010]:.3f} 2019={sig[2019]:.3f} {yend}={sig[yend]:.3f}")

    betas = [beta_convergence(panel,2010,2019),
             beta_convergence(panel,2019,yend),
             beta_convergence(panel,2022,yend)]
    for b in betas:
        print(f"beta {b['window']}: {b['beta']:+.4f} (se={b['se']:.4f}, "
              f"p={b['p']:.4f}, R2={b['r2']:.3f}, n={b['n']})")
    pd.DataFrame(betas).to_csv(f"{OUT_DIR}/table_beta_{tag}.csv", index=False)

    rows=[]
    for y in sorted(set([2010,2015,2019,2022,yend]) & set(panel.YEAR.unique())):
        T,Tb,Tw = theil_decomposition(panel[panel.YEAR==y])
        rows.append(dict(YEAR=y,Theil=T,Between=Tb,Within=Tw,Between_pct=100*Tb/T))
        print(f"Theil {y}: T={T:.4f} between={Tb:.4f} ({100*Tb/T:.1f}%)")
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/table_theil_{tag}.csv", index=False)

    for grp in ["Priced","Non-priced"]:
        F,pv,shift = chow_2020(panel, grp)
        print(f"Chow@{BREAK_YEAR} {grp}: F={F:.2f} p={pv:.4f} trend-shift={shift:+.4f}/yr")

    r19 = panel[panel.YEAR==2019].set_index("STATE_NAME").CO2.rank(ascending=False)
    rE  = panel[panel.YEAR==yend ].set_index("STATE_NAME").CO2.rank(ascending=False)
    rho,pv = stats.spearmanr(r19,rE)
    print(f"Rank mobility (Spearman 2019 vs {yend}): rho={rho:.3f} (p={pv:.2e})")

    make_figures(panel, tag)
    panel.to_csv(f"{OUT_DIR}/panel_{tag}.csv", index=False)

if __name__ == "__main__":
    raw = load_panel()
    annual, january = build_panels(raw)
    run(january, "january")     # seasonally anchored series (manuscript baseline)
    run(annual,  "annual")      # full-year robustness (check partial last year!)
    print(f"\nAll outputs written to {OUT_DIR}/")
