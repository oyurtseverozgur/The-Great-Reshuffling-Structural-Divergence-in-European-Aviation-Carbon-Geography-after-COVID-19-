import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
import statsmodels.api as sm
from analysis import load, panel, balance, sigma
from macro import ISO, wb

B, O, G, Y, GR = "#4472C4", "#ED7D31", "#A5A5A5", "#FFC000", "#70AD47"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 9, "axes.edgecolor": "#808080",
    "axes.linewidth": .8, "axes.grid": True, "grid.color": "#D9D9D9", "grid.linewidth": .5, "grid.linestyle": "--",
    "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 100,
    "axes.spines.top": False, "axes.spines.right": False})
df = load(); jan = balance(panel(df, [1])); ann = balance(panel(df, list(range(1, 13)), range(2010, 2026)))
g = jan.groupby(["YEAR", "GROUP"]).CO2.sum().unstack(); yrs = g.index.values
OUT = "fig/"
import os; os.makedirs(OUT, exist_ok=True)

# ---------- Figure 1
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.85, 3.0))
for grp, c, mk, lab in [("Core", B, "o", "ETS-participating states"), ("Periphery", O, "s", "Non-ETS states")]:
    idx = 100 * g[grp] / g.loc[2019, grp]
    a1.plot(yrs, idx, color=c, marker=mk, ms=3.2, lw=1.6, label=lab)
    pre = np.log(g[grp].loc[2010:2020]); t = pre.index.values.astype(float)
    m = sm.OLS(pre.values, sm.add_constant(t)).fit()
    tt = np.arange(2019, 2027); a1.plot(tt, 100 * np.exp(m.params[0] + m.params[1] * tt) / g.loc[2019, grp],
                                       color=c, ls="--", lw=1.0, alpha=.8)
a1.axvline(2020.5, color=G, lw=.8, ls=":")
a1.set_ylabel("Index (January 2019 = 100)"); a1.set_title("a", fontsize=10, loc="left", fontweight="bold")
a1.legend(loc="upper left", fontsize=7.5)
a1.set_xticks(range(2010, 2027, 4))
sh = 100 * g.Periphery / g.sum(axis=1)
a2.bar(yrs, sh, color=O, width=.7)
for x, v in zip(yrs, sh):
    if x in (2010, 2014, 2019, 2021, 2022, 2026): a2.text(x, v + .2, f"{v:.1f}", ha="center", fontsize=6.8)
a2.set_ylabel("Non-ETS share of total CO$_2$ (%)"); a2.set_title("b", fontsize=10, loc="left", fontweight="bold")
a2.set_xticks(range(2010, 2027, 4)); a2.set_ylim(0, 13)
plt.tight_layout(); plt.savefig(OUT + "Fig1.png"); plt.savefig(OUT + "Fig1.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"}); plt.close()

# ---------- Figure 2
sig = sigma(jan)
pcs = {}
for y in yrs:
    py = min(y, 2025); s = jan[(jan.YEAR == y) & jan.STATE_NAME.isin(ISO)].set_index("STATE_NAME")
    pcs[y] = np.std(np.log(s.CO2 / wb.loc[s.index.map(ISO), f"pop{py}"].values if f"pop{py}" in wb else np.nan), ddof=1) if f"pop{py}" in wb else np.nan
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.85, 3.0))
a1.plot(yrs, sig.values, color=B, marker="o", ms=3.2, lw=1.6, label="Total CO$_2$ (39 states)")
pcy = [y for y in yrs if not np.isnan(pcs[y])]
a1.plot(pcy, [pcs[y] for y in pcy], color=GR, marker="D", ms=3.2, lw=1.6, label="CO$_2$ per capita (38 states; WDI years)")
a1.axvline(2020.5, color=G, lw=.8, ls=":"); a1.set_ylabel("SD of log CO$_2$"); a1.legend(fontsize=7.5)
a1.set_title("a", fontsize=10, loc="left", fontweight="bold"); a1.set_xticks(range(2010, 2027, 4))
a = jan[jan.YEAR == 2019].set_index("STATE_NAME"); b = jan[jan.YEAR == 2026].set_index("STATE_NAME").loc[a.index]
x = np.log(a.CO2); gr = (np.log(b.CO2) - x) / 7 * 100
for grp, c, mk, lab in [("Core", B, "o", "ETS-participating states"), ("Periphery", O, "s", "Non-ETS states")]:
    m_ = a.GROUP == grp; a2.scatter(x[m_], gr[m_], s=20, color=c, marker=mk, edgecolor="white", lw=.4, zorder=3, label=lab)
sl, ic, *_ = stats.linregress(x, gr); xx = np.linspace(x.min(), x.max(), 50)
a2.plot(xx, ic + sl * xx, color=G, ls="--", lw=1.2, label="OLS fit")
for s_ in ["TURKIYE", "ALBANIA", "MOLDOVA", "HUNGARY", "SWEDEN", "MONTENEGRO"]:
    a2.annotate(s_.title().replace("Turkiye", "Türkiye"), (x[s_], gr[s_]), fontsize=6.3, xytext=(3, 2), textcoords="offset points")
a2.axhline(0, color="#808080", lw=.6)
a2.set_xlabel("ln CO$_2$, January 2019 (t)"); a2.set_ylabel("Annualised growth 2019–2026 (%/yr)")
a2.set_title("b", fontsize=10, loc="left", fontweight="bold"); a2.legend(fontsize=7, loc="upper right")
plt.tight_layout(); plt.savefig(OUT + "Fig2.png"); plt.savefig(OUT + "Fig2.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"}); plt.close()

# ---------- Figure 3
ch = ((b.CO2 / a.CO2 - 1) * 100).sort_values()
fig, ax = plt.subplots(figsize=(5.5, 6.6))
cols = [O if a.GROUP[s] == "Periphery" else B for s in ch.index]
ax.barh(range(len(ch)), ch.values, color=cols, height=.7)
ax.set_yticks(range(len(ch))); ax.set_yticklabels([s.title().replace("Turkiye", "Türkiye").replace(" And ", " and ") for s in ch.index], fontsize=7)
for i, v in enumerate(ch.values):
    ax.text(v + (2 if v >= 0 else -2), i, f"{v:+.0f}", va="center", ha="left" if v >= 0 else "right", fontsize=6.3)
ax.axvline(0, color="#595959", lw=.8); ax.grid(axis="y", visible=False); ax.set_xlim(-55, 185)
ax.set_xlabel("Change in January CO$_2$, 2019→2026 (%)")
ax.legend(handles=[Patch(fc=B, label="ETS-participating states"), Patch(fc=O, label="Non-ETS states")], loc="lower right", fontsize=7.5)
plt.tight_layout(); plt.savefig(OUT + "Fig3.png"); plt.savefig(OUT + "Fig3.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"}); plt.close()

# ---------- Figure 4
def inten(p):
    return p.groupby(["YEAR", "GROUP"]).apply(lambda z: z.CO2.sum() / z.TF.sum(), include_groups=False).unstack()
ij, ia = inten(jan), inten(ann)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.85, 2.9), sharey=True)
for ax, d, ttl in [(a1, ij, "a"), (a2, ia, "b")]:
    ax.plot(d.index, d.Core, color=B, marker="o", ms=3.2, lw=1.6, label="ETS-participating states")
    ax.plot(d.index, d.Periphery, color=O, marker="s", ms=3.2, lw=1.6, label="Non-ETS states")
    ax.axvline(2020.5, color=G, lw=.8, ls=":"); ax.set_title(ttl, fontsize=10, loc="left", fontweight="bold"); ax.set_xticks(range(2010, 2027, 4))
a1.set_ylabel("CO$_2$ per departing flight (t)"); a1.legend(fontsize=7.5, loc="upper left")
plt.tight_layout(); plt.savefig(OUT + "Fig4.png"); plt.savefig(OUT + "Fig4.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"}); plt.close()

# ---------- Figure 5: growth against initial income
a = jan[jan.YEAR == 2019].set_index("STATE_NAME"); a = a[a.index.isin(ISO)]
b = jan[jan.YEAR == 2026].set_index("STATE_NAME").loc[a.index]
gr = (np.log(b.CO2) - np.log(a.CO2)) / 7 * 100
inc = np.log(wb.loc[a.index.map(ISO), "gdppc2019"].values)
inc = pd.Series(inc, index=a.index)
fig, ax = plt.subplots(figsize=(4.6, 3.3))
for grp, c, mk, lab in [("Core", B, "o", "ETS-participating states"), ("Periphery", O, "s", "Non-ETS states")]:
    m_ = a.GROUP == grp
    ax.scatter(inc[m_], gr[m_], s=22, color=c, marker=mk, edgecolor="white", lw=.4, zorder=3, label=lab)
sl, ic, *_ = stats.linregress(inc, gr); xx = np.linspace(inc.min(), inc.max(), 50)
ax.plot(xx, ic + sl * xx, color=G, ls="--", lw=1.2, label="OLS fit, all states")
off = {"TURKIYE": (14, 22), "POLAND": (4, 3)}
for s_ in ["TURKIYE", "ALBANIA", "HUNGARY", "POLAND", "SWEDEN", "MONTENEGRO", "GERMANY"]:
    ap = dict(arrowstyle="-", color="#595959", lw=.5) if s_ == "TURKIYE" else None
    ax.annotate(s_.title().replace("Turkiye", "Türkiye"), (inc[s_], gr[s_]), fontsize=6.3, xytext=off.get(s_, (3, 2)), textcoords="offset points", arrowprops=ap)
ax.axhline(0, color="#808080", lw=.6)
ax.set_xlabel("ln GDP per capita, 2019 (constant 2015 US\\$)"); ax.set_ylabel("Annualised CO$_2$ growth 2019–2026 (%/yr)")
ax.legend(fontsize=7, loc="upper right")
plt.tight_layout(); plt.savefig(OUT + "Fig5.png"); plt.savefig(OUT + "Fig5.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"}); plt.close()
print("fig5 slope %.3f" % sl)
json.dump({"pc_sigma": {int(k): v for k, v in pcs.items()}, "int_jan": ij.round(2).to_dict(), "int_ann": ia.round(2).to_dict()},
          open("out/fig_numbers.json", "w"), indent=1, default=float)
print("ok", {k: round(v, 3) for k, v in pcs.items()})
# Springer requires RGB (8 bits/channel) TIFF
from PIL import Image
for i in range(1, 6):
    Image.open(OUT + f"Fig{i}.tif").convert("RGB").save(OUT + f"Fig{i}.tif", dpi=(600, 600), compression="tiff_lzw")
