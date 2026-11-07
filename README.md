# The Great Reshuffling: Post-Pandemic Redistribution of European Aviation CO₂ across the Emissions-Trading Boundary (2010–2026)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Replication package for the manuscript

**Yurtsever, Ö.** *The Great Reshuffling: Post-Pandemic Redistribution of European Aviation CO₂ across the Emissions-Trading Boundary (2010–2026).* Revised submission (R1) to the *Journal of Aviation Technology and Engineering*, November 2026.

The code reproduces every table, figure and statistic of the revised manuscript from the public EUROCONTROL state-level CO₂ files and a World Bank extract.

## Repository structure

```
code/
  analysis.py        panels, σ/β-convergence, Theil, Spearman, robustness windows (Tables 1, 3, 6, 7)
  macro.py           macro-conditioned β and per-capita σ/β (World Bank covariates)
  tab4_t.py          Table 4 and Table A3 with HC3 standard errors (t reference distribution)
  trend.py           trend-path diagnostics (Table 2)
  revision_jate.py   HC3 confidence intervals, influence diagnostics, break sensitivity,
                     leave-one-out, income-based grouping, market maturity (Tables 3, 4-M5, 5, A2, A4)
  make_figs.py       Figures 1–5 (PNG 300 dpi, TIFF 600 dpi)
data/
  wb_macro.csv                    World Bank WDI extract (GDP per capita, constant 2015 US$; population)
  derived/derived_data_panels.xlsx  January, January–May and full-year state panels and the WDI extract
legacy_initial_submission/        code of the initial submission (kept for transparency; superseded)
```

## Data

* **EUROCONTROL** "CO₂ emissions by State – SET" monthly files, January 2010 – May 2026
  (<https://ansperformance.eu/reference/dataset/emissions/>, downloaded 4 July 2026).
  The files are not redistributed here. Download `co2_emmissions_by_state_2010.csv` … `co2_emmissions_by_state_2026.csv` and place them in `data/`.
* **World Bank** World Development Indicators, NY.GDP.PCAP.KD and SP.POP.TOTL for 2010, 2019, 2022 and 2025
  (<https://data.worldbank.org>, retrieved 26 September 2026), in `data/wb_macro.csv`.

## How to run

From the repository root (Python ≥ 3.10):

```bash
pip install -r requirements.txt
mkdir -p out fig
python code/analysis.py       # -> out/results.json
python code/macro.py          # -> out/macro_results.json
python code/tab4_t.py         # -> out/table4_t.json
python code/trend.py          # -> out/trend_results.json
python code/revision_jate.py  # -> out/revision_results.json
python code/make_figs.py      # -> fig/
```

In the code, `Core` denotes ETS-participating states and `Periphery` non-ETS states.
Figure files map to the manuscript as follows: `Fig1` → Figure 1, `Fig2` → Figure 2, `Fig5` → Figure 3, `Fig3` → Figure 4, `Fig4` → Figure 5.

## Key design choices

* All merges are keyed on `STATE_NAME`. `STATE_CODE` is not unique (LY = Montenegro and Serbia; LS = Liechtenstein and Switzerland).
* Monaco and Liechtenstein are dropped (sporadic or negligible traffic). The balanced 39-state panel excludes the late entrants Latvia (2011), Georgia (2014) and Estonia (2018).
* The baseline uses January observations; January 2020 is treated as pre-pandemic. Break tests exclude January 2021 and allow level and slope shifts from 2022; alternative break definitions are in `revision_jate.py`.
* The Canary Islands are excluded from regressions that use World Bank covariates (no separate national statistics).

## Note on the initial submission

`legacy_initial_submission/` contains the code used for the first version of the manuscript. It is kept only for transparency. Several statistics in that version were corrected in the revision (see the response to reviewers); use the scripts in `code/` to reproduce the revised paper.

## License

Code: MIT (see `LICENSE`). Data remain subject to the terms of EUROCONTROL and the World Bank.
