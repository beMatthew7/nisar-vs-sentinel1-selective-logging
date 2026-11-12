"""Step 1. Explore the DETER dataset to understand what it offers.

Run (from the project folder):
    python step1_deter/explore_deter.py "../deter_brasilia_dataset/deter-amz-deter-public.shp"

Writes to step1_deter/figures/:
    fig1_classes.png      number of alerts per class
    fig2_over_time.png    selective logging per year and per month of the year
    fig3_area.png         size of the selective-logging polygons
    fig4_study_area.png   selective logging in the study area, 2026 season
    summary.txt           the main numbers
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pyogrio

SHP = sys.argv[1] if len(sys.argv) > 1 else "../deter_brasilia_dataset/deter-amz-deter-public.shp"
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

SELECTIVE = ["CS_DESORDENADO", "CS_GEOMETRICO"]       # the two selective-logging classes
AOI = (-49.0, -4.5, -47.0, -2.5)                      # lon_min, lat_min, lon_max, lat_max (Paragominas)
SEASON_START = "2026-06-17"                            # first calibrated NISAR data
RED, GREY, BLUE = "#d1495b", "#9a9a9a", "#2e86ab"

# ---------------------------------------------------------------- 1. attribute table (no geometry = fast)
print("Reading the DETER attribute table...")
t = pyogrio.read_dataframe(SHP, read_geometry=False)
t["VIEW_DATE"] = pd.to_datetime(t["VIEW_DATE"])
t["year"], t["month"] = t.VIEW_DATE.dt.year, t.VIEW_DATE.dt.month
cs = t[t.CLASSNAME.isin(SELECTIVE)]

summary = [f"Total alerts: {len(t):,}  ({t.VIEW_DATE.min().date()} -> {t.VIEW_DATE.max().date()})",
       f"Columns: {', '.join(t.columns[:-2])}", "",
       "Alerts per class:", t.CLASSNAME.value_counts().to_string(), "",
       f"Selective logging: {len(cs):,} polygons, {cs.AREAMUNKM.sum():,.0f} km2",
       f"Area of a selective-logging polygon (km2): median {cs.AREAMUNKM.median():.2f}, "
       f"min {cs.AREAMUNKM.min():.3f}, max {cs.AREAMUNKM.max():.1f}", "",
       "Satellites behind the selective-logging alerts:", cs.SATELLITE.value_counts().to_string(), "",
       "States with most selective logging:", cs.UF.value_counts().head(5).to_string()]

# ---------------------------------------------------------------- fig 1: classes
fig, ax = plt.subplots(figsize=(7, 3.6))
n = t.CLASSNAME.value_counts().sort_values()
ax.barh(n.index, n.values, color=[RED if c in SELECTIVE else GREY for c in n.index])
for y, v in enumerate(n.values):
    ax.text(v, y, f" {v:,}", va="center", fontsize=8)
ax.set_title("DETER alerts per class (red = selective logging)", loc="left", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0, n.max() * 1.15)
fig.tight_layout(); fig.savefig(OUT / "fig1_classes.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- fig 2: over time
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6))
per_year = cs.groupby(["year", "CLASSNAME"]).size().unstack(fill_value=0)
per_year.plot.bar(stacked=True, ax=a, color=[RED, BLUE], width=0.8)
a.set_title("Selective logging per year", loc="left", fontsize=11); a.set_xlabel(""); a.legend(fontsize=8, frameon=False)
per_month = cs.groupby("month").size().reindex(range(1, 13), fill_value=0) / cs.year.nunique()
b.bar(per_month.index, per_month.values, color=[RED if 6 <= m <= 11 else GREY for m in per_month.index])
b.set_xticks(range(1, 13)); b.set_xticklabels(list("JFMAMJJASOND"))
b.set_title("Mean per month of the year (red = study season)", loc="left", fontsize=11)
for x in (a, b):
    x.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "fig2_over_time.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- fig 3: polygon area
fig, ax = plt.subplots(figsize=(7, 3.6))
bins = [0.03 * 1.35 ** i for i in range(26)]
for c, col in zip(SELECTIVE, [RED, BLUE]):
    ax.hist(cs[cs.CLASSNAME == c].AREAMUNKM, bins=bins, histtype="step", lw=1.8, color=col, label=c)
ax.set_xscale("log"); ax.set_xlabel("polygon area (km2, log scale)")
ax.set_title("Size of the selective-logging polygons", loc="left", fontsize=11)
ax.legend(fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "fig3_area.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- fig 4: map of the study area (geometry is read here)
print("Reading the polygons in the study area...")
z = gpd.read_file(SHP, bbox=AOI)
z["VIEW_DATE"] = pd.to_datetime(z["VIEW_DATE"])
zs = z[z.CLASSNAME.isin(SELECTIVE) & (z.VIEW_DATE >= SEASON_START)].copy()
older = z[z.VIEW_DATE < SEASON_START]
fig, ax = plt.subplots(figsize=(7.5, 7))
older.plot(ax=ax, color="#e3e3e3")
zs["month"] = zs.VIEW_DATE.dt.strftime("%Y-%m")
zs.plot(ax=ax, column="month", cmap="plasma", legend=True, legend_kwds={"fontsize": 8, "frameon": False, "title": "seen by DETER in"})
ax.set_xlim(AOI[0], AOI[2]); ax.set_ylim(AOI[1], AOI[3])
ax.set_title(f"Study area: {len(zs)} selective-logging polygons after {SEASON_START}\n(grey = all older alerts, since 2016)", loc="left", fontsize=11)
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
fig.tight_layout(); fig.savefig(OUT / "fig4_study_area.png", dpi=150); plt.close(fig)

summary += ["", f"Study area {AOI}: {len(zs)} selective-logging polygons after {SEASON_START}",
        zs.groupby(["month", "CLASSNAME"]).size().unstack(fill_value=0).to_string(),
        f"Median polygon area in the study area: {zs.AREAMUNKM.median():.2f} km2"]
(OUT / "summary.txt").write_text("\n".join(summary))
print("\n".join(summary))
print(f"\nDone. Figures are in {OUT}/")