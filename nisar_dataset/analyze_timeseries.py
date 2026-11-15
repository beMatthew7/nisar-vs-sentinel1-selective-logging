"""Step 2e. A fairer look at the L-band time series (uses timeseries_nisar.csv, no new downloads).

Why: in fig3 the polygon is ~1.4 dB *brighter* than the ring around it at all times. That offset says
nothing about logging - it comes from the ring containing old clearings, roads and pasture (dark pixels),
which pull its 10th percentile down. So instead of comparing places, we compare each place WITH ITSELF:

    change(t)  = value(t) - mean(value before 2026-07-20)          for inside and for around
    signal(t)  = change_inside(t) - change_around(t)               ("difference in differences")

The ring still removes effects that hit both (rain, season, calibration), but its constant offset is gone.
Only polygons first seen by DETER after 2026-08-20 are used, so each has at least 3 images before logging.

Run (from the project folder):
    python nisar_dataset/analyze_timeseries.py                         # median, as before
    python nisar_dataset/analyze_timeseries.py --stat p10_db           # darkest 10% of each polygon
    python nisar_dataset/analyze_timeseries.py --stat p10_db --tag p10 # save figures with a suffix

Writes:
    nisar_dataset/figures/fig4_change_by_date.png    signal per calendar date, early vs late polygons
    nisar_dataset/figures/fig5_change_vs_deter.png   signal by days relative to the DETER date
    nisar_dataset/figures/fig6_examples.png          12 single polygons, to see what one logging event looks like
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
FIG = HERE / "figures"
REF_END = pd.Timestamp("2026-07-20")         # images before this date = "before logging" reference
MIN_VIEW = pd.Timestamp("2026-08-20")        # keep polygons DETER saw after this date
ap = argparse.ArgumentParser()
ap.add_argument("--stat", default="median_db", choices=["median_db", "p10_db"],
                help="median_db = whole polygon; p10_db = darkest 10%% (closer to the actual gaps)")
ap.add_argument("--tag", default="", help="suffix added to the figure names")
args = ap.parse_args()
STAT = args.stat
TAG = f"_{args.tag}" if args.tag else ""
RED, GREY, BLUE = "#d1495b", "#7a7a7a", "#2e86ab"

ts = pd.read_csv(HERE / "timeseries_nisar.csv", parse_dates=["date", "view_date"])
ts = ts.groupby(["poly_id", "date", "pol", "zone", "view_date", "classname", "area_km2"],
                as_index=False)[[STAT, "n_pixels"]].mean()                  # two files on the same date -> mean
ts = ts[ts.n_pixels >= 20]                                                  # need enough pixels in each zone

ref = ts[ts.date <= REF_END].groupby(["poly_id", "pol", "zone"])[STAT].agg(["mean", "count"])
MIN_REF = {"HH": 2, "HV": 1}                 # HV exists only on 2026-07-06 before the reference end
ref = ref[ref["count"] >= ref.index.get_level_values("pol").map(MIN_REF)]["mean"].rename("ref")
ts = ts.join(ref, on=["poly_id", "pol", "zone"], how="inner")
ts["change"] = ts[STAT] - ts["ref"]

w = ts.pivot_table(index=["poly_id", "date", "pol", "view_date", "classname", "area_km2"],
                   columns="zone", values="change").dropna().reset_index()
w["signal"] = w["inside"] - w["around"]
w["days"] = (w.date - w.view_date).dt.days
late = w[w.view_date >= MIN_VIEW]
print(f"Polygons with a valid reference: {w.poly_id.nunique()} | seen by DETER after {MIN_VIEW.date()}: {late.poly_id.nunique()}")


def mean_ci(df, by, col):
    m = df.groupby(by)[col].agg(["mean", "sem", "count"])
    return m[m["count"] >= 5]


# ---------------------------------------------------------------- fig 4: by calendar date
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, pol in zip(axes, ["HH", "HV"]):
    d = w[w.pol == pol]
    for name, sub, col in [("seen by DETER after Aug 20", d[d.view_date >= MIN_VIEW], RED),
                           ("seen by DETER before Aug 20", d[d.view_date < MIN_VIEW], GREY)]:
        m = mean_ci(sub, "date", "signal")
        ax.errorbar(m.index, m["mean"], yerr=1.96 * m["sem"], marker="o", color=col, label=f"{name} ({sub.poly_id.nunique()})")
    ax.axhline(0, color="grey", lw=0.6); ax.axvline(REF_END, color="k", ls=":", lw=0.8)
    ax.set_title(f"L-band {pol}: change inside minus change around", loc="left", fontsize=10)
    ax.tick_params(axis="x", rotation=45); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("dB (0 = same change as the forest around)")
axes[0].legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(FIG / f"fig4_change_by_date{TAG}.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- fig 5: relative to DETER date
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, pol in zip(axes, ["HH", "HV"]):
    d = late[late.pol == pol].copy()
    d["bin"] = (d.days // 24) * 24 + 12
    m = mean_ci(d, "bin", "signal")
    ax.errorbar(m.index, m["mean"], yerr=1.96 * m["sem"], marker="o", color=RED)
    for b, row in m.iterrows():
        ax.annotate(int(row["count"]), (b, row["mean"]), textcoords="offset points", xytext=(0, 8), fontsize=7, ha="center")
    ax.axhline(0, color="grey", lw=0.6); ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_title(f"L-band {pol}: polygons seen after Aug 20", loc="left", fontsize=10)
    ax.set_xlabel("days relative to DETER view date (24-day bins; numbers = observations)")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("change inside minus change around (dB)")
fig.tight_layout(); fig.savefig(FIG / f"fig5_change_vs_deter{TAG}.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- fig 6: single polygons
hh = late[late.pol == "HH"]
final = hh.sort_values("date").groupby("poly_id").signal.last().sort_values()
pick = list(final.index[:6]) + list(final.sample(min(6, len(final)), random_state=0).index)
fig, axes = plt.subplots(3, 4, figsize=(13, 7.5), sharey=True)
for ax, pid in zip(axes.flat, pick):
    p = ts[(ts.poly_id == pid) & (ts.pol == "HH")].sort_values("date")
    for zone, col in [("inside", RED), ("around", GREY)]:
        z = p[p.zone == zone]
        ax.plot(z.date, z.change, marker="o", ms=3, color=col, label=zone)
    vd = p.view_date.iloc[0]
    ax.axvline(vd, color="k", ls="--", lw=0.8)
    ax.set_title(f"{pid} {p.classname.iloc[0][3:6]} {p.area_km2.iloc[0]:.1f} km2", fontsize=8, loc="left")
    ax.tick_params(axis="x", labelsize=6, rotation=45); ax.axhline(0, color="grey", lw=0.5)
axes[0, 0].legend(fontsize=7, frameon=False); axes[0, 0].set_ylabel("HH change (dB)")
fig.suptitle("Single polygons (HH). Top row: strongest drops; others: random. Dashed = DETER date", fontsize=10, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(FIG / f"fig6_examples{TAG}.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- numbers for the report
summary = late.assign(after=late.days >= 0).groupby(["pol", "after"]).signal.agg(["mean", "sem", "count"]).round(3)
print("\nSignal before (after=False) and after (after=True) the DETER date, polygons seen after Aug 20:")
print(summary.to_string())
# simple test: is the mean signal after the DETER date different from before? (Welch t-test)
from scipy import stats
for pol, d in late.groupby("pol"):
    b, a = d[d.days < 0].signal, d[d.days >= 0].signal
    if len(b) > 4 and len(a) > 4:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        print(f"{pol}: after - before = {a.mean() - b.mean():+.2f} dB, p = {p:.3f} ({d.poly_id.nunique()} polygons)")
print(f"\nStatistic: {STAT}. Saved fig4_change_by_date{TAG}.png, fig5_change_vs_deter{TAG}.png, fig6_examples{TAG}.png")