"""Step 2f / 3b. Is the radar change in logging polygons different from the change in control forest?

Works for both sensors:
    --sensor nisar : nisar_dataset/timeseries_nisar.csv + timeseries_nisar_neg.csv   (L-band, HH / HV)
    --sensor s1    : s1_dataset/timeseries_s1_o<N>.csv + ..._neg.csv                 (C-band, VV / VH)
                     --orbit 97 for one orbit, --orbit 97,170 to merge orbits (each orbit gets its own reference)
For s1, only polygons that also exist in the NISAR files are used, so both sensors see the same polygons.
--thin 2 keeps every second date (Sentinel-1 at 12 days instead of 6, same as NISAR).

Why this replaces the ring: in fig6 the "around" ring often got darker by 1-2 dB (new clearings or fires
next to the polygon that DETER had not mapped yet), which made "inside minus around" positive although
nothing happened inside. Here the reference is REGIONAL: for each date, the median change of all control
polygons that are still forest. One value per polygon, so the test is honest (no repeated observations).

    change(t)   = value(t) - mean of the polygon's own images before 2026-07-20
    regional(t) = median change of the forest control polygons on the same date
    signal(t)   = change(t) - regional(t)
    effect      = mean signal after the DETER date - mean signal before it        (one number per polygon)
For a control polygon the "DETER date" is the one of the logging polygon it was copied from.

Run (from the project folder):
    python nisar_dataset/compare_controls.py                              # NISAR, 10th percentile
    python nisar_dataset/compare_controls.py --stat median_db
    python nisar_dataset/compare_controls.py --sensor s1 --orbit 97           # Sentinel-1, one orbit (12 days)
    python nisar_dataset/compare_controls.py --sensor s1 --orbit 97,170       # two orbits merged (~6 days)

Writes nisar_dataset/figures/fig7_controls_<sensor>_<stat>[_thin2].png and prints the test.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
ROOT = HERE.parent
REF_END = pd.Timestamp("2026-07-20")
MIN_VIEW = pd.Timestamp("2026-08-20")
FOREST_DB = -15.0             # cross-pol (HV or VH) median before REF_END above this = forest
RED, GREY = "#d1495b", "#7a7a7a"

ap = argparse.ArgumentParser()
ap.add_argument("--stat", default="p10_db", choices=["p10_db", "median_db"])
ap.add_argument("--sensor", default="nisar", choices=["nisar", "s1"])
ap.add_argument("--thin", type=int, default=1, help="keep every n-th date")
ap.add_argument("--orbit", default="97", help="Sentinel-1 orbit(s), e.g. 97 or 97,170")
A = ap.parse_args()
STAT = A.stat
if A.sensor == "nisar":
    FILES, CO, CROSS = (ROOT / "nisar_dataset" / "timeseries_nisar.csv", ROOT / "nisar_dataset" / "timeseries_nisar_neg.csv"), "HH", "HV"
    MIN_REF = {"HH": 2, "HV": 1}                     # HV exists only once before REF_END
else:
    ORBITS = [o.strip() for o in A.orbit.split(",")]
    FILES = ([ROOT / "s1_dataset" / f"timeseries_s1_o{o}.csv" for o in ORBITS],
             [ROOT / "s1_dataset" / f"timeseries_s1_o{o}_neg.csv" for o in ORBITS])
    CO, CROSS = "VV", "VH"
    MIN_REF = {"VV": 2, "VH": 2}
POLS = [CO, CROSS]
if A.sensor == "nisar":
    FILES = ([FILES[0]], [FILES[1]])
NAME = f"{A.sensor}{'_o' + A.orbit.replace(',', '-') if A.sensor == 's1' else ''}_{STAT}" + (f"_thin{A.thin}" if A.thin > 1 else "")


def load(paths, group):
    t = pd.concat([pd.read_csv(p, parse_dates=["date", "view_date"]) for p in paths])
    if "orbit" not in t:
        t["orbit"] = 0                                       # NISAR: one track
    t = t[(t.zone == "inside") & (t.n_pixels >= 20)]
    cols = list(dict.fromkeys([STAT, "median_db"]))          # no duplicate column when STAT is median_db
    t = t.groupby(["poly_id", "date", "orbit", "pol", "view_date"], as_index=False)[cols].mean()
    t["group"] = group
    return t


ts = pd.concat([load(FILES[0], "logging"), load(FILES[1], "control")])
if A.thin > 1:
    keep = np.sort(ts.date.unique())[::A.thin]
    ts = ts[ts.date.isin(keep)]
if A.sensor == "s1":                                   # same polygons as NISAR
    nisar_ids = set(pd.read_csv(ROOT / "nisar_dataset" / "timeseries_nisar.csv", usecols=["poly_id"]).poly_id) | \
                set(pd.read_csv(ROOT / "nisar_dataset" / "timeseries_nisar_neg.csv", usecols=["poly_id"]).poly_id)
    ts = ts[ts.poly_id.isin(nisar_ids)]
print(f"{A.sensor}: {ts.date.nunique()} dates, median gap "
      f"{np.median(np.diff(np.sort(ts.date.unique())).astype('timedelta64[D]').astype(int)):.0f} days")

# forest check on both groups: cross-pol median before REF_END must look like forest
x_ref = ts[(ts.pol == CROSS) & (ts.date <= REF_END)].groupby("poly_id").median_db.mean()
forest = set(x_ref[x_ref > FOREST_DB].index)
ts = ts[ts.poly_id.isin(forest) & (ts.view_date >= MIN_VIEW)]

ref = ts[ts.date <= REF_END].groupby(["poly_id", "orbit", "pol"])[STAT].agg(["mean", "count"])
ref = ref[ref["count"] >= ref.index.get_level_values("pol").map(MIN_REF)]["mean"].rename("ref")
ts = ts.join(ref, on=["poly_id", "orbit", "pol"], how="inner")      # each orbit vs its own reference
ts["change"] = ts[STAT] - ts["ref"]
regional = ts[ts.group == "control"].groupby(["date", "orbit", "pol"]).change.median().rename("regional")
ts = ts.join(regional, on=["date", "orbit", "pol"], how="inner")
ts["signal"] = ts.change - ts.regional
ts["days"] = (ts.date - ts.view_date).dt.days

per_poly = (ts.assign(after=ts.days >= 0).groupby(["group", "pol", "poly_id", "after"]).signal.mean()
            .unstack("after").dropna())
per_poly["effect"] = per_poly[True] - per_poly[False]

print(f"Statistic: {STAT} | forest polygons seen after {MIN_VIEW.date()}: "
      f"{ts[ts.group == 'logging'].poly_id.nunique()} logging, {ts[ts.group == 'control'].poly_id.nunique()} control\n")
for pol in POLS:
    if pol not in per_poly.index.get_level_values("pol"):
        continue
    L = per_poly.loc[("logging", pol), "effect"] if ("logging", pol) in per_poly.index.droplevel(2) else pd.Series(dtype=float)
    C = per_poly.loc[("control", pol), "effect"] if ("control", pol) in per_poly.index.droplevel(2) else pd.Series(dtype=float)
    if len(L) < 5 or len(C) < 5:
        print(f"{pol}: not enough polygons"); continue
    u, p = stats.mannwhitneyu(L, C)
    thr = C.quantile(0.05)                                   # control's 5% most negative effect
    print(f"{pol}: effect logging median {L.median():+.2f} dB (n={len(L)}), control {C.median():+.2f} dB (n={len(C)}), "
          f"Mann-Whitney p = {p:.3f}")
    print(f"    logging polygons darker than 95% of controls: {(L < thr).mean():.0%} | "
          f"brighter than 95% of controls: {(L > C.quantile(0.95)).mean():.0%}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, pol in zip(axes, POLS):
    for group, col in [("control", GREY), ("logging", RED)]:
        d = ts[(ts.group == group) & (ts.pol == pol)].copy()
        if d.empty:
            continue
        d["bin"] = (d.days // 24) * 24 + 12
        m = d.groupby("bin").signal.agg(["mean", "sem", "count"])
        m = m[m["count"] >= 5]
        ax.errorbar(m.index, m["mean"], yerr=1.96 * m["sem"], marker="o", color=col,
                    label=f"{group} ({d.poly_id.nunique()})")
    ax.axhline(0, color="grey", lw=0.6); ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_title(f"{A.sensor.upper()} {pol} ({STAT}), relative to regional forest", loc="left", fontsize=10)
    ax.set_xlabel("days relative to DETER view date (24-day bins)")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("change minus regional change (dB)"); axes[0].legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(HERE / "figures" / f"fig7_controls_{NAME}.png", dpi=150)
print(f"\nSaved figures/fig7_controls_{NAME}.png")