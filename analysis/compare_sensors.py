"""Step 4. L-band vs C-band on the SAME polygons, with one summary table and figures for the report.

Uses the files made before:
    nisar_dataset/timeseries_nisar.csv  + timeseries_nisar_neg.csv          (L-band, HH / HV, 12 days)
    s1_dataset/timeseries_s1_o97.csv    + timeseries_s1_o97_neg.csv         (C-band, VV / VH, 12 days)
    s1_dataset/timeseries_s1_o170.csv   + timeseries_s1_o170_neg.csv        (C-band, merged with o97: ~6 days)

Same method as compare_controls.py (change vs own reference, minus the regional change of the controls,
one effect per polygon = mean after DETER date - mean before), but:
  - only polygons that have a value in ALL three variants are kept (fair comparison);
  - curves show MEDIANS (the means were pulled down by a few controls with large drops);
  - controls with a large drop are listed with coordinates, to check by eye if they are unmapped
    clearings / fires (then they are not "stable forest" and should be removed).

Run (from the project folder):
    python analysis/compare_sensors.py "deter_brasilia_dataset/deter-amz-deter-public.shp"
    python analysis/compare_sensors.py "deter_brasilia_dataset/deter-amz-deter-public.shp" --exclude-controls analysis/bad_controls.txt

Writes to analysis/:
    summary_table.csv           the table for the report
    controls_large_drop.csv     controls to check by eye (lon, lat, drop in each sensor)
    figures/fig8_effects.png    per-polygon effects, logging vs control, each sensor
    figures/fig9_curves.png     median change vs days to DETER date, each sensor
"""
import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from shapely import make_valid
from shapely.affinity import translate

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
(OUT / "figures").mkdir(parents=True, exist_ok=True)

AOI = (-49.0, -4.5, -47.0, -2.5)
START = "2026-06-17"
REF_END = pd.Timestamp("2026-07-20")
MIN_VIEW = pd.Timestamp("2026-08-20")
FOREST_DB = -15.0
STAT = "p10_db"
SELECTIVE = ["CS_DESORDENADO", "CS_GEOMETRICO"]
RED, GREY = "#d1495b", "#7a7a7a"

ap = argparse.ArgumentParser()
ap.add_argument("shp")
ap.add_argument("--exclude-controls", help="text file with control ids (N012 ...) to drop, one per line")
args = ap.parse_args()
excluded = set()
if args.exclude_controls:
    excluded = {l.strip() for l in Path(args.exclude_controls).read_text().split() if l.strip()}

N, S = ROOT / "nisar_dataset", ROOT / "s1_dataset"
VARIANTS = {   # name: (logging files, control files, co-pol, cross-pol, min reference images per pol)
    "L-band 12 d (NISAR)": ([N / "timeseries_nisar.csv"], [N / "timeseries_nisar_neg.csv"], "HH", "HV", {"HH": 2, "HV": 1}),
    "C-band 12 d (S1 o97)": ([S / "timeseries_s1_o97.csv"], [S / "timeseries_s1_o97_neg.csv"], "VV", "VH", {"VV": 2, "VH": 2}),
    "C-band ~6 d (S1 o97+170)": ([S / "timeseries_s1_o97.csv", S / "timeseries_s1_o170.csv"],
                                 [S / "timeseries_s1_o97_neg.csv", S / "timeseries_s1_o170_neg.csv"], "VV", "VH",
                                 {"VV": 2, "VH": 2}),
}


def load(paths, group):
    t = pd.concat([pd.read_csv(p, parse_dates=["date", "view_date"]) for p in paths])
    if "orbit" not in t:
        t["orbit"] = 0
    t = t[(t.zone == "inside") & (t.n_pixels >= 20)]
    t = t.groupby(["poly_id", "date", "orbit", "pol", "view_date"], as_index=False)[[STAT, "median_db"]].mean()
    t["group"] = group
    return t


def signals(log_files, ctl_files, cross, min_ref):
    """Per polygon and date: change vs own reference minus the regional (control median) change."""
    ts = pd.concat([load(log_files, "logging"), load(ctl_files, "control")])
    ts = ts[~ts.poly_id.isin(excluded)]
    x_ref = ts[(ts.pol == cross) & (ts.date <= REF_END)].groupby("poly_id").median_db.mean()
    ts = ts[ts.poly_id.isin(x_ref[x_ref > FOREST_DB].index) & (ts.view_date >= MIN_VIEW)]
    ref = ts[ts.date <= REF_END].groupby(["poly_id", "orbit", "pol"])[STAT].agg(["mean", "count"])
    ref = ref[ref["count"] >= ref.index.get_level_values("pol").map(min_ref)]["mean"].rename("ref")
    ts = ts.join(ref, on=["poly_id", "orbit", "pol"], how="inner")
    ts["change"] = ts[STAT] - ts["ref"]
    regional = ts[ts.group == "control"].groupby(["date", "orbit", "pol"]).change.median().rename("regional")
    ts = ts.join(regional, on=["date", "orbit", "pol"], how="inner")
    ts["signal"] = ts.change - ts.regional
    ts["days"] = (ts.date - ts.view_date).dt.days
    return ts


def effects(ts):
    e = ts.assign(after=ts.days >= 0).groupby(["group", "pol", "poly_id", "after"]).signal.mean().unstack("after")
    e = e.dropna()
    return (e[True] - e[False]).rename("effect").reset_index()


sig, eff = {}, {}
for name, (lf, cf, co, cross, mr) in VARIANTS.items():
    sig[name] = signals(lf, cf, cross, mr)
    e = effects(sig[name])
    e["kind"] = np.where(e.pol == co, "co-pol", "cross-pol")
    eff[name] = e

# keep only polygons present in every variant (per kind of polarization)
common = {}
for kind in ["co-pol", "cross-pol"]:
    ids = [set(e[e.kind == kind].poly_id) for e in eff.values()]
    common[kind] = set.intersection(*ids)

rows = []
for name, e in eff.items():
    for kind in ["co-pol", "cross-pol"]:
        d = e[(e.kind == kind) & e.poly_id.isin(common[kind])]
        L, C = d[d.group == "logging"].effect, d[d.group == "control"].effect
        if len(L) < 5 or len(C) < 5:
            continue
        u, p = stats.mannwhitneyu(L, C)
        rows.append({"variant": name, "pol": d.pol.iloc[0], "n_logging": len(L), "n_control": len(C),
                     "logging_median_dB": round(L.median(), 2), "control_median_dB": round(C.median(), 2),
                     "AUC": round(u / (len(L) * len(C)), 2), "p_MannWhitney": round(p, 4),
                     "detected_brighter_at_5%FAR": round((L > C.quantile(0.95)).mean(), 2),
                     "detected_darker_at_5%FAR": round((L < C.quantile(0.05)).mean(), 2)})
table = pd.DataFrame(rows)
table.to_csv(OUT / "summary_table.csv", index=False)
print("Same polygons in every variant:",
      {k: f"{sum(i.startswith('P') for i in v)} logging, {sum(i.startswith('N') for i in v)} control" for k, v in common.items()})
print("\n" + table.to_string(index=False))
print("\nAUC: 0.5 = no difference; above 0.5 = logging polygons brighter than controls, below 0.5 = darker.")

# ---------------------------------------------------------------- fig 8: per-polygon effects
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
for ax, kind in zip(axes, ["co-pol", "cross-pol"]):
    data, labels, colors = [], [], []
    for name, e in eff.items():
        d = e[(e.kind == kind) & e.poly_id.isin(common[kind])]
        for group, col in [("control", GREY), ("logging", RED)]:
            data.append(d[d.group == group].effect.values)
            labels.append(f"{name.split(' (')[0]}\n{d.pol.iloc[0] if len(d) else ''} {group}")
            colors.append(col)
    bp = ax.boxplot(data, patch_artist=True, showfliers=True, widths=0.6)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.6)
    ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_title(f"{kind}: effect per polygon (after - before DETER date)", loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("dB")
fig.tight_layout(); fig.savefig(OUT / "figures" / "fig8_effects.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- fig 9: median curves, cross-pol
fig, axes = plt.subplots(1, len(sig), figsize=(5 * len(sig), 4), sharey=True)
for ax, (name, ts) in zip(axes, sig.items()):
    cross = VARIANTS[name][3]
    d = ts[(ts.pol == cross) & ts.poly_id.isin(common["cross-pol"])].copy()
    d["bin"] = (d.days // 24) * 24 + 12
    for group, col in [("control", GREY), ("logging", RED)]:
        g = d[d.group == group].groupby("bin").signal
        m = pd.DataFrame({"med": g.median(), "q25": g.quantile(.25), "q75": g.quantile(.75), "n": g.size()})
        m = m[m.n >= 5]
        ax.plot(m.index, m.med, marker="o", color=col, label=group)
        ax.fill_between(m.index, m.q25, m.q75, color=col, alpha=0.15)
    ax.axhline(0, color="grey", lw=0.6); ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_title(f"{name}, {cross}: median and 25-75%", loc="left", fontsize=10)
    ax.set_xlabel("days relative to DETER view date")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("change minus regional change (dB)"); axes[0].legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "figures" / "fig9_curves.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- controls with a large drop: where are they?
deter = gpd.read_file(args.shp, bbox=AOI)
deter["VIEW_DATE"] = pd.to_datetime(deter["VIEW_DATE"])
polys = deter[deter.CLASSNAME.isin(SELECTIVE) & (deter.VIEW_DATE >= START)].reset_index(drop=True)
polys["poly_id"] = [f"P{i:03d}" for i in range(len(polys))]
alerts = deter[deter.VIEW_DATE >= "2020-01-01"]


def make_controls(polys, alerts, n_per_poly=2, seed=42):
    """Identical to the extraction scripts, so the ids match."""
    rng = np.random.default_rng(seed)
    utm = polys.estimate_utm_crs()
    p_m, a_m = polys.to_crs(utm), alerts.to_crs(utm)
    a_m["geometry"] = a_m.geometry.apply(make_valid).buffer(0)
    idx, taken, out = a_m.sindex, [], []
    for p in p_m.itertuples():
        made = 0
        for _ in range(200):
            if made == n_per_poly:
                break
            ang, dist = rng.uniform(0, 2 * np.pi), rng.uniform(3000, 15000)
            g = translate(p.geometry, dist * np.cos(ang), dist * np.sin(ang))
            if len(idx.query(g.buffer(1000), predicate="intersects")) or any(g.intersects(t) for t in taken):
                continue
            taken.append(g); made += 1
            out.append({"poly_id": f"N{len(out):03d}", "geometry": g})
    return gpd.GeoDataFrame(out, crs=utm).to_crs(4326)


ctl = make_controls(polys, alerts)
pt = ctl.geometry.representative_point()
ctl["lon"], ctl["lat"] = pt.x.round(5), pt.y.round(5)
wide = pd.concat([e[(e.group == "control") & (e.kind == "cross-pol")].set_index("poly_id").effect.rename(n)
                  for n, e in eff.items()], axis=1)
wide = wide.join(ctl.set_index("poly_id")[["lon", "lat"]])
wide["most_negative"] = wide.drop(columns=["lon", "lat"]).min(axis=1)
bad = wide[wide.most_negative < -1.0].sort_values("most_negative")
bad.round(2).to_csv(OUT / "controls_large_drop.csv")
print(f"\nControls with a cross-pol drop > 1 dB in at least one sensor: {len(bad)} (see controls_large_drop.csv)")
print(bad.head(10).round(2).to_string())
print("\nSaved summary_table.csv, figures/fig8_effects.png, figures/fig9_curves.png")