"""Step 2d. L-band time series inside the DETER logging polygons, compared with the forest around them.

For every selective-logging polygon (seen by DETER since 2026-06-17) and every NISAR file in data/nisar/:
  - inside  = pixels inside the polygon
  - around  = pixels in a ring 300-1000 m around it (local "control" forest)
We store the median and the 10th percentile (the darkest part) in dB, for HH and HV.
Pixels flagged as bad in the NISAR `mask` layer are ignored.

Run (from the project folder):
    python nisar_dataset/timeseries_nisar.py "deter_brasilia_dataset/deter-amz-deter-public.shp"

Writes:
    nisar_dataset/timeseries_nisar.csv                one row per polygon x date x zone
    nisar_dataset/figures/fig3_timeseries.png         mean difference inside - around, by days to DETER date
"""
import re
import sys
from pathlib import Path

import geopandas as gpd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from shapely import make_valid
from shapely.geometry import box

SHP = sys.argv[1] if len(sys.argv) > 1 else "deter_brasilia_dataset/deter-amz-deter-public.shp"
HERE = Path(__file__).parent
FILES = sorted(Path("data/nisar").glob("*.h5"))
GRID = "/science/LSAR/GCOV/grids/frequencyA"
AOI = (-49.0, -4.5, -47.0, -2.5)
START = "2026-06-17"
SELECTIVE = ["CS_DESORDENADO", "CS_GEOMETRICO"]
RING = (300, 1000)                                  # metres around the polygon used as control


def to_db(power):
    return 10 * np.log10(np.where(power > 0, power, np.nan))


def stats(values):
    v = values[np.isfinite(values)]
    if v.size < 5:
        return np.nan, np.nan, v.size
    return np.median(v), np.percentile(v, 10), v.size


if not FILES:
    raise SystemExit("No .h5 files in data/nisar/. Download them first.")

deter = gpd.read_file(SHP, bbox=AOI)
deter["VIEW_DATE"] = pd.to_datetime(deter["VIEW_DATE"])
polys = deter[deter.CLASSNAME.isin(SELECTIVE) & (deter.VIEW_DATE >= START)].reset_index(drop=True)
polys["poly_id"] = [f"P{i:03d}" for i in range(len(polys))]
other_alerts = deter[deter.VIEW_DATE >= "2020-01-01"]          # rings must not touch any recent alert

rows, broken = [], []
for f in FILES:
    date = pd.Timestamp(re.search(r"_(\d{8})T", f.name).group(1))
    try:
        h5 = h5py.File(f, "r")
    except OSError as e:                                    # e.g. a download that stopped half way
        print(f"SKIPPED (cannot open, probably incomplete download): {f.name}\n    {e}")
        broken.append(f.name)
        continue
    with h5:
        g = h5[GRID]
        x, y = g["xCoordinates"][:], g["yCoordinates"][:]
        epsg = int(np.asarray(g["projection"][()]).item())
        dx, dy = x[1] - x[0], y[1] - y[0]
        pol_m = polys.to_crs(epsg)
        pol_m["geometry"] = pol_m.geometry.apply(make_valid).buffer(0)       # DETER has some broken polygons
        alerts_m = other_alerts.to_crs(epsg)
        alerts_m["geometry"] = alerts_m.geometry.apply(make_valid).buffer(0)
        alerts_index = alerts_m.sindex
        layers = [k for k in ("HHHH", "HVHV") if k in g]
        print(f"{date.date()}  layers: {layers}")
        for p in pol_m.itertuples():
            ring = p.geometry.buffer(RING[1]).difference(p.geometry.buffer(RING[0]))
            for k in alerts_index.query(ring, predicate="intersects"):     # cut out nearby alerts, one by one
                try:
                    ring = ring.difference(alerts_m.geometry.iloc[k].buffer(50))
                except Exception:
                    pass                                                       # skip a geometry GEOS cannot handle
            x0, y0, x1, y1 = p.geometry.buffer(RING[1] + 50).bounds
            c0, c1 = np.searchsorted(x, [x0, x1])
            r0, r1 = np.searchsorted(-y, [-y1, -y0])                  # y decreases down the image
            if c1 - c0 < 3 or r1 - r0 < 3:
                continue                                              # polygon outside this image
            tr = from_origin(x[c0] - dx / 2, y[r0] - dy / 2, dx, -dy)
            shape_ = (r1 - r0, c1 - c0)
            inside = geometry_mask([p.geometry], shape_, tr, invert=True)
            around = geometry_mask([ring], shape_, tr, invert=True) if not ring.is_empty else np.zeros(shape_, bool)
            bad = np.zeros(shape_, bool)
            if "mask" in g:
                m = g["mask"][r0:r1, c0:c1]
                bad = (m == 0) | (m == 255)                           # 0 = no data, 255 = fill (see product spec)
            for layer in layers:
                img = to_db(g[layer][r0:r1, c0:c1].astype("float64"))
                img[bad] = np.nan
                for zone, sel in (("inside", inside), ("around", around)):
                    med, p10, n = stats(img[sel])
                    rows.append({"poly_id": p.poly_id, "date": date, "pol": layer[:2], "zone": zone,
                                 "median_db": med, "p10_db": p10, "n_pixels": n,
                                 "view_date": p.VIEW_DATE, "classname": p.CLASSNAME, "area_km2": p.AREAMUNKM})

if broken:
    print("\nIncomplete files (delete them and download again):")
    for b in broken:
        print("   data/nisar/" + b)
ts = pd.DataFrame(rows)
ts.to_csv(HERE / "timeseries_nisar.csv", index=False)
print(f"Saved {len(ts)} rows for {ts.poly_id.nunique()} polygons and {ts.date.nunique()} dates.")

# ---------------------------------------------------------------- figure: inside minus around, by days to DETER
wide = ts.pivot_table(index=["poly_id", "date", "pol", "view_date"], columns="zone", values="p10_db").dropna().reset_index()
wide["diff"] = wide["inside"] - wide["around"]
wide["days"] = (wide.date - wide.view_date).dt.days
wide["bin"] = (wide.days // 24) * 24 + 12
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, pol in zip(axes, ["HH", "HV"]):
    w = wide[wide.pol == pol]
    if w.empty:
        ax.set_title(f"{pol}: no data"); continue
    m = w.groupby("bin")["diff"].agg(["mean", "sem", "count"])
    m = m[m["count"] >= 5]
    ax.errorbar(m.index, m["mean"], yerr=1.96 * m["sem"], marker="o", color="#d1495b")
    ax.axvline(0, color="k", ls="--", lw=0.8); ax.axhline(0, color="grey", lw=0.6)
    ax.set_title(f"L-band {pol}: logging polygon minus surrounding forest", loc="left", fontsize=10)
    ax.set_xlabel("days relative to DETER view date (24-day bins)")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("difference of 10th percentiles (dB)")
fig.tight_layout(); fig.savefig(HERE / "figures" / "fig3_timeseries.png", dpi=150)
print("Saved figures/fig3_timeseries.png")