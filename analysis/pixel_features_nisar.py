"""Pixel-level before/after features from NISAR L-band (HH, HV), one row per polygon.

Follows the variable set of Flores-Anderson et al. (2026): backscatter before / after / difference,
temporal coefficient of variation (CoV) and the Radar Forest Degradation Index (RFDI), all computed in
POWER scale; dB is used only for the final differences. Added here (selective logging affects only part
of a polygon): the share of pixels with a strong drop or rise.

For each polygon (logging or control):
  before = images up to 2026-07-20, after = images on or after the polygon's DETER view date
  per pixel:   B = temporal mean power before, A = temporal mean power after   (3x3 boxcar on both)
               d = 10*log10(A / B)
  forest pixels only: 5x5-smoothed HV of 2026-07-06 above -14.5 dB (radar forest mask, before logging)

Run (from the project folder):
    python analysis/pixel_features_nisar.py "deter_brasilia_dataset/deter-amz-deter-public.shp"
Writes analysis/features_nisar.csv
"""
import re
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from scipy.ndimage import uniform_filter
from shapely import make_valid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REF_END, load_polygons

warnings.filterwarnings("ignore", category=RuntimeWarning)      # all-NaN windows outside the swath
SHP = sys.argv[1]
OUT = Path(__file__).resolve().parent
FILES = sorted(Path("data/nisar").glob("*_2005_*.h5"))          # 20 MHz mode only (20 m pixels)
GRID = "/science/LSAR/GCOV/grids/frequencyA"
MIN_VIEW = pd.Timestamp("2026-08-01")
FOREST_HV_DB = -14.5
THR_DB = 1.5                                                    # "strong" change of a pixel
PAD = 5                                                         # pixels read around the polygon bbox


def db(p):
    return 10 * np.log10(p)


def smooth(a, n=3):
    """Boxcar mean that ignores NaN."""
    ok = np.isfinite(a)
    s = uniform_filter(np.where(ok, a, 0.0), n, mode="nearest")
    w = uniform_filter(ok.astype(float), n, mode="nearest")
    return np.where(w > 0.5, s / np.maximum(w, 1e-9), np.nan)


logging, controls = load_polygons(SHP)
polys = pd.concat([logging, controls], ignore_index=True)
polys = polys[polys.VIEW_DATE >= MIN_VIEW].reset_index(drop=True)
print(f"Polygons seen by DETER after {MIN_VIEW.date()}: {(polys.CLASSNAME != 'CONTROL').sum()} logging, "
      f"{(polys.CLASSNAME == 'CONTROL').sum()} control")

# ---------------------------------------------------------------- 1. read a small window per polygon and date
stacks = {pid: {"HH": {}, "HV": {}} for pid in polys.poly_id}
windows, centers = {}, {}
for f in FILES:
    date = pd.Timestamp(re.search(r"_(\d{8})T", f.name).group(1))
    try:
        h5 = h5py.File(f, "r")
    except OSError:
        print("skip (cannot open):", f.name); continue
    with h5:
        g = h5[GRID]
        x, y = g["xCoordinates"][:], g["yCoordinates"][:]
        epsg = int(np.asarray(g["projection"][()]).item())
        if not windows:
            pm = polys.to_crs(epsg)
            pm["geometry"] = pm.geometry.apply(make_valid).buffer(0)
            for p in pm.itertuples():
                x0, y0, x1, y1 = p.geometry.bounds
                c0, c1 = np.searchsorted(x, [x0, x1]); r0, r1 = np.searchsorted(-y, [-y1, -y0])
                c0, r0, c1, r1 = c0 - PAD, r0 - PAD, c1 + PAD, r1 + PAD
                if c0 < 0 or r0 < 0 or c1 > x.size or r1 > y.size:
                    continue                                              # polygon not fully inside the frame
                tr = from_origin(x[c0] - 10, y[r0] + 10, 20, 20)
                inside = geometry_mask([p.geometry], (r1 - r0, c1 - c0), tr, invert=True)
                windows[p.poly_id] = (r0, r1, c0, c1, inside)
                centers[p.poly_id] = (p.geometry.centroid.x, p.geometry.centroid.y)
        for pid, (r0, r1, c0, c1, _) in windows.items():
            m = g["mask"][r0:r1, c0:c1]
            for layer, pol in (("HHHH", "HH"), ("HVHV", "HV")):
                if layer not in g:
                    continue
                a = g[layer][r0:r1, c0:c1].astype("float64")
                a[(m == 0) | (m == 255) | ~np.isfinite(a) | (a <= 0)] = np.nan
                prev = stacks[pid][pol].get(date)                         # two files on one date -> mean
                stacks[pid][pol][date] = a if prev is None else np.nanmean([prev, a], axis=0)
    print(date.date(), "read")

# ---------------------------------------------------------------- 2. features per polygon
meta = polys.set_index("poly_id")
rows = []
for pid, (r0, r1, c0, c1, inside) in windows.items():
    view = meta.loc[pid, "VIEW_DATE"]
    row = {"poly_id": pid, "label": int(pid.startswith("P")), "view_date": view.date(),
           "classname": meta.loc[pid, "CLASSNAME"], "area_km2": meta.loc[pid, "AREAMUNKM"]}
    hv_ref = [a for d, a in stacks[pid]["HV"].items() if d <= REF_END]
    if not hv_ref:
        continue
    forest = db(smooth(np.nanmean(hv_ref, axis=0), 5)) > FOREST_HV_DB
    sel = inside & forest
    row["x_m"], row["y_m"] = centers[pid]
    row["forest_frac"] = forest[inside].mean()
    row["n_pixels"] = int(sel.sum())
    if sel.sum() < 100:
        continue
    means = {}
    for pol in ("HH", "HV"):
        bef = [a for d, a in stacks[pid][pol].items() if d <= REF_END]
        aft = [a for d, a in stacks[pid][pol].items() if d >= view]
        row[f"{pol}_n_bef"], row[f"{pol}_n_aft"] = len(bef), len(aft)
        if not bef or not aft:
            continue
        B, A = smooth(np.nanmean(bef, axis=0)), smooth(np.nanmean(aft, axis=0))
        means[pol] = (B, A)
        d = db(A / B)[sel]
        d = d[np.isfinite(d)]
        if d.size < 100:
            continue                                                      # polygon outside the radar swath
        row[f"{pol}_bef"], row[f"{pol}_aft"] = db(np.nanmedian(B[sel])), db(np.nanmedian(A[sel]))
        row[f"{pol}_diff"] = np.median(d)
        row[f"{pol}_diff_p10"], row[f"{pol}_diff_p90"] = np.percentile(d, [10, 90])
        row[f"{pol}_diff_std"] = d.std()
        row[f"{pol}_frac_drop"], row[f"{pol}_frac_rise"] = (d < -THR_DB).mean(), (d > THR_DB).mean()
        for name, st in (("bef", bef), ("aft", aft)):                     # temporal CoV needs >= 2 images
            if len(st) >= 2:
                cov = np.nanstd(st, axis=0) / np.nanmean(st, axis=0) * 100
                row[f"{pol}cov_{name}"] = np.nanmedian(cov[sel])
        if f"{pol}cov_bef" in row and f"{pol}cov_aft" in row:
            row[f"{pol}cov_diff"] = row[f"{pol}cov_aft"] - row[f"{pol}cov_bef"]
    # placebo: change between two BEFORE periods (first image vs the other reference images). If logging
    # polygons already differ from controls here, the change started before our first image.
    hh_ref = sorted(d for d in stacks[pid]["HH"] if d <= REF_END)
    if len(hh_ref) >= 3:
        p0 = smooth(stacks[pid]["HH"][hh_ref[0]])
        p1 = smooth(np.nanmean([stacks[pid]["HH"][d] for d in hh_ref[1:]], axis=0))
        row["HH_placebo"] = np.nanmedian(db(p1 / p0)[sel])
    if "HH" in means and "HV" in means:                                   # RFDI on dates that have both channels
        hv_dates = set(stacks[pid]["HV"])
        hb = smooth(np.nanmean([stacks[pid]["HH"][d] for d in hv_dates if d <= REF_END], axis=0))
        ha = [stacks[pid]["HH"][d] for d in hv_dates if d >= view and d in stacks[pid]["HH"]]
        if ha:
            ha = smooth(np.nanmean(ha, axis=0))
            vb, va = means["HV"]
            rb, ra = (hb - vb) / (hb + vb), (ha - va) / (ha + va)
            row["RFDI_bef"], row["RFDI_aft"] = np.nanmedian(rb[sel]), np.nanmedian(ra[sel])
            row["RFDI_diff"] = np.nanmedian((ra - rb)[sel])
            row["RFDI_diff_p90"] = np.nanpercentile((ra - rb)[sel], 90)
    if "HH_diff" in row:
        rows.append(row)

feat = pd.DataFrame(rows)
feat.to_csv(OUT / "features_nisar.csv", index=False)
print(f"Saved {OUT / 'features_nisar.csv'}: {len(feat)} polygons "
      f"({feat.label.sum()} logging, {(feat.label == 0).sum()} control)")
