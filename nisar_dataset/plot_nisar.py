"""Step 2c. Look at one NISAR L-band image and where the logging polygons fall on it.

Run (from the project folder):
    python step2_nisar/plot_nisar.py data/nisar/<file>.h5 "../deter_brasilia_dataset/deter-amz-deter-public.shp"

Writes step2_nisar/figures/fig2_lband_<date>.png with three panels:
    HH backscatter, HV backscatter (whole frame) and a full-resolution HV zoom on the logging area.
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
from shapely.geometry import box

H5 = Path(sys.argv[1])
SHP = sys.argv[2] if len(sys.argv) > 2 else "../deter_brasilia_dataset/deter-amz-deter-public.shp"
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

SELECTIVE = ["CS_DESORDENADO", "CS_GEOMETRICO"]
START = "2026-06-17"
GRID = "/science/LSAR/GCOV/grids/frequencyA"      # where the images live inside the HDF5 file
ZOOM_KM = 12                                      # half-size of the zoom window


def to_db(power):
    """GCOV stores gamma0 as linear power; radar people look at it in decibels."""
    power = np.asarray(power, dtype="float64")
    return 10 * np.log10(np.where(power > 0, power, np.nan))


with h5py.File(H5, "r") as f:
    g = f[GRID]
    print("Layers in the file:", [k for k in g.keys()])
    x, y = g["xCoordinates"][:], g["yCoordinates"][:]
    epsg = int(np.asarray(g["projection"][()]).item())
    posting = abs(x[1] - x[0])
    print(f"Image size: {y.size} x {x.size} pixels, pixel = {posting:.0f} m, projection EPSG:{epsg}")
    step = max(1, max(x.size, y.size) // 1500)     # read every `step`-th pixel for the overview
    hh = to_db(g["HHHH"][::step, ::step])
    hv = to_db(g["HVHV"][::step, ::step])

    # DETER selective-logging polygons that fall inside this image
    frame = box(x.min(), y.min(), x.max(), y.max())
    aoi = gpd.GeoSeries([frame], crs=epsg).to_crs(4674).total_bounds
    deter = gpd.read_file(SHP, bbox=tuple(aoi))
    deter["VIEW_DATE"] = pd.to_datetime(deter["VIEW_DATE"])
    polys = deter[deter.CLASSNAME.isin(SELECTIVE) & (deter.VIEW_DATE >= START)].to_crs(epsg)
    polys = polys[polys.within(frame)]
    print(f"Selective-logging polygons inside this image: {len(polys)}")

    # zoom: full resolution around the polygon with most neighbours
    if len(polys):
        c = polys.geometry.centroid
        neighbours = [(c.distance(p) < 15000).sum() for p in c]
        dense = c.iloc[int(np.argmax(neighbours))]
        cx, cy = dense.x, dense.y
    else:
        cx, cy = x.mean(), y.mean()
    cols = np.where(np.abs(x - cx) <= ZOOM_KM * 1000)[0]
    rows = np.where(np.abs(y - cy) <= ZOOM_KM * 1000)[0]
    zoom = to_db(g["HVHV"][rows.min():rows.max() + 1, cols.min():cols.max() + 1])
    zx, zy = x[cols], y[rows]

extent = [x.min(), x.max(), y.min(), y.max()]
fig, ax = plt.subplots(1, 3, figsize=(17, 6))
for a, img, name, lim in [(ax[0], hh, "HH", (-14, -2)), (ax[1], hv, "HV", (-20, -8))]:
    im = a.imshow(img, cmap="gray", vmin=lim[0], vmax=lim[1], extent=extent)
    if len(polys):
        polys.boundary.plot(ax=a, color="#ff3b30", lw=0.6)
    a.plot([zx.min(), zx.max(), zx.max(), zx.min(), zx.min()], [zy.min(), zy.min(), zy.max(), zy.max(), zy.min()], color="#ffd60a", lw=1)
    a.set_title(f"L-band {name} (dB), whole frame", loc="left", fontsize=11)
    fig.colorbar(im, ax=a, shrink=0.7)
im = ax[2].imshow(zoom, cmap="gray", vmin=-20, vmax=-8, extent=[zx.min(), zx.max(), zy.min(), zy.max()])
if len(polys):
    polys.boundary.plot(ax=ax[2], color="#ff3b30", lw=1)
ax[2].set_xlim(zx.min(), zx.max()); ax[2].set_ylim(zy.min(), zy.max())
ax[2].set_title(f"HV zoom ({2 * ZOOM_KM} km, full resolution); red = DETER selective logging", loc="left", fontsize=11)
fig.colorbar(im, ax=ax[2], shrink=0.7)
for a in ax:
    a.set_xticks([]); a.set_yticks([])
date = re.search(r"_(\d{8})T", H5.name)
out = OUT / f"fig2_lband_{date.group(1) if date else H5.stem}.png"
fig.tight_layout(); fig.savefig(out, dpi=130)
print("Saved", out)