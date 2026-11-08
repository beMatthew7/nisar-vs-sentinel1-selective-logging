"""Step 2a. Find which NISAR L-band scenes (GCOV product) cover the study area.

Nothing is downloaded here. No account is needed for searching.

Run (from the project folder):
    python step2_nisar/search_nisar.py "../deter_brasilia_dataset/deter-amz-deter-public.shp"

Writes to step2_nisar/:
    scenes.csv                    one row per scene (date, track, frame, polarization, size, url)
    figures/fig1_footprints.png   scene footprints drawn over the DETER selective-logging polygons
and prints a table that tells you which track/frame to download.
"""
import re
import sys
from pathlib import Path

import asf_search as asf
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import box, shape

SHP = sys.argv[1] if len(sys.argv) > 1 else "../deter_brasilia_dataset/deter-amz-deter-public.shp"
HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)

AOI = (-49.0, -4.5, -47.0, -2.5)                 # lon_min, lat_min, lon_max, lat_max (Paragominas)
START, END = "2026-06-17", "2026-11-12"          # calibrated NISAR data start on 2026-06-17
SELECTIVE = ["CS_DESORDENADO", "CS_GEOMETRICO"]

# Scene names look like:
# NISAR_L2_PR_GCOV_033_117_A_176_2005_DHDH_A_20261020T084803_..._001
#                  cycle track dir frame mode pol     start time
NAME = re.compile(r"GCOV_(\d{3})_(\d{3})_([AD])_(\d{3})_(\w{4})_(\w{4})_\w_(\d{8})T")

# ---------------------------------------------------------------- 1. search the ASF catalogue
print("Searching ASF for NISAR GCOV scenes...")
results = asf.search(dataset=["NISAR"], processingLevel=["GCOV"], intersectsWith=box(*AOI).wkt,
                     start=START, end=END, maxResults=2000)
urls = {u.rsplit("/", 1)[-1][:-3]: u for u in results.find_urls(extension=".h5") if "QA_STATS" not in u}

def h5_size_gb(props, name):
    """Size of the .h5 file in GB. ASF gives `bytes` either as a number or as a dict per file."""
    b = props.get("bytes")
    if isinstance(b, dict):
        sizes = [v.get("bytes", 0) if isinstance(v, dict) else v for k, v in b.items() if str(k).endswith(".h5")]
        b = sizes[0] if sizes else 0
    try:
        return round(float(b or 0) / 1e9, 2)
    except (TypeError, ValueError):
        return float("nan")


rows = []
for r in results:
    name = r.properties.get("sceneName", "")
    m = NAME.search(name)
    if not m or name not in urls:
        continue
    cycle, track, direction, frame, mode, pol, day = m.groups()
    rows.append({"scene": name, "date": pd.Timestamp(day), "track": int(track), "direction": direction,
                 "frame": int(frame), "mode": mode, "pol": pol,
                 "dual_pol": pol[0] in "DQ",                    # D = HH+HV, Q = quad-pol, S = HH only
                 "size_gb": h5_size_gb(r.properties, name),
                 "url": urls[name], "geometry": shape(r.geometry)})
scenes = gpd.GeoDataFrame(rows, crs=4326).drop_duplicates("scene").sort_values("date")
scenes.drop(columns="geometry").to_csv(HERE / "scenes.csv", index=False)
print(f"Found {len(scenes)} scenes, {int(scenes.dual_pol.sum())} of them dual-pol (HH + HV).")

# ---------------------------------------------------------------- 2. how many logging polygons does each frame cover?
deter = gpd.read_file(SHP, bbox=AOI)
deter["VIEW_DATE"] = pd.to_datetime(deter["VIEW_DATE"])
logging_polys = deter[deter.CLASSNAME.isin(SELECTIVE) & (deter.VIEW_DATE >= START)].to_crs(4326)
centroids = logging_polys.geometry.representative_point()

table = []
for (track, direction, frame), g in scenes.groupby(["track", "direction", "frame"]):
    dual = g[g.dual_pol]
    table.append({"track": track, "dir": direction, "frame": frame, "scenes": len(g), "dual_pol": len(dual),
                  "first": g.date.min().date(), "last": g.date.max().date(),
                  "GB_dual": round(dual.size_gb.sum(), 1),
                  "logging_polygons": int(centroids.within(g.geometry.iloc[0]).sum())})
table = pd.DataFrame(table).sort_values(["logging_polygons", "dual_pol"], ascending=False)
print(f"\nSelective-logging polygons in the study area since {START}: {len(logging_polys)}\n")
print(table.to_string(index=False))
table.to_csv(HERE / "coverage.csv", index=False)

# ---------------------------------------------------------------- 3. map
fig, ax = plt.subplots(figsize=(8, 7))
colors = plt.cm.tab10.colors
for i, ((track, direction, frame), g) in enumerate(scenes.groupby(["track", "direction", "frame"])):
    gpd.GeoSeries([g.geometry.iloc[0]]).boundary.plot(ax=ax, color=colors[i % 10], lw=1.6,
                                                     label=f"{track}{direction} frame {frame}")
logging_polys.plot(ax=ax, color="#d1495b")
gpd.GeoSeries([box(*AOI)]).boundary.plot(ax=ax, color="k", ls="--", lw=1)
ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
ax.set_title("NISAR GCOV footprints (lines) and DETER selective logging (red)", loc="left", fontsize=11)
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
fig.tight_layout(); fig.savefig(HERE / "figures" / "fig1_footprints.png", dpi=150)
print("\nSaved scenes.csv, coverage.csv and figures/fig1_footprints.png")
print("Pick the row with many logging polygons AND many dual-pol scenes; that is the track/frame to download.")