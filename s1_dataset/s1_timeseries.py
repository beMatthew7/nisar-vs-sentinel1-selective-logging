"""Step 3a. Sentinel-1 (C-band) time series for the SAME polygons as NISAR, from Google Earth Engine.

Nothing is downloaded: Earth Engine computes, for every polygon and every Sentinel-1 date, the median and
the 10th percentile of VV and VH (in dB) and sends back only those numbers.

The logging polygons and the control polygons are rebuilt exactly as in nisar_dataset/timeseries_nisar.py
(same filters, same random seed), so poly_id P012 or N034 means the same polygon in both sensors.

Before the first run:
    pip install earthengine-api
    earthengine authenticate

Run (from the project folder), first only to see which orbits exist:
    python s1_dataset/s1_timeseries.py "deter_brasilia_dataset/deter-amz-deter-public.shp" --project <ID> --list-orbits
then with the chosen orbit:
    python s1_dataset/s1_timeseries.py "deter_brasilia_dataset/deter-amz-deter-public.shp" --project <ID> --orbit <N>

Writes s1_dataset/timeseries_s1_o<N>.csv and s1_dataset/timeseries_s1_o<N>_neg.csv (same columns as the
NISAR files, plus the orbit). Run it once per orbit you want to use.
"""
import argparse
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import make_valid
from shapely.affinity import translate

HERE = Path(__file__).parent
AOI = (-49.0, -4.5, -47.0, -2.5)
START = "2026-06-17"                 # same start as NISAR
END = "2026-10-31"
SELECTIVE = ["CS_DESORDENADO", "CS_GEOMETRICO"]

ap = argparse.ArgumentParser()
ap.add_argument("shp")
ap.add_argument("--project", required=True, help="your Google Cloud project ID (not its display name)")
ap.add_argument("--orbit", type=int, help="Sentinel-1 relative orbit to use")
ap.add_argument("--list-orbits", action="store_true")
args = ap.parse_args()

ee.Initialize(project=args.project)
aoi = ee.Geometry.Rectangle(list(AOI))
col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi).filterDate(START, END)
       .filter(ee.Filter.eq("instrumentMode", "IW"))
       .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))

# ---------------------------------------------------------------- 1. which orbits pass over the study area
info = col.reduceColumns(ee.Reducer.toList(4), ["relativeOrbitNumber_start", "orbitProperties_pass",
                                                "platform_number", "system:time_start"]).get("list").getInfo()
meta = pd.DataFrame(info, columns=["orbit", "pass", "platform", "time"])
meta["date"] = pd.to_datetime(meta.time, unit="ms").dt.normalize()
summary = (meta.groupby(["orbit", "pass"])
           .agg(dates=("date", "nunique"), platforms=("platform", lambda s: "".join(sorted(set(s)))),
                first=("date", "min"), last=("date", "max")).reset_index())
summary["median_gap_days"] = [np.median(np.diff(np.sort(meta[meta.orbit == o].date.unique())).astype("timedelta64[D]").astype(int))
                              for o in summary.orbit]
print("Sentinel-1 orbits over the study area:")
print(summary.sort_values("dates", ascending=False).to_string(index=False))
if args.list_orbits or args.orbit is None:
    raise SystemExit("\nPick the orbit with most dates (ideally descending, like NISAR 154 D) and run again with --orbit N.")

# ---------------------------------------------------------------- 2. rebuild the same polygons as for NISAR
deter = gpd.read_file(args.shp, bbox=AOI)
deter["VIEW_DATE"] = pd.to_datetime(deter["VIEW_DATE"])
polys = deter[deter.CLASSNAME.isin(SELECTIVE) & (deter.VIEW_DATE >= START)].reset_index(drop=True)
polys["poly_id"] = [f"P{i:03d}" for i in range(len(polys))]
other_alerts = deter[deter.VIEW_DATE >= "2020-01-01"]


def make_controls(polys, alerts, n_per_poly=2, seed=42):
    """Identical to timeseries_nisar.py: move each polygon 3-15 km to a spot >= 1 km from any alert."""
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
            out.append({"poly_id": f"N{len(out):03d}", "source": p.poly_id, "VIEW_DATE": p.VIEW_DATE,
                        "CLASSNAME": "CONTROL", "AREAMUNKM": p.AREAMUNKM, "geometry": g})
    return gpd.GeoDataFrame(out, crs=utm).to_crs(polys.crs)


controls = make_controls(polys, other_alerts)
print(f"\nLogging polygons: {len(polys)} | control polygons: {len(controls)}")


def to_ee(gdf):
    """Send the polygons to Earth Engine (simplified to 5 m to keep the request small)."""
    g = gdf[["poly_id", "geometry"]].copy()
    utm = g.estimate_utm_crs()
    g["geometry"] = g.to_crs(utm).geometry.apply(make_valid).buffer(0).simplify(5).to_crs(4326).values
    return ee.FeatureCollection(g.__geo_interface__)


# ---------------------------------------------------------------- 3. one mosaic per date, stats per polygon
orbit_col = col.filter(ee.Filter.eq("relativeOrbitNumber_start", args.orbit))
dates = sorted(meta[meta.orbit == args.orbit].date.unique())
reducer = (ee.Reducer.median().combine(ee.Reducer.percentile([10]), sharedInputs=True)
           .combine(ee.Reducer.count(), sharedInputs=True))

for name, gdf in [(f"timeseries_s1_o{args.orbit}.csv", polys), (f"timeseries_s1_o{args.orbit}_neg.csv", controls)]:
    fc = to_ee(gdf)
    attrs = gdf.set_index("poly_id")
    rows = []
    for d in dates:
        day = pd.Timestamp(d)
        img = (orbit_col.filterDate(str(day.date()), str((day + pd.Timedelta(days=1)).date()))
               .mosaic().select(["VV", "VH"]))                       # GRD in Earth Engine is already in dB
        feats = img.reduceRegions(collection=fc, reducer=reducer, scale=20, tileScale=4).getInfo()["features"]
        for f in feats:
            p = f["properties"]
            a = attrs.loc[p["poly_id"]]
            for pol in ["VV", "VH"]:
                rows.append({"poly_id": p["poly_id"], "date": day.date(), "orbit": args.orbit, "pol": pol, "zone": "inside",
                             "median_db": p.get(f"{pol}_median"), "p10_db": p.get(f"{pol}_p10"),
                             "n_pixels": p.get(f"{pol}_count", 0),
                             "view_date": a.VIEW_DATE.date(), "classname": a.CLASSNAME, "area_km2": a.AREAMUNKM})
        print(f"{name}: {day.date()} done")
    pd.DataFrame(rows).dropna(subset=["median_db"]).to_csv(HERE / name, index=False)
    print(f"Saved {HERE / name}")