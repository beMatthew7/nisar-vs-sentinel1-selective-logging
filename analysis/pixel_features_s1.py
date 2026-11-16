"""Pixel-level before/after features from Sentinel-1 C-band (VV, VH), one row per polygon, via Earth Engine.

Mirror of pixel_features_nisar.py, so the two sensors can be compared feature by feature:
  same polygons (ids from features_nisar.csv), same "before" period (up to 2026-07-20), same "after"
  rule (images on/after the polygon's DETER view date), 20 m grid in the same UTM zone, 3x3 boxcar,
  power scale (collection COPERNICUS/S1_GRD_FLOAT), same +-1.5 dB threshold, mRFDI instead of RFDI.
Only orbit 97 (descending, 12 days, one day before each NISAR pass) is used.

Run (from the project folder):
    python analysis/pixel_features_s1.py "deter_brasilia_dataset/deter-amz-deter-public.shp" --project <ID>
Writes analysis/features_s1.csv
"""
import argparse
import sys
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import make_valid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AOI, REF_END, START, load_polygons

OUT = Path(__file__).resolve().parent
END = "2026-10-31"
CRS = "EPSG:32722"                 # same UTM zone as the NISAR frame
FOREST_VH_DB = -15.0
THR_DB = 1.5

ap = argparse.ArgumentParser()
ap.add_argument("shp")
ap.add_argument("--project", required=True)
ap.add_argument("--orbit", type=int, default=97)
args = ap.parse_args()
ee.Initialize(project=args.project)

ids = set(pd.read_csv(OUT / "features_nisar.csv").poly_id)
logging, controls = load_polygons(args.shp)
polys = pd.concat([logging, controls], ignore_index=True)
polys = gpd.GeoDataFrame(polys[polys.poly_id.isin(ids)], crs=logging.crs).reset_index(drop=True)
utm = polys.estimate_utm_crs()
polys["geometry"] = polys.to_crs(utm).geometry.apply(make_valid).buffer(0).simplify(5).to_crs(4326).values
polys = polys.set_crs(4326, allow_override=True)
print(f"Polygons: {len(polys)}")

col = (ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT").filterBounds(ee.Geometry.Rectangle(list(AOI)))
       .filterDate(START, END).filter(ee.Filter.eq("instrumentMode", "IW"))
       .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
       .filter(ee.Filter.eq("relativeOrbitNumber_start", args.orbit)).select(["VV", "VH"])
       .map(lambda im: im.updateMask(im.gt(0))))
times = col.aggregate_array("system:time_start").getInfo()
dates = sorted(set(pd.to_datetime(times, unit="ms").normalize()))
print("Sentinel-1 dates:", [str(d.date()) for d in dates])

box = ee.Kernel.square(1, "pixels")                       # 3x3 at the 20 m output scale


def to_db(img):
    return img.log10().multiply(10)


before = col.filterDate(START, str((REF_END + pd.Timedelta(days=1)).date()))
B = before.mean().convolve(box)
forest = to_db(before.mean().select("VH").convolve(ee.Kernel.square(2, "pixels"))).gt(FOREST_VH_DB)
cov_b = before.reduce(ee.Reducer.stdDev()).divide(before.mean()).multiply(100).rename(["VVcov_b", "VHcov_b"])
rf_b = B.normalizedDifference(["VV", "VH"]).rename("RF_b")            # (VV - VH) / (VV + VH) in power = mRFDI
reducer = (ee.Reducer.median().combine(ee.Reducer.percentile([10, 90]), sharedInputs=True)
           .combine(ee.Reducer.stdDev(), sharedInputs=True).combine(ee.Reducer.mean(), sharedInputs=True)
           .combine(ee.Reducer.count(), sharedInputs=True))

rows = []
for view, grp in polys.groupby("VIEW_DATE"):
    n_aft = sum(d >= view for d in dates)
    if n_aft == 0:
        continue                                                          # no image after the DETER date
    after = col.filterDate(str(view.date()), END)
    A = after.mean().convolve(box)
    d = to_db(A.divide(B)).rename(["VV_d", "VH_d"])
    bands = [d, d.lt(-THR_DB).rename(["VV_drop", "VH_drop"]), d.gt(THR_DB).rename(["VV_rise", "VH_rise"]),
             B.rename(["VV_B", "VH_B"]), A.rename(["VV_A", "VH_A"]), cov_b, rf_b]
    rf_a = A.normalizedDifference(["VV", "VH"]).rename("RF_a")
    bands += [rf_a, rf_a.subtract(rf_b).rename("RF_d")]
    if n_aft >= 2:
        bands.append(after.reduce(ee.Reducer.stdDev()).divide(after.mean()).multiply(100).rename(["VVcov_a", "VHcov_a"]))
    img = ee.Image.cat(bands).updateMask(forest)
    fc = ee.FeatureCollection(grp[["poly_id", "geometry"]].__geo_interface__)
    feats = img.reduceRegions(collection=fc, reducer=reducer, scale=20, crs=CRS, tileScale=4).getInfo()["features"]
    for f in feats:
        p = f["properties"]
        if not p.get("VV_d_count"):
            continue
        row = {"poly_id": p["poly_id"], "n_pixels_s1": p["VV_d_count"], "S1_n_aft": n_aft}
        for pol in ("VV", "VH"):
            row[f"{pol}_bef"] = 10 * np.log10(p[f"{pol}_B_median"])
            row[f"{pol}_aft"] = 10 * np.log10(p[f"{pol}_A_median"])
            row[f"{pol}_diff"] = p[f"{pol}_d_median"]
            row[f"{pol}_diff_p10"], row[f"{pol}_diff_p90"] = p[f"{pol}_d_p10"], p[f"{pol}_d_p90"]
            row[f"{pol}_diff_std"] = p[f"{pol}_d_stdDev"]
            row[f"{pol}_frac_drop"], row[f"{pol}_frac_rise"] = p[f"{pol}_drop_mean"], p[f"{pol}_rise_mean"]
            row[f"{pol}cov_bef"] = p[f"{pol}cov_b_median"]
            if f"{pol}cov_a_median" in p and p[f"{pol}cov_a_median"] is not None:
                row[f"{pol}cov_aft"] = p[f"{pol}cov_a_median"]
                row[f"{pol}cov_diff"] = row[f"{pol}cov_aft"] - row[f"{pol}cov_bef"]
        row["mRFDI_bef"], row["mRFDI_aft"] = p["RF_b_median"], p["RF_a_median"]
        row["mRFDI_diff"], row["mRFDI_diff_p90"] = p["RF_d_median"], p["RF_d_p90"]
        rows.append(row)
    print(f"view date {view.date()}: {len(grp)} polygons, {n_aft} images after")

pd.DataFrame(rows).to_csv(OUT / "features_s1.csv", index=False)
print(f"Saved {OUT / 'features_s1.csv'}: {len(rows)} polygons")
