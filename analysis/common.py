"""Shared definitions: study area, DETER logging polygons and matched control polygons.

Every script imports these, so polygon ids (P012 = logging, N034 = control) mean the same polygon everywhere.
A control is a copy of a logging polygon moved 3-15 km to a place at least 1 km away from any DETER alert
since 2020. It inherits the DETER date of its source polygon ("matched" design: same shape, same dates).
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import make_valid
from shapely.affinity import translate

AOI = (-49.0, -4.5, -47.0, -2.5)            # lon_min, lat_min, lon_max, lat_max (Paragominas, Para)
START = "2026-06-17"                        # first calibrated NISAR data
REF_END = pd.Timestamp("2026-07-20")        # images up to this date = "before"
SELECTIVE = ["CS_DESORDENADO", "CS_GEOMETRICO"]


def load_polygons(shp):
    """Return (logging polygons, control polygons) in the DETER CRS, with poly_id and view_date."""
    deter = gpd.read_file(shp, bbox=AOI)
    deter["VIEW_DATE"] = pd.to_datetime(deter["VIEW_DATE"])
    polys = deter[deter.CLASSNAME.isin(SELECTIVE) & (deter.VIEW_DATE >= START)].reset_index(drop=True)
    polys["poly_id"] = [f"P{i:03d}" for i in range(len(polys))]
    alerts = deter[deter.VIEW_DATE >= "2020-01-01"]
    return polys, make_controls(polys, alerts)


def make_controls(polys, alerts, n_per_poly=2, seed=42):
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
