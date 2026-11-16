"""Bootstrap confidence intervals for the classification AUCs and for the differences between sensors.

Same data, filters, features and spatial cross-validation as run_experiment.py. The out-of-fold probabilities
of each model are resampled by polygon (2000 times) to get 95 % intervals.

Run:  python analysis/bootstrap_ci.py      -> analysis/results/bootstrap_ci.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
CHANGE = ("_diff", "_diff_p10", "_diff_p90", "_diff_std", "_frac_drop", "_frac_rise")
SENSORS = {"L": ("HH", "HV", "HHcov", "HVcov", "RFDI"), "C": ("VV", "VH", "VVcov", "VHcov", "mRFDI")}

df = pd.read_csv(HERE / "features_nisar.csv", parse_dates=["view_date"])
s1 = pd.read_csv(HERE / "features_s1.csv").drop(columns=["label", "view_date", "n_pixels"], errors="ignore")
df = df.merge(s1, on="poly_id", how="inner")
df = df[(df.forest_frac >= 0.9) & (df.view_date >= "2026-08-01")].reset_index(drop=True)
blocks = (np.floor(df.x_m / 10000).astype(int).astype(str) + "_" + np.floor(df.y_m / 10000).astype(int).astype(str))


def cols(sensor):
    return [c for c in df.columns if c.endswith(CHANGE) and c.split("_")[0] in SENSORS[sensor] and df[c].notna().mean() > 0.9]


def oof(c):
    X = df[c].fillna(df[c].median())
    p = np.zeros(len(df))
    for tr, te in GroupKFold(5).split(X, df.label, groups=blocks):
        rf = RandomForestClassifier(500, min_samples_leaf=3, class_weight="balanced", random_state=0)
        p[te] = rf.fit(X.iloc[tr], df.label.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
    return p


P = {"L": oof(cols("L")), "C": oof(cols("C")), "C+L": oof(cols("C") + cols("L")), "placebo": oof(["HH_placebo"])}
y = df.label.values
rng = np.random.default_rng(0)
boot = {k: [] for k in P}
for _ in range(2000):
    i = rng.integers(0, len(y), len(y))
    if y[i].min() == y[i].max():
        continue
    for k in P:
        boot[k].append(roc_auc_score(y[i], P[k][i]))
B = pd.DataFrame(boot)
rows = [{"quantity": f"AUC {k}", "value": roc_auc_score(y, P[k]), "lo": B[k].quantile(.025), "hi": B[k].quantile(.975)} for k in P]
for a, b in [("C", "L"), ("C+L", "C"), ("C+L", "L")]:
    d = B[a] - B[b]
    rows.append({"quantity": f"AUC {a} - AUC {b}", "value": roc_auc_score(y, P[a]) - roc_auc_score(y, P[b]),
                 "lo": d.quantile(.025), "hi": d.quantile(.975)})
out = pd.DataFrame(rows)
out.to_csv(HERE / "results" / "bootstrap_ci.csv", index=False)
print(f"{int(y.sum())} logging, {int((y == 0).sum())} control")
print(out.round(3).to_string(index=False))
