"""Case-study experiment: can before/after radar features separate DETER selective logging from intact forest?

Inputs (one row per polygon): analysis/features_nisar.csv (L-band) and, if present, analysis/features_s1.csv
(C-band). Only CHANGE features are used for classification (differences after - before, share of pixels
with a strong drop / rise). The "before" and "after" LEVELS are reported but never given to the classifier:
logging and control forests already differ before logging, so levels would leak "where" instead of "what changed".

Steps
  1. keep polygons that are >= 90 % radar forest before logging (L-band HV mask) and present in every sensor
  2. per feature: median of logging vs control, AUC and Mann-Whitney p
  3. Random Forest on change features, 5-fold cross-validation on 10 km spatial blocks:
     L only, C only, C + L -> confusion matrix, accuracy, precision, recall, F1, AUC
  4. variable importance of the full model (as in Flores-Anderson et al. 2026)

Run:  python analysis/run_experiment.py [--min-view 2026-08-20]
Writes analysis/results/: feature_table.csv, classification.csv, importance.csv, fig_features.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RES.mkdir(exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--min-view", default="2026-08-01", help="keep polygons first seen by DETER on/after this date")
ap.add_argument("--min-forest", type=float, default=0.9)
ap.add_argument("--block-km", type=float, default=10)
A = ap.parse_args()

CHANGE = ("_diff", "_diff_p10", "_diff_p90", "_diff_std", "_frac_drop", "_frac_rise")
SENSORS = {"L": ("HH", "HV", "HHcov", "HVcov", "RFDI"), "C": ("VV", "VH", "VVcov", "VHcov", "mRFDI")}


def change_cols(df, sensor):
    return [c for c in df.columns if c.endswith(CHANGE) and c.split("_")[0] in SENSORS[sensor]
            and df[c].notna().mean() > 0.9]


df = pd.read_csv(HERE / "features_nisar.csv", parse_dates=["view_date"])
have_c = (HERE / "features_s1.csv").exists()
if have_c:
    s1 = pd.read_csv(HERE / "features_s1.csv").drop(columns=["label", "view_date", "n_pixels"], errors="ignore")
    df = df.merge(s1, on="poly_id", how="inner")                        # same polygons in both sensors
df = df[(df.forest_frac >= A.min_forest) & (df.view_date >= A.min_view)].reset_index(drop=True)
print(f"Polygons: {int(df.label.sum())} logging, {int((df.label == 0).sum())} control "
      f"(forest >= {A.min_forest:.0%}, DETER date >= {A.min_view}, C-band: {'yes' if have_c else 'not yet'})")

# ---------------------------------------------------------------- 2. one feature at a time
rows = []
feature_cols = [c for c in df.columns if c.split("_")[0] in SENSORS["L"] + SENSORS["C"] and "_n_" not in c]
for c in feature_cols:
    a, b = df[df.label == 1][c].dropna(), df[df.label == 0][c].dropna()
    if len(a) < 10 or len(b) < 10:
        continue
    u, p = mannwhitneyu(a, b)
    rows.append({"feature": c, "used_by_classifier": c.endswith(CHANGE), "n_logging": len(a), "n_control": len(b),
                 "logging_median": a.median(), "control_median": b.median(), "AUC": u / len(a) / len(b), "p": p})
ft = pd.DataFrame(rows).sort_values("p")
ft.to_csv(RES / "feature_table.csv", index=False)
print("\nSingle features (AUC 0.5 = no difference; <0.5 = lower in logging polygons):")
print(ft.round(4).to_string(index=False))

# ---------------------------------------------------------------- 3. Random Forest, spatial block CV
blocks = (np.floor(df.x_m / (A.block_km * 1000)).astype(int).astype(str) + "_"
          + np.floor(df.y_m / (A.block_km * 1000)).astype(int).astype(str))
print(f"\nSpatial blocks of {A.block_km:.0f} km: {blocks.nunique()}")


def evaluate(cols, name):
    X = df[cols].fillna(df[cols].median())
    proba = np.zeros(len(df))
    for tr, te in GroupKFold(5).split(X, df.label, groups=blocks):
        rf = RandomForestClassifier(500, min_samples_leaf=3, class_weight="balanced", random_state=0)
        proba[te] = rf.fit(X.iloc[tr], df.label.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
    pred = proba >= 0.5
    tn, fp, fn, tp = confusion_matrix(df.label, pred).ravel()
    thr = np.quantile(proba[df.label == 0], 0.95)                        # threshold giving 5 % false alarms
    return {"model": name, "n_features": len(cols), "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "accuracy": (tp + tn) / len(df), "precision": precision_score(df.label, pred, zero_division=0),
            "recall": recall_score(df.label, pred), "F1": f1_score(df.label, pred),
            "AUC": roc_auc_score(df.label, proba), "detection_at_5%FAR": (proba[df.label == 1] > thr).mean()}


models = {"L-band (HH only)": [c for c in change_cols(df, "L") if c.startswith("HH")],
          "L-band (HH + HV + RFDI)": change_cols(df, "L")}
if have_c:
    models["C-band (VV + VH + mRFDI)"] = change_cols(df, "C")
    models["C + L"] = change_cols(df, "C") + change_cols(df, "L")
models["placebo: L-band HH, before vs before"] = ["HH_placebo"]
cls = pd.DataFrame([evaluate(cols, name) for name, cols in models.items() if cols])
cls.to_csv(RES / "classification.csv", index=False)
print("\nRandom Forest, change features only, spatial 5-fold CV:")
print(cls.round(3).to_string(index=False))
print(f"Majority-class accuracy (always 'control'): {(df.label == 0).mean():.3f}")

# ---------------------------------------------------------------- 4. importance of the fullest model
cols = list(models["C + L"] if have_c else models["L-band (HH + HV + RFDI)"])
rf = RandomForestClassifier(500, min_samples_leaf=3, class_weight="balanced", random_state=0)
rf.fit(df[cols].fillna(df[cols].median()), df.label)
imp = pd.Series(rf.feature_importances_, cols).sort_values(ascending=False)
imp.to_csv(RES / "importance.csv", header=["mean_decrease_gini"])
print("\nVariable importance (mean decrease Gini):")
print(imp.head(10).round(3).to_string())

# ---------------------------------------------------------------- figure: the main change features
show = [c for c in ["HH_diff", "HV_diff", "HV_frac_drop", "RFDI_diff", "VV_diff", "VH_diff", "VH_frac_drop", "mRFDI_diff"]
        if c in df]
fig, axes = plt.subplots(1, len(show), figsize=(2.6 * len(show), 3.6))
for ax, c in zip(np.atleast_1d(axes), show):
    data = [df[df.label == 0][c].dropna(), df[df.label == 1][c].dropna()]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
    for patch, col in zip(bp["boxes"], ["#9a9a9a", "#d1495b"]):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["control", "logging"], fontsize=8)
    ax.set_title(c, fontsize=9, loc="left"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(RES / "fig_features.png", dpi=150)
print("\nSaved results/feature_table.csv, classification.csv, importance.csv, fig_features.png")
