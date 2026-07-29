#!/usr/bin/env python3
"""Predict La Ronde fireworks nights from BIXI demand -- INFERENCE ONLY.

Loads the model bundle trained + exported by `firework_anomaly_train.ipynb`
(`output/firework_model.joblib`), runs the shared `firework_features` preprocessing
on a season's trip CSV, and ranks the nights most likely to have hosted a fireworks
show. No training happens here -- retrain by re-running the notebook.

Preprocessing is imported from firework_features.py (the exact same code the model was
trained with), so there is no train/serve skew.

Example
-------
    python predict_fireworks.py \\
        --predict "/Volumes/Extreme SSD/SUMO Data/DonneesOuvertes2026_<full-season>.csv"

Note: the fireworks run late-June to early-August. A predict CSV that stops before July
(e.g. the Jan-Apr 2026 export) contains no fireworks season and the script says so.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

import firework_features as ff


def load_bundle(path):
    path = Path(path)
    if not path.exists():
        sys.exit(f"ERROR: model bundle not found at {path}\n"
                 f"       Train and export it first by running firework_anomaly_train.ipynb.")
    bundle = joblib.load(path)
    required = {"models", "features", "threshold", "default_model", "use_weather"}
    missing = required - set(bundle)
    if missing:
        sys.exit(f"ERROR: model bundle is missing keys {missing}; re-export from the notebook.")
    return bundle


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predict", required=True, help="season CSV to score")
    ap.add_argument("--model-path", default="output/firework_model.joblib",
                    help="bundle from firework_anomaly_train.ipynb")
    ap.add_argument("--model", default=None,
                    help="which saved model to use (default: the bundle's default_model)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the bundle's saved decision threshold")
    ap.add_argument("--top-n", type=int, default=12, help="ranked nights to print")
    ap.add_argument("--no-weather", action="store_true", help="skip Open-Meteo (offline)")
    ap.add_argument("--cache-dir", default="output")
    ap.add_argument("--fireworks-json", default="fireworks_dates.json",
                    help="labels DB, used to validate when the predicted year is known")
    ap.add_argument("--out", default=None, help="write the full ranked table to this CSV")
    args = ap.parse_args(argv)

    bundle = load_bundle(args.model_path)
    features = bundle["features"]
    model_name = args.model or bundle["default_model"]
    if model_name not in bundle["models"]:
        sys.exit(f"ERROR: model '{model_name}' not in bundle; available: {list(bundle['models'])}")
    clf = bundle["models"][model_name]
    threshold = args.threshold if args.threshold is not None else bundle["threshold"]

    print(f"Model bundle: {args.model_path}")
    print(f"  trained on {bundle.get('train_years')} | using '{model_name}' | "
          f"threshold {threshold:.3f}")
    print(f"  features: {features}")

    # The model needs weather columns iff it was trained with them.
    need_weather = bundle["use_weather"]
    if need_weather and args.no_weather:
        sys.exit("ERROR: this model was trained WITH weather features, so --no-weather cannot\n"
                 "       satisfy its inputs. Retrain a no-weather model or drop --no-weather.")

    print(f"\nLoading PREDICT season: {args.predict}")
    pred, weather_ok = ff.build_features(args.predict, use_weather=need_weather,
                                         cache_dir=args.cache_dir)
    if pred is None:
        sys.exit("ERROR: predict CSV has no June-August nights in the viewshed.\n"
                 "       The fireworks run July-early August -- a CSV that stops before July\n"
                 "       (e.g. the Jan-Apr 2026 export) does not contain the season yet.")
    if need_weather and not weather_ok:
        sys.exit("ERROR: model needs weather features but the weather fetch failed for this\n"
                 "       season. Re-run with a working connection (Open-Meteo archive).")

    pred_year = ff.season_year(pred)
    missing_cols = [c for c in features if c not in pred.columns]
    if missing_cols:
        sys.exit(f"ERROR: preprocessed data is missing model features {missing_cols}.")

    X = pred[features].fillna(0.0)
    pred = pred.copy()
    pred["fireworks_proba"] = clf.predict_proba(X)[:, 1]
    ranked = pred.sort_values("fireworks_proba", ascending=False)

    print(f"\n=== Predicted fireworks nights for {pred_year} (proba >= {threshold:.2f}) ===")
    called = ranked[ranked["fireworks_proba"] >= threshold]
    if called.empty:
        print("  (none above threshold)")
    for night, row in called.iterrows():
        print(f"  {night.date()}  {night.day_name():9s}  p={row['fireworks_proba']:.2f}  "
              f"escape_netout={int(row['escape_netout'])}  escape_rides={int(row['escape_rides'])}")

    print(f"\n=== Top {args.top_n} nights by probability ===")
    cols = ["fireworks_proba", "escape_rides", "escape_netout", "escape_share", "netout_share"]
    view = ranked.head(args.top_n)[cols].copy()
    view.index = [f"{d.date()} {d.day_name()[:3]}" for d in view.index]
    print(view.to_string(float_format=lambda v: f"{v:.2f}"))

    # Validate against ground truth when the predicted season's calendar is known.
    fw_dates = ff.load_fireworks_dates(args.fireworks_json)
    if pred_year in fw_dates:
        truth = ff.label_series(pred.index, fw_dates[pred_year]).astype(bool)
        called_mask = pred["fireworks_proba"] >= threshold
        tp = int((called_mask & truth).sum())
        fp = int((called_mask & ~truth).sum())
        fn = int((~called_mask & truth).sum())
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        try:
            from sklearn.metrics import average_precision_score
            ap = average_precision_score(truth, pred["fireworks_proba"])
        except Exception:
            ap = float("nan")
        print(f"\n=== Validation vs known {pred_year} calendar ===")
        print(f"  precision {prec:.2f}  recall {rec:.2f}  PR-AUC {ap:.2f}  "
              f"(TP={tp} FP={fp} FN={fn}, {int(truth.sum())} real shows)")

    if args.out:
        out_cols = ["fireworks_proba", "escape_rides", "escape_netout",
                    "escape_share", "netout_share", "pre_arrival_share", "dow"]
        ranked[out_cols].to_csv(args.out)
        print(f"\nWrote ranked table -> {args.out}")


if __name__ == "__main__":
    main()
