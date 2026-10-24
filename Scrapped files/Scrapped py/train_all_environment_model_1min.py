"""Train and evaluate a one-minute-ahead multi-output ventilation forecaster.

Expected files beside this script:
  data/training_data_2.csv
  data/ventilation_test.csv

Run: python train_all_environment_model_1min.py

The script predicts eight continuous environmental variables one minute ahead.
It compares a trained multi-output Random Forest with persistence and a capped
one-minute CO2 trend baseline. The independent test file is never used for
model selection or training.
"""
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
TRAIN_FILE = DATA_DIR / "training_data_2.csv"
TEST_FILE = DATA_DIR / "ventilation_test.csv"

TIME_COL = "time"
TARGETS = [
    "Room CO2 (ppm)", "Temperature (°C)", "Relative Humidity (%)",
    "Pressure (mbar)", "Supply airflow (l/s)", "Extract airflow (l/s)",
    "Supply air CO2 (ppm)", "Exhaust air CO2 (ppm)",
]
# Occupant count is included because it is a ground-truth measurement.
# The ML-estimated occupancy column is deliberately excluded to avoid leakage.
BASE_FEATURES = TARGETS + ["Occupant count"]
LAGS_MINUTES = [0, 1, 2, 3, 5]
HORIZON_MINUTES = 1
CO2_TREND_CAP = 75.0


def load_data(path):
    df = pd.read_csv(path)
    missing = [c for c in [TIME_COL, *BASE_FEATURES, "Occupant count (ML estimate)"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL)
    # If duplicate timestamps exist, keep the last recorded row.
    df = df.drop_duplicates(TIME_COL, keep="last").set_index(TIME_COL)
    for col in BASE_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def make_supervised(df):
    """Join each row's features to the measured values exactly one minute later."""
    feats = pd.DataFrame(index=df.index)
    for lag in LAGS_MINUTES:
        shifted = df[BASE_FEATURES].copy()
        shifted.index = shifted.index + pd.Timedelta(minutes=lag)
        for col in BASE_FEATURES:
            feats[f"{col}__lag_{lag}m"] = shifted[col].reindex(df.index)
    future = df[TARGETS].copy()
    future.index = future.index - pd.Timedelta(minutes=HORIZON_MINUTES)
    future.columns = [f"target__{c}" for c in TARGETS]
    joined = feats.join(future, how="inner")
    # Current target values serve as persistence predictions and current CO2
    # plus its one-minute change serves as the trend baseline.
    joined["baseline__Room CO2 (ppm)__current"] = df["Room CO2 (ppm)"].reindex(joined.index)
    joined["baseline__Room CO2 (ppm)__previous"] = df["Room CO2 (ppm)"].shift(1).reindex(joined.index)
    joined["baseline__Temperature (°C)__current"] = df["Temperature (°C)"].reindex(joined.index)
    return joined.replace([np.inf, -np.inf], np.nan).dropna()


def metric_row(target, method, y, pred, selected=False):
    return {
        "target": target,
        "method": method,
        "n": int(len(y)),
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)) if len(y) > 1 else float("nan"),
        "selected_by_validation": bool(selected),
    }


def evaluate_set(frame, model, selected_methods=None):
    X = frame[[f"{c}__lag_{lag}m" for lag in LAGS_MINUTES for c in BASE_FEATURES]]
    y = frame[[f"target__{c}" for c in TARGETS]].to_numpy()
    rf_pred = model.predict(X)
    rows, pred_frame = [], pd.DataFrame({"time": frame.index})
    for j, target in enumerate(TARGETS):
        actual = y[:, j]
        current = frame[f"{target}__lag_0m"].to_numpy()
        persistence = current
        methods = {"persistence": persistence, "random_forest": rf_pred[:, j]}
        if target == "Room CO2 (ppm)":
            previous = frame[f"{target}__lag_1m"].to_numpy()
            change = current - previous
            trend = current + np.clip(change, -CO2_TREND_CAP, CO2_TREND_CAP)
            methods["trend_capped"] = trend
        chosen = (selected_methods or {}).get(target, "random_forest")
        for method, pred in methods.items():
            rows.append(metric_row(target, method, actual, pred, method == chosen))
        pred_frame[f"actual__{target}"] = actual
        for method, pred in methods.items():
            pred_frame[f"{method}__{target}"] = pred
        pred_frame[f"selected__{target}"] = methods.get(chosen, rf_pred[:, j])
    return pd.DataFrame(rows), pred_frame


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        raise FileNotFoundError(
            "Expected data/training_data_2.csv and data/ventilation_test.csv. "
            "Place the script in the project root and keep these CSVs in data/."
        )

    train_df = load_data(TRAIN_FILE)
    test_df = load_data(TEST_FILE)
    train_all = make_supervised(train_df)
    test_all = make_supervised(test_df)
    if len(train_all) < 100 or len(test_all) < 20:
        raise ValueError("Too few exact one-minute timestamp pairs after removing gaps/missing values.")

    # Chronological validation; no random shuffling for time-series data.
    cut = int(len(train_all) * 0.8)
    train_part, val_part = train_all.iloc[:cut], train_all.iloc[cut:]
    feature_cols = [f"{c}__lag_{lag}m" for lag in LAGS_MINUTES for c in BASE_FEATURES]
    target_cols = [f"target__{c}" for c in TARGETS]
    val_model = MultiOutputRegressor(RandomForestRegressor(
        n_estimators=30, min_samples_leaf=2, max_features=0.8,
        random_state=42, n_jobs=1,
    ), n_jobs=1)
    val_model.fit(train_part[feature_cols], train_part[target_cols])
    val_metrics, val_predictions = evaluate_set(val_part, val_model)

    # Select per-target method on validation only, comparing MAE.
    selected = {}
    for target in TARGETS:
        rows = val_metrics[val_metrics["target"] == target]
        candidates = rows[rows["method"].isin(["persistence", "random_forest", "trend_capped"])]
        selected[target] = candidates.sort_values("MAE").iloc[0]["method"]
        # Trend is only a candidate for room CO2; for all other targets the
        # shared trend rule is not evaluated or selected.
    val_metrics["selected_by_validation"] = val_metrics.apply(
        lambda r: r["method"] == selected[r["target"]], axis=1
    )

    # Refit the final model on all development data (not the independent test).
    final_model = MultiOutputRegressor(RandomForestRegressor(
        n_estimators=30, min_samples_leaf=2, max_features=0.8,
        random_state=42, n_jobs=1,
    ), n_jobs=1)
    final_model.fit(train_all[feature_cols], train_all[target_cols])
    holdout_metrics, holdout_predictions = evaluate_set(test_all, final_model, selected)
    holdout_metrics["selected_by_validation"] = holdout_metrics.apply(
        lambda r: r["method"] == selected[r["target"]], axis=1
    )

    # Save a simple prediction-ready bundle and explicit metadata.
    bundle = {
        "model": final_model,
        "feature_columns": feature_cols,
        "targets": TARGETS,
        "lags_minutes": LAGS_MINUTES,
        "horizon_minutes": HORIZON_MINUTES,
        "selected_methods": selected,
        "co2_trend_cap_ppm": CO2_TREND_CAP,
        "note": "Regression forecaster; accuracy/precision/recall/F1 are not appropriate for continuous targets.",
    }
    joblib.dump(bundle, MODEL_DIR / "all_environment_model_1min.joblib")
    val_metrics.to_csv(MODEL_DIR / "all_environment_1min_validation_metrics.csv", index=False)
    holdout_metrics.to_csv(MODEL_DIR / "all_environment_1min_holdout_metrics.csv", index=False)
    val_predictions.to_csv(MODEL_DIR / "all_environment_1min_validation_predictions.csv", index=False)
    holdout_predictions.to_csv(MODEL_DIR / "all_environment_1min_holdout_predictions.csv", index=False)
    with open(MODEL_DIR / "all_environment_1min_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "horizon_minutes": HORIZON_MINUTES,
            "targets": TARGETS,
            "lags_minutes": LAGS_MINUTES,
            "selected_methods_by_validation_mae": selected,
            "train_rows": int(len(train_all)),
            "validation_rows": int(len(val_part)),
            "holdout_rows": int(len(test_all)),
            "model": "MultiOutputRegressor(RandomForestRegressor)",
            "random_forest": {"n_estimators": 30, "min_samples_leaf": 2, "max_features": 0.8, "random_state": 42},
            "classification_metrics": "Not applicable to continuous regression outputs; define separate classes before computing F1/precision/recall.",
        }, f, indent=2)

    print("\n=== VALIDATION MAE (lower is better) ===")
    print(val_metrics.pivot(index="target", columns="method", values="MAE").round(4).to_string())
    print("\nSelected method per target (validation only):")
    for target, method in selected.items():
        print(f"  {target}: {method}")
    print("\n=== INDEPENDENT ONE-MINUTE HOLDOUT RESULTS ===")
    print(holdout_metrics.sort_values(["target", "method"])[["target", "method", "n", "MAE", "RMSE", "R2", "selected_by_validation"]].round(4).to_string(index=False))
    print("\nSaved model and reports under:", MODEL_DIR)
    print("Metrics are regression metrics. Accuracy, precision, recall and F1 require a separate classification target.")

if __name__ == "__main__":
    main()
