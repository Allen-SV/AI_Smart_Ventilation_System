"""Evaluate the existing five-minute ventilation forecasting model.

Expected project layout:
    models/all_environment_model_5min.joblib
    data/ventilation_test.csv
    data/training_data_2.csv   (used only to derive Low/Medium/High tertile thresholds)

Run from the project root:
    python evaluate_five_minute_model.py

The script does NOT train or modify the model. It compares predictions made at time T
against recorded sensor values at exactly T+5 minutes. Missing history/future matches
are excluded, never counted as passes. Tertile categories are statistical bins, not
health or safety thresholds.
"""
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, recall_score, r2_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "all_environment_model_5min.joblib"
if not MODEL_PATH.exists() and (ROOT / "all_environment_model_5min.joblib").exists():
    MODEL_PATH = ROOT / "all_environment_model_5min.joblib"
TEST_PATH = ROOT / "data" / "ventilation_test.csv"
if not TEST_PATH.exists():
    # Convenient fallback for running beside uploaded CSVs during review.
    TEST_PATH = ROOT / "ventilation_test(2).csv" if (ROOT / "ventilation_test(2).csv").exists() else ROOT / "ventilation_test.csv"
TRAIN_PATH = ROOT / "data" / "training_data_2.csv"
if not TRAIN_PATH.exists() and (ROOT / "training_data_2.csv").exists():
    TRAIN_PATH = ROOT / "training_data_2.csv"
OUTPUT_DIR = ROOT / "models"
TIME_COL = "time"
TARGETS = ["Room CO2 (ppm)", "Temperature (°C)", "Relative Humidity (%)"]
BASE_FEATURES = [
    "Room CO2 (ppm)", "Temperature (°C)", "Relative Humidity (%)",
    "Pressure (mbar)", "Supply airflow (l/s)", "Extract airflow (l/s)",
    "Supply air CO2 (ppm)", "Exhaust air CO2 (ppm)", "Occupant count",
]
# Tolerances are explicit evaluation criteria, not safety thresholds.
TOLERANCES = {
    "Room CO2 (ppm)": [15, 20, 30, 40, 50],              # ppm
    "Temperature (°C)": [0.05, 0.10, 0.20, 0.50],        # degrees C
    "Relative Humidity (%)": [0.5, 1.0, 2.0, 3.0],       # percentage points
}


def load_data(path):
    df = pd.read_csv(path)
    required = [TIME_COL, *BASE_FEATURES, *TARGETS]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL)
    df = df.drop_duplicates(TIME_COL, keep="last").set_index(TIME_COL)
    for col in set(BASE_FEATURES + TARGETS):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def row_at(df, timestamp):
    if timestamp not in df.index:
        return None
    row = df.loc[timestamp]
    return row.iloc[-1] if isinstance(row, pd.DataFrame) else row


def main():
    for path in (MODEL_PATH, TEST_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}\nCheck the project folder paths.")

    # Loading a joblib model requires trusting the model file. Version warnings may occur
    # if scikit-learn differs from the version used when the model was saved.
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        bundle = joblib.load(MODEL_PATH)

    horizon = int(bundle.get("horizon_minutes", -1))
    if horizon != 5:
        raise ValueError(f"Expected a 5-minute model, but bundle horizon is {horizon}.")
    if list(bundle.get("targets", [])) != TARGETS:
        raise ValueError(f"Unexpected model targets: {bundle.get('targets')}")

    df = load_data(TEST_PATH)
    feature_cols = bundle["feature_columns"]
    lags = [int(x) for x in bundle.get("lags_minutes", [0, 1, 2, 3, 5])]
    methods = bundle.get("selected_methods", {})
    cap = float(bundle.get("co2_trend_cap_ppm", 75.0))
    records = []
    skip_history = 0
    skip_future = 0

    # Each forecast uses only readings at T and earlier. Actual T+5 values are joined
    # only afterward for evaluation.
    for timestamp in df.index:
        cur = row_at(df, timestamp)
        future_time = timestamp + pd.Timedelta(minutes=horizon)
        actual_future = row_at(df, future_time)
        if actual_future is None:
            skip_future += 1
            continue

        feature_values = {}
        history_ok = True
        for lag in lags:
            source_time = timestamp - pd.Timedelta(minutes=lag)
            source_row = row_at(df, source_time)
            if source_row is None:
                history_ok = False
                break
            for col in BASE_FEATURES:
                feature_values[f"{col}__lag_{lag}m"] = source_row[col]
        if not history_ok:
            skip_history += 1
            continue

        X = pd.DataFrame([[feature_values.get(c, np.nan) for c in feature_cols]], columns=feature_cols)
        if X.isna().any().any():
            skip_history += 1
            continue
        pred = {}
        # The saved selected methods are persistence/trend_capped, so no Random Forest
        # inference is needed for those targets. Only call the model for a target whose
        # selected method falls back to its learned regression output.
        fallback_targets = [
            t for t in TARGETS
            if methods.get(t, "random_forest") != "persistence"
            and not (methods.get(t) == "trend_capped" and t == "Room CO2 (ppm)")
        ]
        raw_pred = {}
        if fallback_targets:
            raw = np.asarray(bundle["model"].predict(X)).reshape(-1)
            raw_pred = {target: float(raw[i]) for i, target in enumerate(TARGETS)}
        for target in TARGETS:
            method = methods.get(target, "random_forest")
            if method == "persistence":
                pred[target] = float(cur[target])
            elif method == "trend_capped" and target == "Room CO2 (ppm)":
                prev = row_at(df, timestamp - pd.Timedelta(minutes=1))
                if prev is None or pd.isna(prev[target]) or pd.isna(cur[target]):
                    pred[target] = np.nan
                else:
                    rate = float(cur[target]) - float(prev[target])
                    pred[target] = float(cur[target] + np.clip(rate * horizon, -cap, cap))
            else:
                pred[target] = raw_pred[target]

        rec = {
            "Input timestamp": timestamp,
            "Forecast timestamp": future_time,
            "Current occupant count": cur.get("Occupant count", np.nan),
        }
        for target in TARGETS:
            short = target
            actual = float(actual_future[target]) if pd.notna(actual_future[target]) else np.nan
            current = float(cur[target]) if pd.notna(cur[target]) else np.nan
            prediction = pred[target]
            rec[f"Current {short}"] = current
            rec[f"Predicted {short}"] = prediction
            rec[f"Actual {short} at forecast"] = actual
            rec[f"Absolute error {short}"] = abs(prediction - actual) if pd.notna(prediction) and pd.notna(actual) else np.nan
        records.append(rec)

    results = pd.DataFrame(records)
    if results.empty:
        raise RuntimeError("No valid forecasts could be evaluated. Check timestamps and lag history.")

    # Regression metrics and tolerance pass rates.
    metric_rows, tolerance_rows = [], []
    for target in TARGETS:
        err_col = f"Absolute error {target}"
        pred_col = f"Predicted {target}"
        actual_col = f"Actual {target} at forecast"
        valid = results[[pred_col, actual_col, err_col]].dropna()
        y_pred = valid[pred_col].to_numpy()
        y_true = valid[actual_col].to_numpy()
        errors = y_pred - y_true
        metric_rows.append({
            "Target": target,
            "N": len(valid),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred) if len(valid) > 1 else np.nan,
            "Mean bias (pred-actual)": float(np.mean(errors)),
        })
        for tol in TOLERANCES[target]:
            passed = valid[err_col] <= tol
            tolerance_rows.append({
                "Target": target,
                "Tolerance": tol,
                "Unit": "ppm" if target == TARGETS[0] else ("°C" if target == TARGETS[1] else "percentage points"),
                "Valid forecasts": len(valid),
                "Passed": int(passed.sum()),
                "Failed": int((~passed).sum()),
                "Pass rate (%)": float(passed.mean() * 100),
            })

    metrics = pd.DataFrame(metric_rows)
    tolerances = pd.DataFrame(tolerance_rows)
    OUTPUT_DIR.mkdir(exist_ok=True)
    results_path = OUTPUT_DIR / "five_minute_model_accuracy_results.csv"
    metrics_path = OUTPUT_DIR / "five_minute_model_metrics.csv"
    tolerances_path = OUTPUT_DIR / "five_minute_model_tolerance_pass_rates.csv"
    results.to_csv(results_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    tolerances.to_csv(tolerances_path, index=False)

    print("\n=== FIVE-MINUTE REGRESSION METRICS ===")
    print(metrics.to_string(index=False, formatters={
        "MAE": lambda x: f"{x:.6f}", "RMSE": lambda x: f"{x:.6f}",
        "R2": lambda x: f"{x:.6f}", "Mean bias (pred-actual)": lambda x: f"{x:.6f}"
    }))
    print("\n=== TOLERANCE PASS RATES ===")
    print(tolerances.to_string(index=False, formatters={"Pass rate (%)": lambda x: f"{x:.2f}%"}))

    # Optional statistical Low/Medium/High classification. Thresholds are computed from
    # training data only (tertiles), then applied to actual and predicted future values.
    if TRAIN_PATH.exists():
        train = load_data(TRAIN_PATH)
        f1_rows = []
        print("\n=== LOW / MEDIUM / HIGH CLASSIFICATION (TRAINING TERTILES) ===")
        for target in TARGETS:
            q = train[target].dropna().quantile([1/3, 2/3]).to_numpy()
            valid = results[[f"Predicted {target}", f"Actual {target} at forecast"]].dropna()
            y_true = np.digitize(valid[f"Actual {target} at forecast"].to_numpy(), q, right=True)
            y_pred = np.digitize(valid[f"Predicted {target}"].to_numpy(), q, right=True)
            f1_rows.append({
                "Target": target, "N": len(valid),
                "Training tertile 1 threshold": q[0], "Training tertile 2 threshold": q[1],
                "Accuracy": accuracy_score(y_true, y_pred),
                "Macro precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
                "Macro recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
                "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            })
        f1_df = pd.DataFrame(f1_rows)
        f1_path = OUTPUT_DIR / "five_minute_model_category_metrics.csv"
        f1_df.to_csv(f1_path, index=False)
        print(f1_df.to_string(index=False, formatters={
            "Accuracy": lambda x: f"{x:.4f}", "Macro precision": lambda x: f"{x:.4f}",
            "Macro recall": lambda x: f"{x:.4f}", "Macro F1": lambda x: f"{x:.4f}"
        }))
        print("Note: tertile categories are statistical bins, not safety thresholds.")

    print("\nRows evaluated:", len(results))
    print("Skipped for missing lag history/values:", skip_history)
    print("Skipped because exact T+5 reading was unavailable:", skip_future)
    print("\nSaved files:")
    print(" -", results_path)
    print(" -", metrics_path)
    print(" -", tolerances_path)
    if TRAIN_PATH.exists():
        print(" -", OUTPUT_DIR / "five_minute_model_category_metrics.csv")


if __name__ == "__main__":
    main()
