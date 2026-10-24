"""
Analyze where the current Random Forest is underperforming.

Run from the project root:
    python analyze_errors.py

Reads:
    data/training_data_2.csv
    data/ventilation_test.csv
    models/environment_forecasting_model.joblib

Writes:
    models/error_analysis.csv
    models/test_predictions.csv

This script does not retrain or overwrite the model.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = Path("data")
MODEL_PATH = Path("models") / "environment_forecasting_model.joblib"
TEST_PATH = DATA_DIR / "ventilation_test.csv"
OUTPUT_DIR = Path("models")
TIME_COL = "time"


def load_test_data(path: Path, targets: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}")
    df = pd.read_csv(path)

    required = {TIME_COL, "Occupant count", *targets}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Test CSV is missing required columns: {missing}")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    if df[TIME_COL].isna().any():
        raise ValueError("The test CSV contains invalid timestamps.")

    df = df.sort_values(TIME_COL).drop_duplicates(TIME_COL).reset_index(drop=True)
    for col in ["Occupant count", *targets]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["Occupant count", *targets]).reset_index(drop=True)


def make_test_samples(df: pd.DataFrame, targets: list[str],
                      lags: list[int], horizon: int,
                      feature_names: list[str]):
    by_time = df.set_index(TIME_COL)
    available_times = set(by_time.index)
    xs, ys, sample_times = [], [], []

    for current_time in by_time.index:
        needed = [current_time - pd.Timedelta(minutes=lag) for lag in lags]
        future_time = current_time + pd.Timedelta(minutes=horizon)

        # Require exact timestamps for every lag and the future target.
        if not all(t in available_times for t in needed):
            continue
        if future_time not in available_times:
            continue

        row = []
        valid = True
        for lag in lags:
            historical = by_time.loc[current_time - pd.Timedelta(minutes=lag)]
            for col in targets:
                value = historical[col]
                if pd.isna(value):
                    valid = False
                row.append(float(value))
            if lag == 0:
                occ = by_time.loc[current_time, "Occupant count"]
                if pd.isna(occ):
                    valid = False
                row.append(float(occ))

        future = by_time.loc[future_time, targets]
        if not valid or future.isna().any():
            continue

        xs.append(row)
        ys.append(future.to_numpy(dtype=float))
        sample_times.append(current_time)

    if not xs:
        raise ValueError("No valid test samples were generated.")

    return (
        pd.DataFrame(xs, columns=feature_names),
        np.asarray(ys, dtype=float),
        pd.to_datetime(sample_times),
    )


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {MODEL_PATH}. Run `python train_model.py` first."
        )

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    targets = list(bundle["target_names"])
    lags = list(bundle["lags_minutes"])
    horizon = int(bundle["horizon_minutes"])
    feature_names = list(bundle["feature_names"])

    df = load_test_data(TEST_PATH, targets)
    X, y_true, sample_times = make_test_samples(
        df, targets, lags, horizon, feature_names
    )
    y_pred = model.predict(X)
    persistence_cols = [f"{col}__lag_0m" for col in targets]
    y_persistence = X[persistence_cols].to_numpy(dtype=float)

    print(f"Test samples analyzed: {len(X):,}")
    print(f"Forecast horizon:      {horizon} minutes")
    print("\nPer-target error comparison (lower MAE is better):")
    print("-" * 92)

    report = []
    for i, target in enumerate(targets):
        abs_error = np.abs(y_true[:, i] - y_pred[:, i])
        base_error = np.abs(y_true[:, i] - y_persistence[:, i])
        report.append({
            "target": target,
            "persistence_MAE": mean_absolute_error(y_true[:, i], y_persistence[:, i]),
            "random_forest_MAE": mean_absolute_error(y_true[:, i], y_pred[:, i]),
            "MAE_difference_RF_minus_persistence": (
                mean_absolute_error(y_true[:, i], y_pred[:, i])
                - mean_absolute_error(y_true[:, i], y_persistence[:, i])
            ),
            "RF_RMSE": np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])),
            "RF_error_p90": np.percentile(abs_error, 90),
            "persistence_error_p90": np.percentile(base_error, 90),
        })

    report_df = pd.DataFrame(report).sort_values(
        "MAE_difference_RF_minus_persistence", ascending=False
    )
    print(report_df.to_string(index=False, formatters={
        col: (lambda x: f"{x:.3f}")
        for col in report_df.select_dtypes(include="number").columns
    }))

    # Detailed sample-level predictions/errors for plots and investigation.
    out = pd.DataFrame({"time": sample_times})
    for i, target in enumerate(targets):
        safe_name = (
            target.replace(" ", "_")
            .replace("(", "").replace(")", "")
            .replace("/", "_")
        )
        out[f"actual__{safe_name}"] = y_true[:, i]
        out[f"rf_pred__{safe_name}"] = y_pred[:, i]
        out[f"persistence_pred__{safe_name}"] = y_persistence[:, i]
        out[f"rf_abs_error__{safe_name}"] = np.abs(y_true[:, i] - y_pred[:, i])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "error_analysis.csv"
    predictions_path = OUTPUT_DIR / "test_predictions.csv"
    report_df.to_csv(report_path, index=False)
    out.to_csv(predictions_path, index=False)

    print(f"\nSaved summary:     {report_path}")
    print(f"Saved predictions:{predictions_path}")
    print("\nNext: share this terminal output. We will use it to decide which targets")
    print("need different features/models and which should retain the baseline.")


if __name__ == "__main__":
    main()
