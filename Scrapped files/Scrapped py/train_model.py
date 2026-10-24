"""
Train a multivariate 5-minute-ahead environmental forecaster.

Run from the project root:
    python train_model.py

Expected files:
    data/training_data_2.csv
    data/ventilation_test.csv

This script:
- Uses training_data_2.csv for model development.
- Makes a chronological train/validation split.
- Never creates a sample across a timestamp gap.
- Excludes the existing "Occupant count (ML estimate)" column.
- Compares a Random Forest multi-output model with a persistence baseline.
- Evaluates ventilation_test.csv only after fitting/model selection.
- Saves the fitted model and metadata to models/environment_forecasting_model.joblib.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
TRAIN_FILE = DATA_DIR / "training_data_2.csv"
TEST_FILE = DATA_DIR / "ventilation_test.csv"
TIME_COL = "time"

# Forecast the values five minutes after the latest input timestamp.
HORIZON_MINUTES = 5

# Start with measurements present in training_data_2.csv and test file.
TARGETS = [
    "Room CO2 (ppm)",
    "Temperature (°C)",
    "Relative Humidity (%)",
    "Pressure (mbar)",
    "Supply airflow (l/s)",
    "Extract airflow (l/s)",
    "Supply air CO2 (ppm)",
    "Exhaust air CO2 (ppm)",
]

# Current values plus lagged readings; occupancy is a known input.
LAGS_MINUTES = [0, 1, 2, 3, 5]
OCCUPANCY_COL = "Occupant count"
FORBIDDEN_FEATURES = {"Occupant count (ML estimate)"}


def load_and_validate(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Put the CSV file in the project's data folder."
        )

    df = pd.read_csv(path)
    required = {TIME_COL, OCCUPANCY_COL, *TARGETS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    if df[TIME_COL].isna().any():
        raise ValueError(f"{path} contains invalid timestamps.")

    df = df.sort_values(TIME_COL).drop_duplicates(TIME_COL).reset_index(drop=True)
    for col in [OCCUPANCY_COL, *TARGETS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # No imputation across long gaps. Drop rows with missing model values.
    df = df.dropna(subset=[OCCUPANCY_COL, *TARGETS]).reset_index(drop=True)
    return df


def make_samples(df: pd.DataFrame):
    """
    Build a sample only when every required lag and the future target timestamp
    exists at the exact expected minute. This avoids treating gaps as adjacent.
    """
    by_time = df.set_index(TIME_COL)
    available_times = set(by_time.index)
    feature_names = []
    for lag in LAGS_MINUTES:
        feature_names.extend([f"{col}__lag_{lag}m" for col in TARGETS])
        if lag == 0:
            feature_names.append(f"{OCCUPANCY_COL}__current")

    xs, ys, sample_times = [], [], []
    for current_time in by_time.index:
        needed = [current_time - pd.Timedelta(minutes=lag) for lag in LAGS_MINUTES]
        future_time = current_time + pd.Timedelta(minutes=HORIZON_MINUTES)

        if not all(t in available_times for t in needed):
            continue
        if future_time not in available_times:
            continue

        row = []
        valid = True
        for lag in LAGS_MINUTES:
            historical = by_time.loc[current_time - pd.Timedelta(minutes=lag)]
            for col in TARGETS:
                value = historical[col]
                if pd.isna(value):
                    valid = False
                row.append(float(value))
            if lag == 0:
                occ = by_time.loc[current_time][OCCUPANCY_COL]
                if pd.isna(occ):
                    valid = False
                row.append(float(occ))

        future = by_time.loc[future_time, TARGETS]
        if not valid or future.isna().any():
            continue

        xs.append(row)
        ys.append(future.to_numpy(dtype=float))
        sample_times.append(current_time)

    if not xs:
        raise ValueError(
            "No valid samples were created. Check timestamp format, cadence, "
            "column names and whether the dataset contains enough continuous minutes."
        )

    X = pd.DataFrame(xs, columns=feature_names)
    Y = pd.DataFrame(ys, columns=TARGETS)
    times = pd.Series(pd.to_datetime(sample_times), name=TIME_COL)
    return X, Y, times


def metric_rows(y_true, y_pred, targets, label):
    rows = []
    for i, target in enumerate(targets):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        rows.append({"model": label, "target": target, "MAE": mae, "RMSE": rmse})
    return rows


def print_metrics(rows, heading):
    print(f"\n{heading}")
    print("-" * len(heading))
    frame = pd.DataFrame(rows)
    if frame.empty:
        print("No metrics.")
        return
    print(frame.to_string(index=False, formatters={
        "MAE": lambda x: f"{x:.3f}",
        "RMSE": lambda x: f"{x:.3f}",
    }))


def main():
    warnings.filterwarnings("default")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_and_validate(TRAIN_FILE)
    test_df = load_and_validate(TEST_FILE)

    print(f"Development data: {len(train_df):,} rows")
    print(f"Final test data:  {len(test_df):,} rows")
    print(f"Forecast horizon: {HORIZON_MINUTES} minutes")
    print(f"Targets: {', '.join(TARGETS)}")

    X_all, Y_all, times_all = make_samples(train_df)

    # Chronological split based on the current/input timestamp.
    cutoff = times_all.quantile(0.80)
    # Leave a horizon-sized embargo so no training target lands in the
    # validation period. Validation samples begin at the chronological cutoff.
    train_mask = (times_all + pd.Timedelta(minutes=HORIZON_MINUTES)) < cutoff
    val_mask = times_all >= cutoff

    X_train, Y_train = X_all.loc[train_mask], Y_all.loc[train_mask]
    X_val, Y_val = X_all.loc[val_mask], Y_all.loc[val_mask]

    if len(X_train) < 20 or len(X_val) < 10:
        raise ValueError(
            f"Not enough valid chronological samples (train={len(X_train)}, "
            f"validation={len(X_val)}). Check timestamp gaps or choose another horizon."
        )

    print(f"\nValid development samples: {len(X_all):,}")
    print(f"Training samples:          {len(X_train):,}")
    print(f"Validation samples:        {len(X_val):,}")
    print(f"Validation starts at:      {cutoff}")

    model = RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, Y_train)

    val_pred = model.predict(X_val)
    # Persistence baseline: the current values are the lag-0 environmental inputs.
    persistence_cols = [f"{col}__lag_0m" for col in TARGETS]
    persistence_pred = X_val[persistence_cols].to_numpy()

    val_rows = metric_rows(Y_val.to_numpy(), persistence_pred, TARGETS, "Persistence baseline")
    val_rows += metric_rows(Y_val.to_numpy(), val_pred, TARGETS, "Random Forest")
    print_metrics(val_rows, "Chronological validation metrics (original units)")

    # Final holdout is not used for model fitting or selection.
    X_test, Y_test, _ = make_samples(test_df)
    test_pred = model.predict(X_test)
    test_persistence = X_test[persistence_cols].to_numpy()
    test_rows = metric_rows(Y_test.to_numpy(), test_persistence, TARGETS, "Persistence baseline")
    test_rows += metric_rows(Y_test.to_numpy(), test_pred, TARGETS, "Random Forest")
    print_metrics(test_rows, "Final unseen test metrics (original units)")

    # Retrain selected model on all development data only after validation.
    final_model = RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X_all, Y_all)

    bundle = {
        "model": final_model,
        "feature_names": list(X_all.columns),
        "target_names": TARGETS,
        "horizon_minutes": HORIZON_MINUTES,
        "lags_minutes": LAGS_MINUTES,
        "occupancy_column": OCCUPANCY_COL,
        "time_column": TIME_COL,
        "model_type": "RandomForestRegressor multi-output",
        "note": "Use only trusted, current sensor readings at inference time. "
                "The existing ML occupancy estimate is intentionally excluded.",
    }
    model_path = MODEL_DIR / "environment_forecasting_model.joblib"
    joblib.dump(bundle, model_path)

    report_path = MODEL_DIR / "evaluation_metrics.csv"
    pd.DataFrame(val_rows + test_rows).to_csv(report_path, index=False)

    print(f"\nSaved model bundle: {model_path}")
    print(f"Saved metrics:      {report_path}")
    print("\nImportant: test metrics are an evaluation, not a guarantee of real-world performance.")
    print("Review outliers and sensor units before using the forecasts to set control thresholds.")


if __name__ == "__main__":
    main()
