from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
TRAIN_FILE = DATA_DIR / "training_data_2.csv"
HORIZON_MINUTES = 5
LAGS_MINUTES = [0, 1, 2, 3, 5]
TARGET = "Room CO2 (ppm)"
EXCLUDED = {"Occupant count (ML estimate)"}

def load_data(path):
    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError(f"{path} must contain a 'time' column.")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    df = df.drop_duplicates(subset=["time"]).set_index("time")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def make_samples(df):
    if TARGET not in df.columns:
        raise ValueError(f"Could not find target column: {TARGET}")
    times = df.index
    available = set(times)
    feature_cols = [c for c in df.columns if c not in EXCLUDED]
    rows, targets, timestamps = [], [], []

    for t in times:
        target_time = t + pd.Timedelta(minutes=HORIZON_MINUTES)
        if target_time not in available:
            continue
        if any(t - pd.Timedelta(minutes=lag) not in available for lag in LAGS_MINUTES):
            continue

        row = {}
        valid = True
        for lag in LAGS_MINUTES:
            source = df.loc[t - pd.Timedelta(minutes=lag)]
            for col in feature_cols:
                value = source[col]
                if pd.isna(value):
                    valid = False
                    break
                row[f"{col}__lag{lag}"] = float(value)
            if not valid:
                break
        target_value = df.loc[target_time, TARGET]
        if not valid or pd.isna(target_value):
            continue

        rows.append(row)
        targets.append(float(target_value))
        timestamps.append(t)

    X = pd.DataFrame(rows)
    y = np.asarray(targets, dtype=float)
    t = pd.DatetimeIndex(timestamps)
    return X, y, t

def metrics(y, pred):
    return {
        "MAE_ppm": mean_absolute_error(y, pred),
        "RMSE_ppm": mean_squared_error(y, pred) ** 0.5,
    }

def trend_forecast(df, timestamps):
    """Linear extrapolation using only readings at prediction time and earlier."""
    preds = []
    for t in timestamps:
        values = []
        # Fit a simple least-squares line to CO2 readings at t, t-1, t-2, t-3, t-5.
        # Exact timestamps are required, so gaps are not bridged.
        for lag in LAGS_MINUTES:
            reading = df.loc[t - pd.Timedelta(minutes=lag), TARGET]
            values.append(float(reading))
        x = np.asarray([-lag for lag in LAGS_MINUTES], dtype=float)
        y = np.asarray(values, dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        preds.append(intercept + slope * HORIZON_MINUTES)
    return np.asarray(preds)

def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {TRAIN_FILE}. Run this script from your project root: "
            r"C:\Users\HP\OneDrive\Documents\AI_Smart_Ventilation_System"
        )

    df = load_data(TRAIN_FILE)
    X, y, timestamps = make_samples(df)
    if len(y) < 20:
        raise ValueError(f"Only {len(y)} usable samples were found.")

    # Chronological split, with an embargo of one forecast horizon between train and validation.
    split_index = int(len(y) * 0.80)
    split_time = timestamps[split_index]
    validation_start = split_time + pd.Timedelta(minutes=HORIZON_MINUTES)
    train_mask = timestamps < split_time
    val_mask = timestamps >= validation_start

    val_times = timestamps[val_mask]
    y_val = y[val_mask]
    val_current = np.asarray([df.loc[t, TARGET] for t in val_times], dtype=float)
    persistence = val_current.copy()
    trend = trend_forecast(df, val_times)

    # Also report a simple linear-trend forecast clipped only to plausible nonnegative CO2.
    # Keep the unclipped version as the primary trend result; the clipped result is secondary.
    trend_clipped = np.maximum(trend, 0)

    results = []
    for name, pred in [
        ("Persistence baseline", persistence),
        ("Linear trend baseline", trend),
        ("Linear trend (minimum 0 ppm)", trend_clipped),
    ]:
        row = {"model": name, "samples": len(y_val)}
        row.update(metrics(y_val, pred))
        results.append(row)

    results_df = pd.DataFrame(results)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODEL_DIR / "room_co2_trend_comparison_validation.csv"
    predictions_path = MODEL_DIR / "room_co2_trend_predictions_validation.csv"

    # Save validation predictions for later inspection.
    pred_df = pd.DataFrame({
        "timestamp": val_times,
        "current_CO2_ppm": val_current,
        "actual_CO2_plus_5min_ppm": y_val,
        "persistence_prediction_ppm": persistence,
        "linear_trend_prediction_ppm": trend,
        "linear_trend_clipped_prediction_ppm": trend_clipped,
        "actual_change_5min_ppm": y_val - val_current,
        "trend_prediction_error_ppm": trend - y_val,
        "persistence_error_ppm": persistence - y_val,
    })
    results_df.to_csv(out_path, index=False)
    pred_df.to_csv(predictions_path, index=False)

    print("ROOM CO2 FORECAST BASELINE COMPARISON")
    print(f"Forecast horizon: {HORIZON_MINUTES} minutes")
    print(f"Usable samples: {len(y)}")
    print(f"Training period ends before: {split_time}")
    print(f"Validation period starts at: {validation_start}")
    print(f"Validation samples: {len(y_val)}")
    print("\nValidation results (lower is better):")
    print(results_df.round(3).to_string(index=False))
    baseline_mae = results_df.loc[results_df["model"] == "Persistence baseline", "MAE_ppm"].iloc[0]
    trend_mae = results_df.loc[results_df["model"] == "Linear trend baseline", "MAE_ppm"].iloc[0]
    improvement = (baseline_mae - trend_mae) / baseline_mae * 100
    print(f"\nTrend MAE change vs persistence: {improvement:+.2f}% improvement")
    if trend_mae < baseline_mae:
        print("Interpretation: the trend baseline improved MAE on this validation split.")
    else:
        print("Interpretation: the trend baseline did not improve MAE on this validation split.")
    print("\nSaved:")
    print(f"  {out_path}")
    print(f"  {predictions_path}")
    print("\nNo machine-learning model was trained, and no existing model or dataset was modified.")

if __name__ == "__main__":
    main()
