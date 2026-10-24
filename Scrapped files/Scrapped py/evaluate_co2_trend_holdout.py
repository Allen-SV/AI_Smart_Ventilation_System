from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_FILE = Path("data/ventilation_test.csv")
OUTPUT_DIR = Path("models")
TARGET = "Room CO2 (ppm)"
HORIZON = 5
LAGS = [0, 1, 2, 3, 5]

def load_data(path):
    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError("The dataset must contain a 'time' column.")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    df = df.drop_duplicates(subset=["time"]).set_index("time")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")
    return df

def metrics(actual, predicted):
    return (
        mean_absolute_error(actual, predicted),
        mean_squared_error(actual, predicted) ** 0.5
    )

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {DATA_FILE}. Run this script from your project root."
        )

    df = load_data(DATA_FILE)
    time_set = set(df.index)
    records = []

    # Require exact timestamps at every required lag and at the 5-minute target.
    # This avoids bridging the large sampling gaps in the dataset.
    for t in df.index:
        target_t = t + pd.Timedelta(minutes=HORIZON)
        required_times = [t - pd.Timedelta(minutes=lag) for lag in LAGS]
        if target_t not in time_set or any(rt not in time_set for rt in required_times):
            continue

        values = [df.loc[t - pd.Timedelta(minutes=lag), TARGET] for lag in LAGS]
        actual = df.loc[target_t, TARGET]
        if any(pd.isna(v) for v in values) or pd.isna(actual):
            continue

        current = float(df.loc[t, TARGET])
        one_minute_ago = float(df.loc[t - pd.Timedelta(minutes=1), TARGET])
        rate_per_minute = current - one_minute_ago
        raw_trend = current + HORIZON * rate_per_minute
        capped_trend = current + np.clip(HORIZON * rate_per_minute, -75, 75)

        # Previous experiment's 5-minute linear regression trend baseline.
        x = np.asarray([-lag for lag in LAGS], dtype=float)
        y = np.asarray(values, dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        linear_trend = intercept + slope * HORIZON

        records.append({
            "timestamp": t,
            "current_CO2_ppm": current,
            "actual_CO2_plus_5min_ppm": float(actual),
            "persistence_prediction_ppm": current,
            "linear_trend_prediction_ppm": float(linear_trend),
            "one_minute_trend_prediction_ppm": float(raw_trend),
            "capped_one_minute_trend_prediction_ppm": float(capped_trend),
            "actual_change_5min_ppm": float(actual) - current,
            "one_minute_rate_ppm_per_min": rate_per_minute,
        })

    pred_df = pd.DataFrame(records)
    if len(pred_df) < 10:
        raise ValueError(
            f"Only {len(pred_df)} valid samples found. Check timestamp cadence and column names."
        )

    actual = pred_df["actual_CO2_plus_5min_ppm"].to_numpy()
    candidates = [
        ("Persistence baseline", "persistence_prediction_ppm"),
        ("Linear trend baseline", "linear_trend_prediction_ppm"),
        ("1-minute trend, uncapped", "one_minute_trend_prediction_ppm"),
        ("1-minute trend, adjustment capped ±75 ppm", "capped_one_minute_trend_prediction_ppm"),
    ]

    results = []
    for name, col in candidates:
        mae, rmse = metrics(actual, pred_df[col].to_numpy())
        results.append({
            "model_or_baseline": name,
            "samples": len(pred_df),
            "MAE_ppm": mae,
            "RMSE_ppm": rmse,
        })
    results_df = pd.DataFrame(results)
    persistence_mae = results_df.loc[
        results_df["model_or_baseline"] == "Persistence baseline", "MAE_ppm"
    ].iloc[0]
    results_df["MAE_improvement_vs_persistence_percent"] = (
        (persistence_mae - results_df["MAE_ppm"]) / persistence_mae * 100
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / "co2_holdout_trend_evaluation.csv"
    predictions_path = OUTPUT_DIR / "co2_holdout_trend_predictions.csv"
    results_df.to_csv(metrics_path, index=False)
    pred_df.to_csv(predictions_path, index=False)

    print("FINAL HOLDOUT EVALUATION — ROOM CO2")
    print("Forecast horizon: 5 minutes")
    print("Dataset: data/ventilation_test.csv")
    print(f"Valid holdout samples: {len(pred_df)}")
    print(f"First prediction timestamp: {pred_df['timestamp'].min()}")
    print(f"Last prediction timestamp: {pred_df['timestamp'].max()}")
    print("\nResults (lower MAE and RMSE are better):")
    print(results_df.round(3).to_string(index=False))
    winner = results_df.sort_values("MAE_ppm").iloc[0]
    print(f"\nLowest holdout MAE: {winner['model_or_baseline']} ({winner['MAE_ppm']:.3f} ppm)")
    print("\nSaved:")
    print(f"  {metrics_path}")
    print(f"  {predictions_path}")
    print("\nImportant: this holdout result is for final evaluation, not further tuning.")
    print("No existing model or source dataset was modified.")

if __name__ == "__main__":
    main()
