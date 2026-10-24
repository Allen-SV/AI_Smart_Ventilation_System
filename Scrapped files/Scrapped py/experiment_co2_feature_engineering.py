from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_FILE = Path("data/training_data_2.csv")
OUTPUT_DIR = Path("models")
TARGET = "Room CO2 (ppm)"
OCCUPANCY = "Occupant count"
HORIZON = 5
EXCLUDED = {"Occupant count (ML estimate)"}

def load_data(path):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    df = df.drop_duplicates(subset=["time"]).set_index("time")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def build_sample_frame(df):
    times = df.index
    time_set = set(times)
    samples = []
    for t in times:
        future_t = t + pd.Timedelta(minutes=HORIZON)
        if future_t not in time_set:
            continue
        # Require exact historical timestamps so gaps are never silently bridged.
        required = [t - pd.Timedelta(minutes=m) for m in [0, 1, 2, 3, 5, 10]]
        if any(rt not in time_set for rt in required):
            continue
        row = {"timestamp": t}
        for lag in [0, 1, 2, 3, 5, 10]:
            src_t = t - pd.Timedelta(minutes=lag)
            for col in df.columns:
                if col in EXCLUDED:
                    continue
                value = df.loc[src_t, col]
                if pd.isna(value):
                    row = None
                    break
                row[f"{col}__lag{lag}"] = float(value)
            if row is None:
                break
        if row is None:
            continue
        actual = df.loc[future_t, TARGET]
        if pd.isna(actual):
            continue
        row["actual_future"] = float(actual)
        row["current_co2"] = float(df.loc[t, TARGET])
        row["delta_1m"] = float(df.loc[t, TARGET] - df.loc[t - pd.Timedelta(minutes=1), TARGET])
        row["delta_2m"] = float(df.loc[t, TARGET] - df.loc[t - pd.Timedelta(minutes=2), TARGET]) / 2
        row["delta_3m"] = float(df.loc[t, TARGET] - df.loc[t - pd.Timedelta(minutes=3), TARGET]) / 3
        row["delta_5m"] = float(df.loc[t, TARGET] - df.loc[t - pd.Timedelta(minutes=5), TARGET]) / 5
        row["delta_10m"] = float(df.loc[t, TARGET] - df.loc[t - pd.Timedelta(minutes=10), TARGET]) / 10
        row["occupancy_now"] = float(df.loc[t, OCCUPANCY]) if OCCUPANCY in df.columns else np.nan
        row["occupancy_5m_ago"] = float(df.loc[t - pd.Timedelta(minutes=5), OCCUPANCY]) if OCCUPANCY in df.columns else np.nan
        row["occupancy_change_5m"] = row["occupancy_now"] - row["occupancy_5m_ago"]
        row["time_of_day_hour"] = t.hour + t.minute / 60
        samples.append(row)
    return pd.DataFrame(samples)

def score(actual, pred):
    return mean_absolute_error(actual, pred), mean_squared_error(actual, pred) ** 0.5

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing {DATA_FILE}. Run from the project root.")
    df = load_data(DATA_FILE)
    data = build_sample_frame(df)
    if len(data) < 100:
        raise ValueError(f"Only {len(data)} usable samples found.")

    # Same approximate chronological split as the earlier comparison; 5-minute embargo.
    split_idx = int(len(data) * 0.80)
    split_time = pd.Timestamp(data.iloc[split_idx]["timestamp"])
    validation_start = split_time + pd.Timedelta(minutes=HORIZON)
    train = data[data["timestamp"] < split_time].copy()
    val = data[data["timestamp"] >= validation_start].copy()
    if len(val) == 0:
        raise ValueError("No validation samples after the chronological embargo.")

    actual = val["actual_future"].to_numpy()
    current = val["current_co2"].to_numpy()
    delta_1 = val["delta_1m"].to_numpy()
    delta_3 = val["delta_3m"].to_numpy()
    delta_5 = val["delta_5m"].to_numpy()
    delta_10 = val["delta_10m"].to_numpy()

    # Trend extrapolation variants. Rates are ppm/minute.
    preds = {
        "Persistence": current,
        "Trend 1-minute": current + 5 * delta_1,
        "Trend 3-minute": current + 5 * delta_3,
        "Trend 5-minute": current + 5 * delta_5,
        "Trend 10-minute": current + 5 * delta_10,
        "Blend 50% 3m + 50% 10m": current + 5 * (0.5 * delta_3 + 0.5 * delta_10),
        "Blend 70% 3m + 30% 10m": current + 5 * (0.7 * delta_3 + 0.3 * delta_10),
    }

    # Cap the trend adjustment, not the CO2 prediction itself, to limit extreme extrapolation.
    for name in ["Trend 1-minute", "Trend 3-minute", "Trend 5-minute", "Trend 10-minute",
                 "Blend 50% 3m + 50% 10m", "Blend 70% 3m + 30% 10m"]:
        adjustment = preds[name] - current
        preds[name + " (adjustment capped ±75 ppm)"] = current + np.clip(adjustment, -75, 75)

    # Add models trained only on the training period using selected, available-at-time features.
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    engineered_features = [
        "current_co2", "delta_1m", "delta_2m", "delta_3m", "delta_5m", "delta_10m",
        "occupancy_now", "occupancy_5m_ago", "occupancy_change_5m", "time_of_day_hour",
    ]
    # Add current values and recent lag values for all other environmental sensors.
    sensor_cols = [
        c for c in df.columns
        if c not in EXCLUDED and c != TARGET
    ]
    for lag in [0, 1, 3, 5]:
        for c in sensor_cols:
            key = f"{c}__lag{lag}"
            if key in data.columns and key not in engineered_features:
                engineered_features.append(key)

    X_train = train[engineered_features]
    X_val = val[engineered_features]
    et = ExtraTreesRegressor(
        n_estimators=300, min_samples_leaf=5, max_features=0.8,
        random_state=42, n_jobs=-1
    )
    hgb = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42
        )
    )
    et.fit(X_train, train["actual_future"])
    hgb.fit(X_train, train["actual_future"])
    preds["Engineered Extra Trees"] = et.predict(X_val)
    preds["Engineered Hist Gradient Boosting"] = hgb.predict(X_val)

    results = []
    for name, prediction in preds.items():
        mae, rmse = score(actual, prediction)
        results.append({
            "model_or_baseline": name,
            "samples": len(actual),
            "MAE_ppm": mae,
            "RMSE_ppm": rmse,
        })
    results_df = pd.DataFrame(results).sort_values("MAE_ppm")
    baseline_mae = results_df.loc[
        results_df["model_or_baseline"] == "Persistence", "MAE_ppm"
    ].iloc[0]
    results_df["MAE_improvement_vs_persistence_percent"] = (
        (baseline_mae - results_df["MAE_ppm"]) / baseline_mae * 100
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "co2_feature_engineering_comparison.csv"
    predictions_path = OUTPUT_DIR / "co2_feature_engineering_predictions.csv"
    results_df.to_csv(results_path, index=False)

    pred_out = pd.DataFrame({
        "timestamp": val["timestamp"].to_numpy(),
        "actual_future_CO2_ppm": actual,
        "current_CO2_ppm": current,
        "occupancy_now": val["occupancy_now"].to_numpy(),
        "occupancy_change_5m": val["occupancy_change_5m"].to_numpy(),
        "actual_change_5m_ppm": actual - current,
    })
    for name, prediction in preds.items():
        safe = name.lower().replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "").replace("±", "plusminus").replace("=", "")
        while "__" in safe:
            safe = safe.replace("__", "_")
        pred_out[safe + "_prediction_ppm"] = prediction
    pred_out.to_csv(predictions_path, index=False)

    print("CO2 FEATURE-ENGINEERING EXPERIMENT — VALIDATION ONLY")
    print(f"Usable samples: {len(data)}")
    print(f"Training samples: {len(train)}")
    print(f"Validation samples: {len(val)}")
    print(f"Training period ends before: {split_time}")
    print(f"Validation starts at: {validation_start}")
    print("\nResults sorted by MAE (lower is better):")
    print(results_df.round(3).to_string(index=False))
    best = results_df.iloc[0]
    print(f"\nBest validation method: {best['model_or_baseline']}")
    print(f"MAE: {best['MAE_ppm']:.3f} ppm")
    print(f"MAE improvement vs persistence: {best['MAE_improvement_vs_persistence_percent']:+.2f}%")
    print("\nSaved:")
    print(f"  {results_path}")
    print(f"  {predictions_path}")
    print("\nNo holdout test data was used. No existing model or source dataset was changed.")

if __name__ == "__main__":
    main()
