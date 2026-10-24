from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
TRAIN_FILE = DATA_DIR / "training_data_2.csv"
TARGET = "Room CO2 (ppm)"
OCCUPANCY = "Occupant count"
HORIZON = 5
LAGS = [0, 1, 2, 3, 5]
EXCLUDED = {"Occupant count (ML estimate)"}

def load_data(path):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    df = df.drop_duplicates(subset=["time"]).set_index("time")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def make_samples(df):
    if TARGET not in df.columns:
        raise ValueError(f"Missing required target column: {TARGET}")
    times = df.index
    time_set = set(times)
    feature_cols = [c for c in df.columns if c not in EXCLUDED]
    rows, targets, sample_times = [], [], []
    for t in times:
        target_t = t + pd.Timedelta(minutes=HORIZON)
        if target_t not in time_set:
            continue
        if any(t - pd.Timedelta(minutes=lag) not in time_set for lag in LAGS):
            continue
        features, valid = {}, True
        for lag in LAGS:
            src = df.loc[t - pd.Timedelta(minutes=lag)]
            for col in feature_cols:
                value = src[col]
                if pd.isna(value):
                    valid = False
                    break
                features[f"{col}__lag{lag}"] = float(value)
            if not valid:
                break
        target = df.loc[target_t, TARGET]
        if not valid or pd.isna(target):
            continue
        rows.append(features)
        targets.append(float(target))
        sample_times.append(t)
    return pd.DataFrame(rows), np.asarray(targets), pd.DatetimeIndex(sample_times)

def mae_rmse(actual, pred):
    return mean_absolute_error(actual, pred), mean_squared_error(actual, pred) ** 0.5

def trend_predict(df, times):
    pred = []
    x = np.asarray([-lag for lag in LAGS], dtype=float)
    for t in times:
        y = np.asarray([
            float(df.loc[t - pd.Timedelta(minutes=lag), TARGET])
            for lag in LAGS
        ])
        slope, intercept = np.polyfit(x, y, 1)
        pred.append(intercept + slope * HORIZON)
    return np.asarray(pred)

def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Missing {TRAIN_FILE}. Run from the project root.")
    df = load_data(TRAIN_FILE)
    X, y, times = make_samples(df)
    if len(y) < 100:
        raise ValueError(f"Too few usable samples: {len(y)}")

    # Keep the same chronological validation boundary used in the previous trend experiment.
    split_idx = int(len(y) * 0.80)
    split_time = times[split_idx]
    val_start = split_time + pd.Timedelta(minutes=HORIZON)
    train_mask = times < split_time
    val_mask = times >= val_start
    X_train, y_train = X.loc[train_mask].copy(), y[train_mask]
    X_val, y_val = X.loc[val_mask].copy(), y[val_mask]
    val_times = times[val_mask]

    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, max_features=0.8,
            random_state=42, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=5, max_features=0.8,
            random_state=42, n_jobs=-1
        ),
        "Hist Gradient Boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                l2_regularization=1.0, random_state=42
            )
        ),
    }

    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_val)

    current = np.asarray([df.loc[t, TARGET] for t in val_times], dtype=float)
    trend = trend_predict(df, val_times)
    persistence = current.copy()

    results = []
    for name, pred in [
        ("Persistence baseline", persistence),
        ("Linear trend baseline", trend),
        *[(name, pred) for name, pred in predictions.items()],
    ]:
        mae, rmse = mae_rmse(y_val, pred)
        results.append({"model": name, "samples": len(y_val), "MAE_ppm": mae, "RMSE_ppm": rmse})

    results_df = pd.DataFrame(results).sort_values("MAE_ppm")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    result_path = MODEL_DIR / "room_co2_fair_model_comparison_validation.csv"
    pred_path = MODEL_DIR / "room_co2_fair_model_predictions_validation.csv"
    results_df.to_csv(result_path, index=False)

    pred_df = pd.DataFrame({
        "timestamp": val_times,
        "actual_CO2_plus_5min_ppm": y_val,
        "current_CO2_ppm": current,
        "persistence_prediction_ppm": persistence,
        "linear_trend_prediction_ppm": trend,
    })
    for name, pred in predictions.items():
        col = name.lower().replace(" ", "_") + "_prediction_ppm"
        pred_df[col] = pred
    pred_df.to_csv(pred_path, index=False)

    print("FAIR ROOM CO2 MODEL COMPARISON — VALIDATION ONLY")
    print(f"Usable samples: {len(y)}")
    print(f"Training samples: {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")
    print(f"Training period ends before: {split_time}")
    print(f"Validation starts at: {val_start}")
    print("\nResults sorted by MAE (lower is better):")
    print(results_df.round(3).to_string(index=False))
    base_mae = results_df.loc[results_df.model == "Persistence baseline", "MAE_ppm"].iloc[0]
    print("\nMAE improvement vs persistence:")
    for _, row in results_df.iterrows():
        if row["model"] == "Persistence baseline":
            continue
        improvement = (base_mae - row["MAE_ppm"]) / base_mae * 100
        print(f"  {row['model']}: {improvement:+.2f}%")
    print("\nSaved:")
    print(f"  {result_path}")
    print(f"  {pred_path}")
    print("\nNo model files were saved or overwritten. The holdout test dataset was not used.")

if __name__ == "__main__":
    main()
