from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
TRAIN_FILE = DATA_DIR / "training_data_2.csv"
TEST_FILE = DATA_DIR / "ventilation_test.csv"
TARGET = "Room CO2 (ppm)"
HORIZON_MINUTES = 5
LAGS = [0, 1, 2, 3, 5]
OCCUPANCY = "Occupant count"
EXCLUDED_FEATURES = ["Occupant count (ML estimate)"]

def load_data(path):
    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError(f"{path} has no 'time' column.")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates("time")
    for col in df.columns:
        if col != "time":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    needed = [TARGET, OCCUPANCY]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df.reset_index(drop=True)

def build_samples(df):
    # Inputs are only information available at or before each prediction time.
    # Labels are the room CO2 reading exactly five minutes later.
    df = df.set_index("time").sort_index()
    available_cols = [
        c for c in df.columns
        if c not in EXCLUDED_FEATURES and c != TARGET
    ]
    feature_cols = []
    for lag in LAGS:
        for c in available_cols:
            feature_cols.append(f"{c}__lag{lag}")
    feature_cols += [f"{TARGET}__lag{lag}" for lag in LAGS]

    records = []
    labels = []
    timestamps = []
    index_times = df.index
    lookup = set(index_times)

    for current_time in index_times:
        target_time = current_time + pd.Timedelta(minutes=HORIZON_MINUTES)
        if target_time not in lookup:
            continue
        row = {}
        valid = True
        for lag in LAGS:
            lag_time = current_time - pd.Timedelta(minutes=lag)
            if lag_time not in lookup:
                valid = False
                break
            source = df.loc[lag_time]
            for c in available_cols:
                row[f"{c}__lag{lag}"] = source[c]
            row[f"{TARGET}__lag{lag}"] = source[TARGET]
        if not valid:
            continue
        y = df.loc[target_time, TARGET]
        if pd.isna(y) or pd.isna(df.loc[current_time, TARGET]):
            continue
        records.append(row)
        labels.append(float(y))
        timestamps.append(current_time)

    X = pd.DataFrame(records, columns=feature_cols)
    y = np.asarray(labels, dtype=float)
    times = pd.DatetimeIndex(timestamps)
    # Drop rows with no usable features, but impute remaining feature gaps in the pipeline.
    keep = ~X.isna().all(axis=1).to_numpy()
    return X.loc[keep].reset_index(drop=True), y[keep], times[keep]

def metrics(y, pred):
    return (
        mean_absolute_error(y, pred),
        mean_squared_error(y, pred) ** 0.5,
    )

def main():
    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        raise FileNotFoundError(
            "Expected data/training_data_2.csv and data/ventilation_test.csv. "
            "Run this script from the project root."
        )

    train_df = load_data(TRAIN_FILE)
    test_df = load_data(TEST_FILE)
    X_all, y_all, t_all = build_samples(train_df)
    X_test, y_test, t_test = build_samples(test_df)

    if len(X_all) < 100 or len(X_test) < 50:
        raise ValueError(f"Not enough samples: train={len(X_all)}, test={len(X_test)}")

    # Chronological validation with an embargo so training labels do not cross cutoff.
    cutoff = t_all[int(len(t_all) * 0.80)]
    train_mask = (t_all + pd.Timedelta(minutes=HORIZON_MINUTES)) < cutoff
    val_mask = t_all >= cutoff
    X_train, y_train = X_all.loc[train_mask].copy(), y_all[train_mask]
    X_val, y_val = X_all.loc[val_mask].copy(), y_all[val_mask]
    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Chronological split produced an empty partition.")

    # Persistence prediction: current room CO2 value is the prediction.
    current_co2_col = f"{TARGET}__lag0"
    val_persistence = X_val[current_co2_col].to_numpy(dtype=float)
    test_persistence = X_test[current_co2_col].to_numpy(dtype=float)

    candidates = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, max_features=0.8,
            random_state=42, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=5, max_features=0.8,
            random_state=42, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42
        ),
    }

    print("ROOM CO2 FORECAST EXPERIMENT")
    print(f"Target: {TARGET}")
    print(f"Horizon: {HORIZON_MINUTES} minutes")
    print(f"Features: historical readings at lags {LAGS} minutes")
    print("Occupancy estimate column excluded; ground-truth Occupant count may be used.")
    print(f"Development samples: {len(X_all):,}")
    print(f"Training samples:    {len(X_train):,}")
    print(f"Validation samples:  {len(X_val):,}")
    print(f"Validation starts:   {cutoff}")
    print(f"Holdout test samples:{len(X_test):,}")
    print("\nValidation results (choose models using this split only):")
    val_rows = []
    fitted = {}
    for name, estimator in candidates.items():
        model = make_pipeline(SimpleImputer(strategy="median"), estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        pred = model.predict(X_val)
        mae, rmse = metrics(y_val, pred)
        base_mae, base_rmse = metrics(y_val, val_persistence)
        val_rows.append((name, mae, rmse, base_mae, base_rmse))
        fitted[name] = model
    val_table = pd.DataFrame(
        val_rows,
        columns=["model", "MAE", "RMSE", "persistence_MAE", "persistence_RMSE"]
    ).sort_values("MAE")
    print(val_table.round(3).to_string(index=False))

    best_name = val_table.iloc[0]["model"]
    best_val_mae = float(val_table.iloc[0]["MAE"])
    base_val_mae = float(val_table.iloc[0]["persistence_MAE"])
    if best_val_mae >= base_val_mae:
        print("\nNo candidate beats persistence on validation. Do not promote a learned model yet.")
        selected_name = "Persistence"
    else:
        selected_name = str(best_name)
        print(f"\nSelected on validation: {selected_name}")

    # Only after selection is frozen, report performance once on holdout test.
    print("\nFinal holdout test results:")
    test_rows = []
    p_mae, p_rmse = metrics(y_test, test_persistence)
    test_rows.append(("Persistence", p_mae, p_rmse))
    print(f"{'Persistence':24s} MAE={p_mae:.3f} | RMSE={p_rmse:.3f}")
    for name, model in fitted.items():
        pred = model.predict(X_test)
        mae, rmse = metrics(y_test, pred)
        test_rows.append((name, mae, rmse))
        print(f"{name:24s} MAE={mae:.3f} | RMSE={rmse:.3f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    val_table.to_csv(MODEL_DIR / "room_co2_model_comparison_validation.csv", index=False)
    pd.DataFrame(test_rows, columns=["model", "MAE", "RMSE"]).to_csv(
        MODEL_DIR / "room_co2_model_comparison_test.csv", index=False
    )

    # This is a separate experimental artifact. It never overwrites the original bundle.
    if selected_name != "Persistence":
        final_model = make_pipeline(SimpleImputer(strategy="median"), candidates[selected_name])
        X_train_all = X_all.copy()
        y_train_all = y_all.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final_model.fit(X_train_all, y_train_all)
        bundle = {
            "model": final_model,
            "target": TARGET,
            "horizon_minutes": HORIZON_MINUTES,
            "lags_minutes": LAGS,
            "feature_columns": list(X_all.columns),
            "selected_on_validation": selected_name,
            "note": "Experimental room CO2 model; original model bundle untouched.",
        }
        joblib.dump(bundle, MODEL_DIR / "room_co2_experimental_model.joblib")
        print("\nSaved experimental model: models/room_co2_experimental_model.joblib")
    else:
        print("\nNo experimental model saved because persistence was better on validation.")

    print("\nSaved reports:")
    print("  models/room_co2_model_comparison_validation.csv")
    print("  models/room_co2_model_comparison_test.csv")
    print("\nImportant: use validation to choose models; the holdout test is for final comparison.")

if __name__ == "__main__":
    main()
