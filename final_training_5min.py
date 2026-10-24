"""Train and evaluate a five-minute-ahead indoor air-quality forecaster.

Expected files beside this script:
  data/training_data_2.csv
  data/ventilation_test.csv

Run: python final_training_5min.py

The script predicts Room CO2, temperature, and relative humidity exactly five minutes ahead.
It uses other environmental readings and measured occupant count as inputs, compares Random
Forest with persistence, and evaluates a capped five-minute CO2 trend baseline. It also evaluates
selected forecasts as Low/Medium/High categories using training-data tertiles only. The independent
test file is never used for model selection, training, or threshold selection.
"""
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_recall_fscore_support, r2_score,
)
from sklearn.multioutput import MultiOutputRegressor

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
TRAIN_FILE = DATA_DIR / "training_data_2.csv"
TEST_FILE = DATA_DIR / "ventilation_test.csv"

TIME_COL = "time"
TARGETS = ["Room CO2 (ppm)", "Temperature (°C)", "Relative Humidity (%)"]

# Other measured environmental variables remain available as historical input features.
INPUT_ENVIRONMENT_FEATURES = [
    "Room CO2 (ppm)", "Temperature (°C)", "Relative Humidity (%)",
    "Pressure (mbar)", "Supply airflow (l/s)", "Extract airflow (l/s)",
    "Supply air CO2 (ppm)", "Exhaust air CO2 (ppm)",
]
# Use measured occupant count as an input; deliberately exclude ML-estimated occupancy.
# Retain historical environmental readings as features, but predict only the three targets.
BASE_FEATURES = INPUT_ENVIRONMENT_FEATURES + ["Occupant count"]
LAGS_MINUTES = [0, 1, 2, 3, 5]
HORIZON_MINUTES = 5
# Maximum absolute CO2 change allowed by the trend baseline over the full horizon.
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
    """Join each row's lagged features to measurements exactly five minutes later."""
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
    # Current target values serve as persistence predictions. The CO2 trend baseline
    # extrapolates the latest one-minute change across the five-minute horizon.
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
            change_per_minute = current - previous
            projected_change = change_per_minute * HORIZON_MINUTES
            trend = current + np.clip(projected_change, -CO2_TREND_CAP, CO2_TREND_CAP)
            methods["trend_capped"] = trend
        chosen = (selected_methods or {}).get(target, "random_forest")
        for method, pred in methods.items():
            rows.append(metric_row(target, method, actual, pred, method == chosen))
        pred_frame[f"actual__{target}"] = actual
        for method, pred in methods.items():
            pred_frame[f"{method}__{target}"] = pred
        pred_frame[f"selected__{target}"] = methods.get(chosen, rf_pred[:, j])
    return pd.DataFrame(rows), pred_frame



def calculate_training_quantile_edges(train_df):
    """Create Low/Medium/High boundaries from development/training values only."""
    edges_by_target = {}
    for target in TARGETS:
        values = pd.to_numeric(train_df[target], errors="coerce").dropna().to_numpy()
        if len(values) == 0:
            raise ValueError(f"No valid training values available for target: {target}")
        q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
        # Quantile classification needs two distinct cut points.
        if not q1 < q2:
            raise ValueError(
                f"Cannot create three distinct quantile categories for {target}: "
                f"training tertile cut points are {q1} and {q2}."
            )
        edges_by_target[target] = [float(q1), float(q2)]
    return edges_by_target


def category_ids(values, edges):
    """Map numeric values to 0=Low, 1=Medium, 2=High using fixed training cut points."""
    return np.digitize(np.asarray(values, dtype=float), bins=edges, right=False)


def evaluate_selected_classification(train_df, holdout_predictions, output_dir):
    """
    Evaluate selected continuous forecasts as Low/Medium/High categories.
    Thresholds come only from the development training data, never from holdout data.
    """
    labels = ["Low", "Medium", "High"]
    edges_by_target = calculate_training_quantile_edges(train_df)

    summary_rows = []
    per_class_rows = []
    confusion_rows = []
    prediction_output = pd.DataFrame({"time": holdout_predictions["time"]})
    threshold_rows = []

    for target in TARGETS:
        edges = edges_by_target[target]
        actual_col = f"actual__{target}"
        predicted_col = f"selected__{target}"

        actual_values = holdout_predictions[actual_col].to_numpy(dtype=float)
        predicted_values = holdout_predictions[predicted_col].to_numpy(dtype=float)
        y_true = category_ids(actual_values, edges)
        y_pred = category_ids(predicted_values, edges)

        accuracy = accuracy_score(y_true, y_pred)
        macro_all = f1_score(y_true, y_pred, labels=[0, 1, 2],
                             average="macro", zero_division=0)
        present_labels = sorted(np.unique(y_true).tolist())
        macro_present = f1_score(y_true, y_pred, labels=present_labels,
                                 average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2],
                               average="weighted", zero_division=0)
        weighted_precision, weighted_recall, _, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2], average="weighted", zero_division=0
        )

        summary_rows.append({
            "target": target,
            "n": int(len(y_true)),
            "accuracy": float(accuracy),
            "macro_f1_all_defined": float(macro_all),
            "macro_f1_present_actual_classes": float(macro_present),
            "weighted_precision": float(weighted_precision),
            "weighted_recall": float(weighted_recall),
            "weighted_f1": float(weighted_f1),
            "actual_classes_present": ", ".join(labels[i] for i in present_labels),
            "actual_high_support": int(np.sum(y_true == 2)),
            "note": (
                "High category has no actual holdout samples; its performance is not evaluated."
                if 2 not in present_labels else ""
            ),
        })

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
        )
        for class_id, class_name in enumerate(labels):
            per_class_rows.append({
                "target": target,
                "class": class_name,
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(f1[class_id]),
                "support": int(support[class_id]),
                "present_in_actual_holdout": bool(support[class_id] > 0),
            })

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        for actual_id, actual_name in enumerate(labels):
            for predicted_id, predicted_name in enumerate(labels):
                confusion_rows.append({
                    "target": target,
                    "actual_class": actual_name,
                    "predicted_class": predicted_name,
                    "count": int(cm[actual_id, predicted_id]),
                })

        prediction_output[f"actual_class__{target}"] = [labels[i] for i in y_true]
        prediction_output[f"predicted_class__{target}"] = [labels[i] for i in y_pred]

        threshold_rows.append({
            "target": target,
            "low_medium_cutpoint": edges[0],
            "medium_high_cutpoint": edges[1],
            "threshold_source": "training_data_2.csv only",
            "category_definition": "Low <= first tertile; Medium > first and <= second tertile; High > second tertile",
        })

    summary = pd.DataFrame(summary_rows)
    per_class = pd.DataFrame(per_class_rows)
    confusion = pd.DataFrame(confusion_rows)
    thresholds = pd.DataFrame(threshold_rows)

    summary.to_csv(output_dir / "all_environment_5min_quantile_classification_metrics.csv", index=False)
    per_class.to_csv(output_dir / "all_environment_5min_quantile_classification_per_class.csv", index=False)
    confusion.to_csv(output_dir / "all_environment_5min_quantile_confusion_matrices.csv", index=False)
    prediction_output.to_csv(output_dir / "all_environment_5min_quantile_classification_predictions.csv", index=False)
    thresholds.to_csv(output_dir / "all_environment_5min_quantile_category_thresholds.csv", index=False)

    print("\n=== CLASSIFICATION RESULTS: SELECTED FORECASTS ===")
    print(summary[[
        "target", "n", "accuracy", "macro_f1_all_defined",
        "macro_f1_present_actual_classes", "weighted_f1", "actual_classes_present"
    ]].round(4).to_string(index=False))
    print("\nClassification categories are statistical training-data tertiles, not safety thresholds.")
    print("If a category has zero actual holdout samples, performance for that category is untested.")

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
        raise ValueError("Too few exact five-minute timestamp pairs after removing gaps/missing values.")

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
        "note": "Five-minute-ahead regression forecaster for room CO2, temperature, and relative humidity. Occupant count and other environmental readings are inputs; statistical category reports are not safety thresholds.",
    }
    joblib.dump(bundle, MODEL_DIR / "all_environment_model_5min.joblib")
    val_metrics.to_csv(MODEL_DIR / "all_environment_5min_validation_metrics.csv", index=False)
    holdout_metrics.to_csv(MODEL_DIR / "all_environment_5min_holdout_metrics.csv", index=False)
    val_predictions.to_csv(MODEL_DIR / "all_environment_5min_validation_predictions.csv", index=False)
    holdout_predictions.to_csv(MODEL_DIR / "all_environment_5min_holdout_predictions.csv", index=False)
    # Add category-level evaluation without changing the continuous forecasting model.
    evaluate_selected_classification(train_df, holdout_predictions, MODEL_DIR)
    with open(MODEL_DIR / "all_environment_5min_config.json", "w", encoding="utf-8") as f:
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
    print("\n=== INDEPENDENT FIVE-MINUTE HOLDOUT RESULTS ===")
    print(holdout_metrics.sort_values(["target", "method"])[["target", "method", "n", "MAE", "RMSE", "R2", "selected_by_validation"]].round(4).to_string(index=False))
    print("\nSaved model and reports under:", MODEL_DIR)
    print("Targets: room CO2, temperature, and relative humidity. Occupant count is an input feature.")
    print("Metrics are regression metrics; category reports use training-data tertiles, not safety thresholds.")

if __name__ == "__main__":
    main()
