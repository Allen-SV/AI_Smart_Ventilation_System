"""
Re-evaluate all eight one-minute forecasts using training-set quantile categories.

- Quantile cut points are calculated ONLY from data/training_data_2.csv.
- The same cut points are applied to actual and predicted holdout values.
- Holdout labels are not used to choose category thresholds.
- Regression metrics remain the primary forecasting evaluation.

Run from the project root:
    python evaluate_quantile_categories.py

Requires pandas and scikit-learn.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent
TRAIN_FILE = ROOT / "data" / "training_data_2.csv"
PRED_FILE = ROOT / "models" / "all_environment_1min_holdout_predictions.csv"
OUT_DIR = ROOT / "models"

TARGETS = {
    "Room CO2 (ppm)": ("actual__Room CO2 (ppm)", "selected__Room CO2 (ppm)"),
    "Temperature (°C)": ("actual__Temperature (°C)", "selected__Temperature (°C)"),
    "Relative Humidity (%)": ("actual__Relative Humidity (%)", "selected__Relative Humidity (%)"),
    "Pressure (mbar)": ("actual__Pressure (mbar)", "selected__Pressure (mbar)"),
    "Supply airflow (l/s)": ("actual__Supply airflow (l/s)", "selected__Supply airflow (l/s)"),
    "Extract airflow (l/s)": ("actual__Extract airflow (l/s)", "selected__Extract airflow (l/s)"),
    "Supply air CO2 (ppm)": ("actual__Supply air CO2 (ppm)", "selected__Supply air CO2 (ppm)"),
    "Exhaust air CO2 (ppm)": ("actual__Exhaust air CO2 (ppm)", "selected__Exhaust air CO2 (ppm)"),
}

# Corresponding raw columns in training_data_2.csv
TRAIN_COLUMNS = {
    "Room CO2 (ppm)": "Room CO2 (ppm)",
    "Temperature (°C)": "Temperature (°C)",
    "Relative Humidity (%)": "Relative Humidity (%)",
    "Pressure (mbar)": "Pressure (mbar)",
    "Supply airflow (l/s)": "Supply airflow (l/s)",
    "Extract airflow (l/s)": "Extract airflow (l/s)",
    "Supply air CO2 (ppm)": "Supply air CO2 (ppm)",
    "Exhaust air CO2 (ppm)": "Exhaust air CO2 (ppm)",
}


def make_edges(training_values):
    """Get unique tertile cut points from training data only."""
    vals = pd.to_numeric(training_values, errors="coerce").dropna().to_numpy()
    if len(vals) == 0:
        raise ValueError("No numeric training values available.")
    q1, q2 = np.quantile(vals, [1 / 3, 2 / 3])
    # If quantiles coincide, collapse to two classes rather than inventing
    # a cut point. Constant targets cannot meaningfully support classification.
    edges = np.unique(np.array([-np.inf, q1, q2, np.inf], dtype=float))
    if len(edges) < 3:
        return edges, ["Low", "High"] if len(edges) == 3 else ["All"]
    if len(edges) == 3:
        labels = ["Low", "High"]
    else:
        labels = ["Low", "Medium", "High"]
    return edges, labels


def categorize(series, edges, labels):
    if labels == ["All"]:
        return pd.Series(["All"] * len(series), index=series.index, dtype="object")
    result = pd.cut(
        pd.to_numeric(series, errors="coerce"),
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    return result.astype("object")


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_FILE}")
    if not PRED_FILE.exists():
        raise FileNotFoundError(
            f"Holdout prediction file not found: {PRED_FILE}. "
            "Run train_all_environment_model_1min.py first."
        )

    train = pd.read_csv(TRAIN_FILE)
    holdout = pd.read_csv(PRED_FILE)

    summaries, class_rows, cm_rows, pred_rows, threshold_rows = [], [], [], [], []

    for target, (actual_col, pred_col) in TARGETS.items():
        raw_col = TRAIN_COLUMNS[target]
        if raw_col not in train.columns:
            print(f"SKIPPING {target}: training column '{raw_col}' missing.")
            continue
        if actual_col not in holdout.columns or pred_col not in holdout.columns:
            print(f"SKIPPING {target}: holdout prediction columns missing.")
            continue

        edges, labels = make_edges(train[raw_col])
        # Store the numerical boundaries so the categories are reproducible.
        finite_edges = [float(x) for x in edges if np.isfinite(x)]
        threshold_rows.append({
            "target": target,
            "training_column": raw_col,
            "category_labels": " | ".join(labels),
            "finite_cut_points": json.dumps(finite_edges),
            "training_rows_used": int(pd.to_numeric(train[raw_col], errors="coerce").notna().sum()),
            "note": "Cut points derived from training data only; categories are relative statistical bands."
        })

        actual = pd.to_numeric(holdout[actual_col], errors="coerce")
        predicted = pd.to_numeric(holdout[pred_col], errors="coerce")
        valid = actual.notna() & predicted.notna()
        actual = actual[valid]
        predicted = predicted[valid]

        y_true = categorize(actual, edges, labels)
        y_pred = categorize(predicted, edges, labels)
        keep = y_true.notna() & y_pred.notna()
        y_true, y_pred = y_true[keep].astype(str), y_pred[keep].astype(str)
        n = len(y_true)

        accuracy = accuracy_score(y_true, y_pred)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        )
        summaries.append({
            "target": target, "n": n, "categories": len(labels),
            "accuracy": accuracy, "macro_precision": p_macro,
            "macro_recall": r_macro, "macro_f1": f1_macro,
            "weighted_precision": p_weighted, "weighted_recall": r_weighted,
            "weighted_f1": f1_weighted,
        })

        report = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        )
        for label in labels:
            stats = report[label]
            class_rows.append({
                "target": target, "category": label,
                "precision": stats["precision"], "recall": stats["recall"],
                "f1": stats["f1-score"], "support": stats["support"],
            })

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        for i, label in enumerate(labels):
            row = {"target": target, "actual_category": label}
            for j, pred_label in enumerate(labels):
                row[f"predicted_{pred_label}"] = int(cm[i, j])
            cm_rows.append(row)

        indices = y_true.index
        for idx in indices:
            pred_rows.append({
                "time": holdout.loc[idx, "time"] if "time" in holdout.columns else None,
                "target": target,
                "actual_value": float(actual.loc[idx]),
                "predicted_value": float(predicted.loc[idx]),
                "actual_category": y_true.loc[idx],
                "predicted_category": y_pred.loc[idx],
                "category_correct": bool(y_true.loc[idx] == y_pred.loc[idx]),
            })

    if not summaries:
        raise ValueError("No target columns could be evaluated. Check CSV schemas.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_DIR / "all_environment_quantile_classification_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(OUT_DIR / "all_environment_quantile_classification_per_class.csv", index=False)
    pd.DataFrame(cm_rows).to_csv(OUT_DIR / "all_environment_quantile_confusion_matrices.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(OUT_DIR / "all_environment_quantile_classification_predictions.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(OUT_DIR / "all_environment_quantile_category_thresholds.csv", index=False)

    print("\n=== TRAINING-QUANTILE CATEGORY CLASSIFICATION RESULTS ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved reports in models/:")
    for filename in [
        "all_environment_quantile_classification_metrics.csv",
        "all_environment_quantile_classification_per_class.csv",
        "all_environment_quantile_confusion_matrices.csv",
        "all_environment_quantile_classification_predictions.csv",
        "all_environment_quantile_category_thresholds.csv",
    ]:
        print(f"  {filename}")
    print("\nNote: categories are relative to the training-data distribution,")
    print("not validated health, safety, or ventilation-control thresholds.")


if __name__ == "__main__":
    main()
