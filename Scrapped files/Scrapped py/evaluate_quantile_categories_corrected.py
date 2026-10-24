"""
Re-evaluate one-minute forecasts using training-set tertile categories.

This version reports:
1. Metrics across all categories (including empty categories, explicitly labeled).
2. Metrics across categories represented in the holdout actual labels only.
3. Per-category metrics and supports, including categories with zero support.
4. Category thresholds derived from training data only.

Important: excluding an absent actual class from macro averaging does NOT evaluate
performance on that class. It is reported separately with zero support.
Categories are relative statistical bands, not validated safety thresholds.

Run from project root:
    python evaluate_quantile_categories.py
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
    vals = pd.to_numeric(training_values, errors="coerce").dropna().to_numpy()
    if len(vals) == 0:
        raise ValueError("No numeric training values available.")
    q1, q2 = np.quantile(vals, [1 / 3, 2 / 3])
    edges = np.unique(np.array([-np.inf, q1, q2, np.inf], dtype=float))
    if len(edges) == 3:
        return edges, ["Low", "High"]
    return edges, ["Low", "Medium", "High"]


def categorize(series, edges, labels):
    return pd.cut(
        pd.to_numeric(series, errors="coerce"),
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
    ).astype("object")


def metric_values(y_true, y_pred, labels):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return p, r, f1


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_FILE}")
    if not PRED_FILE.exists():
        raise FileNotFoundError(
            f"Holdout predictions not found: {PRED_FILE}. "
            "Run train_all_environment_model_1min.py first."
        )

    train = pd.read_csv(TRAIN_FILE)
    holdout = pd.read_csv(PRED_FILE)

    summary_rows, class_rows, cm_rows, pred_rows, threshold_rows = [], [], [], [], []

    for target, (actual_col, pred_col) in TARGETS.items():
        train_col = TRAIN_COLUMNS[target]
        if train_col not in train.columns:
            print(f"SKIPPING {target}: training column '{train_col}' missing.")
            continue
        if actual_col not in holdout.columns or pred_col not in holdout.columns:
            print(f"SKIPPING {target}: expected holdout columns missing.")
            continue

        edges, labels = make_edges(train[train_col])
        finite_edges = [float(x) for x in edges if np.isfinite(x)]
        threshold_rows.append({
            "target": target,
            "training_column": train_col,
            "category_labels": " | ".join(labels),
            "finite_cut_points": json.dumps(finite_edges),
            "training_rows_used": int(pd.to_numeric(train[train_col], errors="coerce").notna().sum()),
            "note": "Cut points from training data only; relative statistical bands."
        })

        actual = pd.to_numeric(holdout[actual_col], errors="coerce")
        predicted = pd.to_numeric(holdout[pred_col], errors="coerce")
        valid = actual.notna() & predicted.notna()
        actual = actual[valid]
        predicted = predicted[valid]

        y_true_all = categorize(actual, edges, labels)
        y_pred_all = categorize(predicted, edges, labels)
        valid_labels = y_true_all.notna() & y_pred_all.notna()
        y_true = y_true_all[valid_labels].astype(str)
        y_pred = y_pred_all[valid_labels].astype(str)
        n = len(y_true)

        # Macro metrics including all defined classes, including absent classes.
        p_all, r_all, f1_all = metric_values(y_true, y_pred, labels)
        # Macro metrics only across classes that have at least one actual holdout sample.
        present_labels = [label for label in labels if (y_true == label).any()]
        p_present, r_present, f1_present = metric_values(y_true, y_pred, present_labels)

        summary_rows.append({
            "target": target,
            "n": n,
            "defined_categories": len(labels),
            "actual_categories_present": len(present_labels),
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_precision_all_defined": p_all,
            "macro_recall_all_defined": r_all,
            "macro_f1_all_defined": f1_all,
            "macro_precision_present_actual_classes": p_present,
            "macro_recall_present_actual_classes": r_present,
            "macro_f1_present_actual_classes": f1_present,
            "weighted_precision": precision_recall_fscore_support(
                y_true, y_pred, labels=labels, average="weighted", zero_division=0
            )[0],
            "weighted_recall": precision_recall_fscore_support(
                y_true, y_pred, labels=labels, average="weighted", zero_division=0
            )[1],
            "weighted_f1": precision_recall_fscore_support(
                y_true, y_pred, labels=labels, average="weighted", zero_division=0
            )[2],
        })

        report = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        )
        actual_counts = pd.Series(y_true).value_counts()
        predicted_counts = pd.Series(y_pred).value_counts()
        for label in labels:
            stats = report[label]
            support = int(actual_counts.get(label, 0))
            class_rows.append({
                "target": target,
                "category": label,
                "precision": stats["precision"],
                "recall": stats["recall"],
                "f1": stats["f1-score"],
                "support": support,
                "actual_category_present": support > 0,
                "predicted_count": int(predicted_counts.get(label, 0)),
                "note": "" if support > 0 else "No actual holdout samples; class performance cannot be assessed."
            })

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        for i, label in enumerate(labels):
            row = {"target": target, "actual_category": label}
            for j, predicted_label in enumerate(labels):
                row[f"predicted_{predicted_label}"] = int(cm[i, j])
            cm_rows.append(row)

        for idx in y_true.index:
            pred_rows.append({
                "time": holdout.loc[idx, "time"] if "time" in holdout.columns else None,
                "target": target,
                "actual_value": float(actual.loc[idx]),
                "predicted_value": float(predicted.loc[idx]),
                "actual_category": y_true.loc[idx],
                "predicted_category": y_pred.loc[idx],
                "category_correct": bool(y_true.loc[idx] == y_pred.loc[idx]),
            })

    if not summary_rows:
        raise ValueError("No targets evaluated. Check input CSV column names.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "all_environment_quantile_classification_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(OUT_DIR / "all_environment_quantile_classification_per_class.csv", index=False)
    pd.DataFrame(cm_rows).to_csv(OUT_DIR / "all_environment_quantile_confusion_matrices.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(OUT_DIR / "all_environment_quantile_classification_predictions.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(OUT_DIR / "all_environment_quantile_category_thresholds.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    display_cols = [
        "target", "n", "accuracy", "macro_f1_all_defined",
        "macro_f1_present_actual_classes", "weighted_f1"
    ]
    print("\n=== CLASSIFICATION RESULTS: ALL DEFINED VS PRESENT ACTUAL CLASSES ===")
    print(summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved updated reports in models/.")
    print("For absent classes, present-actual macro metrics exclude them; this does not")
    print("measure performance on those classes. Check per-class support before reporting.")
    print("Categories are statistical bands, not validated ventilation safety thresholds.")


if __name__ == "__main__":
    main()
