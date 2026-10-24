"""
Evaluate one-minute forecasts for all eight environmental variables.

This script converts each continuous target into project-defined categories,
then calculates accuracy, precision, recall and F1-score. It also saves the
existing regression metrics for context.

IMPORTANT:
- Category bands below are INITIAL PROJECT EVALUATION BANDS, not health,
  regulatory, or engineering standards. Confirm them with your project guide
  before presenting them as operational ventilation thresholds.
- Accuracy/F1 measure category agreement, not exact numerical accuracy.
- Run train_all_environment_model_1min.py first.
"""

from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
INPUT = MODELS / "all_environment_1min_holdout_predictions.csv"

# Initial evaluation bands. Values are grouped in increasing order.
# These are project-defined for evaluation and must be validated before
# being used to control real ventilation equipment.
CATEGORIES = {
    "Room CO2 (ppm)": {
        "column_actual": "actual__Room CO2 (ppm)",
        "column_pred": "selected__Room CO2 (ppm)",
        "bins": [-float("inf"), 800, 1000, float("inf")],
        "labels": ["Low", "Moderate", "High"],
    },
    "Temperature (°C)": {
        "column_actual": "actual__Temperature (°C)",
        "column_pred": "selected__Temperature (°C)",
        "bins": [-float("inf"), 20, 24, 28, float("inf")],
        "labels": ["Cool", "Comfortable-low", "Comfortable-high", "Warm"],
    },
    "Relative Humidity (%)": {
        "column_actual": "actual__Relative Humidity (%)",
        "column_pred": "selected__Relative Humidity (%)",
        "bins": [-float("inf"), 30, 40, 60, 70, float("inf")],
        "labels": ["Very-dry", "Dry", "Typical", "Humid", "Very-humid"],
    },
    "Pressure (mbar)": {
        "column_actual": "actual__Pressure (mbar)",
        "column_pred": "selected__Pressure (mbar)",
        "bins": [-float("inf"), 990, 1000, 1010, float("inf")],
        "labels": ["Low", "Moderately-low", "Moderately-high", "High"],
    },
    "Supply airflow (l/s)": {
        "column_actual": "actual__Supply airflow (l/s)",
        "column_pred": "selected__Supply airflow (l/s)",
        "bins": [-float("inf"), 20, 40, 60, float("inf")],
        "labels": ["Low", "Medium", "High", "Very-high"],
    },
    "Extract airflow (l/s)": {
        "column_actual": "actual__Extract airflow (l/s)",
        "column_pred": "selected__Extract airflow (l/s)",
        "bins": [-float("inf"), 20, 40, 60, float("inf")],
        "labels": ["Low", "Medium", "High", "Very-high"],
    },
    "Supply air CO2 (ppm)": {
        "column_actual": "actual__Supply air CO2 (ppm)",
        "column_pred": "selected__Supply air CO2 (ppm)",
        "bins": [-float("inf"), 600, 800, 1000, float("inf")],
        "labels": ["Low", "Moderate", "Elevated", "High"],
    },
    "Exhaust air CO2 (ppm)": {
        "column_actual": "actual__Exhaust air CO2 (ppm)",
        "column_pred": "selected__Exhaust air CO2 (ppm)",
        "bins": [-float("inf"), 600, 800, 1000, float("inf")],
        "labels": ["Low", "Moderate", "Elevated", "High"],
    },
}


def classify(values, bins, labels):
    return pd.cut(values, bins=bins, labels=labels, right=False).astype(str)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT}. Run train_all_environment_model_1min.py first."
        )

    df = pd.read_csv(INPUT)
    summary_rows = []
    per_class_rows = []
    prediction_frames = []
    confusion_matrices = []

    for target, cfg in CATEGORIES.items():
        actual_col = cfg["column_actual"]
        pred_col = cfg["column_pred"]
        if actual_col not in df.columns or pred_col not in df.columns:
            print(f"SKIPPING {target}: required columns missing.")
            continue

        actual_values = pd.to_numeric(df[actual_col], errors="coerce")
        pred_values = pd.to_numeric(df[pred_col], errors="coerce")
        valid = actual_values.notna() & pred_values.notna()
        actual_values = actual_values[valid]
        pred_values = pred_values[valid]

        actual_labels = classify(actual_values, cfg["bins"], cfg["labels"])
        predicted_labels = classify(pred_values, cfg["bins"], cfg["labels"])
        labels = cfg["labels"]

        accuracy = accuracy_score(actual_labels, predicted_labels)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            actual_labels, predicted_labels, labels=labels,
            average="macro", zero_division=0
        )
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
            actual_labels, predicted_labels, labels=labels,
            average="weighted", zero_division=0
        )

        summary_rows.append({
            "target": target,
            "n": len(actual_labels),
            "accuracy": accuracy,
            "macro_precision": p_macro,
            "macro_recall": r_macro,
            "macro_f1": f1_macro,
            "weighted_precision": p_weighted,
            "weighted_recall": r_weighted,
            "weighted_f1": f1_weighted,
        })

        report = classification_report(
            actual_labels, predicted_labels, labels=labels,
            output_dict=True, zero_division=0
        )
        for label in labels:
            per_class_rows.append({
                "target": target,
                "class": label,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"],
            })

        cm = confusion_matrix(actual_labels, predicted_labels, labels=labels)
        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual: {label}" for label in labels],
            columns=[f"Predicted: {label}" for label in labels],
        )
        cm_df.insert(0, "target", target)
        cm_df.insert(1, "actual_class", labels)
        confusion_matrices.append(cm_df.reset_index(drop=True))

        frame = pd.DataFrame({
            "target": target,
            "actual_value": actual_values.to_numpy(),
            "predicted_value": pred_values.to_numpy(),
            "actual_category": actual_labels.to_numpy(),
            "predicted_category": predicted_labels.to_numpy(),
        })
        if "time" in df.columns:
            frame.insert(0, "time", df.loc[valid, "time"].to_numpy())
        frame["category_correct"] = frame["actual_category"] == frame["predicted_category"]
        prediction_frames.append(frame)

    if not summary_rows:
        raise ValueError(
            "No target columns matched the expected prediction CSV schema. "
            "Inspect the CSV column names and update CATEGORIES."
        )

    summary = pd.DataFrame(summary_rows).sort_values("target")
    per_class = pd.DataFrame(per_class_rows)
    confusion = pd.concat(confusion_matrices, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    summary.to_csv(MODELS / "all_environment_classification_metrics.csv", index=False)
    per_class.to_csv(MODELS / "all_environment_classification_per_class.csv", index=False)
    confusion.to_csv(MODELS / "all_environment_classification_confusion_matrices.csv", index=False)
    predictions.to_csv(MODELS / "all_environment_classification_predictions.csv", index=False)

    print("\n=== ONE-MINUTE CLASSIFICATION RESULTS: ALL ENVIRONMENTAL TARGETS ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved:")
    print(MODELS / "all_environment_classification_metrics.csv")
    print(MODELS / "all_environment_classification_per_class.csv")
    print(MODELS / "all_environment_classification_confusion_matrices.csv")
    print(MODELS / "all_environment_classification_predictions.csv")
    print("\nReminder: confirm the category bands with your project guide before")
    print("describing them as real-world operating or safety thresholds.")


if __name__ == "__main__":
    main()
