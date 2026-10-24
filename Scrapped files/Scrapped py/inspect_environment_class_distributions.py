from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
INPUT = MODELS / "all_environment_classification_predictions.csv"
OUTPUT = MODELS / "all_environment_class_distribution.csv"

if not INPUT.exists():
    raise FileNotFoundError(
        f"Missing {INPUT}. Run evaluate_all_environment_classification.py first."
    )

df = pd.read_csv(INPUT)

required = {"target", "actual_category", "predicted_category"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing expected columns: {sorted(missing)}")

rows = []
for target, group in df.groupby("target", sort=True):
    actual_counts = group["actual_category"].value_counts(dropna=False)
    predicted_counts = group["predicted_category"].value_counts(dropna=False)
    categories = sorted(set(actual_counts.index.astype(str)) |
                        set(predicted_counts.index.astype(str)))

    total = len(group)
    for category in categories:
        actual_n = int(actual_counts.get(category, 0))
        predicted_n = int(predicted_counts.get(category, 0))
        rows.append({
            "target": target,
            "category": category,
            "actual_count": actual_n,
            "actual_percent": 100 * actual_n / total if total else 0,
            "predicted_count": predicted_n,
            "predicted_percent": 100 * predicted_n / total if total else 0,
            "total_samples": total,
        })

result = pd.DataFrame(rows)
result.to_csv(OUTPUT, index=False)

print("\n=== ACTUAL VS PREDICTED CATEGORY DISTRIBUTIONS ===")
for target, group in result.groupby("target", sort=True):
    print(f"\n{target}")
    print(group[[
        "category", "actual_count", "actual_percent",
        "predicted_count", "predicted_percent"
    ]].to_string(
        index=False,
        formatters={
            "actual_percent": "{:.2f}%".format,
            "predicted_percent": "{:.2f}%".format,
        }
    ))

print(f"\nSaved report: {OUTPUT}")
