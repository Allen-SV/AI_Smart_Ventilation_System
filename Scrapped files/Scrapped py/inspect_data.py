from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
FILES = [
    "training_data_1.csv",
    "training_data_2.csv",
    "ventilation_test.csv",
]

TIME_COL = "time"
TARGET_CANDIDATES = [
    "Room CO2 (ppm)",
    "Temperature (°C)",
    "Relative Humidity (%)",
    "Pressure (mbar)",
    "Supply airflow (l/s)",
    "Extract airflow (l/s)",
    "Supply air CO2 (ppm)",
    "Exhaust air CO2 (ppm)",
    "Occupant count",
]

def inspect_file(path: Path) -> None:
    print("\n" + "=" * 88)
    print(f"FILE: {path}")
    print("=" * 88)

    df = pd.read_csv(path)
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")

    if TIME_COL not in df.columns:
        print(f"\nERROR: required timestamp column {TIME_COL!r} is missing.")
        return

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    invalid_times = int(df[TIME_COL].isna().sum())
    print(f"\nInvalid timestamps: {invalid_times}")

    valid_times = df[TIME_COL].dropna()
    if not valid_times.empty:
        print(f"Time range: {valid_times.min()} -> {valid_times.max()}")
        deltas = valid_times.sort_values().diff().dropna().dt.total_seconds() / 60
        print("\nMost common time gaps (minutes):")
        print(deltas.value_counts().head(8).to_string())
        print(f"Duplicate timestamps: {int(df[TIME_COL].duplicated().sum())}")

    print("\nMissing values:")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(missing.to_string() if not missing.empty else "  None")

    if "Occupant count" in df.columns:
        print("\nGround-truth occupant count distribution:")
        print(df["Occupant count"].value_counts(dropna=False).sort_index().to_string())

    print("\nNumeric summary:")
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols):
        print(df[numeric_cols].describe().round(3).to_string())

def main():
    if not DATA_DIR.exists():
        print(f"Missing folder: {DATA_DIR.resolve()}")
        print("Create a folder named 'data' and place the three CSV files inside it.")
        return

    for filename in FILES:
        path = DATA_DIR / filename
        if path.exists():
            inspect_file(path)
        else:
            print(f"\nWARNING: Could not find {path}")

if __name__ == "__main__":
    main()
