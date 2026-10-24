from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data")
OUTPUT_DIR = Path("models")
FILES = [
    DATA_DIR / "training_data_2.csv",
    DATA_DIR / "ventilation_test.csv",
]

EXPECTED_COLUMNS = [
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

def load_csv(path):
    if not path.exists():
        print(f"\n[SKIP] File not found: {path}")
        return None

    df = pd.read_csv(path)
    if "time" not in df.columns:
        print(f"\n[SKIP] No 'time' column in {path}")
        return None

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df

def inspect_file(path):
    df = load_csv(path)
    if df is None:
        return None

    print("\n" + "=" * 100)
    print(f"FILE: {path}")
    print("=" * 100)
    print(f"Rows: {len(df):,}")
    print(f"Time range: {df['time'].min()} -> {df['time'].max()}")
    print(f"Duplicate timestamps: {df['time'].duplicated().sum():,}")

    diffs = df["time"].diff().dropna().dt.total_seconds()
    if len(diffs):
        print("\nTimestamp interval (seconds):")
        print(f"  Median: {diffs.median():.1f}")
        print(f"  Minimum: {diffs.min():.1f}")
        print(f"  Maximum: {diffs.max():.1f}")
        print(f"  Gaps > 90 seconds: {(diffs > 90).sum():,}")
        print(f"  Gaps > 5 minutes: {(diffs > 300).sum():,}")
        print(f"  Non-positive intervals: {(diffs <= 0).sum():,}")

    missing_expected = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_expected:
        print("\nExpected columns missing from this file:")
        for c in missing_expected:
            print(f"  - {c}")

    numeric_cols = [c for c in EXPECTED_COLUMNS if c in df.columns]
    if numeric_cols:
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        print("\nMissing/non-numeric values:")
        for c in numeric_cols:
            count = int(df[c].isna().sum())
            if count:
                print(f"  {c}: {count:,}")

        print("\nNumeric summary:")
        print(df[numeric_cols].describe(percentiles=[0.01, 0.50, 0.99]).T[
            ["min", "1%", "50%", "99%", "max", "mean", "std"]
        ].round(3).to_string())

        print("\nPotentially suspicious values (flags only; no rows are removed):")
        checks = {
            "Room CO2 (ppm)": lambda s: (s < 250) | (s > 5000),
            "Temperature (°C)": lambda s: (s < 5) | (s > 45),
            "Relative Humidity (%)": lambda s: (s < 0) | (s > 100),
            "Pressure (mbar)": lambda s: (s < 800) | (s > 1200),
            "Supply airflow (l/s)": lambda s: (s < 0),
            "Extract airflow (l/s)": lambda s: (s < 0),
            "Supply air CO2 (ppm)": lambda s: (s < 250) | (s > 5000),
            "Exhaust air CO2 (ppm)": lambda s: (s < 250) | (s > 5000),
            "Occupant count": lambda s: (s < 0) | (s > 100),
        }
        for c, rule in checks.items():
            if c in df.columns:
                mask = df[c].notna() & rule(df[c])
                print(f"  {c}: {int(mask.sum()):,} flagged")

        # List a small number of extreme examples for human review.
        for c in ["Relative Humidity (%)", "Supply airflow (l/s)", "Extract airflow (l/s)"]:
            if c in df.columns:
                bad = df[c].notna() & (
                    (df[c] < 0) | (df[c] > 100 if c == "Relative Humidity (%)" else False)
                )
                if bad.any():
                    print(f"\nExamples to review for {c}:")
                    print(df.loc[bad, ["time", c]].head(8).to_string(index=False))

    return df

def main():
    print("DATA QUALITY CHECK")
    print("Read-only analysis: original CSV files are not changed.")
    print("Thresholds below are screening flags, not automatic proof of bad data.")

    loaded = []
    for path in FILES:
        df = inspect_file(path)
        if df is not None:
            loaded.append((path, df))

    if loaded:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        rows = []
        for path, df in loaded:
            for c in EXPECTED_COLUMNS:
                if c not in df.columns:
                    continue
                s = pd.to_numeric(df[c], errors="coerce")
                rows.append({
                    "file": str(path),
                    "column": c,
                    "count": int(s.notna().sum()),
                    "missing": int(s.isna().sum()),
                    "min": s.min(),
                    "p01": s.quantile(0.01),
                    "median": s.median(),
                    "p99": s.quantile(0.99),
                    "max": s.max(),
                })
        report = pd.DataFrame(rows)
        output = OUTPUT_DIR / "data_quality_summary.csv"
        report.to_csv(output, index=False)
        print(f"\nSaved compact summary: {output}")
    else:
        print("\nNo input files were analyzed. Check that both CSV files are inside the data folder.")

if __name__ == "__main__":
    main()
