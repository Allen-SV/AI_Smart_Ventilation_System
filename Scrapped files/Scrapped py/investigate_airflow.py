from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data")
OUT_DIR = Path("models")
FILES = [DATA_DIR / "training_data_2.csv", DATA_DIR / "ventilation_test.csv"]

AIRFLOW_COLS = ["Supply airflow (l/s)", "Extract airflow (l/s)"]
TARGET_COLS = [
    "Room CO2 (ppm)", "Temperature (°C)", "Relative Humidity (%)",
    "Pressure (mbar)", "Supply airflow (l/s)", "Extract airflow (l/s)",
    "Supply air CO2 (ppm)", "Exhaust air CO2 (ppm)",
]

def inspect(path):
    if not path.exists():
        print(f"[SKIP] Missing file: {path}")
        return None

    df = pd.read_csv(path)
    if "time" not in df:
        print(f"[SKIP] No time column: {path}")
        return None

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    for c in TARGET_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    print("\n" + "=" * 95)
    print(f"{path} — focused airflow and timestamp inspection")
    print("=" * 95)

    diffs = df["time"].diff().dt.total_seconds()
    gaps = df.loc[diffs > 90, ["time"]].copy()
    gaps["gap_seconds"] = diffs[diffs > 90].values
    if len(gaps):
        print("\nTimestamp gaps > 90 seconds (time is the first row after each gap):")
        print(gaps.to_string(index=False))
    else:
        print("\nNo timestamp gaps > 90 seconds.")

    for c in AIRFLOW_COLS:
        if c not in df:
            continue
        s = df[c]
        neg = df.loc[s < 0, ["time", c]].copy()
        print(f"\n{c}:")
        print(f"  Negative readings: {len(neg)} / {s.notna().sum()}")
        print(f"  Minimum: {s.min():.6f}; median: {s.median():.6f}; maximum: {s.max():.6f}")
        if len(neg):
            print("  Negative-reading ranges (separated by gaps > 90 seconds):")
            mask = (df[c] < 0).fillna(False)
            # Start a new group when negativity starts or timestamps are discontinuous.
            new_group = mask & (~mask.shift(1, fill_value=False) | (diffs > 90).fillna(False))
            groups = new_group.cumsum()
            for _, group in df.loc[mask].assign(_group=groups[mask]).groupby("_group"):
                start = group["time"].iloc[0]
                end = group["time"].iloc[-1]
                vals = group[c]
                print(f"    {start} -> {end} | rows={len(group)} | min={vals.min():.6f}")

        # Display nearby readings around the first few negative values.
        if len(neg):
            print("  First negative readings with previous/next context:")
            for idx in neg.index[:3]:
                lo, hi = max(0, idx - 2), min(len(df), idx + 3)
                print(df.loc[lo:hi, ["time", c]].to_string(index=False))
                print("  ---")

    # Show whether airflows and occupancy differ between negative and nonnegative samples.
    if "Occupant count" in df.columns:
        print("\nAirflow and occupancy around negative readings:")
        for c in AIRFLOW_COLS:
            if c in df:
                neg_mask = df[c] < 0
                pos_mask = df[c] >= 0
                print(f"  {c}:")
                print(f"    Mean occupancy when negative: {df.loc[neg_mask, 'Occupant count'].mean():.3f}")
                print(f"    Mean occupancy when nonnegative: {df.loc[pos_mask, 'Occupant count'].mean():.3f}")

    return df

def main():
    print("Focused investigation only: no input CSVs or trained models are modified.")
    loaded = [(p, inspect(p)) for p in FILES]
    loaded = [(p, df) for p, df in loaded if df is not None]

    if loaded:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        summaries = []
        for path, df in loaded:
            for c in AIRFLOW_COLS:
                if c not in df:
                    continue
                s = df[c]
                summaries.append({
                    "file": str(path),
                    "column": c,
                    "negative_count": int((s < 0).sum()),
                    "valid_count": int(s.notna().sum()),
                    "negative_percent": float((s < 0).sum() / max(s.notna().sum(), 1) * 100),
                    "minimum": s.min(),
                    "median": s.median(),
                    "maximum": s.max(),
                })
        out = OUT_DIR / "airflow_investigation.csv"
        pd.DataFrame(summaries).to_csv(out, index=False)
        print(f"\nSaved summary: {out}")

if __name__ == "__main__":
    main()
