from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
TRAIN_FILE = DATA_DIR / "training_data_2.csv"
TEST_FILE = DATA_DIR / "ventilation_test.csv"
TARGET = "Room CO2 (ppm)"
OCCUPANCY = "Occupant count"
HORIZON = 5
LAGS = [0, 1, 2, 3, 5]
EXCLUDED = {"Occupant count (ML estimate)"}

def load(path):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates("time").set_index("time")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def build(df):
    cols = [c for c in df.columns if c not in EXCLUDED and c != TARGET]
    records, ys, ts, current, future, occ = [], [], [], [], [], []
    times = df.index
    lookup = set(times)
    for t in times:
        future_t = t + pd.Timedelta(minutes=HORIZON)
        if future_t not in lookup:
            continue
        if any(t - pd.Timedelta(minutes=lag) not in lookup for lag in LAGS):
            continue
        row = {}
        ok = True
        for lag in LAGS:
            source = df.loc[t - pd.Timedelta(minutes=lag)]
            for c in cols:
                row[f"{c}__lag{lag}"] = source[c]
            row[f"{TARGET}__lag{lag}"] = df.loc[t - pd.Timedelta(minutes=lag), TARGET]
        y = df.loc[future_t, TARGET]
        y0 = df.loc[t, TARGET]
        if pd.isna(y) or pd.isna(y0):
            continue
        records.append(row)
        ys.append(float(y))
        ts.append(t)
        current.append(float(y0))
        future.append(float(y))
        occ.append(df.loc[t, OCCUPANCY] if OCCUPANCY in df.columns else np.nan)
    X = pd.DataFrame(records)
    return X, np.asarray(ys), pd.DatetimeIndex(ts), np.asarray(current), np.asarray(future), np.asarray(occ)

def metric(y, p):
    return mean_absolute_error(y, p), mean_squared_error(y, p) ** 0.5

def condition_report(name, y, pred, current, future, occupancy):
    delta = future - current
    abs_delta = np.abs(delta)
    error = np.abs(y - pred)
    # Mutually exclusive change categories defined using observed 5-minute CO2 change.
    change_bins = [-np.inf, -25, -10, 10, 25, np.inf]
    change_labels = ["Drop >25 ppm", "Drop 10–25 ppm", "Stable (-10 to +10)", "Rise 10–25 ppm", "Rise >25 ppm"]
    change_cat = pd.cut(delta, bins=change_bins, labels=change_labels, right=False)
    rows = []
    for category in change_labels:
        mask = np.asarray(change_cat == category)
        if mask.sum() == 0:
            continue
        rows.append({
            "dataset": name, "analysis": "observed_CO2_change",
            "condition": category, "samples": int(mask.sum()),
            "persistence_MAE": metric(y[mask], current[mask])[0],
            "persistence_RMSE": metric(y[mask], current[mask])[1],
            "persistence_mean_abs_5min_change": float(abs_delta[mask].mean()),
        })
    occ_valid = np.isfinite(occupancy)
    for label, mask in [
        ("Occupancy = 0", occ_valid & (occupancy == 0)),
        ("Occupancy > 0", occ_valid & (occupancy > 0)),
    ]:
        if mask.sum():
            rows.append({
                "dataset": name, "analysis": "occupancy",
                "condition": label, "samples": int(mask.sum()),
                "persistence_MAE": metric(y[mask], current[mask])[0],
                "persistence_RMSE": metric(y[mask], current[mask])[1],
                "persistence_mean_abs_5min_change": float(abs_delta[mask].mean()),
            })
    # Current CO2 regimes, based only on values available at prediction time.
    co2_bins = [-np.inf, 600, 800, 1000, 1200, np.inf]
    co2_labels = ["<600 ppm", "600–800 ppm", "800–1000 ppm", "1000–1200 ppm", ">1200 ppm"]
    co2_cat = pd.cut(current, bins=co2_bins, labels=co2_labels, right=False)
    for category in co2_labels:
        mask = np.asarray(co2_cat == category)
        if mask.sum():
            rows.append({
                "dataset": name, "analysis": "current_CO2_level",
                "condition": category, "samples": int(mask.sum()),
                "persistence_MAE": metric(y[mask], current[mask])[0],
                "persistence_RMSE": metric(y[mask], current[mask])[1],
                "persistence_mean_abs_5min_change": float(abs_delta[mask].mean()),
            })
    return rows

def main():
    for f in [TRAIN_FILE, TEST_FILE]:
        if not f.exists():
            raise FileNotFoundError(f"Missing {f}; run from project root.")

    train = load(TRAIN_FILE)
    test = load(TEST_FILE)
    _, y_all, t_all, current_all, future_all, occ_all = build(train)
    _, y_test, t_test, current_test, future_test, occ_test = build(test)

    # Validation region matches the earlier 80% chronological cutoff, with a horizon embargo.
    cutoff = t_all[int(len(t_all) * 0.80)]
    val_mask = t_all >= cutoff
    val_y = y_all[val_mask]
    val_current = current_all[val_mask]
    val_future = future_all[val_mask]
    val_occ = occ_all[val_mask]

    reports = []
    reports += condition_report("validation", val_y, val_current, val_current, val_future, val_occ)
    reports += condition_report("holdout_test", y_test, current_test, current_test, future_test, occ_test)
    report = pd.DataFrame(reports)

    # Extra dynamics summary, per dataset.
    dynamics = []
    for name, current, future, occ in [
        ("validation", val_current, val_future, val_occ),
        ("holdout_test", current_test, future_test, occ_test),
    ]:
        delta = future - current
        dynamics.append({
            "dataset": name,
            "samples": len(delta),
            "mean_current_CO2_ppm": float(np.mean(current)),
            "mean_future_CO2_ppm": float(np.mean(future)),
            "mean_signed_5min_change_ppm": float(np.mean(delta)),
            "mean_absolute_5min_change_ppm": float(np.mean(np.abs(delta))),
            "median_absolute_5min_change_ppm": float(np.median(np.abs(delta))),
            "percent_change_over_10ppm": float(np.mean(np.abs(delta) > 10) * 100),
            "percent_change_over_25ppm": float(np.mean(np.abs(delta) > 25) * 100),
            "percent_occupancy_zero": float(np.mean(occ == 0) * 100) if np.isfinite(occ).any() else np.nan,
        })
    dyn = pd.DataFrame(dynamics)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    report_path = MODEL_DIR / "co2_condition_analysis.csv"
    dynamics_path = MODEL_DIR / "co2_dynamics_summary.csv"
    report.to_csv(report_path, index=False)
    dyn.to_csv(dynamics_path, index=False)

    print("ROOM CO2 DYNAMICS ANALYSIS")
    print(f"Forecast horizon: {HORIZON} minutes")
    print("\nCO2 changes over the five-minute horizon:")
    print(dyn.round(3).to_string(index=False))
    print("\nPersistence performance by operating condition:")
    print(report.round(3).to_string(index=False))
    print("\nSaved reports:")
    print(f"  {report_path}")
    print(f"  {dynamics_path}")
    print("\nThis analysis is descriptive. Conditions use observed future changes for grouping, not as model inputs.")
    print("No source data or model files were modified.")

if __name__ == "__main__":
    main()
