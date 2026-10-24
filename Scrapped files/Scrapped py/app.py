from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "all_environment_model_1min.joblib"
TIME_COL = "time"
TARGETS = [
    "Room CO2 (ppm)", "Temperature (°C)", "Relative Humidity (%)",
    "Pressure (mbar)", "Supply airflow (l/s)", "Extract airflow (l/s)",
    "Supply air CO2 (ppm)", "Exhaust air CO2 (ppm)",
]
BASE_FEATURES = TARGETS + ["Occupant count"]
DEFAULT_CSVS = ["training_data_2.csv", "ventilation_test.csv"]

st.set_page_config(page_title="Smart Ventilation | CSV Explorer", page_icon="🌿", layout="wide")
st.markdown("""
<style>
.stApp { background:#0b1220; color:#e7eef8; }
[data-testid="stSidebar"] { background:#101b2d; }
.hero { padding:1.1rem 1.4rem; border-radius:16px; background:linear-gradient(120deg,#132b3b,#163c3c); border:1px solid #285653; }
div[data-testid="stMetric"] { background:#142238; border:1px solid #263a55; padding:12px; border-radius:12px; }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🌿 Smart Room Ventilation</h1><p>CSV-driven sensor monitoring · Automatic one-minute-ahead forecasting</p></div>', unsafe_allow_html=True)
st.write("Upload or select a CSV. The app processes its rows in timestamp order automatically, scores the observed conditions during warm-up, and begins forecasting as soon as the required history is available.")

@st.cache_resource
def load_bundle(path):
    return joblib.load(path)

def prepare_csv(df):
    required = [TIME_COL, *BASE_FEATURES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))
    out = df.copy()
    out[TIME_COL] = pd.to_datetime(out[TIME_COL], errors="coerce")
    out = out.dropna(subset=[TIME_COL]).sort_values(TIME_COL)
    out = out.drop_duplicates(TIME_COL, keep="last").set_index(TIME_COL)
    for c in BASE_FEATURES:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def make_feature_row(df, timestamp, feature_columns, lags):
    values = {}
    for lag in lags:
        source_time = timestamp - pd.Timedelta(minutes=int(lag))
        if source_time not in df.index:
            raise ValueError(f"Missing exact timestamp {source_time} required for lag {lag} minute(s).")
        row = df.loc[source_time]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        for col in BASE_FEATURES:
            values[f"{col}__lag_{lag}m"] = row[col]
    X = pd.DataFrame([[values[c] for c in feature_columns]], columns=feature_columns)
    if X.isna().any().any():
        raise ValueError("Missing sensor values in the required lag history.")
    return X

def forecast_for_row(df, timestamp, bundle):
    feature_columns = bundle["feature_columns"]
    targets = bundle.get("targets", TARGETS)
    lags = bundle.get("lags_minutes", [0, 1, 2, 3, 5])
    horizon = int(bundle.get("horizon_minutes", 1))
    raw = np.asarray(bundle["model"].predict(make_feature_row(df, timestamp, feature_columns, lags))).reshape(-1)
    rf_values = {target: float(raw[i]) for i, target in enumerate(targets)}
    current = df.loc[timestamp]
    if isinstance(current, pd.DataFrame):
        current = current.iloc[-1]
    previous_time = timestamp - pd.Timedelta(minutes=1)
    previous = df.loc[previous_time] if previous_time in df.index else None
    methods = bundle.get("selected_methods", {})
    cap = float(bundle.get("co2_trend_cap_ppm", 75.0))
    selected = {}
    for target in targets:
        method = methods.get(target, "random_forest")
        if method == "persistence":
            selected[target] = float(current[target])
        elif method == "trend_capped" and target == "Room CO2 (ppm)" and previous is not None:
            change = float(current[target]) - float(previous[target])
            selected[target] = float(current[target]) + float(np.clip(change, -cap, cap))
        else:
            selected[target] = rf_values[target]
    return selected, rf_values, timestamp + pd.Timedelta(minutes=horizon), methods

def score_ventilation(co2, supply=None, extract=None):
    """
    Project-defined heuristic indicator, not a certified IAQ/safety rating.
    CO2 is the primary driver; airflow imbalance can increase concern modestly.
    """
    if pd.isna(co2):
        return None, "Unavailable", "No CO₂ value available"
    co2 = float(co2)
    # Transparent CO2-based points: 100 = lower concern, 0 = highest concern.
    if co2 <= 800:
        score, label, note = 100, "LOW CONCERN", "CO₂ is at or below the project's 800 ppm reference."
    elif co2 <= 1000:
        score, label, note = 80, "MODERATE", "CO₂ is above 800 ppm; keep monitoring."
    elif co2 <= 1500:
        score, label, note = 55, "HIGH", "CO₂ is elevated under the project's chosen criteria."
    else:
        score, label, note = 25, "CRITICAL", "CO₂ exceeds the project's 1500 ppm reference."
    # If both airflows are valid, a large imbalance modestly lowers score.
    if supply is not None and extract is not None and pd.notna(supply) and pd.notna(extract):
        s, e = float(supply), float(extract)
        if max(abs(s), abs(e)) > 0 and abs(s - e) / max(abs(s), abs(e)) > 0.35:
            score = max(0, score - 10)
            note += " Supply/extract airflow differs by more than 35%, adding a project-defined penalty."
    return int(score), label, note

def build_sequence(df, bundle):
    lags = bundle.get("lags_minutes", [0, 1, 2, 3, 5])
    targets = bundle.get("targets", TARGETS)
    rows = []
    for pos, timestamp in enumerate(df.index):
        current = df.loc[timestamp]
        if isinstance(current, pd.DataFrame):
            current = current.iloc[-1]
        actual_co2 = current.get("Room CO2 (ppm)", np.nan)
        actual_supply = current.get("Supply airflow (l/s)", np.nan)
        actual_extract = current.get("Extract airflow (l/s)", np.nan)
        current_score, current_label, current_note = score_ventilation(actual_co2, actual_supply, actual_extract)
        try:
            pred, raw, forecast_time, methods = forecast_for_row(df, timestamp, bundle)
            pscore, plabel, pnote = score_ventilation(
                pred.get("Room CO2 (ppm)", np.nan),
                pred.get("Supply airflow (l/s)", np.nan),
                pred.get("Extract airflow (l/s)", np.nan),
            )
            actual_next = df.loc[forecast_time] if forecast_time in df.index else None
            rows.append({
                "Input row": pos + 1, "Input timestamp": timestamp,
                "Stage": "Forecast ready", "Forecast timestamp": forecast_time,
                "Current CO₂ (ppm)": actual_co2,
                "Current status": current_label, "Current score": current_score,
                "Predicted CO₂ (ppm)": pred.get("Room CO2 (ppm)", np.nan),
                "Predicted ventilation status": plabel, "Predicted score / 100": pscore,
                "Actual next CO₂ (ppm)": actual_next["Room CO2 (ppm)"] if actual_next is not None else np.nan,
                "Prediction available": True, "_prediction": pred, "_rf": raw,
                "_methods": methods, "_current_note": current_note, "_pred_note": pnote,
            })
        except Exception:
            rows.append({
                "Input row": pos + 1, "Input timestamp": timestamp,
                "Stage": "Warm-up / missing history", "Forecast timestamp": pd.NaT,
                "Current CO₂ (ppm)": actual_co2,
                "Current status": current_label, "Current score": current_score,
                "Predicted CO₂ (ppm)": np.nan, "Predicted ventilation status": "Waiting for history",
                "Predicted score / 100": np.nan, "Actual next CO₂ (ppm)": np.nan,
                "Prediction available": False, "_prediction": None, "_rf": None,
                "_methods": None, "_current_note": current_note, "_pred_note": "",
            })
    return pd.DataFrame(rows)

if not MODEL_PATH.exists():
    st.error(f"Model bundle not found: {MODEL_PATH}")
    st.info("Place models/all_environment_model_1min.joblib in the project folder. It must be the bundle created by your training script.")
    st.stop()
try:
    bundle = load_bundle(str(MODEL_PATH))
    if not isinstance(bundle, dict) or "model" not in bundle or "feature_columns" not in bundle:
        raise ValueError("Unexpected model file format. Use the prediction-ready bundle created by final_training.py.")
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()

with st.sidebar:
    st.header("Dataset")
    source = st.radio("Choose CSV source", ["Project data folder", "Upload a CSV"])
    df_raw, dataset_label = None, ""
    if source == "Project data folder":
        options = [n for n in DEFAULT_CSVS if (DATA_DIR / n).exists()]
        options += sorted(p.name for p in DATA_DIR.glob("*.csv") if p.name not in options)
        if options:
            chosen = st.selectbox("CSV file", options)
            csv_path = DATA_DIR / chosen
            df_raw = pd.read_csv(csv_path)
            dataset_label = str(csv_path.relative_to(ROOT))
        else:
            st.warning("No CSV files found in the data folder.")
    else:
        uploaded = st.file_uploader("Upload dataset", type=["csv"])
        if uploaded is not None:
            df_raw = pd.read_csv(uploaded)
            dataset_label = uploaded.name
    st.caption("Rows are processed automatically in timestamp order. No random readings are generated.")

if df_raw is None:
    st.info("Choose a CSV from the project data folder or upload one to begin.")
    st.stop()
try:
    df = prepare_csv(df_raw)
except Exception as exc:
    st.error(str(exc))
    st.stop()
if df.empty:
    st.error("The selected CSV has no valid timestamped rows.")
    st.stop()

sequence = build_sequence(df, bundle)
ready = sequence[sequence["Prediction available"]]
latest = sequence.iloc[-1]
st.caption(f"Dataset: `{dataset_label}` · {len(df):,} rows · {df.index.min()} to {df.index.max()}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("CSV rows processed", f"{len(df):,}")
c2.metric("Forecast-ready rows", f"{len(ready):,}")
c3.metric("Warm-up / skipped rows", f"{len(df) - len(ready):,}")
c4.metric("Latest predicted CO₂", f"{latest['Predicted CO₂ (ppm)']:.1f} ppm" if pd.notna(latest["Predicted CO₂ (ppm)"]) else "Waiting for history")

tab_live, tab_data, tab_charts, tab_model = st.tabs(["🟢 Automatic run", "📄 CSV data", "📈 Trends", "⚙️ Model details"])

with tab_live:
    st.subheader("Automatic row-by-row processing")
    st.write("The full CSV is processed in timestamp order after upload/selection. Early rows receive a score from their actual observed values while the model warms up; once the required lag history exists, the predicted next-minute values receive their own ventilation score.")
    if pd.notna(latest["Predicted score / 100"]):
        score = int(latest["Predicted score / 100"])
        label = latest["Predicted ventilation status"]
        a, b, c = st.columns([1, 1, 2])
        a.metric("Latest predicted ventilation score", f"{score}/100")
        b.metric("Predicted status", label)
        c.metric("Forecast time", str(latest["Forecast timestamp"]))
        st.progress(score / 100)
        st.info(str(latest["_pred_note"]))
    else:
        a, b = st.columns(2)
        a.metric("Latest current-condition score", f"{int(latest['Current score'])}/100" if pd.notna(latest["Current score"]) else "Unavailable")
        b.metric("Current status", latest["Current status"])
        st.warning("The dataset has not yet supplied the required lag history for a forecast. The current condition is still scored.")
        st.info(str(latest["_current_note"]))

    st.markdown("#### Processing timeline")
    timeline_cols = [
        "Input row", "Input timestamp", "Stage", "Forecast timestamp",
        "Current CO₂ (ppm)", "Current status", "Current score",
        "Predicted CO₂ (ppm)", "Predicted ventilation status", "Predicted score / 100",
        "Actual next CO₂ (ppm)",
    ]
    st.dataframe(sequence[timeline_cols], use_container_width=True, hide_index=True, height=420)
    st.caption("The timeline makes the warm-up period visible: rows without the required lag timestamps are scored using current values, and forecast rows are scored using predicted values.")
    if len(ready):
        st.markdown("#### Latest available forecast details")
        latest_ready = ready.iloc[-1]
        st.caption(f"Input timestamp: {latest_ready['Input timestamp']} → forecast timestamp: {latest_ready['Forecast timestamp']}")
        pred = latest_ready["_prediction"]
        actual_next = df.loc[latest_ready["Forecast timestamp"]] if latest_ready["Forecast timestamp"] in df.index else None
        detail_rows = []
        for target in TARGETS:
            actual = actual_next[target] if actual_next is not None else np.nan
            pv = pred[target]
            detail_rows.append({
                "Parameter": target, "Predicted value": round(float(pv), 4),
                "Actual at forecast time": round(float(actual), 4) if pd.notna(actual) else np.nan,
                "Absolute error": round(abs(float(pv) - float(actual)), 4) if pd.notna(actual) else np.nan,
                "Selected method": latest_ready["_methods"].get(target, "random_forest"),
            })
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
    st.download_button(
        "Download automatic scoring and forecast timeline",
        sequence[timeline_cols].to_csv(index=False).encode("utf-8"),
        "ventilation_forecast_timeline.csv", "text/csv"
    )

with tab_data:
    st.subheader("Actual CSV readings")
    st.caption("These are the original values loaded from the CSV.")
    st.dataframe(df.reset_index(), use_container_width=True, height=420)
    st.download_button("Download loaded data", df.reset_index().to_csv(index=False).encode("utf-8"), "loaded_dataset.csv", "text/csv")

with tab_charts:
    st.subheader("Recorded and predicted trends")
    selected_targets = st.multiselect("Actual parameters to plot", TARGETS, default=["Room CO2 (ppm)", "Supply airflow (l/s)", "Extract airflow (l/s)"])
    if selected_targets:
        st.line_chart(df[selected_targets])
    if len(ready):
        chart = ready.set_index("Forecast timestamp")[["Predicted CO₂ (ppm)", "Actual next CO₂ (ppm)"]].dropna(how="all")
        if not chart.empty:
            st.markdown("#### Predicted vs actual room CO₂")
            st.line_chart(chart)

with tab_model:
    st.write("**Model file:**", str(MODEL_PATH.relative_to(ROOT)))
    st.write("**Forecast horizon:**", f"{bundle.get('horizon_minutes', 1)} minute(s)")
    st.write("**Lag history used:**", bundle.get("lags_minutes", [0, 1, 2, 3, 5]), "minute offsets")
    st.write("**Feature columns:**", len(bundle.get("feature_columns", [])))
    st.json(bundle.get("selected_methods", {}))
    st.warning("Ventilation scores and labels are project-defined heuristic indicators, not certified safety ratings or regulatory thresholds. They use predicted values whenever a forecast is available; current readings are scored during warm-up.")
    st.caption("This is a software dashboard. It does not control physical fans or ventilation hardware.")
