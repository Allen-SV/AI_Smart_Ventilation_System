from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "all_environment_model_5min.joblib"
TIME_COL = "time"
TARGETS = ["Room CO2 (ppm)", "Temperature (°C)", "Relative Humidity (%)"]
OCCUPANCY_COL = "Occupant count"
ENVIRONMENT_INPUTS = [
    "Room CO2 (ppm)", "Temperature (°C)", "Relative Humidity (%)",
    "Pressure (mbar)", "Supply airflow (l/s)", "Extract airflow (l/s)",
    "Supply air CO2 (ppm)", "Exhaust air CO2 (ppm)",
]
BASE_FEATURES = ENVIRONMENT_INPUTS + [OCCUPANCY_COL]
DEFAULT_CSVS = ["training_data_2.csv", "ventilation_test.csv"]

st.set_page_config(page_title="Smart Room Ventilation | 5-Minute Forecast", page_icon="🌿", layout="wide")
st.markdown("""
<style>
.stApp { background:#0b1220; color:#e7eef8; }
[data-testid="stSidebar"] { background:#101b2d; }
.hero { padding:1.2rem 1.5rem; border-radius:16px; background:linear-gradient(120deg,#132b3b,#163c3c); border:1px solid #285653; margin-bottom:1rem; }
div[data-testid="stMetric"] { background:#142238; border:1px solid #263a55; padding:12px; border-radius:12px; }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🌿 Smart Room Ventilation</h1><p>Five-minute indoor condition forecasting · CSV-driven simulation</p></div>', unsafe_allow_html=True)
st.write("Process sensor readings in timestamp order and forecast room CO₂, temperature, and relative humidity five minutes ahead. Actual future readings are compared only when the exact forecast timestamp exists.")

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

def row_at(df, timestamp):
    if timestamp not in df.index:
        return None
    row = df.loc[timestamp]
    return row.iloc[-1] if isinstance(row, pd.DataFrame) else row

def make_feature_row(df, timestamp, feature_columns, lags):
    values = {}
    for lag in lags:
        source_time = timestamp - pd.Timedelta(minutes=int(lag))
        row = row_at(df, source_time)
        if row is None:
            raise ValueError(f"Missing exact timestamp {source_time} required for lag {lag} minute(s).")
        for col in BASE_FEATURES:
            values[f"{col}__lag_{lag}m"] = row[col]
    missing = [c for c in feature_columns if c not in values]
    if missing:
        raise ValueError("Could not construct model features: " + ", ".join(missing[:5]))
    X = pd.DataFrame([[values[c] for c in feature_columns]], columns=feature_columns)
    if X.isna().any().any():
        raise ValueError("Missing sensor values in the required lag history.")
    return X

def forecast_for_row(df, timestamp, bundle):
    targets = bundle.get("targets", [])
    if list(targets) != TARGETS:
        raise ValueError("Model targets must be room CO₂, temperature, and relative humidity.")
    horizon = int(bundle.get("horizon_minutes", -1))
    if horizon != 5:
        raise ValueError(f"Expected a five-minute model; bundle horizon is {horizon} minute(s).")
    lags = bundle.get("lags_minutes", [0, 1, 2, 3, 5])
    X = make_feature_row(df, timestamp, bundle["feature_columns"], lags)
    raw = np.asarray(bundle["model"].predict(X)).reshape(-1)
    if len(raw) != len(targets):
        raise ValueError("Model output count does not match its target list.")
    raw_predictions = {t: float(raw[i]) for i, t in enumerate(targets)}
    current = row_at(df, timestamp)
    previous = row_at(df, timestamp - pd.Timedelta(minutes=1))
    methods = bundle.get("selected_methods", {})
    cap = float(bundle.get("co2_trend_cap_ppm", 75.0))
    selected = {}
    for target in targets:
        method = methods.get(target, "random_forest")
        if method == "persistence":
            selected[target] = float(current[target])
        elif method == "trend_capped" and target == "Room CO2 (ppm)":
            if previous is None or pd.isna(previous[target]) or pd.isna(current[target]):
                raise ValueError("CO₂ trend prediction requires valid current and previous-minute readings.")
            change_per_minute = float(current[target]) - float(previous[target])
            selected[target] = float(current[target] + np.clip(change_per_minute * horizon, -cap, cap))
        else:
            selected[target] = raw_predictions[target]
    return selected, timestamp + pd.Timedelta(minutes=horizon), methods

def score_ventilation(co2):
    """Project-defined heuristic only; not a certified safety or regulatory rating."""
    if pd.isna(co2):
        return None, "Unavailable", "No predicted CO₂ value is available."
    co2 = float(co2)
    if co2 <= 800:
        return 100, "LOW CONCERN", "Predicted CO₂ is at or below the project's 800 ppm reference."
    if co2 <= 1000:
        return 80, "MODERATE", "Predicted CO₂ is above 800 ppm; continue monitoring."
    if co2 <= 1500:
        return 55, "HIGH", "Predicted CO₂ is elevated under the project's chosen criteria."
    return 25, "CRITICAL", "Predicted CO₂ exceeds the project's 1500 ppm reference."

def build_sequence(df, bundle):
    records = []
    for pos, timestamp in enumerate(df.index):
        cur = row_at(df, timestamp)
        rec = {
            "Input row": pos + 1, "Input timestamp": timestamp,
            "Occupant count": cur.get(OCCUPANCY_COL, np.nan),
            "Current CO₂ (ppm)": cur.get(TARGETS[0], np.nan),
            "Current temperature (°C)": cur.get(TARGETS[1], np.nan),
            "Current relative humidity (%)": cur.get(TARGETS[2], np.nan),
            "Stage": "Forecast unavailable", "Forecast timestamp": pd.NaT,
            "Predicted CO₂ (ppm)": np.nan, "Predicted temperature (°C)": np.nan,
            "Predicted relative humidity (%)": np.nan,
            "Actual CO₂ at forecast (ppm)": np.nan,
            "Actual temperature at forecast (°C)": np.nan,
            "Actual relative humidity at forecast (%)": np.nan,
            "CO₂ absolute error (ppm)": np.nan, "Temperature absolute error (°C)": np.nan,
            "Humidity absolute error (percentage points)": np.nan,
            "Ventilation score / 100": np.nan, "Ventilation status": "Waiting for history",
            "Prediction available": False, "_prediction": None, "_methods": None, "_error": "",
        }
        try:
            pred, forecast_time, methods = forecast_for_row(df, timestamp, bundle)
            future = row_at(df, forecast_time)
            score, status, note = score_ventilation(pred[TARGETS[0]])
            rec.update({
                "Stage": "Forecast ready", "Forecast timestamp": forecast_time,
                "Predicted CO₂ (ppm)": pred[TARGETS[0]],
                "Predicted temperature (°C)": pred[TARGETS[1]],
                "Predicted relative humidity (%)": pred[TARGETS[2]],
                "Ventilation score / 100": score, "Ventilation status": status,
                "Prediction available": True, "_prediction": pred, "_methods": methods, "_note": note,
            })
            if future is not None:
                pairs = [
                    (TARGETS[0], "Predicted CO₂ (ppm)", "Actual CO₂ at forecast (ppm)", "CO₂ absolute error (ppm)"),
                    (TARGETS[1], "Predicted temperature (°C)", "Actual temperature at forecast (°C)", "Temperature absolute error (°C)"),
                    (TARGETS[2], "Predicted relative humidity (%)", "Actual relative humidity at forecast (%)", "Humidity absolute error (percentage points)"),
                ]
                for target, pred_col, actual_col, error_col in pairs:
                    actual = future[target]
                    rec[actual_col] = actual
                    if pd.notna(actual):
                        rec[error_col] = abs(float(pred[target]) - float(actual))
        except Exception as exc:
            rec["_error"] = str(exc)
            rec["Stage"] = "Warm-up / missing history" if pos < max(bundle.get("lags_minutes", [0,1,2,3,5])) else "Forecast unavailable"
        records.append(rec)
    return pd.DataFrame(records)

if not MODEL_PATH.exists():
    st.error(f"Five-minute model bundle not found: {MODEL_PATH}")
    st.info("Train final_training_5min.py first. Expected: models/all_environment_model_5min.joblib")
    st.stop()
try:
    bundle = load_bundle(str(MODEL_PATH))
    if not isinstance(bundle, dict) or "model" not in bundle or "feature_columns" not in bundle:
        raise ValueError("Unexpected model bundle format.")
    if int(bundle.get("horizon_minutes", -1)) != 5 or list(bundle.get("targets", [])) != TARGETS:
        raise ValueError("The loaded bundle must be the five-minute model for CO₂, temperature, and humidity.")
except Exception as exc:
    st.error(f"Could not load the five-minute model: {exc}")
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
            df_raw, dataset_label = pd.read_csv(csv_path), str(csv_path.relative_to(ROOT))
        else:
            st.warning("No CSV files found in the data folder.")
    else:
        uploaded = st.file_uploader("Upload dataset", type=["csv"])
        if uploaded is not None:
            df_raw, dataset_label = pd.read_csv(uploaded), uploaded.name
    st.caption("Rows are processed chronologically. No random sensor readings are generated.")
if df_raw is None:
    st.info("Choose a CSV or upload one to begin.")
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
a, b, c, d = st.columns(4)
a.metric("CSV rows processed", f"{len(df):,}")
b.metric("Forecast-ready rows", f"{len(ready):,}")
c.metric("Rows without forecast", f"{len(df)-len(ready):,}")
d.metric("Latest forecast time", str(latest["Forecast timestamp"]) if pd.notna(latest["Forecast timestamp"]) else "Unavailable")

tab_forecast, tab_timeline, tab_data, tab_trends, tab_model = st.tabs(["🌡️ Forecast dashboard", "🧭 Simulation timeline", "📄 CSV data", "📈 Trends", "⚙️ Model details"])
with tab_forecast:
    st.subheader("Latest five-minute forecast")
    if bool(latest["Prediction available"]):
        st.caption(f"Input time: {latest['Input timestamp']} → forecast time: {latest['Forecast timestamp']}")
        specs = [
            ("Room CO₂", "ppm", "Current CO₂ (ppm)", "Predicted CO₂ (ppm)"),
            ("Temperature", "°C", "Current temperature (°C)", "Predicted temperature (°C)"),
            ("Relative humidity", "%", "Current relative humidity (%)", "Predicted relative humidity (%)"),
        ]
        cols = st.columns(3)
        for col, (label, unit, current_col, pred_col) in zip(cols, specs):
            current, predicted = latest[current_col], latest[pred_col]
            if pd.notna(predicted):
                delta = float(predicted) - float(current) if pd.notna(current) else None
                col.metric(f"{label} forecast", f"{float(predicted):.2f} {unit}", delta=f"{delta:+.2f} {unit} vs current" if delta is not None else None)
            else:
                col.metric(f"{label} forecast", "Unavailable")
        st.markdown("#### Ventilation recommendation")
        score = latest["Ventilation score / 100"]
        x, y = st.columns([1, 2])
        x.metric("Project-defined CO₂ score", f"{int(score)}/100" if pd.notna(score) else "Unavailable")
        y.metric("Forecast status", latest["Ventilation status"])
        if pd.notna(score): st.progress(float(score)/100)
        st.info(latest.get("_note", "Recommendation is based on the project's CO₂ heuristic."))
        st.caption("This is a project-defined indicator, not a certified safety rating. CO₂ does not measure every indoor pollutant.")
        st.markdown("#### Actual values at the forecast timestamp")
        future = row_at(df, latest["Forecast timestamp"])
        detail = []
        for target, label, unit, pred_col, actual_col in [
            (TARGETS[0], "Room CO₂", "ppm", "Predicted CO₂ (ppm)", "Actual CO₂ at forecast (ppm)"),
            (TARGETS[1], "Temperature", "°C", "Predicted temperature (°C)", "Actual temperature at forecast (°C)"),
            (TARGETS[2], "Relative humidity", "%", "Predicted relative humidity (%)", "Actual relative humidity at forecast (%)"),
        ]:
            actual = future[target] if future is not None else np.nan
            predicted = latest[pred_col]
            detail.append({"Parameter": label, "Predicted": predicted, "Actual at forecast time": actual,
                           "Absolute error": abs(float(predicted)-float(actual)) if pd.notna(predicted) and pd.notna(actual) else np.nan, "Unit": unit})
        st.dataframe(pd.DataFrame(detail), use_container_width=True, hide_index=True)
        if future is None: st.info("No reading exists at the exact forecast timestamp; actual values and errors are unavailable.")
    else:
        st.warning("A forecast is not available for the latest row. Check the timeline for missing timestamps or sensor readings.")
        if latest.get("_error"): st.code(str(latest["_error"]))
        for col, label, value, unit in zip(st.columns(3), ["Current CO₂", "Current temperature", "Current humidity"], [latest["Current CO₂ (ppm)"], latest["Current temperature (°C)"], latest["Current relative humidity (%)"]], ["ppm", "°C", "%"]):
            col.metric(label, f"{float(value):.2f} {unit}" if pd.notna(value) else "Unavailable")
    st.markdown("#### Latest input readings")
    for col, label, value in zip(st.columns(4), ["Occupant count", "Current CO₂", "Current temperature", "Current humidity"], [latest["Occupant count"], latest["Current CO₂ (ppm)"], latest["Current temperature (°C)"], latest["Current relative humidity (%)"]]):
        col.metric(label, f"{float(value):.2f}" if pd.notna(value) else "Unavailable")

with tab_timeline:
    st.subheader("Row-by-row simulation")
    st.write("Each input timestamp produces a forecast five minutes ahead. Actual values and errors appear only when the exact forecast timestamp is present.")
    cols = ["Input row", "Input timestamp", "Occupant count", "Stage", "Forecast timestamp",
            "Current CO₂ (ppm)", "Predicted CO₂ (ppm)", "Actual CO₂ at forecast (ppm)", "CO₂ absolute error (ppm)",
            "Current temperature (°C)", "Predicted temperature (°C)", "Actual temperature at forecast (°C)", "Temperature absolute error (°C)",
            "Current relative humidity (%)", "Predicted relative humidity (%)", "Actual relative humidity at forecast (%)", "Humidity absolute error (percentage points)", "Ventilation score / 100", "Ventilation status"]
    st.dataframe(sequence[cols], use_container_width=True, hide_index=True, height=460)
    errors = sequence.loc[~sequence["Prediction available"] & sequence["_error"].astype(str).ne(""), ["Input timestamp", "Stage", "_error"]]
    if not errors.empty:
        with st.expander("Why some forecasts were unavailable"):
            st.dataframe(errors.rename(columns={"_error": "Reason"}), use_container_width=True, hide_index=True)
    st.download_button("Download five-minute forecast timeline", sequence[cols].to_csv(index=False).encode("utf-8"), "ventilation_forecast_timeline_5min.csv", "text/csv")

with tab_data:
    st.subheader("Source CSV readings")
    st.caption("Original readings after timestamp parsing and duplicate timestamp removal.")
    st.dataframe(df.reset_index(), use_container_width=True, height=420)
    st.download_button("Download loaded data", df.reset_index().to_csv(index=False).encode("utf-8"), "loaded_dataset.csv", "text/csv")

with tab_trends:
    st.subheader("Actual and predicted trends")
    selected = st.multiselect("Recorded parameters", TARGETS, default=TARGETS)
    if selected:
        st.markdown("#### Recorded readings")
        st.line_chart(df[selected])
    if not ready.empty:
        for title, pred_col, actual_col in [
            ("Room CO₂: predicted vs actual", "Predicted CO₂ (ppm)", "Actual CO₂ at forecast (ppm)"),
            ("Temperature: predicted vs actual", "Predicted temperature (°C)", "Actual temperature at forecast (°C)"),
            ("Relative humidity: predicted vs actual", "Predicted relative humidity (%)", "Actual relative humidity at forecast (%)"),
        ]:
            chart = ready.set_index("Forecast timestamp")[[pred_col, actual_col]].dropna(how="all")
            if not chart.empty:
                st.markdown(f"#### {title}")
                st.line_chart(chart)

with tab_model:
    st.subheader("Five-minute model configuration")
    st.write("**Model file:**", str(MODEL_PATH.relative_to(ROOT)))
    st.write("**Forecast horizon:**", f"{bundle.get('horizon_minutes', 5)} minute(s)")
    st.write("**Historical offsets:**", bundle.get("lags_minutes", [0, 1, 2, 3, 5]), "minute offsets")
    st.write("**Prediction targets:**", TARGETS)
    st.write("**Input feature count:**", len(bundle.get("feature_columns", [])))
    st.write("**Selected method per target:**")
    st.json(bundle.get("selected_methods", {}))
    st.write("**Occupancy input:**", OCCUPANCY_COL)
    st.warning("Ventilation scores are project-defined heuristics, not certified safety ratings or regulatory thresholds. CO₂ does not represent every indoor pollutant.")
    st.caption("This is a software dashboard. It does not control physical fans or ventilation hardware.")
