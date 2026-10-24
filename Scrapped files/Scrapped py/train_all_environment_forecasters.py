"""Evaluate the shared one-minute trend method for all environmental targets.

Run from the project root:
    python train_all_environment_forecasters.py

Inputs:
    data/training_data_2.csv
    data/ventilation_test.csv

If those paths do not exist, the script also checks the script's directory.

The independent test file is used only after the method for each target has been
selected using a chronological validation split from training_data_2.csv.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HORIZON_MINUTES = 5
VALIDATION_FRACTION = 0.20
CO2_CAP = 75.0
TARGETS = [
    'Room CO2 (ppm)',
    'Temperature (°C)',
    'Relative Humidity (%)',
    'Pressure (mbar)',
    'Supply airflow (l/s)',
    'Extract airflow (l/s)',
    'Supply air CO2 (ppm)',
    'Exhaust air CO2 (ppm)',
]
TIME_COL = 'time'
OCCUPANCY_COL = 'Occupant count'


def locate_file(filename: str) -> Path:
    candidates = [Path('data') / filename, Path(__file__).resolve().parent / 'data' / filename,
                  Path(filename), Path(__file__).resolve().parent / filename]
    for path in candidates:
        if path.exists():
            return path
    checked = ', '.join(str(p) for p in candidates)
    raise FileNotFoundError(f'Could not find {filename}. Checked: {checked}')


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TIME_COL not in df.columns:
        raise ValueError(f"Missing required column '{TIME_COL}' in {path}")
    missing = [c for c in TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f'Missing target columns in {path}: {missing}')
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors='coerce')
    df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL).drop_duplicates(TIME_COL, keep='last')
    for col in TARGETS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.reset_index(drop=True)


def make_examples(df: pd.DataFrame) -> pd.DataFrame:
    """Pair current readings with exact one-minute history and exact +5-minute targets."""
    indexed = df.set_index(TIME_COL)
    current_times = indexed.index
    previous_times = current_times - pd.Timedelta(minutes=1)
    future_times = current_times + pd.Timedelta(minutes=HORIZON_MINUTES)
    has_previous = previous_times.isin(indexed.index)
    has_future = future_times.isin(indexed.index)
    keep = has_previous & has_future
    times = current_times[keep]
    current = indexed.loc[times, TARGETS]
    previous = indexed.loc[times - pd.Timedelta(minutes=1), TARGETS]
    future = indexed.loc[times + pd.Timedelta(minutes=HORIZON_MINUTES), TARGETS]
    out = pd.DataFrame({'time': times})
    for col in TARGETS:
        out[f'current::{col}'] = current[col].to_numpy()
        out[f'previous::{col}'] = previous[col].to_numpy()
        out[f'actual::{col}'] = future[col].to_numpy()
        out[f'trend::{col}'] = current[col].to_numpy() + HORIZON_MINUTES * (
            current[col].to_numpy() - previous[col].to_numpy())
    out = out.replace([np.inf, -np.inf], np.nan)
    required = [f'current::{c}' for c in TARGETS] + [f'previous::{c}' for c in TARGETS] + [f'actual::{c}' for c in TARGETS]
    return out.dropna(subset=required).reset_index(drop=True)


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    if len(actual) == 0:
        return {'n': 0, 'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan}
    errors = predicted - actual
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - np.sum(errors ** 2) / ss_tot if ss_tot > 0 else np.nan
    return {'n': len(actual), 'MAE': float(np.mean(np.abs(errors))),
            'RMSE': float(np.sqrt(np.mean(errors ** 2))), 'R2': float(r2)}


def main():
    output_dir = Path('models')
    if not output_dir.exists():
        output_dir = Path(__file__).resolve().parent / 'models'
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = locate_file('training_data_2.csv')
    test_path = locate_file('ventilation_test.csv')
    train_raw = load_data(train_path)
    test_raw = load_data(test_path)
    train = make_examples(train_raw)
    test = make_examples(test_raw)
    if len(train) < 20:
        raise ValueError(f'Too few aligned training examples: {len(train)}')
    split = int(len(train) * (1 - VALIDATION_FRACTION))
    fit_rows = train.iloc[:split].copy()
    val_rows = train.iloc[split:].copy()
    if len(val_rows) == 0:
        raise ValueError('Validation set is empty')

    # Caps are determined only from the earlier training segment. CO2 retains
    # the previously tested 75 ppm cap; all other caps are scaled to each target.
    caps = {}
    for col in TARGETS:
        if col == 'Room CO2 (ppm)':
            caps[col] = CO2_CAP
        else:
            raw_adjustment = np.abs(fit_rows[f'trend::{col}'] - fit_rows[f'current::{col}'])
            caps[col] = float(raw_adjustment.quantile(0.95))
            if not np.isfinite(caps[col]) or caps[col] <= 0:
                caps[col] = float(raw_adjustment.max()) if len(raw_adjustment) else 0.0

    def candidate_predictions(rows, col):
        current = rows[f'current::{col}'].to_numpy(float)
        trend = rows[f'trend::{col}'].to_numpy(float)
        cap = caps[col]
        capped = current + np.clip(trend - current, -cap, cap)
        return {'persistence': current, 'trend_uncapped': trend, 'trend_capped': capped}

    validation_metrics = []
    selected = {}
    for col in TARGETS:
        actual = val_rows[f'actual::{col}'].to_numpy(float)
        candidates = candidate_predictions(val_rows, col)
        per_target = []
        for method, pred in candidates.items():
            m = regression_metrics(actual, pred)
            per_target.append((method, m))
            validation_metrics.append({'split': 'validation', 'target': col, 'method': method,
                                      **m, 'cap': caps[col] if method == 'trend_capped' else np.nan})
        selected[col] = min(per_target, key=lambda item: item[1]['MAE'])[0]

    test_metrics = []
    prediction_rows = []
    for col in TARGETS:
        actual = test[f'actual::{col}'].to_numpy(float)
        candidates = candidate_predictions(test, col)
        baseline = regression_metrics(actual, candidates['persistence'])
        for method, pred in candidates.items():
            m = regression_metrics(actual, pred)
            improvement = ((baseline['MAE'] - m['MAE']) / baseline['MAE'] * 100
                           if baseline['MAE'] and np.isfinite(baseline['MAE']) else np.nan)
            test_metrics.append({'split': 'holdout_test', 'target': col, 'method': method,
                                 **m, 'MAE_improvement_vs_persistence_pct': improvement,
                                 'cap': caps[col] if method == 'trend_capped' else np.nan,
                                 'selected_by_validation': method == selected[col]})
            if method == selected[col]:
                prediction_rows.append(pd.DataFrame({
                    'time': test['time'], 'target': col,
                    'actual_5min_ahead': actual, 'current_value': test[f'current::{col}'].to_numpy(float),
                    'prediction': pred, 'selected_method': method,
                }))

    val_df = pd.DataFrame(validation_metrics)
    test_df = pd.DataFrame(test_metrics)
    pred_df = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    val_path = output_dir / 'all_environment_validation_metrics.csv'
    test_metrics_path = output_dir / 'all_environment_holdout_metrics.csv'
    pred_path = output_dir / 'all_environment_holdout_predictions.csv'
    config_path = output_dir / 'all_environment_forecasting_config.json'
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_metrics_path, index=False)
    pred_df.to_csv(pred_path, index=False)
    config = {
        'forecast_horizon_minutes': HORIZON_MINUTES,
        'history_interval_minutes': 1,
        'targets': TARGETS,
        'validation_fraction': VALIDATION_FRACTION,
        'selected_methods_by_validation_MAE': selected,
        'trend_adjustment_caps_by_target': caps,
        'cap_method': 'Room CO2 uses the previously tested fixed +/-75 ppm; other caps are the 95th percentile of absolute 5-minute trend adjustments in the earlier training segment.',
        'training_file': str(train_path),
        'holdout_file': str(test_path),
        'occupancy_columns_not_used_as_targets_or_predictors': ['Occupant count', 'Occupant count (ML estimate)'],
        'note': 'This is a deterministic trend-forecasting rule, not a fitted machine-learning estimator. Holdout data was not used to select methods or caps.'
    }
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')

    print('\n=== VALIDATION RESULTS (used to select a method per target) ===')
    print(val_df.pivot(index='target', columns='method', values='MAE').round(4).to_string())
    print('\nSelected method per target:')
    for col in TARGETS:
        print(f'  {col}: {selected[col]} (cap={caps[col]:.6g})')
    print('\n=== INDEPENDENT HOLDOUT RESULTS ===')
    print(test_df[['target', 'method', 'n', 'MAE', 'RMSE', 'R2', 'MAE_improvement_vs_persistence_pct', 'selected_by_validation']].round(4).to_string(index=False))
    print('\nSaved files:')
    print(f'  {val_path.resolve()}')
    print(f'  {test_metrics_path.resolve()}')
    print(f'  {pred_path.resolve()}')
    print(f'  {config_path.resolve()}')
    print('\nNote: MAE/RMSE/R2 are regression metrics. F1 is not appropriate for continuous forecasts.')

if __name__ == '__main__':
    main()
