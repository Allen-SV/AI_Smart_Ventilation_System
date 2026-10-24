# AI-Based Smart Room Ventilation System — Starter

## Current goal
Build a simulation-ready, multivariate forecasting system that predicts future room conditions from current/historical sensor readings and occupancy, then produces a simulated ventilation control signal.

## Stage 1: Inspect the datasets
1. Create a folder named `data`.
2. Put these files inside it, using the exact names:
   - `training_data_1.csv`
   - `training_data_2.csv`
   - `ventilation_test.csv`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run:
   ```bash
   python inspect_data.py
   ```

## Important modelling rules
- `time` is a timestamp, not a prediction target.
- `Occupant count` is the ground-truth occupancy measurement.
- Do not use `Occupant count (ML estimate)` as an input feature; it is an existing model estimate and may leak information.
- Forecast targets should be future values, not values from the same timestamp as the inputs.
- Do not blindly join every file: the files cover different periods and have different columns.
- Keep `ventilation_test.csv` untouched until model selection is complete.
- A future fan command does not change prerecorded CSV readings. A later simulation-response module will be needed to model the effect of ventilation actions.

## Next stage
After inspecting timestamp gaps, missingness and column overlap, implement chronological window creation and compare multi-output regression baselines. The initial forecast horizon can be configured (for example, 5 minutes) rather than hardcoded.
