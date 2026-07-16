# Turbofan Engine Predictive Maintenance

A full ML system that predicts Remaining Useful Life (RUL) for aircraft turbofan engines from live sensor data — trained, tested, containerized, and deployed end-to-end.

**Live demo:** [Dashboard](https://predictive-maintenance-dashboard-jomb.onrender.com/) · [API](https://predictive-maintenance-api-fyyu.onrender.com/)

## What this does

Given 30 cycles of recent sensor readings from a jet engine, this system predicts how many operating cycles remain before failure — the kind of prediction that lets maintenance teams act *before* something breaks instead of after.

## Results

| Model | MAE (cycles) | RMSE (cycles) |
|---|---|---|
| Random Forest (baseline) | 13.57 | 18.77 |
| **LSTM (final model)** | **6.60** | **9.19** |

The LSTM's accuracy improves further as engines approach failure — MAE of just **1.89 cycles** in the 0-20 RUL range, which is exactly where predictions matter most for real maintenance decisions.

## Architecture

```
┌───────────────────┐         ┌───────────────────┐
│   Streamlit         │  HTTP   │   FastAPI           │
│   Dashboard          │ ──────▶ │   Model Server       │
│   (port 8501)        │         │   (port 8000)        │
└───────────────────┘         └───────────────────┘
                                          │
                                          ▼
                                  Trained LSTM (PyTorch)
```

Both services are containerized independently and deployed as separate services on Render, communicating over HTTP via an environment-configured API URL.

## Tech stack

- **Data & modeling**: pandas, NumPy, scikit-learn, PyTorch (LSTM)
- **Serving**: FastAPI, Uvicorn
- **Dashboard**: Streamlit, Plotly
- **Testing**: pytest
- **CI**: GitHub Actions (automated tests on every push)
- **Containerization**: Docker, docker-compose
- **Deployment**: Render

## Dataset

[NASA C-MAPSS Turbofan Engine Degradation Simulation](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) (FD001 subset) — 100 simulated engines run to failure under one operating condition, with 21 sensor channels recorded per cycle.

## How it works

1. **RUL labeling**: for training data, Remaining Useful Life is calculated per row as `(engine's final cycle) - (current cycle)`, then capped at 125 cycles — a standard technique in this field, since early-life sensor readings can't meaningfully distinguish "200 cycles left" from "300 cycles left."
2. **Sequence windowing**: 30-cycle sliding windows are built per engine so the LSTM can learn from *trends*, not just single snapshots.
3. **Baseline**: a Random Forest trained on individual rows establishes a benchmark.
4. **LSTM**: a 2-layer LSTM trained on windowed sequences beats the baseline by ~40%, confirming degradation is a genuinely time-dependent pattern.
5. **Serving**: the trained model is wrapped in a FastAPI endpoint (`/predict`) that accepts a 30×24 sensor reading window and returns a predicted RUL.
6. **Dashboard**: a Streamlit app lets you pick any test engine and see its live prediction alongside the true answer (available since this is a benchmark dataset).

## Running locally

**With Docker (recommended):**
```bash
docker-compose up --build
```
Dashboard: http://localhost:8501 · API: http://localhost:8000

**Without Docker:**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Terminal 1
cd api && uvicorn main:app --reload

# Terminal 2
streamlit run app.py
```

## Running tests

```bash
pytest tests/ -v
```

Tests cover RUL calculation correctness, multi-engine independence, capping logic, and API input validation. These run automatically on every push via GitHub Actions.

## Project structure

```
predictive-maintenance/
├── archive/CMaps/               # CMAPSS dataset
├── src/
│   ├── data_loader.py           # Data loading utilities
│   ├── feature_engineering.py   # RUL labeling + capping
│   └── sequence_prep.py         # Sliding window generation
├── api/
│   ├── main.py                  # FastAPI model server
│   └── Dockerfile
├── tests/                       # pytest suite
├── app.py                       # Streamlit dashboard
├── Dockerfile.dashboard
├── docker-compose.yml
└── .github/workflows/tests.yml  # CI pipeline
```

## Known limitations

- Predictions beyond ~80 cycles remaining are inherently less precise, since the RUL cap of 125 means the model isn't trained to distinguish exact values in that range — this is by design, not a flaw.
- Free-tier hosting means the API/dashboard may take 30-60s to respond after periods of inactivity (cold start).
- Trained only on FD001 (single operating condition, single fault mode) — the more complex CMAPSS subsets (FD002-FD004) would require retraining.
