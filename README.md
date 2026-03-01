# ICU Census Prediction API

> Production ML system that forecasts next-day ICU bed occupancy using arrival modeling, length-of-stay prediction, and discharge hazard analysis. Built with real de-identified hospital data, served via FastAPI, containerized with Docker, and deployed on GCP Cloud Run.

**Live API:** [https://icu-census-api-460987588868.us-central1.run.app/docs](https://icu-census-api-460987588868.us-central1.run.app/docs)

## Why This Matters

ICU capacity planning is one of the highest-stakes operational challenges in healthcare. Underestimating demand leads to patient diversions and worse outcomes; overestimating wastes expensive staffed beds. This project builds an **end-to-end ML pipeline** that combines three modeling components — arrival forecasting, LOS regression, and discharge probability estimation — into a unified ICU census forecast served as a REST API that hospital operations teams can integrate into dashboards and scheduling systems.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Arrival RF  │     │  LOS Ridge   │     │  Hazard Table    │
│  (400 trees) │     │  (log-scale) │     │  (empirical S(t))│
└──────┬───────┘     └──────┬───────┘     └────────┬─────────┘
       │                    │                      │
       ▼                    ▼                      ▼
┌──────────────────────────────────────────────────────────────┐
│              Census Balance Equation                         │
│   Census_t+1 = Census_t + ICU_arrivals − Expected_discharges│
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  FastAPI     │
                    │  REST API    │
                    │  (6 endpoints)│
                    └──────┬───────┘
                           │
                    ┌──────┴───────┐
                    │ GCP Cloud Run│
                    └──────────────┘
```

## Approach

| Component | Purpose | Method |
|-----------|---------|--------|
| **Arrival Forecasting** | Predict daily hospital admissions | Random Forest on lag/rolling/calendar features |
| **LOS Modeling** | Estimate total length of stay | Ridge Regression (log-transformed target) |
| **Discharge Hazard** | P(discharge \| days in ICU) | Empirical survival curve from LOS distribution |
| **Census Forecast** | Project next-day bed occupancy | Balance equation: Census₊₁ = Census + Arrivals − Discharges |
| **Short-Stay Classifier** | Segment ≤2-day stays | Logistic Regression (balanced class weights) |

Data covers **Feb 2024 – Jan 2025** from two hospital sources: daily inpatient level-of-care records (~1,700 encounters) and TeleTracking bed-request timestamps (~5,000 requests).

## Key Results

| Task | Model | Metric | Value |
|------|-------|--------|-------|
| Arrival Forecast (14-day holdout) | Random Forest | MAE | 5.29 |
| Arrival Forecast (14-day holdout) | Random Forest | MAPE | 9.85% |
| LOS Prediction (20% holdout) | Ridge Regression | MAE | 6.85 days |
| LOS Prediction (20% holdout) | Naive Median Baseline | MAE | 7.72 days |
| Short-Stay Classification | Logistic Regression | AUC | 0.797 |
| Short-Stay Classification | Logistic Regression | F1 | 0.623 |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/model-info` | GET | Model metrics and metadata |
| `/predict/arrivals` | POST | Predict next-day arrivals from lag features |
| `/predict/los` | POST | Predict length of stay for a single patient |
| `/predict/los/batch` | POST | Batch LOS prediction (up to 500 patients) |
| `/predict/short-stay` | POST | Classify short-stay probability |
| `/predict/census` | POST | **Full next-day ICU census forecast** |

### Example: Census Forecast

```bash
curl -X POST https://icu-census-api-460987588868.us-central1.run.app/predict/census \
  -H "Content-Type: application/json" \
  -d '{
    "current_census": 28,
    "icu_days_completed": [0,1,2,3,5,7,10,1,2,0,3,4,5,6,8,1,2,3,4,5,0,1,6,7,12,15,2,3],
    "predicted_arrivals": 55,
    "icu_share": 0.15
  }'
```

Response:
```json
{
  "current_census": 28,
  "predicted_total_arrivals": 55.0,
  "icu_share": 0.15,
  "predicted_icu_arrivals": 8.2,
  "expected_discharges": 5.72,
  "forecasted_census_tomorrow": 30.5
}
```

## Project Structure

```
ICU-Census-Prediction/
├── src/
│   ├── api.py                   # FastAPI REST API (6 endpoints)
│   ├── predict.py               # Inference engine (loads serialized models)
│   ├── pipeline.py              # End-to-end training orchestrator
│   ├── data_loader.py           # Load & clean Excel sources
│   ├── feature_engineering.py   # Arrival features, LOS features, hazard table
│   ├── models.py                # Train & evaluate all models
│   ├── census_simulator.py      # Forecast ICU census via balance equation
│   └── visualizations.py        # Dashboard and plot generation
├── models/
│   ├── arrival_rf.joblib        # Trained arrival Random Forest
│   ├── los_ridge.joblib         # Trained LOS Ridge model
│   ├── short_stay_logit.joblib  # Trained short-stay classifier
│   ├── icu_hazard_table.csv     # Empirical discharge probabilities
│   └── model_metadata.json      # Features, metrics, config
├── tests/
│   └── test_api.py              # 12 integration tests (all passing)
├── save_models.py               # Train and serialize all models
├── Dockerfile                   # Container config
├── .gcloudignore                # Cloud Run deploy config
├── requirements.txt
└── README.md
```

## Setup & Run

### Local Development

```bash
git clone https://github.com/yashraj10/ICU-Census-Prediction.git
cd ICU-Census-Prediction
pip install -r requirements.txt

# Train models (requires data files in project root)
python save_models.py

# Start API
uvicorn src.api:app --reload --port 8080

# Open http://localhost:8080/docs for interactive Swagger UI

# Run tests
pytest tests/test_api.py -v
```

### Docker

```bash
docker build -t icu-census-api .
docker run -p 8080:8080 icu-census-api
```

### Deploy to GCP Cloud Run

```bash
gcloud run deploy icu-census-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi
```

## Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.11 |
| ML | scikit-learn (Random Forest, Ridge, Logistic Regression) |
| API | FastAPI, Pydantic v2, Uvicorn |
| Testing | pytest (12 integration tests) |
| Container | Docker |
| Cloud | GCP Cloud Run, Artifact Registry, Cloud Build |
| Data | pandas, NumPy, openpyxl |

## What I'd Improve Next

- Incorporate **clinical features** (diagnosis codes, acuity scores) to improve LOS prediction
- Add **survival analysis** (Cox PH / Kaplan-Meier) as an alternative to the empirical hazard model
- Build a **Streamlit dashboard** for interactive scenario exploration (e.g., "what if arrivals spike 20%?")
- Model **ICU admission share** separately instead of using a static 15% parameter
- Add **CI/CD pipeline** with GitHub Actions for automated testing and deployment
- Extend framework to **PCU and Telemetry** units

## Author

**Yashraj Jadhav** — MS Business Analytics, USC Marshall
[LinkedIn](https://www.linkedin.com/in/yashrajjadhav/) · [Email](mailto:yjadhav@marshall.usc.edu)
