# Housing Price Predictor

A production-ready MLOps service that generates instant, data-driven price estimates for residential property listings from 12 property attributes.

[![MLOps Quality Gate](https://github.com/carlos13suarez/G2_MLOPS_Project/actions/workflows/ci.yml/badge.svg)](https://github.com/carlos13suarez/G2_MLOPS_Project/actions/workflows/ci.yml)
[![Live on Render](https://img.shields.io/badge/API-Live%20on%20Render-brightgreen)](https://housing-price-predictor-jfz4.onrender.com/health)

---

## Business Objective

Real estate agencies price new listings inconsistently. Without a formal appraisal, agents rely on intuition — the same property receives different estimates depending on who handles it. Overpriced listings sit on the market; underpriced ones close fast but leave revenue on the table.

This service is an automated first-pass valuation tool. The moment a new listing is registered, agents input 12 attributes they already collect at intake — size, layout, amenities, location indicators — and receive a data-driven reference price before any further assessment is needed.

**Users:** Listing agents (primary), sellers, agency management, and financial institutions seeking an independent cross-check on declared property values.

**Deployment condition:** Model output is always reviewed by an agent before being communicated to a seller. It is never surfaced as a final price.

---

## Success Metrics

**Business KPIs:**
- Pricing turnaround — estimate available at listing creation (0 wait time)
- Agent consistency — every agent starts from the same model-generated reference
- Estimation accuracy — predicted price within ±20% of actual sale price at the median

*At the median sale price of 4.34M, ±20% = ±868,000. The achieved MAE of 747,580 falls within this band, making predictions commercially useful as a first-pass anchor.*

**Technical acceptance criteria:**

| Criterion | Threshold | Result |
|---|---|---|
| R² | ≥ 0.65 | 0.653 |
| Adjusted R² | ≥ 0.64 | 0.644 |
| MAE | ≤ 868,000 (±20% of median) | 747,580 |
| RMSE | — | 1,039,102 |
| CV stability | No single fold deviates > 5% R² from mean | Confirmed across 5 folds |

---

## Architecture Overview

The system is a linear pipeline from raw data to a live REST endpoint:

```
data/raw/Housing.csv → main.py → W&B (metrics + artifacts) → models/model.joblib → api.py → Render
```

**MLOps layers:**

| Layer | Implementation |
|---|---|
| Configuration | `config.yaml` — single source of truth for all non-secret settings |
| Secrets | `.env` — loaded at runtime via `python-dotenv`, never committed |
| Logging | `src/logger.py` — dual `StreamHandler` + `FileHandler` output |
| Experiment tracking | Weights & Biases — metrics, model artifacts, inference logs |
| CI/CD | GitHub Actions — quality gate on PRs, deploy hook on Release |
| Deployment | Render — containerised FastAPI service |

**Model comparison — five models were developed; Model 5 was selected:**

| Model | Approach | R² | Adj. R² |
|---|---|---|---|
| 1 — Baseline | Binary encoding + OHE + StandardScaler | ~0.63 | ~0.62 |
| 2 — Log Price | + log(price) + log(area) | ~0.65 | ~0.64 |
| 3 — Log + Lasso | + LassoCV feature selection | ~0.64 | ~0.63 |
| 4 — Log + Outlier Removal | + IQR outlier removal on training set | ~0.65 | ~0.64 |
| **5 — K-Fold CV** | **Model 2 preprocessing + 5-fold cross-validation** | **0.653** | **0.644** |

Model 5 was selected because K-Fold CV ensures performance is not an artefact of a single favourable train/test split — the result holds across all five data partitions.

---

## Repository Structure

```text
G2_MLOPS_Project/
├── README.md
├── config.yaml                        # All non-secret runtime settings (60+ keys)
├── environment.yml                    # Conda environment specification
├── conda-lock.yml                     # Pinned Linux-64 lockfile for reproducibility
├── Dockerfile                         # Container image for the FastAPI service
├── .dockerignore                      # Allowlist — only src/, models/, config.yaml copied
├── pytest.ini                         # Test discovery configuration
│
├── .github/
│   └── workflows/
│       ├── ci.yml                     # Quality gate — runs on all PRs to main
│       └── deploy.yml                 # CD — triggers on published GitHub Release
│
├── src/
│   ├── __init__.py
│   ├── main.py                        # Pipeline orchestrator (entry point)
│   ├── logger.py                      # Root logging configuration
│   ├── load_data.py                   # Data ingestion (local CSV or Kaggle)
│   ├── clean_data.py                  # Deterministic cleaning and encoding
│   ├── schema.py                      # Column contracts and domain constants
│   ├── validate.py                    # Schema, dtype, and value checks
│   ├── features.py                    # Unfitted ColumnTransformer recipe
│   ├── train.py                       # 5-fold CV training + final refit
│   ├── evaluate.py                    # Metrics + diagnostic plots
│   ├── infer.py                       # Inference on new data
│   ├── api.py                         # FastAPI serving layer
│   └── utils.py                       # File I/O helpers
│
├── data/
│   ├── raw/                           # Housing.csv (not committed)
│   ├── processed/                     # clean.csv (generated)
│   └── inference/                     # housing_inference.csv
│
├── models/
│   └── model.joblib                   # Trained pipeline artifact
│
├── reports/
│   ├── actual_vs_predicted.png
│   ├── residuals.png
│   └── predictions.csv
│
├── notebooks/
│   └── HousingPricesPrediction.ipynb  # Exploratory analysis (read-only sandbox)
│
├── logs/
│   └── pipeline.log                   # Runtime log output (not committed)
│
└── tests/
    ├── __init__.py
    ├── mock_data/
    │   └── housing_small.csv
    ├── test_api.py
    ├── test_clean_data.py
    ├── test_evaluate.py
    ├── test_features.py
    ├── test_infer.py
    ├── test_load_data.py
    ├── test_main.py
    ├── test_schema.py
    ├── test_train.py
    ├── test_utils.py
    └── test_validate.py
```

---

## Setup — Local Development

**Prerequisites:** conda, Docker Desktop

### 1. Clone the repository

```bash
git clone https://github.com/carlos13suarez/G2_MLOPS_Project.git
cd G2_MLOPS_Project
```

### 2. Create `.env` with your secrets

```bash
# .env — never commit this file
WANDB_API_KEY="paste_your_40_character_key_here"
WANDB_ENTITY="paste_your_wandb_username_or_team_here"
MODEL_SOURCE="local"
WANDB_MODEL_ALIAS="prod"
```

### 3. Install the environment from the lockfile

```bash
conda-lock install -n housing_prices_mlops conda-lock.yml
```

### 4. Activate

```bash
conda activate housing_prices_mlops
```

### 5. Add the dataset

Download `Housing.csv` from [Kaggle — yasserh/housing-prices-dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset) and place it at `data/raw/Housing.csv`.

### 6. Run the training pipeline

```bash
python -m src.main
```

This produces `models/model.joblib`, `data/processed/clean.csv`, `reports/actual_vs_predicted.png`, `reports/residuals.png`, and `reports/predictions.csv`. Metrics and artifacts are logged to W&B if `WANDB_API_KEY` is set and `run.log_to_wandb: true` in `config.yaml`.

---

## Running the API Locally

**Native (with hot reload):**

```bash
MODEL_SOURCE=local uvicorn src.api:app --reload
```

**Docker:**

```bash
docker build -t housing-api:latest .
docker run -p 8000:8000 --env-file .env housing-api:latest
```

**Health check:**

```bash
curl http://127.0.0.1:8000/health
```

**Interactive docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## API Usage

| Environment | Base URL |
|---|---|
| Local | `http://127.0.0.1:8000` |
| Live (Render) | `https://housing-price-predictor-jfz4.onrender.com` |

### GET /health

```bash
curl https://housing-price-predictor-jfz4.onrender.com/health
```

```json
{"status": "ok", "model_version": "model.joblib"}
```

Returns `503` with `{"status": "unavailable", "model_version": "none"}` if the model has not loaded.

### POST /predict

```bash
curl -X POST https://housing-price-predictor-jfz4.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [{
      "area": 7420,
      "bedrooms": 4,
      "bathrooms": 2,
      "stories": 3,
      "mainroad": "yes",
      "guestroom": "no",
      "basement": "no",
      "hotwaterheating": "no",
      "airconditioning": "yes",
      "parking": 2,
      "prefarea": "yes",
      "furnishingstatus": "furnished"
    }]
  }'
```

```json
{"predictions": [{"prediction": 6823541.25}]}
```

The endpoint accepts batches — include multiple objects in `records` to get multiple predictions in a single call. Extra fields return `422`. Missing fields return `422`.

---

## W&B Experiment Tracking

Project: [https://wandb.ai/charliesuarez10-ie/housing-price-mlops](https://wandb.ai/charliesuarez10-ie/housing-price-mlops)

Each training run logs:

- **Data:** raw row count, clean row count
- **Metrics:** CV RMSE, MAE, R², Adjusted R² (mean over 5 folds)
- **Artifacts:** processed dataset, trained model, predictions CSV
- **Plots:** actual vs. predicted, residuals panel
- **API inference logs:** buffered in `src/api.py` (batch size 50) and flushed as a W&B Table

To disable W&B (e.g. for local dev without credentials), set `run.log_to_wandb: false` in `config.yaml` or `WANDB_MODE=disabled` in the environment.

---

## CI/CD

### ci.yml — Quality Gate

Triggers on every pull request and push to `main`. Steps:

1. Checkout repository
2. Setup Miniconda
3. Install exact environment from `conda-lock.yml`
4. `pytest -q` — full test suite
5. `docker build` — validates the container builds without errors

W&B is fully disabled in CI (`WANDB_MODE=disabled`, `MODEL_SOURCE=local`). No secrets are required.

[View runs →](https://github.com/carlos13suarez/G2_MLOPS_Project/actions)

### deploy.yml — Continuous Deployment

Triggers only when a human explicitly publishes a GitHub Release from `main`. Sends a deploy hook to Render, which pulls the latest image and restarts the service. The `RENDER_DEPLOY_HOOK_URL` secret is set in GitHub repository settings — it is never embedded in code.

---

## Model Card

| Field | Detail |
|---|---|
| **Model type** | Linear Regression — scikit-learn `Pipeline` with `ColumnTransformer` |
| **Training data** | Kaggle Housing Prices dataset — 545 observations × 13 columns, no missing values |
| **Target** | `price` — house sale price (range: 1.75M–13.3M, median: 4.34M) |
| **Features** | 12 inputs: `area` (log1p + StandardScaler), `bedrooms`, `bathrooms`, `stories`, `parking` (StandardScaler); `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea` (binary 0/1); `furnishingstatus` (one-hot, drop-first) |
| **Preprocessing** | Binary encoding of yes/no columns → log1p on `area` → StandardScaler on numeric features → OneHotEncoder on `furnishingstatus`. All transforms fit on training data only (no leakage). |
| **Evaluation** | 5-fold cross-validation: R² 0.653, Adj R² 0.644, MAE 747,580, RMSE 1,039,102 |
| **Intended use** | First-pass automated valuation for residential listing agents. Always reviewed by a human before communicating to sellers. |
| **Limitations** | Trained on 545 rows from a single market. Performance may degrade on properties outside the training distribution (high-value outliers above ~10M, non-residential properties, geographies not represented in the data). Not suitable for financial instruments or legal valuations. |

---

## Changelog

### [1.0.0] — 2026-03-22

#### Added

- Full MLOps upgrade from Phase 1 notebook-converted pipeline
- `config.yaml` expanded to 60+ keys across 10 sections
- `src/logger.py` with dual `StreamHandler` + `FileHandler` output
- W&B experiment tracking in `main.py` (metrics, artifacts, plots)
- `src/api.py` — FastAPI serving layer with `/health` and `/predict` endpoints
- Pydantic strict input contract (`extra="forbid"`)
- HTTP middleware with correlation IDs and latency logging
- Async W&B inference log buffer (batch size 50)
- `Dockerfile` + `.dockerignore` (allowlist strategy)
- `conda-lock.yml` for reproducible Linux-64 environment
- `.github/workflows/ci.yml` — quality gate on all PRs
- `.github/workflows/deploy.yml` — CD triggered by GitHub Release
- Render deployment at <https://housing-price-predictor-jfz4.onrender.com>
- `tests/test_api.py` — 8 API tests including 422 contract enforcement
- `pytest.ini`, `src/__init__.py`, `tests/__init__.py`

---

## Authors

**Group 2:** Tom Biefel, Kishan Dhulashia, Álvaro Perez La Rosa, Robyn Rothlin, Carlos Suarez Álvarez, Natalia Urrea

**Course:** MLOps — IE University MsC Business Analytics & Data Science, March 2026
