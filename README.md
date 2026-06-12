# 🏥 End-to-End FastAPI Endpoint for ML

A production-style machine learning API that predicts **insurance premium categories** (Low / Medium / High) for individuals based on their demographic and lifestyle data. Built with **FastAPI** for the backend, a **Random Forest Classifier** as the ML model, and **Streamlit** for a simple interactive frontend.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Folder Structure](#folder-structure)
- [Module Breakdown](#module-breakdown)
- [API Reference](#api-reference)
- [Feature Engineering Pipeline](#feature-engineering-pipeline)
- [Setup & Installation](#setup--installation)
- [Running the App](#running-the-app)
- [Tech Stack](#tech-stack)

---

## Overview

This project demonstrates how to wrap a trained scikit-learn model inside a FastAPI application and expose it as a REST endpoint. The key design choices include:

- **Pydantic v2 schemas** with `@computed_field` to handle all feature engineering server-side, so callers only need to send raw user inputs (age, weight, height, city, etc.)
- **Clean separation of concerns** — model loading, input validation, feature engineering, and response formatting each live in their own modules
- **Streamlit frontend** that calls the API locally, acting as a lightweight demo UI

---

## Architecture

```
User Input (Streamlit UI)
        │
        ▼
  frontend.py  ──POST /predict──►  backend.py (FastAPI)
                                        │
                                        ▼
                               schema/schema.py
                          (Pydantic validation +
                           computed feature engineering)
                                        │
                                        ▼
                              Model/predict.py
                          (loads model.pkl, runs
                           predict + predict_proba)
                                        │
                                        ▼
                           schema/response_model.py
                          (PredictionResponse output)
                                        │
                                        ▼
                    JSON Response → { predicted_category,
                                      confidence,
                                      class_probabilities }
```

---

## Folder Structure

```
End-to-End-FastAPI-Endpoint-for-ML/
│
├── backend.py                          # FastAPI app — route definitions and entrypoint
├── frontend.py                         # Streamlit UI — collects user input and calls the API
├── insurance.csv                       # Raw dataset used to train the model
├── requirements.txt                    # Python dependencies
├── Random_Forest_Classifier_Toy_Model.ipynb  # Jupyter notebook — EDA, training, and model export
│
├── Model/
│   ├── model.pkl                       # Serialized trained Random Forest model (pickle)
│   └── predict.py                      # Model loader + inference logic (predict & predict_proba)
│
├── schema/
│   ├── schema.py                       # Pydantic input schema with computed feature engineering
│   └── response_model.py               # Pydantic response model (predicted_category, confidence, class_probabilities)
│
└── config/
    └── city_tier.py                    # Static lookup lists mapping Indian cities to Tier 1 / Tier 2
```

---

## Module Breakdown

### `backend.py` — FastAPI Application

The main application file. Defines three routes:

| Route | Method | Description |
|---|---|---|
| `/` | GET | Health check — returns a welcome message |
| `/health` | GET | Returns `status: OK` and the current model version (useful for AWS/container health checks) |
| `/predict` | POST | Accepts a `Schema` body, runs inference, returns a `PredictionResponse` |

The `/predict` endpoint extracts the six model-ready features (`bmi`, `age_group`, `lifestyle_risk`, `city_tier`, `income_lpa`, `occupation`) from the validated Pydantic model and passes them to `predict_output()`.

---

### `frontend.py` — Streamlit UI

A lightweight browser-based form that:
- Collects raw user inputs: age, weight, height, income, smoker status, city, and occupation
- Sends them as a JSON `POST` to `http://localhost:8000/predict`
- Displays the predicted premium category on success, or a clear error message if the API is unreachable

> **Note:** The frontend sends raw inputs — all feature engineering (BMI, age group, lifestyle risk, city tier) happens inside the API via Pydantic's `@computed_field`.

---

### `schema/schema.py` — Input Schema + Feature Engineering

The `Schema` class (Pydantic `BaseModel`) accepts raw user inputs and computes derived features automatically before the model ever sees the data:

| Field | Type | Description |
|---|---|---|
| `age` | `int` | Age of the user (1–119) |
| `weight` | `float` | Weight in kilograms |
| `height` | `float` | Height in metres |
| `income_lpa` | `int` | Annual income in Lakhs Per Annum |
| `smoker` | `bool` | Whether the user is a smoker |
| `city` | `str` | City of residence (auto-normalized to Title Case) |
| `occupation` | `Literal[...]` | One of 7 valid occupation categories |

**Computed fields (derived automatically):**

| Computed Field | Logic |
|---|---|
| `bmi` | `round(weight / height², 2)` |
| `age_group` | `young` (<25) / `adult` (<45) / `middle_aged` (<60) / `senior` (60+) |
| `lifestyle_risk` | `high` (smoker + BMI > 30) / `medium` (smoker or BMI > 27) / `low` (otherwise) |
| `city_tier` | `1` (metro) / `2` (major city) / `3` (other) — looked up from `config/city_tier.py` |

A `@field_validator` on `city` strips whitespace and normalizes casing before tier lookup.

---

### `schema/response_model.py` — Response Schema

The `PredictionResponse` model structures the API output:

| Field | Type | Description |
|---|---|---|
| `predicted_category` | `str` | The predicted premium tier (`Low`, `Medium`, or `High`) |
| `confidence` | `float` | The model's max class probability (0–1) |
| `class_probabilities` | `Dict[str, float]` | Full probability distribution across all classes |

---

### `Model/predict.py` — Inference Module

Loads `model.pkl` at import time (once, on startup) using `pickle`. Exposes:

- `model` — the loaded scikit-learn `RandomForestClassifier`
- `MODEL_VERSION = '1.0.0'` — hardcoded version string (in production, this would come from an MLflow model registry)
- `predict_output(user_input: dict)` — builds a single-row DataFrame from the 6 engineered features, calls `model.predict()` and `model.predict_proba()`, and returns the structured prediction dict

---

### `Model/model.pkl` — Trained Model

The serialized Random Forest Classifier trained on `insurance.csv`. The full training process, including EDA, feature engineering decisions, hyperparameter choices, and model export, is documented in `Random_Forest_Classifier_Toy_Model.ipynb`.

---

### `config/city_tier.py` — City Tier Configuration

Two Python lists used by `schema.py` to assign a city tier (1, 2, or 3):

- `tier_1_cities`: The 7 major metros — Mumbai, Delhi, Bangalore, Chennai, Kolkata, Hyderabad, Pune
- `tier_2_cities`: ~48 major Indian cities (Jaipur, Lucknow, Indore, Noida, etc.)
- Any city not in either list defaults to **Tier 3**

---

### `Random_Forest_Classifier_Toy_Model.ipynb` — Training Notebook

The Jupyter notebook covers:
- Exploratory data analysis on `insurance.csv`
- Feature engineering (the same transformations that are later replicated in `schema.py`)
- Model training and evaluation
- Exporting the final model to `Model/model.pkl`

---

## API Reference

### `POST /predict`

**Request Body:**
```json
{
  "age": 35,
  "weight": 80.0,
  "height": 1.75,
  "income_lpa": 12,
  "smoker": false,
  "city": "Bangalore",
  "occupation": "private_job"
}
```

**Occupation options:** `retired`, `freelancer`, `student`, `government_job`, `business_owner`, `unemployed`, `private_job`

**Response (200 OK):**
```json
{
  "predicted_category": "Medium",
  "confidence": 0.7231,
  "class_probabilities": {
    "Low": 0.1543,
    "Medium": 0.7231,
    "High": 0.1226
  }
}
```

---

## Feature Engineering Pipeline

The API applies these transformations to raw inputs before passing features to the model:

```
age          ──►  age_group     (young / adult / middle_aged / senior)
weight + height ──►  bmi        (weight / height²)
smoker + bmi  ──►  lifestyle_risk  (low / medium / high)
city          ──►  city_tier    (1 / 2 / 3  via lookup)
income_lpa    ──►  passed through directly
occupation    ──►  passed through directly
```

The 6 features fed to the model: `bmi`, `age_group`, `lifestyle_risk`, `city_tier`, `income_lpa`, `occupation`

---

## Setup & Installation

**Prerequisites:** Python 3.9+

```bash
# Clone the repository
git clone https://github.com/HarshRaj4343/End-to-End-FastAPI-Endpoint-for-ML.git
cd End-to-End-FastAPI-Endpoint-for-ML

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Running the App

**Start the FastAPI backend:**
```bash
uvicorn backend:app --reload
```
The API will be available at `http://localhost:8000`

**Interactive API docs (Swagger UI):** `http://localhost:8000/docs`

**Start the Streamlit frontend** (in a separate terminal):
```bash
streamlit run frontend.py
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| Data Validation | Pydantic v2 |
| ML Model | scikit-learn `RandomForestClassifier` |
| Frontend | Streamlit |
| Data Processing | pandas |
| Model Serialization | pickle |
| Server | Uvicorn |
