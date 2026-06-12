# 🏥 Insurance Premium Category Predictor

An end-to-end ML serving pipeline built with **FastAPI** (backend) and **Streamlit** (frontend). Submit personal health and lifestyle details and instantly receive a predicted **insurance premium category** powered by a trained Random Forest classifier.

---

## 🗂️ Project Structure

```
.
├── backend.py                          # FastAPI app — model loading, Pydantic schema, /predict endpoint
├── frontend.py                         # Streamlit UI — form inputs, API call, result display
├── model.pkl                           # Serialised Random Forest model
├── insurance.csv                       # Training dataset
├── Random_Forest_Classifier_Toy_Model.ipynb  # Model training notebook
└── requirements.txt                    # Python dependencies
```

---

## ⚙️ How It Works

### Input Features (user-supplied)

| Field | Type | Description |
|---|---|---|
| `age` | `int` | Age in years (1–119) |
| `height` | `float` | Height in metres |
| `weight` | `float` | Weight in kilograms |
| `income_lpa` | `int` | Annual income in lakhs |
| `smoker` | `bool` | Smoking status |
| `city` | `str` | City of residence |
| `occupation` | `Literal` | One of 7 occupation categories |

### Computed Features (auto-derived by the API)

| Field | Logic |
|---|---|
| `bmi` | `weight / height²` |
| `age_group` | `young / adult / middle_aged / senior` |
| `lifestyle_risk` | `high / medium / low` (based on BMI + smoking) |
| `city_tier` | `1 / 2 / 3` (Tier-1, Tier-2, or other city) |

The model receives only the four computed features plus `income_lpa` and `occupation`, keeping the prediction surface clean and the raw inputs private.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/HarshRaj4343/End-to-End-FastAPI-Endpoint-for-ML.git
cd End-to-End-FastAPI-Endpoint-for-ML
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the FastAPI backend

```bash
uvicorn backend:app --reload
```

The API will be available at `http://localhost:8000`.  
Interactive docs (Swagger UI): `http://localhost:8000/docs`

### 4. Start the Streamlit frontend

Open a second terminal:

```bash
streamlit run frontend.py
```

The UI will open at `http://localhost:8501`.

---

## 📡 API Reference

### `POST /predict`

**Request body (JSON):**

```json
{
  "age": 32,
  "height": 1.75,
  "weight": 78.0,
  "income_lpa": 12,
  "smoker": false,
  "city": "Mumbai",
  "occupation": "private_job"
}
```

**Response:**

```json
{
  "predicted_category": "Silver"
}
```

---

## 🛠️ Tech Stack

- **FastAPI** — high-performance REST API with automatic OpenAPI docs
- **Pydantic v2** — request validation and computed fields (`@computed_field`)
- **scikit-learn** — Random Forest classifier, serialised with `pickle`
- **pandas** — DataFrame construction for model inference
- **Streamlit** — lightweight interactive frontend

---

## 📓 Model Training

The training notebook (`Random_Forest_Classifier_Toy_Model.ipynb`) covers:

- Feature engineering (BMI, age group, lifestyle risk, city tier)
- Random Forest training on `insurance.csv`
- Model serialisation to `model.pkl`

---

## 👤 Author

**Harsh Raj** — [GitHub](https://github.com/HarshRaj4343) · [Portfolio](https://harshraj4343.github.io)  
B.Tech AI & Robotics, IIT Mandi (2025–2029)
