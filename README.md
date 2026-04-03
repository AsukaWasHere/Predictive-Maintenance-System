# 🔧 Predictive Maintenance System

## Problem Statement
Industrial machines fail unexpectedly, causing costly downtime. This project uses the **AI4I 2020 dataset** to predict machine failures before they occur, identify root causes, and serve predictions via a containerized REST API.

---

## Dataset Description
The **AI4I 2020 Predictive Maintenance Dataset** contains:
- **Sensor readings**: Air temperature, process temperature, rotational speed, torque, tool wear
- **Operational data**: Machine type (L/M/H), product quality variant
- **Failure labels**: Binary machine failure + 5 failure mode sub-labels (TWF, HDF, PWF, OSF, RNF)
- **Rows**: 10,000 | **Features**: 7 (after cleaning)

---

## Objectives
1. **Predict failures** — Binary classification: Will the machine fail? (0 = No, 1 = Yes)
2. **Analyze feature importance** — Which sensor/operational factors drive failures?
3. **Explain predictions** — Per-prediction explainability via SHAP
4. **Serve predictions** — REST API via FastAPI, containerized with Docker

---

## Models

| Model | Accuracy | Precision | Recall | Missed Failures | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 82.9% | 14.5% | 82.4% | 12 | 0.908 |
| Random Forest | **96.7%** | **51.0%** | 76.5% | 16 | **0.976** |
| Neural Network (PyTorch) | 92.2% | 29.3% | **92.6%** | **5** | **0.976** |
| XGBoost | 89.1% | 22.7% | **92.6%** | **5** | **0.976** |

**Best for precision → Random Forest** (fewest false alarms, deployed in API)
**Best for recall → Neural Network / XGBoost** (fewest missed failures)

---

## Key Findings

- `Torque [Nm]` and `Rotational speed [rpm]` drive **61%** of failure prediction
- `Tool wear [min]` ranks 3rd — time-based degradation matters
- Machine type (`L/M/H`) has almost **no predictive value**
- SMOTE was critical — raw 3% class imbalance would fool any naive model
- RF, XGBoost, and NN are statistically equivalent on AUC (0.976)
- SHAP confirms: high torque + high rotational speed = the dangerous combination

---

## Pipeline
```
Raw CSV
  ↓
Drop identifiers + leakage columns
  ↓
Encode categorical (Type) + StandardScaler
  ↓
Train/Test Split (80/20, stratified)
  ↓
SMOTE (balance 3% minority class)
  ↓
Train 4 models
  ↓
Evaluate (Accuracy, Precision, Recall, AUC)
  ↓
SHAP explainability (summary + waterfall)
  ↓
FastAPI endpoint → Docker container
```

---

## Folder Structure
```
predictive_maintenance/
├── data/
│   └── ai4i2020.csv
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── outputs/
│   ├── models/
│   │   ├── rf_model.pkl
│   │   ├── scaler.pkl
│   │   └── feature_columns.pkl
│   └── plots/
│       ├── feature_importance.png
│       ├── model_comparison.png
│       ├── roc_curves.png
│       ├── shap_summary.png
│       └── shap_waterfall_*.png
├── app.py
├── main.py
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

---

## Key Concepts
- **Classification** — Predict binary outcome (fail / no fail)
- **Imbalanced data** — Failures are rare (~3%); handled with SMOTE
- **Feature importance** — Torque + Rotational speed = 61% of signal
- **ROC-AUC** — Threshold-independent model comparison
- **SHAP** — Explains why each individual prediction was made
- **REST API** — FastAPI wraps model into a live endpoint
- **Docker** — Containerized for portable deployment

---

## SHAP Explainability

### Global — Summary Plot
Shows which features matter most across all predictions.
- High torque (red, right) → strongest push toward failure
- Low tool wear (blue, left) → pushes toward safe

### Local — Waterfall Plot
Explains a single machine's prediction step by step.
- Red bars → push toward failure
- Blue bars → push toward safe
- Starts from base rate `E[f(X)]`, ends at final prediction `f(x)`

---

## Sample Prediction

**Input** — High torque + High tool wear:
```json
{
  "air_temp": 302.0,
  "process_temp": 310.0,
  "rotational_speed": 1350,
  "torque": 60.0,
  "tool_wear": 220,
  "type_L": 0,
  "type_M": 1
}
```

**Output:**
```json
{
  "prediction": "Machine likely to fail",
  "confidence": 86.0,
  "alert": true
}
```

---

## Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Train models + generate plots
python main.py

# Start API
uvicorn app:app --reload

# Visit interactive docs
http://127.0.0.1:8000/docs
```

## Run with Docker
```bash
# Build image
docker build -t predictive-maintenance .

# Run container
docker run -p 8000:8000 predictive-maintenance

# Visit interactive docs
http://127.0.0.1:8000/docs
```

---

## Requirements
```
pandas
numpy
scikit-learn
imbalanced-learn
torch
xgboost
shap
matplotlib
fastapi
uvicorn
joblib
```

---

## Future Work
- [ ] Threshold tuning per business cost (miss vs false alarm)
- [ ] Add SHAP explanations to API response
- [ ] Push Docker image to Docker Hub
- [ ] Add real-time sensor streaming simulation
- [ ] Experiment with LightGBM
- [ ] Add CI/CD pipeline (GitHub Actions)