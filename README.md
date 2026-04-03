# 🔧 Predictive Maintenance System

## Problem Statement
Industrial machines fail unexpectedly, causing costly downtime. This project uses the **AI4I 2020 dataset** to predict machine failures before they occur and identify the root causes.

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

---

## Models

| Model | Accuracy | Precision | Recall | Missed Failures | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 82.9% | 14.5% | 82.4% | 12 | 0.908 |
| Random Forest | **96.7%** | **51.0%** | 76.5% | 16 | **0.976** |
| Neural Network (PyTorch) | 92.2% | 29.3% | **92.6%** | **5** | **0.976** |
| XGBoost | 89.1% | 22.7% | **92.6%** | **5** | **0.976** |

**Best for precision → Random Forest** (fewest false alarms)
**Best for recall → Neural Network / XGBoost** (fewest missed failures)

---

## Key Findings

- `Torque [Nm]` and `Rotational speed [rpm]` drive **61%** of failure prediction
- `Tool wear [min]` ranks 3rd — time-based degradation matters
- Machine type (`L/M/H`) has almost **no predictive value**
- SMOTE was critical — raw 3% class imbalance would fool any naive model
- RF, XGBoost, NN are statistically equivalent on AUC (0.976)

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
Feature Importance
  ↓
FastAPI endpoint → live predictions
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
│       └── roc_curves.png
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

---

## Key Concepts
- **Classification** — Predict binary outcome (fail / no fail)
- **Imbalanced data** — Failures are rare (~3%); handled with SMOTE
- **Feature importance** — Torque + Rotational speed = 61% of signal
- **ROC-AUC** — Threshold-independent model comparison
- **REST API** — FastAPI wraps model into a live endpoint

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

## Run the API
```bash
uvicorn app:app --reload
# Visit http://127.0.0.1:8000/docs
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
matplotlib
fastapi
uvicorn
joblib
```

---

## Future Work
- [ ] Tune Neural Network (learning rate scheduler, more epochs)
- [ ] Threshold tuning per business cost (miss vs false alarm)
- [ ] Add SHAP values for per-prediction explainability
- [ ] Dockerize the FastAPI app
- [ ] Add real-time sensor streaming simulation