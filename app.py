from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app    = FastAPI(title="Predictive Maintenance API")
model  = joblib.load("outputs/models/rf_model.pkl")
scaler = joblib.load("outputs/models/scaler.pkl")
cols   = joblib.load("outputs/models/feature_columns.pkl")

class SensorInput(BaseModel):
    air_temp        : float
    process_temp    : float
    rotational_speed: float
    torque          : float
    tool_wear       : float
    type_L          : int
    type_M          : int

@app.post("/predict")
def predict(data: SensorInput):

    # Step 1: Create raw input
    row = pd.DataFrame([{
        "Air temperature K"     : data.air_temp,
        "Process temperature K" : data.process_temp,
        "Rotational speed rpm"  : data.rotational_speed,
        "Torque Nm"             : data.torque,
        "Tool wear min"         : data.tool_wear,
        "Type_L"                : data.type_L,
        "Type_M"                : data.type_M
    }])

    # Step 2: Align with training columns (VERY IMPORTANT)
    row = row.reindex(columns=cols, fill_value=0)

    # Step 3: Scale
    row_scaled = pd.DataFrame(
        scaler.transform(row), columns=cols
    )

    # Step 4: Predict
    prob  = model.predict_proba(row_scaled)[0][1]
    label = "Machine likely to fail" if prob >= 0.5 else "Machine OK"

    return {
    "prediction": label,
    "confidence": float(round(prob * 100, 2)),
    "alert": bool(prob >= 0.5)
}