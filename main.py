from src.preprocess import load_and_preprocess
from src.train import train_logistic, train_random_forest
from src.evaluate import evaluate

# update unpack
X_train, X_test, y_train, y_test, scaler, feat_cols = load_and_preprocess()

# Logistic Regression
lr = train_logistic(X_train, y_train)
evaluate("Logistic Regression", y_test, lr.predict(X_test))

# Random Forest
rf = train_random_forest(X_train, y_train)
evaluate("Random Forest", y_test, rf.predict(X_test))

from src.train import train_logistic, train_random_forest
from src.train import train_neural_network, predict_neural_network

# Neural Network
nn_model = train_neural_network(X_train, y_train)
y_pred_nn = predict_neural_network(nn_model, X_test)
evaluate("Neural Network", y_test, y_pred_nn)

from src.evaluate import evaluate, plot_feature_importance, plot_model_comparison

plot_feature_importance(rf, X_test.columns)
plot_model_comparison()

from src.evaluate import predict_single



# Sample input — high torque + high wear = likely failure
test_input = {
    "Air temperature K"     : 302.0,
    "Process temperature K" : 310.0,
    "Rotational speed rpm"  : 1350,
    "Torque Nm"             : 60.0,
    "Tool wear min"         : 220,
    "Type_L"                : 0,
    "Type_M"                : 1
}

predict_single(test_input, scaler,
               model_type="rf", rf_model=rf,
               feature_columns=feat_cols)


from src.train import train_xgboost

xgb = train_xgboost(X_train, y_train)
evaluate("XGBoost", y_test, xgb.predict(X_test))

import torch
import numpy as np
from src.evaluate import plot_roc_curves

# Get probability scores (not just 0/1 predictions)
lr_probs  = lr.predict_proba(X_test)[:, 1]
rf_probs  = rf.predict_proba(X_test)[:, 1]
xgb_probs = xgb.predict_proba(X_test)[:, 1]

# Neural Network probs
nn_model.eval()
with torch.no_grad():
    x_t      = torch.tensor(X_test.values, dtype=torch.float32)
    nn_probs = nn_model(x_t).squeeze().numpy()

models_probs = {
    "Logistic Regression" : lr_probs,
    "Random Forest"       : rf_probs,
    "Neural Network"      : nn_probs,
    "XGBoost"             : xgb_probs
}

plot_roc_curves(models_probs, y_test)

import joblib
import os

os.makedirs("outputs/models", exist_ok=True)
joblib.dump(rf, "outputs/models/rf_model.pkl")
joblib.dump(scaler, "outputs/models/scaler.pkl")
joblib.dump(feat_cols, "outputs/models/feature_columns.pkl")
print("Models saved.")

from src.evaluate import plot_shap_summary

plot_shap_summary(rf, X_test, feat_cols)

from src.evaluate import explain_single_prediction

# Explain a high-risk machine (pick a failure case from test set)
failure_indices = y_test[y_test == 1].index
sample_idx      = failure_indices[0]

# Get the row from X_test by index position
row_pos = X_test.index.get_loc(sample_idx)
row     = X_test.iloc[[row_pos]]

explain_single_prediction(rf, row, feat_cols, index=row_pos)