from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, classification_report, confusion_matrix
)

def evaluate(model_name, y_test, y_pred):
    print(f"\n===== {model_name} =====")
    print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
    print("Precision:", round(precision_score(y_test, y_pred), 4))
    print("Recall   :", round(recall_score(y_test, y_pred), 4))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_feature_importance(rf_model, feature_names):
    importance = pd.Series(
        rf_model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    importance.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Feature Importance — Random Forest")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()

    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig("outputs/plots/feature_importance.png")
    plt.show()
    print("Saved → outputs/plots/feature_importance.png")

def plot_model_comparison():
    models     = ["Logistic Reg", "Random Forest", "Neural Network"]
    precision  = [0.1455, 0.5098, 0.2681]
    recall     = [0.8235, 0.7647, 0.9265]

    x = range(len(models))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - 0.2 for i in x], precision, width=0.4, label="Precision", color="salmon")
    ax.bar([i + 0.2 for i in x], recall,    width=0.4, label="Recall",    color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Comparison — Precision vs Recall")
    ax.legend()
    plt.tight_layout()

    plt.savefig("outputs/plots/model_comparison.png")
    plt.show()
    print("Saved → outputs/plots/model_comparison.png")

import numpy as np

def predict_single(input_dict, scaler, model_type="rf",
                   rf_model=None, nn_model=None, 
                   xgb_model=None, feature_columns=None):

    row = pd.DataFrame([input_dict])[feature_columns]

    row_scaled = pd.DataFrame(
        scaler.transform(row), columns=feature_columns
    )

    if model_type == "rf":
        prob = rf_model.predict_proba(row_scaled)[0][1]
    elif model_type == "xgb":
        prob = xgb_model.predict_proba(row_scaled)[0][1]
    elif model_type == "nn":
        import torch
        x = torch.tensor(row_scaled.values, dtype=torch.float32)
        nn_model.eval()
        with torch.no_grad():
            prob = nn_model(x).item()

    label = "⚠️  Machine likely to fail" if prob >= 0.5 else "✅  Machine OK"
    print(f"\n{label}  (Confidence: {prob*100:.1f}%)")

from sklearn.metrics import roc_curve, auc

def plot_roc_curves(models_probs, y_test):
    """
    models_probs: dict of {model_name: y_prob_array}
    """
    plt.figure(figsize=(8, 6))

    for name, y_prob in models_probs.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    # Random baseline
    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.500)")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — All Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("outputs/plots/roc_curves.png")
    plt.show()
    print("Saved → outputs/plots/roc_curves.png")

import shap

def plot_shap_summary(model, X_test, feature_names):
    import shap
    import pandas as pd
    import matplotlib.pyplot as plt

    # Ensure DataFrame
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=feature_names)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    print("Generating SHAP summary plot...")

    # 🔥 HANDLE ALL CASES
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]

    elif len(shap_values.shape) == 3:
        shap_values_to_plot = shap_values[:, :, 1]

    else:
        shap_values_to_plot = shap_values

    # Final check
    assert shap_values_to_plot.shape == X_test.shape, \
        f"Shape mismatch: {shap_values_to_plot.shape} vs {X_test.shape}"

    shap.summary_plot(
        shap_values_to_plot,
        X_test,
        feature_names=feature_names,
        show=False
    )

    plt.tight_layout()
    plt.savefig("outputs/plots/shap_summary.png")
    plt.close()

    print("Saved → outputs/plots/shap_summary.png")

def explain_single_prediction(model, row, feature_names, index=0):
    import shap
    import pandas as pd

    if not isinstance(row, pd.DataFrame):
        row = pd.DataFrame(row, columns=feature_names)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row)

    # Handle all SHAP version formats
    if isinstance(shap_values, list):
        sv   = shap_values[1][0]
        base = explainer.expected_value[1]

    elif len(shap_values.shape) == 3:
        sv   = shap_values[0, :, 1]       # (samples, features, classes)
        base = explainer.expected_value[1]

    else:
        sv   = shap_values[0]             # single class output
        base = explainer.expected_value

    print(f"\nExplaining prediction for sample {index}:")
    shap.waterfall_plot(
        shap.Explanation(
            values        = sv,
            base_values   = base,
            data          = row.iloc[0].values,
            feature_names = feature_names
        ),
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"outputs/plots/shap_waterfall_{index}.png")
    plt.show()
    print(f"Saved → outputs/plots/shap_waterfall_{index}.png")