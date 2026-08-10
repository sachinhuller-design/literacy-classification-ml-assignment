import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="Literacy Classification", page_icon="📚", layout="wide")
st.title("📚 Literacy Level Classification")
st.write("Compare five classification models and predict literacy level from student/education features.")

MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}

@st.cache_resource
def load_model(path):
    return joblib.load(path)

metrics_path = os.path.join(MODEL_DIR, "metrics.csv")
if os.path.exists(metrics_path):
    metrics = pd.read_csv(metrics_path)
    st.subheader("Model comparison")
    st.dataframe(metrics.style.format({c: "{:.4f}" for c in metrics.columns if c != "Model"}), use_container_width=True)

selected = st.selectbox("Select model", list(MODEL_FILES.keys()))
uploaded = st.file_uploader("Upload test CSV", type=["csv"])

default_path = "test_data.csv"
if uploaded is not None:
    data = pd.read_csv(uploaded)
else:
    data = pd.read_csv(default_path)

target = "Literacy_Level"
has_target = target in data.columns
X = data.drop(columns=[target]) if has_target else data.copy()

model = load_model(os.path.join(MODEL_DIR, MODEL_FILES[selected]))
pred = model.predict(X)

st.subheader("Predictions")
out = X.copy()
out["Predicted_Literacy_Level"] = pred
st.dataframe(out.head(20), use_container_width=True)

if has_target:
    y_true = data[target].astype(str)
    proba = model.predict_proba(X)

    st.subheader("Evaluation on uploaded test data")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    values = [
        accuracy_score(y_true, pred),
        roc_auc_score(y_true, proba, multi_class="ovr", labels=model.classes_),
        precision_score(y_true, pred, average="weighted", zero_division=0),
        recall_score(y_true, pred, average="weighted", zero_division=0),
        f1_score(y_true, pred, average="weighted", zero_division=0),
        matthews_corrcoef(y_true, pred)
    ]
    labels = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    for c, label, value in zip([col1,col2,col3,col4,col5,col6], labels, values):
        c.metric(label, f"{value:.4f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, pred, labels=model.classes_)
    st.dataframe(pd.DataFrame(cm, index=model.classes_, columns=model.classes_))

    st.subheader("Classification Report")
    report = classification_report(y_true, pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4))
else:
    st.info("No Literacy_Level column was supplied, so only predictions are shown. Upload a labeled test CSV to display evaluation metrics.")
