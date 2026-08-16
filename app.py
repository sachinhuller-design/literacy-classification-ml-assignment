import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Literacy Classification",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS - DECORATION
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef7 100%);
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        color: #1f3c88;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #5c677d;
        margin-bottom: 30px;
    }

    /* Section headers */
    .section-title {
        font-size: 26px;
        font-weight: 700;
        color: #1f3c88;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    /* Cards */
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 5px solid #1f77b4;
        margin-bottom: 15px;
    }

    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #172554 0%, #1e3a8a 100%);
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    /* Dataframes */
    .stDataFrame {
        border-radius: 10px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📚 Literacy Level Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Student Literacy Prediction & Model Comparison'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-card">
<b>📌 About this application</b><br>
This application compares five machine learning classification models
and predicts the literacy level of students using educational and
student-related features.
</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_DIR = "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(path):
    return joblib.load(path)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## ⚙️ Controls")

selected = st.sidebar.selectbox(
    "🤖 Select Prediction Model",
    list(MODEL_FILES.keys())
)

st.sidebar.markdown("---")

uploaded = st.sidebar.file_uploader(
    "📂 Upload Test CSV",
    type=["csv"]
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
### 📊 Available Models

- Logistic Regression
- Decision Tree
- kNN
- Naive Bayes
- Random Forest
""")

st.sidebar.markdown("---")

st.sidebar.info(
    "💡 Upload a labeled CSV containing `Literacy_Level` "
    "to calculate evaluation metrics."
)


# ============================================================
# MODEL COMPARISON
# ============================================================

metrics_path = os.path.join(MODEL_DIR, "metrics.csv")

if os.path.exists(metrics_path):

    metrics = pd.read_csv(metrics_path)

    st.markdown(
        '<div class="section-title">🏆 Model Performance Comparison</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Find best model
    # --------------------------------------------------------

    if "Accuracy" in metrics.columns:

        best_idx = metrics["Accuracy"].idxmax()
        best_model = metrics.loc[best_idx, "Model"]
        best_accuracy = metrics.loc[best_idx, "Accuracy"]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "🏆 Best Model",
                str(best_model)
            )

        with col2:
            st.metric(
                "🎯 Best Accuracy",
                f"{best_accuracy:.4f}"
            )

        with col3:
            st.metric(
                "🤖 Models Compared",
                len(metrics)
            )

    # --------------------------------------------------------
    # Metrics Table
    # --------------------------------------------------------

    st.subheader("📋 Detailed Model Metrics")

    numeric_cols = [
        c for c in metrics.columns
        if c != "Model" and pd.api.types.is_numeric_dtype(metrics[c])
    ]

    format_dict = {
        c: "{:.4f}" for c in numeric_cols
    }

    st.dataframe(
        metrics.style.format(format_dict),
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # PLOTS
    # ========================================================

    st.markdown(
        '<div class="section-title">📊 Visual Model Comparison</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Plot 1 - Accuracy
    # --------------------------------------------------------

    if "Accuracy" in metrics.columns:

        st.subheader("🎯 Accuracy Comparison")

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.barplot(
            data=metrics,
            x="Model",
            y="Accuracy",
            ax=ax
        )

        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Model")
        ax.set_title("Accuracy of Different Classification Models")

        plt.xticks(rotation=25)

        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%.3f",
                padding=3
            )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    # --------------------------------------------------------
    # Plot 2 - Precision Recall F1
    # --------------------------------------------------------

    available_prf = [
        c for c in ["Precision", "Recall", "F1"]
        if c in metrics.columns
    ]

    if len(available_prf) > 0:

        st.subheader("📈 Precision, Recall & F1 Comparison")

        melted = metrics.melt(
            id_vars="Model",
            value_vars=available_prf,
            var_name="Metric",
            value_name="Score"
        )

        fig, ax = plt.subplots(figsize=(11, 6))

        sns.barplot(
            data=melted,
            x="Model",
            y="Score",
            hue="Metric",
            ax=ax
        )

        ax.set_ylim(0, 1.05)
        ax.set_title(
            "Precision, Recall and F1 Score Comparison"
        )
        ax.set_ylabel("Score")
        ax.set_xlabel("Model")

        plt.xticks(rotation=25)
        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    # --------------------------------------------------------
    # Plot 3 - AUC
    # --------------------------------------------------------

    if "AUC" in metrics.columns:

        st.subheader("📈 ROC-AUC Comparison")

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.barplot(
            data=metrics,
            x="Model",
            y="AUC",
            ax=ax
        )

        ax.set_ylim(0, 1.05)
        ax.set_ylabel("ROC-AUC")
        ax.set_xlabel("Model")
        ax.set_title("ROC-AUC Score Comparison")

        plt.xticks(rotation=25)

        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%.3f",
                padding=3
            )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    # --------------------------------------------------------
    # Plot 4 - MCC
    # --------------------------------------------------------

    if "MCC" in metrics.columns:

        st.subheader("🎯 Matthews Correlation Coefficient")

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.barplot(
            data=metrics,
            x="Model",
            y="MCC",
            ax=ax
        )

        ax.set_ylabel("MCC")
        ax.set_xlabel("Model")
        ax.set_title(
            "Matthews Correlation Coefficient Comparison"
        )

        plt.xticks(rotation=25)

        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%.3f",
                padding=3
            )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


else:

    st.warning(
        "⚠️ metrics.csv was not found in the model folder. "
        "Model comparison charts cannot be displayed."
    )


# ============================================================
# LOAD DATA
# ============================================================

target = "Literacy_Level"

default_path = "test_data.csv"

if uploaded is not None:

    data = pd.read_csv(uploaded)

    st.success(
        f"✅ Uploaded dataset loaded successfully: "
        f"{data.shape[0]} rows × {data.shape[1]} columns"
    )

else:

    if os.path.exists(default_path):

        data = pd.read_csv(default_path)

        st.info(
            f"📁 Using default dataset: {default_path}"
        )

    else:

        st.error(
            "❌ test_data.csv was not found. "
            "Please upload a test CSV file."
        )

        st.stop()


# ============================================================
# DATA INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">📄 Dataset Information</div>',
    unsafe_allow_html=True
)

info1, info2, info3 = st.columns(3)

with info1:
    st.metric(
        "📌 Number of Rows",
        data.shape[0]
    )

with info2:
    st.metric(
        "📊 Number of Features",
        data.shape[1] - (1 if target in data.columns else 0)
    )

with info3:
    st.metric(
        "🎯 Target Available",
        "Yes" if target in data.columns else "No"
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

has_target = target in data.columns

X = (
    data.drop(columns=[target])
    if has_target
    else data.copy()
)


# ============================================================
# PREDICTION
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    MODEL_FILES[selected]
)

model = load_model(model_path)

pred = model.predict(X)


# ============================================================
# PREDICTIONS
# ============================================================

st.markdown(
    '<div class="section-title">🔮 Literacy Level Predictions</div>',
    unsafe_allow_html=True
)

out = X.copy()

out["Predicted_Literacy_Level"] = pred

st.dataframe(
    out,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# EVALUATION
# ============================================================

if has_target:

    y_true = data[target].astype(str)

    # Make prediction labels strings for safe comparison
    pred = pd.Series(pred).astype(str).values

    proba = model.predict_proba(X)

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(y_true, pred)

    auc = roc_auc_score(
        y_true,
        proba,
        multi_class="ovr",
        labels=model.classes_
    )

    precision = precision_score(
        y_true,
        pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        pred,
        average="weighted",
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_true,
        pred
    )

    # ========================================================
    # EVALUATION METRICS
    # ========================================================

    st.markdown(
        '<div class="section-title">📊 Evaluation on Test Data</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    col1.metric("🎯 Accuracy", f"{accuracy:.4f}")
    col2.metric("📈 AUC", f"{auc:.4f}")
    col3.metric("🔵 Precision", f"{precision:.4f}")

    col4.metric("🟢 Recall", f"{recall:.4f}")
    col5.metric("⭐ F1 Score", f"{f1:.4f}")
    col6.metric("🎯 MCC", f"{mcc:.4f}")


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.markdown(
        '<div class="section-title">🔥 Confusion Matrix</div>',
        unsafe_allow_html=True
    )

    cm = confusion_matrix(
        y_true,
        pred,
        labels=model.classes_
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_,
        ax=ax
    )

    ax.set_xlabel("Predicted Literacy Level")
    ax.set_ylabel("Actual Literacy Level")
    ax.set_title(
        f"Confusion Matrix - {selected}"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    st.markdown(
        '<div class="section-title">📋 Classification Report</div>',
        unsafe_allow_html=True
    )

    report = classification_report(
        y_true,
        pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).T.round(4)

    st.dataframe(
        report_df,
        use_container_width=True
    )


    # ========================================================
    # ACTUAL VS PREDICTED
    # ========================================================

    st.markdown(
        '<div class="section-title">🔍 Actual vs Predicted Literacy Level</div>',
        unsafe_allow_html=True
    )

    comparison = pd.DataFrame({
        "Actual Literacy Level": y_true.values,
        "Predicted Literacy Level": pred
    })

    comparison["Prediction"] = comparison.apply(
        lambda row:
        "✅ Correct"
        if row["Actual Literacy Level"]
        == row["Predicted Literacy Level"]
        else "❌ Incorrect",
        axis=1
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )


else:

    st.info(
        "ℹ️ No Literacy_Level column was supplied, so only "
        "predictions are shown. Upload a labeled test CSV to "
        "display evaluation metrics."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
<div style="text-align:center; padding:20px; color:#6b7280;">
    <b>📚 Literacy Level Classification System</b><br>
    Machine Learning Classification Project<br>
    Logistic Regression • Decision Tree • kNN • Naive Bayes • Random Forest
</div>
""", unsafe_allow_html=True)
