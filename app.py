import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Literacy Classification",
    page_icon="📚",
    layout="wide"
)

# ============================================================
# LIGHT PROFESSIONAL UI
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #f8fafc;
}

/* =========================
   BLACK TITLES
   ========================= */

.main-title {
    font-size: 40px;
    font-weight: 800;
    text-align: center;
    color: #000000 !important;
    margin-top: 10px;
    margin-bottom: 6px;
    animation: fadeDown 0.9s ease-out;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    color: #475569 !important;
    margin-bottom: 25px;
    animation: fadeUp 1s ease-out;
}

.section-title {
    font-size: 25px;
    font-weight: 750;
    color: #000000 !important;
    margin-top: 28px;
    margin-bottom: 15px;
    padding-bottom: 8px;
    border-bottom: 2px solid #e2e8f0;
    animation: fadeUp 0.7s ease-out;
}

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
    color: #000000 !important;
}

/* =========================
   ANIMATIONS
   ========================= */

@keyframes fadeDown {
    from {
        opacity: 0;
        transform: translateY(-18px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(15px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes softPulse {
    0%, 100% {
        transform: scale(1);
    }
    50% {
        transform: scale(1.015);
    }
}

/* =========================
   INFO CARD
   ========================= */

.info-card {
    background-color: white;
    padding: 18px 22px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #cbd5e1;
    box-shadow: 0 2px 8px rgba(15,23,42,0.05);
    color: #334155;
    animation: fadeUp 0.8s ease-out;
}

/* =========================
   SIDEBAR
   ========================= */

section[data-testid="stSidebar"] {
    background-color: #f8fafc;
    border-right: 1px solid #e2e8f0;
}

section[data-testid="stSidebar"] * {
    color: #000000 !important;
}

/* =========================
   METRIC CARDS
   ========================= */

div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px;
    box-shadow: 0 2px 8px rgba(15,23,42,0.04);
    transition: all 0.25s ease;
    animation: fadeUp 0.7s ease-out;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 7px 18px rgba(15,23,42,0.10);
    border-color: #cbd5e1;
}

div[data-testid="stMetricLabel"] {
    color: #64748b !important;
}

div[data-testid="stMetricValue"] {
    color: #000000 !important;
}

/* =========================
   DATA TABLES
   ========================= */

[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    overflow: hidden;
    background-color: white;
    animation: fadeUp 0.6s ease-out;
}

/* =========================
   FILE UPLOADER
   ========================= */

[data-testid="stFileUploader"] {
    background-color: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
}

/* =========================
   SELECT BOX
   ========================= */

div[data-baseweb="select"] > div {
    background-color: white;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
}

/* =========================
   BUTTON
   ========================= */

.stButton > button {
    background-color: white;
    color: #000000;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.25s ease;
}

.stButton > button:hover {
    border-color: #94a3b8;
    transform: translateY(-2px);
    box-shadow: 0 4px 10px rgba(15,23,42,0.08);
}

/* =========================
   ALERTS
   ========================= */

div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* =========================
   PLOT ANIMATION
   ========================= */

[data-testid="stImage"],
[data-testid="stPyplotGraph"] {
    animation: fadeUp 0.9s ease-out;
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

# Small visual loading animation
progress = st.progress(0)
for value in range(0, 101, 20):
    progress.progress(value)
progress.empty()

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
    "Random Forest": "random_forest.joblib"
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

st.sidebar.info(
    "Upload a labeled CSV containing "
    "`Literacy_Level` to calculate evaluation metrics."
)


# ============================================================
# MODEL COMPARISON
# ============================================================

metrics_path = os.path.join(
    MODEL_DIR,
    "metrics.csv"
)

if os.path.exists(metrics_path):

    metrics = pd.read_csv(metrics_path)

    st.markdown(
        '<div class="section-title">'
        '🏆 Model Performance Comparison'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if "Accuracy" in metrics.columns:

        best_idx = metrics["Accuracy"].idxmax()

        best_model = metrics.loc[
            best_idx, "Model"
        ]

        best_accuracy = metrics.loc[
            best_idx, "Accuracy"
        ]

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "🏆 Best Model",
            best_model
        )

        c2.metric(
            "🎯 Best Accuracy",
            f"{best_accuracy:.4f}"
        )

        c3.metric(
            "🤖 Models Compared",
            len(metrics)
        )

    # --------------------------------------------------------
    # METRICS TABLE
    # --------------------------------------------------------

    st.subheader("📋 Detailed Model Metrics")

    numeric_cols = [
        c for c in metrics.columns
        if c != "Model" and
        pd.api.types.is_numeric_dtype(metrics[c])
    ]

    st.dataframe(
        metrics.style.format(
            {c: "{:.4f}" for c in numeric_cols}
        ),
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # ACCURACY PLOT
    # ========================================================

    if "Accuracy" in metrics.columns:

        st.subheader("🎯 Accuracy Comparison")

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        sns.barplot(
            data=metrics,
            x="Model",
            y="Accuracy",
            ax=ax
        )

        ax.set_ylim(0, 1.05)
        ax.set_title(
            "Accuracy of Classification Models",
            color="#475569"
        )
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Model")

        plt.xticks(rotation=25)

        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%.3f",
                padding=3
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ========================================================
    # PRECISION / RECALL / F1
    # ========================================================

    prf_cols = [
        c for c in
        ["Precision", "Recall", "F1"]
        if c in metrics.columns
    ]

    if prf_cols:

        st.subheader(
            "📈 Precision, Recall & F1 Comparison"
        )

        melted = metrics.melt(
            id_vars="Model",
            value_vars=prf_cols,
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
            "Precision, Recall and F1 Score",
            color="#475569"
        )

        ax.set_ylabel("Score")
        ax.set_xlabel("Model")

        plt.xticks(rotation=25)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    # ========================================================
    # AUC
    # ========================================================

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

        ax.set_title(
            "ROC-AUC Score Comparison",
            color="#475569"
        )

        ax.set_ylabel("ROC-AUC")
        ax.set_xlabel("Model")

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

    # ========================================================
    # MCC
    # ========================================================

    if "MCC" in metrics.columns:

        st.subheader(
            "🎯 Matthews Correlation Coefficient"
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.barplot(
            data=metrics,
            x="Model",
            y="MCC",
            ax=ax
        )

        ax.set_title(
            "MCC Comparison",
            color="#475569"
        )

        ax.set_ylabel("MCC")
        ax.set_xlabel("Model")

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
        "⚠️ metrics.csv was not found in the model folder."
    )


# ============================================================
# LOAD DATA
# ============================================================

default_path = "test_data.csv"
target = "Literacy_Level"

if uploaded is not None:

    data = pd.read_csv(uploaded)

    st.success(
        f"✅ Dataset loaded: "
        f"{data.shape[0]} rows × {data.shape[1]} columns"
    )

elif os.path.exists(default_path):

    data = pd.read_csv(default_path)

    st.info(
        f"📁 Using default dataset: {default_path}"
    )

else:

    st.error(
        "❌ test_data.csv not found. "
        "Please upload a CSV file."
    )

    st.stop()


# ============================================================
# DATA INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📄 Dataset Information'
    '</div>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "📌 Rows",
    data.shape[0]
)

c2.metric(
    "📊 Features",
    data.shape[1] -
    (1 if target in data.columns else 0)
)

c3.metric(
    "🎯 Target Available",
    "Yes" if target in data.columns else "No"
)


# ============================================================
# PREPARE DATA
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
# PREDICTIONS TABLE
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🔮 Literacy Level Predictions'
    '</div>',
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

    pred = pd.Series(pred).astype(str).values

    proba = model.predict_proba(X)

    accuracy = accuracy_score(
        y_true,
        pred
    )

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
        '<div class="section-title">'
        '📊 Evaluation on Test Data'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    c1.metric(
        "🎯 Accuracy",
        f"{accuracy:.4f}"
    )

    c2.metric(
        "📈 AUC",
        f"{auc:.4f}"
    )

    c3.metric(
        "🔵 Precision",
        f"{precision:.4f}"
    )

    c4.metric(
        "🟢 Recall",
        f"{recall:.4f}"
    )

    c5.metric(
        "⭐ F1 Score",
        f"{f1:.4f}"
    )

    c6.metric(
        "🎯 MCC",
        f"{mcc:.4f}"
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🔥 Confusion Matrix'
        '</div>',
        unsafe_allow_html=True
    )

    cm = confusion_matrix(
        y_true,
        pred,
        labels=model.classes_
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_,
        linewidths=0.5,
        linecolor="#e2e8f0",
        ax=ax
    )

    ax.set_xlabel(
        "Predicted Literacy Level"
    )

    ax.set_ylabel(
        "Actual Literacy Level"
    )

    ax.set_title(
        f"Confusion Matrix - {selected}",
        color="#475569"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📋 Classification Report'
        '</div>',
        unsafe_allow_html=True
    )

    report = classification_report(
        y_true,
        pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).T.round(4)

    st.dataframe(
        report_df,
        use_container_width=True
    )


    # ========================================================
    # ACTUAL VS PREDICTED
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🔍 Actual vs Predicted'
        '</div>',
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
        "ℹ️ No Literacy_Level column was supplied. "
        "Only predictions are shown. Upload a labeled "
        "test CSV to display evaluation metrics."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
<div style="
text-align:center;
padding:20px;
color:#94a3b8;
font-size:14px;
">
<b>📚 Literacy Level Classification System</b><br>
Machine Learning Classification Project<br>
Logistic Regression • Decision Tree • kNN • Naive Bayes • Random Forest
</div>
""", unsafe_allow_html=True)
