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

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Literacy Classification",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# LIGHT YELLOW DESIGN
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #fffdf5;
}

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: 700;
    color: #8a6d1d;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    color: #756b4d;
    margin-bottom: 25px;
}

.section-title {
    font-size: 24px;
    font-weight: 650;
    color: #8a6d1d;
    border-bottom: 2px solid #f1df9a;
    padding-bottom: 7px;
    margin-top: 25px;
}

.info-card {
    background-color: #fffaf0;
    border: 1px solid #f1df9a;
    border-left: 4px solid #e8c95b;
    border-radius: 10px;
    padding: 18px;
    color: #665b3c;
}

section[data-testid="stSidebar"] {
    background-color: #fffaf0;
    border-right: 1px solid #f1df9a;
}

section[data-testid="stSidebar"] * {
    color: #665b3c !important;
}

div[data-testid="stMetric"] {
    background-color: #fffaf0;
    border: 1px solid #f1df9a;
    border-radius: 10px;
    padding: 12px;
}

div[data-testid="stMetricValue"] {
    color: #8a6d1d !important;
}

div[data-testid="stMetricLabel"] {
    color: #756b4d !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #f1df9a;
    border-radius: 8px;
}

[data-testid="stFileUploader"] {
    background-color: #fffaf0;
    border: 1px solid #f1df9a;
    border-radius: 8px;
}

div[data-baseweb="select"] > div {
    background-color: #fffdf8;
    border: 1px solid #e2cf86;
}

/* Simple animation */

.main-title {
    animation: appear 0.8s ease-in;
}

.section-title {
    animation: appear 0.6s ease-in;
}

div[data-testid="stMetric"] {
    transition: transform 0.2s;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
}

@keyframes appear {
    from {
        opacity: 0;
        transform: translateY(-8px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
    '<div class="main-title">📚 Literacy Level Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Compare machine learning models and predict literacy level'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# MODEL FILES
# --------------------------------------------------

MODEL_DIR = "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib"
}


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model(path):
    return joblib.load(path)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("⚙️ Controls")

selected = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_FILES.keys())
)

uploaded = st.sidebar.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

st.sidebar.write("Available Models:")
st.sidebar.write("- Logistic Regression")
st.sidebar.write("- Decision Tree")
st.sidebar.write("- kNN")
st.sidebar.write("- Naive Bayes")
st.sidebar.write("- Random Forest")


# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

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

    # Best model
    if "Accuracy" in metrics.columns:

        best = metrics.loc[
            metrics["Accuracy"].idxmax()
        ]

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "🏆 Best Model",
            best["Model"]
        )

        c2.metric(
            "🎯 Best Accuracy",
            f'{best["Accuracy"]:.4f}'
        )

        c3.metric(
            "🤖 Models",
            len(metrics)
        )

    # Metrics table
    st.subheader("📋 Model Metrics")

    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True
    )

    # Accuracy plot
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

        plt.xticks(rotation=25)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    # Precision, Recall and F1
    columns = [
        x for x in
        ["Precision", "Recall", "F1"]
        if x in metrics.columns
    ]

    if columns:

        st.subheader(
            "📈 Precision, Recall & F1"
        )

        data_plot = metrics.melt(
            id_vars="Model",
            value_vars=columns,
            var_name="Metric",
            value_name="Score"
        )

        fig, ax = plt.subplots(figsize=(11, 6))

        sns.barplot(
            data=data_plot,
            x="Model",
            y="Score",
            hue="Metric",
            ax=ax
        )

        ax.set_ylim(0, 1.05)

        plt.xticks(rotation=25)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    # AUC
    if "AUC" in metrics.columns:

        st.subheader("📈 AUC Comparison")

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.barplot(
            data=metrics,
            x="Model",
            y="AUC",
            ax=ax
        )

        ax.set_ylim(0, 1.05)

        plt.xticks(rotation=25)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    # MCC
    if "MCC" in metrics.columns:

        st.subheader("🎯 MCC Comparison")

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.barplot(
            data=metrics,
            x="Model",
            y="MCC",
            ax=ax
        )

        plt.xticks(rotation=25)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)


# --------------------------------------------------
# LOAD TEST DATA
# --------------------------------------------------

target = "Literacy_Level"
default_file = "test_data.csv"

if uploaded is not None:

    data = pd.read_csv(uploaded)

elif os.path.exists(default_file):

    data = pd.read_csv(default_file)

else:

    st.error("test_data.csv not found.")
    st.stop()


# --------------------------------------------------
# DATA INFORMATION
# --------------------------------------------------

st.markdown(
    '<div class="section-title">'
    '📄 Dataset Information'
    '</div>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)

c1.metric("Rows", data.shape[0])

c2.metric(
    "Features",
    data.shape[1] -
    (1 if target in data.columns else 0)
)

c3.metric(
    "Target Available",
    "Yes" if target in data.columns else "No"
)


# --------------------------------------------------
# PREPARE DATA
# --------------------------------------------------

has_target = target in data.columns

if has_target:
    X = data.drop(columns=[target])
else:
    X = data.copy()


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

model_path = os.path.join(
    MODEL_DIR,
    MODEL_FILES[selected]
)

model = load_model(model_path)

pred = model.predict(X)


# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

st.markdown(
    '<div class="section-title">'
    '🔮 Literacy Level Predictions'
    '</div>',
    unsafe_allow_html=True
)

result = X.copy()

result["Predicted_Literacy_Level"] = pred

st.dataframe(
    result,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

if has_target:

    y_true = data[target].astype(str)

    pred = pd.Series(pred).astype(str).values

    probability = model.predict_proba(X)

    accuracy = accuracy_score(
        y_true, pred
    )

    auc = roc_auc_score(
        y_true,
        probability,
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

    # Metrics
    st.markdown(
        '<div class="section-title">'
        '📊 Evaluation Results'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    c1.metric("Accuracy", f"{accuracy:.4f}")
    c2.metric("AUC", f"{auc:.4f}")
    c3.metric("Precision", f"{precision:.4f}")
    c4.metric("Recall", f"{recall:.4f}")
    c5.metric("F1 Score", f"{f1:.4f}")
    c6.metric("MCC", f"{mcc:.4f}")


    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

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

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=model.classes_,
        yticklabels=model.classes_,
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # --------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------

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

    st.dataframe(
        pd.DataFrame(report).T.round(4),
        use_container_width=True
    )


    # --------------------------------------------------
    # ACTUAL VS PREDICTED
    # --------------------------------------------------

    comparison = pd.DataFrame({
        "Actual": y_true.values,
        "Predicted": pred
    })

    comparison["Result"] = comparison.apply(
        lambda row:
        "Correct"
        if row["Actual"] == row["Predicted"]
        else "Incorrect",
        axis=1
    )

    st.markdown(
        '<div class="section-title">'
        '🔍 Actual vs Predicted'
        '</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No Literacy_Level column was supplied. "
        "Only predictions are displayed."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        padding:15px;
        color:#9a8b63;
    ">
        📚 Literacy Level Classification System<br>
        Machine Learning Classification Project
    </div>
    """,
    unsafe_allow_html=True
)
