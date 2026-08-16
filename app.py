import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# Page settings
st.set_page_config(
    page_title="Literacy Classification",
    layout="wide"
)


# Page design
st.markdown("""
<style>

.stApp {
    background-color: #ffffff;
    color: #000000;
}

/* Main title */
.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: 700;
    color: #000000 !important;
    margin-top: 15px;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 17px;
    color: #444444 !important;
    margin-bottom: 30px;
}

/* Section headings */
.section-title {
    font-size: 24px;
    font-weight: 700;
    color: #000000 !important;
    border-bottom: 1px solid #dddddd;
    padding-bottom: 8px;
    margin-top: 28px;
    margin-bottom: 15px;
}

/* Normal headings */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {
    color: #000000 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #dddddd;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label {
    color: #000000 !important;
}

/* Select box */
div[data-baseweb="select"] {
    background-color: #ffffff !important;
}

div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1px solid #cccccc !important;
    color: #000000 !important;
}

div[data-baseweb="select"] span {
    color: #000000 !important;
}

div[data-baseweb="select"] input {
    color: #000000 !important;
}

/* Dropdown */
ul[data-testid="stSelectboxVirtualDropdown"] {
    background-color: #ffffff !important;
}

ul[data-testid="stSelectboxVirtualDropdown"] li {
    color: #000000 !important;
    background-color: #ffffff !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #ffffff !important;
    border: 1px solid #cccccc !important;
    border-radius: 6px;
}

[data-testid="stFileUploader"] * {
    color: #000000 !important;
}

/* Metric boxes */
div[data-testid="stMetric"] {
    background-color: #ffffff !important;
    border: 1px solid #dddddd !important;
    border-radius: 8px;
    padding: 14px;
    transition: transform 0.2s;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
}

div[data-testid="stMetricLabel"] {
    color: #444444 !important;
}

div[data-testid="stMetricValue"] {
    color: #000000 !important;
}

/* Tables */
[data-testid="stDataFrame"] {
    background-color: #ffffff !important;
    border: 1px solid #dddddd !important;
    border-radius: 6px;
}

/* Buttons */
.stButton button {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #cccccc !important;
}

/* Messages */
[data-testid="stAlert"] {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #dddddd;
}

/* Animation */
.main-title {
    animation: appear 0.7s ease-in;
}

.section-title {
    animation: appear 0.5s ease-in;
}

@keyframes appear {
    from {
        opacity: 0;
        transform: translateY(-6px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

</style>
""", unsafe_allow_html=True)


# Title
st.markdown(
    '<div class="main-title">Literacy Level Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Compare machine learning models and predict literacy level'
    '</div>',
    unsafe_allow_html=True
)


# Model files
MODEL_DIR = "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib"
}


# Load trained model
@st.cache_resource
def load_model(path):
    return joblib.load(path)


# Sidebar
st.sidebar.header("Model Selection")

selected = st.sidebar.selectbox(
    "Select a model",
    list(MODEL_FILES.keys())
)

uploaded = st.sidebar.file_uploader(
    "Upload test data",
    type=["csv"]
)


# Model comparison
metrics_path = os.path.join(
    MODEL_DIR,
    "metrics.csv"
)

if os.path.exists(metrics_path):

    metrics = pd.read_csv(metrics_path)

    st.markdown(
        '<div class="section-title">'
        'Model Performance Comparison'
        '</div>',
        unsafe_allow_html=True
    )

    # Best model
    if "Accuracy" in metrics.columns:

        best = metrics.loc[
            metrics["Accuracy"].idxmax()
        ]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Best Model",
            best["Model"]
        )

        col2.metric(
            "Best Accuracy",
            f'{best["Accuracy"]:.4f}'
        )

        col3.metric(
            "Number of Models",
            len(metrics)
        )


    # Model metrics table
    st.markdown(
        '<div class="section-title">Model Metrics</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True
    )


    # Accuracy comparison
    if "Accuracy" in metrics.columns:

        st.markdown(
            '<div class="section-title">'
            'Accuracy Comparison'
            '</div>',
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        sns.barplot(
            data=metrics,
            x="Model",
            y="Accuracy",
            ax=ax
        )

        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy")

        plt.xticks(rotation=25)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)


    # Precision, Recall and F1
    performance_columns = [
        column
        for column in [
            "Precision",
            "Recall",
            "F1"
        ]
        if column in metrics.columns
    ]

    if performance_columns:

        st.markdown(
            '<div class="section-title">'
            'Precision, Recall and F1 Score'
            '</div>',
            unsafe_allow_html=True
        )

        plot_data = metrics.melt(
            id_vars="Model",
            value_vars=performance_columns,
            var_name="Metric",
            value_name="Score"
        )

        fig, ax = plt.subplots(
            figsize=(11, 6)
        )

        sns.barplot(
            data=plot_data,
            x="Model",
            y="Score",
            hue="Metric",
            ax=ax
        )

        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")

        plt.xticks(rotation=25)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)


    # AUC comparison
    if "AUC" in metrics.columns:

        st.markdown(
            '<div class="section-title">'
            'AUC Comparison'
            '</div>',
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        sns.barplot(
            data=metrics,
            x="Model",
            y="AUC",
            ax=ax
        )

        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Model")
        ax.set_ylabel("AUC")

        plt.xticks(rotation=25)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)


    # MCC comparison
    if "MCC" in metrics.columns:

        st.markdown(
            '<div class="section-title">'
            'Matthews Correlation Coefficient'
            '</div>',
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        sns.barplot(
            data=metrics,
            x="Model",
            y="MCC",
            ax=ax
        )

        ax.set_xlabel("Model")
        ax.set_ylabel("MCC")

        plt.xticks(rotation=25)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)


# Load test data
target = "Literacy_Level"
default_file = "test_data.csv"

if uploaded is not None:

    data = pd.read_csv(uploaded)

elif os.path.exists(default_file):

    data = pd.read_csv(default_file)

else:

    st.error(
        "test_data.csv was not found. "
        "Please upload a CSV file."
    )

    st.stop()


# Dataset information
st.markdown(
    '<div class="section-title">'
    'Dataset Information'
    '</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Rows",
    data.shape[0]
)

col2.metric(
    "Features",
    data.shape[1] -
    (1 if target in data.columns else 0)
)

col3.metric(
    "Target Available",
    "Yes" if target in data.columns else "No"
)


# Prepare data
has_target = target in data.columns

if has_target:
    X = data.drop(columns=[target])
else:
    X = data.copy()


# Load selected model
model_path = os.path.join(
    MODEL_DIR,
    MODEL_FILES[selected]
)

model = load_model(model_path)


# Make predictions
pred = model.predict(X)


# Prediction results
st.markdown(
    '<div class="section-title">'
    'Literacy Level Predictions'
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


# Evaluation
if has_target:

    y_true = data[target].astype(str)

    pred = pd.Series(
        pred
    ).astype(str).values

    probability = model.predict_proba(X)

    accuracy = accuracy_score(
        y_true,
        pred
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


    # Evaluation results
    st.markdown(
        '<div class="section-title">'
        'Evaluation Results'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    col1.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

    col2.metric(
        "AUC",
        f"{auc:.4f}"
    )

    col3.metric(
        "Precision",
        f"{precision:.4f}"
    )

    col4.metric(
        "Recall",
        f"{recall:.4f}"
    )

    col5.metric(
        "F1 Score",
        f"{f1:.4f}"
    )

    col6.metric(
        "MCC",
        f"{mcc:.4f}"
    )


    # Confusion matrix
    st.markdown(
        '<div class="section-title">'
        'Confusion Matrix'
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
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


    # Classification report
    st.markdown(
        '<div class="section-title">'
        'Classification Report'
        '</div>',
        unsafe_allow_html=True
    )

    report = classification_report(
        y_true,
        pred,
        output_dict=True,
        zero_division=0
    )

    report_table = pd.DataFrame(
        report
    ).T.round(4)

    st.dataframe(
        report_table,
        use_container_width=True
    )


    # Actual versus predicted
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
        'Actual vs Predicted'
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
        "The Literacy_Level column is not available. "
        "Only predictions are displayed."
    )


# Footer
st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        padding:15px;
        color:#555555;
    ">
        Literacy Level Classification System<br>
        Machine Learning Classification Project
    </div>
    """,
    unsafe_allow_html=True
)
