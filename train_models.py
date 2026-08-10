#Train Model save the dataset in the folder where you want to run this script
import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

DATA_FILE = "literacy_classification.csv"
MODEL_DIR = "model"
RANDOM_STATE = 42

df = pd.read_csv(DATA_FILE)
target = "Literacy_Level"
X = df.drop(columns=[target])
y = df[target].astype(str)

categorical = X.select_dtypes(include=["object"]).columns.tolist()
numeric = [c for c in X.columns if c not in categorical]

pre_sparse = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

pre_dense = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), categorical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

models = {
    "logistic_regression": (LogisticRegression(max_iter=2000), pre_sparse),
    "decision_tree": (DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE), pre_sparse),
    "knn": (KNeighborsClassifier(n_neighbors=9), pre_sparse),
    "naive_bayes": (GaussianNB(), pre_dense),
    "random_forest": (RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
    ), pre_sparse),
}

os.makedirs(MODEL_DIR, exist_ok=True)
rows = []

for name, (model, preprocessor) in models.items():
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)

    rows.append({
        "Model": name.replace("_", " ").title(),
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, proba, multi_class="ovr", labels=pipeline.classes_),
        "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
        "F1": f1_score(y_test, pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred)
    })

    joblib.dump(pipeline, os.path.join(MODEL_DIR, f"{name}.joblib"))

metrics = pd.DataFrame(rows)
metrics.to_csv(os.path.join(MODEL_DIR, "metrics.csv"), index=False)
X_test.assign(Literacy_Level=y_test).to_csv("test_data.csv", index=False)

with open(os.path.join(MODEL_DIR, "classes.json"), "w") as f:
    json.dump(sorted(y.unique().tolist()), f, indent=2)

print("\nModel comparison:\n")
print(metrics.round(4).to_string(index=False))
print("\nSaved models in:", MODEL_DIR)
