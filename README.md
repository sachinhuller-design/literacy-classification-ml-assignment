# Literacy Level Classification

## 1. Problem statement

This project treats literacy level as a **3-class classification problem**: Low, Medium, or High. The objective is to compare Logistic Regression, Decision Tree, kNN, Gaussian Naive Bayes, and Random Forest using the six metrics required in the assignment: Accuracy, AUC, Precision, Recall, F1 and MCC.

## 2. Dataset description

The project uses an education/student-performance dataset concept from Kaggle as the basis and prepares a literacy-classification version.

**Important:** `literacy_classification.csv` in this package is a **prepared/derived learning dataset**, not the original Kaggle download. The target `Literacy_Level` is constructed from reading performance:
- Low: reading score < 60
- Medium: 60–79.99
- High: >= 80

The final prepared file contains 1,000 rows, 13 input features and 1 target column. The reading score itself is intentionally not used as an input feature to avoid direct target leakage.

Public source used for the dataset concept:
https://www.kaggle.com/datasets/sanlavisingh/education-dataset-csv-23-72-kb

The source dataset is described as containing 500 student records and education-related metrics.

## 3. Repository contents

GITHUB Link: https://github.com/sachinhuller-design/literacy-classification-ml-assignment

Repository structure

```text
literacy_classification_project/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── literacy_classification.csv
├── test_data.csv
├── model_metrics.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
    └── metrics.csv
```

## 4. Models used

The model specifies five model rows in its comparison table: Logistic Regression, Decision Tree, kNN, Naive Bayes and Random Forest.

### Model comparison from the prepared 80/20 stratified split

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8400 | 0.9455 | 0.8389 | 0.8400 | 0.8367 | 0.6419 |
| Decision Tree | 0.8500 | 0.8408 | 0.8482 | 0.8500 | 0.8484 | 0.6682 |
| kNN | 0.8350 | 0.8968 | 0.8514 | 0.8350 | 0.8162 | 0.6146 |
| Naive Bayes | 0.8000 | 0.9385 | 0.8207 | 0.8000 | 0.8050 | 0.6104 |
| Random Forest | 0.8500 | 0.9444 | 0.8512 | 0.8500 | 0.8487 | 0.6709 |

## 5. Observations

- **Logistic Regression:** Strong AUC and balanced overall performance. It provides a useful linear baseline.
- **Decision Tree:** Highest accuracy among the individual tree-style models in this run and easy to interpret, but its AUC is lower than the two strongest probabilistic models.
- **kNN:** Competitive but slightly weaker overall; scaling is important because distance-based learning is sensitive to feature magnitude.
- **Naive Bayes:** Fast and reasonably strong AUC, but its independence assumptions reduce overall classification quality compared with the best models.
- **Random Forest:** Tied for the best accuracy and achieved the highest MCC in this run, making it the **overall winner** for this prepared dataset.

**Overall winner: Random Forest**, based on the highest MCC and tied-highest accuracy.

## 6. How to run

### Step 1 – Install packages

```bash
pip install -r requirements.txt
```

### Step 2 – Train/save models

```bash
python train_models.py
```

### Step 3 – Start Streamlit

```bash
streamlit run app.py
```

The browser should open the Streamlit interface.

## 7. Streamlit features
Link to Streamlit app in https://literacy-level-classification.streamlit.app/
Virtual machine: http://localhost:8501/
The app includes:
1. CSV upload
2. Model-selection dropdown
3. Accuracy, AUC, Precision, Recall, F1 and MCC
4. Confusion matrix
5. Classification report
6. Prediction table
