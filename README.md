# Literacy Level Classification

## 1. Problem statement

This project treats literacy level as a **3-class classification problem**: Low, Medium, or High. The objective is to compare Logistic Regression, Decision Tree, kNN, Gaussian Naive Bayes, and Random Forest using the six metrics required in the assignment: Accuracy, AUC, Precision, Recall, F1 and MCC.

The project also provides an interactive Streamlit application for model selection, performance comparison, test-data upload, and literacy-level prediction.

## 2. Dataset description

The project uses an education/student-performance dataset concept from Kaggle as the basis and prepares a literacy-classification version.

**Important:** `literacy_classification.csv` in this package is a **prepared/derived learning dataset**, not the original Kaggle download. The target `Literacy_Level` is constructed from reading performance:

- Low: reading score < 60
- Medium: 60-79.99
- High: >= 80

The final prepared file contains **1,000 rows, 13 input features and 1 target column**. The reading score itself is intentionally not used as an input feature to avoid direct target leakage.

### Public source used for the dataset concept

https://www.kaggle.com/datasets/sanlavisingh/education-dataset-csv-23-72-kb

The source dataset is described as containing 500 student records and education-related metrics.

## 3. Repository contents

**GitHub Repository:**  
https://github.com/sachinhuller-design/literacy-classification-ml-assignment

**Live Streamlit App:**  
https://literacy-level-classification.streamlit.app/

### Repository structure

```text
literacy_classification_project/
 app.py
 train_models.py
 requirements.txt
 README.md
 literacy_classification.csv
 test_data.csv
 model_metrics.csv
 model/
     logistic_regression.joblib
     decision_tree.joblib
     knn.joblib
     naive_bayes.joblib
     random_forest.joblib
     metrics.csv
```

## 4. Models used

The project implements five classification models:

1. Logistic Regression
2. Decision Tree
3. k-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest

### Model comparison from the prepared 80/20 stratified split

The following results are from the latest model-training execution used by the Streamlit application:

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8400 | 0.9458 | 0.8389 | 0.8400 | 0.8367 | 0.6419 |
| **Decision Tree** | **0.8700** | 0.8756 | **0.8701** | **0.8700** | **0.8694** | **0.7153** |
| kNN | 0.8250 | 0.9143 | 0.8395 | 0.8250 | 0.8056 | 0.5873 |
| Naive Bayes | 0.8000 | 0.9378 | 0.8207 | 0.8000 | 0.8050 | 0.6104 |
| Random Forest | 0.8500 | 0.9396 | 0.8512 | 0.8500 | 0.8487 | 0.6709 |

## 5. Observations

- **Logistic Regression:** Provides a strong baseline with 84% accuracy and the highest AUC (0.9458) among the five models. Its performance is balanced across the classification metrics.

- **Decision Tree:** Achieves the **highest accuracy (87%)**, highest precision (0.8701), highest recall (0.8700), highest F1 score (0.8694), and highest MCC (0.7153). Although its AUC is lower than Logistic Regression and Random Forest, it provides the strongest overall classification performance in this run.

- **kNN:** Achieves 82.5% accuracy. Its performance is competitive, but it is weaker than the Decision Tree and Random Forest on the main classification metrics. Scaling is important because kNN is distance-based.

- **Naive Bayes:** Provides reasonably strong AUC (0.9378) and is computationally efficient, but its accuracy and F1 score are lower than the other stronger models.

- **Random Forest:** Achieves 85% accuracy and a strong AUC of 0.9396. It performs better than Logistic Regression, kNN, and Naive Bayes in accuracy, but it does not outperform the Decision Tree in this particular run.

### Overall Winner

**Decision Tree** is the overall winner for this prepared dataset in the latest run.

It achieves:

- **Accuracy:** 0.8700
- **Precision:** 0.8701
- **Recall:** 0.8700
- **F1 Score:** 0.8694
- **MCC:** 0.7153

The Streamlit application also identifies **Decision Tree** as the best model with a best accuracy of **0.8700**.

## 6. How to run

### Step 1 - Install packages

```bash
pip install -r requirements.txt
```

### Step 2 - Train and save models

```bash
python train_models.py
```

### Step 3 - Start Streamlit locally

```bash
streamlit run app.py
```

The browser should open the Streamlit interface.

## 7. Streamlit application

### Live application

https://literacy-level-classification.streamlit.app/

The deployed application provides an interactive frontend for comparing the five trained classification models.

### Streamlit features

1. **Test-data CSV upload**
2. **Model-selection dropdown**
3. **Model performance comparison**
4. **Best model display**
5. **Best accuracy display**
6. **Accuracy, AUC, Precision, Recall, F1 and MCC**
7. **Confusion matrix**
8. **Classification report**
9. **Prediction table**
10. **Literacy-level prediction using the selected model**

The current application displays:

- **Best Model:** Decision Tree
- **Best Accuracy:** 0.8700
- **Number of Models:** 5

The upload section is intended for test data, consistent with the assignment requirement for the Streamlit free tier.

## 8. Conclusion

Five classification algorithms were trained and evaluated on the prepared literacy-classification dataset using six required evaluation metrics.

Among the tested models, the **Decision Tree performed best in the latest run**, achieving **87% accuracy** and the highest MCC score of **0.7153**. Therefore, Decision Tree is selected as the overall winner for this dataset and is also identified as the best model by the Streamlit application.

