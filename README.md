# Breast Cancer Diagnosis — Multi-Model Classification & Streamlit App

An end-to-end machine learning project that trains, evaluates, and deploys **five
classification models** on the Breast Cancer Wisconsin (Diagnostic) dataset, with
an interactive **Streamlit** web app for inference on uploaded test data.

---

## a. Problem Statement

Breast cancer is one of the most common cancers worldwide, and early, accurate
diagnosis significantly improves patient outcomes. Given 30 quantitative features
computed from digitized images of a fine-needle aspirate (FNA) of a breast mass,
the task is a **binary classification** problem: predict whether a tumour is
**malignant** (`0`) or **benign** (`1`).

The goal of this project is to:

1. Train five classical classifiers on the same dataset.
2. Compare them across six evaluation metrics.
3. Serve the trained models through an interactive Streamlit app that scores an
   uploaded test CSV **without retraining**.

---

## b. Dataset Description

| Property | Value |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic) |
| Source | UCI ML Repository / `sklearn.datasets.load_breast_cancer` |
| Task | Binary classification |
| Instances | **569** (≥ 500 required) |
| Features | **30** numeric (≥ 12 required) |
| Target | `diagnosis` → `0` = malignant, `1` = benign |
| Class balance | 212 malignant / 357 benign |
| Missing values | None |

The 30 features are derived from ten real-valued characteristics of each cell
nucleus (radius, texture, perimeter, area, smoothness, compactness, concavity,
concave points, symmetry, fractal dimension), each reported as a **mean**,
**standard error**, and **worst (largest)** value.

**Train/test split:** 80% train (455 rows) / 20% test (114 rows), stratified on
the target, `random_state = 2026`. The 114-row test split is exported as
[`test_data.csv`](test_data.csv) and is the file to upload in the Streamlit app.

---

## c. GitHub Repository Link

> 🔗 **Repository:** https://github.com/<your-github-username>/ml-classification-assignment

*(Replace `<your-github-username>` with your actual GitHub username after you push
the project.)*

**Repository structure**

```
ml-classification-assignment/
├── app.py                 # Streamlit application (inference only)
├── train_models.py        # Training + evaluation + model persistence
├── requirements.txt       # Pinned dependencies
├── README.md              # This file
├── test_data.csv          # Held-out test set (features + target)
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    └── metadata.joblib     # feature names, class names, cached metrics
```

> 🔴 **Live Streamlit App:** https://<your-app-name>.streamlit.app
> *(Fill in after deploying on Streamlit Community Cloud.)*

---

## d. Models Used

All five models are implemented on the **same dataset** and persisted as complete
scikit-learn `Pipeline`s (`StandardScaler` + estimator), so feature scaling is
applied identically at train and inference time.

1. **Logistic Regression** — `LogisticRegression(max_iter=5000, C=1.0)`
2. **Decision Tree Classifier** — `DecisionTreeClassifier(max_depth=5)`
3. **K-Nearest Neighbor** — `KNeighborsClassifier(n_neighbors=7, weights="distance")`
4. **Naive Bayes (Gaussian)** — `GaussianNB()`
5. **Support Vector Machine** — `SVC(kernel="rbf", C=2.0, gamma="scale")`

### Evaluation Metrics Comparison Table

Metrics are computed on the held-out **test set** (114 rows).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9737 | 0.9927 | 0.9726 | 0.9861 | 0.9793 | 0.9433 |
| Decision Tree | 0.9298 | 0.9395 | 0.9848 | 0.9028 | 0.9420 | 0.8588 |
| KNN | 0.9649 | 0.9947 | 0.9595 | 0.9861 | 0.9726 | 0.9245 |
| Naive Bayes | 0.9211 | 0.9848 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| Support Vector Machine | 0.9649 | 0.9960 | 0.9857 | 0.9583 | 0.9718 | 0.9260 |

> AUC uses predicted class probabilities (or the SVM decision-function scores) for
> the positive class. All metrics are reproducible by running
> `python train_models.py`.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Best overall performer — highest Accuracy (0.9737), F1 (0.9793) and MCC (0.9433). The classes are nearly linearly separable after standardization, so a linear boundary generalizes extremely well. Its high recall (0.9861) means very few malignant tumours are missed, which is exactly what matters clinically. |
| **Decision Tree** | Weakest on ranking (AUC 0.9395) and the lowest F1 among the top models. It has the highest precision (0.9848) but noticeably lower recall (0.9028), i.e. it misses more malignant cases. A single tree uses axis-aligned splits and tends to overfit/underfit relative to smoother models, even when depth is capped at 5. |
| **KNN** | Strong and well-balanced — joint-best recall (0.9861) and a very high AUC (0.9947). Distance-weighted voting over `k=7` neighbours in the scaled feature space works well because similar tumours cluster together. Precision is slightly lower, producing a few more false positives. |
| **Naive Bayes** | Lowest Accuracy (0.9211) and MCC (0.8341). The conditional-independence assumption is strongly violated here (radius, perimeter and area are highly correlated), which hurts hard predictions. Interestingly its AUC stays high (0.9848), showing it still ranks cases well even when its probability calibration is off. |
| **Support Vector Machine** | A very close second to Logistic Regression — the **best AUC (0.9960)** and the best precision among high-recall models (0.9857), with a strong MCC (0.9260). The RBF kernel captures mild non-linearities in the decision boundary, trading a little recall for very clean positive predictions. |
| **Overall, Winner for this dataset?** | **Logistic Regression** is the winner by Accuracy, F1 and MCC, with **SVM** an extremely close runner-up (and the best AUC). For a diagnostic setting where **recall/sensitivity is paramount**, Logistic Regression and KNN tie for the best recall (0.9861), reinforcing Logistic Regression as the best all-round choice. |

---

## How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) retrain the models — regenerates model/ and test_data.csv
python train_models.py

# 3. Launch the app
streamlit run app.py
```

Then upload [`test_data.csv`](test_data.csv) in the sidebar, pick a model, and view
the metrics, confusion matrix, classification report, and the all-model comparison.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. **New app** → select this repo → branch `main` → main file `app.py` → **Deploy**.
4. Share the generated `*.streamlit.app` link.

## App Features

- 📤 **Test-data upload (CSV)** — scores only uploaded test data (no retraining).
- 🔽 **Model-selection dropdown** — switch between all five classifiers.
- 📊 **Evaluation metrics** — Accuracy, AUC, Precision, Recall, F1, MCC.
- 🧩 **Confusion matrix + classification report** for the selected model.
- 🏁 **All-model comparison table** computed live on the uploaded data.
