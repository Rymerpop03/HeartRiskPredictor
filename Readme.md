# Heart Disease Risk Predictor — How It Works

This document explains the full pipeline: from raw data to the Streamlit UI that outputs a 10-year coronary heart disease (CHD) risk percentage.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Dataset](#2-dataset)
3. [Data Preprocessing](#3-data-preprocessing)
4. [The 15 Input Features](#4-the-15-input-features)
5. [Model Training](#5-model-training)
   - [Gradient Boosting](#gradient-boosting)
   - [Random Forest](#random-forest)
   - [Support Vector Machine (SVM)](#support-vector-machine-svm)
   - [Neural Network](#neural-network)
6. [How Risk Probability Is Computed](#6-how-risk-probability-is-computed)
   - [Individual Models](#individual-models)
   - [Ensemble Mode](#ensemble-mode)
7. [Risk Tier Classification](#7-risk-tier-classification)
8. [The Streamlit Application](#8-the-streamlit-application)
9. [Project File Structure](#9-project-file-structure)

---

## 1. Overview

The predictor estimates the probability that a person will develop coronary heart disease (CHD) within the next 10 years. It does this by passing 15 clinical and demographic features through one or more trained machine learning models, then converting the resulting probability into a percentage and a named risk tier (Low / Moderate / High / Very High).

The application is built with **Streamlit** and can run any combination of four trained models: Gradient Boosting, Random Forest, SVM, and a PyTorch Neural Network.

---

## 2. Dataset

**Source:** The [Framingham Heart Study](https://www.framinghamheartstudy.org/) — a long-running cardiovascular cohort study.

**File:** `data/raw/framingham.csv`

**Target variable:** `TenYearCHD` — binary (0 = no CHD event in next 10 years, 1 = CHD event).

The dataset is imbalanced: CHD-positive cases are a minority. Two models (Random Forest, SVM) directly address this with `class_weight='balanced'`; the Neural Network handles it via a weighted loss function.

---

## 3. Data Preprocessing

Preprocessing is done in `dataPreprocessing.ipynb` and mirrored in each training script.

### Missing Value Imputation

| Column Group | Columns | Strategy |
|---|---|---|
| Numerical | `cigsPerDay`, `totChol`, `BMI`, `heartRate`, `glucose` | Median |
| Categorical | `education`, `BPMeds` | Most frequent (mode) |

Imputers are **fit on the training set only**, then applied to the test set to prevent data leakage.

### Train / Test Split

```
80% training  /  20% test
stratify=y  →  class proportions preserved in both splits
random_state=42
```

The split is saved to CSV files in `data/interim/` and loaded by each training script.

### Feature Scaling

Tree-based models (Gradient Boosting, Random Forest) do **not** require scaling.

Distance/gradient-based models (SVM, Neural Network) **do** require it. Both use `sklearn.preprocessing.StandardScaler` (zero mean, unit variance), fit on training data only. The fitted scaler objects are saved alongside each model so they can be re-applied identically at inference time.

---

## 4. The 15 Input Features

These are the exact features passed to every model, in this order:

| # | Feature | Type | Description |
|---|---|---|---|
| 1 | `male` | Binary | Biological sex (1 = male, 0 = female) |
| 2 | `age` | Integer | Age in years |
| 3 | `education` | Ordinal (1–4) | Highest education level |
| 4 | `currentSmoker` | Binary | Currently smokes cigarettes |
| 5 | `cigsPerDay` | Integer | Cigarettes smoked per day (0 if non-smoker) |
| 6 | `BPMeds` | Binary | Currently taking blood pressure medication |
| 7 | `prevalentStroke` | Binary | Prior history of stroke |
| 8 | `prevalentHyp` | Binary | Diagnosed with hypertension |
| 9 | `diabetes` | Binary | Diagnosed with diabetes |
| 10 | `totChol` | Float | Total cholesterol (mg/dL) |
| 11 | `sysBP` | Float | Systolic blood pressure (mmHg) |
| 12 | `diaBP` | Float | Diastolic blood pressure (mmHg) |
| 13 | `BMI` | Float | Body Mass Index (kg/m²) |
| 14 | `heartRate` | Float | Resting heart rate (bpm) |
| 15 | `glucose` | Float | Fasting blood glucose (mg/dL) |

The app collects all 15 values from the user through the Streamlit UI, then assembles them into a single-row `pandas.DataFrame` in the order above before passing them to the model.

---

## 5. Model Training

### Gradient Boosting

**Script:** `scripts/train_gradient_boosting.py`
**Saved to:** `models/gradient_boosting.pkl`

```
GradientBoostingClassifier(
    n_estimators  = 200,   # 200 sequential decision trees
    learning_rate = 0.05,  # small step size — reduces overfitting
    max_depth     = 4,     # shallow trees for regularization
    subsample     = 0.8,   # 80% of training data per tree (stochastic GB)
    random_state  = 42
)
```

Gradient Boosting builds trees sequentially, where each new tree corrects the residual errors of the ensemble so far. With `n_estimators=200` and a low learning rate, it slowly and precisely fits the training signal.

No scaler is saved for this model — tree splits are invariant to monotonic feature transformations.

---

### Random Forest

**Script:** `scripts/train_random_forest.py`
**Saved to:** `models/random_forest.pkl`

```
RandomForestClassifier(
    n_estimators     = 200,   # 200 parallel decision trees
    max_depth        = 8,     # limits tree growth to prevent overfitting
    min_samples_split= 10,    # node must have ≥10 samples to split
    min_samples_leaf = 4,     # leaf must contain ≥4 samples
    class_weight     = 'balanced',  # up-weights minority class (CHD=1)
    random_state     = 42,
    n_jobs           = -1     # use all CPU cores
)
```

Random Forest builds 200 trees in parallel, each on a bootstrap sample of the data and a random subset of features. Predictions are averaged over all trees. `class_weight='balanced'` automatically adjusts sample weights inversely proportional to class frequencies, compensating for the dataset imbalance.

---

### Support Vector Machine (SVM)

**Script:** `scripts/train_svm.py`
**Saved to:** `models/svm.pkl` + `models/svm_scaler.pkl`

```
SVC(
    C            = 1.0,       # regularization strength
    kernel       = 'rbf',     # radial basis function kernel
    probability  = True,      # enables predict_proba() via Platt scaling
    class_weight = 'balanced',
    random_state = 42
)
```

SVM finds the maximum-margin hyperplane separating CHD-positive and CHD-negative cases in feature space. The RBF kernel allows it to learn non-linear decision boundaries. Because SVM is distance-based, features **must be scaled** — the `StandardScaler` is saved and applied at inference time.

`probability=True` uses Platt scaling (a logistic regression fit on cross-validated scores) to convert the SVM's distance-from-hyperplane into a calibrated probability.

Feature importance for SVM is computed via **permutation importance** (the native SVM has no built-in importances for the RBF kernel): features are shuffled one at a time and the mean decrease in ROC-AUC is measured.

---

### Neural Network

**Script:** `scripts/train_neural_network.py`
**Saved to:** `models/neural_network.pt` + `models/nn_scaler.pkl`
**Requires:** PyTorch

Architecture:

```
Input (15)
  → Linear(15 → 64) → ReLU → BatchNorm1d(64) → Dropout(0.3)
  → Linear(64 → 32) → ReLU → BatchNorm1d(32) → Dropout(0.2)
  → Linear(32 → 1)
  → [Sigmoid applied at inference only]
```

Training details:

| Setting | Value |
|---|---|
| Loss | `BCEWithLogitsLoss` with `pos_weight` |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| LR scheduler | StepLR — halves every 20 epochs |
| Epochs | 60 |
| Batch size | 64 |
| Device | CUDA if available, otherwise CPU |

**`pos_weight`** = `(# negative samples) / (# positive samples)` — this makes each positive (CHD=1) sample count more in the loss, compensating for class imbalance without discarding any data.

The final layer outputs a raw **logit** (unbounded real number). At inference, `torch.sigmoid()` converts it to a probability in [0, 1].

The network is loaded in `HeartRiskPredictor.py` using a locally redefined `HeartNet` class (same architecture, no final sigmoid) and `net.load_state_dict(...)`.

---

## 6. How Risk Probability Is Computed

### Individual Models

The function `predict_proba(model_name, features)` in `HeartRiskPredictor.py`:

1. Assembles user inputs into a single-row DataFrame, enforcing the 15-feature order.
2. If the model requires scaling (SVM or Neural Network), applies the saved `StandardScaler`.
3. Calls the model:
   - **Gradient Boosting / Random Forest:** `model.predict_proba(X)[0][1]` — returns the probability of class 1 (CHD).
   - **SVM:** same as above, after scaling.
   - **Neural Network:** passes the scaled tensor through the network, then applies `torch.sigmoid()` to the raw logit output to get a probability.
4. Returns a single float in the range [0.0, 1.0].

### Ensemble Mode

The function `ensemble_proba(features)` loops over all loaded models, calls `predict_proba` for each, and returns the **arithmetic mean** of all probabilities:

```python
probs = [predict_proba(name, features) for name in models]
return float(np.mean(probs))
```

This simple averaging reduces variance and generally produces more robust estimates than any single model alone, especially when the individual models disagree.

---

## 7. Risk Tier Classification

The raw probability is mapped to a named tier by `get_risk_tier(prob)`:

| Probability Range | Tier | Color |
|---|---|---|
| < 10% | Low Risk | Green |
| 10% – 19.9% | Moderate Risk | Yellow |
| 20% – 34.9% | High Risk | Orange/Red |
| ≥ 35% | Very High Risk | Dark Red |

The UI also evaluates individual feature thresholds (age ≥ 55, sysBP ≥ 140, BMI ≥ 30, glucose ≥ 126, etc.) to generate a plain-language list of the user's specific risk flags, presented alongside the model output.

---

## 8. The Streamlit Application

**Entry point:** `HeartRiskPredictor.py`

### Startup

All models are loaded once at startup using `@st.cache_resource` — they are kept in memory across user interactions rather than reloaded on every prediction.

### Input Sections

| Section | Fields Collected |
|---|---|
| About You | Sex, Age, Education level |
| Lifestyle | Smoking status, Cigarettes per day |
| Medical History | BP medication, stroke history, hypertension, diabetes |
| Vitals & Lab Results | Systolic/Diastolic BP, Total Cholesterol, Fasting Glucose, Heart Rate, BMI (direct entry or calculated from height/weight in metric or imperial) |

### Prediction Flow

```
User clicks "Calculate My Heart Risk"
        ↓
features dict assembled from all 15 inputs
        ↓
predict_proba() or ensemble_proba() called
        ↓
prob (float 0–1) returned
        ↓
get_risk_tier(prob) → tier name, CSS class, icon, color
        ↓
pct = round(prob * 100, 1)
        ↓
Results displayed: percentage, tier badge, progress bar,
explanatory text, key risk factor flags
        ↓
(Ensemble only) expandable table showing each model's
individual probability and tier
```

### Model Selection

Users choose from:
- **Ensemble (All Models Combined)** — recommended, averages all loaded models
- **Gradient Boosting** — individual model
- **Random Forest** — individual model
- **SVM** — individual model
- **Neural Network** — individual model (only available if PyTorch is installed and the `.pt` file exists)

---

## 9. Project File Structure

```
HeartDisease/
├── HeartRiskPredictor.py          # Streamlit application (main entry point)
├── dataPreprocessing.ipynb        # EDA, imputation, train/test split
│
├── scripts/
│   ├── train_gradient_boosting.py
│   ├── train_random_forest.py
│   ├── train_svm.py
│   └── train_neural_network.py
│
├── models/
│   ├── gradient_boosting.pkl
│   ├── random_forest.pkl
│   ├── svm.pkl
│   ├── svm_scaler.pkl
│   ├── neural_network.pt
│   └── nn_scaler.pkl
│
└── data/
    ├── raw/
    │   └── framingham.csv
    └── interim/
        ├── X_train.csv / y_train.csv
        ├── X_test.csv  / y_test.csv
        ├── feature_importance.png       (Gradient Boosting)
        ├── roc_curve.png                (Gradient Boosting)
        ├── feature_importance_rf.png    (Random Forest)
        ├── roc_curve_rf.png             (Random Forest)
        ├── feature_importance_svm.png   (SVM permutation importance)
        ├── roc_curve_svm.png            (SVM)
        ├── training_loss_nn.png         (Neural Network)
        └── roc_curve_nn.png             (Neural Network)
```

---

> **Medical Disclaimer:** This tool is for educational purposes only and is based on statistical patterns from the Framingham Heart Study. It is not a medical diagnosis. Always consult a qualified healthcare provider for personalised medical advice.
