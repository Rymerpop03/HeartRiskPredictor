import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import joblib
import os

# --- Load data ---
X_train = pd.read_csv('data/interim/X_train.csv')
X_test  = pd.read_csv('data/interim/X_test.csv')
y_train = pd.read_csv('data/interim/y_train.csv').squeeze()
y_test  = pd.read_csv('data/interim/y_test.csv').squeeze()

# --- Impute missing values ---
numerical_cols   = ['cigsPerDay', 'totChol', 'BMI', 'heartRate', 'glucose']
categorical_cols = ['education', 'BPMeds']

num_imputer = SimpleImputer(strategy='median')
X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
X_test[numerical_cols]  = num_imputer.transform(X_test[numerical_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
X_test[categorical_cols]  = cat_imputer.transform(X_test[categorical_cols])

# --- Scale features (required for SVM) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# --- Train model ---
model = SVC(
    C=1.0,
    kernel='rbf',
    probability=True,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train_scaled, y_train)

# --- Evaluate ---
y_pred      = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_prob):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Permutation importance (RBF SVM has no native feature importances) ---
print("\nComputing permutation importance (this may take a moment)...")
perm = permutation_importance(
    model, X_test_scaled, y_test,
    n_repeats=10, random_state=42, scoring='roc_auc'
)
feat_importance = pd.Series(perm.importances_mean, index=X_train.columns)
feat_importance = feat_importance.sort_values(ascending=True)

plt.figure(figsize=(8, 6))
feat_importance.plot(kind='barh')
plt.title('Permutation Importance - SVM')
plt.xlabel('Mean decrease in ROC-AUC')
plt.tight_layout()
plt.savefig('data/interim/feature_importance_svm.png', dpi=150)
plt.show()

# --- ROC curve ---
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
auc = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - SVM')
plt.legend()
plt.tight_layout()
plt.savefig('data/interim/roc_curve_svm.png', dpi=150)
plt.show()

# --- Save model and scaler ---
os.makedirs('models', exist_ok=True)
joblib.dump(model,  'models/svm.pkl')
joblib.dump(scaler, 'models/svm_scaler.pkl')
print("\nModel saved to models/svm.pkl")
print("Scaler saved to models/svm_scaler.pkl")
