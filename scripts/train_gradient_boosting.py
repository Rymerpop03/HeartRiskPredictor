import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
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

# --- Impute missing values (mirrors notebook preprocessing) ---
numerical_cols   = ['cigsPerDay', 'totChol', 'BMI', 'heartRate', 'glucose']
categorical_cols = ['education', 'BPMeds']

num_imputer = SimpleImputer(strategy='median')
X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
X_test[numerical_cols]  = num_imputer.transform(X_test[numerical_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
X_test[categorical_cols]  = cat_imputer.transform(X_test[categorical_cols])

# --- Train model ---
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_prob):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Feature importance plot ---
feature_importance = pd.Series(model.feature_importances_, index=X_train.columns)
feature_importance = feature_importance.sort_values(ascending=True)

plt.figure(figsize=(8, 6))
feature_importance.plot(kind='barh')
plt.title('Feature Importances - Gradient Boosting')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('data/interim/feature_importance.png', dpi=150)
plt.show()

# --- ROC curve ---
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
auc = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Gradient Boosting')
plt.legend()
plt.tight_layout()
plt.savefig('data/interim/roc_curve.png', dpi=150)
plt.show()

# --- Save model ---
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/gradient_boosting.pkl')
print("\nModel saved to models/gradient_boosting.pkl")
