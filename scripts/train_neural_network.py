import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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

# --- Scale features (important for neural networks) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# --- Convert to tensors ---
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_t  = torch.tensor(X_test_scaled,  dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_t  = torch.tensor(y_test.values,  dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True
)

# --- Model ---
class HeartDiseaseNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = HeartDiseaseNet(input_dim=X_train_t.shape[1]).to(device)

# --- Class imbalance: weight positive class higher ---
pos_weight = torch.tensor(
    [(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32
).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Rebuild model without final Sigmoid so BCEWithLogitsLoss handles it
class HeartDiseaseNetLogits(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = HeartDiseaseNetLogits(input_dim=X_train_t.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# --- Training loop ---
EPOCHS = 60
train_losses = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(X_batch)

    train_losses.append(epoch_loss / len(X_train_t))
    scheduler.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {train_losses[-1]:.4f}")

# --- Evaluate ---
model.eval()
with torch.no_grad():
    logits     = model(X_test_t.to(device)).cpu().squeeze()
    y_pred_prob = torch.sigmoid(logits).numpy()
    y_pred      = (y_pred_prob >= 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_prob):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Training loss plot ---
plt.figure(figsize=(7, 4))
plt.plot(range(1, EPOCHS + 1), train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss - Neural Network')
plt.tight_layout()
plt.savefig('data/interim/training_loss_nn.png', dpi=150)
plt.show()

# --- ROC curve ---
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
auc = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Neural Network')
plt.legend()
plt.tight_layout()
plt.savefig('data/interim/roc_curve_nn.png', dpi=150)
plt.show()

# --- Save model and scaler ---
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/neural_network.pt')
joblib.dump(scaler, 'models/nn_scaler.pkl')
print("\nModel saved to models/neural_network.pt")
print("Scaler saved to models/nn_scaler.pkl")
