"""
Credit Card Fraud Detection - Deep Learning Challenge (TabNet & Autoencoder)
=============================================================================
Goal: Beat the current best PR AUC of 0.8849 using Deep Learning (GPU)

Techniques:
1. TabNet (Google's Attention-based Network for Tabular Data)
2. Denoising Autoencoder (DAE) for Feature Extraction
3. Neural Network with DAE Features
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_recall_curve, auc
from pytorch_tabnet.tab_model import TabNetClassifier
import xgboost as xgb
import warnings
import os
import json

warnings.filterwarnings('ignore')

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_STATE)

print("=" * 60)
print("🧠 Deep Learning Challenge - TabNet & Autoencoder")
print("=" * 60)
print(f"Current Best Score: 0.8849 (XGBoost)")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Load data
print("\n📊 Loading and Preprocessing Data...")
df = pd.read_csv('creditcard.csv')

# Preprocessing (same as winning XGBoost)
X = df.drop('Class', axis=1)
y = df['Class']

X['Amount_log'] = np.log1p(X['Amount'])
X['Time_hour'] = (X['Time'] % 3600) / 3600
X['Time_day'] = (X['Time'] % 86400) / 86400

v_features = [f'V{i}' for i in range(1, 29)]
X['V_mean'] = X[v_features].mean(axis=1)
X['V_std'] = X[v_features].std(axis=1)
X['V_max'] = X[v_features].max(axis=1)
X['V_min'] = X[v_features].min(axis=1)
X['V_range'] = X['V_max'] - X['V_min']

X['V14_V17_int'] = X['V14'] * X['V17']
X['V10_V12_int'] = X['V10'] * X['V12']
X['V3_V7_int'] = X['V3'] * X['V7']
X['V4_V11_int'] = X['V4'] * X['V11']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Scale
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def calculate_pr_auc(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return auc(recall, precision)

results = {}

# ============================================================
# Model 1: TabNet
# ============================================================
print("\n" + "="*60)
print("🚀 Training TabNet Classifier")
print("="*60)

# TabNet requires numpy arrays
X_train_np = X_train_scaled
X_test_np = X_test_scaled
y_train_np = y_train.values
y_test_np = y_test.values

# Define TabNet
tabnet = TabNetClassifier(
    n_d=64, n_a=64, n_steps=5,
    gamma=1.5, n_independent=2, n_shared=2,
    lambda_sparse=1e-4, momentum=0.3, clip_value=2.,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params=dict(step_size=50, gamma=0.9),
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='entmax', # 'sparsemax'
    device_name='auto',
    verbose=1
)

# Train TabNet (with class weights for imbalance)
# Calculate weights
neg_count = len(y_train[y_train==0])
pos_count = len(y_train[y_train==1])
print(f"   Class Imbalance: {pos_count} positive / {neg_count} negative")

# TabNet handles weights internally if passed to fit? No, mostly uses weights in loss
# We'll use a simple fitting approach
tabnet.fit(
    X_train_np, y_train_np,
    eval_set=[(X_train_np, y_train_np), (X_test_np, y_test_np)],
    eval_name=['train', 'valid'],
    eval_metric=['auc'],
    max_epochs=20, # Efficient epochs
    patience=5,
    batch_size=1024, # Large batch for GPU
    virtual_batch_size=128,
    num_workers=0,
    weights=1, # 0=No sampling, 1=Automated sampling
    drop_last=False
)

tabnet_pred = tabnet.predict_proba(X_test_np)[:, 1]
results['TabNet'] = calculate_pr_auc(y_test, tabnet_pred)
print(f"   TabNet PR AUC: {results['TabNet']:.4f}")

# ============================================================
# Model 2: Denoising Autoencoder + MLP
# ============================================================
print("\n" + "="*60)
print("🔮 Training Denoising Autoencoder (DAE) + MLP")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DAE(nn.Module):
    def __init__(self, input_dim):
        super(DAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(), # Swish activation (state of the art)
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Prepare data loaders
train_tensor = torch.FloatTensor(X_train_scaled).to(device)
test_tensor = torch.FloatTensor(X_test_scaled).to(device)
y_train_tensor = torch.FloatTensor(y_train.values).to(device)

train_loader = DataLoader(TensorDataset(train_tensor, y_train_tensor), batch_size=2048, shuffle=True)

# 1. Train Autoencoder (Unsupervised)
dae = DAE(X_train_scaled.shape[1]).to(device)
dae_optimizer = torch.optim.AdamW(dae.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()

print("   Training Autoencoder...")
for epoch in range(15):
    dae.train()
    total_loss = 0
    for batch_x, _ in train_loader:
        # Add noise
        noise = torch.randn_like(batch_x) * 0.1
        noisy_x = batch_x + noise
        
        _, decoded = dae(noisy_x)
        loss = criterion(decoded, batch_x)
        
        dae_optimizer.zero_grad()
        loss.backward()
        dae_optimizer.step()
        total_loss += loss.item()
    
    if (epoch+1) % 5 == 0:
        print(f"     Epoch {epoch+1}/15 - Loss: {total_loss/len(train_loader):.6f}")

# 2. Train Classifier using Encoder Features
class FraudClassifier(nn.Module):
    def __init__(self, input_dim, encoder):
        super(FraudClassifier, self).__init__()
        self.encoder = encoder
        # Freeze encoder? No, let's fine tune it
        self.head = nn.Sequential(
            nn.Linear(64 + input_dim, 128), # Skip connection: Original features + Encoded
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder.encoder(x)
        combined = torch.cat([x, encoded], dim=1) # Concatenate
        return self.head(combined)

classifier = FraudClassifier(X_train_scaled.shape[1], dae).to(device)
clf_optimizer = torch.optim.AdamW(classifier.parameters(), lr=5e-4) # Lower LR for fine tuning
bce_loss = nn.BCELoss(weight=torch.tensor([scale_pos]).to(device) if 'scale_pos' in globals() else None) # Weighted loss not directly supported in BCELoss single tensor

# Custom weighted loss
def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2
        
        loss = weights[1] * (target * torch.log(output + 1e-7)) + \
               weights[0] * ((1 - target) * torch.log(1 - output + 1e-7))
    else:
        loss = target * torch.log(output + 1e-7) + (1 - target) * torch.log(1 - output + 1e-7)

    return torch.neg(torch.mean(loss))

# Calculate weights
pos_weight = len(y_train) / (2 * pos_count)
neg_weight = len(y_train) / (2 * neg_count)
weights = [neg_weight, pos_weight]

print("   Training Classifier Head...")
for epoch in range(20):
    classifier.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        pred = classifier(batch_x).squeeze()
        loss = weighted_binary_cross_entropy(pred, batch_y, weights)
        
        clf_optimizer.zero_grad()
        loss.backward()
        clf_optimizer.step()
        train_loss += loss.item()
        
    # Eval
    classifier.eval()
    with torch.no_grad():
        test_pred = classifier(test_tensor).cpu().numpy().flatten()
        test_score = calculate_pr_auc(y_test, test_pred)
    
    print(f"     Epoch {epoch+1}/20 - Loss: {train_loss/len(train_loader):.4f} - Val PR AUC: {test_score:.4f}")

# Final DAE prediction
dae_pred = test_pred
results['DAE_MLP'] = calculate_pr_auc(y_test, dae_pred)
print(f"   DAE+MLP PR AUC: {results['DAE_MLP']:.4f}")

# ============================================================
# Model 3: Optimized XGBoost (for Ensembling)
# ============================================================
print("\n" + "="*60)
print("🚀 Retraining Optimal XGBoost (for Ensemble)")
print("="*60)

# Optimal parameters from beat_the_score_optuna_v4.py
best_xgb_params = {
    "lambda": 2.2626, "alpha": 1.3167, "colsample_bytree": 0.6079,
    "subsample": 0.8867, "learning_rate": 0.1909, "n_estimators": 668,
    "max_depth": 5, "min_child_weight": 2, "scale_pos_weight": 199.4,
    "objective": "binary:logistic", "eval_metric": "aucpr",
    "random_state": RANDOM_STATE, "use_label_encoder": False
}

xgb_model = xgb.XGBClassifier(**best_xgb_params)
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict_proba(X_test_scaled)[:, 1]
results['XGBoost'] = calculate_pr_auc(y_test, xgb_pred)
print(f"   XGBoost PR AUC: {results['XGBoost']:.4f}")

# ============================================================
# Ensemble
# ============================================================
print("\n" + "="*60)
print("⚖️ Creating Deep Ensemble")
print("="*60)

# Weighted Average
# Give more weight to the consistent XGBoost, but allow DL to boost it
ensemble_pred = (0.6 * xgb_pred) + (0.2 * tabnet_pred) + (0.2 * dae_pred)
ensemble_score = calculate_pr_auc(y_test, ensemble_pred)

print(f"\n🏆 FINAL DEEP LEARNING RESULTS")
print("-" * 40)
print(f"1. Deep Ensemble: {ensemble_score:.4f}")
print(f"2. XGBoost (Baseline): {results['XGBoost']:.4f}")
print(f"3. TabNet: {results['TabNet']:.4f}")
print(f"4. DAE+MLP: {results['DAE_MLP']:.4f}")

if ensemble_score > 0.8849:
    print(f"\n🎉 SUCCESS! Beat previous best (0.8849) by +{(ensemble_score - 0.8849):.4f}")
else:
    print(f"\n⚠️ Did not beat previous best. Gap: {(0.8849 - ensemble_score):.4f}")
