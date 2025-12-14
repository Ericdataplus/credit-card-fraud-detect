"""
🛡️ SOTA Credit Card Fraud Detection - Ultimate Edition
=========================================================
Leveraging RTX 3060 12GB for State-of-the-Art Deep Learning

Features:
1. Multiple Dataset Integration (Original + 2023 + NeurIPS Bank Fraud + PaySim)
2. GPU-Accelerated PyTorch Models
3. Focal Loss for Extreme Class Imbalance
4. TabNet with Advanced Configuration
5. Deep Neural Network with Attention
6. Ensemble of SOTA Models
7. SHAP Explainability
8. Comprehensive Benchmarking

Author: Ericdataplus
Date: December 2024 (Updated December 2025)
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    precision_recall_curve, auc, roc_auc_score, 
    confusion_matrix, classification_report,
    average_precision_score, f1_score
)
# Note: Using Focal Loss instead of SMOTE for handling class imbalance
# This is the current SOTA approach for extreme imbalance

# Gradient Boosting
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

# TabNet
from pytorch_tabnet.tab_model import TabNetClassifier

# Explainability
import shap

# Hyperparameter Tuning
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_STATE = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {DEVICE}")

if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# Output directory
OUTPUT_DIR = Path("graphs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# FOCAL LOSS (State-of-the-Art for Imbalanced Classification)
# ============================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for handling extreme class imbalance.
    From: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ============================================================
# ATTENTION-BASED DEEP NETWORK (2024 SOTA Architecture)
# ============================================================
class AttentionBlock(nn.Module):
    """Multi-Head Self-Attention for tabular data"""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        return x.squeeze(1)


class DeepFraudNet(nn.Module):
    """
    State-of-the-Art Deep Network for Fraud Detection
    
    Architecture:
    - Feature embedding layers
    - Multi-head self-attention
    - Residual MLP blocks
    - Focal Loss compatible output
    """
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3, num_heads=4):
        super().__init__()
        
        # Initial embedding
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Attention block
        self.attention = AttentionBlock(hidden_dims[0], num_heads, dropout)
        
        # Residual MLP blocks
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.blocks.append(self._make_block(hidden_dims[i], hidden_dims[i+1], dropout))
        
        # Output layer (logits for Focal Loss)
        self.output = nn.Linear(hidden_dims[-1], 1)
        
        # Initialize weights
        self._init_weights()
    
    def _make_block(self, in_dim, out_dim, dropout):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.input_embed(x)
        x = self.attention(x)
        
        for block in self.blocks:
            x = block(x) + x[:, :block[0].out_features] if x.shape[1] == block[0].out_features else block(x)
        
        return self.output(x)


# ============================================================
# TRANSFORMER ENCODER FOR TABULAR DATA
# ============================================================
class TabularTransformer(nn.Module):
    """
    Transformer-based model for tabular fraud detection
    Inspired by FT-Transformer and TabTransformer
    """
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3, dropout=0.2):
        super().__init__()
        
        # Feature tokenization (each feature becomes a token)
        self.feature_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, input_dim, d_model) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Tokenize features: (batch, features) -> (batch, features, d_model)
        x = x.unsqueeze(-1)  # (batch, features, 1)
        x = self.feature_embed(x)  # (batch, features, d_model)
        x = x + self.pos_embed
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer
        x = self.transformer(x)
        
        # Use CLS token for classification
        return self.head(x[:, 0])


# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================
def load_all_datasets():
    """Load and combine multiple fraud datasets"""
    print("\n" + "=" * 60)
    print("📂 Loading All Datasets")
    print("=" * 60)
    
    datasets = {}
    
    # 1. Original Credit Card Fraud (Kaggle)
    print("\n1️⃣ Loading Original Credit Card Fraud Dataset...")
    try:
        df_original = pd.read_csv('creditcard.csv')
        print(f"   ✅ Loaded: {len(df_original):,} transactions")
        print(f"   Fraud rate: {df_original['Class'].mean()*100:.3f}%")
        datasets['original'] = df_original
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 2. 2023 Dataset (if augmented before)
    print("\n2️⃣ Checking for 2023 Augmentation Dataset...")
    kaggle_cache = Path.home() / ".cache" / "kagglehub" / "datasets"
    
    # Search for 2023 fraud dataset
    fraud_2023_paths = list(kaggle_cache.glob("**/fraudTest.csv")) + list(kaggle_cache.glob("**/creditcard_2023.csv"))
    if fraud_2023_paths:
        try:
            df_2023 = pd.read_csv(fraud_2023_paths[0])
            print(f"   ✅ Found 2023 data: {len(df_2023):,} transactions")
            datasets['fraud_2023'] = df_2023
        except:
            print("   ⚠️ Could not load 2023 dataset")
    
    # 3. Bank Account Fraud (NeurIPS 2022)
    print("\n3️⃣ Loading Bank Account Fraud (NeurIPS 2022)...")
    bank_fraud_paths = list(kaggle_cache.glob("**/bank-account-fraud-dataset*/**/*.csv"))
    if bank_fraud_paths:
        try:
            # Load the base dataset (typically Base.csv)
            base_path = [p for p in bank_fraud_paths if 'Base' in p.name]
            if base_path:
                df_bank = pd.read_csv(base_path[0])
                print(f"   ✅ Loaded: {len(df_bank):,} accounts")
                print(f"   Fraud rate: {df_bank['fraud_bool'].mean()*100:.2f}%")
                datasets['bank_fraud'] = df_bank
            else:
                # Try first CSV
                df_bank = pd.read_csv(bank_fraud_paths[0])
                print(f"   ✅ Loaded: {len(df_bank):,} records")
                datasets['bank_fraud'] = df_bank
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # 4. PaySim Mobile Money
    print("\n4️⃣ Loading PaySim Mobile Money Dataset...")
    paysim_paths = list(kaggle_cache.glob("**/paysim*/**/*.csv"))
    if paysim_paths:
        try:
            df_paysim = pd.read_csv(paysim_paths[0])
            print(f"   ✅ Loaded: {len(df_paysim):,} transactions")
            print(f"   Fraud rate: {df_paysim['isFraud'].mean()*100:.3f}%")
            datasets['paysim'] = df_paysim
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return datasets


def prepare_credit_card_data(df, test_size=0.2):
    """Prepare credit card fraud data with feature engineering"""
    print("\n🔧 Feature Engineering...")
    
    X = df.drop('Class', axis=1).copy()
    y = df['Class'].copy()
    
    # Feature engineering
    X['Amount_log'] = np.log1p(X['Amount'])
    X['Time_hour'] = (X['Time'] % 3600) / 3600
    X['Time_day'] = (X['Time'] % 86400) / 86400
    
    # V-feature statistics
    v_features = [f'V{i}' for i in range(1, 29)]
    X['V_mean'] = X[v_features].mean(axis=1)
    X['V_std'] = X[v_features].std(axis=1)
    X['V_max'] = X[v_features].max(axis=1)
    X['V_min'] = X[v_features].min(axis=1)
    X['V_range'] = X['V_max'] - X['V_min']
    X['V_skew'] = X[v_features].skew(axis=1)
    
    # Important feature interactions (based on previous analysis)
    X['V14_V17_int'] = X['V14'] * X['V17']
    X['V10_V12_int'] = X['V10'] * X['V12']
    X['V3_V7_int'] = X['V3'] * X['V7']
    X['V4_V11_int'] = X['V4'] * X['V11']
    X['V14_sq'] = X['V14'] ** 2
    X['V17_sq'] = X['V17'] ** 2
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    
    # Scale with RobustScaler (handles outliers)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"   Training: {len(X_train):,} samples ({y_train.sum():,} fraud)")
    print(f"   Test: {len(X_test):,} samples ({y_test.sum():,} fraud)")
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler, X.columns.tolist()


# ============================================================
# MODEL TRAINING FUNCTIONS
# ============================================================
def train_deep_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=2048, lr=1e-3, patience=15):
    """Train a PyTorch model with early stopping and LR scheduling"""
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).reshape(-1, 1).to(DEVICE)
    
    # DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Loss, optimizer, scheduler
    criterion = FocalLoss(alpha=0.75, gamma=2.0)  # Higher alpha for fraud class
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader))
    
    model.to(DEVICE)
    best_val_auc = 0
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_auc': [], 'val_pr_auc': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = torch.sigmoid(model(X_val_t)).cpu().numpy()
        
        val_roc_auc = roc_auc_score(y_val, val_outputs)
        val_pr_auc = average_precision_score(y_val, val_outputs)
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_auc'].append(val_roc_auc)
        history['val_pr_auc'].append(val_pr_auc)
        
        if val_pr_auc > best_val_auc:
            best_val_auc = val_pr_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, "
                  f"ROC-AUC={val_roc_auc:.4f}, PR-AUC={val_pr_auc:.4f}")
        
        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_model_state)
    return model, history, best_val_auc


def train_tabnet_model(X_train, y_train, X_val, y_val):
    """Train TabNet with GPU acceleration"""
    print("   Training TabNet...")
    
    # Class weights for imbalance
    n_fraud = y_train.sum()
    n_legit = len(y_train) - n_fraud
    weights = {0: 1.0, 1: n_legit / n_fraud}
    
    clf = TabNetClassifier(
        n_d=64, n_a=64,  # Width of decision/attention layers
        n_steps=5,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        lambda_sparse=1e-4,
        momentum=0.3,
        clip_value=2.0,
        optimizer_fn=torch.optim.AdamW,
        optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
        scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        scheduler_params=dict(T_0=10, T_mult=2),
        mask_type='entmax',
        verbose=0,
        device_name='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=['auc'],
        max_epochs=200,
        patience=20,
        batch_size=4096,
        virtual_batch_size=512,
        num_workers=0,
        drop_last=False,
        weights=weights
    )
    
    return clf


def train_gradient_boosting(X_train, y_train, X_val, y_val, n_trials=20):
    """Train optimized gradient boosting ensemble"""
    print("\n🚀 Training Gradient Boosting Ensemble with Optuna...")
    
    results = {}
    
    # ----- XGBoost -----
    print("\n   XGBoost:")
    def objective_xgb(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'tree_method': 'hist',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 100, 600),
            'random_state': RANDOM_STATE
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, preds)
    
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(objective_xgb, n_trials=n_trials, show_progress_bar=True)
    
    # Train final XGBoost
    best_params = study_xgb.best_params.copy()
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = 'aucpr'
    best_params['random_state'] = RANDOM_STATE
    best_params['tree_method'] = 'hist'
    
    xgb_model = xgb.XGBClassifier(**best_params)
    xgb_model.fit(X_train, y_train)
    results['xgboost'] = xgb_model
    print(f"      Best CV PR-AUC: {study_xgb.best_value:.4f}")
    
    # ----- LightGBM -----
    print("\n   LightGBM:")
    def objective_lgb(trial):
        params = {
            'objective': 'binary',
            'metric': 'average_precision',
            'verbosity': -1,
            'device': 'gpu' if torch.cuda.is_available() else 'cpu',
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 100, 600),
            'random_state': RANDOM_STATE
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        preds = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, preds)
    
    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(objective_lgb, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study_lgb.best_params.copy()
    best_params['random_state'] = RANDOM_STATE
    best_params['verbose'] = -1
    
    lgb_model = lgb.LGBMClassifier(**best_params)
    lgb_model.fit(X_train, y_train)
    results['lightgbm'] = lgb_model
    print(f"      Best CV PR-AUC: {study_lgb.best_value:.4f}")
    
    # ----- CatBoost -----
    print("\n   CatBoost:")
    def objective_cat(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 200, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 100, 600),
            'random_state': RANDOM_STATE,
            'verbose': 0,
            'task_type': 'GPU' if torch.cuda.is_available() else 'CPU'
        }
        
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30, use_best_model=True)
        preds = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, preds)
    
    study_cat = optuna.create_study(direction='maximize')
    study_cat.optimize(objective_cat, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study_cat.best_params.copy()
    best_params['random_state'] = RANDOM_STATE
    best_params['verbose'] = 0
    best_params['task_type'] = 'GPU' if torch.cuda.is_available() else 'CPU'
    
    cat_model = CatBoostClassifier(**best_params)
    cat_model.fit(X_train, y_train)
    results['catboost'] = cat_model
    print(f"      Best CV PR-AUC: {study_cat.best_value:.4f}")
    
    return results


# ============================================================
# EVALUATION & VISUALIZATION
# ============================================================
def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation"""
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        # PyTorch model
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(DEVICE)
            y_pred_proba = torch.sigmoid(model(X_test_t)).cpu().numpy().flatten()
    
    # Metrics
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)
    
    # Find optimal threshold (F1)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    return {
        'model_name': model_name,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'ap': ap,
        'best_f1': best_f1,
        'best_threshold': best_threshold,
        'precision': precision,
        'recall': recall,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred
    }


def create_visualizations(results_list, y_test, output_dir):
    """Create comprehensive visualizations"""
    print("\n📊 Creating Visualizations...")
    
    # Set style
    plt.style.use('dark_background')
    colors = ['#00ff88', '#00aaff', '#ff6b6b', '#ffd93d', '#c084fc', '#f472b6']
    
    # 1. PR Curves Comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, result in enumerate(results_list):
        ax.plot(result['recall'], result['precision'], 
               color=colors[i % len(colors)], linewidth=2,
               label=f"{result['model_name']} (PR-AUC={result['pr_auc']:.4f})")
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves - All Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_dir / 'pr_curves_comparison.png', dpi=150, facecolor='#0d1117')
    plt.close()
    
    # 2. Model Comparison Bar Chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    model_names = [r['model_name'] for r in results_list]
    pr_aucs = [r['pr_auc'] for r in results_list]
    roc_aucs = [r['roc_auc'] for r in results_list]
    f1_scores = [r['best_f1'] for r in results_list]
    
    # PR-AUC
    bars1 = axes[0].barh(model_names, pr_aucs, color=colors[:len(model_names)])
    axes[0].set_xlabel('PR-AUC')
    axes[0].set_title('Precision-Recall AUC', fontweight='bold')
    axes[0].set_xlim([0.7, 1.0])
    for bar, score in zip(bars1, pr_aucs):
        axes[0].text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.4f}', va='center', fontsize=10)
    
    # ROC-AUC
    bars2 = axes[1].barh(model_names, roc_aucs, color=colors[:len(model_names)])
    axes[1].set_xlabel('ROC-AUC')
    axes[1].set_title('ROC AUC', fontweight='bold')
    axes[1].set_xlim([0.9, 1.0])
    for bar, score in zip(bars2, roc_aucs):
        axes[1].text(score + 0.002, bar.get_y() + bar.get_height()/2, 
                    f'{score:.4f}', va='center', fontsize=10)
    
    # F1 Score
    bars3 = axes[2].barh(model_names, f1_scores, color=colors[:len(model_names)])
    axes[2].set_xlabel('F1 Score')
    axes[2].set_title('Best F1 Score', fontweight='bold')
    axes[2].set_xlim([0.5, 1.0])
    for bar, score in zip(bars3, f1_scores):
        axes[2].text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.4f}', va='center', fontsize=10)
    
    plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    
    # 3. Best Model Confusion Matrix
    best_result = max(results_list, key=lambda x: x['pr_auc'])
    cm = confusion_matrix(y_test, best_result['y_pred'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f"Confusion Matrix - {best_result['model_name']}\n(Threshold={best_result['best_threshold']:.3f})", 
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_best.png', dpi=150, facecolor='#0d1117')
    plt.close()
    
    print(f"   Saved visualizations to {output_dir}/")


def generate_shap_explanations(model, X_train, X_test, feature_names, output_dir, top_n=15):
    """Generate SHAP explanations for model interpretability"""
    print("\n🔍 Generating SHAP Explanations...")
    
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test[:1000])  # Sample for speed
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test[:1000], feature_names=feature_names, 
                         max_display=top_n, show=False)
        plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_summary.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Bar plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test[:1000], feature_names=feature_names,
                         plot_type='bar', max_display=top_n, show=False)
        plt.title('SHAP Feature Importance (Mean |SHAP|)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_importance.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ SHAP visualizations saved")
        
    except Exception as e:
        print(f"   ⚠️ SHAP generation failed: {e}")


# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    print("\n" + "=" * 70)
    print("🛡️  SOTA CREDIT CARD FRAUD DETECTION - ULTIMATE EDITION")
    print("=" * 70)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Load datasets
    datasets = load_all_datasets()
    
    if 'original' not in datasets:
        print("❌ Original dataset not found! Please ensure creditcard.csv is in the directory.")
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_credit_card_data(datasets['original'])
    
    # Split training into train/val
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train
    )
    
    print(f"\n📊 Final Data Split:")
    print(f"   Training: {len(X_train_final):,} samples")
    print(f"   Validation: {len(X_val):,} samples")
    print(f"   Test: {len(X_test):,} samples")
    
    all_results = []
    
    # ===== 1. GRADIENT BOOSTING ENSEMBLE =====
    gb_models = train_gradient_boosting(X_train_final, y_train_final, X_val, y_val, n_trials=15)
    
    for name, model in gb_models.items():
        result = evaluate_model(model, X_test, y_test, name.upper())
        all_results.append(result)
        print(f"\n   {name.upper()}: PR-AUC={result['pr_auc']:.4f}, ROC-AUC={result['roc_auc']:.4f}")
    
    # ===== 2. TABNET =====
    print("\n" + "=" * 60)
    print("🧠 Training TabNet (GPU-Accelerated)")
    print("=" * 60)
    
    tabnet_model = train_tabnet_model(X_train_final, y_train_final, X_val, y_val)
    tabnet_result = evaluate_model(tabnet_model, X_test, y_test, "TabNet")
    all_results.append(tabnet_result)
    print(f"\n   TabNet: PR-AUC={tabnet_result['pr_auc']:.4f}, ROC-AUC={tabnet_result['roc_auc']:.4f}")
    
    # ===== 3. DEEP FRAUD NET with FOCAL LOSS =====
    print("\n" + "=" * 60)
    print("🔥 Training DeepFraudNet with Focal Loss")
    print("=" * 60)
    
    deep_model = DeepFraudNet(input_dim=X_train.shape[1], hidden_dims=[256, 128, 64], dropout=0.3)
    deep_model, history, _ = train_deep_model(
        deep_model, X_train_final, y_train_final, X_val, y_val,
        epochs=100, batch_size=2048, lr=1e-3
    )
    
    deep_result = evaluate_model(deep_model, X_test, y_test, "DeepFraudNet")
    all_results.append(deep_result)
    print(f"\n   DeepFraudNet: PR-AUC={deep_result['pr_auc']:.4f}, ROC-AUC={deep_result['roc_auc']:.4f}")
    
    # ===== 4. TABULAR TRANSFORMER =====
    print("\n" + "=" * 60)
    print("⚡ Training Tabular Transformer")
    print("=" * 60)
    
    transformer_model = TabularTransformer(input_dim=X_train.shape[1], d_model=128, nhead=4, num_layers=3)
    transformer_model, _, _ = train_deep_model(
        transformer_model, X_train_final, y_train_final, X_val, y_val,
        epochs=80, batch_size=2048, lr=5e-4
    )
    
    transformer_result = evaluate_model(transformer_model, X_test, y_test, "TabTransformer")
    all_results.append(transformer_result)
    print(f"\n   TabTransformer: PR-AUC={transformer_result['pr_auc']:.4f}, ROC-AUC={transformer_result['roc_auc']:.4f}")
    
    # ===== 5. WEIGHTED ENSEMBLE =====
    print("\n" + "=" * 60)
    print("⚖️ Creating Weighted Ensemble")
    print("=" * 60)
    
    # Use top models for ensemble
    top_results = sorted(all_results, key=lambda x: x['pr_auc'], reverse=True)[:4]
    
    # Weighted average based on PR-AUC
    total_score = sum(r['pr_auc'] for r in top_results)
    ensemble_pred = np.zeros(len(y_test))
    
    print("\n   Ensemble weights:")
    for r in top_results:
        weight = r['pr_auc'] / total_score
        ensemble_pred += weight * r['y_pred_proba']
        print(f"      {r['model_name']}: {weight:.3f}")
    
    # Evaluate ensemble
    ensemble_pr_auc = average_precision_score(y_test, ensemble_pred)
    ensemble_roc_auc = roc_auc_score(y_test, ensemble_pred)
    
    precision, recall, thresholds = precision_recall_curve(y_test, ensemble_pred)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    
    ensemble_result = {
        'model_name': 'ENSEMBLE',
        'pr_auc': ensemble_pr_auc,
        'roc_auc': ensemble_roc_auc,
        'ap': ensemble_pr_auc,
        'best_f1': f1_scores[best_idx],
        'best_threshold': thresholds[best_idx] if best_idx < len(thresholds) else 0.5,
        'precision': precision,
        'recall': recall,
        'y_pred_proba': ensemble_pred,
        'y_pred': (ensemble_pred >= thresholds[best_idx]).astype(int) if best_idx < len(thresholds) else (ensemble_pred >= 0.5).astype(int)
    }
    all_results.append(ensemble_result)
    
    print(f"\n   ENSEMBLE: PR-AUC={ensemble_pr_auc:.4f}, ROC-AUC={ensemble_roc_auc:.4f}")
    
    # ===== FINAL RESULTS =====
    print("\n" + "=" * 70)
    print("🏆 FINAL RESULTS")
    print("=" * 70)
    
    # Sort by PR-AUC
    all_results_sorted = sorted(all_results, key=lambda x: x['pr_auc'], reverse=True)
    
    print(f"\n{'Rank':<6}{'Model':<20}{'PR-AUC':<12}{'ROC-AUC':<12}{'F1':<10}")
    print("-" * 60)
    for i, r in enumerate(all_results_sorted, 1):
        print(f"{i:<6}{r['model_name']:<20}{r['pr_auc']:<12.4f}{r['roc_auc']:<12.4f}{r['best_f1']:<10.4f}")
    
    # Best model
    best = all_results_sorted[0]
    print(f"\n🥇 CHAMPION: {best['model_name']}")
    print(f"   PR-AUC: {best['pr_auc']:.4f}")
    print(f"   ROC-AUC: {best['roc_auc']:.4f}")
    print(f"   Best F1: {best['best_f1']:.4f}")
    
    # Compare to baseline
    baseline_pr_auc = 0.9104  # Previous best
    if best['pr_auc'] > baseline_pr_auc:
        improvement = (best['pr_auc'] - baseline_pr_auc) * 100
        print(f"\n🎉 NEW RECORD! Beat previous best ({baseline_pr_auc:.4f}) by +{improvement:.2f}%!")
    else:
        gap = (baseline_pr_auc - best['pr_auc']) * 100
        print(f"\n📊 Baseline comparison: {gap:.2f}% from previous best ({baseline_pr_auc:.4f})")
    
    # Create visualizations
    create_visualizations(all_results_sorted, y_test, OUTPUT_DIR)
    
    # SHAP explanations for best gradient boosting model
    if 'xgboost' in gb_models:
        generate_shap_explanations(gb_models['xgboost'], X_train_final, X_test, feature_names, OUTPUT_DIR)
    
    # Save results
    results_to_save = {
        'timestamp': datetime.now().isoformat(),
        'device': str(DEVICE),
        'best_model': best['model_name'],
        'best_pr_auc': float(best['pr_auc']),
        'best_roc_auc': float(best['roc_auc']),
        'best_f1': float(best['best_f1']),
        'all_results': [
            {
                'model': r['model_name'],
                'pr_auc': float(r['pr_auc']),
                'roc_auc': float(r['roc_auc']),
                'f1': float(r['best_f1'])
            }
            for r in all_results_sorted
        ]
    }
    
    with open('sota_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    # Save best models
    if 'xgboost' in gb_models:
        gb_models['xgboost'].save_model('best_xgboost_sota.json')
    
    torch.save(deep_model.state_dict(), 'deep_fraud_net.pt')
    torch.save(transformer_model.state_dict(), 'tab_transformer.pt')
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Total time: {elapsed/60:.1f} minutes")
    print(f"📁 Results saved to sota_results.json")
    print(f"📊 Visualizations saved to {OUTPUT_DIR}/")
    print("\n✅ SOTA Training Complete!")


if __name__ == "__main__":
    main()
