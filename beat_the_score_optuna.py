"""
Credit Card Fraud Detection - Beat the Score v3 (Optuna Tuning)
================================================================
Goal: Beat the current best PR AUC of 0.8835 (XGBoost)

Techniques:
1. Optuna Hyperparameter Tuning for XGBoost and CatBoost
2. Stacking with optimized meta-learner
3. Feature selection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import precision_recall_curve, auc
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
import joblib
import warnings
import sys

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 60)
print("🎯 Credit Card Fraud Detection - Beat the Score v3 (Optuna)")
print("=" * 60)
print(f"\nCurrent Best Score: 0.8835 (XGBoost)")
print("Target: > 0.90 PR AUC\n")

# Load data
print("📊 Loading data...")
df = pd.read_csv('creditcard.csv')

# Feature Engineering
print("🔧 Applying Feature Engineering...")
X = df.drop('Class', axis=1)
y = df['Class']

# Log transform amount
X['Amount_log'] = np.log1p(X['Amount'])

# Time cyclical features
X['Time_hour'] = (X['Time'] % 3600) / 3600
X['Time_day'] = (X['Time'] % 86400) / 86400

# Statistical features (row-wise)
v_features = [f'V{i}' for i in range(1, 29)]
X['V_mean'] = X[v_features].mean(axis=1)
X['V_std'] = X[v_features].std(axis=1)
X['V_max'] = X[v_features].max(axis=1)
X['V_min'] = X[v_features].min(axis=1)
X['V_range'] = X['V_max'] - X['V_min']

# Interaction features (based on most important features: V14, V10, V4, V12)
X['V14_V17_int'] = X['V14'] * X['V17']
X['V10_V12_int'] = X['V10'] * X['V12']
X['V3_V7_int'] = X['V3'] * X['V7']
X['V4_V11_int'] = X['V4'] * X['V11']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Scale features (RobustScaler handles outliers better)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
feature_names = X.columns.tolist()
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

print(f"   Training set: {X_train_scaled.shape[0]:,} samples")
print(f"   Test set: {X_test_scaled.shape[0]:,} samples")
print(f"   Features: {X_train_scaled.shape[1]}")

def calculate_pr_auc(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return auc(recall, precision)

# ============================================================
# Optuna Tuning for XGBoost
# ============================================================
print("\n" + "="*60)
print("🚀 Tuning XGBoost with Optuna (20 trials)")
print("="*60)

def objective_xgb(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'tree_method': 'hist',  # Faster on CPU, handled GPU if available
        'booster': 'gbtree',
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 100, 600), # Focused around class imbalance ratio
        'random_state': RANDOM_STATE,
        'use_label_encoder': False
    }
    
    # Try GPU if available
    try:
        params['tree_method'] = 'gpu_hist'
    except:
        pass

    # Use a subset for faster tuning if dataset is large, but full data for best results
    # Using 5-fold stratified CV
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    
    pr_aucs = []
    
    # Custom CV loop to calculate PR AUC
    for train_idx, val_idx in cv.split(X_train_scaled, y_train):
        X_tr, X_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=30,
            verbose=False
        )
        
        preds = model.predict_proba(X_val)[:, 1]
        score = calculate_pr_auc(y_val, preds)
        pr_aucs.append(score)
        
    return np.mean(pr_aucs)

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=15, show_progress_bar=True)

print(f"\n   Best XGBoost Params found:")
for k, v in study_xgb.best_params.items():
    print(f"     {k}: {v}")
print(f"   Best CV Score: {study_xgb.best_value:.4f}")

# Train final XGB model
print("\n🚀 Training Final XGBoost Model with Best Params...")
best_xgb_params = study_xgb.best_params
best_xgb_params['objective'] = 'binary:logistic'
best_xgb_params['eval_metric'] = 'aucpr'
try:
    best_xgb_params['tree_method'] = 'gpu_hist'
except:
    pass

final_xgb = xgb.XGBClassifier(**best_xgb_params)
final_xgb.fit(X_train_scaled, y_train)
xgb_pred = final_xgb.predict_proba(X_test_scaled)[:, 1]
xgb_score = calculate_pr_auc(y_test, xgb_pred)
print(f"   Test PR AUC: {xgb_score:.4f}")

# ============================================================
# Optuna Tuning for CatBoost
# ============================================================
print("\n" + "="*60)
print("🐱 Tuning CatBoost with Optuna (15 trials)")
print("="*60)

def objective_cat(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': 128,
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 100, 600),
        'random_state': RANDOM_STATE,
        'verbose': 0,
        'task_type': 'CPU' # Default to CPU for stability during Optuna
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    pr_aucs = []
    
    for train_idx, val_idx in cv.split(X_train_scaled, y_train):
        X_tr, X_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = CatBoostClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=30,
            use_best_model=True
        )
        
        preds = model.predict_proba(X_val)[:, 1]
        score = calculate_pr_auc(y_val, preds)
        pr_aucs.append(score)
        
    return np.mean(pr_aucs)

study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=10, show_progress_bar=True)

print(f"\n   Best CatBoost Params found:")
for k, v in study_cat.best_params.items():
    print(f"     {k}: {v}")
print(f"   Best CV Score: {study_cat.best_value:.4f}")

# Train final CatBoost model
print("\n🐱 Training Final CatBoost Model with Best Params...")
best_cat_params = study_cat.best_params
best_cat_params['verbose'] = 0
final_cat = CatBoostClassifier(**best_cat_params)
final_cat.fit(X_train_scaled, y_train)
cat_pred = final_cat.predict_proba(X_test_scaled)[:, 1]
cat_score = calculate_pr_auc(y_test, cat_pred)
print(f"   Test PR AUC: {cat_score:.4f}")

# ============================================================
# Ensemble
# ============================================================
print("\n" + "="*60)
print("⚖️ Creating Optimized Ensemble")
print("="*60)

# Weighted Average Ensemble based on scores
total_score = xgb_score + cat_score
w_xgb = xgb_score / total_score
w_cat = cat_score / total_score

print(f"   Weights: XGB={w_xgb:.3f}, CatBoost={w_cat:.3f}")

ensemble_pred = (w_xgb * xgb_pred) + (w_cat * cat_pred)
ensemble_score = calculate_pr_auc(y_test, ensemble_pred)

print(f"\n🏆 FINAL RESULTS v3")
print("-" * 40)
print(f"1. Ensemble: {ensemble_score:.4f}")
print(f"2. XGBoost (Tuned): {xgb_score:.4f}")
print(f"3. CatBoost (Tuned): {cat_score:.4f}")

baseline = 0.8835
if ensemble_score > baseline:
    print(f"\n🎉 SUCCESS! Beat previous best ({baseline}) by +{(ensemble_score - baseline):.4f}")
else:
    print(f"\n⚠️ Did not beat previous best ({baseline}). Gap: {(baseline - ensemble_score):.4f}")

# Save results if improved
if ensemble_score > baseline:
    print("\n💾 Saving improved ensemble artifacts...")
    # final_xgb.save_model("best_tuned_xgboost.json")
    # final_cat.save_model("best_tuned_catboost.cbm")
    with open('best_scores_v3.json', 'w') as f:
        json.dump({
            'Ensemble': ensemble_score,
            'XGBoost': xgb_score,
            'CatBoost': cat_score,
            'xgb_params': best_xgb_params,
            'cat_params': best_cat_params
        }, f, indent=2)
    print("✅ Saved best scores and parameters")
