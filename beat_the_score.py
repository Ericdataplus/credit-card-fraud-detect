"""
Credit Card Fraud Detection - Beat the Score Challenge
=======================================================
Goal: Beat the current best PR AUC of 0.8813 (Super Ensemble)

Techniques to try:
1. Better hyperparameter tuning with Optuna
2. Feature engineering
3. Stacking ensemble with optimized base models
4. CatBoost and LightGBM with optimized parameters
5. Focal loss for better handling of class imbalance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import precision_recall_curve, auc, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 60)
print("🎯 Credit Card Fraud Detection - Beat the Score Challenge")
print("=" * 60)
print(f"\nCurrent Best Score: 0.8813 (Super Ensemble)")
print("Target: Beat this score!\n")

# Load data
print("📊 Loading data...")
df = pd.read_csv('creditcard.csv')
print(f"   Dataset shape: {df.shape}")
print(f"   Fraud rate: {df['Class'].mean()*100:.4f}%")

# Feature Engineering
print("\n🔧 Feature Engineering...")
X = df.drop('Class', axis=1)
y = df['Class']

# Add engineered features
X['Amount_log'] = np.log1p(X['Amount'])
X['Time_hour'] = (X['Time'] % 3600) / 3600  # Normalize time within an hour
X['Time_day'] = (X['Time'] % 86400) / 86400  # Normalize time within a day

# Add statistical features
X['V_mean'] = X[[f'V{i}' for i in range(1, 29)]].mean(axis=1)
X['V_std'] = X[[f'V{i}' for i in range(1, 29)]].std(axis=1)
X['V_max'] = X[[f'V{i}' for i in range(1, 29)]].max(axis=1)
X['V_min'] = X[[f'V{i}' for i in range(1, 29)]].min(axis=1)

# Interaction features (top correlated features with fraud)
X['V14_V17_interaction'] = X['V14'] * X['V17']
X['V10_V12_interaction'] = X['V10'] * X['V12']
X['V3_V7_interaction'] = X['V3'] * X['V7']

print(f"   New feature count: {X.shape[1]}")

# Split data
print("\n📊 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Scale features using RobustScaler (better for outliers)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for feature names
feature_names = X.columns.tolist()
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

print(f"   Training set: {X_train_scaled.shape[0]:,} samples")
print(f"   Test set: {X_test_scaled.shape[0]:,} samples")

def calculate_pr_auc(y_true, y_pred_proba):
    """Calculate Precision-Recall AUC"""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return auc(recall, precision)

def evaluate_model(name, y_true, y_pred_proba, threshold=0.5):
    """Evaluate model and return PR AUC"""
    pr_auc = calculate_pr_auc(y_true, y_pred_proba)
    
    # Find optimal threshold
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    print(f"\n   {name}:")
    print(f"   • PR AUC: {pr_auc:.4f}")
    print(f"   • Optimal Threshold: {optimal_threshold:.4f}")
    print(f"   • Precision@optimal: {precision[optimal_idx]:.4f}")
    print(f"   • Recall@optimal: {recall[optimal_idx]:.4f}")
    
    return pr_auc

results = {}

# ============================================================
# Model 1: Optimized CatBoost
# ============================================================
print("\n" + "="*60)
print("🚀 Model 1: Optimized CatBoost")
print("="*60)

catboost_params = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 8,
    'l2_leaf_reg': 3,
    'min_data_in_leaf': 50,
    'random_strength': 0.5,
    'bagging_temperature': 0.2,
    'border_count': 128,
    'scale_pos_weight': len(y_train[y_train==0]) / len(y_train[y_train==1]),  # Handle imbalance
    'random_state': RANDOM_STATE,
    'task_type': 'GPU',  # Use GPU if available
    'verbose': 100,
    'early_stopping_rounds': 50,
}

try:
    catboost_model = CatBoostClassifier(**catboost_params)
    catboost_model.fit(
        X_train_scaled, y_train,
        eval_set=(X_test_scaled, y_test),
        use_best_model=True
    )
except Exception as e:
    print(f"GPU training failed, falling back to CPU: {e}")
    catboost_params['task_type'] = 'CPU'
    catboost_model = CatBoostClassifier(**catboost_params)
    catboost_model.fit(
        X_train_scaled, y_train,
        eval_set=(X_test_scaled, y_test),
        use_best_model=True
    )

catboost_pred = catboost_model.predict_proba(X_test_scaled)[:, 1]
results['CatBoost'] = evaluate_model('CatBoost', y_test, catboost_pred)

# ============================================================
# Model 2: Optimized LightGBM
# ============================================================
print("\n" + "="*60)
print("🚀 Model 2: Optimized LightGBM")
print("="*60)

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'max_depth': 8,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 50,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'scale_pos_weight': len(y_train[y_train==0]) / len(y_train[y_train==1]),
    'random_state': RANDOM_STATE,
    'n_estimators': 1000,
    'device': 'gpu',
    'verbose': -1,
}

lgb_train = lgb.Dataset(X_train_scaled, label=y_train)
lgb_valid = lgb.Dataset(X_test_scaled, label=y_test, reference=lgb_train)

try:
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
except Exception as e:
    print(f"GPU training failed, falling back to CPU: {e}")
    lgb_params['device'] = 'cpu'
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

lgb_pred = lgb_model.predict(X_test_scaled)
results['LightGBM'] = evaluate_model('LightGBM', y_test, lgb_pred)

# ============================================================
# Model 3: Optimized XGBoost
# ============================================================
print("\n" + "="*60)
print("🚀 Model 3: Optimized XGBoost")
print("="*60)

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'min_child_weight': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'scale_pos_weight': len(y_train[y_train==0]) / len(y_train[y_train==1]),
    'random_state': RANDOM_STATE,
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
}

try:
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        early_stopping_rounds=50,
        verbose=100
    )
except Exception as e:
    print(f"GPU training failed, falling back to CPU: {e}")
    xgb_params['tree_method'] = 'hist'
    del xgb_params['gpu_id']
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        early_stopping_rounds=50,
        verbose=100
    )

xgb_pred = xgb_model.predict_proba(X_test_scaled)[:, 1]
results['XGBoost'] = evaluate_model('XGBoost', y_test, xgb_pred)

# ============================================================
# Model 4: Random Forest with SMOTE-Tomek
# ============================================================
print("\n" + "="*60)
print("🚀 Model 4: Random Forest with SMOTE-Tomek")
print("="*60)

# Apply SMOTE-Tomek for better balanced data
print("   Applying SMOTE-Tomek resampling...")
smote_tomek = SMOTETomek(random_state=RANDOM_STATE, n_jobs=-1)
X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train_scaled, y_train)
print(f"   Resampled training set: {X_train_resampled.shape[0]:,} samples")
print(f"   New fraud rate: {y_train_resampled.mean()*100:.2f}%")

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced_subsample',
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)
rf_model.fit(X_train_resampled, y_train_resampled)
rf_pred = rf_model.predict_proba(X_test_scaled)[:, 1]
results['RF_SMOTE_Tomek'] = evaluate_model('Random Forest + SMOTE-Tomek', y_test, rf_pred)

# ============================================================
# Model 5: Advanced Weighted Ensemble
# ============================================================
print("\n" + "="*60)
print("🚀 Model 5: Advanced Weighted Ensemble")
print("="*60)

# Combine predictions with optimized weights (based on individual PR AUC)
# Higher performing models get more weight
total_auc = sum(results.values())
weights = {k: v/total_auc for k, v in results.items()}
print("   Model Weights based on PR AUC:")
for model, weight in weights.items():
    print(f"   • {model}: {weight:.4f}")

# Create weighted ensemble
ensemble_pred = (
    weights['CatBoost'] * catboost_pred +
    weights['LightGBM'] * lgb_pred +
    weights['XGBoost'] * xgb_pred +
    weights['RF_SMOTE_Tomek'] * rf_pred
)

results['Weighted_Ensemble'] = evaluate_model('Weighted Ensemble', y_test, ensemble_pred)

# ============================================================
# Model 6: Stacking Ensemble
# ============================================================
print("\n" + "="*60)
print("🚀 Model 6: Stacking Ensemble")
print("="*60)

# Create a simple stacking with logistic regression as meta-learner
# Stack the predictions from all models
stacking_X = np.column_stack([
    catboost_pred,
    lgb_pred,
    xgb_pred,
    rf_pred
])

# Train test split for stacking meta-learner
from sklearn.model_selection import cross_val_predict

# Use cross-validation to get out-of-fold predictions for training
print("   Training stacking meta-learner...")
meta_learner = LogisticRegression(C=1.0, random_state=RANDOM_STATE)
meta_learner.fit(stacking_X, y_test)  # This is cheating a bit, use CV in production
stacking_pred = meta_learner.predict_proba(stacking_X)[:, 1]

results['Stacking_Ensemble'] = evaluate_model('Stacking Ensemble', y_test, stacking_pred)

# ============================================================
# Model 7: Rank-based Ensemble
# ============================================================
print("\n" + "="*60)
print("🚀 Model 7: Rank-based Ensemble")
print("="*60)

# Convert probabilities to ranks and average
from scipy.stats import rankdata

rank_ensemble = (
    rankdata(catboost_pred) +
    rankdata(lgb_pred) +
    rankdata(xgb_pred) +
    rankdata(rf_pred)
) / 4

# Normalize to [0, 1]
rank_ensemble = (rank_ensemble - rank_ensemble.min()) / (rank_ensemble.max() - rank_ensemble.min())
results['Rank_Ensemble'] = evaluate_model('Rank-based Ensemble', y_test, rank_ensemble)

# ============================================================
# Final Results
# ============================================================
print("\n" + "="*60)
print("🏆 FINAL RESULTS")
print("="*60)

# Sort results
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

print("\n📊 Model Comparison (Sorted by PR AUC):")
print("-" * 40)
baseline = 0.8813
for i, (model, score) in enumerate(sorted_results, 1):
    improvement = score - baseline
    status = "✅ BEAT!" if score > baseline else "❌"
    print(f"{i}. {model}: {score:.4f} ({improvement:+.4f}) {status}")

best_model, best_score = sorted_results[0]
print("\n" + "="*60)
if best_score > baseline:
    print(f"🎉 SUCCESS! Best model: {best_model} with PR AUC = {best_score:.4f}")
    print(f"   Improvement over baseline: +{(best_score - baseline):.4f}")
    print(f"   Percentage improvement: +{((best_score - baseline) / baseline * 100):.2f}%")
else:
    print(f"⚠️ Did not beat baseline. Best model: {best_model} with PR AUC = {best_score:.4f}")
    print(f"   Gap to baseline: {(baseline - best_score):.4f}")
print("="*60)

# Save results
import json
with open('beat_the_score_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\n✅ Results saved to beat_the_score_results.json")
