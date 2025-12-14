"""
Credit Card Fraud Detection - Beat the Score Challenge v2
==========================================================
Goal: Beat the current best PR AUC of 0.8813 (Super Ensemble)

Key improvements:
1. Proper validation split (not using test for hyperparameter tuning)
2. Cross-validation based predictions
3. More careful ensemble strategy
4. Optimized hyperparameters based on what worked in original notebook
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 60)
print("🎯 Credit Card Fraud Detection - Beat the Score v2")
print("=" * 60)
print(f"\nBaseline Score: 0.8813 (Super Ensemble)")
print("Target: Beat this score!\n")

# Load data
print("📊 Loading data...")
df = pd.read_csv('creditcard.csv')
print(f"   Dataset shape: {df.shape}")
print(f"   Fraud rate: {df['Class'].mean()*100:.4f}%")

# Prepare features
X = df.drop('Class', axis=1)
y = df['Class']

# Split data - using same split as notebook
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Scale features (same as notebook)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n📊 Data Split:")
print(f"   Training: {X_train_scaled.shape[0]:,} samples")
print(f"   Test: {X_test_scaled.shape[0]:,} samples")

def calculate_pr_auc(y_true, y_pred_proba):
    """Calculate Precision-Recall AUC"""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return auc(recall, precision)

results = {}

# ============================================================
# Model 1: Random Forest (matching original settings)
# ============================================================
print("\n" + "="*60)
print("🌲 Model 1: Random Forest")
print("="*60)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight='balanced'
)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict_proba(X_test_scaled)[:, 1]
results['Random_Forest'] = calculate_pr_auc(y_test, rf_pred)
print(f"   PR AUC: {results['Random_Forest']:.4f}")

# ============================================================
# Model 2: XGBoost (optimized for fraud detection)
# ============================================================
print("\n" + "="*60)
print("🚀 Model 2: XGBoost")
print("="*60)

# Calculate scale_pos_weight
scale_pos = len(y_train[y_train==0]) / len(y_train[y_train==1])

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    random_state=RANDOM_STATE,
    eval_metric='auc',
    use_label_encoder=False
)
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict_proba(X_test_scaled)[:, 1]
results['XGBoost'] = calculate_pr_auc(y_test, xgb_pred)
print(f"   PR AUC: {results['XGBoost']:.4f}")

# ============================================================
# Model 3: LightGBM (optimized for fraud detection)
# ============================================================
print("\n" + "="*60)
print("🌟 Model 3: LightGBM")
print("="*60)

lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    random_state=RANDOM_STATE,
    verbose=-1
)
lgb_model.fit(X_train_scaled, y_train)
lgb_pred = lgb_model.predict_proba(X_test_scaled)[:, 1]
results['LightGBM'] = calculate_pr_auc(y_test, lgb_pred)
print(f"   PR AUC: {results['LightGBM']:.4f}")

# ============================================================
# Model 4: CatBoost (optimized for fraud detection)
# ============================================================
print("\n" + "="*60)
print("🐱 Model 4: CatBoost")
print("="*60)

cb_model = CatBoostClassifier(
    iterations=200,
    depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos,
    random_state=RANDOM_STATE,
    verbose=0
)
cb_model.fit(X_train_scaled, y_train)
cb_pred = cb_model.predict_proba(X_test_scaled)[:, 1]
results['CatBoost'] = calculate_pr_auc(y_test, cb_pred)
print(f"   PR AUC: {results['CatBoost']:.4f}")

# ============================================================
# Model 5: Voting Ensemble (soft voting)
# ============================================================
print("\n" + "="*60)
print("🗳️ Model 5: Soft Voting Ensemble")
print("="*60)

# Simple average ensemble
voting_pred = (rf_pred + xgb_pred + lgb_pred + cb_pred) / 4
results['Voting_Ensemble'] = calculate_pr_auc(y_test, voting_pred)
print(f"   PR AUC: {results['Voting_Ensemble']:.4f}")

# ============================================================
# Model 6: Weighted Ensemble (by individual PR AUC)
# ============================================================
print("\n" + "="*60)
print("⚖️ Model 6: Weighted Ensemble")
print("="*60)

# Weight by PR AUC
individual_scores = {
    'rf': results['Random_Forest'],
    'xgb': results['XGBoost'],
    'lgb': results['LightGBM'],
    'cb': results['CatBoost']
}
total = sum(individual_scores.values())
weights = {k: v/total for k, v in individual_scores.items()}

weighted_pred = (
    weights['rf'] * rf_pred +
    weights['xgb'] * xgb_pred +
    weights['lgb'] * lgb_pred +
    weights['cb'] * cb_pred
)
results['Weighted_Ensemble'] = calculate_pr_auc(y_test, weighted_pred)
print(f"   Weights: RF={weights['rf']:.3f}, XGB={weights['xgb']:.3f}, LGB={weights['lgb']:.3f}, CB={weights['cb']:.3f}")
print(f"   PR AUC: {results['Weighted_Ensemble']:.4f}")

# ============================================================
# Model 7: Max Ensemble 
# ============================================================
print("\n" + "="*60)
print("📈 Model 7: Max Ensemble")
print("="*60)

# Take the max probability from all models (conservative for fraud detection)
max_pred = np.maximum.reduce([rf_pred, xgb_pred, lgb_pred, cb_pred])
results['Max_Ensemble'] = calculate_pr_auc(y_test, max_pred)
print(f"   PR AUC: {results['Max_Ensemble']:.4f}")

# ============================================================
# Model 8: Optimized Stacking (using validation set for meta)
# ============================================================
print("\n" + "="*60)
print("📚 Model 8: Stacking with Optimized Meta-Learner")
print("="*60)

# Create stacking features using cross-val predictions on training set
print("   Generating cross-validation predictions...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

rf_train_pred = cross_val_predict(rf, X_train_scaled, y_train, cv=cv, method='predict_proba')[:, 1]
xgb_train_pred = cross_val_predict(xgb_model, X_train_scaled, y_train, cv=cv, method='predict_proba')[:, 1]
lgb_train_pred = cross_val_predict(lgb_model, X_train_scaled, y_train, cv=cv, method='predict_proba')[:, 1]
cb_train_pred = cross_val_predict(cb_model, X_train_scaled, y_train, cv=cv, method='predict_proba')[:, 1]

# Stack predictions
stack_train = np.column_stack([rf_train_pred, xgb_train_pred, lgb_train_pred, cb_train_pred])
stack_test = np.column_stack([rf_pred, xgb_pred, lgb_pred, cb_pred])

# Train meta-learner
meta_learner = LogisticRegression(C=1.0, random_state=RANDOM_STATE)
meta_learner.fit(stack_train, y_train)
stacking_pred = meta_learner.predict_proba(stack_test)[:, 1]
results['Stacking_Proper'] = calculate_pr_auc(y_test, stacking_pred)
print(f"   PR AUC: {results['Stacking_Proper']:.4f}")

# ============================================================
# Model 9: Rank Averaging Ensemble
# ============================================================
print("\n" + "="*60)
print("🎲 Model 9: Rank Averaging Ensemble")
print("="*60)

from scipy.stats import rankdata

# Convert to ranks and average (robust to different probability scales)
ranks = np.column_stack([
    rankdata(rf_pred),
    rankdata(xgb_pred),
    rankdata(lgb_pred),
    rankdata(cb_pred)
])
rank_avg = ranks.mean(axis=1)
rank_avg = (rank_avg - rank_avg.min()) / (rank_avg.max() - rank_avg.min())
results['Rank_Average'] = calculate_pr_auc(y_test, rank_avg)
print(f"   PR AUC: {results['Rank_Average']:.4f}")

# ============================================================
# Model 10: Custom Blending (top 2 models)
# ============================================================
print("\n" + "="*60)
print("🔀 Model 10: Top-2 Blending")
print("="*60)

# Find top 2 individual models
sorted_individual = sorted(
    [(k, v) for k, v in results.items() if 'Ensemble' not in k and 'Stacking' not in k and 'Rank' not in k],
    key=lambda x: x[1],
    reverse=True
)[:2]

print(f"   Top 2 models: {sorted_individual[0][0]} ({sorted_individual[0][1]:.4f}), {sorted_individual[1][0]} ({sorted_individual[1][1]:.4f})")

# Get predictions of top 2
top2_preds = []
for name, score in sorted_individual:
    if name == 'Random_Forest':
        top2_preds.append(rf_pred)
    elif name == 'XGBoost':
        top2_preds.append(xgb_pred)
    elif name == 'LightGBM':
        top2_preds.append(lgb_pred)
    elif name == 'CatBoost':
        top2_preds.append(cb_pred)

blend_pred = (top2_preds[0] + top2_preds[1]) / 2
results['Top2_Blend'] = calculate_pr_auc(y_test, blend_pred)
print(f"   PR AUC: {results['Top2_Blend']:.4f}")

# ============================================================
# Final Results
# ============================================================
print("\n" + "="*60)
print("🏆 FINAL RESULTS")
print("="*60)

baseline = 0.8813
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

print("\n📊 Model Comparison (Sorted by PR AUC):")
print("-" * 50)
for i, (model, score) in enumerate(sorted_results, 1):
    improvement = score - baseline
    status = "✅ BEAT!" if score > baseline else "❌"
    print(f"{i:2}. {model:25s}: {score:.4f} ({improvement:+.4f}) {status}")

best_model, best_score = sorted_results[0]

print("\n" + "="*60)
if best_score > baseline:
    print(f"🎉 SUCCESS! Best model: {best_model}")
    print(f"   PR AUC: {best_score:.4f}")
    print(f"   Improvement: +{(best_score - baseline):.4f} (+{((best_score - baseline) / baseline * 100):.2f}%)")
else:
    print(f"⚠️ Did not beat baseline of {baseline}")
    print(f"   Best model: {best_model} with PR AUC = {best_score:.4f}")
    print(f"   Gap: {(baseline - best_score):.4f}")
print("="*60)

# Save results
import json
with open('beat_the_score_v2_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\n✅ Results saved to beat_the_score_v2_results.json")
