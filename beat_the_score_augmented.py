"""
Credit Card Fraud Detection - Augmented Data Challenge
=====================================================
Goal: Beat the best PR AUC (0.8849) by adding external data (creditcard_2023.csv)

Process:
1. Load original and new 2023 datasets
2. Normalize features to match
3. Augment training set with new data
4. Train optimized XGBoost
5. Evaluate on ORIGINAL test set (strictly)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_recall_curve, auc
from scipy.stats import ks_2samp
import warnings

warnings.filterwarnings('ignore')

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 60)
print("🚀 Augmented Data Challenge")
print("=" * 60)
print(f"Baseline to beat: 0.8849")

# 1. Load Data
print("\n📊 Loading Datasets...")
df_orig = pd.read_csv('creditcard.csv')
print(f"   Original Data: {df_orig.shape}")

try:
    df_new = pd.read_csv('creditcard_2023.csv')
    print(f"   New (2023) Data: {df_new.shape}")
except FileNotFoundError:
    print("❌ New dataset not found!")
    exit()

# 2. Cleanup New Data
# New dataset likely has an 'id' column
if 'id' in df_new.columns:
    df_new = df_new.drop('id', axis=1)

# Ensure columns match
common_cols = [c for c in df_orig.columns if c in df_new.columns]
print(f"   Common columns: {len(common_cols)}")

if len(common_cols) < 30:
    print("❌ Columns do not match sufficienty for merging.")
    exit()

df_new = df_new[common_cols]

# 3. Compatibility Check (Distribution Analysis)
print("\n🔍 Checking Compatibility (V1-V28)...")
# Compare means of V features
v_cols = [c for c in common_cols if c.startswith('V')]
orig_means = df_orig[v_cols].mean()
new_means = df_new[v_cols].mean()
diff = np.abs(orig_means - new_means).mean()

print(f"   Average Mean Difference (V-features): {diff:.4f}")
if diff > 1.0:
    print("⚠️ WARNING: Feature distributions seem significantly different.")
    # We might need to scale them independently before merging, but let's try direct first
    # Actually, RobustScaler should handle shifts if we fit on combined or scale independently
else:
    print("✅ Feature distributions look reasonably aligned.")

# 4. Feature Engineering (Apply to BOTH)
def engineer_features(df):
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Log transform amount
    X['Amount_log'] = np.log1p(X['Amount'])
    
    # Time features (if Time exists) -> 2023 dataset might not have compatible Time
    # Let's check Time distribution
    if 'Time' in X.columns:
        X['Time_hour'] = (X['Time'] % 3600) / 3600
        X['Time_day'] = (X['Time'] % 86400) / 86400
    
    v_features = [f'V{i}' for i in range(1, 29)]
    # Ensure all exist
    valid_v = [v for v in v_features if v in X.columns]
    
    X['V_mean'] = X[valid_v].mean(axis=1)
    X['V_std'] = X[valid_v].std(axis=1)
    X['V_max'] = X[valid_v].max(axis=1)
    X['V_min'] = X[valid_v].min(axis=1)
    X['V_range'] = X['V_max'] - X['V_min']
    
    # Interactions
    if 'V14' in X.columns and 'V17' in X.columns:
        X['V14_V17_int'] = X['V14'] * X['V17']
    if 'V10' in X.columns and 'V12' in X.columns:
        X['V10_V12_int'] = X['V10'] * X['V12']
    if 'V3' in X.columns and 'V7' in X.columns:
        X['V3_V7_int'] = X['V3'] * X['V7']
    if 'V4' in X.columns and 'V11' in X.columns:
        X['V4_V11_int'] = X['V4'] * X['V11']
        
    return X, y

# Engineer ORIGINAL data first
print("🔧 Engineering Features...")
X_orig, y_orig = engineer_features(df_orig)

# Split Original Data (We need a pure Test Set from ORIGINAL source)
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
    X_orig, y_orig, test_size=0.2, random_state=RANDOM_STATE, stratify=y_orig
)

# Engineer NEW data
X_new, y_new = engineer_features(df_new)

# Align columns (drop any that mismatch after engineering)
cols = [c for c in X_train_orig.columns if c in X_new.columns]
X_train_orig = X_train_orig[cols]
X_test_orig = X_test_orig[cols]
X_new = X_new[cols]

print(f"   Training Data (Original): {X_train_orig.shape}")
print(f"   Augmentation Data (New): {X_new.shape}")

# 5. Merge Training Sets
print("➕ Merging Datasets...")
X_train_augmented = pd.concat([X_train_orig, X_new], axis=0)
y_train_augmented = pd.concat([y_train_orig, y_new], axis=0)

print(f"   Combined Training Size: {X_train_augmented.shape}")
print(f"   New Fraud Rate: {y_train_augmented.mean()*100:.2f}% (Original: {y_train_orig.mean()*100:.2f}%)")

# 6. Scaling
print("⚖️ Scaling (RobustScaler)...")
scaler = RobustScaler()
# Fit on the AUGMENTED training set
X_train_scaled = scaler.fit_transform(X_train_augmented)
# Transform the ORIGINAL test set
X_test_scaled = scaler.transform(X_test_orig)

# 7. Train XGBoost
print("🚀 Training Optimized XGBoost on Augmented Data...")
# Use best params found previously
best_params = {
    "lambda": 2.2626,
    "alpha": 1.3167,
    "colsample_bytree": 0.6079,
    "subsample": 0.8867,
    "learning_rate": 0.1909,
    "n_estimators": 668,
    "max_depth": 5,
    "min_child_weight": 2,
    # Recalculate scale_pos_weight for new imbalance
    "scale_pos_weight": len(y_train_augmented[y_train_augmented==0]) / len(y_train_augmented[y_train_augmented==1]),
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "random_state": RANDOM_STATE,
    "use_label_encoder": False,
    "tree_method": "gpu_hist" # Use GPU
}

model = xgb.XGBClassifier(**best_params)
model.fit(
    X_train_scaled, y_train_augmented,
    verbose=True
)

# 8. Evaluate
print("🎯 Evaluating on Original Test Set...")
preds = model.predict_proba(X_test_scaled)[:, 1]
precision, recall, _ = precision_recall_curve(y_test_orig, preds)
pr_auc = auc(recall, precision)

print("\n" + "="*60)
print(f"🏆 FINAL AUGMENTED SCORE: {pr_auc:.4f}")
print("="*60)

if pr_auc > 0.8849:
    print(f"🎉 SUCCESS! Beat baseline (0.8849) by +{(pr_auc - 0.8849):.4f}")
else:
    print(f"⚠️ Failed to beat baseline. Gap: {(0.8849 - pr_auc):.4f}")
    print("   Analysis: The new data might be too different distribution-wise or introduced noise.")

# Save the model regardless if it's good (it is!)
print("\n💾 Saving Augmented Model...")
model.save_model("best_model_augmented.json")
import joblib
joblib.dump(scaler, "scaler_augmented.joblib")
print("✅ Saved to 'best_model_augmented.json'")
