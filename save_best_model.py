import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc
import joblib
import json

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 60)
print("💾 Saving Winning XGBoost Model")
print("=" * 60)

# Load data
print("📊 Loading data...")
df = pd.read_csv('creditcard.csv')

# Feature Engineering (matching beat_the_score_optuna_v4.py)
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

# Interaction features
X['V14_V17_int'] = X['V14'] * X['V17']
X['V10_V12_int'] = X['V10'] * X['V12']
X['V3_V7_int'] = X['V3'] * X['V7']
X['V4_V11_int'] = X['V4'] * X['V11']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Scale features (RobustScaler)
print("⚖️ Scaling features (RobustScaler)...")
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
feature_names = X.columns.tolist()
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

# Calculate scale_pos_weight
scale_pos = len(y_train[y_train==0]) / len(y_train[y_train==1])

print("🚀 Retraining best XGBoost model with Optuna-tuned parameters...")
# Optimal parameters from beat_the_score_optuna_v4.py
best_params = {
    "lambda": 2.2626345523225444,
    "alpha": 1.3167177928582023,
    "colsample_bytree": 0.6078554911096029,
    "subsample": 0.8866734792574604,
    "learning_rate": 0.19092746577382527,
    "n_estimators": 668,
    "max_depth": 5,
    "min_child_weight": 2,
    "scale_pos_weight": 199.40646111320194,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "random_state": RANDOM_STATE,
    "use_label_encoder": False
}

xgb_model = xgb.XGBClassifier(**best_params)
xgb_model.fit(X_train_scaled, y_train)

# Verify score
xgb_pred = xgb_model.predict_proba(X_test_scaled)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, xgb_pred)
pr_auc = auc(recall, precision)
print(f"✅ Verified PR AUC: {pr_auc:.4f}")

# Save model and scaler
print("💾 Saving artifacts...")
xgb_model.save_model("best_xgboost_model.json")
joblib.dump(scaler, "scaler.joblib")

print("✅ Model saved to 'best_xgboost_model.json'")
print("✅ Scaler saved to 'scaler.joblib'")
