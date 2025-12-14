import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json

# Load model and scaler
print("Loading model and scaler...")
# Use Booster directly to avoid sklearn wrapper issues with saved JSON
model = xgb.Booster()
model.load_model("best_model_augmented.json")
scaler = joblib.load("scaler_augmented.joblib")

def preprocess_input(df):
    """
    Preprocess input dataframe to match the training configuration
    """
    df_processed = df.copy()
    
    # Feature Engineering (must match training exactly)
    df_processed['Amount_log'] = np.log1p(df_processed['Amount'])
    
    df_processed['Time_hour'] = (df_processed['Time'] % 3600) / 3600
    df_processed['Time_day'] = (df_processed['Time'] % 86400) / 86400
    
    v_features = [f'V{i}' for i in range(1, 29)]
    df_processed['V_mean'] = df_processed[v_features].mean(axis=1)
    df_processed['V_std'] = df_processed[v_features].std(axis=1)
    df_processed['V_max'] = df_processed[v_features].max(axis=1)
    df_processed['V_min'] = df_processed[v_features].min(axis=1)
    df_processed['V_range'] = df_processed['V_max'] - df_processed['V_min']
    
    df_processed['V14_V17_int'] = df_processed['V14'] * df_processed['V17']
    df_processed['V10_V12_int'] = df_processed['V10'] * df_processed['V12']
    df_processed['V3_V7_int'] = df_processed['V3'] * df_processed['V7']
    df_processed['V4_V11_int'] = df_processed['V4'] * df_processed['V11']
    
    # Drop Class if it exists (ground truth)
    if 'Class' in df_processed.columns:
        df_processed = df_processed.drop('Class', axis=1)
        
    # Scale
    df_scaled = scaler.transform(df_processed)
    
    return df_scaled

# Example usage with random sample from dataset
print("\nLoading a sample transaction...")
df = pd.read_csv('creditcard.csv').sample(1)
print(f"Sample transaction:\n{df[['Time', 'Amount']].to_string(index=False)}")

# Preprocess
X_sample = preprocess_input(df)

# Predict using DMatrix for Booster
dtest = xgb.DMatrix(pd.DataFrame(X_sample, columns=scaler.get_feature_names_out()))
prob = model.predict(dtest)[0] # Booster predict returns probabilities directly
is_fraud = prob > 0.5

print(f"\nFRAUD PROBABILITY: {prob:.4f}")
print(f"PREDICTION: {'🚨 FRAUD' if is_fraud else '✅ LEGITIMATE'}")
print(f"ACTUAL CLASS: {df['Class'].values[0]}")
