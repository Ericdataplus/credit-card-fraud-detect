"""
Verify Data Integrity - Check for Data Leakage
==============================================
Goal: Ensure the '2023' dataset does not contain duplicates of the original Test Set.
If it does, our 0.91 score is invalid (leakage).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print("=" * 60)
print("🕵️ Data Integrity Check (Leakage Detection)")
print("=" * 60)

# 1. Load Original Data & Re-create Split
print("Loading Original Data...")
df_orig = pd.read_csv('creditcard.csv')
X = df_orig.drop('Class', axis=1)
y = df_orig['Class']

# Use EXACT same split parameters as training script
RANDOM_STATE = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Original Test Set Size: {len(X_test)}")

# 2. Load Augmented Data
print("Loading External (2023) Data...")
try:
    df_new = pd.read_csv('creditcard_2023.csv')
    if 'id' in df_new.columns:
        df_new = df_new.drop('id', axis=1)
    
    # Keep only common columns for comparison
    common_cols = [c for c in X.columns if c in df_new.columns]
    df_new = df_new[common_cols]
    
    print(f"External Data Size: {len(df_new)}")
except FileNotFoundError:
    print("External data not found.")
    exit()

# 3. Check for Leakage
# We need to see if rows in X_test exist in df_new.
# Exact float comparison can be tricky, so we round to 6 decimal places for 'V' features
print("\nChecking for duplicates (Potential Leakage)...")

# Focus on V features + Amount (Time might differ)
compare_cols = [c for c in common_cols if c.startswith('V') or c == 'Amount']
print(f"Comparing on features: {compare_cols[:5]} ... {compare_cols[-1]}")

# Create signatures (strings) for faster comparison than merge on 500k rows
# Rounding is important for float comparison stability
X_test['signature'] = X_test[compare_cols].round(5).apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
df_new['signature'] = df_new[compare_cols].round(5).apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

# Find intersection
leakage = X_test[X_test['signature'].isin(df_new['signature'])]
num_leaks = len(leakage)

print("-" * 30)
if num_leaks > 0:
    print(f"🚨 CRITICAL ALERT: Found {num_leaks} duplicate rows!")
    print(f"   ({num_leaks / len(X_test) * 100:.2f}% of Test Set is in External Data)")
    print("   conclusion: The score is likely inflated due to data leakage.")
else:
    print("✅ INTEGRITY CONFIRMED: 0 duplicates found.")
    print("   Conclusion: The Test Set is completely unique. The score is valid.")
print("-" * 30)
