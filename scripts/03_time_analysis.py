"""
03 - Time Analysis
Analyzes when fraudulent transactions occur
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_path = os.path.join(project_dir, 'creditcard.csv')
output_path = os.path.join(project_dir, 'graphs', '03_time_analysis.png')

# Load data
df = pd.read_csv(data_path)

# Convert Time to hours (Time is in seconds from first transaction)
df['Hour'] = (df['Time'] / 3600) % 24

# Separate fraud and legitimate
fraud = df[df['Class'] == 1]
legit = df[df['Class'] == 0]

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.patch.set_facecolor('#0d1117')

# ----- Plot 1: Transaction count by hour -----
ax1 = axes[0]
ax1.set_facecolor('#0d1117')

# Histogram of transactions by hour
hours = np.arange(0, 24)
legit_counts, _ = np.histogram(legit['Hour'], bins=24, range=(0, 24))
fraud_counts, _ = np.histogram(fraud['Hour'], bins=24, range=(0, 24))

# Normalize for comparison
legit_norm = legit_counts / legit_counts.sum()
fraud_norm = fraud_counts / fraud_counts.sum()

width = 0.35
ax1.bar(hours - width/2, legit_norm, width, label='Legitimate', color='#56d364', alpha=0.8)
ax1.bar(hours + width/2, fraud_norm, width, label='Fraud', color='#f85149', alpha=0.8)

ax1.set_xlabel('Hour of Day', color='white', fontsize=12)
ax1.set_ylabel('Proportion of Transactions', color='white', fontsize=12)
ax1.set_title('Transaction Timing by Class', color='white', fontsize=16, fontweight='bold')
ax1.legend(facecolor='#161b22', labelcolor='white')
ax1.set_xticks([0, 6, 12, 18, 24])
ax1.set_xticklabels(['12 AM', '6 AM', '12 PM', '6 PM', '12 AM'])
ax1.tick_params(colors='white')
for spine in ax1.spines.values():
    spine.set_color('#30363d')

# ----- Plot 2: Fraud rate by hour -----
ax2 = axes[1]
ax2.set_facecolor('#0d1117')

# Calculate fraud rate by hour
all_counts, _ = np.histogram(df['Hour'], bins=24, range=(0, 24))
fraud_rate = np.divide(fraud_counts, all_counts, where=all_counts!=0) * 100

# Color based on rate
colors = plt.cm.RdYlGn_r(fraud_rate / fraud_rate.max())
bars = ax2.bar(hours, fraud_rate, color=colors, edgecolor='none')

# Add average line
avg_rate = df['Class'].mean() * 100
ax2.axhline(y=avg_rate, color='#f0883e', linestyle='--', linewidth=2, label=f'Avg: {avg_rate:.3f}%')

ax2.set_xlabel('Hour of Day', color='white', fontsize=12)
ax2.set_ylabel('Fraud Rate (%)', color='white', fontsize=12)
ax2.set_title('Fraud Rate by Hour', color='white', fontsize=16, fontweight='bold')
ax2.legend(facecolor='#161b22', labelcolor='white')
ax2.set_xticks([0, 6, 12, 18])
ax2.set_xticklabels(['12 AM', '6 AM', '12 PM', '6 PM'])
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_color('#30363d')

plt.tight_layout()
plt.savefig(output_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
plt.close()

print(f"✅ Saved: {output_path}")
print(f"   Peak fraud hour: {hours[fraud_rate.argmax()]}:00")
