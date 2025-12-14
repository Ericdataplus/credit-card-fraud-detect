"""
02 - Transaction Amount Analysis
Compares transaction amounts between fraud and legitimate transactions
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_path = os.path.join(project_dir, 'creditcard.csv')
output_path = os.path.join(project_dir, 'graphs', '02_amount_analysis.png')

# Load data
df = pd.read_csv(data_path)

# Separate fraud and legitimate
fraud = df[df['Class'] == 1]
legit = df[df['Class'] == 0]

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.patch.set_facecolor('#0d1117')

# ----- Plot 1: Amount distribution comparison -----
ax1 = axes[0]
ax1.set_facecolor('#0d1117')

# Histogram for both classes
ax1.hist(legit['Amount'], bins=50, alpha=0.7, label='Legitimate', color='#56d364', density=True)
ax1.hist(fraud['Amount'], bins=50, alpha=0.7, label='Fraud', color='#f85149', density=True)

ax1.set_xlabel('Transaction Amount ($)', color='white', fontsize=12)
ax1.set_ylabel('Density', color='white', fontsize=12)
ax1.set_title('Transaction Amount Distribution', color='white', fontsize=16, fontweight='bold')
ax1.legend(facecolor='#161b22', labelcolor='white', fontsize=11)
ax1.set_xlim(0, 500)  # Focus on common range
ax1.tick_params(colors='white')
for spine in ax1.spines.values():
    spine.set_color('#30363d')

# ----- Plot 2: Box plot comparison -----
ax2 = axes[1]
ax2.set_facecolor('#0d1117')

# Prepare data for box plot
bp_data = [legit['Amount'], fraud['Amount']]
bp = ax2.boxplot(bp_data, labels=['Legitimate', 'Fraud'], patch_artist=True)

# Style boxes
colors = ['#56d364', '#f85149']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for whisker in bp['whiskers']:
    whisker.set_color('white')
for cap in bp['caps']:
    cap.set_color('white')
for median in bp['medians']:
    median.set_color('#fff')
    median.set_linewidth(2)
for flier in bp['fliers']:
    flier.set(marker='o', markerfacecolor='#666', markersize=3, alpha=0.3)

# Add stats
ax2.text(1, legit['Amount'].median() + 10, f'Median: ${legit["Amount"].median():.2f}', 
         ha='center', color='white', fontsize=10)
ax2.text(2, fraud['Amount'].median() + 50, f'Median: ${fraud["Amount"].median():.2f}', 
         ha='center', color='white', fontsize=10)

ax2.set_ylabel('Amount ($)', color='white', fontsize=12)
ax2.set_title('Amount by Transaction Type', color='white', fontsize=16, fontweight='bold')
ax2.set_ylim(0, 1000)
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_color('#30363d')

plt.tight_layout()
plt.savefig(output_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
plt.close()

print(f"✅ Saved: {output_path}")
print(f"   Legit median: ${legit['Amount'].median():.2f}")
print(f"   Fraud median: ${fraud['Amount'].median():.2f}")
