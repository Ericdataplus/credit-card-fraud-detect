"""
01 - Class Distribution Analysis
Visualizes the extreme imbalance between fraud and non-fraud transactions
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_path = os.path.join(project_dir, 'creditcard.csv')
output_path = os.path.join(project_dir, 'graphs', '01_class_distribution.png')

# Load data
df = pd.read_csv(data_path)

# Class counts
class_counts = df['Class'].value_counts()
fraud_count = class_counts[1]
legit_count = class_counts[0]

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.patch.set_facecolor('#0d1117')

# ----- Plot 1: Bar chart (log scale) -----
ax1 = axes[0]
ax1.set_facecolor('#0d1117')

colors = ['#56d364', '#f85149']
bars = ax1.bar(['Legitimate', 'Fraud'], [legit_count, fraud_count], color=colors, width=0.6)

# Add value labels
for bar, val in zip(bars, [legit_count, fraud_count]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000, f'{val:,}',
             ha='center', va='bottom', color='white', fontsize=14, fontweight='bold')

ax1.set_yscale('log')
ax1.set_ylabel('Count (log scale)', color='white', fontsize=12)
ax1.set_title('Transaction Class Distribution', color='white', fontsize=16, fontweight='bold')
ax1.tick_params(colors='white')
for spine in ax1.spines.values():
    spine.set_color('#30363d')

# ----- Plot 2: Pie chart showing imbalance -----
ax2 = axes[1]
ax2.set_facecolor('#0d1117')

# Create donut chart
sizes = [legit_count, fraud_count]
labels = ['Legitimate', 'Fraud']
explode = (0, 0.1)  # Explode fraud slice

wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=labels, autopct='%1.2f%%',
                                    colors=colors, textprops={'color': 'white', 'fontsize': 12},
                                    startangle=90)

# Draw center circle for donut effect
centre_circle = plt.Circle((0, 0), 0.5, fc='#0d1117')
ax2.add_patch(centre_circle)

# Add center text
ax2.text(0, 0, f'{fraud_count/len(df)*100:.2f}%\nFraud', ha='center', va='center',
         fontsize=20, fontweight='bold', color='#f85149')

ax2.set_title('Fraud Rate: Extreme Imbalance', color='white', fontsize=16, fontweight='bold')

for autotext in autotexts:
    autotext.set_fontweight('bold')
    autotext.set_fontsize(11)

plt.tight_layout()
plt.savefig(output_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
plt.close()

print(f"✅ Saved: {output_path}")
print(f"   Fraud rate: {fraud_count/len(df)*100:.4f}%")
