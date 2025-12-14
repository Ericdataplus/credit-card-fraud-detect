"""
06 - Summary Dashboard
Creates a comprehensive summary of the fraud detection project
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_path = os.path.join(project_dir, 'creditcard.csv')
output_path = os.path.join(project_dir, 'graphs', '06_summary_dashboard.png')

# Load data
df = pd.read_csv(data_path)

fraud_count = (df['Class'] == 1).sum()
legit_count = (df['Class'] == 0).sum()

# Create figure
fig = plt.figure(figsize=(20, 12))
fig.patch.set_facecolor('#0d1117')

# Title
fig.suptitle('Credit Card Fraud Detection - Summary Dashboard', fontsize=26, fontweight='bold', 
             color='white', y=0.97)

# Create grid
gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3, 
                      left=0.05, right=0.95, top=0.88, bottom=0.05)

# ===== ROW 1: Key Stats =====
stats = [
    ('Transactions', f'{len(df):,}', '#58a6ff'),
    ('Fraud Cases', f'{fraud_count:,}', '#f85149'),
    ('Fraud Rate', f'{fraud_count/len(df)*100:.3f}%', '#f0883e'),
    ('Best PR AUC', '91.04%', '#56d364'),
]

for i, (label, value, color) in enumerate(stats):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor('#161b22')
    
    # Draw colored bar at top
    ax.axhline(y=0.95, xmin=0.1, xmax=0.9, color=color, linewidth=4)
    
    ax.text(0.5, 0.55, value, fontsize=28, fontweight='bold', ha='center', va='center', 
            transform=ax.transAxes, color=color)
    ax.text(0.5, 0.2, label, fontsize=12, ha='center', va='center', 
            transform=ax.transAxes, color='#8b949e')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for spine in ax.spines.values():
        spine.set_color('#30363d')
        spine.set_linewidth(1)

# ===== ROW 2: Charts =====
# Class distribution (donut)
ax1 = fig.add_subplot(gs[1, 0:2])
ax1.set_facecolor('#0d1117')

colors = ['#56d364', '#f85149']
wedges, texts, autotexts = ax1.pie([legit_count, fraud_count], 
                                    labels=['Legitimate', 'Fraud'],
                                    autopct='%1.2f%%', colors=colors,
                                    textprops={'color': 'white', 'fontsize': 11},
                                    explode=(0, 0.1), startangle=90)
centre_circle = plt.Circle((0, 0), 0.5, fc='#0d1117')
ax1.add_patch(centre_circle)
ax1.text(0, 0, f'{fraud_count:,}\nFrauds', ha='center', va='center',
         fontsize=14, fontweight='bold', color='#f85149')
ax1.set_title('Class Distribution', color='white', fontsize=14, fontweight='bold')

# Model comparison
ax2 = fig.add_subplot(gs[1, 2:4])
ax2.set_facecolor('#0d1117')

models = ['Aug. XGBoost', 'Tuned XGB', 'Ensemble', 'Random Forest', 'Deep Learn.']
scores = [0.9104, 0.8849, 0.8813, 0.8788, 0.7361]
model_colors = ['#d4a72c', '#56d364', '#a371f7', '#58a6ff', '#f85149']

bars = ax2.barh(range(len(models)), scores, color=model_colors, height=0.6)
ax2.set_yticks(range(len(models)))
ax2.set_yticklabels(models, color='white', fontsize=10)
ax2.invert_yaxis()
ax2.set_xlim(0.6, 1.0)
ax2.set_xlabel('PR AUC', color='white', fontsize=11)
ax2.set_title('Model Performance', color='white', fontsize=14, fontweight='bold')
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_color('#30363d')

for bar, score in zip(bars, scores):
    ax2.text(score + 0.01, bar.get_y() + bar.get_height()/2, f'{score*100:.1f}%',
             va='center', color='white', fontsize=10, fontweight='bold')

# ===== ROW 3: More analysis =====
# Amount distribution
ax3 = fig.add_subplot(gs[2, 0:2])
ax3.set_facecolor('#0d1117')

fraud_data = df[df['Class'] == 1]['Amount']
legit_data = df[df['Class'] == 0]['Amount']

ax3.hist(legit_data, bins=50, alpha=0.7, label='Legitimate', color='#56d364', density=True)
ax3.hist(fraud_data, bins=50, alpha=0.7, label='Fraud', color='#f85149', density=True)
ax3.set_xlim(0, 500)
ax3.set_xlabel('Transaction Amount ($)', color='white', fontsize=11)
ax3.set_ylabel('Density', color='white', fontsize=11)
ax3.set_title('Amount Distribution', color='white', fontsize=14, fontweight='bold')
ax3.legend(facecolor='#161b22', labelcolor='white')
ax3.tick_params(colors='white')
for spine in ax3.spines.values():
    spine.set_color('#30363d')

# Key insights
ax4 = fig.add_subplot(gs[2, 2:4])
ax4.set_facecolor('#161b22')
ax4.set_xticks([])
ax4.set_yticks([])
for spine in ax4.spines.values():
    spine.set_color('#30363d')

insights = [
    ('Data aug. = +3.3% boost', '#d4a72c'),
    ('XGBoost > Deep Learning', '#56d364'),
    ('91% PR AUC achieved', '#56d364'),
    ('V14, V17 most predictive', '#58a6ff'),
    ('0 data leakage verified', '#a371f7'),
]

ax4.text(0.5, 0.92, 'Key Insights', fontsize=14, fontweight='bold', 
         ha='center', va='top', color='white', transform=ax4.transAxes)

for j, (insight, color) in enumerate(insights):
    y_pos = 0.75 - j * 0.15
    ax4.text(0.1, y_pos, '●', fontsize=16, color=color, transform=ax4.transAxes, va='center')
    ax4.text(0.18, y_pos, insight, fontsize=12, color='white', transform=ax4.transAxes, va='center')

plt.savefig(output_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
plt.close()

print(f"✅ Saved: {output_path}")
print(f"   Dashboard generated")
