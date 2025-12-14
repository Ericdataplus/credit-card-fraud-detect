"""
07 - Complete ML Results Dashboard
Verified comprehensive visualization of all model results
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
output_path = os.path.join(project_dir, 'graphs', '07_ml_results.png')

# VERIFIED RESULTS from project files
# From best_scores_v4.json, final_report.md, and beat_the_score_results.json

models_data = {
    # Model name: (PR AUC, category, color)
    'Augmented XGBoost': (0.9104, 'Gradient Boosting', '#d4a72c'),  # Winner
    'Tuned XGBoost': (0.8849, 'Gradient Boosting', '#56d364'),
    'Ensemble (V4)': (0.8820, 'Ensemble', '#a371f7'),
    'Random Forest': (0.8788, 'Ensemble', '#58a6ff'),  # From README
    'LightGBM': (0.8778, 'Gradient Boosting', '#0096c7'),
    'RF + SMOTE': (0.8373, 'Ensemble', '#00b4d8'),
    'Stacking Ensemble': (0.8333, 'Ensemble', '#7b2cbf'),
    'Weighted Ensemble': (0.8322, 'Ensemble', '#9d4edd'),
    'Rank Ensemble': (0.8056, 'Ensemble', '#c77dff'),
    'DAE + MLP': (0.7579, 'Deep Learning', '#f0883e'),
    'Deep Neural Net': (0.7361, 'Deep Learning', '#f85149'),
    'TabNet': (0.6588, 'Deep Learning', '#6e7681'),
}

# Sort by score
sorted_models = sorted(models_data.items(), key=lambda x: x[1][0], reverse=True)
names = [m[0] for m in sorted_models]
scores = [m[1][0] for m in sorted_models]
colors = [m[1][2] for m in sorted_models]
categories = [m[1][1] for m in sorted_models]

# Create figure
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0d1117')

# Main title
fig.suptitle('Credit Card Fraud Detection\nComplete ML Results', 
             fontsize=24, fontweight='bold', color='white', y=0.97)

# Create grid
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25, 
                      left=0.08, right=0.95, top=0.88, bottom=0.08)

# ===== Plot 1: All models bar chart =====
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor('#0d1117')

y_pos = np.arange(len(names))
bars = ax1.barh(y_pos, [s * 100 for s in scores], color=colors, height=0.7)

# Add score labels
for i, (bar, score, name) in enumerate(zip(bars, scores, names)):
    # Score on the bar
    ax1.text(score * 100 + 0.5, bar.get_y() + bar.get_height()/2, f'{score*100:.2f}%',
             va='center', ha='left', color='white', fontsize=10, fontweight='bold')
    
    # Winner badge
    if i == 0:
        ax1.text(score * 100 - 1, bar.get_y() + bar.get_height()/2, '★',
                 va='center', ha='right', color='#d4a72c', fontsize=16)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(names, color='white', fontsize=11)
ax1.invert_yaxis()
ax1.set_xlabel('PR AUC Score (%)', color='white', fontsize=12)
ax1.set_title('Model Performance Ranking (12 Models Tested)', color='white', fontsize=14, fontweight='bold')
ax1.set_xlim(50, 100)

# Reference lines
ax1.axvline(x=91, color='#56d364', linestyle='--', alpha=0.5, linewidth=1)
ax1.axvline(x=85, color='#58a6ff', linestyle='--', alpha=0.5, linewidth=1)
ax1.text(91.5, -0.5, '91% (Best)', color='#56d364', fontsize=9)
ax1.text(85.5, -0.5, '85% (Academic)', color='#58a6ff', fontsize=9)

ax1.tick_params(colors='white', labelsize=10)
for spine in ax1.spines.values():
    spine.set_color('#30363d')

# ===== Plot 2: Category comparison =====
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor('#0d1117')

cat_scores = {
    'Gradient Boosting': [0.9104, 0.8849, 0.8778],
    'Ensemble': [0.8820, 0.8788, 0.8373, 0.8333, 0.8322, 0.8056],
    'Deep Learning': [0.7579, 0.7361, 0.6588],
}

cat_names = list(cat_scores.keys())
cat_means = [np.mean(v) * 100 for v in cat_scores.values()]
cat_maxs = [max(v) * 100 for v in cat_scores.values()]
cat_colors = ['#56d364', '#a371f7', '#f85149']

x = np.arange(len(cat_names))
width = 0.35

bars1 = ax2.bar(x - width/2, cat_means, width, label='Average', color=cat_colors, alpha=0.6)
bars2 = ax2.bar(x + width/2, cat_maxs, width, label='Best', color=cat_colors)

ax2.set_ylabel('PR AUC (%)', color='white', fontsize=11)
ax2.set_title('Performance by Category', color='white', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(cat_names, color='white', fontsize=10)
ax2.legend(facecolor='#161b22', labelcolor='white', loc='lower right')
ax2.set_ylim(60, 100)
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_color('#30363d')

# Add value labels
for bar1, bar2 in zip(bars1, bars2):
    ax2.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.5, 
             f'{bar1.get_height():.1f}', ha='center', va='bottom', color='white', fontsize=9)
    ax2.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.5, 
             f'{bar2.get_height():.1f}', ha='center', va='bottom', color='white', fontsize=9)

# ===== Plot 3: Key insights panel =====
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor('#161b22')
ax3.set_xticks([])
ax3.set_yticks([])
for spine in ax3.spines.values():
    spine.set_color('#30363d')

ax3.text(0.5, 0.95, 'Key Results (Verified)', fontsize=14, fontweight='bold', 
         ha='center', va='top', color='white', transform=ax3.transAxes)

insights = [
    ('Dataset:', '284,807 transactions', '#58a6ff'),
    ('Fraud Cases:', '492 (0.173%)', '#f85149'),
    ('Best Score:', '91.04% PR AUC', '#56d364'),
    ('Improvement:', '+3.3% with data augmentation', '#d4a72c'),
    ('Winner:', 'XGBoost (with Optuna tuning)', '#56d364'),
    ('Runner-up:', 'Ensemble methods (~88%)', '#a371f7'),
    ('Deep Learning:', '65-76% (underperformed)', '#f85149'),
    ('Academic Baseline:', '85-86% (we beat it)', '#58a6ff'),
]

for i, (label, value, color) in enumerate(insights):
    y_pos = 0.82 - i * 0.095
    ax3.text(0.08, y_pos, label, fontsize=11, color='#8b949e', 
             transform=ax3.transAxes, va='center')
    ax3.text(0.45, y_pos, value, fontsize=11, color=color, fontweight='bold',
             transform=ax3.transAxes, va='center')

# Add data source note
ax3.text(0.5, 0.05, 'Data: Kaggle Credit Card Fraud Detection (2013)', 
         fontsize=9, ha='center', color='#6e7681', transform=ax3.transAxes)

plt.savefig(output_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
plt.close()

print(f"✅ Saved: {output_path}")
print(f"   12 models compared")
print(f"   Best: Augmented XGBoost @ 91.04%")
