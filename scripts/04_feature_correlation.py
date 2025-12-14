"""
04 - Feature Correlation Analysis
Identifies which PCA features are most correlated with fraud
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_path = os.path.join(project_dir, 'creditcard.csv')
output_path = os.path.join(project_dir, 'graphs', '04_feature_correlation.png')

# Load data
df = pd.read_csv(data_path)

# Calculate correlation with Class (fraud)
feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
correlations = df[feature_cols + ['Class']].corr()['Class'].drop('Class')

# Sort by absolute correlation
sorted_corr = correlations.abs().sort_values(ascending=False)
top_features = sorted_corr.head(15).index
top_corr = correlations[top_features]

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.patch.set_facecolor('#0d1117')

# ----- Plot 1: Top feature correlations -----
ax1 = axes[0]
ax1.set_facecolor('#0d1117')

colors = ['#f85149' if c < 0 else '#56d364' for c in top_corr]
bars = ax1.barh(range(len(top_corr)), top_corr.values, color=colors)

ax1.set_yticks(range(len(top_corr)))
ax1.set_yticklabels(top_corr.index, color='white', fontsize=10)
ax1.invert_yaxis()
ax1.set_xlabel('Correlation with Fraud', color='white', fontsize=12)
ax1.set_title('Top 15 Features Correlated with Fraud', color='white', fontsize=16, fontweight='bold')
ax1.axvline(x=0, color='white', linestyle='-', linewidth=0.5)
ax1.tick_params(colors='white')
for spine in ax1.spines.values():
    spine.set_color('#30363d')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, top_corr.values)):
    ax1.text(val + 0.01 if val > 0 else val - 0.01, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', ha='left' if val > 0 else 'right',
             color='white', fontsize=9)

# ----- Plot 2: Feature importance heatmap-style -----
ax2 = axes[1]
ax2.set_facecolor('#0d1117')

# Show all V features as a grid
all_corr = correlations[[f'V{i}' for i in range(1, 29)]]
grid_data = all_corr.values.reshape(4, 7)

im = ax2.imshow(grid_data, cmap='RdYlGn_r', aspect='auto', vmin=-0.3, vmax=0.3)

# Add labels
for i in range(4):
    for j in range(7):
        feat_num = i * 7 + j + 1
        if feat_num <= 28:
            val = all_corr[f'V{feat_num}']
            ax2.text(j, i, f'V{feat_num}\n{val:.2f}', ha='center', va='center',
                    color='white' if abs(val) > 0.15 else '#888', fontsize=9)

ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title('All Feature Correlations with Fraud', color='white', fontsize=16, fontweight='bold')

# Colorbar
cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
cbar.set_label('Correlation', color='white')
cbar.ax.tick_params(colors='white')

plt.tight_layout()
plt.savefig(output_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
plt.close()

print(f"✅ Saved: {output_path}")
print(f"   Most correlated: {top_corr.index[0]} ({top_corr.values[0]:.3f})")
