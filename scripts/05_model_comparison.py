"""
05 - Model Performance Comparison
Visualizes the performance of different ML models
"""
import matplotlib.pyplot as plt
import numpy as np
import os

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
output_path = os.path.join(project_dir, 'graphs', '05_model_comparison.png')

# Model results (from final_report.md)
models = [
    ('Augmented XGBoost', 0.9104, '#d4a72c'),  # Gold - winner
    ('Tuned XGBoost', 0.8849, '#56d364'),
    ('Super Ensemble', 0.8813, '#a371f7'),
    ('Random Forest', 0.8788, '#58a6ff'),
    ('RF + SMOTE', 0.8764, '#0096c7'),
    ('XGBoost Base', 0.8712, '#0077b6'),
    ('DAE + MLP', 0.7579, '#f0883e'),
    ('Deep Neural Net', 0.7361, '#f85149'),
    ('TabNet', 0.6588, '#6e7681'),
]

names = [m[0] for m in models]
scores = [m[1] for m in models]
colors = [m[2] for m in models]

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

# Create horizontal bar chart
y_pos = np.arange(len(models))
bars = ax.barh(y_pos, scores, color=colors, height=0.7, edgecolor='white', linewidth=0.5)

# Add value labels and difference from best
best_score = max(scores)
for i, (bar, score) in enumerate(zip(bars, scores)):
    # Score label
    ax.text(score + 0.005, bar.get_y() + bar.get_height()/2, f'{score*100:.2f}%',
            va='center', ha='left', color='white', fontsize=12, fontweight='bold')
    
    # Difference from best (except winner)
    if score < best_score:
        diff = (best_score - score) * 100
        ax.text(0.01, bar.get_y() + bar.get_height()/2, f'-{diff:.1f}%',
                va='center', ha='left', color='#f85149', fontsize=9, alpha=0.8)

# Winner badge
ax.text(scores[0] - 0.02, 0, '🏆', fontsize=20, va='center', ha='right')

# Styling
ax.set_yticks(y_pos)
ax.set_yticklabels(names, color='white', fontsize=12)
ax.invert_yaxis()
ax.set_xlabel('PR AUC Score', color='white', fontsize=14)
ax.set_title('Model Performance Comparison\nPrecision-Recall Area Under Curve', 
             color='white', fontsize=18, fontweight='bold', pad=20)
ax.set_xlim(0.5, 1.0)

# Add vertical lines for reference
ax.axvline(x=0.9, color='#56d364', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(x=0.85, color='#58a6ff', linestyle='--', alpha=0.5, linewidth=1)
ax.text(0.9, len(models) + 0.3, '90%', color='#56d364', fontsize=10, ha='center')
ax.text(0.85, len(models) + 0.3, '85%', color='#58a6ff', fontsize=10, ha='center')

ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_color('#30363d')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add legend for model types
ax.text(0.52, 0.5, 'Gradient Boosting', color='#56d364', fontsize=10, fontweight='bold')
ax.text(0.52, 2.5, 'Ensemble Methods', color='#a371f7', fontsize=10)
ax.text(0.52, 6.5, 'Deep Learning', color='#f0883e', fontsize=10)

plt.tight_layout()
plt.savefig(output_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
plt.close()

print(f"✅ Saved: {output_path}")
print(f"   Best model: {names[0]} ({scores[0]*100:.2f}%)")
