"""
Mobile-Optimized Graphs Generator for Credit Card Fraud Detection
Generates 6 key graphs optimized for phone screens with:
- Larger fonts (16pt+ titles, 14pt+ labels)
- Simpler layouts (single focus per graph)
- Portrait orientation (600x800)
- High contrast dark theme
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_path = os.path.join(project_dir, 'creditcard.csv')
output_dir = os.path.join(project_dir, 'graphs_mobile')

os.makedirs(output_dir, exist_ok=True)

# Mobile style settings
MOBILE_CONFIG = {
    'figsize': (6, 8),  # Portrait for phones
    'title_size': 20,
    'label_size': 16,
    'tick_size': 14,
    'value_size': 18,
    'bg_color': '#0d1117',
    'text_color': '#ffffff',
    'grid_color': '#30363d',
}

def setup_mobile_style():
    """Set matplotlib defaults for mobile viewing"""
    plt.rcParams['font.size'] = MOBILE_CONFIG['tick_size']
    plt.rcParams['axes.titlesize'] = MOBILE_CONFIG['title_size']
    plt.rcParams['axes.labelsize'] = MOBILE_CONFIG['label_size']
    plt.rcParams['xtick.labelsize'] = MOBILE_CONFIG['tick_size']
    plt.rcParams['ytick.labelsize'] = MOBILE_CONFIG['tick_size']
    plt.rcParams['figure.facecolor'] = MOBILE_CONFIG['bg_color']
    plt.rcParams['axes.facecolor'] = MOBILE_CONFIG['bg_color']
    plt.rcParams['text.color'] = MOBILE_CONFIG['text_color']
    plt.rcParams['axes.labelcolor'] = MOBILE_CONFIG['text_color']
    plt.rcParams['xtick.color'] = MOBILE_CONFIG['text_color']
    plt.rcParams['ytick.color'] = MOBILE_CONFIG['text_color']

def style_axes(ax):
    """Apply consistent styling to axes"""
    ax.set_facecolor(MOBILE_CONFIG['bg_color'])
    for spine in ax.spines.values():
        spine.set_color(MOBILE_CONFIG['grid_color'])
        spine.set_linewidth(2)

# ============================================================
# GRAPH 1: Fraud Rate - Big Number Focus
# ============================================================
def generate_fraud_rate():
    print("📱 Generating: Fraud Rate (mobile)")
    
    df = pd.read_csv(data_path)
    fraud_count = df['Class'].sum()
    total_count = len(df)
    fraud_rate = fraud_count / total_count * 100
    
    fig, ax = plt.subplots(figsize=MOBILE_CONFIG['figsize'])
    style_axes(ax)
    ax.axis('off')
    
    # Big fraud rate number
    ax.text(0.5, 0.55, f'{fraud_rate:.2f}%', fontsize=72, fontweight='bold',
            ha='center', va='center', color='#f85149', transform=ax.transAxes)
    
    ax.text(0.5, 0.40, 'FRAUD RATE', fontsize=24, fontweight='bold',
            ha='center', va='center', color='#8b949e', transform=ax.transAxes)
    
    # Context numbers
    ax.text(0.5, 0.22, f'{fraud_count:,} frauds', fontsize=18,
            ha='center', va='center', color='#f85149', transform=ax.transAxes)
    ax.text(0.5, 0.14, f'out of {total_count:,} transactions', fontsize=14,
            ha='center', va='center', color='#8b949e', transform=ax.transAxes)
    
    # Title at top
    ax.text(0.5, 0.92, 'Dataset Overview', fontsize=MOBILE_CONFIG['title_size'],
            ha='center', va='center', color='white', fontweight='bold', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_fraud_rate.png'), 
                dpi=200, facecolor=MOBILE_CONFIG['bg_color'], bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 01_fraud_rate.png")

# ============================================================
# GRAPH 2: Winner Model Highlight
# ============================================================
def generate_winner():
    print("📱 Generating: Winner Model (mobile)")
    
    fig, ax = plt.subplots(figsize=MOBILE_CONFIG['figsize'])
    style_axes(ax)
    ax.axis('off')
    
    # Trophy emoji or star
    ax.text(0.5, 0.75, '🏆', fontsize=80, ha='center', va='center', transform=ax.transAxes)
    
    # Big score
    ax.text(0.5, 0.52, '91.04%', fontsize=64, fontweight='bold',
            ha='center', va='center', color='#56d364', transform=ax.transAxes)
    
    ax.text(0.5, 0.38, 'PR AUC Score', fontsize=20,
            ha='center', va='center', color='#8b949e', transform=ax.transAxes)
    
    # Model name
    ax.text(0.5, 0.22, 'XGBoost', fontsize=28, fontweight='bold',
            ha='center', va='center', color='white', transform=ax.transAxes)
    ax.text(0.5, 0.14, 'with Data Augmentation', fontsize=16,
            ha='center', va='center', color='#58a6ff', transform=ax.transAxes)
    
    # Title
    ax.text(0.5, 0.92, 'Best Model', fontsize=MOBILE_CONFIG['title_size'],
            ha='center', va='center', color='white', fontweight='bold', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_winner.png'),
                dpi=200, facecolor=MOBILE_CONFIG['bg_color'], bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 02_winner.png")

# ============================================================
# GRAPH 3: Top 5 Models - Simple Horizontal Bar
# ============================================================
def generate_top_models():
    print("📱 Generating: Top 5 Models (mobile)")
    
    models = [
        ('XGBoost +Aug', 91.04, '#d4a72c'),
        ('XGBoost', 88.49, '#56d364'),
        ('Ensemble', 88.20, '#a371f7'),
        ('Random Forest', 87.88, '#58a6ff'),
        ('LightGBM', 87.78, '#0096c7'),
    ]
    
    fig, ax = plt.subplots(figsize=MOBILE_CONFIG['figsize'])
    style_axes(ax)
    
    names = [m[0] for m in models]
    scores = [m[1] for m in models]
    colors = [m[2] for m in models]
    
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, scores, color=colors, height=0.6)
    
    # Add score labels on bars
    for bar, score in zip(bars, scores):
        ax.text(score - 3, bar.get_y() + bar.get_height()/2, f'{score:.1f}%',
                va='center', ha='right', color='white', fontsize=16, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=14)
    ax.invert_yaxis()
    ax.set_xlim(80, 95)
    ax.set_xlabel('PR AUC (%)', fontsize=14)
    ax.set_title('Top 5 Models', fontsize=MOBILE_CONFIG['title_size'], fontweight='bold', pad=20)
    
    # Reference line
    ax.axvline(x=85, color='#58a6ff', linestyle='--', alpha=0.7, linewidth=2)
    ax.text(85.5, 4.8, 'Academic\nBaseline', fontsize=10, color='#58a6ff', va='top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_top_models.png'),
                dpi=200, facecolor=MOBILE_CONFIG['bg_color'], bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 03_top_models.png")

# ============================================================
# GRAPH 4: Deep Learning vs Gradient Boosting
# ============================================================
def generate_comparison():
    print("📱 Generating: DL vs GB Comparison (mobile)")
    
    fig, ax = plt.subplots(figsize=MOBILE_CONFIG['figsize'])
    style_axes(ax)
    
    categories = ['Gradient\nBoosting', 'Deep\nLearning']
    best_scores = [91.04, 75.79]
    colors = ['#56d364', '#f85149']
    
    bars = ax.bar(categories, best_scores, color=colors, width=0.6, edgecolor='white', linewidth=2)
    
    # Add value labels
    for bar, score in zip(bars, best_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{score:.1f}%', ha='center', va='bottom', fontsize=20, fontweight='bold', color='white')
    
    # Add delta
    ax.annotate('', xy=(0.15, 91), xytext=(0.85, 76),
                arrowprops=dict(arrowstyle='->', color='#d4a72c', lw=3))
    ax.text(0.5, 83, '+15%', fontsize=24, fontweight='bold', color='#d4a72c', ha='center',
            transform=ax.get_xaxis_transform())
    
    ax.set_ylabel('Best PR AUC (%)', fontsize=14)
    ax.set_ylim(0, 100)
    ax.set_title('Why XGBoost Won', fontsize=MOBILE_CONFIG['title_size'], fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_comparison.png'),
                dpi=200, facecolor=MOBILE_CONFIG['bg_color'], bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 04_comparison.png")

# ============================================================
# GRAPH 5: Feature Importance - Top 5
# ============================================================
def generate_features():
    print("📱 Generating: Top Features (mobile)")
    
    # Top 5 features from SHAP analysis
    features = [
        ('V14', 0.28),
        ('V17', 0.19),
        ('V12', 0.15),
        ('V10', 0.12),
        ('V4', 0.08),
    ]
    
    fig, ax = plt.subplots(figsize=MOBILE_CONFIG['figsize'])
    style_axes(ax)
    
    names = [f[0] for f in features]
    importance = [f[1] for f in features]
    colors = ['#f85149', '#ff6b6b', '#ff8c8c', '#ffadad', '#ffcece']
    
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, importance, color=colors, height=0.6)
    
    # Add percentage labels
    for bar, imp in zip(bars, importance):
        ax.text(imp + 0.01, bar.get_y() + bar.get_height()/2, f'{imp:.0%}',
                va='center', ha='left', color='white', fontsize=16, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=16, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlim(0, 0.35)
    ax.set_xlabel('Importance Score', fontsize=14)
    ax.set_title('Top Fraud Indicators', fontsize=MOBILE_CONFIG['title_size'], fontweight='bold', pad=20)
    
    # Subtitle
    ax.text(0.5, -0.08, 'SHAP Feature Importance', fontsize=12, color='#8b949e',
            ha='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_features.png'),
                dpi=200, facecolor=MOBILE_CONFIG['bg_color'], bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 05_features.png")

# ============================================================
# GRAPH 6: Key Takeaways Summary
# ============================================================
def generate_summary():
    print("📱 Generating: Key Takeaways (mobile)")
    
    fig, ax = plt.subplots(figsize=MOBILE_CONFIG['figsize'])
    style_axes(ax)
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'Key Takeaways', fontsize=MOBILE_CONFIG['title_size'],
            ha='center', va='center', color='white', fontweight='bold', transform=ax.transAxes)
    
    takeaways = [
        ('1', '91% PR AUC', 'Beats academic baselines', '#56d364'),
        ('2', '+3.3% Boost', 'From data augmentation', '#d4a72c'),
        ('3', 'XGBoost Wins', 'Over deep learning', '#58a6ff'),
        ('4', 'V14, V17, V12', 'Top fraud indicators', '#f85149'),
    ]
    
    for i, (num, headline, subtext, color) in enumerate(takeaways):
        y = 0.78 - i * 0.20
        
        # Number circle
        circle = plt.Circle((0.12, y), 0.05, transform=ax.transAxes, 
                           facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(circle)
        ax.text(0.12, y, num, fontsize=18, fontweight='bold', color='white',
                ha='center', va='center', transform=ax.transAxes)
        
        # Text
        ax.text(0.22, y + 0.02, headline, fontsize=18, fontweight='bold',
                color='white', va='center', transform=ax.transAxes)
        ax.text(0.22, y - 0.04, subtext, fontsize=14,
                color='#8b949e', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_summary.png'),
                dpi=200, facecolor=MOBILE_CONFIG['bg_color'], bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 06_summary.png")

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("\n📱 Generating Mobile-Optimized Graphs")
    print("=" * 50)
    
    setup_mobile_style()
    
    generate_fraud_rate()
    generate_winner()
    generate_top_models()
    generate_comparison()
    generate_features()
    generate_summary()
    
    print("\n" + "=" * 50)
    print(f"✅ All mobile graphs saved to: {output_dir}")
    print("📱 6 graphs optimized for phone viewing")
