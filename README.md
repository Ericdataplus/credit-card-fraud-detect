# 🛡️ Credit Card Fraud Detection

> 📊 **Inspired by:** [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
>
> Machine learning project achieving **91% PR AUC** for detecting fraudulent transactions using ensemble methods, deep learning, and data augmentation.

🔗 **[View Live Dashboard](https://ericdataplus.github.io/credit-card-fraud-detect/)**

![Model Comparison](graphs/pr_curves_comparison.png)

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Best PR AUC | **91.04%** (Augmented) |
| Best on Original Data | **87.91%** (XGBoost) |
| Dataset Size | 284,807 transactions |
| Fraud Rate | 0.17% |
| SOTA Models | 7 |
| GPU Used | RTX 3060 12GB |

## 🏆 December 2025 SOTA Benchmark

| Model | PR AUC | ROC AUC | F1 | Dataset |
|-------|--------|---------|-----|---------|
| **XGBoost** | **87.91%** | 97.43% | 88.9% | Original |
| CatBoost | 87.20% | 96.92% | 89.5% | Original |
| Ensemble | 87.10% | 96.21% | 88.5% | Original |
| LightGBM | 86.75% | 97.25% | 87.0% | Original |
| TabTransformer | 73.72% | 96.09% | 78.7% | Original |
| TabNet | 73.37% | 98.18% | 77.0% | Original |
| DeepFraudNet | 68.72% | 97.90% | 80.2% | Original |
| **Augmented XGBoost** | **91.04%** | - | - | +2023 Data |

## 🔍 Key Findings

1. **Data Augmentation = +3.3% Boost** — Adding external 2023 fraud data increased training fraud rate from 0.17% to 55.6%
2. **XGBoost Beats Deep Learning** — Gradient boosting (88-91%) outperformed neural networks (68-74%) on structured tabular data
3. **No Data Leakage** — Rigorous integrity check confirmed 0 duplicates between augmented data and test set
4. **Beats Academic Baselines** — Our 87.91% exceeds typical research benchmarks of 85-86%
5. **SHAP Explainability** — V14, V17, V12, V10 are the most important fraud indicators

## 🧠 SOTA Techniques Implemented

### Deep Learning (GPU)
- **TabNet** - Google's attention-based tree mimic
- **DeepFraudNet** - Custom architecture with Focal Loss
- **TabularTransformer** - Feature tokenization + transformer encoder

### Gradient Boosting (GPU-Accelerated)
- **XGBoost** with Optuna hyperparameter tuning (25 trials)
- **CatBoost** GPU training
- **LightGBM** ensemble

### Advanced Features
- **Focal Loss** for extreme class imbalance (α=0.75, γ=2.0)
- **SHAP Explainability** for model interpretability
- **Weighted Ensemble** combining top models

## 📁 Project Structure

```
credit-card-fraud-detect/
├── index.html                    # Interactive Dashboard
├── graphs/                       # Visualizations
│   ├── pr_curves_comparison.png  # All models PR curves
│   ├── shap_summary.png          # SHAP feature importance
│   ├── model_comparison.png      # Bar chart comparison
│   └── confusion_matrix_best.png
├── sota_fraud_detection.py       # SOTA training script (GPU)
├── sota_augmented.py             # Augmented data training
├── sota_results.json             # Benchmark results
├── predict_fraud.py              # Production prediction script
├── best_xgboost_model.json       # Saved best model
├── deep_fraud_net.pt             # PyTorch DeepFraudNet
├── tab_transformer.pt            # PyTorch Transformer
└── creditcard.csv                # Dataset (not in repo)
```

## 🛠️ Tech Stack

- **Python** - Core language
- **XGBoost / CatBoost / LightGBM** - Gradient boosting
- **PyTorch** - Deep learning (TabNet, Transformer, Focal Loss)
- **Optuna** - Hyperparameter optimization
- **SHAP** - Model explainability
- **Scikit-learn** - Preprocessing & metrics

## 📦 Data Sources

- Primary: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) - 284K transactions
- Augmentation: [Credit Card Fraud 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023) - 568K transactions
- Bank Fraud: [NeurIPS 2022](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) - 1M accounts
- PaySim: [Mobile Money](https://www.kaggle.com/datasets/ealaxi/paysim1) - 6.3M transactions

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/Ericdataplus/credit-card-fraud-detect.git
cd credit-card-fraud-detect

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle and place creditcard.csv in root

# Run SOTA training (requires GPU)
python sota_fraud_detection.py

# Or run prediction on new data
python predict_fraud.py
```

## 📈 Why PR AUC?

For **highly imbalanced datasets** (0.17% fraud), accuracy is misleading. A model predicting "not fraud" always gets 99.83% accuracy!

**Precision-Recall AUC** measures:
- **Precision**: Of flagged transactions, how many are actually fraud?
- **Recall**: Of all frauds, how many did we catch?

This is the industry standard for fraud detection.

## 💰 Business Impact

To deploy commercially, tune the decision threshold based on:
- **False Negative Cost**: Lost transaction + chargeback fees (~$180 average)
- **False Positive Cost**: Lost sale + customer friction (~$40)

At optimal threshold, catching 85% of fraud while only flagging 5% false positives can save millions annually.

---

Made with 🛡️ by [Ericdataplus](https://github.com/Ericdataplus) | December 2025 (SOTA Update)