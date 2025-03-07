# Credit Card Fraud Detection

Machine learning project for credit card fraud detection using ensemble methods and deep learning.

## Project Overview

This project focuses on building a robust machine learning system to detect fraudulent credit card transactions. Using a dataset of credit card transactions, we developed and compared various models to identify fraudulent activities while minimizing false positives.

## Dataset

The dataset used in this project is the Credit Card Fraud Detection dataset, which contains transactions made by credit cards in September 2013 by European cardholders.

**Due to its size (144MB), the dataset is not included in this repository.**

You can download it from Kaggle:
[Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

After downloading, place the `creditcard.csv` file in the root directory of this project.

## Key Features

- Data exploration and visualization of credit card transactions
- Handling extreme class imbalance (only 0.17% fraudulent transactions)
- Implementation of multiple machine learning models:
  - Random Forest (PR AUC: 0.8788)
  - XGBoost (PR AUC: 0.8712)
  - Random Forest with SMOTE (PR AUC: 0.8764)
  - Super Ensemble with GPU acceleration (PR AUC: 0.8813)
  - Deep Neural Network (PR AUC: 0.7361)
- Experiment tracking with Weights & Biases
- GPU acceleration for model training

## Setup and Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Download the dataset from Kaggle and place it in the root directory
4. Create a `.env` file with your Weights & Biases API key:
   ```
   WANDB_API_KEY=your_api_key_here
   ```

## Running the Notebook

Open and run the `script.ipynb` notebook in Jupyter:
```
jupyter notebook script.ipynb
```

## Results

The best performing model was the Super Ensemble with GPU acceleration, achieving a Precision-Recall AUC of 0.8813. This model combines predictions from multiple models to achieve better performance than any individual model.

## Future Work

- Explore anomaly detection techniques like Isolation Forests
- Implement cost-sensitive learning to better reflect business impact
- Investigate time-based features and transaction sequences
- Deploy the model as a real-time fraud detection system

## License

This project is open source and available under the [MIT License](LICENSE).