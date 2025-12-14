from kaggle.api.kaggle_api_extended import KaggleApi
import os

api = KaggleApi()
api.authenticate()

print("Downloading dataset: nelgiriyewithana/credit-card-fraud-detection-dataset-2023")
api.dataset_download_files('nelgiriyewithana/credit-card-fraud-detection-dataset-2023', path='.', unzip=True)
print("Download complete.")

# Check files
files = os.listdir('.')
print(f"Files: {[f for f in files if f.endswith('.csv')]}")
