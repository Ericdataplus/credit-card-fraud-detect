from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

print("Searching for datasets...")
try:
    datasets = api.dataset_list(search="credit card fraud")
    for d in datasets:
        print(f"Ref: {d.ref}")
        print(f"Title: {d.title}")
        print("-" * 30)
except Exception as e:
    print(f"Error: {e}")
