import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

dataset_path = project_root / "data" / "Crop_recommendation.csv"

print("Dataset path:")
print(dataset_path)

if not dataset_path.exists():
    print("ERROR: Dataset not found!")
    exit()

# Load tab-separated dataset
df = pd.read_csv(dataset_path, sep="\t")

print("\nDataset loaded successfully!")

print("\nShape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())