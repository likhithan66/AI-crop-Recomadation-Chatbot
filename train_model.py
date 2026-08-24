import pandas as pd
from pathlib import Path

# Find the project root
project_root = Path(__file__).resolve().parent.parent

# Dataset path
dataset_path = project_root / "data" / "Crop_recommendation.csv"

# Load dataset
df = pd.read_csv(dataset_path, sep="\t")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# -----------------------------
# STEP 2: X and y
# -----------------------------

X = df[
    [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]
]

y = df["label"]

print("\nInput features (X):")
print(X.head())

print("\nTarget variable (y):")
print(y.head())


# -----------------------------
# STEP 3: Train/Test Split
# -----------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# -----------------------------
# STEP 4: Feature Scaling
# -----------------------------

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Create scalers
mx = MinMaxScaler()
sc = StandardScaler()

# Fit on training data
X_train_minmax = mx.fit_transform(X_train)

# Transform test data using the same scaler
X_test_minmax = mx.transform(X_test)

# Standard scaling
X_train_scaled = sc.fit_transform(X_train_minmax)
X_test_scaled = sc.transform(X_test_minmax)

print("\nScaling completed!")

print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)
# -----------------------------
# STEP 5: Random Forest Model
# -----------------------------

from sklearn.ensemble import RandomForestClassifier

# Create Random Forest model
# -----------------------------
# STEP 5: Random Forest Model
# -----------------------------

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("\nRandom Forest model trained successfully!")
# -----------------------------
# STEP 6: Model Evaluation
# -----------------------------

from sklearn.metrics import accuracy_score

# Predict using test data
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

print("\nAccuracy Percentage:")
print(f"{accuracy * 100:.2f}%")
# -----------------------------
# STEP 7: Save the ML Model
# -----------------------------

import joblib

# Model folder
model_folder = project_root / "model"
model_folder.mkdir(exist_ok=True)

# Save Random Forest model
model_path = model_folder / "crop_model.pkl"

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("Model location:")
print(model_path)
# -----------------------------
# STEP 8: Save the Scalers
# -----------------------------

# Save MinMaxScaler
minmax_path = model_folder / "minmax_scaler.pkl"
joblib.dump(mx, minmax_path)

# Save StandardScaler
standard_path = model_folder / "standard_scaler.pkl"
joblib.dump(sc, standard_path)

print("\nScalers saved successfully!")

print("MinMaxScaler:")
print(minmax_path)

print("StandardScaler:")
print(standard_path)