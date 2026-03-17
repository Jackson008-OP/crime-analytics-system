# src/train_model.py

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("Dataset/cleaned_crime_data.csv")

print("Dataset Loaded Successfully")
print(data.head())


# -----------------------------
# Feature Selection
# -----------------------------

features = [
    "MURDER",
    "RAPE",
    "KIDNAPPING_&_ABDUCTION",
    "RIOTS",
    "DOWRY_DEATHS"
]

target = "TOTAL_IPC_CRIMES"


X = data[features]
y = data[target]


# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


print("Training Data Size:", X_train.shape)
print("Testing Data Size:", X_test.shape)


# -----------------------------
# Train Model
# -----------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Training Completed")


# -----------------------------
# Prediction
# -----------------------------

y_pred = model.predict(X_test)


# -----------------------------
# Model Evaluation
# -----------------------------

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)


print("\nModel Performance")
print("-----------------------")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)


# -----------------------------
# Save Model
# -----------------------------

with open("models/crime_prediction_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved successfully in models/crime_prediction_model.pkl")