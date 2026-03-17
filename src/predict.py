# src/predict.py

import joblib
import numpy as np

print("Loading trained model...")

# Load trained model
model = joblib.load("models/crime_prediction_model.pkl")

print("Model loaded successfully")

# Example input data
# [MURDER, RAPE, KIDNAPPING_&_ABDUCTION, RIOTS, DOWRY_DEATHS]

input_data = np.array([[50, 120, 200, 300, 40]])

# Make prediction
prediction = model.predict(input_data)

print("\nPredicted Total IPC Crimes:")
print(prediction[0])