# src/crime_clustering.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("Loading dataset...")

# Load dataset
data = pd.read_csv("Dataset/cleaned_crime_data.csv")

print("Dataset loaded successfully")
print("Dataset Shape:", data.shape)

# Features used for clustering
features = [
    "MURDER",
    "RAPE",
    "KIDNAPPING_&_ABDUCTION",
    "RIOTS",
    "DOWRY_DEATHS",
    "THEFT",
    "ROBBERY"
]

# Extract feature data
X = data[features]

print("\nSelected Features:")
print(features)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nData scaling completed")

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)

data["crime_cluster"] = kmeans.fit_predict(X_scaled)

print("\nClustering completed")

# Show cluster distribution
print("\nCluster Distribution:")
print(data["crime_cluster"].value_counts())

# Rename clusters for understanding
cluster_labels = {
    0: "Low Crime",
    1: "Moderate Crime",
    2: "High Crime"
}

data["crime_level"] = data["crime_cluster"].map(cluster_labels)

print("\nSample Results:")
print(data[["STATE/UT", "crime_cluster", "crime_level"]].head())

# Save clustered dataset
output_path = "Dataset/crime_clustered_data.csv"
data.to_csv(output_path, index=False)

print("\nClustered dataset saved successfully")
print("Saved at:", output_path)