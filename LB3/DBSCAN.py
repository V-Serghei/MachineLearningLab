import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import save_figure

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Step 1: Load data
data = pd.read_csv("../data/lb3/cluster_moons.csv")
print(data.head())

# Step 2: Preprocessing
X = data[["X1", "X2"]].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Visualise raw data
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c="blue", s=50)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Raw data — Moons")
plt.grid()
save_figure("dbscan_raw_data")
plt.show()

# Step 4: DBSCAN clustering
# eps — neighbourhood radius; min_samples — minimum points to form a core point
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# Step 5: Visualise clusters
plt.figure(figsize=(8, 5))
unique_labels = set(labels)
colors = ["red", "blue", "green", "cyan", "magenta", "yellow", "black", "orange"]
for label in unique_labels:
    cluster = X[labels == label]
    plt.scatter(cluster[:, 0], cluster[:, 1], s=50, c=colors[label % len(colors)], label=f"Cluster {label}")

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("DBSCAN clustering — Moons")
plt.legend()
plt.grid()
save_figure("dbscan_clusters")
plt.show()

# Step 6: Evaluate clustering quality
if len(unique_labels) > 1 and -1 not in unique_labels:
    silhouette_avg = silhouette_score(X_scaled, labels)
    ch_score = calinski_harabasz_score(X_scaled, labels)
    db_score = davies_bouldin_score(X_scaled, labels)
    print(f"Average silhouette score: {silhouette_avg:.3f}")
    print(f"Calinski-Harabasz index: {ch_score:.3f}")
    print(f"Davies-Bouldin index: {db_score:.3f}")
else:
    print("Evaluation not possible: all points in one cluster or noise is present.")
