import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import save_figure

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score

# Step 1: Load data
data = pd.read_csv("../data/lb3/Mall_Customers.csv")
print(data.head())

# Step 2: Preprocessing
# Encode gender from categorical to numeric
le = LabelEncoder()
data["Gender"] = le.fit_transform(data["Gender"])

# Select only the relevant columns for clustering
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# Step 3: Visualise raw data
plt.figure(figsize=(8, 5))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c="blue", s=50)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Raw customer data")
plt.grid()
save_figure("kmeans_raw_data")
plt.show()

# Step 4: Elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, "bx-")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS (within-cluster sum of squares)")
plt.title("Elbow method — optimal number of clusters")
plt.grid()
save_figure("kmeans_elbow")
plt.show()

# The elbow is typically around 5 clusters
n_clusters = 5

# Step 5: K-Means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X)

X = X.copy()
X["Cluster"] = labels

# Step 6: Visualise clustering results
plt.figure(figsize=(8, 5))
colors = ["red", "blue", "green", "cyan", "magenta"]

for i in range(n_clusters):
    plt.scatter(
        X[X["Cluster"] == i].iloc[:, 0],
        X[X["Cluster"] == i].iloc[:, 1],
        s=50,
        c=colors[i],
        label=f"Cluster {i + 1}",
    )

# Visualise centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c="yellow", marker="X", label="Centroids")

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("K-Means customer segmentation")
plt.legend()
plt.grid()
save_figure("kmeans_clusters")
plt.show()

# Step 7: Evaluate clustering quality
silhouette_avg = silhouette_score(X.iloc[:, :2], labels)
print(f"Average silhouette score for {n_clusters} clusters: {silhouette_avg:.3f}")
