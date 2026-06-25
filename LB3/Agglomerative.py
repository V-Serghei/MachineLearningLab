import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import save_figure

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score

# Step 1: Load data
data = pd.read_csv("../data/lb3/Wholesale_customers_data.csv")
print(data.head())

# Select features for clustering
X = data[["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Plot dendrogram to determine the number of clusters
plt.figure(figsize=(16, 10))
Z = linkage(X_scaled, method="ward")
dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=45, leaf_font_size=12, color_threshold=20)
plt.title("Agglomerative clustering dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean distance")
save_figure("agg_dendrogram_truncated")
plt.show()

plt.figure(figsize=(16, 10))
Z = linkage(X_scaled, method="ward")
dendrogram(Z, leaf_rotation=45, leaf_font_size=12, color_threshold=20)
plt.title("Agglomerative clustering dendrogram (full)")
plt.xlabel("Customers")
plt.ylabel("Euclidean distance")
save_figure("agg_dendrogram_full")
plt.show()

# Step 3: Agglomerative clustering with the chosen number of clusters
n_clusters = 4
agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
labels = agg_clustering.fit_predict(X_scaled)

# Step 4: Visualise clusters
plt.figure(figsize=(8, 5))
colors = ["red", "blue", "green", "cyan", "magenta"]
for cluster in np.unique(labels):
    cluster_points = X[labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=colors[cluster], label=f"Cluster {cluster + 1}")

plt.xlabel("Fresh")
plt.ylabel("Milk")
plt.title("Agglomerative clustering")
plt.legend()
save_figure("agg_clusters")
plt.show()

# Step 5: Evaluate clustering quality
silhouette_avg = silhouette_score(X_scaled, labels)
print(f"Average silhouette score: {silhouette_avg:.3f}")
