import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import save_figure

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load data
df = pd.read_csv("../data/lb3/Live.csv")

# Drop irrelevant columns
df.drop(columns=["status_id"], inplace=True)

# One-hot encode status_type
df = pd.get_dummies(df, columns=["status_type"], drop_first=True)

# Extract day-of-week and hour from the publication timestamp as numeric features
df["status_published"] = pd.to_datetime(df["status_published"])
df["day_of_week"] = df["status_published"].dt.dayofweek
df["hour"] = df["status_published"].dt.hour
df.drop(columns=["status_published"], inplace=True)  # Drop original datetime column

# Drop any remaining unused columns
for col in ["Column1", "Column2", "Column3", "Column4"]:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# 2. Standardise
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# 3. DBSCAN clustering
dbscan = DBSCAN(eps=2.0, min_samples=5)  # eps can be tuned
df["DBSCAN_Cluster"] = dbscan.fit_predict(df_scaled)

# 4. Reduce to 2 components with PCA for visualisation
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)
df["PCA1"] = df_pca[:, 0]
df["PCA2"] = df_pca[:, 1]

# 5. Visualise clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df["PCA1"], y=df["PCA2"], hue=df["DBSCAN_Cluster"], palette="viridis", legend="full")
plt.xlabel("Principal Component 1 (PCA1)")
plt.ylabel("Principal Component 2 (PCA2)")
plt.title("DBSCAN clustering — Facebook Live data")
save_figure("dbscan_live_clusters")
plt.show()
