# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Load data
data = pd.read_csv("../data/lb3/Iris.csv")
print(data.head())

# Encode species labels as integers
le = LabelEncoder()
data['Species'] = le.fit_transform(data['Species'])

# Select features for clustering
X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y_true = data['Species'].values

# Standardise
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Visualise raw data (3D)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='blue', s=50)
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')
plt.title('Raw data - Iris')
plt.show()

# Step 3: Gaussian Mixture Model clustering
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X_scaled)
labels = gmm.predict(X_scaled)

data['Cluster'] = labels

# Step 4: Visualise GMM clusters (3D)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
colors = ['red', 'green', 'blue']
for cluster in np.unique(labels):
    cluster_points = X[labels == cluster]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], s=50, color=colors[cluster], label=f'Cluster {cluster}')

ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')
plt.title('GMM clustering - Iris')
plt.legend()
plt.show()

# Step 5: Visualise true species labels (3D)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
true_colors = ['red', 'green', 'blue']
species_names = ['Setosa', 'Versicolor', 'Virginica']
for species in np.unique(y_true):
    species_points = X[y_true == species]
    ax.scatter(species_points[:, 0], species_points[:, 1], species_points[:, 2], s=50, color=true_colors[species], label=species_names[species])

ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')
plt.title('True classes - Iris')
plt.legend()
plt.show()

# Step 6: Evaluate clustering quality
silhouette_avg = silhouette_score(X_scaled, labels)
print(f'Average silhouette score: {silhouette_avg:.3f}')
