import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import save_figure

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load heart.csv — patient records with a heart disease target label.
data = pd.read_csv("../data/heart.csv")

X = data.drop(columns=["target"])
y = data["target"].copy()

# Visualise class distribution to assess balance before training.
plt.figure(figsize=(6, 4))
sns.countplot(x=y, hue=y, palette="coolwarm", legend=False)
plt.title("Class distribution — Heart Disease")
save_figure("rbf_heart_class_dist")
plt.show()

# Stratified split — preserves class proportions in both sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 100)
#%%
print(data.head())
#%%
print(data.info())
#%%
print(data.describe())
#%%
# Standardise features — required for correct distance computation in KMeans and RBF.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Find 10 cluster centres with KMeans; they serve as RBF kernel anchors.
n_centers = 10
kmeans = KMeans(n_clusters=n_centers, random_state=42, n_init=10)
kmeans.fit(X_train_scaled)
centers = kmeans.cluster_centers_


def rbf_transform(X, centers, sigma=1.0):
    """Apply Gaussian RBF kernel: map each sample to distances from cluster centres.

    Args:
        X:       Input array of shape (n_samples, n_features).
        centers: Cluster centre array of shape (n_centers, n_features).
        sigma:   Bandwidth parameter.

    Returns:
        Transformed array of shape (n_samples, n_centers).
    """
    return np.exp(-np.linalg.norm(X[:, np.newaxis, :] - centers, axis=2) ** 2 / (2 * sigma**2))


#%%
sigma = 1.0
X_train_rbf = rbf_transform(X_train_scaled, centers, sigma)
X_test_rbf = rbf_transform(X_test_scaled, centers, sigma)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_rbf, y_train)

y_pred = clf.predict(X_test_rbf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")
#%%

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))
#%%

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion matrix — RBF Classifier (Heart)")
save_figure("rbf_heart_cm")
plt.show()
#%%

y_prob = clf.predict_proba(X_test_rbf)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve — RBF Classifier (Heart)")
plt.legend()
save_figure("rbf_heart_roc")
plt.show()
