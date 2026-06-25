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

# 1. Load data — creditcard.csv has transaction records; 'Class' is the target label.
data = pd.read_csv("../data/creditcard.csv")

X = data.drop(columns=["Class"])
y = data["Class"].copy()

# 2. Artificially shift class distribution:
#    Change 1/6 of class-0 samples to class 1 to create a less extreme imbalance for demonstration.
zero_indices = y[y == 0].index
num_to_change = len(zero_indices) // 6
np.random.seed(42)
indices_to_change = np.random.choice(zero_indices, size=num_to_change, replace=False)
y.loc[indices_to_change] = 1

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 100)
#%%
print(data.head())
#%%
print(data.info())
#%%
print(data.describe())
#%%

# 3. Visualise the new class distribution.
plt.figure(figsize=(6, 4))
sns.countplot(x=y, hue=y, palette="coolwarm", legend=False)
plt.title("Class distribution after relabelling")
save_figure("rbf_credit_class_dist")
plt.show()
#%%

# 4. Split with stratification to preserve class proportions.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Standardise features — important when distances are used (KMeans, RBF).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Find cluster centres with KMeans — 20 centres used as RBF kernel anchors.
n_centers = 20
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
    # X[:, np.newaxis, :] adds a dimension for broadcasting the centre subtraction.
    return np.exp(-np.linalg.norm(X[:, np.newaxis, :] - centers, axis=2) ** 2 / (2 * sigma**2))


sigma = 2.0
X_train_rbf = rbf_transform(X_train_scaled, centers, sigma)
X_test_rbf = rbf_transform(X_test_scaled, centers, sigma)

# 8. Train logistic regression on RBF-transformed features.
#    class_weight='balanced' compensates for class imbalance.
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train_rbf, y_train)
#%%

y_pred = clf.predict(X_test_rbf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")
#%%

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))
#%%

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion matrix — RBF Classifier (Credit Card)")
save_figure("rbf_credit_cm")
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
plt.title("ROC curve — RBF Classifier (Credit Card)")
plt.legend()
save_figure("rbf_credit_roc")
plt.show()
