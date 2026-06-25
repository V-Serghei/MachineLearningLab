import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load heart.csv — contains patient records with a heart disease target label.
data = pd.read_csv('../data/heart.csv')

# Separate features (X) and target variable (y). 'target' is the class label.
X = data.drop(columns=['target'])
y = data['target'].copy()

# Visualise class distribution to assess balance before training.
plt.figure(figsize=(6, 4))
sns.countplot(x=y, hue=y, palette='coolwarm', legend=False)
plt.title('Class distribution')
plt.show()

# Stratified split — preserves class proportions in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 100)
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

# RBF transform: maps each sample to its Gaussian-weighted distances to cluster centres,
# enabling logistic regression to capture nonlinear patterns without an explicit kernel.
def rbf_transform(X, centers, sigma=1.0):
    return np.exp(-np.linalg.norm(X[:, np.newaxis, :] - centers, axis=2) ** 2 / (2 * sigma ** 2))
#%%

# Apply RBF transform to both splits.
sigma = 1.0
X_train_rbf = rbf_transform(X_train_scaled, centers, sigma)
X_test_rbf = rbf_transform(X_test_scaled, centers, sigma)

# Train logistic regression on RBF-transformed features.
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_rbf, y_train)

# Predict and compute overall accuracy.
y_pred = clf.predict(X_test_rbf)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.4f}')
#%%

# Classification report — precision, recall, f1-score per class.
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))
#%%

# Confusion matrix — shows correct vs. incorrect predictions per class.
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion matrix')
plt.show()
#%%

# ROC curve and AUC — summarise the model's discriminative ability.
y_prob = clf.predict_proba(X_test_rbf)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend()
plt.show()
#%%


# 1. Data: heart.csv contains heart disease risk data.
#
# 2. Train/test split: stratified to preserve class proportions.
#
# 3. Feature scaling: standardisation is critical for KMeans distances and RBF.
#
# 4. KMeans: 10 cluster centres extracted as RBF anchors.
#
# 5. RBF transform: projects feature space into Gaussian-weighted distances to cluster
#    centres, allowing logistic regression to capture nonlinear relationships.
#
# 6. Logistic regression: fitted on RBF-transformed features — a kernel-method-like
#    approach without an explicit kernel.
#
# 7. Evaluation:
#    Accuracy: 0.5902 (59.02%).
#    Class 0: precision=1.0000, recall=0.1071, f1-score=0.1935.
#    Class 1: precision=0.5690, recall=1.0000, f1-score=0.7253.
#    Strong asymmetry: all class-1 samples found but class-0 recall is very low.
#
# 8. Confusion matrix: most class-0 samples misclassified as class 1.
#
# 9. ROC / AUC: overall discriminative power is limited given the metric asymmetry.
#
# Conclusions:
# - Model is biased toward the positive class.
# - Recommendations: tune n_centers/sigma, apply class_weight='balanced', or
#   use algorithms more robust to imbalanced data.
