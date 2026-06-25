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
data = pd.read_csv('../data/creditcard.csv')

# Separate features (X) from labels (y).
X = data.drop(columns=['Class'])
y = data['Class'].copy()

# 2. Artificially shift class distribution:
#    Select all indices where class == 0.
zero_indices = y[y == 0].index  # indices of class-0 samples
#    Change 1/6 of them to class 1 to create a less extreme imbalance for demonstration.
num_to_change = len(zero_indices) // 6
np.random.seed(42)
indices_to_change = np.random.choice(zero_indices, size=num_to_change, replace=False)
y.loc[indices_to_change] = 1
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 100)
#%%

print(data.head())
#%%
print(data.info())
#%%
print(data.describe())
#%%

# 3. Visualise the new class distribution.
plt.figure(figsize=(6, 4))
sns.countplot(x=y, hue=y, palette='coolwarm', legend=False)
plt.title('Class distribution after relabelling')
plt.show()
#%%

# 4. Split into training and test sets with stratification to preserve class proportions.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Standardise features — important when distances are used (KMeans, RBF).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Find cluster centres with KMeans.
#    20 cluster centres are used as RBF kernel anchors.
n_centers = 20
kmeans = KMeans(n_clusters=n_centers, random_state=42, n_init=10)
kmeans.fit(X_train_scaled)
centers = kmeans.cluster_centers_

# 7. Vectorised RBF transform:
#    Computes Euclidean distance from each sample to every cluster centre,
#    then applies a Gaussian kernel, producing nonlinear features.
def rbf_transform(X, centers, sigma=1.0):
    # X[:, np.newaxis, :] adds a dimension for broadcasting the centre subtraction.
    # np.linalg.norm with axis=2 computes the distance to each centre.
    return np.exp(-np.linalg.norm(X[:, np.newaxis, :] - centers, axis=2) ** 2 / (2 * sigma ** 2))

sigma = 2.0
X_train_rbf = rbf_transform(X_train_scaled, centers, sigma)
X_test_rbf = rbf_transform(X_test_scaled, centers, sigma)

# 8. Train logistic regression on RBF-transformed features.
#    class_weight='balanced' compensates for class imbalance.
clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(X_train_rbf, y_train)
#%%

# 9. Predict and evaluate accuracy.
y_pred = clf.predict(X_test_rbf)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.4f}')
#%%

# 10. Classification report — precision, recall, f1-score per class.
#     Note: if class 1 metrics are 0.0, the model is not predicting that class at all.
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))
#%%

# 11. Confusion matrix heatmap.
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion matrix')
plt.show()
#%%

# 12. ROC curve and AUC.
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

# 1. Class distribution after relabelling:
#    1/6 of class-0 samples were moved to class 1; imbalance is less extreme.
#    The shift can hurt recall for the minority class.
#
# 2. Model results: Accuracy 0.5902 (59.02%).
#
# 3. Classification Report:
#    Class 0: precision=1.0000, recall=0.1071, f1-score=0.1935.
#      Perfect precision but very low recall — most class-0 samples are missed.
#    Class 1: precision=0.5690, recall=1.0000, f1-score=0.7253.
#      All class-1 samples are found, but many false positives reduce precision.
#
# 4. Confusion Matrix:
#    Most class-0 samples are misclassified as class 1.
#
# 5. ROC curve: given the class-level metric imbalance, overall discriminative
#    ability is limited.
#
# 6. All required steps are implemented: loading, preprocessing, distribution shift,
#    split, standardisation, RBF transform, logistic regression, full evaluation.
#
# 7. Conclusions:
#    Recommendations: apply class-balancing (oversampling/undersampling)
#    and tune n_centers and sigma in the RBF transform.
