import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import save_figure

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.decomposition import NMF

#%%
# Load spam.csv — contains spam and ham messages.
df = pd.read_csv("../data/spam.csv", encoding="latin-1")
df = df[["Category", "Message"]]

# Map labels: 'ham' -> 0 (not spam), 'spam' -> 1 (spam)
df["Label"] = df["Category"].map({"ham": 0, "spam": 1})

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 100)
#%%
print(df.head())
#%%
print(df.info())
#%%
print(df.describe())
#%%
# TF-IDF vectorisation — weights each word by frequency in the document vs.
# rarity across the corpus, de-emphasising stop words automatically.
vectorizer = TfidfVectorizer(stop_words="english")
X_features = vectorizer.fit_transform(df["Message"])  # Sparse feature matrix
y = df["Label"].values  # 0 = not spam, 1 = spam
#%%
# 70/30 train/test split
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.3, random_state=42)

# Complement NB works well with sparse data typical of text classification.
clf = ComplementNB(alpha=1.0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("\nModel results on test set:")
print("Accuracy:", acc)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
#%%

# TF-IDF space has very high dimensionality. Apply NMF to reduce it to 2 components
# for decision-boundary plots. NMF is used because its output is non-negative,
# which is compatible with the NB model.
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()

nmf = NMF(n_components=2, random_state=42)
X_train_nmf = nmf.fit_transform(X_train_dense)
X_test_nmf = nmf.transform(X_test_dense)

# Train a second ComplementNB on NMF-reduced data for 2-D visualisation only.
clf_nmf = ComplementNB(alpha=1.0)
clf_nmf.fit(X_train_nmf, y_train)


def plot_decision_boundary(clf, X, y, title, step=0.01):
    """Plot 2-D decision boundary with overlaid data points."""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Contour fill shows the decision regions
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(("lightgreen", "salmon")))
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # Overlay the data points coloured by class
    for i, j in enumerate(np.unique(y)):
        plt.scatter(X[y == j, 0], X[y == j, 1], color=ListedColormap(("green", "red"))(i), label=f"Class {j}", edgecolor="k")
    plt.title(title)
    plt.xlabel("General information signal")
    plt.ylabel("Spam features")
    plt.legend()


#%%
# Decision boundary on training set (NMF space)
plt.figure(figsize=(8, 6))
plot_decision_boundary(clf_nmf, X_train_nmf, y_train, title="ComplementNB — Training set (NMF)")
save_figure("cnb_boundary_train")
plt.show()
#%%

# Decision boundary on test set (NMF space)
plt.figure(figsize=(8, 6))
plot_decision_boundary(clf_nmf, X_test_nmf, y_test, title="ComplementNB — Test set (NMF)")
save_figure("cnb_boundary_test")
plt.show()
#%%

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print("Classification Report:\n", classification_report(y_test, y_pred))
#%%

# Confusion matrix heatmap
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.title("Confusion matrix — ComplementNB")
save_figure("cnb_cm")
plt.show()
#%%

# ROC curve
y_proba = clf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve — ComplementNB")
plt.legend(loc="lower right")
save_figure("cnb_roc")
plt.show()
