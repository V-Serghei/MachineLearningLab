#%%
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import save_figure

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif

data = pd.read_csv("../data/gender_submission.csv")

# Two synthetic features added for feature-selection demonstration only.
data["Feature1"] = data["PassengerId"] % 3  # PassengerId modulo 3
data["Feature2"] = (data["PassengerId"] // 10) % 5  # (PassengerId // 10) modulo 5

# X — feature matrix: PassengerId, Feature1, Feature2
# y — target: whether the passenger survived (Survived)
X = data[["PassengerId", "Feature1", "Feature2"]]
y = data["Survived"]

# Replace missing values with column means
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 100)

print(data.head())
print(data.info())
print(data.describe())

# SelectKBest with f_classif (ANOVA F-test) scores each feature by its ability
# to explain variance in the target; select the top 2.
selector = SelectKBest(score_func=f_classif, k=2)
X_new = selector.fit_transform(X, y)

all_features = np.array(["PassengerId", "Feature1", "Feature2"])
selected_features = all_features[selector.get_support()]
print("Selected features:", selected_features)

# Split selected features (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=42)

# Train CART decision tree with limited depth to prevent overfitting
clf = DecisionTreeClassifier(random_state=42, max_depth=3)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

print("Classification report:")
print(classification_report(y_test, y_pred, digits=4))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
#%%

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# filled=True colours nodes by majority class
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=selected_features, class_names=["Not Survived", "Survived"])
plt.title("Decision tree (CART)")
save_figure("cart_tree")
plt.show()

# Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.title("Confusion matrix — CART")
save_figure("cart_cm")
plt.show()

# ROC curve — plots TPR vs. FPR at varying classification thresholds.
y_proba = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve — CART")
plt.legend(loc="lower right")
save_figure("cart_roc")
plt.show()

# Precision-Recall curve — especially useful for imbalanced class distributions.
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker=".", label="Precision-Recall curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall curve — CART")
plt.legend(loc="best")
save_figure("cart_pr_curve")
plt.show()

# Learning curve — shows training and CV accuracy vs. training set size.
# Helps detect overfitting (large train/CV gap) or underfitting (low scores on both).
train_sizes, train_scores, test_scores = learning_curve(
    clf, X, y, cv=5, scoring="accuracy", n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training set")
plt.plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation")
plt.title("Learning curve — CART")
plt.xlabel("Number of training samples")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.grid(True)
save_figure("cart_learning_curve")
plt.show()

# Class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x="Survived", hue="Survived", data=data, palette="pastel", legend=False)
plt.title("Class distribution (Survived / Not survived)")
plt.xlabel("Survived")
plt.ylabel("Sample count")
save_figure("cart_class_dist")
plt.show()
