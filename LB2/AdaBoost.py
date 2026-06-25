import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split  # Split data into training and test sets
from sklearn.preprocessing import StandardScaler  # Feature scaling
from sklearn.ensemble import AdaBoostClassifier  # AdaBoost algorithm
from sklearn.tree import DecisionTreeClassifier  # Base estimator (decision tree)
from sklearn.impute import SimpleImputer  # Fill missing values
from sklearn.metrics import (  # Model evaluation metrics
    classification_report, confusion_matrix, accuracy_score, roc_curve, auc
)


# 1. Load dataset from CSV
data = pd.read_csv("../data/breast_cancer_data.csv")
# Drop 'Unnamed' columns if present (may appear as an artifact in CSV files)
data = data.loc[:, ~data.columns.str.contains('Unnamed')]

# 2. Preprocessing
# Drop 'id' column — not informative for the model
if 'id' in data.columns:
    data.drop(columns=['id'], inplace=True)

# Encode target: M (malignant) -> 1, B (benign) -> 0
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Define features (X) and target variable (y)
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']

# Fill missing values with column medians
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 100)

print(data.head())

print(data.info())

print(data.describe())

# 3. Split into training and test sets (70/30 with fixed random_state for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# 4. Feature scaling — standardise with StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Initialise base estimator (decision stump: depth-1 tree)
base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)

# 6. Initialise and train AdaBoostClassifier — 50 weak estimators, learning_rate=1.0
ada = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
ada.fit(X_train_scaled, y_train)


# 7. Prediction and evaluation
y_pred = ada.predict(X_test_scaled)

# Classification report: precision, recall, f1-score per class
print("Classification report:")
print(classification_report(y_test, y_pred, digits=4))
# Precision: fraction of positive predictions that are correct.
# Recall: fraction of actual positives the model found.
# F1-score: harmonic mean of precision and recall.
# Model accuracy: 97% — well-balanced, very few errors.

# Confusion matrix
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Accuracy
print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))


# 8. ROC curve
y_proba = ada.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for AdaBoost')
plt.legend(loc="lower right")
plt.show()


# Confusion matrix heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.title("Confusion matrix")
plt.show()




# 9. Target variable distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=y, hue=y, palette="coolwarm", legend=False)
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Target variable distribution")
plt.show()


# 10. Correlation matrix
plt.figure(figsize=(12, 10))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
plt.title("Feature correlation matrix")
plt.show()


# 11. Feature distribution histograms
X.hist(figsize=(12, 10), bins=20, color='steelblue', edgecolor='black')
plt.suptitle("Feature distribution histograms")
plt.show()


# 12. Feature importances
feature_importances = ada.feature_importances_
features = X.columns
# Sort by descending importance
sorted_indices = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances[sorted_indices], y=features[sorted_indices], hue=features[sorted_indices], palette="viridis", legend=False)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.title("Feature importances in AdaBoost")
plt.show()

# Conclusions:
# 1. Full AdaBoost pipeline with decision stumps is implemented correctly.
# 2. Data is cleaned, standardised, and split during preprocessing.
# 3. Metrics show high model quality: Accuracy 97.08%, few errors in confusion matrix.
# 4. Visualisations (ROC curve, target distribution, correlation matrix,
#    feature histograms, feature importances) enable a detailed multi-faceted analysis.
# 5. Results are strong for this dataset. Further experiments with other datasets
#    can verify model robustness.
