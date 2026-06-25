import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# - train_test_split to split data into training and test sets.
# - classification_report, confusion_matrix, accuracy_score to evaluate the model.
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# CN2Learner from Orange3 — a rule-based classifier.
# Table, Domain, DiscreteVariable are needed to work with Orange data structures.
from Orange.classification import CN2Learner
from Orange.data import Table, Domain, DiscreteVariable

# 1. Load data
# mushrooms.csv contains mushroom records; 'class' indicates edible or poisonous.
data = pd.read_csv("../data/mushrooms.csv")

# 2. Cast all columns to category dtype.
# All features are categorical; this is required for encoding and Orange domain creation.
for column in data.columns:
    data[column] = data[column].astype('category')

# Define features (X) and target variable (y)
X = data.drop(columns=['class'])
y = data['class']

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 100)
#%%

print(data.head())
#%%
print(data.info())
#%%
print(data.describe())


#%%

# Split data into training and test sets.
# stratify=y keeps class proportions equal in both splits.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Create Orange3 Domain
# Domain describes features and the target variable for Orange algorithms.
# Each feature is a DiscreteVariable with its possible category values.
feature_vars = [DiscreteVariable(name, values=list(data[name].cat.categories)) for name in X.columns]
# Target variable 'class' is also a DiscreteVariable.
class_var = DiscreteVariable('class', values=list(y.cat.categories))
# Combine features and target into a single domain.
domain = Domain(feature_vars, class_var)

# 4. Encode categorical data as integer codes.
# Orange requires numeric input; cat.codes converts each category to an integer.
X_train_enc = np.column_stack([X_train[col].cat.codes for col in X_train.columns])
y_train_enc = y_train.cat.codes.to_numpy().reshape(-1, 1)

X_test_enc = np.column_stack([X_test[col].cat.codes for col in X_test.columns])
y_test_enc = y_test.cat.codes.to_numpy().reshape(-1, 1)

# 5. Convert to Orange3 Table format
train_data = Table(domain, np.hstack((X_train_enc, y_train_enc)))
test_data = Table(domain, np.hstack((X_test_enc, y_test_enc)))

# 6. Train CN2 model
# CN2Learner is a rule-based classifier that generates a set of classification rules.
cn2 = CN2Learner()
model = cn2(train_data)

# 7. Predict on the test set
y_pred = np.array([model(x) for x in test_data])
#%%

# 8. Evaluate results
# precision: fraction of correct positive predictions among all predicted positives.
# recall: fraction of correct positive predictions among all actual positives.
# f1-score: harmonic mean of precision and recall.
print("\nClassification report:")
print(classification_report(y_test_enc, y_pred, digits=4))
#%%

# Overall accuracy — fraction of correctly classified samples.
accuracy = accuracy_score(y_test_enc, y_pred)
print(f"Accuracy: {accuracy:.4f}")
#%%

# 9. Visualisations

## 9.1. Confusion matrix heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test_enc, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion matrix")
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.show()
#%%

## 9.2. Predicted class histogram
plt.figure(figsize=(6, 4))
sns.histplot(y_pred, bins=np.arange(len(class_var.values)+1)-0.5, kde=False, color="purple")
plt.xticks(range(len(class_var.values)), class_var.values)
plt.title("Predicted class distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()
#%%

## 9.3. Predicted class pie chart
plt.figure(figsize=(6, 6))
plt.pie(np.bincount(y_pred), labels=class_var.values, autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
plt.title("Predicted class distribution")
plt.show()
#%%

## 9.4. True vs. predicted classes scatter plot
y_test_num = y_test.cat.codes
y_pred_num = y_pred

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test_num)), y_test_num, color='blue', label='True classes', alpha=0.5)
plt.scatter(range(len(y_pred_num)), y_pred_num, color='red', label='Predicted classes', alpha=0.5)
plt.title("True vs. predicted classes")
plt.xlabel("Sample index")
plt.ylabel("Class")
plt.legend()
plt.show()
y_test_num = y_test.cat.codes
y_pred_num = y_pred
#%%

# Difference between true and predicted labels
errors = y_test_num - y_pred_num

plt.figure(figsize=(10, 6))
plt.stem(range(len(errors)), errors)
plt.title("Difference between true and predicted classes (errors)")
plt.xlabel("Sample index")
plt.ylabel("Difference (True - Predicted)")
plt.show()
#%%

# Data split:
#    stratify=y preserves class proportions in both splits.
#
# Orange3 Domain:
#    Built from all categorical features and the target variable.
#    Required to convert data into the Orange Table format.
#
# Categorical encoding:
#    cat.codes converts categories to integers for CN2 training.
#
# CN2 training:
#    CN2Learner generates a set of classification rules.
#
# Evaluation:
#    Precision, recall, f1-score, and accuracy ≈ 99.75% — model is well-trained.
#
# Visualisations:
#    Confusion matrix, histogram, pie chart, and scatter plot provide
#    a visual assessment of model quality and class distribution.
#
# Conclusions:
#    CN2 achieved accuracy ≈ 99.75% — excellent class separation in the mushroom dataset.
#    All metrics are very high for both classes.
#
