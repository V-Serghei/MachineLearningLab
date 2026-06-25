# Import required libraries
import pandas as pd                   # Tabular data handling
import numpy as np                    # Numerical computation
from sklearn.naive_bayes import ComplementNB  # Complement Naive Bayes classifier
from sklearn.model_selection import train_test_split  # Train/test split
from sklearn.metrics import accuracy_score, classification_report  # Model evaluation

# =======================
# 1. Synthetic dataset
# =======================
# Small dataset simulating spam filtering.
# Each row represents a message; columns are keyword occurrence counts.
# 'discount', 'win', 'prize', 'buy' — spam-indicative words.
# 'hello' — common in legitimate messages.
# Labels: 1 = spam, 0 = not spam.
data = {
    'discount': [2, 0, 1, 0, 3, 0, 0, 1, 2, 0],
    'win':      [1, 0, 0, 0, 2, 0, 0, 1, 1, 0],
    'prize':    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'buy':      [1, 0, 0, 0, 1, 0, 0, 1, 1, 0],
    'hello':    [0, 1, 0, 1, 0, 1, 1, 0, 0, 1]
}
labels = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0]  # 1 - spam, 0 - not spam

df = pd.DataFrame(data)
df['label'] = labels

# =======================
# 2. Prepare data for training
# =======================
X = df.drop('label', axis=1)
y = df['label']

# 70% training, 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# =======================
# 3. Train Complement Naive Bayes
# =======================
# alpha provides Laplace smoothing to avoid zero-probability issues
# when a feature is absent from the training set.
clf = ComplementNB(alpha=1.0)

clf.fit(X_train, y_train)

# =======================
# 4. Predict and evaluate
# =======================
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Complement Naive Bayes Classifier")
print("Model accuracy on test set:", accuracy)
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# =======================
# Notes:
# =======================
# 1. Libraries:
#    pandas and numpy for data handling.
#    ComplementNB implements Complement Naive Bayes.
#    train_test_split creates training and test splits.
#    accuracy_score and classification_report evaluate the model.
#
# 2. Dataset:
#    Models a spam vs. not-spam classification task.
#    Each column is the count of a keyword in a message.
#    Can be extended with more features or real data.
#
# 3. Data preparation: X/y split followed by train_test_split.
#
# 4. Training: ComplementNB with alpha smoothing, fitted via fit().
#
# 5. Evaluation: accuracy_score and classification_report.
#
# 6. Interpretation:
#    Complement NB is especially useful for imbalanced datasets because it models
#    the complement of each class rather than the class itself.
