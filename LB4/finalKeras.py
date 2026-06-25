import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import save_figure

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping

# Load bank churners dataset — target column 'Attrition_Flag' indicates customer churn.
try:
    df = pd.read_csv("../data/lab4/BankChurners.csv")
    print("Data loaded successfully from BankChurners.csv")
except FileNotFoundError:
    print("Error: BankChurners.csv not found.")
    exit()

# Drop columns not useful for prediction.
naive_bayes_cols = [col for col in df.columns if "Naive_Bayes_Classifier" in col]
cols_to_drop = naive_bayes_cols + ["CLIENTNUM"]
df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

# Binary target: 1 = churned customer, 0 = existing customer.
df["churn_flag"] = df["Attrition_Flag"].apply(lambda x: 1 if x == "Attrited Customer" else 0)

print(df.head())
df.info()
print(df.describe(include="all"))
print(f"Total rows: {len(df)}")

df.dropna(inplace=True)
print(f"Rows after dropping nulls: {len(df)}")

# --- Class distribution ---
plt.figure(figsize=(8, 5))
sns.countplot(x="Attrition_Flag", hue="Attrition_Flag", data=df, palette="viridis", legend=False)
plt.title("Customer status distribution (Churn)", fontsize=15)
plt.xlabel("Customer status", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.tight_layout()
save_figure("keras_target_dist")
plt.show()

key_numeric_features = [
    "Total_Trans_Amt", "Total_Trans_Ct", "Credit_Limit",
    "Total_Revolving_Bal", "Avg_Utilization_Ratio",
    "Months_Inactive_12_mon", "Contacts_Count_12_mon",
]

# --- Correlation of numeric features with churn_flag ---
numeric_df = df.select_dtypes(include=np.number)
correlation_with_churn = numeric_df.corr()["churn_flag"].sort_values(ascending=False)
print("\nCorrelation of numeric features with churn_flag:")
print(correlation_with_churn)

plt.figure(figsize=(15, 8))
correlation_with_churn.drop("churn_flag").plot(kind="bar", color="coral")
plt.title("Correlation of numeric features with churn (churn_flag = 1)", fontsize=16)
plt.xlabel("Features", fontsize=12)
plt.ylabel("Correlation coefficient", fontsize=12)
plt.xticks(rotation=60, ha="right", fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
save_figure("keras_feat_corr")
plt.show()

# --- Full correlation matrix ---
plt.figure(figsize=(14, 10))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature correlation matrix")
plt.tight_layout()
save_figure("keras_corr_matrix")
plt.show()

# --- Scatter: transaction count vs amount, coloured by churn ---
df_sample = df.sample(n=min(5000, len(df)), random_state=42)
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df_sample, x="Total_Trans_Ct", y="Total_Trans_Amt",
    hue="Attrition_Flag", palette="coolwarm", alpha=0.7,
    size="Months_Inactive_12_mon", sizes=(20, 200),
)
plt.title("Transaction amount vs count (colour = churn, size = inactivity months)", fontsize=14)
plt.xlabel("Total_Trans_Ct")
plt.ylabel("Total_Trans_Amt")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
save_figure("keras_scatter_transactions")
plt.show()

# --- Preprocessing for modelling ---
if "Attrition_Flag" in df.columns:
    df.drop("Attrition_Flag", axis=1, inplace=True)

categorical_columns = list(df.select_dtypes(exclude=np.number).columns)
if categorical_columns:
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

feature_names = df.drop("churn_flag", axis=1).columns
X = df.drop("churn_flag", axis=1).values
y = df["churn_flag"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101, stratify=y
)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Model: 3-layer MLP with dropout for regularisation ---
n_features = X_train.shape[1]
model = Sequential([
    Dense(78, activation="relu", input_shape=(n_features,)),
    Dropout(0.5),
    Dense(39, activation="relu"),
    Dropout(0.5),
    Dense(19, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=15)
history = model.fit(
    X_train, y_train,
    epochs=100, batch_size=256,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1,
)

# --- Training curves ---
losses = pd.DataFrame(history.history)

plt.figure(figsize=(12, 5))
plt.plot(losses["loss"], label="Train loss")
plt.plot(losses["val_loss"], label="Validation loss")
plt.title("Model loss curve")
plt.xlabel("Epoch")
plt.ylabel("Loss (binary_crossentropy)")
plt.legend()
plt.grid(True, linestyle="--")
plt.tight_layout()
save_figure("keras_training_loss")
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(losses["accuracy"], label="Train accuracy")
plt.plot(losses["val_accuracy"], label="Validation accuracy")
plt.title("Model accuracy curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, linestyle="--")
plt.tight_layout()
save_figure("keras_training_accuracy")
plt.show()

# --- Evaluation ---
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Existing (0)", "Attrited (1)"]))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Existing (0)", "Attrited (1)"],
    yticklabels=["Existing (0)", "Attrited (1)"],
)
plt.title("Confusion matrix — Keras MLP (Churn)", fontsize=15)
plt.xlabel("Predicted")
plt.ylabel("True")
save_figure("keras_cm")
plt.show()

# --- Feature importance via first-layer weight magnitudes ---
weights = model.layers[0].get_weights()[0]
importance = pd.Series(np.abs(weights).sum(axis=1), index=feature_names)
importance.nlargest(10).plot(kind="barh")
plt.title("Top 10 most important features (sum of absolute weights)")
plt.xlabel("Sum of absolute weights")
plt.tight_layout()
save_figure("keras_feat_importance")
plt.show()

print("\n--- Analysis complete ---")
