import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import save_figure

# STEP 1: Import pandas, numpy, matplotlib, and seaborn.
print("Step 1")

import pandas as pds
import numpy
import matplotlib.pyplot as plt
import seaborn as sns

print("All libraries imported")
# Display all columns in the console
pds.set_option("display.max_columns", None)
pds.set_option("display.width", 1000)

# Step 2: Read the file movies.csv into a DataFrame named movies
print("Step 2")

movies = pds.read_csv("../data/movies.csv")
print("File loaded")

# Step 3: Print the first rows, info, and describe
print("Step 3")
print(movies.head())
print(movies.info())
print(movies.describe())

# Replace all missing data with mean values
movies["gross"] = movies["gross"].fillna(movies["gross"].mean())
movies["budget"] = movies["budget"].fillna(movies["budget"].mean())
movies["runtime"] = movies["runtime"].fillna(movies["runtime"].mean())
movies["score"] = movies["score"].fillna(movies["score"].mean())
movies["votes"] = movies["votes"].fillna(movies["votes"].mean())

# Remove random duplicates
movies.drop_duplicates(inplace=True)

# Use seaborn to create a jointplot to compare budget and gross.
# Observations show a correlation: budget data directly affects gross income
sns.jointplot(x="budget", y="gross", data=movies, kind="scatter")
plt.suptitle("Relationship between Box Office Gross and Budget", y=1.02)
save_figure("lr_budget_gross")
plt.show()

# STEP 4: Jointplot — relationship between runtime and gross income
# A correlation can be observed but it is very insignificant — more coincidence than dependency
sns.jointplot(x="runtime", y="gross", data=movies, kind="scatter")
plt.suptitle("Step 4: Relationship between Box Office Gross and Runtime", y=1.02)
save_figure("lr_runtime_gross")
plt.show()

# STEP 5 — Use jointplot() to compare runtime and score.
# Runtime doesn't particularly influence the score either.
sns.jointplot(x="runtime", y="score", data=movies, kind="hex")
plt.suptitle("Step 5: Relationship between Movie Runtime and Score", y=1.02)
save_figure("lr_runtime_score")
plt.show()

# STEP 6 — Investigate relationships with pairplot.
# The strongest correlation is between budget and gross income.
sns.pairplot(movies[["budget", "runtime", "score", "votes", "gross"]])
plt.suptitle("Step 6: Exploring All Features", y=1.02)
save_figure("lr_pairplot")
plt.show()

# STEP 7 — Linear regression model plot using lmplot
sns.lmplot(x="budget", y="gross", data=movies, line_kws={"color": "red"})
plt.title("Step 7: Linear Model - Gross Income vs. Budget")
save_figure("lr_lmplot")
plt.show()

# Step 8: Define X (features) and y (target: gross)
print("Step 8")
X = movies[["budget", "runtime", "score", "votes"]]
y = movies["gross"]

print("First 5 rows of features (X):")
print(X.head())
print("\nFirst 5 values of the target variable (y):")
print(y.head())

# Step 9: Train/test split
print("Step 9")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Size of X_train: {X_train.shape}")
print(f"Size of X_test: {X_test.shape}")

# Step 10-12: Train LinearRegression model
from sklearn.linear_model import LinearRegression

lm = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1)
lm.fit(X_train, y_train)
print("Model training complete.")

# Step 13: Print coefficients
print("Step 13")
print("Model coefficients:", lm.coef_)
print("Intercept:", lm.intercept_)

# Step 14: Predict on test set
y_pred = lm.predict(X_test)

# Step 15: Scatter plot of true vs. predicted values
print("Step 15 - On the graph")
plt.scatter(y_test, y_pred)
plt.xlabel("True Values (y_test)")
plt.ylabel("Predicted Values (y_pred)")
plt.title("Step 15 — True vs. Predicted Values")
save_figure("lr_true_vs_pred")
plt.show()

# Step 16: MAE, MSE, RMSE
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
print("Step 16")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Step 17: Residual histogram
print("Step 17 - On the graph")
residuals = y_test - y_pred
sns.histplot(residuals, kde=True, bins=20)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Step 17 — Residual Distribution")
save_figure("lr_residuals")
plt.show()

# Step 18: Coefficients DataFrame
print("Step 18")
coefficients = pds.DataFrame(lm.coef_, X.columns, columns=["Coefficient"])
print(coefficients)
print("Steps 19 and 20 - conclusions in the comments")

# Step 19: Coefficient interpretation
# budget: 2.54
#   An increase in budget by $1 increases gross income by an average of $2.54.
# runtime: -268634.47
#   An increase in runtime by 1 minute decreases gross income by ~$268,634.
# votes: 418.31
#   One more vote correlates with ~$418 more gross income.
# score: -1,637,712
#   Higher rating correlates with lower gross — not intuitive but driven by data.

# Step 20: What to prioritise to maximise gross income?
# Budget is the strongest positive predictor (higher budget → higher revenue).
# Votes also positively correlate (more viewers → more revenue).
# Rating and runtime are not reliable predictors of gross income.
