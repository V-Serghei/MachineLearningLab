# STEP 1: Import pandas, numpy, matplotlib, and seaborn. Then set %matplotlib inline.
# (Later, import sklearn if necessary.)
print("Step 1")

import pandas as pds
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.ma.core import correlate

print("All libraries imported")
# Display all columns in the console
# Also extend the output width to fit in the console
pds.set_option('display.max_columns', None)
pds.set_option('display.width', 1000)

# Step 2: Read the file movies.csv into a DataFrame named movies
print("Step 2")

movies = pds.read_csv("../data/movies.csv")
print("File loaded")

# Step 3: Print the first rows (head) of the film data and get its information
# using the functions info() and describe().
print("Step 3")
print(movies.head())
print(movies.info())
print(movies.describe())

# Replace all missing data with mean values
movies['gross'] = movies['gross'].fillna(movies['gross'].mean())
movies['budget'] = movies['budget'].fillna(movies['budget'].mean())
movies['runtime'] = movies['runtime'].fillna(movies['runtime'].mean())
movies['score'] = movies['score'].fillna(movies['score'].mean())
movies['votes'] = movies['votes'].fillna(movies['votes'].mean())
# Unsure if this is allowed
# movies = movies[movies['gross'] < movies['gross'].quantile(0.95)]
# movies = movies[movies['budget'] < movies['budget'].quantile(0.95)]

# Remove random duplicates
movies.drop_duplicates(inplace=True)

# Use seaborn to create a jointplot to compare the columns budget and gross.
# Is there a correlation? Observations show a correlation: budget data directly affects gross income
sns.jointplot(x='budget', y='gross', data=movies, kind='scatter')
plt.suptitle("Relationship between Box Office Gross and Budget", y=1.02)
plt.show()

# STEP 4: Jointplot – relationship between runtime and gross income
# Regarding runtime and revenue, a correlation can be observed, but it is very insignificant,
# so it's more of a coincidence than a dependency
sns.jointplot(x='runtime', y='gross', data=movies, kind='scatter')
plt.suptitle("Step 4: Relationship between Box Office Gross and Runtime", y=1.02)
plt.show()

# STEP 5 - Use jointplot() to create a 2D plot comparing runtime and score.
# Similarly, runtime doesn't particularly influence the score either.
sns.jointplot(x='runtime', y='score', data=movies, kind='hex')
plt.suptitle("Step 5: Relationship between Movie Runtime and Score", y=1.02)
plt.show()

# STEP 6 - Investigate relationships in the entire dataset. Use pairplot to recreate
# the provided chart. (Don't worry about the colors.)
# Observing all relationships, the strongest correlation is between budget and gross income.
sns.pairplot(movies[['budget', 'runtime', 'score', 'votes', 'gross']])
plt.suptitle("Step 6: Exploring All Features", y=1.02)
plt.show()

# STEP 7
# Create a linear regression model plot using lmplot from seaborn for budget vs. gross data
sns.lmplot(x='budget', y='gross', data=movies, line_kws={'color': 'red'})
plt.title("Step 7: Linear Model - Gross Income vs. Budget")
plt.show()

# Step 8: Define variable X containing the numerical features of the movies,
# and variable y containing the gross column.
print("Step 8")
X = movies[['budget', 'runtime', 'score', 'votes']]
y = movies['gross']

print("First 5 rows of features (X):")
print(X.head())
print("\nFirst 5 values of the target variable (y):")
print(y.head())

# Step 9: Use model_selection.train_test_split from sklearn to split the data into
# training and testing datasets. Set test_size=0.3 and random_state=101.
print("Step 9")
from sklearn.model_selection import train_test_split

# Splitting the data into training and testing datasets
# 30% of the data goes into the testing set, 70% into the training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Size of X_train: {X_train.shape}")
print(f"Size of X_test: {X_test.shape}")
print(f"Size of y_train: {y_train.shape}")
print(f"Size of y_test: {y_test.shape}")

# Now train the model using the training dataset!
# Step 10: Import LinearRegression from sklearn.linear_model
from sklearn.linear_model import LinearRegression

# Step 11: Create an instance of the LinearRegression model named lm.
lm = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1)

# Step 12: Train lm using the training data.
lm.fit(X_train, y_train)
print("Steps 10, 11, 12: Imported lm, set it up for training, and trained it on the data.")
print("Model training complete.")

# Step 13: Print the model coefficients.
print("Step 13")
print("Model coefficients:", lm.coef_)
print("Intercept:", lm.intercept_)

# Assess the model's performance by predicting test values!
# Step 14: Use lm.predict() to predict values for X_test.
y_pred = lm.predict(X_test)
print("Step 14: Predicted values for X_test")

# Step 15: Create a scatterplot of the true test data vs. predicted values.
print("Step 15 - On the graph")
plt.scatter(y_test, y_pred)
plt.xlabel("True Values (y_test)")
plt.ylabel("Predicted Values (y_pred)")
plt.title("Step 15 - True vs. Predicted Values")
plt.show()

# Evaluate the model's performance by calculating the sum of squared residuals and the coefficient of determination (R^2).
# Step 16: Calculate the Mean Absolute Error (MAE), Mean Squared Error (MSE), and
# Root Mean Squared Error (RMSE). Refer to the lecture or Wikipedia for formulas.
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print("Step 16")

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Quickly investigate the residuals to ensure the data is valid.
# Step 17: Plot a histogram of the residuals to ensure they are normally distributed.
# Use either seaborn distplot or plt.hist().
print("Step 17 - On the graph")

residuals = y_test - y_pred
sns.histplot(residuals, kde=True, bins=20)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Step 17 - Residual Distribution")
plt.show()

# Still aiming to answer the original question: should we focus on the budget, or some other feature?
# What is the best indicator of gross income?
# Step 18: Recreate the dataset shown below
print("Step 18")

coefficients = pds.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
print(coefficients)
print("Steps 19 and 20 - conclusions in the comments")

# Step 19: Answer the question: How can these coefficients be interpreted?
# budget: 2.54
# An increase in a movie's budget by 1 dollar increases gross income by an average of 2.54 dollars
# runtime: -268634.47
# This means that an increase in movie runtime by 1 minute decreases gross income by an average of 268,634 dollars
# votes: 418.31
# This means that an increase in the number of audience votes by 1 correlates with an average gross income increase of 418 dollars
# score: -1.637712e+06
# Not logical, but the results indicate an increase in rating correlates with a decrease in gross income

# Step 20: What should be prioritized to maximize gross income?
# It is clear that the higher the budget, the higher the revenue (likely due to better effects, cast, etc.).
# Similarly, more votes indicate more people watched the movie, thus increasing gross revenue—this is logical.
# However, rating's and runtime's negative coefficients suggest that they are not the deciding factors.
# Ratings, especially at release, may not always influence revenue. Similarly, runtime is not a significant determinant.
