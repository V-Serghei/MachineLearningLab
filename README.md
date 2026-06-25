# Machine Learning Lab

Educational Python project covering supervised and unsupervised machine learning algorithms across three lab exercises.

## Tech Stack

| Library | Purpose |
|---------|---------|
| Python 3.10+ | Runtime |
| pandas, numpy | Data manipulation |
| matplotlib, seaborn | Visualisation |
| scikit-learn | ML algorithms |
| Orange3 | CN2 rule-based classifier |
| scipy | Hierarchical clustering |

## Project Structure

```
MachineLearningLab/
├── LB1/                              # Lab 1 — Linear Regression
│   └── linear_regression.py          # Movie budget vs. gross revenue
├── LB2/                              # Lab 2 — Supervised Classification
│   ├── AdaBoost.py                   # AdaBoost on breast cancer dataset
│   ├── CN2.py                        # CN2 rule learner on mushrooms
│   ├── ClassificationAndRegressionTrees.py  # CART on Titanic data
│   ├── ComplementNaiveBayes_classifier.py   # Complement NB spam classifier
│   ├── RBFClassifier.py              # RBF + logistic regression (credit card)
│   └── RBFClassifierHeart.py         # RBF + logistic regression (heart disease)
├── LB3/                              # Lab 3 — Unsupervised Clustering
│   ├── K_Means.py                    # K-Means on mall customers
│   ├── DBSCAN.py                     # DBSCAN on moon-shaped data
│   ├── GMM.py                        # Gaussian Mixture Model on Iris
│   ├── Algomerative.py               # Agglomerative clustering on wholesale data
│   └── KMeans_p1.py                  # DBSCAN + PCA on Facebook Live data
├── guide/
│   └── kernel_svm.py                 # Kernel SVM (RBF kernel) example
├── data/                             # Datasets (see table below)
├── start.bat                         # Environment setup script
└── stop.bat                          # Shutdown script (no persistent services)
```

## Quick Start

```bash
# Option 1 — automated setup
start.bat

# Option 2 — manual setup
pip install pandas numpy matplotlib seaborn scikit-learn orange3 scipy

# Extract the credit card dataset (required by RBFClassifier.py only)
cd data && tar -xvf creditcard.tar.xz && cd ..

# Run any script from the repository root
python LB2/AdaBoost.py
python LB3/K_Means.py
```

## Datasets

| File | Used by |
|------|---------|
| `breast_cancer_data.csv` | AdaBoost |
| `mushrooms.csv` | CN2 |
| `gender_submission.csv` | CART |
| `spam.csv` | Complement Naive Bayes |
| `creditcard.tar.xz` | RBFClassifier — **extract before use** |
| `heart.csv` | RBFClassifierHeart |
| `movies.csv` | Linear Regression |
| `Social_Network_Ads.csv` | Kernel SVM |
| `lb3/Iris.csv` | GMM |
| `lb3/Mall_Customers.csv` | K-Means |
| `lb3/cluster_moons.csv` | DBSCAN |
| `lb3/Wholesale_customers_data.csv` | Agglomerative |
| `lb3/Live.csv` | DBSCAN + PCA |
