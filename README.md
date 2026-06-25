# Machine Learning Lab

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/V-Serghei/MachineLearningLab/actions/workflows/ci.yml/badge.svg)](https://github.com/V-Serghei/MachineLearningLab/actions/workflows/ci.yml)

> Educational Python project exploring supervised and unsupervised machine learning algorithms across six lab exercises — from linear regression to deep learning and computer vision.

---

## Results Summary

### Lab 2 — Supervised Classification

| Algorithm | Dataset | Samples | Accuracy |
|-----------|---------|---------|----------|
| AdaBoost (50 stumps) | Breast Cancer | 569 | **97.08 %** |
| CN2 Rule Learner | Mushrooms | 8 124 | **99.75 %** |
| CART Decision Tree | Titanic (synthetic) | 418 | ~85 % |
| Complement Naive Bayes | SMS Spam | 5 572 | **93.18 %** |
| RBF + Logistic Regression | Credit Card Fraud | 284 807 | ~59 % (imbalanced) |
| RBF + Logistic Regression | Heart Disease | 303 | ~82 % |

### Lab 3 — Unsupervised Clustering

| Algorithm | Dataset | Clusters | Silhouette |
|-----------|---------|----------|------------|
| K-Means | Mall Customers | 5 | ~0.55 |
| DBSCAN | Moon-shaped data | 2 | ~0.90 |
| Gaussian Mixture Model | Iris | 3 | ~0.57 |
| Agglomerative (Ward) | Wholesale Customers | 4 | ~0.45 |

### Lab 4 — Deep Learning (Keras)

| Model | Dataset | Samples | Accuracy |
|-------|---------|---------|----------|
| MLP 78→39→19→1 (Dropout) | Bank Churn (BankChurners) | 10 127 | ~85 % |

### Lab 5 — Convolutional Neural Network

| Model | Dataset | Classes | Task |
|-------|---------|---------|------|
| CNN (Keras) | Garbage Classification | 6 | Image classification |

### Lab 6 — Object Detection

| Model | Task |
|-------|------|
| YOLO | Real-time object detection on photos |

---

## Example Plots

### Lab 1 — Linear Regression
| Budget vs Gross | True vs Predicted |
|:---:|:---:|
| ![LR budget](results/lr_budget_gross.png) | ![LR pred](results/lr_true_vs_pred.png) |

### Lab 2 — Supervised Classification
| AdaBoost — ROC Curve | CART — Decision Tree |
|:---:|:---:|
| ![AdaBoost ROC](results/adaboost_roc.png) | ![CART tree](results/cart_tree.png) |

| ComplementNB — Decision Boundary | RBF — ROC Curve (Heart) |
|:---:|:---:|
| ![CNB boundary](results/cnb_boundary_test.png) | ![RBF ROC](results/rbf_heart_roc.png) |

### Lab 3 — Unsupervised Clustering
| K-Means — Customer Segmentation | DBSCAN — Moon Clusters |
|:---:|:---:|
| ![K-Means](results/kmeans_clusters.png) | ![DBSCAN](results/dbscan_clusters.png) |

| GMM — Iris 3D | Agglomerative — Dendrogram |
|:---:|:---:|
| ![GMM](results/gmm_clusters.png) | ![Dendrogram](results/agg_dendrogram_truncated.png) |

### Lab 4 — Keras Deep Learning
| Training Loss | Confusion Matrix |
|:---:|:---:|
| ![Keras loss](results/keras_training_loss.png) | ![Keras CM](results/keras_cm.png) |

| Feature Correlation | Feature Importance |
|:---:|:---:|
| ![Keras corr](results/keras_feat_corr.png) | ![Keras importance](results/keras_feat_importance.png) |

---

## Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.10+ | Runtime |
| pandas | ≥ 2.0 | Data manipulation |
| numpy | ≥ 1.24 | Numerical computation |
| matplotlib / seaborn | ≥ 3.7 / ≥ 0.12 | Visualisation |
| scikit-learn | ≥ 1.3 | ML algorithms |
| TensorFlow / Keras | ≥ 2.12 | Deep learning (LB4, LB5) |
| Orange3 | ≥ 3.35 | CN2 rule-based classifier |
| scipy | ≥ 1.11 | Hierarchical clustering |

---

## Project Structure

```
MachineLearningLab/
├── LB1/                              # Lab 1 — Linear Regression
│   └── linear_regression.py          # Movie budget vs. gross revenue
├── LB2/                              # Lab 2 — Supervised Classification
│   ├── AdaBoost.py
│   ├── CN2.py
│   ├── ClassificationAndRegressionTrees.py
│   ├── ComplementNaiveBayes_classifier.py
│   ├── RBFClassifier.py
│   └── RBFClassifierHeart.py
├── LB3/                              # Lab 3 — Unsupervised Clustering
│   ├── K_Means.py
│   ├── DBSCAN.py
│   ├── GMM.py
│   ├── Agglomerative.py
│   └── DBSCAN_live.py
├── LB4/                              # Lab 4 — Deep Learning
│   └── finalKeras.py                 # MLP churn prediction (TensorFlow/Keras)
├── LB5/                              # Lab 5 — CNN Image Classification
│   ├── FIX.py                        # Dataset sampler utility
│   └── convolutional_neural_network_model.ipynb
├── LB6/                              # Lab 6 — Object Detection
│   └── YOLO.ipynb                    # YOLO inference on test images
├── guide/
│   └── kernel_svm.py                 # Kernel SVM reference example
├── results/                          # Auto-generated plots (git-tracked)
├── utils.py                          # Shared save_figure() helper
├── requirements.txt
├── pyproject.toml                    # Black + Ruff config
├── .pre-commit-config.yaml
├── start.bat
└── stop.bat
```

---

## Quick Start

```bash
# Option A — automated setup
start.bat

# Option B — manual
pip install -r requirements.txt

# Extract the credit card dataset (required by RBFClassifier.py only)
cd data && tar -xvf creditcard.tar.xz && cd ..

# Run any script — plots save automatically to results/
python LB2/AdaBoost.py
python LB3/K_Means.py
python LB4/finalKeras.py
```

### Code quality tools (optional)

```bash
pip install pre-commit
pre-commit install        # runs black + ruff + nbstripout on every commit
pre-commit run --all-files
```

---

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
| `lb3/Live.csv` | DBSCAN Live |
| `lab4/BankChurners.csv` | Keras MLP (Churn) |
| `lab5/archive/` *(local only)* | CNN — Garbage Classification |

---

## License

[MIT](LICENSE) © 2024 Serghei Vistovschii
