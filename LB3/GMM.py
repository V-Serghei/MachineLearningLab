# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

# Шаг 1: Загрузка данных
data = pd.read_csv("../data/lb3/Iris.csv")
print(data.head())

# Преобразование категориальных данных в числовые
le = LabelEncoder()
data['Species'] = le.fit_transform(data['Species'])

# Выбираем признаки для кластеризации
X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values

# Стандартизация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Шаг 2: Визуализация исходных данных (3D-график)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='blue', s=50)
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')
plt.title('Исходные данные - Iris')
plt.show()

# Шаг 3: Кластеризация методом Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X_scaled)
labels = gmm.predict(X_scaled)

# Добавляем кластеры в данные
data['Cluster'] = labels

# Шаг 4: Визуализация кластеров (3D-график)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
colors = ['red', 'green', 'blue']
for cluster in np.unique(labels):
    cluster_points = X[labels == cluster]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], s=50, color=colors[cluster], label=f'Кластер {cluster}')

ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')
plt.title('Кластеризация методом GMM - Iris')
plt.legend()
plt.show()

# Шаг 5: Оценка качества кластеризации
silhouette_avg = silhouette_score(X_scaled, labels)
print(f'Средний силуэтный коэффициент: {silhouette_avg:.3f}')
