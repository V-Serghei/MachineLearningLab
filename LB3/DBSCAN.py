# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Шаг 1: Загрузка данных
data = pd.read_csv("../data/lb3/cluster_moons.csv")
print(data.head())

# Шаг 2: Предварительная обработка данных
X = data[['X1', 'X2']].values

# Стандартизация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Шаг 3: Визуализация исходных данных
plt.figure(figsize=(8,5))
plt.scatter(X[:, 0], X[:, 1], c='blue', s=50)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Исходные данные - Moons')
plt.grid()
plt.show()

# Шаг 4: Кластеризация методом DBSCAN
# Параметры: eps - радиус окружности поиска, min_samples - минимальное количество точек для кластера
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# Шаг 5: Визуализация кластеров
plt.figure(figsize=(8,5))
unique_labels = set(labels)
colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'orange']
for label in unique_labels:
    cluster = X[labels == label]
    plt.scatter(cluster[:, 0], cluster[:, 1], s=50, c=colors[label % len(colors)], label=f'Кластер {label}')

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('DBSCAN Кластеризация - Moons')
plt.legend()
plt.grid()
plt.show()

# Шаг 6: Оценка качества кластеризации
if len(unique_labels) > 1:
    silhouette_avg = silhouette_score(X_scaled, labels)
    print(f'Средний силуэтный коэффициент: {silhouette_avg:.3f}')
else:
    print('Оценка силуэта невозможна, так как все точки в одном кластере.')
