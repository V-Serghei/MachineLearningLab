# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score

# Шаг 1: Загрузка данных
data = pd.read_csv("../data/lb3/Wholesale_customers_data.csv")
print(data.head())

# Выбор признаков для кластеризации
X = data[['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']].values

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Шаг 2: Построение дендрограммы для определения числа кластеров
plt.figure(figsize=(16, 10))
Z = linkage(X_scaled, method='ward')
dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=45, leaf_font_size=12, color_threshold=20)
plt.title('Дендрограмма агломеративной кластеризации')
plt.xlabel('Покупатели')
plt.ylabel('Евклидово расстояние')
plt.show()

plt.figure(figsize=(16, 10))
Z = linkage(X_scaled, method='ward')
dendrogram(Z, leaf_rotation=45, leaf_font_size=12, color_threshold=20)
plt.title('Дендрограмма агломеративной кластеризации')
plt.xlabel('Покупатели')
plt.ylabel('Евклидово расстояние')
plt.show()

# Шаг 3: Агломеративная кластеризация с выбранным числом кластеров (например, 4)
n_clusters = 4
agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
labels = agg_clustering.fit_predict(X_scaled)


# Шаг 4: Визуализация кластеров
plt.figure(figsize=(8, 5))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for cluster in np.unique(labels):
    cluster_points = X[labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=colors[cluster], label=f'Кластер {cluster+1}')

plt.xlabel('Fresh')
plt.ylabel('Milk')
plt.title('Кластеризация методом агломерации')
plt.legend()
plt.show()

# Шаг 5: Оценка качества кластеризации
silhouette_avg = silhouette_score(X_scaled, labels)
print(f'Средний силуэтный коэффициент: {silhouette_avg:.3f}')
