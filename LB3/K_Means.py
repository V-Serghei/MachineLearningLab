# Импортируем необходимые библиотеки
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score

# Шаг 1: Загрузка данных
data = pd.read_csv("../data/lb3/Mall_Customers.csv")

# Предварительный просмотр данных
print(data.head())

# Шаг 2: Предварительная обработка данных
# Преобразуем пол из категориальных переменных в числовые
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])

# Выбираем только нужные столбцы для кластеризации
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Шаг 3: Визуализация исходных данных
plt.figure(figsize=(8,5))
plt.scatter(X.iloc[:,0], X.iloc[:,1], c='blue', s=50)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Исходные данные о клиентах')
plt.grid()
plt.show()

# Шаг 4: Определение оптимального количества кластеров (метод локтя)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, 'bx-')
plt.xlabel('Количество кластеров')
plt.ylabel('WCSS (внутрикластерная изменчивость)')
plt.title('Определение оптимального количества кластеров (метод локтя)')
plt.grid()
plt.show()

# Оптимальное количество кластеров — обычно выбирается визуально (например, 5)
n_clusters = 5

# Шаг 5: Выполнение кластеризации методом K-Means
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X)

# Добавим полученные кластеры в исходные данные для наглядности
X = X.copy()
X['Cluster'] = labels

# Шаг 6: Визуализация результатов кластеризации
plt.figure(figsize=(8,5))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']

for i in range(n_clusters):
    plt.scatter(X[X['Cluster'] == i].iloc[:,0],
                X[X['Cluster'] == i].iloc[:,1],
                s=50, c=colors[i], label=f'Кластер {i+1}')

# Визуализация центроидов
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c='yellow', marker='X', label='Центроиды')

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Кластеризация клиентов методом K-Means')
plt.legend()
plt.grid()
plt.show()

# Шаг 7: Оценка качества кластеризации
silhouette_avg = silhouette_score(X.iloc[:, :2], labels)
print(f'Средний силуэтный коэффициент для {n_clusters} кластеров: {silhouette_avg:.3f}')