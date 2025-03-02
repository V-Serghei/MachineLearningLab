import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Загрузка данных
data = pd.read_csv('../data/creditcard.csv')
# Отделяем признаки от меток
X = data.drop(columns=['Class'])
y = data['Class']

# Визуализация распределения классов
plt.figure(figsize=(6,4))
sns.countplot(x=y, palette='coolwarm')
plt.title('Распределение классов')
plt.show()

# 2. Разделение данных (с учетом стратификации для сохранения пропорций классов)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Стандартизация признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Определение центров с помощью KMeans
n_centers = 10  # Можно подобрать оптимальное число центров
kmeans = KMeans(n_clusters=n_centers, random_state=42, n_init=10)
kmeans.fit(X_train_scaled)
centers = kmeans.cluster_centers_

# 5. Функция векторизованного RBF-преобразования
def rbf_transform(X, centers, sigma=1.0):
    # Вычисляем евклидовы расстояния между каждой точкой X и каждым центром
    # Затем применяем экспоненциальную функцию
    return np.exp(-np.linalg.norm(X[:, np.newaxis, :] - centers, axis=2) ** 2 / (2 * sigma ** 2))

# Применение RBF-преобразования к обучающей и тестовой выборкам
sigma = 1.0
X_train_rbf = rbf_transform(X_train_scaled, centers, sigma)
X_test_rbf = rbf_transform(X_test_scaled, centers, sigma)

# 6. Обучение логистической регрессии на преобразованных данных
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_rbf, y_train)

# Предсказание на тестовой выборке
y_pred = clf.predict(X_test_rbf)
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели: {accuracy:.4f}')

# 7. Вычисление дополнительных метрик
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказано')
plt.ylabel('Истинное значение')
plt.title('Матрица ошибок')
plt.show()

# 8. Построение ROC-кривой и расчет AUC
y_prob = clf.predict_proba(X_test_rbf)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend()
plt.show()
