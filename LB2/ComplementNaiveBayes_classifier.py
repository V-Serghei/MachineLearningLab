# Импорт необходимых библиотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Библиотеки для обработки текста и построения TF-IDF признаков
from sklearn.feature_extraction.text import TfidfVectorizer

# Импорт модели Комплементарного наивного Байеса
from sklearn.naive_bayes import ComplementNB

# Импорт функций для разделения данных и оценки модели
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Импорт NMF для уменьшения размерности (сохранение неотрицательности)
from sklearn.decomposition import NMF
from matplotlib.colors import ListedColormap

# =====================================
# 1. Загрузка и предобработка данных
# =====================================
# Предполагается, что датасет находится в файле "spam.csv" и содержит колонки "Category" и "Message".
df = pd.read_csv("../data/spam.csv", encoding="latin-1")
# Оставляем только нужные колонки
df = df[['Category', 'Message']]

# Преобразуем метки: 'ham' → 0 (не спам), 'spam' → 1 (спам)
df['Label'] = df['Category'].map({'ham': 0, 'spam': 1})

print("Пример датасета:")
print(df.head())

# =====================================
# 2. Векторизация текста
# =====================================
# Преобразуем текст в числовые признаки с помощью TF-IDF векторизации.
vectorizer = TfidfVectorizer(stop_words='english')
X_features = vectorizer.fit_transform(df['Message'])  # Разреженная матрица признаков
y = df['Label'].values

# =====================================
# 3. Разделение на обучающую и тестовую выборки
# =====================================
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.3, random_state=42)

# =====================================
# 4. Обучение модели ComplementNB
# =====================================
clf = ComplementNB(alpha=1.0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Оценка результатов
acc = accuracy_score(y_test, y_pred)
print("\nРезультаты модели на тестовой выборке:")
print("Accuracy:", acc)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# =====================================
# 5. Визуализация результатов с помощью NMF (сохранение неотрицательности)
# =====================================
# Преобразуем разреженные матрицы в плотный формат
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()

# Применяем NMF для уменьшения размерности до 2-х (NMF сохраняет неотрицательные значения)
nmf = NMF(n_components=2, random_state=42)
X_train_nmf = nmf.fit_transform(X_train_dense)
X_test_nmf = nmf.transform(X_test_dense)

# Обучаем новый экземпляр модели ComplementNB на данных, преобразованных NMF
clf_nmf = ComplementNB(alpha=1.0)
clf_nmf.fit(X_train_nmf, y_train)


# Функция для построения решающих границ
def plot_decision_boundary(clf, X, y, title, step=0.01):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))

    # Предсказание классов для каждой точки сетки
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Строим контурную заливку
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(('lightgreen', 'salmon')))
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # Отображаем точки исходного набора данных
    for i, j in enumerate(np.unique(y)):
        plt.scatter(X[y == j, 0], X[y == j, 1],
                    color=ListedColormap(('green', 'red'))(i), label=f"Class {j}", edgecolor='k')
    plt.title(title)
    plt.xlabel('Компонента 1')
    plt.ylabel('Компонента 2')
    plt.legend()
    plt.show()


# Визуализация на обучающем наборе (NMF-пространство)
plot_decision_boundary(clf_nmf, X_train_nmf, y_train, title="ComplementNB - Обучающая выборка (NMF)")

# Визуализация на тестовой выборке (NMF-пространство)
plot_decision_boundary(clf_nmf, X_test_nmf, y_test, title="ComplementNB - Тестовая выборка (NMF)")
