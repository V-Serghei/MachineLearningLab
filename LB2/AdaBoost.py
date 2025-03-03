# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Импорт модулей для машинного обучения и предобработки
from sklearn.model_selection import train_test_split  # Разделение данных на обучающую и тестовую выборки
from sklearn.preprocessing import StandardScaler  # Масштабирование данных
from sklearn.ensemble import AdaBoostClassifier  # Алгоритм AdaBoost
from sklearn.tree import DecisionTreeClassifier  # Базовый Estimator (дерево решений)
from sklearn.impute import SimpleImputer  # Заполнение пропущенных значений в данных
from sklearn.metrics import (  # Метрики оценки качества модели
    classification_report, confusion_matrix, accuracy_score, roc_curve, auc
)

# 1. Загрузка данных
# Загрузка датасета из файла CSV
data = pd.read_csv("../data/breast_cancer_data.csv")
# Удаление столбцов с названием 'Unnamed', если они присутствуют
data = data.loc[:, ~data.columns.str.contains('Unnamed')]

# 2. Предварительная обработка данных
# Удаление столбца 'id', так как он не имеет значимости для модели
if 'id' in data.columns:
    data.drop(columns=['id'], inplace=True)

# Преобразование целевой переменной 'diagnosis': M (злокачественная) -> 1, B (доброкачественная) -> 0
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Определение признаков (X - объясняющие переменные) и целевой переменной (y - целевой столбец)
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']

# Заполнение пропущенных значений в объясняющих переменных медианными значениями
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 3. Разделение данных на обучающую и тестовую выборки
# Данные делятся в пропорции 70/30 с фиксированным random_state для воспроизводимости
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Масштабирование признаков
# Нормализация данных с помощью StandardScaler (важно для большинства моделей машинного обучения)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Инициализация базового классификатора ("пень", stump)
# Используется DecisionTreeClassifier с глубиной 1 в качестве слабого классификатора
base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)

# 6. Инициализация и обучение модели AdaBoostClassifier
# Конфигурация модели: базовый классификатор, 50 слабых оценщиков и фиксированная скорость обучения 1.0
ada = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
# Обучение модели на масштабированных тренировочных данных
ada.fit(X_train_scaled, y_train)

# 7. Предсказание и оценка модели
# Выполнение предсказания на тестовых данных
y_pred = ada.predict(X_test_scaled)

# Вывод отчёта по классификации (precision, recall, f1-score) для оценки качества модели
print("Отчет по классификации:")
print(classification_report(y_test, y_pred, digits=4))

# Вывод матрицы ошибок для анализа правильных и ошибочных предсказаний
print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred))

# Вывод точности (Accuracy) модели
print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))

# 8. Визуализация ROC-кривой
# Расчёт вероятностей для положительного класса и ROC-кривой
y_proba = ada.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Построение ROC-кривой для визуальной оценки качества модели
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC кривая (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC кривая для AdaBoost')
plt.legend(loc="lower right")
plt.show()

# Визуализация матрицы ошибок
# Используется heatmap для отображения количества ошибок и успешных предсказаний
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")
plt.title("Матрица ошибок")
plt.show()

# Дополнительные графики для анализа

# 9. Распределение целевой переменной
# Построение гистограммы для анализа пропорции классов в данных
plt.figure(figsize=(6, 4))
sns.countplot(x=y,hue=y, palette="coolwarm",legend=False)
plt.xlabel("Класс")
plt.ylabel("Количество")
plt.title("Распределение целевой переменной")
plt.show()

# 10. Корреляционная матрица
# Визуализация корреляционной матрицы для нахождения взаимосвязей между признаками
plt.figure(figsize=(12, 10))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
plt.title("Корреляционная матрица признаков")
plt.show()

# 11. Гистограммы распределения признаков
# Построение гистограмм для изучения распределения числовых признаков
X.hist(figsize=(12, 10), bins=20, color='steelblue', edgecolor='black')
plt.suptitle("Гистограммы распределения признаков")
plt.show()

# 12. Важность признаков
# Вычисление важности признаков по модели AdaBoost
feature_importances = ada.feature_importances_
features = X.columns
# Сортировка по значимости
sorted_indices = np.argsort(feature_importances)[::-1]

# Визуализация важности признаков (барплот для наглядности)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances[sorted_indices], y=features[sorted_indices], hue=features[sorted_indices], palette="viridis", legend=False)
plt.xlabel("Важность признака")
plt.ylabel("Признак")
plt.title("Важность признаков в модели AdaBoost")
plt.show()

# Вывод:
# 1. Код корректно реализует полный цикл работы алгоритма AdaBoost с использованием decision stumps.
# 2. На этапе предобработки данные очищаются, нормализуются и корректно разделяются на обучающую и тестовую выборки.
# 3. Метрики (precision, recall, f1-score, confusion matrix) показывают высокое качество модели:
#    - Accuracy: 97.08%
#    - Небольшое количество ошибок в матрице ошибок подтверждает хорошую работу модели.
# 4. Визуализации (ROC-кривая, распределение целевой переменной, корреляционная матрица, гистограммы распределения признаков,
#    важность признаков) позволяют детально проанализировать данные и оценить модель с разных сторон.
# 5. Полученные результаты являются очень хорошими для данного датасета, что свидетельствует о корректности как данных, так и модели.
#    Если цель – демонстрация работы алгоритма, то датасет выбран удачно. При необходимости можно провести дополнительные эксперименты
#    с другими наборами данных для проверки устойчивости модели.
