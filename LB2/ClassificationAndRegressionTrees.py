import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Импортируем необходимые модули из scikit-learn:
# - train_test_split для разбиения данных на обучающую и тестовую выборки
# - learning_curve для построения кривой обучения
# - DecisionTreeClassifier и plot_tree для создания и визуализации дерева решений (алгоритм CART)
# - classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve для оценки качества модели
# - SimpleImputer для обработки пропущенных значений
# - SelectKBest и f_classif для отбора наиболее значимых признаков с использованием статистического теста ANOVA
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif

# 1. Загрузка данных
# Загружаем CSV-файл с данными. В данном примере используется датасет gender_submission.csv,
# который содержит столбцы 'PassengerId' и 'Survived'. Обычно этот датасет применяется в задачах предсказания выживания (например, в задаче Titanic).
data = pd.read_csv('../data/gender_submission.csv')

# Для примера создаём ещё два искусственных признака, чтобы иметь возможность выбрать из нескольких признаков.
# Эти признаки не обязательно имеют практический смысл, но позволяют продемонстрировать работу алгоритма отбора признаков.
data['Feature1'] = data['PassengerId'] % 3       # Остаток от деления PassengerId на 3
data['Feature2'] = (data['PassengerId'] // 10) % 5  # Остаток от деления целой части от деления PassengerId на 10 на 5

# 2. Определяем признаки и целевую переменную
# X – матрица признаков, включающая 'PassengerId', 'Feature1' и 'Feature2'
# y – целевая переменная, показывающая, выжил ли пассажир (Survived)
X = data[['PassengerId', 'Feature1', 'Feature2']]
y = data['Survived']

# 3. Обработка пропущенных значений
# Создаём объект SimpleImputer, который заменяет пропуски средним значением по столбцу.
# Это важный шаг для обеспечения корректной работы алгоритмов машинного обучения, которые не всегда умеют работать с пропущенными данными.
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# 4. Отбор признаков: выбираем 2 наиболее значимых признака
# Используем SelectKBest с функцией f_classif (ANOVA F-тест) для оценки значимости каждого признака.
# Выбираем те признаки, которые лучше всего объясняют дисперсию целевой переменной.
selector = SelectKBest(score_func=f_classif, k=2)
X_new = selector.fit_transform(X, y)

# Получаем названия выбранных признаков.
# Метод get_support() возвращает булев массив, где True означает, что признак был выбран.
all_features = np.array(['PassengerId', 'Feature1', 'Feature2'])
selected_features = all_features[selector.get_support()]
print("Выбранные признаки:", selected_features)
# Вывод: выбранные признаки, которые показали наилучшие результаты в тесте значимости

# 5. Разделение данных на обучающую и тестовую выборки
# Разбиваем данные с выбранными признаками (X_new) и целевую переменную (y) на обучающую (70%) и тестовую (30%) выборки.
# Параметр random_state=42 обеспечивает воспроизводимость результатов.
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=42)

# 6. Обучение модели CART (Decision Tree) с ограниченной глубиной
# Инициализируем дерево решений (CART) с максимальной глубиной max_depth=3, что помогает избежать переобучения.
# random_state=42 используется для воспроизводимости.
clf = DecisionTreeClassifier(random_state=42, max_depth=3)
clf.fit(X_train, y_train)  # Обучаем модель на обучающей выборке

# 7. Предсказание и оценка результатов
# Предсказываем классы для тестовой выборки
y_pred = clf.predict(X_test)

# Выводим отчёт по классификации, который включает в себя:
# - precision (точность) – отношение правильно предсказанных положительных наблюдений к общему числу предсказанных положительных наблюдений;
# - recall (полнота) – отношение правильно предсказанных положительных наблюдений к количеству фактических положительных наблюдений;
# - f1-score – гармоническое среднее между precision и recall.
print("Отчет по классификации:")
print(classification_report(y_test, y_pred, digits=4))

# Выводим матрицу ошибок (confusion matrix), которая показывает количество истинно положительных, истинно отрицательных,
# ложных положительных и ложных отрицательных предсказаний.
print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred))

# Вычисляем общую точность (accuracy) – долю правильно классифицированных примеров.
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Визуализация дерева решений с уменьшенным числом признаков
# Функция plot_tree используется для визуализации структуры дерева, где:
# - filled=True окрашивает узлы в зависимости от класса,
# - feature_names передаёт имена выбранных признаков,
# - class_names задаёт имена классов для удобства интерпретации.
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=selected_features, class_names=['Not Survived', 'Survived'])
plt.title("Дерево решений (CART) с уменьшенным числом признаков")
plt.show()

#############################################
# Дополнительные графики для анализа модели
#############################################

# График 1. Визуализация дерева решений
# (Дерево уже визуализировано выше с помощью plot_tree)

# График 2. Матрица ошибок (тепловая карта)
# Для наглядности матрицу ошибок также визуализируем с помощью тепловой карты (heatmap) библиотеки seaborn.
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")
plt.title("Матрица ошибок")
plt.show()

# График 3. ROC-кривая
# ROC-кривая (Receiver Operating Characteristic curve) отображает зависимость между True Positive Rate (TPR, чувствительность)
# и False Positive Rate (FPR, специфичность) при различных порогах классификации.
# Для её построения получаем вероятности принадлежности к положительному классу с помощью метода predict_proba.
y_proba = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)  # Вычисляем площадь под ROC-кривой (AUC)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC кривая (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Линия случайного классификатора
plt.xlabel('Ложно-положительная ошибка (False Positive Rate)')
plt.ylabel('Истинно-положительная ошибка (True Positive Rate)')
plt.title('ROC-кривая')
plt.legend(loc="lower right")
plt.show()

# График 4. Precision-Recall кривая
# Эта кривая отображает взаимосвязь между precision и recall при изменении порога.
# Особенно полезна в задачах с несбалансированными классами.
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label='Precision-Recall кривая')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall кривая')
plt.legend(loc='best')
plt.show()

# График 5. Кривая обучения
# Кривая обучения показывает, как изменяется точность модели на обучающей выборке и при кросс-валидации в зависимости от размера обучающей выборки.
# Это помогает выявить проблему переобучения или недообучения.
train_sizes, train_scores, test_scores = learning_curve(
    clf, X, y, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Обучающая выборка")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Кросс-валидация")
plt.title("Кривая обучения")
plt.xlabel("Количество обучающих примеров")
plt.ylabel("Точность")
plt.legend(loc="best")
plt.grid(True)
plt.show()

# График 6. Распределение классов в полном датасете
# Для оценки дисбаланса классов используем countplot, который показывает количество примеров каждого класса.
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived',hue='Survived', data=data, palette='pastel', legend=False)
plt.title("Распределение классов (Выжили/Не выжили)")
plt.xlabel("Survived")
plt.ylabel("Количество примеров")
plt.show()

################################################################################
# Итоговый анализ и выводы:
#
# 1. Загрузка и предобработка данных:
#    - Датасет gender_submission.csv загружается и обогащается искусственными признаками (Feature1, Feature2),
#      что позволяет продемонстрировать процесс отбора признаков.
#    - Обработка пропущенных значений с помощью SimpleImputer гарантирует, что отсутствуют проблемы,
#      связанные с пропущенными данными.
#
# 2. Отбор признаков:
#    - С помощью SelectKBest и f_classif выбираются два наиболее значимых признака из исходного набора.
#      Это упрощает модель, снижает вычислительную сложность и помогает избежать переобучения.
#
# 3. Разделение данных и обучение модели:
#    - Данные разделены на обучающую и тестовую выборки (70/30), что позволяет объективно оценить качество модели.
#    - Модель дерева решений (CART) с ограниченной глубиной (max_depth=3) обучается на обучающей выборке.
#
# 4. Оценка модели:
#    - Для оценки используются метрики: precision, recall, f1-score (отчет по классификации), матрица ошибок и общая точность (accuracy).
#    - Матрица ошибок позволяет увидеть, сколько объектов каждого класса было классифицировано верно и сколько допущено ошибок.
#
# 5. Дополнительные визуализации:
#    - Визуализация дерева решений помогает понять, как модель принимает решения.
#    - ROC-кривая с вычисленным AUC показывает способность модели различать классы.
#    - Precision-Recall кривая важна для оценки качества модели при несбалансированном распределении классов.
#    - Кривая обучения демонстрирует, как меняется точность модели с увеличением числа обучающих примеров,
#      что помогает выявить проблемы переобучения или недообучения.
#    - График распределения классов помогает оценить, насколько сбалансированы данные.
#
# 6. Общий вывод:
#    - Выводы по метрикам (precision, recall, f1-score, confusion_matrix) показывают, что модель имеет определённые
#      проблемы с предсказанием одного из классов (что может быть связано с выбором признаков или дисбалансом классов).
#    - Для улучшения качества модели рекомендуется использовать более информативные признаки (например, демографические данные)
#      и применять методы балансировки классов.
#
# Данный код демонстрирует полный цикл работы с данными для задачи классификации:
# - от загрузки и предобработки данных,
# - через отбор признаков и разделение выборок,
# - до обучения модели и её визуальной и количественной оценки.
################################################################################
