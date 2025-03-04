import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# - train_test_split для разбиения данных на обучающую и тестовую выборки.
# - Метрики classification_report, confusion_matrix, accuracy_score для количественной оценки модели.
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Импортируем CN2Learner из Orange3 – алгоритм, реализующий правило-ориентированное обучение (CN2).
# Также импортируем классы для работы с данными Orange (Table, Domain, DiscreteVariable).
from Orange.classification import CN2Learner
from Orange.data import Table, Domain, DiscreteVariable

# 1. Загрузка данных
# Загружаем датасет "mushrooms.csv", который содержит данные о грибах.
# Целевая переменная 'class' обычно указывает, съедобен гриб или ядовит.
data = pd.read_csv("../data/mushrooms.csv")

# 2. Преобразование категориальных данных в тип 'category'
# Так как все признаки в датасете являются категориальными, приводим каждый столбец к типу category.
# Это необходимо для последующей обработки, кодирования и корректного создания домена для Orange.
for column in data.columns:
    data[column] = data[column].astype('category')

# Определяем признаки (X) и целевую переменную (y)
# X – все столбцы, кроме 'class', а y – непосредственно 'class'.
X = data.drop(columns=['class'])
y = data['class']

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 100)
#%%

print(data.head())
#%%
print(data.info())
#%%
print(data.describe())


#%%

# Разделение данных на обучающую и тестовую выборки.
# Используем stratify=y, чтобы пропорции классов остались примерно одинаковыми в обоих наборах.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Создание домена Orange3
# Для работы с алгоритмами Orange необходимо создать Domain – описание набора признаков и целевой переменной.
# Каждый признак описывается как DiscreteVariable с указанием возможных значений (из категорий столбца).
feature_vars = [DiscreteVariable(name, values=list(data[name].cat.categories)) for name in X.columns]
# Целевая переменная 'class' также определяется как DiscreteVariable.
class_var = DiscreteVariable('class', values=list(y.cat.categories))
# Объединяем признаки и целевую переменную в один домен.
domain = Domain(feature_vars, class_var)

# 4. Преобразование категориальных данных в числовые коды
# Для обучения модели необходимо преобразовать категориальные данные в числовой формат.
# Для каждого столбца обучающей и тестовой выборки получаем числовые коды (cat.codes).
X_train_enc = np.column_stack([X_train[col].cat.codes for col in X_train.columns])
y_train_enc = y_train.cat.codes.to_numpy().reshape(-1, 1)

X_test_enc = np.column_stack([X_test[col].cat.codes for col in X_test.columns])
y_test_enc = y_test.cat.codes.to_numpy().reshape(-1, 1)

# 5. Преобразование данных в формат Orange3 Table
# Объединяем преобразованные признаки и целевую переменную и создаем объекты Table для обучающей и тестовой выборок.
train_data = Table(domain, np.hstack((X_train_enc, y_train_enc)))
test_data = Table(domain, np.hstack((X_test_enc, y_test_enc)))

# 6. Обучение модели CN2
# Инициализируем CN2Learner – правило-ориентированный классификатор.
# Обучаем модель на обучающем наборе данных.
cn2 = CN2Learner()
model = cn2(train_data)

# 7. Предсказание на тестовой выборке
# Для каждого объекта из тестового набора получаем предсказание с помощью обученной модели.
# Результаты собираем в массив y_pred.
y_pred = np.array([model(x) for x in test_data])
#%%

# 8. Оценка результатов
# - precision (точность) – доля верных положительных предсказаний среди всех положительных,
# - recall (полнота) – доля верных положительных предсказаний среди всех реальных положительных,
# - f1-score – гармоническое среднее между precision и recall.
print("\nОтчет по классификации:")
print(classification_report(y_test_enc, y_pred, digits=4))
#%%

# Вычисляем и выводим общую точность модели – долю правильно классифицированных объектов.
accuracy = accuracy_score(y_test_enc, y_pred)
print(f"Accuracy: {accuracy:.4f}")
#%%

# 9. Визуализация результатов

## 9.1. Матрица ошибок
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test_enc, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Матрица ошибок")
plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")
plt.show()
#%%

## 9.2. Гистограмма предсказанных классов
plt.figure(figsize=(6, 4))
sns.histplot(y_pred, bins=np.arange(len(class_var.values)+1)-0.5, kde=False, color="purple")
plt.xticks(range(len(class_var.values)), class_var.values)
plt.title("Распределение предсказанных классов")
plt.xlabel("Классы")
plt.ylabel("Частота")
plt.show()
#%%

## 9.3. Круговая диаграмма предсказанных классов
plt.figure(figsize=(6, 6))
plt.pie(np.bincount(y_pred), labels=class_var.values, autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
plt.title("Распределение предсказанных классов")
plt.show()
#%%

## 9.4. График сравнения реальных и предсказанных классов
y_test_num = y_test.cat.codes
y_pred_num = y_pred

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test_num)), y_test_num, color='blue', label='Реальные классы', alpha=0.5)
plt.scatter(range(len(y_pred_num)), y_pred_num, color='red', label='Предсказанные классы', alpha=0.5)
plt.title("Сравнение реальных и предсказанных классов")
plt.xlabel("Индекс примера")
plt.ylabel("Класс")
plt.legend()
plt.show()
y_test_num = y_test.cat.codes
y_pred_num = y_pred
#%%

# Вычисляем разницу между истинными и предсказанными метками
errors = y_test_num - y_pred_num

plt.figure(figsize=(10, 6))
plt.stem(range(len(errors)), errors)  # убрали use_line_collection
plt.title("Разница между истинными и предсказанными классами (Ошибки)")
plt.xlabel("Индекс примера")
plt.ylabel("Разница (True - Predicted)")
plt.show()
#%%

# Разделение данных:
#    - Данные разбиваются на обучающую и тестовую выборки с сохранением пропорций классов (stratify=y).
#
# Создание домена Orange3:
#    - Формируется Domain, включающий все категориальные признаки и целевую переменную.
#    - Это необходимо для корректного преобразования данных в формат Orange Table.
#
# Преобразование категориальных данных:
#    - Категориальные значения преобразуются в числовые коды (cat.codes), что позволяет обучать модель CN2.
#
# Обучение модели CN2:
#    - Модель CN2Learner обучается на преобразованных данных.
#    - CN2 – правило-ориентированный классификатор, который генерирует набор правил для классификации.
#
# Оценка модели:
#    - Выводятся стандартные метрики (precision, recall, f1-score) и общая точность (accuracy).
#    - Полученные результаты (accuracy ≈ 0.9975, очень высокие метрики) свидетельствуют о качественном обучении модели.
#
# Визуализация результатов:
#    - Матрица ошибок (heatmap) дает наглядное представление о количестве верных и ошибочных предсказаний.
#    - Гистограмма и круговая диаграмма показывают распределение предсказанных классов.
#    - График сравнения реальных и предсказанных классов позволяет увидеть расхождения между ними.
#
# Выводы:
#    - Модель CN2 продемонстрировала высокую точность классификации (accuracy ≈ 99.75%),
#      что указывает на отличное разделение классов в датасете грибов.
#    - Вычисленные метрики (precision, recall, f1-score) для обоих классов находятся на очень высоком уровне.
#    - Построенные графики (матрица ошибок, гистограмма, круговая диаграмма и scatter plot)
#      позволяют визуально оценить качество модели и распределение предсказанных классов.
#    - Дополнительная визуализация, например, набора правил или сравнительного бар-чарта метрик,
#      могла бы еще более явно продемонстрировать результаты работы модели.
#
