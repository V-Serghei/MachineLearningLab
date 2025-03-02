import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Импорт CN2 из Orange3
from Orange.classification import CN2Learner
from Orange.data import Table, Domain, DiscreteVariable

# 1. Загрузка данных
data = pd.read_csv("../data/mushrooms.csv")

# 2. Преобразуем категориальные данные в числовые индексы
for column in data.columns:
    data[column] = data[column].astype('category')

# Определяем X и y
X = data.drop(columns=['class'])
y = data['class']

# Разделяем на train и test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Создаем домен Orange3 с категориальными переменными
feature_vars = [DiscreteVariable(name, values=list(data[name].cat.categories)) for name in X.columns]
class_var = DiscreteVariable('class', values=list(y.cat.categories))
domain = Domain(feature_vars, class_var)

# 4. Преобразуем X_train и y_train в индексы категорий
X_train_enc = np.column_stack([X_train[col].cat.codes for col in X_train.columns])
y_train_enc = y_train.cat.codes.to_numpy().reshape(-1, 1)

X_test_enc = np.column_stack([X_test[col].cat.codes for col in X_test.columns])
y_test_enc = y_test.cat.codes.to_numpy().reshape(-1, 1)

# 5. Преобразуем в формат Orange3 Table
train_data = Table(domain, np.hstack((X_train_enc, y_train_enc)))
test_data = Table(domain, np.hstack((X_test_enc, y_test_enc)))

# 6. Обучение модели CN2
cn2 = CN2Learner()
model = cn2(train_data)

# 7. Предсказание на тестовой выборке
y_pred = np.array([model(x) for x in test_data])

# 8. Оценка результатов
print("\nОтчет по классификации:")
print(classification_report(y_test_enc, y_pred, digits=4))

# Выводим точность
accuracy = accuracy_score(y_test_enc, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# 9. Визуализация результатов

## 9.1. Матрица ошибок
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test_enc, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Матрица ошибок")
plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")
plt.show()

## 9.2. Гистограмма предсказанных классов
plt.figure(figsize=(6, 4))
sns.histplot(y_pred, bins=np.arange(len(class_var.values)+1)-0.5, kde=False, color="purple")
plt.xticks(range(len(class_var.values)), class_var.values)
plt.title("Распределение предсказанных классов")
plt.xlabel("Классы")
plt.ylabel("Частота")
plt.show()

## 9.3. Круговая диаграмма предсказанных классов
plt.figure(figsize=(6, 6))
plt.pie(np.bincount(y_pred), labels=class_var.values, autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
plt.title("Распределение предсказанных классов")
plt.show()

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

