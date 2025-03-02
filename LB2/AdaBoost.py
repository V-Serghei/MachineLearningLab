import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

# 1. Загрузка данных
data = pd.read_csv("../data/breast_cancer_data.csv")
# Удаляем пустые или ненужные столбцы
data = data.loc[:, ~data.columns.str.contains('Unnamed')]

# 2. Предварительная обработка данных
# Удаляем столбец id, так как он не несёт информативной нагрузки
if 'id' in data.columns:
    data.drop(columns=['id'], inplace=True)

# Преобразуем целевую переменную: M -> 1, B -> 0
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Определяем признаки (X) и целевую переменную (y)
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']

# Заполняем пропущенные значения медианой
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 3. Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Масштабирование признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Инициализация базового классификатора ("пень", stump)
base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)

# 6. Инициализация и обучение модели AdaBoost
ada = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
ada.fit(X_train_scaled, y_train)

# 7. Предсказание и оценка модели
y_pred = ada.predict(X_test_scaled)
print("Отчет по классификации:")
print(classification_report(y_test, y_pred, digits=4))
print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))

# 8. Визуализация ROC-кривой
y_proba = ada.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC кривая (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC кривая для AdaBoost')
plt.legend(loc="lower right")
plt.show()

# Визуализация матрицы ошибок
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")
plt.title("Матрица ошибок")
plt.show()

# Дополнительные графики для анализа

# 9. Распределение целевой переменной
plt.figure(figsize=(6, 4))
sns.countplot(x=y, palette="coolwarm")
plt.xlabel("Класс")
plt.ylabel("Количество")
plt.title("Распределение целевой переменной")
plt.show()

# 10. Корреляционная матрица
plt.figure(figsize=(12, 10))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
plt.title("Корреляционная матрица признаков")
plt.show()

# 11. Гистограммы распределения признаков
X.hist(figsize=(12, 10), bins=20, color='steelblue', edgecolor='black')
plt.suptitle("Гистограммы распределения признаков")
plt.show()

# 12. Важность признаков
feature_importances = ada.feature_importances_
features = X.columns
sorted_indices = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances[sorted_indices], y=features[sorted_indices], palette="viridis")
plt.xlabel("Важность признака")
plt.ylabel("Признак")
plt.title("Важность признаков в модели AdaBoost")
plt.show()