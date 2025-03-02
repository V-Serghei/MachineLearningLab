import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.impute import SimpleImputer

# 1. Загрузка данных
data = pd.read_csv('../data/gender_submission.csv')

# 2. Определяем признаки и целевую переменную
X = data[['PassengerId']]
y = data['Survived']

# 3. Обработка пропущенных значений
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# 4. Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Обучение модели CART (Decision Tree)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 6. Предсказание и оценка результатов
y_pred = clf.predict(X_test)
print("Отчет по классификации:")
print(classification_report(y_test, y_pred, digits=4))
print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

#############################################
# Дополнительные графики для анализа модели
#############################################

# График 1. Визуализация дерева решений
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=['PassengerId'], class_names=['Not Survived', 'Survived'])
plt.title("Дерево решений (CART)")
plt.show()

# График 2. Матрица ошибок (тепловая карта)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")
plt.title("Матрица ошибок")
plt.show()

# График 3. ROC-кривая
# Получаем вероятности для положительного класса
y_proba = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC кривая (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Ложно-положительная ошибка (False Positive Rate)')
plt.ylabel('Истинно-положительная ошибка (True Positive Rate)')
plt.title('ROC-кривая')
plt.legend(loc="lower right")
plt.show()

# График 4. Precision-Recall кривая
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label='Precision-Recall кривая')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall кривая')
plt.legend(loc='best')
plt.show()

# График 5. Кривая обучения
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
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', data=data, palette='pastel')
plt.title("Распределение классов (Выжили/Не выжили)")
plt.xlabel("Survived")
plt.ylabel("Количество примеров")
plt.show()
