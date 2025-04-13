
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
import io

# Загрузка данных
try:
    df = pd.read_csv('../data/lab4/BankChurners.csv')
    print("Данные успешно загружены из BankChurners.csv")
except FileNotFoundError:
    print("Ошибка: Файл BankChurners.csv не найден.\nПожалуйста, сохраните ваши данные в этот файл.")
    exit()

# Удаляем ненужные столбцы
naive_bayes_cols = [col for col in df.columns if 'Naive_Bayes_Classifier' in col]
cols_to_drop = naive_bayes_cols + ['CLIENTNUM']
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
print(f"\nУдалены столбцы: {cols_to_drop}")

# Создание бинарной целевой переменной
df['churn_flag'] = df['Attrition_Flag'].apply(lambda x: 1 if x == 'Attrited Customer' else 0)

# Вывод информации о данных
print("\nОбзор данных:")
print(df.head())
print("\nОбщая информация:")
df.info()
print("\nСводная статистика:")
print(df.describe(include='all'))
print(f"\nКоличество строк: {len(df)}")
print("\nПропуски в данных:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Удаление строк с пропусками
df.dropna(inplace=True)
print(f"\nКоличество строк после удаления пропусков: {len(df)}")

# ======================================================================
# Визуализация данных (различные графики и диаграммы)
# ======================================================================
print("\n--- Начало визуализации данных ---")

# 1. Распределение целевой переменной
plt.figure(figsize=(8, 5))
sns.countplot(x='Attrition_Flag', hue='Attrition_Flag', data=df, palette='viridis', legend=False)
plt.title('Распределение статуса клиента (Отток)', fontsize=15)
plt.xlabel('Статус клиента', fontsize=12)
plt.ylabel('Количество', fontsize=12)
plt.tight_layout()
plt.show()

# 2. Гистограммы ключевых числовых признаков
key_numeric_features = ['Total_Trans_Amt', 'Total_Trans_Ct', 'Credit_Limit',
                        'Total_Revolving_Bal', 'Avg_Utilization_Ratio',
                        'Months_Inactive_12_mon', 'Contacts_Count_12_mon']

for feature in key_numeric_features:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[feature], kde=True, bins=50, color='skyblue')
    plt.title(f'Распределение признака: {feature}', fontsize=15)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Частота', fontsize=12)
    plt.tight_layout()
    plt.show()

# 3. Гистограммы с разделением по оттоку
for feature in key_numeric_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=feature, hue='Attrition_Flag', kde=True, bins=50, palette='viridis')
    plt.title(f'Распределение {feature} в зависимости от статуса клиента', fontsize=15)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Частота', fontsize=12)
    plt.legend(title='Статус клиента', labels=['Attrited (Отток)', 'Existing (Существующий)'])
    plt.tight_layout()
    plt.show()


# ======================================================================
# Анализ корреляции
# ======================================================================
numeric_df = df.select_dtypes(include=np.number)
correlation_with_churn = numeric_df.corr()['churn_flag'].sort_values(ascending=False)
print("\nКорреляция числовых признаков с churn_flag (1 = Отток):")
print(correlation_with_churn)

plt.figure(figsize=(15, 8))
correlation_with_churn.drop('churn_flag').plot(kind='bar', color='coral')
plt.title('Корреляция числовых признаков с оттоком клиентов (churn_flag = 1)', fontsize=16)
plt.xlabel('Признаки', fontsize=12)
plt.ylabel('Коэффициент корреляции', fontsize=12)
plt.xticks(rotation=60, ha='right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# 3. Корреляционная матрица признаков
plt.figure(figsize=(14, 10))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Корреляционная матрица признаков')
plt.tight_layout()
plt.show()


# 6. Диаграммы рассеяния для исследования взаимосвязей
if len(df) > 5000:
    df_sample = df.sample(n=5000, random_state=42)
else:
    df_sample = df.copy()

# a) Total_Trans_Amt vs Total_Trans_Ct
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_sample, x='Total_Trans_Ct', y='Total_Trans_Amt',
                hue='Attrition_Flag', palette='coolwarm', alpha=0.7,
                size='Months_Inactive_12_mon', sizes=(20, 200))
plt.title('Сумма vs Количество транзакций (цвет - отток, размер - неактивность)', fontsize=16)
plt.xlabel('Total_Trans_Ct', fontsize=12)
plt.ylabel('Total_Trans_Amt', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Статус клиента', fontsize=10)
plt.tight_layout()
plt.show()

# b) Credit_Limit vs Total_Revolving_Bal
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_sample, x='Total_Revolving_Bal', y='Credit_Limit',
                hue='Attrition_Flag', palette='coolwarm', alpha=0.7,
                size='Avg_Utilization_Ratio', sizes=(20, 200))
plt.title('Кредитный лимит vs Вращающийся баланс (цвет - отток, размер - коэф. исп.)', fontsize=16)
plt.xlabel('Total_Revolving_Bal', fontsize=12)
plt.ylabel('Credit_Limit', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Статус клиента', fontsize=10)
plt.tight_layout()
plt.show()

# c) Avg_Utilization_Ratio vs Total_Revolving_Bal
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_sample, x='Total_Revolving_Bal', y='Avg_Utilization_Ratio',
                hue='Attrition_Flag', palette='coolwarm', alpha=0.7,
                size='Contacts_Count_12_mon', sizes=(20, 200))
plt.title('Коэф. использования vs Вращающийся баланс (цвет - отток, размер - контакты)', fontsize=16)
plt.xlabel('Total_Revolving_Bal', fontsize=12)
plt.ylabel('Avg_Utilization_Ratio', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Статус клиента', fontsize=10)
plt.tight_layout()
plt.show()

# d) Months_Inactive_12_mon vs Total_Trans_Ct
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_sample, x='Months_Inactive_12_mon', y='Total_Trans_Ct',
                hue='Attrition_Flag', palette='coolwarm', alpha=0.7,
                size='Total_Trans_Amt', sizes=(20, 200))
plt.title('Месяцы неактивности vs Кол-во транзакций (цвет - отток, размер - сумма транз.)', fontsize=16)
plt.xlabel('Months_Inactive_12_mon', fontsize=12)
plt.ylabel('Total_Trans_Ct', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Статус клиента', fontsize=10)
plt.tight_layout()
plt.show()

# e) Дополнительная диаграмма: Credit_Limit vs Total_Trans_Amt
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_sample, x='Credit_Limit', y='Total_Trans_Amt',
                hue='Attrition_Flag', palette='Spectral', alpha=0.7)
plt.title('Credit_Limit vs Total_Trans_Amt', fontsize=16)
plt.xlabel('Credit_Limit', fontsize=12)
plt.ylabel('Total_Trans_Amt', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Статус клиента')
plt.tight_layout()
plt.show()

# f) Дополнительная диаграмма: Avg_Utilization_Ratio vs Contacts_Count_12_mon
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_sample, x='Avg_Utilization_Ratio', y='Contacts_Count_12_mon',
                hue='Attrition_Flag', palette='magma', alpha=0.7)
plt.title('Avg_Utilization_Ratio vs Contacts_Count_12_mon', fontsize=16)
plt.xlabel('Avg_Utilization_Ratio', fontsize=12)
plt.ylabel('Contacts_Count_12_mon', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Статус клиента')
plt.tight_layout()
plt.show()
# 2. Диаграмма рассеяния между Credit_Limit и Avg_Open_To_Buy
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Credit_Limit', y='Avg_Open_To_Buy', data=df,
                alpha=0.6, edgecolor='w')
plt.title('Связь между Credit Limit и Avg Open To Buy')
plt.xlabel('Credit Limit')
plt.ylabel('Avg Open To Buy')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 5. Boxplots для сравнения групп
for feature in key_numeric_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Attrition_Flag', y=feature, data=df, palette='coolwarm', hue='Attrition_Flag', legend=False)
    plt.title(f'Взаимосвязь {feature} и статуса клиента', fontsize=15)
    plt.xlabel('Статус клиента', fontsize=12)
    plt.ylabel(feature, fontsize=12)
    plt.tight_layout()
    plt.show()

# 4. Анализ категориальных признаков (общие распределения и с разделением по оттоку)
key_categorical_features = ['Gender', 'Education_Level', 'Marital_Status',
                            'Income_Category', 'Card_Category']

for feature in key_categorical_features:
    plt.figure(figsize=(12, 6))
    order = df[feature].value_counts().index
    sns.countplot(x=feature, data=df, order=order, palette='viridis', hue=feature, legend=False)
    plt.title(f'Распределение признака: {feature}', fontsize=15)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Количество', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 7))
    sns.countplot(x=feature, hue='Attrition_Flag', data=df, order=order, palette='coolwarm')
    plt.title(f'Распределение {feature} в зависимости от статуса клиента', fontsize=15)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Количество', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.legend(title='Статус клиента', labels=['Existing (0)', 'Attrited (1)'])
    plt.tight_layout()
    plt.show()







# ======================================================================
# Предобработка для моделирования
# ======================================================================
if 'Attrition_Flag' in df.columns:
    df.drop('Attrition_Flag', axis=1, inplace=True)
    print("\nСтолбец Attrition_Flag удалён.")

categorical_columns = list(df.select_dtypes(exclude=np.number).columns)
print(f"\nКатегориальные столбцы для преобразования в дамми: {categorical_columns}")
if categorical_columns:
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    print("\nDataFrame после преобразования:")
    print(df.head())
else:
    print("\nНет категориальных столбцов для преобразования.")

print("\nПроверка типов данных:")
print(df.info())

# ======================================================================
# Разделение данных на Train и Test
# ======================================================================
if 'churn_flag' not in df.columns:
    raise ValueError("Целевая переменная 'churn_flag' отсутствует в DataFrame!")

# Для сохранения имен признаков перед преобразованием в массив
feature_names = df.drop('churn_flag', axis=1).columns

X = df.drop('churn_flag', axis=1).values
y = df['churn_flag'].values
print("\nРазмерность X:", X.shape)
print("Размерность y:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=101,
                                                    stratify=y)
print("\nРазмеры выборок:")
print("X_train:", X_train.shape, " y_train:", y_train.shape)
print("X_test:", X_test.shape, "  y_test:", y_test.shape)
print("\nРаспределение классов в y_train:", np.bincount(y_train)/len(y_train))
print("Распределение классов в y_test:", np.bincount(y_test)/len(y_test))

# Нормализация данных
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("\nДанные нормализованы.")
print("Пример X_train после нормализации:", X_train[0, :10])

# ======================================================================
# Создание и обучение модели (фиксированная архитектура 78 → 39 → 19 → 1)
# ======================================================================
n_features = X_train.shape[1]
print(f"\nКоличество признаков для модели: {n_features}")

model = Sequential()
model.add(Dense(units=78, activation='relu', input_shape=(n_features,)))
model.add(Dropout(0.5))
model.add(Dense(units=39, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=19, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
print("\nНачало обучения модели...")
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=256,
                    validation_split=0.2,
                    callbacks=[early_stop],
                    verbose=1)
print("Обучение модели завершено.")

# Графики обучения
losses = pd.DataFrame(history.history)
plt.figure(figsize=(12, 5))
plt.plot(losses['loss'], label='Потери на обучении')
plt.plot(losses['val_loss'], label='Потери на валидации')
plt.title('График потерь модели')
plt.xlabel('Эпохи')
plt.ylabel('Loss (binary_crossentropy)')
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(losses['accuracy'], label='Точность на обучении')
plt.plot(losses['val_accuracy'], label='Точность на валидации')
plt.title('График точности модели')
plt.xlabel('Эпохи')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()

# ======================================================================
# Оценка модели на тестовых данных
# ======================================================================
print("\nОценка модели на тестовых данных:")
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

print("\nОтчет о классификации:")
print(classification_report(y_test, y_pred, target_names=['Existing (0)', 'Attrited (1)']))

print("\nМатрица ошибок:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Existing (0)', 'Attrited (1)'],
            yticklabels=['Existing (0)', 'Attrited (1)'])
plt.title('Матрица ошибок', fontsize=15)
plt.xlabel('Предсказано', fontsize=12)
plt.ylabel('Истинно', fontsize=12)
plt.show()

# ======================================================================
# Дополнительные визуализации
# ======================================================================

# 1. Важность признаков (вычисление суммарных абсолютных весов первого слоя)
# Для сохранения соответствия между признаками и весами используем feature_names
weights = model.layers[0].get_weights()[0]
importance = pd.Series(np.abs(weights).sum(axis=1), index=feature_names)
importance.nlargest(10).plot(kind='barh')
plt.title('Наиболее важные признаки для предсказания')
plt.xlabel('Сумма абсолютных весов')
plt.tight_layout()
plt.show()



print("\n--- Анализ завершен ---")
