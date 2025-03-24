import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from datetime import datetime

# 1. Загружаем данные
df = pd.read_csv("../data/lb3/Live.csv")

# Убираем лишние столбцы
df.drop(columns=["status_id"], inplace=True)

# Кодируем категориальные данные (One-Hot Encoding для status_type)
df = pd.get_dummies(df, columns=["status_type"], drop_first=True)

# Преобразуем дату публикации в удобные признаки (день недели, час)
df["status_published"] = pd.to_datetime(df["status_published"])
df["day_of_week"] = df["status_published"].dt.dayofweek
df["hour"] = df["status_published"].dt.hour
df.drop(columns=["status_published"], inplace=True)  # Удаляем оригинальную дату

# Если есть другие ненужные колонки, удаляем их
for col in ["Column1", "Column2", "Column3", "Column4"]:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# 2. Стандартизация данных
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# 3. Запускаем DBSCAN
dbscan = DBSCAN(eps=2.0, min_samples=5)  # eps можно варьировать
df["DBSCAN_Cluster"] = dbscan.fit_predict(df_scaled)

#4. PCA для снижения размерности
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)
df["PCA1"] = df_pca[:, 0]
df["PCA2"] = df_pca[:, 1]

df_no_outliers = df[df["DBSCAN_Cluster"] != -1]
df_scaled_no_outliers = df_scaled[df["DBSCAN_Cluster"] != -1]



# 5. Визуализация кластеров
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df["PCA1"], y=df["PCA2"], hue=df["DBSCAN_Cluster"], palette="viridis", legend="full")
plt.xlabel("Главный компонент 1 (PCA1)")
plt.ylabel("Главный компонент 2 (PCA2)")
plt.title("DBSCAN Кластеризация")
plt.show()

