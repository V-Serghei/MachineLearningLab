import os
import random
import shutil

# Папка, где лежат ИСХОДНЫЕ изображения:
SOURCE_DIR = "../data/lab5/archive/Garbage classification/Garbage classification"

# Папка, куда сохраним УЖАТЫЙ датасет:
TARGET_DIR = "../data/lab5/new_images"

# Сколько изображений брать из каждого класса
N_PER_CLASS = 20

# Список классов (папок): cardboard, glass, metal, paper, plastic, trash
classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Создаём TARGET_DIR и подпапки
os.makedirs(TARGET_DIR, exist_ok=True)

for c in classes:
    src_folder = os.path.join(SOURCE_DIR, c)
    dst_folder = os.path.join(TARGET_DIR, c)
    os.makedirs(dst_folder, exist_ok=True)

    # Все файлы в исходной папке класса
    all_files = os.listdir(src_folder)
    # Перемешиваем
    random.shuffle(all_files)

    # Берём первые N_PER_CLASS
    selected_files = all_files[:N_PER_CLASS]

    # Копируем
    for f in selected_files:
        shutil.copy(os.path.join(src_folder, f),
                    os.path.join(dst_folder, f))

print("✅ Подвыборка изображений успешно создана!")
