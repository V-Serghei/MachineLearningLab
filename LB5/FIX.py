import os
import random
import shutil

# Source directory containing the full Garbage classification dataset.
SOURCE_DIR = "../data/lab5/archive/Garbage classification/Garbage classification"

# Target directory for the reduced subset used during training.
TARGET_DIR = "../data/lab5/new_images"

# Number of images to sample per class.
N_PER_CLASS = 20

classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

os.makedirs(TARGET_DIR, exist_ok=True)

for c in classes:
    src_folder = os.path.join(SOURCE_DIR, c)
    dst_folder = os.path.join(TARGET_DIR, c)
    os.makedirs(dst_folder, exist_ok=True)

    all_files = os.listdir(src_folder)
    random.shuffle(all_files)
    selected_files = all_files[:N_PER_CLASS]

    for f in selected_files:
        shutil.copy(os.path.join(src_folder, f), os.path.join(dst_folder, f))

print("Image subset created successfully.")
