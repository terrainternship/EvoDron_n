# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 11:09:51 2023

@author: sviat
"""

import os
import shutil
import random

ParentFolder = "D:\\Обучение\\pngdata"

# Путь к папке с изображениями
image_folder = os.path.join(ParentFolder, "jpg")

# Путь к папке с разметкой
label_folder = os.path.join(ParentFolder, "jpgtxt")

# Путь к папке с обучающими данными
train_folder = os.path.join(ParentFolder, "train")

# Путь к папке с тестовыми данными
test_folder = os.path.join(ParentFolder, "valid")


# Создание папок, если они не существуют
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)
os.makedirs(os.path.join(train_folder, "images"), exist_ok=True)
os.makedirs(os.path.join(train_folder, "labels"), exist_ok=True)
os.makedirs(os.path.join(test_folder, "images"), exist_ok=True)
os.makedirs(os.path.join(test_folder, "labels"), exist_ok=True)
os.makedirs(os.path.join(train_folder, "unpaired_files"), exist_ok=True)
os.makedirs(os.path.join(test_folder, "unpaired_files"), exist_ok=True)

# Получение списка файлов из папки с изображениями
image_files = [os.path.splitext(file)[0] for file in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, file))]

# Перемешивание списка файлов
random.shuffle(image_files)

Extention = ".png"
Extention = ".jpg"


# Определение количества файлов для обучающей и тестовой выборки
train_count = int(len(image_files) * 0.8)
test_count = len(image_files) - train_count

for i, file in enumerate(image_files):
    image_path = os.path.join(image_folder, file + Extention)
    label_path = os.path.join(label_folder, file + ".txt")

    if os.path.isfile(label_path):
        if i < train_count:
            # Копирование файлов в обучающую выборку
            shutil.copy(image_path, os.path.join(train_folder, "images"))
            shutil.copy(label_path, os.path.join(train_folder, "labels"))
        else:
            # Копирование файлов в тестовую выборку
            shutil.copy(image_path, os.path.join(test_folder, "images"))
            shutil.copy(label_path, os.path.join(test_folder, "labels"))
    else:
        # Копирование файлов без пар
        if i < train_count:
            shutil.copy(image_path, os.path.join(train_folder, "unpaired_files"))
        else:
            shutil.copy(image_path, os.path.join(test_folder, "unpaired_files"))

print("Разделение датасета завершено.")