# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 00:25:48 2023

@author: sviat
"""

import os
import openpyxl
from PIL import Image

# Путь к папке с фотографиями
folder_path = 'D:\\Обучение\\jpg - добавка\\moved_images'

# Создаем словарь для хранения информации о размерах фотографий и их количестве
size_count = {}

# Функция для определения размера фотографии
def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size

# Перебираем файлы в папке
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(folder_path, filename)
        size = get_image_size(image_path)
        size_str = f"{size[0]}x{size[1]}"
        size_count[size_str] = size_count.get(size_str, 0) + 1

# Создаем новую книгу Excel
workbook = openpyxl.Workbook()
worksheet = workbook.active
worksheet.title = "Фотографии"

# Заголовки для таблицы
worksheet.append(["Размер", "Количество"])

# Заполняем таблицу данными из словаря size_count
for size, count in size_count.items():
    worksheet.append([size, count])

# Сохраняем результаты в файл count.xlsx
workbook.save('D:\\Обучение\\count_jpg.xlsx')