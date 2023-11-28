# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 23:17:13 2023

@author: Вячеслав
"""

import os

# Определение путей к папкам
base_path = 'data'
subfolders = ['train', 'val']

# Создание основной папки и подпапок
if not os.path.exists(base_path):
    os.mkdir(base_path)

for subfolder in subfolders:
    subfolder_path = os.path.join(base_path, subfolder)
    if not os.path.exists(subfolder_path):
        os.mkdir(subfolder_path)

    # Создание папок labels и images в каждой из подпапок
    for inner_folder in ['labels', 'images']:
        inner_folder_path = os.path.join(subfolder_path, inner_folder)
        if not os.path.exists(inner_folder_path):
            os.mkdir(inner_folder_path)

print("Папки успешно созданы!")