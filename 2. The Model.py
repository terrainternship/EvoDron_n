# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 22:58:12 2023

@author: Вячеслав
"""

from ultralytics import YOLO
from pathlib import Path
import torch

# Проверка доступности GPU
print("Is GPU available? ", torch.cuda.is_available())

# Create YOLO model
try:
    #model = YOLO("YOLOv8x.yaml")
    #model = YOLO("YOLOv8x-seg.yaml")
    model = YOLO('yolov8n-seg.pt')
    #model = YOLO('yolov8n.pt')
    print("YOLO model created successfully.")
except Exception as e:
    print("Error creating YOLO model:", e)

# Обучение модели
data_config_path = Path(r"D:\data\config.yaml")

if data_config_path.exists():
    print(f"Config data file found at: {data_config_path}")
else:
    print(f"Error: Config data file not found: {data_config_path}")
    # Завершение выполнения программы, если файл конфигурации не найден
    sys.exit(1)

# Train YOLO model (YOLOv8x-seg.yaml) with custom data
try:
    print("Starting model training...")
    #results = model.train(data=data_config_path, epochs=1, verbose=True)
    results = model.train(data=data_config_path, epochs=1)
    
    print("Model training completed.")
except Exception as e:
    print("Error during model training:", e)
