# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 23:57:15 2023

@author: sviat
"""

import cv2
import os

def main(video_file, frame_rate, save_format="png", quality=0):
    # Проверка и создание папки для сохранения кадров
    filename = os.path.splitext(video_file)[0] + "-opencv"
    os.makedirs(filename, exist_ok=True)

    # Открытие видеофайла
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened(): 
        print("Could not open video file")
        return

    # Получение информации о видео
    fps = cap.get(cv2.CAP_PROP_FPS)  # Частота кадров в видеофайле
    frame_interval = round(fps / frame_rate)  # Интервал между сохраняемыми кадрами

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Общее количество кадров в видео
    max_frames = int(frame_count / frame_interval)  # Максимальное количество сохраняемых кадров
    duration = frame_count / fps  # Длительность видео в секундах

    # Вывод информации о видео
    print("Video Information:")
    print(f"Duration: {duration} seconds")
    print(f"Maximum frames: {max_frames}")
    print(f"Frames per second: {fps}")

    success, frame = cap.read()
    save_count = 0
    frame_count = 0

    while success:
        if frame_count % frame_interval == 0:
            saveframe_name = os.path.join(filename, f"frame_{save_count}.{save_format}")  # Имя файла кадра с выбранным форматом

            # Сохранение кадра с выбранным форматом и степенью сжатия / качеством
            try:
                if save_format.lower() == "png":
                    cv2.imwrite(saveframe_name, frame, [cv2.IMWRITE_PNG_COMPRESSION, quality])  # Сохранение в формате PNG с наилучшим качеством
                elif save_format.lower() == "jpg" or save_format.lower() == "jpeg":
                    cv2.imwrite(saveframe_name, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])  # Сохранение в формате JPEG с выбранным качеством
                else:
                    print("Invalid save format specified")
            except Exception as e:
                print(f"Exception in saving frame: {e}")
            else:
                print(f"{saveframe_name} saved successfully")

            save_count += 1

        success, frame = cap.read()
        frame_count += 1

    cap.release()

    # Проверка доступности записи в директорию
    try:
        with open(os.path.join(filename, "testfile.txt"), 'w') as f:
            f.write("Test")
    except Exception as e:
        print(f"Exception in writing to the directory: {e}")

    print("Frames saved successfully.")

# Вызов функции с видеофайлом, желаемой частотой сохранения кадров, форматом сохранения (по умолчанию PNG) и 
# наилучшим качеством
# при сохранении в формате JPEG и представляет собой значение от 0 до 100, где 0 означает наихудшее качество 
# (минимальное сжатие) и 100 означает наилучшее качество (без сжатия).
#main('C:\\Users\\sviat\\Desktop\\Video\\data_for_datasets_20.mov', 1, save_format="png", quality=0)
main('C:\\Users\\sviat\\Desktop\\Video\\data_for_datasets_20.mov', 1, save_format="jpg", quality=100)

