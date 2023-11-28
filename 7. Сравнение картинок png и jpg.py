# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 20:40:32 2023

@author: sviatoslav
"""
import time
import csv
from PIL import Image
import math
import os

def get_image_paths(directory, extension):
    """
    Возвращает все пути к файлам с заданным расширением в указанной директории.
    """
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(extension)]

def compare_images(image1_path, image2_path, step):
    """
    Сравнивает две картинки и возвращает процент их схожести.
    В случае ошибки бросает исключение.
    """
    # Открыть оба изображения
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

#    if image1.size != image2.size:
#        raise Exception(f"Картинки {image1_path} и {image2_path} имеют разные размеры")

    # Привести изображения к одному формату и грузить все пиксели
    image1 = image1.convert("RGB")
    image2 = image2.convert("RGB")

    pixels1 = image1.load()
    pixels2 = image2.load()

    width, height = image1.size
    diff_pixels = 0

    # Пройдитесь по каждому N-ому пикселю (определяется параметром шага)
    for x in range(0, width, step):
        for y in range(0, height, step):
            # Получить цвет каждого пикселя в формате RGB
            r1, g1, b1 = pixels1[x, y]
            r2, g2, b2 = pixels2[x, y]

            # Вычислитесь общий различия цветов
            diff_pixels += math.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)

    max_diff = math.sqrt(3 * (255 ** 2))
    similarity = (1 - (diff_pixels / ((width // step) * (height // step) * max_diff))) * 100

    return similarity

def find_best_match(image1_directory, image2_directory, csv_output, step):
    """
    Находит для каждой картинки из image1_directory самую похожую картинку в image2_directory.
    В случае ошибок при сравнении выводит сообщение об ошибке и пропускает пару изображений.
    Результаты записываются в CSV-файл.
    """
    image1_paths = get_image_paths(image1_directory, ".jpg")
    image2_paths = get_image_paths(image2_directory, ".png")
    start = time.time()

    with open(csv_output, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image 1", "Best Match", "Similarity", "Remaining Time"])

        total_images = len(image1_paths)
        processed_images = 0

        for image1_path in image1_paths:

            best_matches = [{"similarity": 0, "image2_path": ""} for _ in range(5)]

            for image2_path in image2_paths:
                try:
                    # Compare the current pair of images
                    similarity = compare_images(image1_path, image2_path, step)

                    for i, best_match in enumerate(best_matches):
                        if similarity > best_match["similarity"]:
                            best_matches.insert(i, {"similarity": similarity, "image2_path": image2_path})
                            best_matches.pop()
                            break
                except Exception as e:
                    print(str(e))

            best_match = {"similarity": 0, "image2_path": ""}
            for match in best_matches:
                similarity = compare_images(image1_path, match["image2_path"], step)
                if similarity > best_match["similarity"]:
                    best_match["similarity"] = similarity
                    best_match["image2_path"] = match["image2_path"]

            processed_images += 1
            elapsed = time.time() - start
            remaining_time = (total_images - processed_images) * (elapsed / processed_images)
            remaining_time = round(remaining_time, 2)

            image1_filename = os.path.basename(image1_path)
            best_match_filename = os.path.basename(best_match["image2_path"])
            print(f"Для картинки {image1_filename} максимальное совпадение с файлом {best_match_filename},"
                  f" процент совпадения: {best_match['similarity']:.2f}%, оставшееся время: {remaining_time:.2f} сек.")

            writer.writerow([image1_filename, best_match_filename, f"{best_match['similarity']:.2f}%", f"{remaining_time:.2f} seconds"])
            image2_paths.remove(best_match["image2_path"])

# Используйте параметр шага при вызове функции
#image1_directory = 'C:\\Users\\sviat\\Desktop\\Video\\data_for_datasets_20-jpg\\'
#image2_directory = 'C:\\Users\\sviat\\Desktop\\Video\\data_for_datasets_20-png\\'
image1_directory = 'C:\\Users\\sviat\\Desktop\\Video12\\images\\'
image2_directory = 'C:\\Users\\sviat\\Desktop\\Video12\\images - PNG\\'
csv_output = 'results.csv'
step = 10  # Измените это значение для определения количества пикселей за шаг
find_best_match(image1_directory, image2_directory, csv_output, step)



