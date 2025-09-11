import cv2                         #главная библиотека для компьютерного зрения и обработки изображений
import numpy as np                 #для работы с многомерными массивами и математическими операциями
import matplotlib.pyplot as mp     #для визуализации и построения графиков
from sklearn.datasets import load_sample_images     #Библиотека для машинного обучения, но содержит полезные datasets

import sys
import argparse

#вывод преобразованных изображений + исходник
def displaying_image(img_output, result_1, result_2):
    #создание subplot 1x3
    fig, place_for_images = mp.subplots(1, 3, figsize=(15, 5))

    place_for_images[0].imshow(img_output, cmap='viridis')
    place_for_images[0].set_title('Исходное изображение')

    place_for_images[1].imshow(result_1, cmap='gray')
    place_for_images[1].set_title('Глобальная бинаризация')

    place_for_images[2].imshow(result_2, cmap='gray')
    place_for_images[2].set_title('Адаптивная бинаризация')

    mp.show()

#загрузка изображения
def loading_transformation_displaying(image):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)   #преобразование изображения в двумерный массив (с помощью grayscale)
    img_output = cv2.imread(image, cv2.IMREAD_COLOR)

    #проверка успешной загрузки
    if img is None or img_output is None:
        print(f"Ошибка: Не удалось загрузить изображение '{image}'.")
        sys.exit(1)

    retval, result_bin1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) #пороговая бинаризация
    result_bin2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2) #адаптивная бинаризация

    displaying_image(img_output, result_bin1, result_bin2)

#парсинг входных данных
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Обработка изображений')
    parser.add_argument('input_image', help='исходник')
    
    args = parser.parse_args()
    loading_transformation_displaying(args.input_image)
