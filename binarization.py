import cv2                         #главная библиотека для компьютерного зрения и обработки изображений
import numpy as np                 #для работы с многомерными массивами и математическими операциями
import matplotlib.pyplot as mp     #для визуализации и построения графиков
from sklearn.datasets import load_sample_images     #Библиотека для машинного обучения, но содержит полезные datasets

dataset = load_sample_images()
image = dataset.images[1]

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

img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)   #преобразование изображения (с цветного в чб)
a, b = 255, 127

retval, result_bin1 = cv2.threshold(img, b, a, cv2.THRESH_BINARY) #пороговая бинаризация
result_bin2 = cv2.adaptiveThreshold(img, a, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2) #адаптивная бинаризация

displaying_image(image, result_bin1, result_bin2)
