import cv2                         #главная библиотека для компьютерного зрения и обработки изображений
import numpy as np                 #для работы с многомерными массивами
import matplotlib.pyplot as mp     #для визуализации
import os
import argparse
import sys

def loading(image_path):

    """
    Загружает изображение и возвращает его в grayscale и color форматах.
    
    Args:
        image_path (str): Путь к файлу изображения
        
    Returns:
        Кортеж с grayscale и color версиями изображения или (None, None) в случае ошибки

    Raises:
        FileNotFoundError: Если файл не существует
        ValueError: Если файл не является допустимым изображением
    """

    try:
        #проверка существования файла
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Файл '{image_path}' не найден")
        
        #проверка, что это файл, а не директория
        if not os.path.isfile(image_path):
            raise ValueError(f"'{image_path}' является директорией, а не файлом")

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_output = cv2.imread(image_path, cv2.IMREAD_COLOR)

        #проверка успешной загрузки
        if img is None or img_output is None:
            raise ValueError(f"Ошибка: Не удалось загрузить изображение '{image_path}'.")
            sys.exit(1)
        
        return img, img_output

    except Exception as e:
        print(f"Ошибка при загрузке изображения: {e}")
        return None, None


def global_binarization(image: np.ndarray, threshold: int = 127, max_value: int = 255):

    """
    Применяет глобальную пороговую бинаризацию к изображению.
    
    Args:
        image: Входное изображение в grayscale
        
    Returns:
        np.ndarray: Бинаризованное изображение
    """

    _, result_bin = cv2.threshold(image, threshold, max_value, cv2.THRESH_BINARY)

    return result_bin


def adaptive_binarization(image: np.ndarray, max_value: int = 255, block_size: int = 11,  c: int = 2):
    """
    Применяет адаптивную бинаризацию к изображению.
    
    Args:
        image (np.ndarray): Входное изображение в grayscale

    Returns:
        np.ndarray: Бинаризованное изображение
    """

    result_bin = cv2.adaptiveThreshold(image, max_value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
    
    return result_bin


def binarize_image(image: np.ndarray):
    """
    Применяет оба типа бинаризации к изображению.
    
    Args:
        image (np.ndarray): Входное изображение в grayscale
        
    Returns:
        Кортеж с результатами глобальной и адаптивной бинаризации
    """

    global_bin = global_binarization(image)
    adaptive_bin = adaptive_binarization(image)
    return global_bin, adaptive_bin


def displaying_images(img_output, global_bin, adaptive_bin):
    
    """
    Отображает исходное и обработанные изображения.
    
    Args:
        img_output (np.ndarray): Исходное цветное изображение
        global_bin (np.ndarray): Результат глобальной бинаризации
        adaptive_bin (np.ndarray): Результат адаптивной бинаризации
    """

    #создание subplot 1x3
    fig, place_for_images = mp.subplots(1, 3, figsize=(15, 5))

    #отображение изображений
    place_for_images[0].imshow(img_output, cmap='viridis')
    place_for_images[0].set_title('Исходное изображение')

    place_for_images[1].imshow(global_bin, cmap='gray')
    place_for_images[1].set_title('Глобальная бинаризация')

    place_for_images[2].imshow(adaptive_bin, cmap='gray')
    place_for_images[2].set_title('Адаптивная бинаризация')

    mp.show()

def process_image(image_path: str) -> bool:
    """
    Основная функция обработки изображения.
    
    Args:
        image_path (str): Путь к файлу изображения
        
    Returns:
        bool: True если обработка прошла успешно, False в случае ошибки
    """
    try:
        #загрузка изображения
        img_gray, img_color = loading(image_path)
        if img_gray is None or img_color is None:
            return False
        
        #бинаризация
        global_bin, adaptive_bin = binarize_image(img_gray)
        
        #отображение результатов
        displaying_images(img_color, global_bin, adaptive_bin)
        
        return True
        
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        return False


def main():

    """Основная функция для запуска из командной строки."""

    parser = argparse.ArgumentParser(description='Обработка изображений')
    parser.add_argument('input_image', help='исходник')
    args = parser.parse_args()

    success = process_image(args.input_image)
    sys.exit(0 if success else 1)


if __name__ == "__main__": 
    main()
