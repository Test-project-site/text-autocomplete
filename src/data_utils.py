# Сбор данных. Обработка датасета

import re
import emoji
import pandas as pd


# Очистка и нормализация текста

def preprocess_tweet(text, remove_emoji=True):
    """
    Полный пайплайн очистки твита.
    """
    # 1. Приведение к нижнему регистру
    text = text.lower()
    
    # 2. Удаление ссылок (http, https, www)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # 3. Удаление упоминаний (@user) и хэштегов (#tag)
    # Если хэштеги важны для задачи, уберите '#\w+' из паттерна
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # 4. Удаление эмодзи
    if remove_emoji:
        text = emoji.replace_emoji(text, replace='')
    
    # 5. Удаление ВСЕХ не-буквенно-цифровых символов
    # Оставляем только буквы (латиница + кириллица), цифры и пробелы
    text = re.sub(r'[^а-яa-z0-9\s]', '', text)
    
    # 6. Нормализация пробелов (удаление лишних)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 7. Токенизация (разбиение на слова)
    tokens = text.split()
    
    return tokens



