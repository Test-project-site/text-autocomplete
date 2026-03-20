# Разделение на трейн, валидацию и тест
from sklearn.model_selection import train_test_split

# Формирование обучающих примеров (X → Y)
def create_training_sequences(tokens_list, seq_len=5):
   
    X, y = [], []
    
    for tokens in tokens_list:
        # Пропускаем слишком короткие последовательности
        if len(tokens) < seq_len + 1:
            continue
            
        # Скользящее окно по тексту
        for i in range(len(tokens) - seq_len):
            # Вход: токены [i : i+seq_len]
            # Таргет: токены [i+1 : i+seq_len+1] (сдвиг на 1)
            X.append(tokens[i : i + seq_len])
            y.append(tokens[i + 1 : i + seq_len + 1])
    
    return X, y

# Исходные данные: список списков токенов
all_tweets = df['tokens'].tolist()

# Шаг 1: Отделяем тест (10% от всего датасета)
train_tweets, test_tweets = train_test_split(
    all_tweets, 
    test_size=0.1, 
    random_state=42,  # Фиксируем для воспроизводимости результатов
    shuffle=True      # Перемешиваем перед разделением
)

# Шаг 2: От оставшихся 90% отделяем валидацию 
train_tweets, val_tweets = train_test_split(
    train_tweets, 
    test_size=0.1 / 0.9,  # Коэффициент для получения 10% от общего объема
    random_state=42,
    shuffle=True
)
# Шаг 3: # Сохранение - трейн, валидация и тест
pd.DataFrame({'tokens': train_tweets}).to_csv(
    'data/train.csv', 
    index=False, 
    encoding='utf-8')
pd.DataFrame({'tokens': test_tweets}).to_csv(
    'data/test.csv', 
    index=False, 
    encoding='utf-8')
pd.DataFrame({'tokens': val_tweets}).to_csv(
    'data/val.csv', 
    index=False, 
    encoding='utf-8')



# Шаг 4: # Формирование обучающих примеров
seq_len = 7  # Длина контекста
# Теперь создаём пары (X, y) независимо для каждой части
X_train, y_train = create_training_sequences(train_tweets, seq_len=seq_len)
X_val, y_val = create_training_sequences(val_tweets, seq_len=seq_len)
X_test, y_test = create_training_sequences(test_tweets, seq_len=seq_len)
