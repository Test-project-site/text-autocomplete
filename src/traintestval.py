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


