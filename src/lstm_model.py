import torch
import torch.nn as nn

# ============================================================
# 1. LSTM МОДЕЛЬ
# ============================================================

class LSTMAutoCompleter(nn.Module):
    """
    LSTM модель для задачи автодополнения текста.
    Предсказывает следующий токен по входной последовательности.
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, 
                 num_layers=2, dropout=0.3, pad_idx=0):
        """
        Args:
            vocab_size: размер словаря
            embed_dim: размерность эмбеддингов
            hidden_dim: размерность скрытого состояния LSTM
            num_layers: количество слоёв LSTM
            dropout: коэффициент dropout
            pad_idx: индекс токена паддинга (для Embedding)
        """
        super(LSTMAutoCompleter, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Слой эмбеддингов
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        
        # LSTM слои
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Выходной полносвязный слой
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        """
        Прямой проход через модель.
        
        Args:
            x: тензор размера (batch_size, seq_len)
            hidden: скрытое состояние (h_n, c_n) для продолжения генерации
        
        Returns:
            output: предсказания для каждой позиции (batch_size, seq_len, vocab_size)
            hidden: обновлённое скрытое состояние
        """
        # Embedding: (batch, seq_len) → (batch, seq_len, embed_dim)
        embedded = self.dropout(self.embedding(x))
        
        # LSTM: (batch, seq_len, embed_dim) → (batch, seq_len, hidden_dim)
        if hidden is None:
            output, hidden = self.lstm(embedded)
        else:
            output, hidden = self.lstm(embedded, hidden)
        
        # Dropout после LSTM
        output = self.dropout(output)
        
        # Полносвязный слой: (batch, seq_len, hidden_dim) → (batch, seq_len, vocab_size)
        output = self.fc(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """
        Инициализирует скрытое состояние нулями.
        """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)
    
    def generate_next_token(self, x, hidden=None, temperature=1.0):
        """
        Предсказывает ОДИН следующий токен (для генерации).
        
        Args:
            x: тензор размера (batch_size, seq_len)
            hidden: скрытое состояние с предыдущего шага
            temperature: температура сэмплирования
        
        Returns:
            next_token: индекс предсказанного токена
            hidden: обновлённое скрытое состояние
        """
        self.eval()
        with torch.no_grad():
            output, hidden = self.forward(x, hidden)
            
            # Берём предсказание для последней позиции
            logits = output[:, -1, :] / temperature  # (batch, vocab_size)
            
            # Применяем softmax для получения вероятностей
            probs = torch.softmax(logits, dim=-1)
            
            # Сэмплируем из распределения
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            
            return next_token, hidden


# ============================================================
# 2. ФУНКЦИЯ ГЕНЕРАЦИИ ТЕКСТА
# ============================================================

def generate_text(model, start_tokens, token2idx, idx2token, 
                  max_len=20, temperature=1.0, device='cpu'):
    """
    Генерирует текст, начиная с заданных токенов.
    
    Args:
        model: обученная модель
        start_tokens: начальные токены (список строк или индексов)
        token2idx, idx2token: словари
        max_len: максимальная длина генерации
        temperature: температура сэмплирования
        device: устройство (cpu/cuda)
    
    Returns:
        Сгенерированный текст (список токенов)
    """
    model.eval()
    
    # Если на вход пришли токены (строки), конвертируем в индексы
    if isinstance(start_tokens[0], str):
        current_tokens = [token2idx.get(t, token2idx.get('<unk>', 1)) for t in start_tokens]
    else:
        current_tokens = start_tokens  # Уже индексы
    
    if not current_tokens:
        current_tokens = [2]  # Начинаем с <eos> или случайного токена
    
    hidden = None
    
    for _ in range(max_len):
        # Подготовка входа: (1, seq_len)
        x = torch.tensor([current_tokens], dtype=torch.long).to(device)
        
        # Предсказываем следующий токен
        next_token, hidden = model.generate_next_token(x, hidden, temperature=temperature)
        
        # Добавляем к последовательности
        current_tokens.append(next_token.item())
        
        # Останавливаемся если встретили <eos>
        if next_token.item() == token2idx.get('<eos>', 2):
            break
        
        # Для эффективности передаём только последний токен на следующем шаге
        current_tokens = current_tokens[-1:]
    
    # Конвертируем обратно в токены
    generated_tokens = [idx2token.get(i, '<unk>') for i in current_tokens]
    
    # Фильтруем специальные токены для вывода
    generated_tokens = [t for t in generated_tokens if t not in ['<pad>', '<unk>', '<eos>']]
    
    return generated_tokens
