import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ============================================================
# 1. DATASET КЛАСС
# ============================================================

class NextTokenDataset(Dataset):
    """
    Dataset для задачи предсказания следующего токена.
    Принимает пары (input_sequence, target_sequence) и конвертирует их в тензоры.
    """
    def __init__(self, X, y, token2idx):
        """
        Args:
            X: список входных последовательностей (списки токенов)
            y: список таргет-последовательностей (списки токенов)
            token2idx: словарь токен → индекс
        """
        self.data = []
        unk_idx = token2idx.get('<unk>', 1)
        
        for inp, tgt in zip(X, y):
            # Конвертируем токены в индексы
            inp_ids = [token2idx.get(t, unk_idx) for t in inp]
            tgt_ids = [token2idx.get(t, unk_idx) for t in tgt]
            self.data.append((inp_ids, tgt_ids))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        inp_ids, tgt_ids = self.data[idx]
        return torch.tensor(inp_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

# ============================================================
# 2. COLLATE ФУНКЦИЯ (для паддинга батчей)
# ============================================================

def collate_fn(batch, pad_idx=0):
    """
    Формирует батч из выборок, дополняя последовательности до макс. длины.
    
    Args:
        batch: список кортей (input_tensor, target_tensor)
        pad_idx: индекс токена паддинга
    
    Returns:
        input_padded, target_padded: тензоры размера (batch_size, max_seq_len)
    """
    input_seqs = [item[0] for item in batch]
    target_seqs = [item[1] for item in batch]
    
    # Паддинг до максимальной длины в батче
    input_padded = pad_sequence(input_seqs, batch_first=True, padding_value=pad_idx)
    target_padded = pad_sequence(target_seqs, batch_first=True, padding_value=pad_idx)
    
    return input_padded, target_padded

# ============================================================
# 3. ФУНКЦИЯ СОЗДАНИЯ DATALOADER
# ============================================================

def create_dataloader(X, y, token2idx, batch_size=64, shuffle=True, pad_idx=0):
    """
    Создаёт DataLoader из данных.
    
    Returns:
        DataLoader: готовый итератор для обучения/валидации
    """
    dataset = NextTokenDataset(X, y, token2idx)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=lambda b: collate_fn(b, pad_idx)
    )
    return loader
