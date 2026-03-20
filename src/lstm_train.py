import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

# ============================================================
# 1. ФУНКЦИЯ ОБУЧЕНИЯ НА ОДНОЙ ЭПОХЕ
# ============================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, clip_grad=1.0):
    """
    Проводит обучение модели на одной эпохе.
    
    Returns:
        avg_loss: средняя функция потерь за эпоху
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0
    
    for input_batch, target_batch in train_loader:
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output, _ = model(input_batch)  # (batch, seq_len, vocab_size)
        
        # Reshape для CrossEntropyLoss
        output = output.view(-1, output.size(-1))  # (batch*seq_len, vocab_size)
        target_batch = target_batch.view(-1)       # (batch*seq_len)
        
        # Loss
        loss = criterion(output, target_batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (важно для RNN!)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        
        optimizer.step()
        
        # Учёт потерянных токенов
        total_loss += loss.item() * (target_batch != 0).sum().item()
        total_tokens += (target_batch != 0).sum().item()
    
    return total_loss / max(total_tokens, 1)

# ============================================================
# 2. ФУНКЦИЯ ВАЛИДАЦИИ
# ============================================================

def validate(model, val_loader, criterion, device):
    """
    Проводит валидацию модели.
    
    Returns:
        avg_loss: средняя функция потерь на валидации
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for input_batch, target_batch in val_loader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            output, _ = model(input_batch)
            
            output = output.view(-1, output.size(-1))
            target_batch = target_batch.view(-1)
            
            loss = criterion(output, target_batch)
            
            total_loss += loss.item() * (target_batch != 0).sum().item()
            total_tokens += (target_batch != 0).sum().item()
    
    return total_loss / max(total_tokens, 1)

# ============================================================
# 3. ГЛАВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ
# ============================================================

def train_model(model, train_loader, val_loader, token2idx, idx2token,
                epochs=10, lr=0.001, device='cpu', save_path='best_lstm_model.pth'):
    """
    Полный цикл обучения модели с валидацией и сохранением.
    
    Args:
        model: LSTM модель
        train_loader: DataLoader для обучения
        val_loader: DataLoader для валидации
        token2idx, idx2token: словари для генерации примеров
        epochs: количество эпох
        lr: скорость обучения
        device: устройство (cpu/cuda)
        save_path: путь для сохранения лучшей модели
    
    Returns:
        history: словарь с историей обучения (loss по эпохам)
    """
    # Loss и оптимизатор
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 = PAD
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Для отслеживания лучшей модели
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    print("\n" + "="*70)
    print("НАЧАЛО ОБУЧЕНИЯ LSTM")
    print("="*70)
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # === TRAIN ===
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # === VALIDATION ===
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        # === ВРЕМЯ ===
        epoch_time = time.time() - start_time
        
        # === ИСТОРИЯ ===
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # === ВЫВОД ===
        print(f"\nEpoch {epoch+1}/{epochs} | Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")
        
        # === СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  💾 Сохранена новая лучшая модель (val_loss={val_loss:.4f})")
        
        # === ПРИМЕРЫ ГЕНЕРАЦИИ (каждую 5-ю эпоху и последнюю) ===
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"\n  📝 Пример генерации (Epoch {epoch+1}):")
            from lstm_model import generate_text
            generated = generate_text(model, ['i', 'feel'], token2idx, idx2token, 
                                     max_len=10, temperature=0.8, device=device)
            print(f"    Вход: i feel")
            print(f"    Модель: {' '.join(generated)}")
    
    print("\n" + "="*70)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"Лучший Val Loss: {best_val_loss:.4f}")
    print("="*70)
    
    return history

# ============================================================
# 4. ФУНКЦИЯ ЗАГРУЗКИ МОДЕЛИ
# ============================================================

def load_model(model, checkpoint_path, device='cpu'):
    """
    Загружает веса модели из файла.
    """
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✅ Модель загружена из {checkpoint_path}")
    else:
        print(f"❌ Файл {checkpoint_path} не найден!")
    return model
