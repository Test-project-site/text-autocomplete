import torch
import numpy as np
import time
from rouge_score import rouge_scorer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path
import pandas as pd

# ============================================================
# 1. ЗАГРУЗКА МОДЕЛИ И ТОКЕНИЗАТОРА
# ============================================================

def load_transformer_model(model_name='distilgpt2', device='cpu'):
    """
    Загружает предобученную модель трансформера.
    
    Returns:
        model, tokenizer: загруженные модель и токенизатор
    """
    print(f"📥 Загрузка модели {model_name}...")
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Устанавливаем pad_token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ Модель загружена: {model.config.n_embd} emb, {model.config.n_layer} layers")
    return model, tokenizer

# ============================================================
# 2. ФУНКЦИЯ ГЕНЕРАЦИИ ПРОДОЛЖЕНИЯ
# ============================================================

def generate_continuation_gpt2(model, tokenizer, input_text, 
                                max_length=30, temperature=1.0, 
                                top_k=50, top_p=0.9, device='cpu'):
    """
    Генерирует продолжение текста с помощью GPT-2.
    
    Args:
        model: модель трансформера
        tokenizer: токенизатор
        input_text: входной текст (3/4 исходного)
        max_length: максимальная длина генерации (в токенах)
        temperature: температура сэмплирования
        top_k: ограничение словаря
        top_p: nucleus sampling
        device: устройство
    
    Returns:
        generated_text: сгенерированное продолжение
    """
    model.eval()
    
    # Токенизация входа
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    input_length = input_ids.shape[1]
    
    # Параметры генерации
    do_sample = temperature != 1.0
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=input_length + max_length,
            min_length=input_length + 1,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    
    # Декодируем только сгенерированную часть
    generated_ids = output[0, input_length:].cpu().tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text.strip()

# ============================================================
# 3. ФУНКЦИЯ ОЦЕНКИ ROUGE
# ============================================================

def evaluate_rouge_transformer(model, tokenizer, test_pairs, 
                                generation_config, device='cpu', 
                                max_examples=500):
    """
    Считает метрики ROUGE для трансформера.
    
    Args:
        model: модель трансформера
        tokenizer: токенизатор
        test_pairs: список словарей {'input': str, 'target': str}
        generation_config: словарь с параметрами генерации
        device: устройство
        max_examples: максимальное количество примеров
    
    Returns:
        metrics: словарь с метриками и примерами
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=False)
    
    rouge1_scores = []
    rouge2_scores = []
    examples = []
    
    print(f"  Оценка на {min(len(test_pairs), max_examples)} примерах...")
    start_time = time.time()
    
    for i, pair in enumerate(test_pairs[:max_examples]):
        input_text = pair['input']
        target_text = pair['target']
        
        # Генерация
        generated_text = generate_continuation_gpt2(
            model, tokenizer, input_text,
            max_length=generation_config['max_length'],
            temperature=generation_config['temperature'],
            top_k=generation_config.get('top_k', 50),
            top_p=generation_config.get('top_p', 0.9),
            device=device
        )
        
        # Токенизация для метрики (по словам)
        ref_tokens = target_text.split()
        pred_tokens = generated_text.split()
        
        if not ref_tokens:
            continue
        
        scores = scorer.score(' '.join(ref_tokens), ' '.join(pred_tokens))
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        
        # Сохраняем примеры
        if len(examples) < 5:
            examples.append({
                'input': input_text,
                'reference': target_text,
                'generated': generated_text,
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure
            })
        
        # Прогресс
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"    Обработано {i + 1}/{min(len(test_pairs), max_examples)} ({elapsed:.1f}s)")
    
    elapsed = time.time() - start_time
    
    return {
        'rouge1_mean': np.mean(rouge1_scores) if rouge1_scores else 0,
        'rouge1_std': np.std(rouge1_scores) if rouge1_scores else 0,
        'rouge2_mean': np.mean(rouge2_scores) if rouge2_scores else 0,
        'rouge2_std': np.std(rouge2_scores) if rouge2_scores else 0,
        'examples': examples,
        'n_samples': len(rouge1_scores),
        'time_elapsed': elapsed
    }

# ============================================================
# 4. ПОДБОР ПАРАМЕТРОВ ГЕНЕРАЦИИ
# ============================================================

def tune_generation_params(model, tokenizer, val_pairs, 
                           param_grid, device='cpu', max_examples=200):
    """
    Подбирает оптимальные параметры генерации на валидации.
    
    Args:
        model: модель трансформера
        tokenizer: токенизатор
        val_pairs: валидационные данные
        param_grid: список конфигураций для проверки
        device: устройство
    
    Returns:
        best_config: лучшая конфигурация
        all_results: все результаты для анализа
    """
    print("\n" + "="*70)
    print("ПОДБОР ПАРАМЕТРОВ ГЕНЕРАЦИИ")
    print("="*70)
    
    best_config = None
    best_rouge1 = 0
    all_results = []
    
    for i, config in enumerate(param_grid, 1):
        print(f"\n[{i}/{len(param_grid)}] Тестирование конфигурации:")
        print(f"  {config}")
        
        metrics = evaluate_rouge_transformer(
            model, tokenizer, val_pairs, config, device, max_examples=max_examples
        )
        
        result = {
            'config': config,
            'rouge1': metrics['rouge1_mean'],
            'rouge2': metrics['rouge2_mean'],
            'time': metrics['time_elapsed']
        }
        all_results.append(result)
        
        print(f"  ROUGE-1: {metrics['rouge1_mean']:.4f} ± {metrics['rouge1_std']:.4f}")
        print(f"  ROUGE-2: {metrics['rouge2_mean']:.4f} ± {metrics['rouge2_std']:.4f}")
        print(f"  Время: {metrics['time_elapsed']:.1f}s")
        
        if metrics['rouge1_mean'] > best_rouge1:
            best_rouge1 = metrics['rouge1_mean']
            best_config = config
            print(f"  ⭐ Новый лидер!")
    
    print("\n" + "="*70)
    print(f"ЛУЧШАЯ КОНФИГУРАЦИЯ: {best_config}")
    print(f"ROUGE-1: {best_rouge1:.4f}")
    print("="*70)
    
    return best_config, all_results

# ============================================================
# 5. ВЫВОД И СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================

def print_transformer_results(metrics, model_name='DistilGPT-2'):
    """
    Красиво выводит результаты оценки трансформера.
    """
    print("\n" + "="*70)
    print(f"РЕЗУЛЬТАТЫ ОЦЕНКИ ({model_name})")
    print("="*70)
    print(f"  Примеров оценено: {metrics['n_samples']}")
    print(f"  ROUGE-1 F1: {metrics['rouge1_mean']:.4f} ± {metrics['rouge1_std']:.4f}")
    print(f"  ROUGE-2 F1: {metrics['rouge2_mean']:.4f} ± {metrics['rouge2_std']:.4f}")
    print(f"  Время оценки: {metrics['time_elapsed']:.1f}s")
    print("="*70)
    
    # Вывод примеров
    if metrics['examples']:
        print("\n📝 ПРИМЕРЫ ГЕНЕРАЦИИ:")
        print("-"*70)
        for i, ex in enumerate(metrics['examples'], 1):
            print(f"\n  Пример #{i}")
            print(f"  {'-'*60}")
            print(f"  Вход:      {ex['input'][:50]}...")
            print(f"  Ожидание:  {ex['reference']}")
            print(f"  Модель:    {ex['generated']}")
            print(f"  ROUGE-1:   {ex['rouge1']:.4f} | ROUGE-2: {ex['rouge2']:.4f}")

def save_transformer_results(metrics, config, filepath='results/transformer_metrics.csv'):
    """
    Сохраняет результаты оценки трансформера в CSV.
    """
    Path(filepath).parent.mkdir(exist_ok=True)
    
    # Метрики
    results_df = pd.DataFrame({
        'metric': ['rouge1_mean', 'rouge1_std', 'rouge2_mean', 'rouge2_std', 
                   'n_samples', 'time_elapsed'],
        'value': [
            metrics['rouge1_mean'],
            metrics['rouge1_std'],
            metrics['rouge2_mean'],
            metrics['rouge2_std'],
            metrics['n_samples'],
            metrics['time_elapsed']
        ]
    })
    results_df.to_csv(filepath, index=False, encoding='utf-8')
    
    # Конфигурация
    config_df = pd.DataFrame([config])
    config_df.to_csv(filepath.replace('.csv', '_config.csv'), index=False, encoding='utf-8')
    
    # Примеры
    if metrics['examples']:
        examples_df = pd.DataFrame(metrics['examples'])
        examples_path = filepath.replace('.csv', '_examples.csv')
        examples_df.to_csv(examples_path, index=False, encoding='utf-8')
    
    print(f"✅ Результаты сохранены в {filepath}")

def compare_lstm_vs_transformer(lstm_metrics, transformer_metrics):
    """
    Сравнивает LSTM и трансформер в таблице.
    """
    print("\n" + "="*70)
    print("СРАВНЕНИЕ: LSTM vs DistilGPT-2")
    print("="*70)
    print(f"  {'Модель':<20} {'ROUGE-1':<25} {'ROUGE-2':<25}")
    print(f"  {'-'*70}")
    
    lstm_r1 = f"{lstm_metrics['rouge1_mean']:.4f} ± {lstm_metrics['rouge1_std']:.4f}"
    lstm_r2 = f"{lstm_metrics['rouge2_mean']:.4f} ± {lstm_metrics['rouge2_std']:.4f}"
    trans_r1 = f"{transformer_metrics['rouge1_mean']:.4f} ± {transformer_metrics['rouge1_std']:.4f}"
    trans_r2 = f"{transformer_metrics['rouge2_mean']:.4f} ± {transformer_metrics['rouge2_std']:.4f}"
    
    print(f"  {'LSTM':<20} {lstm_r1:<25} {lstm_r2:<25}")
    print(f"  {'DistilGPT-2':<20} {trans_r1:<25} {trans_r2:<25}")
    print("="*70)
    
    # Определение победителя
    if transformer_metrics['rouge1_mean'] > lstm_metrics['rouge1_mean']:
        print("🏆 DistilGPT-2 показывает лучший ROUGE-1!")
    else:
        print("🏆 LSTM показывает лучший ROUGE-1!")
