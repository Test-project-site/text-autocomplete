import torch
import numpy as np
from rouge_score import rouge_scorer

# ============================================================
# 1. ФУНКЦИЯ ОЦЕНКИ ROUGE
# ============================================================

def evaluate_rouge(model, data_loader, token2idx, idx2token, 
                   device='cpu', max_examples=300, max_gen_len=15, temperature=0.8):
    """
    Считает метрики ROUGE-1 и ROUGE-2 на датасете.
    
    Args:
        model: обученная LSTM модель
        data_loader: DataLoader с данными (input, target)
        token2idx, idx2token: словари
        device: устройство (cpu/cuda)
        max_examples: максимальное количество примеров для оценки
        max_gen_len: максимальная длина генерации
        temperature: температура сэмплирования
    
    Returns:
        metrics: словарь с метриками и примерами
    """
    from lstm_model import generate_text
    
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=False)
    
    rouge1_scores = []
    rouge2_scores = []
    examples = []
    
    print(f"  Оценка на {max_examples} примерах...")
    
    with torch.no_grad():
        for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
            if len(rouge1_scores) >= max_examples:
                break
            
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            # Обрабатываем каждый пример в батче
            for i in range(input_batch.size(0)):
                if len(rouge1_scores) >= max_examples:
                    break
                
                # Получаем последовательности без паддинга
                input_seq = input_batch[i].cpu().tolist()
                target_seq = target_batch[i].cpu().tolist()
                
                # Убираем PAD токены (0)
                input_seq = [t for t in input_seq if t != 0]
                target_seq = [t for t in target_seq if t != 0]
                
                if not input_seq or not target_seq:
                    continue
                
                # Генерация продолжения
                generated = generate_text(
                    model, 
                    input_seq,  # Передаем индексы
                    token2idx, 
                    idx2token, 
                    max_len=max_gen_len, 
                    temperature=temperature, 
                    device=device
                )
                
                # Конвертируем таргет в токены
                ref_tokens = [idx2token.get(t, '<unk>') for t in target_seq]
                pred_tokens = generated
                
                # Убираем специальные токены для метрики
                ref_tokens = [t for t in ref_tokens if t not in ['<pad>', '<unk>', '<eos>']]
                pred_tokens = [t for t in pred_tokens if t not in ['<pad>', '<unk>', '<eos>']]
                
                if not ref_tokens:
                    continue
                
                # Считаем ROUGE
                scores = scorer.score(' '.join(ref_tokens), ' '.join(pred_tokens))
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                
                # Сохраняем примеры для отчёта
                if len(examples) < 5:
                    input_tokens = [idx2token.get(t, '<unk>') for t in input_seq 
                                   if idx2token.get(t, '<unk>') not in ['<pad>', '<unk>', '<eos>']]
                    examples.append({
                        'input': ' '.join(input_tokens),
                        'reference': ' '.join(ref_tokens),
                        'generated': ' '.join(pred_tokens),
                        'rouge1': scores['rouge1'].fmeasure,
                        'rouge2': scores['rouge2'].fmeasure
                    })
            
            # Прогресс
            if (batch_idx + 1) % 10 == 0:
                print(f"    Обработано батчей: {batch_idx + 1}")
    
    return {
        'rouge1_mean': np.mean(rouge1_scores) if rouge1_scores else 0,
        'rouge1_std': np.std(rouge1_scores) if rouge1_scores else 0,
        'rouge2_mean': np.mean(rouge2_scores) if rouge2_scores else 0,
        'rouge2_std': np.std(rouge2_scores) if rouge2_scores else 0,
        'examples': examples,
        'n_samples': len(rouge1_scores)
    }

# ============================================================
# 2. ФУНКЦИЯ ВЫВОДА РЕЗУЛЬТАТОВ
# ============================================================

def print_evaluation_results(metrics, model_name='LSTM'):
    """
    Красиво выводит результаты оценки в консоль.
    """
    print("\n" + "="*70)
    print(f"РЕЗУЛЬТАТЫ ОЦЕНКИ ({model_name})")
    print("="*70)
    print(f"  Примеров оценено: {metrics['n_samples']}")
    print(f"  ROUGE-1 F1: {metrics['rouge1_mean']:.4f} ± {metrics['rouge1_std']:.4f}")
    print(f"  ROUGE-2 F1: {metrics['rouge2_mean']:.4f} ± {metrics['rouge2_std']:.4f}")
    print("="*70)
    
    # Вывод примеров
    if metrics['examples']:
        print("\n📝 ПРИМЕРЫ ГЕНЕРАЦИИ:")
        print("-"*70)
        for i, ex in enumerate(metrics['examples'], 1):
            print(f"\n  Пример #{i}")
            print(f"  {'-'*60}")
            print(f"  Вход:      {ex['input']}")
            print(f"  Ожидание:  {ex['reference']}")
            print(f"  Модель:    {ex['generated']}")
            print(f"  ROUGE-1:   {ex['rouge1']:.4f} | ROUGE-2: {ex['rouge2']:.4f}")

# ============================================================
# 3. ФУНКЦИЯ СРАВНЕНИЯ МОДЕЛЕЙ
# ============================================================

def compare_models(metrics_dict):
    """
    Сравнивает несколько моделей в таблице.
    
    Args:
        metrics_dict: словарь {name: metrics}
    """
    print("\n" + "="*70)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("="*70)
    print(f"  {'Модель':<20} {'ROUGE-1':<20} {'ROUGE-2':<20}")
    print(f"  {'-'*60}")
    
    for name, metrics in metrics_dict.items():
        rouge1 = f"{metrics['rouge1_mean']:.4f} ± {metrics['rouge1_std']:.4f}"
        rouge2 = f"{metrics['rouge2_mean']:.4f} ± {metrics['rouge2_std']:.4f}"
        print(f"  {name:<20} {rouge1:<20} {rouge2:<20}")
    
    print("="*70)

# ============================================================
# 4. ФУНКЦИЯ СОХРАНЕНИЯ РЕЗУЛЬТАТОВ
# ============================================================

def save_results(metrics, filepath='results/lstm_metrics.csv'):
    """
    Сохраняет результаты оценки в CSV.
    """
    import pandas as pd
    from pathlib import Path
    
    # Создаём папку
    Path(filepath).parent.mkdir(exist_ok=True)
    
    # Сохраняем метрики
    results_df = pd.DataFrame({
        'metric': ['rouge1_mean', 'rouge1_std', 'rouge2_mean', 'rouge2_std', 'n_samples'],
        'value': [
            metrics['rouge1_mean'],
            metrics['rouge1_std'],
            metrics['rouge2_mean'],
            metrics['rouge2_std'],
            metrics['n_samples']
        ]
    })
    results_df.to_csv(filepath, index=False, encoding='utf-8')
    
    # Сохраняем примеры
    if metrics['examples']:
        examples_df = pd.DataFrame(metrics['examples'])
        examples_path = filepath.replace('.csv', '_examples.csv')
        examples_df.to_csv(examples_path, index=False, encoding='utf-8')
    
    print(f"✅ Результаты сохранены в {filepath}")
