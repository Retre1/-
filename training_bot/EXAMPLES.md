# 🎯 Практические примеры поэтапного обучения

## 🚀 Команды для быстрого старта

### 1. Демонстрация (5 минут)
```bash
# Интерактивная демонстрация
python demo.py

# Что происходит:
# ✅ Создаются синтетические данные
# ✅ Обучается XGBoost (30 сек)
# ✅ Обучается LSTM (2 мин)  
# ✅ Создается Ensemble (10 сек)
# ✅ Сравниваются результаты
# ✅ Строятся графики
```

### 2. Быстрое тестирование (15 минут)
```bash
# Поэтапное обучение для EURUSD
python run_progressive.py --quick

# Результат:
# 🎯 XGBoost: ~52% точность за 1 мин
# 🧠 LSTM: ~55% точность за 10 мин  
# 🏆 Ensemble: ~58% точность за 1 мин
```

### 3. Профессиональное обучение (3 часа)
```bash
# Максимальная оптимизация
python run_progressive.py --professional --symbol EURUSD

# Результат:
# 🎯 XGBoost: ~58% точность за 30 мин
# 🧠 LSTM: ~62% точность за 2 часа
# 🏆 Ensemble: ~65% точность за 5 мин
```

## 💡 Сценарии использования

### Сценарий 1: "Мне нужна быстрая модель прямо сейчас"
```bash
# Только XGBoost - готово за 30 секунд
python run_progressive.py --xgboost-only --quick

# Плюсы: Очень быстро, стабильно
# Минусы: Ниже точность (~50-55%)
# Когда использовать: Прототипирование, тестирование
```

### Сценарий 2: "Хочу максимальную точность"
```bash
# Полное обучение с максимальными параметрами
python run_progressive.py --professional

# Плюсы: Максимальная точность (~60-65%)
# Минусы: Долго (3-8 часов)
# Когда использовать: Продакшн, реальная торговля
```

### Сценарий 3: "Сравнить разные валютные пары"
```bash
# Обучить модели для нескольких пар
python run_progressive.py --symbol EURUSD --standard
python run_progressive.py --symbol GBPUSD --standard  
python run_progressive.py --symbol USDJPY --standard

# Результат: Модели для разных пар в отдельных папках
```

### Сценарий 4: "У меня есть свои данные"
```bash
# Использование собственных данных
python run_progressive.py --data path/to/my_data.csv --standard

# Формат CSV:
# timestamp,open,high,low,close,volume
# 2024-01-01 00:00:00,1.1000,1.1010,1.0990,1.1005,12345
```

## 📊 Ожидаемые результаты

### Быстрый режим (--quick)
```
🎯 XGBoost:
   • Время: 30 сек - 2 мин
   • Точность: 48-55%
   • MSE: 0.000010-0.000020

🧠 LSTM:
   • Время: 2-10 мин
   • Точность: 50-58%
   • MSE: 0.000008-0.000018

🏆 Ensemble:
   • Время: 10-30 сек
   • Точность: 52-60%
   • MSE: 0.000007-0.000015
```

### Стандартный режим (--standard)
```
🎯 XGBoost:
   • Время: 5-15 мин
   • Точность: 52-58%
   • MSE: 0.000008-0.000015

🧠 LSTM:
   • Время: 30 мин - 2 часа
   • Точность: 55-62%
   • MSE: 0.000006-0.000012

🏆 Ensemble:
   • Время: 1-3 мин
   • Точность: 58-65%
   • MSE: 0.000005-0.000010
```

### Профессиональный режим (--professional)
```
🎯 XGBoost:
   • Время: 15-45 мин
   • Точность: 55-62%
   • MSE: 0.000005-0.000012

🧠 LSTM:
   • Время: 1-4 часа
   • Точность: 58-68%
   • MSE: 0.000004-0.000009

🏆 Ensemble:
   • Время: 3-10 мин
   • Точность: 60-70%
   • MSE: 0.000003-0.000008
```

## 🔧 Расширенные примеры

### Пример 1: Обучение с настройкой логов
```bash
# Создать папку для логов
mkdir -p logs

# Запуск с подробными логами
python run_progressive.py --standard --symbol EURUSD 2>&1 | tee logs/training.log

# Анализ логов
grep "MSE\|Accuracy" logs/training.log
```

### Пример 2: Batch обучение для нескольких пар
```bash
#!/bin/bash
# train_multiple.sh

symbols=("EURUSD" "GBPUSD" "USDJPY" "AUDUSD" "USDCHF")

for symbol in "${symbols[@]}"; do
    echo "🚀 Обучение $symbol..."
    python run_progressive.py --symbol $symbol --standard
    echo "✅ $symbol завершено"
done

echo "🎉 Все модели готовы!"
```

### Пример 3: Сравнение режимов обучения
```bash
# Быстрый режим
python run_progressive.py --quick --symbol EURUSD_QUICK

# Стандартный режим  
python run_progressive.py --standard --symbol EURUSD_STANDARD

# Профессиональный режим
python run_progressive.py --professional --symbol EURUSD_PRO

# Сравнить результаты в папках:
# progressive_models/EURUSD_QUICK/
# progressive_models/EURUSD_STANDARD/  
# progressive_models/EURUSD_PRO/
```

## 📈 Анализ результатов

### Структура выходных файлов
```
progressive_models/EURUSD/
├── xgboost_model.pkl         # XGBoost модель
├── xgboost_results.json      # Метрики XGBoost
├── lstm_model.h5             # LSTM модель  
├── lstm_results.json         # Метрики LSTM
├── ensemble_results.json     # Метрики Ensemble
├── comparison_results.png    # Графики сравнения
└── scalers.pkl              # Нормализаторы данных
```

### Анализ JSON результатов
```python
import json

# Загрузка результатов
with open('progressive_models/EURUSD/ensemble_results.json') as f:
    results = json.load(f)

print(f"MSE: {results['test_mse']:.6f}")
print(f"Точность: {results['directional_accuracy']:.1f}%")
print(f"Веса: XGBoost={results['weights'][0]:.2f}, LSTM={results['weights'][1]:.2f}")
```

### Использование обученных моделей
```python
import pickle
from tensorflow.keras.models import load_model

# Загрузка моделей
with open('progressive_models/EURUSD/xgboost_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

lstm_model = load_model('progressive_models/EURUSD/lstm_model.h5')

# Загрузка скейлеров
with open('progressive_models/EURUSD/scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)

# Предсказание
# X_new = ... ваши новые данные
X_scaled = scalers['xgboost'].transform(X_new)
xgb_pred = xgb_model.predict(X_scaled)

# Ensemble предсказание
with open('progressive_models/EURUSD/ensemble_results.json') as f:
    ensemble_weights = json.load(f)['weights']

ensemble_pred = (ensemble_weights[0] * xgb_pred + 
                ensemble_weights[1] * lstm_pred)
```

## 🚨 Частые проблемы и решения

### Проблема: "GPU не найден"
```bash
# Решение 1: Установка TensorFlow-GPU
pip install tensorflow-gpu

# Решение 2: Использование CPU (медленнее)
export CUDA_VISIBLE_DEVICES=""
python run_progressive.py --quick
```

### Проблема: "Нехватка памяти"
```bash
# Решение: Уменьшение batch size
# Отредактируйте progressive_trainer.py:
# batch_size=16  # вместо 32
```

### Проблема: "Очень медленно"
```bash
# Решение: Уменьшение trials
python run_progressive.py --quick  # Вместо --standard

# Или только XGBoost
python run_progressive.py --xgboost-only --quick
```

### Проблема: "Низкая точность"
```bash
# Решение 1: Больше данных
python data_collector.py  # Собрать больше данных

# Решение 2: Больше оптимизации
python run_progressive.py --professional

# Решение 3: Другая валютная пара
python run_progressive.py --symbol GBPUSD --standard
```

## 💻 Системные требования

### Минимальные (для --quick)
```
CPU: 4 ядра, 2.5 GHz
RAM: 8 GB
GPU: Не требуется
Время: 10-30 минут
```

### Рекомендуемые (для --standard)
```
CPU: 8 ядер, 3.0 GHz  
RAM: 16 GB
GPU: NVIDIA GTX 1660+ (6GB VRAM)
Время: 1-3 часа
```

### Оптимальные (для --professional)
```
CPU: 16 ядер, 3.5 GHz
RAM: 32 GB  
GPU: NVIDIA RTX 3080+ (10GB VRAM)
Время: 3-8 часов
```

## 🎯 Следующие шаги

После успешного обучения:

1. **Интеграция с торговым ботом**
   ```bash
   # Скопировать лучшую модель в основную папку
   cp progressive_models/EURUSD/ensemble_* ../trading_bot/models/
   ```

2. **Backtesting на исторических данных**
   ```bash
   # Тестирование на данных, не участвовавших в обучении
   python ../backtesting/test_model.py --model progressive_models/EURUSD/
   ```

3. **Live тестирование**
   ```bash
   # Запуск на демо-счете
   python ../trading_bot/main.py --demo --model progressive_models/EURUSD/
   ```

4. **Мониторинг производительности**
   ```bash
   # Веб-интерфейс для мониторинга
   python ../web_interface/backend/main.py
   ```

---

🎉 **Поздравляем!** Теперь у вас есть профессиональная система поэтапного обучения AI моделей для торговли на форекс!