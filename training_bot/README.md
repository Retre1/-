# 🤖 Система обучения моделей ForexBot

Эта папка содержит профессиональную систему для обучения AI моделей для торговли на форекс.

## 📂 Структура папки

```
training_bot/
├── README.md                 # Этот файл
├── train_models.py          # Основной скрипт обучения
├── data_collector.py        # Сбор данных из различных источников  
├── training_config.json     # Конфигурация обучения
└── trained_models/          # Папка для сохранения обученных моделей
    ├── EURUSD/
    ├── GBPUSD/
    └── ...
```

## 🚀 Быстрый старт

### 1. Сбор данных

```bash
# Автоматический сбор данных для основных валютных пар
python data_collector.py

# Данные будут сохранены в data/collected/
```

### 2. Обучение моделей

```bash
# Стандартное обучение
python train_models.py

# С кастомной конфигурацией
python train_models.py --config my_training_config.json

# Только XGBoost (быстрое обучение)
python train_models.py --models xgboost
```

### 3. Мониторинг обучения

```bash
# TensorBoard для LSTM моделей
tensorboard --logdir=logs/fit --port=6006

# Optuna dashboard для гиперпараметров
optuna-dashboard sqlite:///optuna_study.db --port=8080
```

## ⚙️ Конфигурация

Основные параметры настраиваются в `training_config.json`:

```json
{
  "data": {
    "symbols": ["EURUSD", "GBPUSD", "USDJPY"],
    "start_date": "2020-01-01",
    "end_date": "2024-01-01"
  },
  "models": {
    "xgboost": {"enabled": true, "trials": 100},
    "lightgbm": {"enabled": true, "trials": 100},
    "lstm": {"enabled": true, "trials": 50}
  }
}
```

### Уровни обучения:

**🏃 Быстрое тестирование (10-30 минут):**
```json
{"models": {"xgboost": {"trials": 20}, "lstm": {"trials": 5}}}
```

**⚡ Стандартное обучение (1-3 часа):**
```json
{"models": {"xgboost": {"trials": 100}, "lstm": {"trials": 50}}}
```

**🚀 Профессиональное обучение (4-8 часов):**
```json
{"models": {"xgboost": {"trials": 300}, "lstm": {"trials": 100}}}
```

**🏆 Экстремальная оптимизация (12-24 часа):**
```json
{"models": {"xgboost": {"trials": 500}, "lstm": {"trials": 200}}}
```

## 🎯 Рекомендуемые платформы

### Локальное обучение
- **GPU**: NVIDIA RTX 3080+ (10GB VRAM)
- **RAM**: 32GB
- **CPU**: 16 ядер 3.5+ GHz

### Google Colab Pro ($10/месяц)
```python
# Colab setup
!git clone https://github.com/your-repo/forex-bot-solana.git
%cd forex-bot-solana/training_bot
!pip install -r ../requirements.txt
!python train_models.py
```

### Google Cloud Platform
```bash
# Создание GPU instance
gcloud compute instances create forex-trainer \
    --machine-type=n1-highmem-8 \
    --accelerator=type=nvidia-tesla-v100,count=1 \
    --image-family=tf2-latest-gpu \
    --preemptible  # 80% скидка
```

## 📊 Источники данных

### 1. MetaTrader 5 (Рекомендуется)
- Реальные тики и спреды
- Минимальная задержка  
- Полная история

```python
from data_collector import ForexDataCollector
collector = ForexDataCollector(data_source="mt5")
```

### 2. Yahoo Finance (Бесплатно)
- Доступно без регистрации
- Ограниченная точность
- Подходит для начала

```python
collector = ForexDataCollector(data_source="yahoo")
```

### 3. Синтетические данные (Тестирование)
- Для разработки алгоритмов
- Полный контроль параметров

```python
data = collector.create_synthetic_data("EURUSD")
```

## 🔧 Технические индикаторы

Система автоматически создает 150+ признаков:

### Базовые индикаторы:
- SMA, EMA (5, 10, 20, 50, 100, 200)
- RSI, MACD, Bollinger Bands
- Stochastic, Williams %R, CCI, ADX

### Продвинутые признаки:
- Лаговые признаки (1-20 периодов)
- Скользящие статистики (мин, макс, std)
- Волатильность (5-50 периодов)
- Позиция в диапазоне
- Z-scores и нормализация

### Паттерны свечей (TA-Lib):
- Doji, Hammer, Shooting Star
- Hanging Man, Engulfing
- 50+ дополнительных паттернов

## 🤖 AI Модели

### 1. XGBoost
```
✅ Лучшая точность (55-60%)
✅ Быстрое обучение (10-30 мин)
✅ Интерпретируемость
⚠️ Требует качественных признаков
```

### 2. LightGBM  
```
✅ Очень быстрое обучение (5-20 мин)
✅ Низкое потребление памяти
✅ Хорошая точность (50-58%)
⚠️ Может переобучаться
```

### 3. LSTM Neural Networks
```
✅ Отлично для временных рядов
✅ Улавливает сложные паттерны (55-62%)
⚠️ Медленное обучение (1-4 часа)
⚠️ Требует GPU
```

### 4. Ensemble (Комбинированный)
```
🏆 Максимальная точность (60-65%)
✅ Устойчивость к переобучению
✅ Автоматическая оптимизация весов
⚠️ Сложность и время обучения
```

## 📈 Результаты обучения

### Метрики качества:
- **MSE**: Среднеквадратичная ошибка
- **MAE**: Средняя абсолютная ошибка  
- **Directional Accuracy**: Точность направления (%)
- **Sharpe Ratio**: Соотношение доходность/риск

### Ожидаемые результаты:
```
Новичок (базовые модели):     45-50% точности
Продвинутый (оптимизация):    50-58% точности  
Эксперт (ансамбли):          58-65% точности
Профи (кастомные модели):     60-70% точности
```

## 🛡️ Валидация и тестирование

### Walk-Forward Analysis
- Обучение на исторических данных
- Тестирование на будущих периодах
- Проверка стабильности во времени

### Cross-Validation
- 5-fold временная валидация
- Предотвращение переобучения
- Оценка устойчивости

### Out-of-Sample Testing
- 20% данных для финального теста
- Никогда не используются при обучении
- Реальная оценка производительности

## 📂 Структура выходных файлов

```
trained_models/EURUSD/
├── xgboost_model.pkl        # XGBoost модель
├── lightgbm_model.pkl       # LightGBM модель  
├── lstm_model.h5            # LSTM модель
├── scalers.pkl              # Нормализаторы данных
├── feature_importance.png   # Важность признаков
├── validation_results.json  # Результаты валидации
└── training_log.txt         # Лог обучения
```

## 🔍 Анализ результатов

### Автоматически создаются графики:
1. **Predictions vs Actual** - Сравнение прогнозов и реальности
2. **Feature Importance** - Важность признаков
3. **Residuals Analysis** - Анализ остатков
4. **Model Comparison** - Сравнение моделей
5. **Optimization History** - История оптимизации

### Статистический анализ:
- Корреляция прогнозов с реальностью
- Стабильность на разных периодах
- Анализ ошибок по времени

## ⚡ Оптимизация производительности

### GPU Ускорение:
```python
# Автоматическая настройка GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

### Параллельное обучение:
```python
# Используйте все ядра CPU
n_jobs = -1  # Все доступные ядра
```

### Кэширование данных:
```python
# Данные автоматически кэшируются
# Повторные запуски будут быстрее
```

## 🚨 Часто встречающиеся проблемы

### 1. Нехватка памяти GPU
```python
# Решение: уменьшите batch_size
"training": {"batch_size": 16}  # вместо 32
```

### 2. Медленное обучение LSTM
```python
# Решение: уменьшите количество trials
"lstm": {"trials": 10}  # вместо 50
```

### 3. Переобучение
```python
# Решение: включите early stopping
"training": {"early_stopping": true}
```

### 4. Отсутствие TA-Lib
```bash
# Ubuntu/Debian
sudo apt-get install ta-lib

# macOS
brew install ta-lib

# Windows - скачайте с https://github.com/mrjbq7/ta-lib
```

## 📞 Поддержка

- 📧 Email: support@forexbot.ai
- 💬 Telegram: @ForexBotSupport  
- 📚 Документация: [docs/TRAINING_GUIDE.md](../docs/TRAINING_GUIDE.md)
- 🐛 Issues: GitHub Issues

## 🔗 Связанные файлы

- [TRAINING_GUIDE.md](../docs/TRAINING_GUIDE.md) - Подробное руководство
- [AI_TRADING_GUIDE.md](../docs/AI_TRADING_GUIDE.md) - Рекомендации по AI
- [requirements.txt](../requirements.txt) - Зависимости
- [config.json.example](../config.json.example) - Основная конфигурация

---

**🎯 Цель**: Создание профессиональных AI моделей для стабильной прибыли в торговле на форекс с интеграцией Solana токена.

**⚠️ Важно**: Торговля на форекс связана с высоким риском. Всегда тестируйте модели на демо-счете перед реальной торговлей.