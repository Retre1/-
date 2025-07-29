# Руководство по обучению моделей для торговли

## 🎯 Где и как обучать модели

### 1. **Локальное обучение (Рекомендуется для начала)**

**Требования для эффективного обучения:**
```
💻 Минимальные требования:
- CPU: 8 ядер, 3.0 GHz
- RAM: 16 GB
- GPU: NVIDIA GTX 1660+ (6GB VRAM)
- Storage: 100 GB SSD

🚀 Рекомендуемые требования:
- CPU: 16 ядер, 3.5 GHz  
- RAM: 32 GB
- GPU: NVIDIA RTX 3080+ (10GB VRAM)
- Storage: 500 GB NVMe SSD
```

**Настройка локальной среды:**
```bash
# 1. Установка CUDA (для GPU)
# Скачайте с nvidia.com и установите CUDA 11.8+

# 2. Проверка GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# 3. Установка дополнительных зависимостей
pip install optuna tensorflow-gpu matplotlib seaborn

# 4. Настройка переменных среды
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

### 2. **Google Colab Pro (Лучший баланс цена/качество)**

**Преимущества:**
- NVIDIA A100 или T4 GPU
- 25+ GB RAM  
- Предустановленные библиотеки
- $10/месяц

**Настройка Colab:**
```python
# В первой ячейке Colab
!pip install MetaTrader5 optuna yfinance

# Подключение Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Загрузка кода
!git clone https://github.com/your-repo/forex-bot-solana.git
%cd forex-bot-solana

# Проверка GPU
import tensorflow as tf
print("GPU доступен:", tf.test.is_gpu_available())
print("Устройства:", tf.config.list_physical_devices())
```

### 3. **Облачные платформы**

#### Google Cloud Platform (Рекомендуется)
```bash
# Создание instance с GPU
gcloud compute instances create forex-trainer \
    --zone=us-central1-a \
    --machine-type=n1-highmem-8 \
    --accelerator=type=nvidia-tesla-v100,count=1 \
    --image-family=tf2-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --preemptible  # Экономия 80% стоимости

# Подключение по SSH
gcloud compute ssh forex-trainer --zone=us-central1-a
```

**Стоимость обучения:**
- **V100 GPU**: ~$2.50/час
- **T4 GPU**: ~$0.35/час  
- **A100 GPU**: ~$4.00/час

#### Amazon EC2
```bash
# Рекомендуемые instance types:
# p3.2xlarge  - V100, $3.06/час
# g4dn.xlarge - T4, $0.526/час  
# p4d.24xlarge - A100, $32.77/час (для больших моделей)
```

### 4. **Paperspace Gradient**

**Преимущества:**
- Простая настройка
- Jupyter notebooks
- Доступные цены

```python
# Paperspace setup
!pip install gradient
!gradient notebooks create --machineType P5000 --container tensorflow/tensorflow:latest-gpu
```

## 🚀 Процесс обучения моделей

### Шаг 1: Подготовка данных

```python
# Запуск скрипта обучения
cd training_bot
python train_models.py

# Или с кастомной конфигурацией
python train_models.py --config custom_training_config.json
```

**Структура данных:**
```python
# Пример получения исторических данных
import MetaTrader5 as mt5
import pandas as pd

def get_forex_data(symbol, timeframe, count=10000):
    """Получение исторических данных из MT5"""
    if not mt5.initialize():
        print("MT5 не инициализирован")
        return None
    
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    return df

# Альтернативно - использование yfinance
import yfinance as yf

def get_yahoo_data(symbol, period="2y"):
    """Получение данных из Yahoo Finance"""
    # Конвертация символов: EURUSD -> EURUSD=X
    yahoo_symbol = f"{symbol}=X" if len(symbol) == 6 else symbol
    data = yf.download(yahoo_symbol, period=period, interval="1h")
    return data
```

### Шаг 2: Оптимизация гиперпараметров

**Конфигурация для разных целей:**

**Быстрое обучение (тестирование):**
```json
{
  "models": {
    "xgboost": {"trials": 20},
    "lightgbm": {"trials": 20},
    "lstm": {"trials": 10, "epochs": 30}
  }
}
```

**Профессиональное обучение:**
```json
{
  "models": {
    "xgboost": {"trials": 200},
    "lightgbm": {"trials": 200},
    "lstm": {"trials": 100, "epochs": 200}
  }
}
```

**Экстремальная оптимизация:**
```json
{
  "models": {
    "xgboost": {"trials": 500},
    "lightgbm": {"trials": 500},
    "lstm": {"trials": 200, "epochs": 500}
  }
}
```

### Шаг 3: Мониторинг обучения

```python
# Отслеживание прогресса с помощью TensorBoard
import tensorboard

# Запуск TensorBoard
%load_ext tensorboard
%tensorboard --logdir logs/fit

# Или через командную строку
tensorboard --logdir=logs/fit --port=6006
```

**Мониторинг Optuna:**
```python
import optuna

# Визуализация оптимизации
study = optuna.load_study(study_name="forex_optimization", 
                         storage="sqlite:///optuna_study.db")

# Графики оптимизации
optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_param_importances(study).show()
optuna.visualization.plot_parallel_coordinate(study).show()
```

## 📊 Лучшие практики обучения

### 1. **Подготовка данных**

```python
def prepare_professional_features(df):
    """Создание профессиональных признаков"""
    
    # Технические индикаторы
    df['rsi'] = ta.RSI(df['close'], timeperiod=14)
    df['macd'], df['signal'], df['hist'] = ta.MACD(df['close'])
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = ta.BBANDS(df['close'])
    
    # Продвинутые индикаторы
    df['adx'] = ta.ADX(df['high'], df['low'], df['close'])
    df['cci'] = ta.CCI(df['high'], df['low'], df['close'])
    df['stoch_k'], df['stoch_d'] = ta.STOCH(df['high'], df['low'], df['close'])
    
    # Волатильность
    df['atr'] = ta.ATR(df['high'], df['low'], df['close'])
    df['volatility'] = df['close'].rolling(20).std()
    
    # Ценовые паттерны
    df['doji'] = ta.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
    df['hammer'] = ta.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
    
    # Объемные индикаторы (если доступны)
    if 'volume' in df.columns:
        df['obv'] = ta.OBV(df['close'], df['volume'])
        df['ad'] = ta.AD(df['high'], df['low'], df['close'], df['volume'])
    
    # Фракталы и уровни
    df['support'] = df['low'].rolling(window=20).min()
    df['resistance'] = df['high'].rolling(window=20).max()
    
    return df
```

### 2. **Оптимизация производительности**

```python
# Настройка TensorFlow для максимальной производительности
import tensorflow as tf

def optimize_tensorflow():
    """Оптимизация TensorFlow"""
    # Настройка GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Mixed precision для ускорения
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            
        except RuntimeError as e:
            print(f"GPU настройка не удалась: {e}")
    
    # Оптимизация CPU
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
```

### 3. **Валидация модели**

```python
def validate_model_robustness(model, X_test, y_test):
    """Проверка устойчивости модели"""
    
    # Walk-forward validation
    results = []
    window_size = len(X_test) // 10
    
    for i in range(10):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        
        X_window = X_test[start_idx:end_idx]
        y_window = y_test[start_idx:end_idx]
        
        predictions = model.predict(X_window)
        mse = mean_squared_error(y_window, predictions)
        
        results.append({
            'window': i,
            'mse': mse,
            'predictions': predictions,
            'actual': y_window
        })
    
    return results

# Проверка на разных временных периодах
def test_temporal_stability(model, data_by_year):
    """Тестирование на разных годах"""
    yearly_performance = {}
    
    for year, data in data_by_year.items():
        X = data.drop(['target'], axis=1)
        y = data['target']
        
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)
        
        yearly_performance[year] = {
            'mse': mse,
            'correlation': np.corrcoef(y, predictions)[0,1],
            'directional_accuracy': np.mean(np.sign(y) == np.sign(predictions))
        }
    
    return yearly_performance
```

## ⚡ Ускорение обучения

### 1. **Распределенное обучение**

```python
# Для TensorFlow с несколькими GPU
strategy = tf.distribute.MirroredStrategy()
print(f'Количество устройств: {strategy.num_replicas_in_sync}')

with strategy.scope():
    model = create_lstm_model()
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
```

### 2. **Оптимизация данных**

```python
# Использование tf.data для ускорения
def create_optimized_dataset(X, y, batch_size=32):
    """Создание оптимизированного датасета"""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.cache()  # Кэширование в память
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Предзагрузка
    return dataset
```

### 3. **Ранняя остановка и checkpoints**

```python
# Умные callbacks для экономии времени
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
]
```

## 💰 Стоимость обучения

### Примерные затраты:

**Локальное обучение:**
- Электричество: $0.1-0.2/час
- Амортизация оборудования: $0.5-1.0/час

**Google Colab Pro:**
- $10/месяц (безлимит)
- Лучший вариант для начинающих

**GCP/AWS:**
- T4 GPU: $0.35/час
- V100 GPU: $2.50/час  
- A100 GPU: $4.00/час

**Время обучения:**
- XGBoost: 10-30 минут
- LightGBM: 5-20 минут
- LSTM: 1-4 часа
- Полный ансамбль: 2-6 часов

### Оптимизация затрат:

```python
# Preemptible instances (80% скидка)
# Автоматическое сохранение прогресса
class TrainingManager:
    def __init__(self):
        self.checkpoint_interval = 100  # trials
        
    def save_progress(self, study, trial_number):
        """Сохранение прогресса оптимизации"""
        if trial_number % self.checkpoint_interval == 0:
            # Сохранение промежуточных результатов
            joblib.dump(study, f'study_checkpoint_{trial_number}.pkl')
            
    def resume_optimization(self, checkpoint_file):
        """Продолжение с последнего checkpoint"""
        return joblib.load(checkpoint_file)
```

## 🎯 Рекомендации по конфигурации

### Для начинающих:
```json
{
  "models": {
    "xgboost": {"enabled": true, "optimize": false},
    "lightgbm": {"enabled": false},
    "lstm": {"enabled": false}
  },
  "data": {
    "symbols": ["EURUSD"],
    "test_size": 0.3
  }
}
```

### Для продвинутых:
```json
{
  "models": {
    "xgboost": {"enabled": true, "optimize": true, "trials": 100},
    "lightgbm": {"enabled": true, "optimize": true, "trials": 100},
    "lstm": {"enabled": true, "optimize": true, "trials": 50},
    "ensemble": {"enabled": true}
  }
}
```

### Для профессионалов:
```json
{
  "models": {
    "xgboost": {"enabled": true, "optimize": true, "trials": 500},
    "lightgbm": {"enabled": true, "optimize": true, "trials": 500},
    "lstm": {"enabled": true, "optimize": true, "trials": 200},
    "transformer": {"enabled": true, "optimize": true, "trials": 100},
    "ensemble": {"enabled": true, "optimize_weights": true}
  }
}
```

## 🛡️ Безопасность и backup

```python
# Автоматическое сохранение в облако
import boto3

def backup_to_s3(local_path, bucket_name, s3_path):
    """Сохранение моделей в S3"""
    s3 = boto3.client('s3')
    s3.upload_file(local_path, bucket_name, s3_path)

# Версионирование моделей
def save_model_version(model, version, metadata):
    """Сохранение версии модели с метаданными"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_path = f"models/v{version}_{timestamp}"
    os.makedirs(model_path, exist_ok=True)
    
    # Сохранение модели
    model.save(f"{model_path}/model.h5")
    
    # Сохранение метаданных
    with open(f"{model_path}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
```

## 📈 Мониторинг результатов

```python
# Отправка уведомлений о завершении обучения
def send_training_notification(results):
    """Отправка уведомления в Telegram"""
    import requests
    
    bot_token = "YOUR_BOT_TOKEN"
    chat_id = "YOUR_CHAT_ID"
    
    message = f"""
🤖 Обучение моделей завершено!

📊 Результаты:
XGBoost MSE: {results['xgboost']['mse']:.6f}
LightGBM MSE: {results['lightgbm']['mse']:.6f}
LSTM MSE: {results['lstm']['mse']:.6f}

🏆 Лучшая модель: {min(results, key=lambda x: results[x]['mse'])}
    """
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": message})
```

---

**Заключение:** Начните с локального обучения простых моделей, затем переходите к облачным платформам для более сложных ансамблей. Google Colab Pro обеспечивает лучший баланс цена/качество для большинства задач.