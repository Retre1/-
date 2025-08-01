# 🎓 **Подробное руководство по обучению ForexBot AI**

## 📋 **Содержание**

1. [Подготовка данных из MetaTrader5](#подготовка-данных-из-metatrader5)
2. [Настройка обучения](#настройка-обучения)
3. [Процесс обучения](#процесс-обучения)
4. [Оценка качества моделей](#оценка-качества-моделей)
5. [Сохранение и загрузка моделей](#сохранение-и-загрузка-моделей)
6. [Оптимизация гиперпараметров](#оптимизация-гиперпараметров)
7. [Мониторинг обучения](#мониторинг-обучения)
8. [Устранение проблем](#устранение-проблем)

---

## 📊 **1. Подготовка данных из MetaTrader5**

### **🔧 Настройка подключения к MT5:**

```json
{
  "mt5": {
    "server": "YourBroker-Server",
    "login": 12345,
    "password": "your_password",
    "symbols": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "NZDUSD", "USDCAD"],
    "timeframes": ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
  }
}
```

### **📊 Сбор исторических данных:**

```bash
# Запуск сбора данных
python data_collection/mt5_data_collector.py
```

**Рекомендуемые параметры сбора данных:**

| Параметр | Значение | Описание |
|----------|----------|----------|
| **Период** | 1-3 года | Достаточно данных для обучения |
| **Таймфреймы** | M1, M5, M15, M30, H1, H4, D1 | Различные временные масштабы |
| **Символы** | Major pairs | Основные валютные пары |
| **Минимум записей** | 10,000 | Для каждого символа/таймфрейма |

### **📈 Требования к качеству данных:**

```python
# Проверка качества данных
def validate_data_quality(df):
    # Проверка на пропуски
    missing_values = df.isnull().sum()
    
    # Проверка на дубликаты
    duplicates = df.index.duplicated().sum()
    
    # Проверка на аномалии
    negative_prices = (df[['open', 'high', 'low', 'close']] < 0).any(axis=1).sum()
    
    # Проверка объема данных
    min_records = 10000
    
    return {
        'total_records': len(df),
        'missing_values': missing_values.to_dict(),
        'duplicates': duplicates,
        'negative_prices': negative_prices,
        'is_sufficient': len(df) >= min_records
    }
```

---

## ⚙️ **2. Настройка обучения**

### **📋 Конфигурация обучения:**

```python
training_config = {
    'symbol': 'EURUSD',
    'timeframe': 'H1',
    'lookback_periods': [20, 50, 100, 200],
    'test_size': 0.2,
    'validation_size': 0.1,
    'sequence_length': 50,
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 10,
    'min_samples': 10000,
    'feature_importance_threshold': 0.01,
    'class_balance_method': 'smote'  # 'smote', 'undersample', 'oversample'
}
```

### **🎯 Выбор целевой переменной:**

```python
def create_target_variable(prices, threshold_buy=0.001, threshold_sell=-0.001):
    """
    Создание целевой переменной для классификации
    
    Args:
        prices: Series с ценами
        threshold_buy: Порог для сигнала покупки (0.1%)
        threshold_sell: Порог для сигнала продажи (-0.1%)
    
    Returns:
        target: Массив классов [0=HOLD, 1=BUY, 2=SELL]
    """
    # Расчет будущих доходностей
    future_returns = prices.shift(-1) / prices - 1
    
    # Создание классов
    target = np.zeros(len(prices))
    target[future_returns > threshold_buy] = 1    # BUY
    target[future_returns < threshold_sell] = 2   # SELL
    # 0 = HOLD (по умолчанию)
    
    return target[:-1]  # Удаляем последний элемент
```

### **🔧 Подготовка признаков:**

```python
# Создание технических индикаторов
def create_features(df):
    # Базовые индикаторы
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # Stochastic
    low_min = df['low'].rolling(window=14).min()
    high_max = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr'] = true_range.rolling(window=14).mean()
    
    return df
```

---

## 🎓 **3. Процесс обучения**

### **🔄 Полный цикл обучения:**

```python
def train_model(symbol, timeframe):
    """
    Полный цикл обучения модели
    
    Args:
        symbol: Валютная пара (например, 'EURUSD')
        timeframe: Таймфрейм (например, 'H1')
    
    Returns:
        bool: Успешность обучения
    """
    
    # 1. Загрузка данных
    df = load_market_data(symbol, timeframe)
    if df is None:
        return False
    
    # 2. Подготовка данных
    features, target, feature_columns = prepare_training_data(df)
    if features is None:
        return False
    
    # 3. Балансировка классов
    features_balanced, target_balanced = balance_classes(features, target)
    
    # 4. Разделение данных
    X_train, X_val, X_test, y_test = split_data(features_balanced, target_balanced)
    
    # 5. Обучение моделей
    training_results = train_models(X_train, X_val, y_train, y_val)
    
    # 6. Оценка качества
    evaluation_results = evaluate_models(X_test, y_test)
    
    # 7. Сохранение моделей
    save_models(symbol, timeframe)
    
    # 8. Генерация отчета
    generate_training_report(symbol, timeframe)
    
    return True
```

### **📊 Разделение данных:**

```python
def split_data(features, target):
    """
    Временное разделение данных для финансовых временных рядов
    
    Args:
        features: Признаки
        target: Целевая переменная
    
    Returns:
        X_train, X_val, X_test, y_test
    """
    # Использование TimeSeriesSplit для временных рядов
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Получаем последний сплит
    for train_idx, test_idx in tscv.split(features):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = target[train_idx], target[test_idx]
    
    # Дополнительное разделение обучающей выборки
    split_point = int(len(X_train) * 0.9)  # 90% для обучения, 10% для валидации
    X_train_final = X_train[:split_point]
    y_train_final = y_train[:split_point]
    X_val = X_train[split_point:]
    y_val = y_train[split_point:]
    
    return X_train_final, X_val, X_test, y_test
```

### **⚖️ Балансировка классов:**

```python
def balance_classes(features, target, method='smote'):
    """
    Балансировка классов для решения проблемы несбалансированности
    
    Args:
        features: Признаки
        target: Целевая переменная
        method: Метод балансировки ('smote', 'undersample', 'oversample')
    
    Returns:
        features_balanced, target_balanced
    """
    if method == 'smote':
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        features_balanced, target_balanced = smote.fit_resample(features, target)
    
    elif method == 'undersample':
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(random_state=42)
        features_balanced, target_balanced = rus.fit_resample(features, target)
    
    elif method == 'oversample':
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=42)
        features_balanced, target_balanced = ros.fit_resample(features, target)
    
    else:
        features_balanced, target_balanced = features, target
    
    return features_balanced, target_balanced
```

---

## 📊 **4. Оценка качества моделей**

### **📈 Метрики качества:**

```python
def evaluate_model_performance(y_true, y_pred, model_name):
    """
    Оценка качества модели
    
    Args:
        y_true: Истинные значения
        y_pred: Предсказания
        model_name: Название модели
    
    Returns:
        dict: Метрики качества
    """
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    
    # Базовая точность
    accuracy = accuracy_score(y_true, y_pred)
    
    # Детальный отчет
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    # Матрица ошибок
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Дополнительные метрики
    precision = class_report['weighted avg']['precision']
    recall = class_report['weighted avg']['recall']
    f1_score = class_report['weighted avg']['f1-score']
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix.tolist()
    }
```

### **🎯 Интерпретация результатов:**

| Метрика | Отличное | Хорошее | Удовлетворительное | Плохое |
|---------|----------|---------|-------------------|--------|
| **Accuracy** | > 0.75 | 0.65-0.75 | 0.55-0.65 | < 0.55 |
| **Precision** | > 0.70 | 0.60-0.70 | 0.50-0.60 | < 0.50 |
| **Recall** | > 0.70 | 0.60-0.70 | 0.50-0.60 | < 0.50 |
| **F1-Score** | > 0.70 | 0.60-0.70 | 0.50-0.60 | < 0.50 |

### **📊 Визуализация результатов:**

```python
def create_evaluation_plots(evaluation_results, save_path):
    """
    Создание визуализаций для оценки моделей
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # График точности моделей
    model_names = list(evaluation_results.keys())
    accuracies = [evaluation_results[name]['accuracy'] for name in model_names]
    
    plt.figure(figsize=(15, 10))
    
    # График точности
    plt.subplot(2, 2, 1)
    plt.bar(model_names, accuracies)
    plt.title('Точность моделей')
    plt.ylabel('Точность')
    plt.xticks(rotation=45)
    
    # Матрицы ошибок
    for i, model_name in enumerate(model_names[:3]):
        plt.subplot(2, 2, 2 + i)
        conf_matrix = np.array(evaluation_results[model_name]['confusion_matrix'])
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Матрица ошибок: {model_name}')
        plt.ylabel('Истинные значения')
        plt.xlabel('Предсказанные значения')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
```

---

## 💾 **5. Сохранение и загрузка моделей**

### **💾 Сохранение моделей:**

```python
def save_trained_models(training_results, symbol, timeframe):
    """
    Сохранение обученных моделей
    
    Args:
        training_results: Результаты обучения
        symbol: Валютная пара
        timeframe: Таймфрейм
    """
    import pickle
    import json
    from pathlib import Path
    
    # Создание директории
    model_dir = Path(f"data/models/{symbol}/{timeframe}")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохранение каждой модели
    for model_name, result in training_results.items():
        model = result['model']
        
        try:
            if hasattr(model, 'save'):
                # Нейронные сети (TensorFlow/Keras)
                model_path = model_dir / f"{model_name}.h5"
                model.save(str(model_path))
            else:
                # Стандартные модели (sklearn, XGBoost, LightGBM)
                model_path = model_dir / f"{model_name}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            print(f"✅ Сохранена модель: {model_path}")
            
        except Exception as e:
            print(f"❌ Ошибка сохранения {model_name}: {e}")
    
    # Сохранение метаданных
    metadata = {
        'symbol': symbol,
        'timeframe': timeframe,
        'training_date': datetime.now().isoformat(),
        'model_performance': evaluation_results,
        'feature_columns': feature_columns
    }
    
    metadata_path = model_dir / 'training_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Сохранены метаданные: {metadata_path}")
```

### **📂 Загрузка моделей:**

```python
def load_trained_models(symbol, timeframe):
    """
    Загрузка обученных моделей
    
    Args:
        symbol: Валютная пара
        timeframe: Таймфрейм
    
    Returns:
        dict: Загруженные модели
    """
    import pickle
    from pathlib import Path
    from tensorflow.keras.models import load_model
    
    model_dir = Path(f"data/models/{symbol}/{timeframe}")
    models = {}
    
    # Загрузка каждой модели
    for model_file in model_dir.glob("*.h5"):
        model_name = model_file.stem
        try:
            model = load_model(str(model_file))
            models[model_name] = model
            print(f"✅ Загружена модель: {model_name}")
        except Exception as e:
            print(f"❌ Ошибка загрузки {model_name}: {e}")
    
    for model_file in model_dir.glob("*.pkl"):
        model_name = model_file.stem
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            models[model_name] = model
            print(f"✅ Загружена модель: {model_name}")
        except Exception as e:
            print(f"❌ Ошибка загрузки {model_name}: {e}")
    
    return models
```

---

## 🔧 **6. Оптимизация гиперпараметров**

### **🎯 Grid Search для оптимизации:**

```python
def optimize_hyperparameters(X_train, y_train, model_type='xgboost'):
    """
    Оптимизация гиперпараметров
    
    Args:
        X_train: Обучающие признаки
        y_train: Обучающая целевая переменная
        model_type: Тип модели
    
    Returns:
        dict: Лучшие параметры
    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer, accuracy_score
    
    if model_type == 'xgboost':
        from xgboost import XGBClassifier
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        model = XGBClassifier(random_state=42)
    
    elif model_type == 'lightgbm':
        import lightgbm as lgb
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        model = lgb.LGBMClassifier(random_state=42)
    
    # Grid Search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=TimeSeriesSplit(n_splits=3),
        scoring=make_scorer(accuracy_score),
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_model': grid_search.best_estimator_
    }
```

### **🔍 Bayesian Optimization:**

```python
def bayesian_optimization(X_train, y_train, model_type='xgboost'):
    """
    Байесовская оптимизация гиперпараметров
    """
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from sklearn.model_selection import cross_val_score
    
    def objective(params):
        if model_type == 'xgboost':
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=int(params[0]),
                max_depth=int(params[1]),
                learning_rate=params[2],
                subsample=params[3],
                colsample_bytree=params[4],
                random_state=42
            )
        
        # Кросс-валидация
        scores = cross_val_score(
            model, X_train, y_train, 
            cv=TimeSeriesSplit(n_splits=3), 
            scoring='accuracy'
        )
        
        return -scores.mean()  # Минимизируем отрицательную точность
    
    # Пространство параметров
    space = [
        Integer(50, 500, name='n_estimators'),
        Integer(3, 10, name='max_depth'),
        Real(0.01, 0.3, name='learning_rate'),
        Real(0.5, 1.0, name='subsample'),
        Real(0.5, 1.0, name='colsample_bytree')
    ]
    
    # Оптимизация
    result = gp_minimize(
        objective, space, n_calls=50, 
        random_state=42, n_jobs=-1
    )
    
    return {
        'best_params': dict(zip(['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree'], result.x)),
        'best_score': -result.fun
    }
```

---

## 📊 **7. Мониторинг обучения**

### **📈 Отслеживание метрик:**

```python
def monitor_training(history, model_name):
    """
    Мониторинг процесса обучения
    
    Args:
        history: История обучения (для нейронных сетей)
        model_name: Название модели
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    # График точности
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Accuracy - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # График потерь
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Loss - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'training_monitor_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
```

### **🔔 Ранняя остановка:**

```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Callbacks для нейронных сетей
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# Использование в обучении
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)
```

---

## 🔧 **8. Устранение проблем**

### **❌ Частые проблемы и решения:**

#### **1. Переобучение (Overfitting):**
```python
# Решения:
# - Увеличение регуляризации
# - Уменьшение сложности модели
# - Больше данных
# - Ранняя остановка

# Для XGBoost:
model = XGBClassifier(
    reg_alpha=0.1,  # L1 регуляризация
    reg_lambda=1.0,  # L2 регуляризация
    max_depth=5,     # Ограничение глубины
    subsample=0.8,   # Случайная выборка
    colsample_bytree=0.8  # Случайная выборка признаков
)
```

#### **2. Недообучение (Underfitting):**
```python
# Решения:
# - Увеличение сложности модели
# - Больше признаков
# - Уменьшение регуляризации
# - Больше эпох обучения

# Для нейронных сетей:
model = Sequential([
    Dense(128, activation='relu', input_shape=(n_features,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])
```

#### **3. Несбалансированные классы:**
```python
# Решения:
# - SMOTE для балансировки
# - Взвешенные классы
# - Изменение порогов классификации

# Взвешенные классы для XGBoost:
class_weights = {
    0: 1.0,  # HOLD
    1: 2.0,  # BUY (больший вес)
    2: 2.0   # SELL (больший вес)
}

model = XGBClassifier(
    scale_pos_weight=2.0,  # Увеличение веса положительных классов
    class_weight=class_weights
)
```

#### **4. Недостаточно данных:**
```python
# Решения:
# - Сбор большего количества данных
# - Аугментация данных
# - Использование предобученных моделей
# - Transfer learning

# Аугментация данных:
def augment_data(df):
    # Добавление шума к ценам
    noise_factor = 0.001
    df_augmented = df.copy()
    df_augmented['close'] += np.random.normal(0, noise_factor, len(df))
    return df_augmented
```

### **🔍 Диагностика проблем:**

```python
def diagnose_training_issues(X_train, y_train, X_val, y_val, model):
    """
    Диагностика проблем обучения
    """
    # Проверка размера данных
    print(f"Размер обучающей выборки: {len(X_train)}")
    print(f"Размер валидационной выборки: {len(X_val)}")
    
    # Проверка распределения классов
    print(f"Распределение классов (train): {np.bincount(y_train)}")
    print(f"Распределение классов (val): {np.bincount(y_val)}")
    
    # Проверка качества признаков
    print(f"Количество признаков: {X_train.shape[1]}")
    print(f"Пропущенные значения: {np.isnan(X_train).sum()}")
    
    # Проверка переобучения
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    
    print(f"Точность на обучающей выборке: {train_score:.4f}")
    print(f"Точность на валидационной выборке: {val_score:.4f}")
    
    if train_score - val_score > 0.1:
        print("⚠️ Возможно переобучение!")
    
    if val_score < 0.5:
        print("⚠️ Возможно недообучение!")
```

---

## 🚀 **Быстрый старт обучения**

### **📋 Пошаговый план:**

1. **Подготовка данных:**
   ```bash
   python data_collection/mt5_data_collector.py
   ```

2. **Настройка конфигурации:**
   ```bash
   # Редактирование config.json
   nano config.json
   ```

3. **Запуск обучения:**
   ```bash
   python training/advanced_training_system.py
   ```

4. **Проверка результатов:**
   ```bash
   # Просмотр отчетов
   ls data/reports/
   
   # Просмотр моделей
   ls data/models/
   ```

### **✅ Чек-лист успешного обучения:**

- [ ] Данные собраны и проверены
- [ ] Конфигурация настроена
- [ ] Модели обучены без ошибок
- [ ] Точность > 60%
- [ ] Нет переобучения (разница train/val < 10%)
- [ ] Модели сохранены
- [ ] Отчеты сгенерированы
- [ ] Визуализации созданы

**🎉 Поздравляем! Ваши модели успешно обучены и готовы к торговле!**