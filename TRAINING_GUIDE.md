# 🧠 Руководство по обучению и сохранению AI моделей

## 📚 Обзор процесса обучения

### 🎯 Этапы обучения моделей:

1. **Подготовка данных** - загрузка и обработка исторических данных
2. **Создание признаков** - технические индикаторы и продвинутые признаки
3. **Обучение моделей** - тренировка различных AI алгоритмов
4. **Валидация** - проверка качества на тестовых данных
5. **Backtesting** - симуляция торговли на исторических данных
6. **Сохранение** - запись моделей и метаданных
7. **Развертывание** - загрузка моделей для торговли

## 🚀 Быстрый старт обучения

### 1. Подготовка окружения

```bash
# Установка зависимостей
pip install -r requirements.txt

# Создание директорий для моделей
mkdir -p data/models
mkdir -p data/backtests
mkdir -p data/reports
```

### 2. Базовое обучение моделей

```python
from training_pipeline import ModelTrainingPipeline

# Конфигурация
config = {
    "ai": {
        "models": ["lstm", "xgboost", "lightgbm"],
        "timeframes": ["M15", "H1", "H4"],
        "retrain_interval": 24,
        "min_accuracy_threshold": 0.65
    }
}

# Создание пайплайна
pipeline = ModelTrainingPipeline(config)

# Обучение моделей
results = pipeline.train_models(
    symbol="EURUSD",
    timeframe="H1",
    start_date="2023-01-01",
    end_date="2023-12-31"
)

print("✅ Обучение завершено!")
```

### 3. Загрузка обученных моделей

```python
# Загрузка моделей
success = pipeline.load_models("EURUSD", "H1")
if success:
    print("✅ Модели загружены успешно!")
else:
    print("❌ Ошибка загрузки моделей")
```

## 📊 Структура сохранения моделей

### 🗂️ Директории моделей

```
data/models/
├── EURUSD/
│   ├── H1/
│   │   ├── lstm/
│   │   │   ├── lstm_20231201_143022.joblib
│   │   │   ├── lstm_20231201_143022_metadata.json
│   │   │   └── lstm_20231201_143022.h5
│   │   ├── xgboost/
│   │   │   ├── xgboost_20231201_143022.joblib
│   │   │   └── xgboost_20231201_143022_metadata.json
│   │   ├── lightgbm/
│   │   │   ├── lightgbm_20231201_143022.joblib
│   │   │   └── lightgbm_20231201_143022_metadata.json
│   │   └── training_results/
│   │       ├── training_results_20231201_143022.json
│   │       └── summary_20231201_143022.json
│   └── H4/
│       └── ...
└── GBPUSD/
    └── ...
```

### 📋 Метаданные моделей

```json
{
  "model_name": "lstm",
  "symbol": "EURUSD",
  "timeframe": "H1",
  "training_date": "2023-12-01T14:30:22",
  "model_path": "data/models/EURUSD/H1/lstm/lstm_20231201_143022.joblib",
  "training_metrics": {
    "accuracy": 0.7234,
    "loss": 0.4567,
    "val_accuracy": 0.6987,
    "val_loss": 0.5123
  },
  "model_type": "AdvancedLSTMModel",
  "feature_count": 45,
  "training_samples": 8760,
  "validation_samples": 2190
}
```

## 🔧 Детальная настройка обучения

### 1. Настройка параметров моделей

```python
# Конфигурация для разных типов моделей
config = {
    "ai": {
        "models": ["lstm", "xgboost", "lightgbm"],
        "timeframes": ["M15", "H1", "H4"],
        "lookback_periods": [50, 100, 200],
        "retrain_interval": 24,
        "min_accuracy_threshold": 0.65,
        "model_params": {
            "lstm": {
                "sequence_length": 20,
                "units": 128,
                "dropout": 0.2,
                "epochs": 100,
                "batch_size": 32
            },
            "xgboost": {
                "n_estimators": 1000,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8
            },
            "lightgbm": {
                "n_estimators": 1000,
                "max_depth": 6,
                "learning_rate": 0.1,
                "num_leaves": 31
            }
        }
    }
}
```

### 2. Настройка признаков

```python
# Создание кастомного инженера признаков
from advanced_ai_models import AdvancedFeatureEngineer

feature_engineer = AdvancedFeatureEngineer()

# Добавление кастомных признаков
def add_custom_features(df):
    # Кастомные технические индикаторы
    df['custom_momentum'] = df['close'].pct_change(periods=5)
    df['custom_volatility'] = df['close'].rolling(window=20).std()
    
    # Кастомные паттерны
    df['price_pattern'] = (df['close'] > df['open']).astype(int)
    
    return df

# Применение кастомных признаков
df = add_custom_features(df)
```

### 3. Настройка валидации

```python
# Кастомная валидация
def custom_validation(X_val, y_val, model):
    predictions = model.predict(X_val)
    
    # Расчет дополнительных метрик
    from sklearn.metrics import classification_report, confusion_matrix
    
    report = classification_report(y_val, predictions, output_dict=True)
    conf_matrix = confusion_matrix(y_val, predictions)
    
    return {
        'accuracy': report['accuracy'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score'],
        'confusion_matrix': conf_matrix.tolist()
    }
```

## 📈 Мониторинг процесса обучения

### 1. Логирование обучения

```python
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Логирование процесса обучения
logger.info("🚀 Начало обучения моделей")
logger.info(f"📊 Данные: {len(X_train)} обучающих, {len(X_val)} валидационных")
logger.info(f"🎯 Целевая переменная: {len(np.unique(y_train))} классов")
```

### 2. Визуализация результатов

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_results(results):
    """Визуализация результатов обучения"""
    
    # График точности моделей
    models = list(results['validation_results'].keys())
    accuracies = [results['validation_results'][m].get('accuracy', 0) for m in models]
    
    plt.figure(figsize=(10, 6))
    plt.bar(models, accuracies)
    plt.title('Точность моделей')
    plt.ylabel('Точность')
    plt.ylim(0, 1)
    
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('data/reports/model_accuracy.png')
    plt.show()
    
    # График результатов backtesting
    backtest = results['backtest_results']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Винрейт
    axes[0, 0].pie([backtest['win_rate'], 100-backtest['win_rate']], 
                    labels=['Прибыльные', 'Убыточные'], autopct='%1.1f%%')
    axes[0, 0].set_title('Винрейт')
    
    # Общая прибыль
    axes[0, 1].bar(['Прибыль'], [backtest['total_profit']])
    axes[0, 1].set_title('Общая прибыль')
    
    # Sharpe Ratio
    axes[1, 0].bar(['Sharpe'], [backtest['sharpe_ratio']])
    axes[1, 0].set_title('Sharpe Ratio')
    
    # Количество сделок
    axes[1, 1].bar(['Сделки'], [backtest['total_trades']])
    axes[1, 1].set_title('Общее количество сделок')
    
    plt.tight_layout()
    plt.savefig('data/reports/backtest_results.png')
    plt.show()
```

## 🔄 Автоматическое переобучение

### 1. Настройка автоматического обучения

```python
from model_manager import ModelManager

# Создание менеджера моделей
model_manager = ModelManager(config)

# Инициализация с автоматическим обучением
success = model_manager.initialize_models(
    symbols=["EURUSD", "GBPUSD", "USDJPY"],
    timeframes=["H1", "H4"]
)

if success:
    print("✅ Модели инициализированы и готовы к торговле!")
```

### 2. Планировщик переобучения

```python
# Настройка расписания переобучения
config = {
    "ai": {
        "auto_training": True,
        "retrain_interval": 24,  # часы
        "min_accuracy_threshold": 0.65,
        "retrain_on_accuracy_drop": True,
        "accuracy_drop_threshold": 0.05
    }
}

# Менеджер автоматически будет переобучать модели каждые 24 часа
model_manager = ModelManager(config)
```

### 3. Мониторинг качества моделей

```python
# Проверка качества моделей
def check_model_quality(model_manager):
    status = model_manager.get_models_status()
    
    for model_key, model_info in status['models_status'].items():
        accuracy = model_info.get('accuracy', 0)
        threshold = status['min_accuracy_threshold']
        
        if accuracy < threshold:
            print(f"⚠️ Низкая точность модели {model_key}: {accuracy:.4f}")
            
            # Принудительное переобучение
            symbol, timeframe = model_key.split('_')
            model_manager.retrain_models(symbol, timeframe, force=True)
        else:
            print(f"✅ Модель {model_key} в норме: {accuracy:.4f}")

# Запуск проверки каждые 6 часов
import schedule
schedule.every(6).hours.do(check_model_quality, model_manager)
```

## 🗄️ Управление версиями моделей

### 1. Сохранение версий

```python
# Автоматическое сохранение версий
def save_model_version(pipeline, symbol, timeframe, version_name=None):
    """Сохранение версии модели"""
    
    if version_name is None:
        version_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Обучение модели
    results = pipeline.train_models(symbol, timeframe, 
                                  start_date="2023-01-01", 
                                  end_date="2023-12-31")
    
    # Сохранение с версией
    model_dir = Path(f"data/models/{symbol}/{timeframe}")
    version_dir = model_dir / f"version_{version_name}"
    version_dir.mkdir(parents=True, exist_ok=True)
    
    # Копирование файлов с версией
    for model_name in results['model_info'].keys():
        source_file = Path(results['model_info'][model_name]['path'])
        target_file = version_dir / f"{model_name}_{version_name}.joblib"
        
        import shutil
        shutil.copy2(source_file, target_file)
    
    print(f"✅ Версия {version_name} сохранена")
    return version_name
```

### 2. Загрузка конкретной версии

```python
# Загрузка конкретной версии
def load_model_version(pipeline, symbol, timeframe, version_name):
    """Загрузка конкретной версии модели"""
    
    model_dir = Path(f"data/models/{symbol}/{timeframe}/version_{version_name}")
    
    if not model_dir.exists():
        print(f"❌ Версия {version_name} не найдена")
        return False
    
    # Загрузка моделей из версии
    for model_file in model_dir.glob("*.joblib"):
        model_name = model_file.stem.split('_')[0]
        model = joblib.load(model_file)
        pipeline.ensemble_model.models[model_name] = model
    
    print(f"✅ Версия {version_name} загружена")
    return True
```

### 3. Сравнение версий

```python
# Сравнение версий моделей
def compare_model_versions(pipeline, symbol, timeframe, versions):
    """Сравнение версий моделей"""
    
    comparison = {}
    
    for version in versions:
        if load_model_version(pipeline, symbol, timeframe, version):
            # Тестирование на валидационных данных
            # ... код тестирования ...
            
            comparison[version] = {
                'accuracy': accuracy,
                'backtest_results': backtest_results
            }
    
    # Создание отчета сравнения
    report_path = f"data/reports/version_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"📋 Отчет сравнения сохранен: {report_path}")
    return comparison
```

## 🚨 Устранение проблем

### 1. Проблемы с обучением

```python
# Диагностика проблем обучения
def diagnose_training_issues(pipeline, symbol, timeframe):
    """Диагностика проблем обучения"""
    
    try:
        # Проверка данных
        df, features, target, feature_names = pipeline.prepare_training_data(
            symbol, timeframe, "2023-01-01", "2023-12-31"
        )
        
        print(f"📊 Данные:")
        print(f"   Записей: {len(features)}")
        print(f"   Признаков: {len(feature_names)}")
        print(f"   Классов: {len(np.unique(target))}")
        
        # Проверка баланса классов
        class_counts = np.bincount(target)
        print(f"   Распределение классов: {class_counts}")
        
        # Проверка на NaN
        nan_count = np.isnan(features).sum()
        print(f"   NaN значений: {nan_count}")
        
        if nan_count > 0:
            print("⚠️ Обнаружены NaN значения!")
        
        # Проверка на бесконечные значения
        inf_count = np.isinf(features).sum()
        print(f"   Бесконечных значений: {inf_count}")
        
        if inf_count > 0:
            print("⚠️ Обнаружены бесконечные значения!")
        
    except Exception as e:
        print(f"❌ Ошибка диагностики: {e}")
```

### 2. Проблемы с сохранением

```python
# Проверка сохранения моделей
def check_model_saving(pipeline, symbol, timeframe):
    """Проверка сохранения моделей"""
    
    try:
        # Попытка сохранения
        results = pipeline.train_models(symbol, timeframe, 
                                      "2023-01-01", "2023-12-31")
        
        # Проверка файлов
        model_dir = Path(f"data/models/{symbol}/{timeframe}")
        
        for model_name in results['model_info'].keys():
            model_path = Path(results['model_info'][model_name]['path'])
            
            if model_path.exists():
                file_size = model_path.stat().st_size
                print(f"✅ {model_name}: {file_size} байт")
                
                # Проверка загрузки
                try:
                    model = joblib.load(model_path)
                    print(f"✅ {model_name}: загрузка успешна")
                except Exception as e:
                    print(f"❌ {model_name}: ошибка загрузки - {e}")
            else:
                print(f"❌ {model_name}: файл не найден")
                
    except Exception as e:
        print(f"❌ Ошибка проверки сохранения: {e}")
```

## 📋 Чек-лист обучения

### ✅ Перед обучением:

- [ ] Установлены все зависимости
- [ ] Созданы необходимые директории
- [ ] Проверены исторические данные
- [ ] Настроена конфигурация
- [ ] Выбран период обучения

### ✅ Во время обучения:

- [ ] Мониторинг процесса обучения
- [ ] Проверка качества данных
- [ ] Валидация моделей
- [ ] Backtesting результатов
- [ ] Сохранение моделей

### ✅ После обучения:

- [ ] Проверка сохраненных файлов
- [ ] Тестирование загрузки моделей
- [ ] Валидация на новых данных
- [ ] Создание отчетов
- [ ] Настройка автоматического переобучения

## 🎯 Лучшие практики

### 1. Качество данных

```python
# Проверка качества данных
def validate_data_quality(df):
    """Проверка качества данных"""
    
    issues = []
    
    # Проверка на пропущенные значения
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        issues.append(f"Пропущенные значения: {missing_data.sum()}")
    
    # Проверка на дубликаты
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"Дубликаты: {duplicates}")
    
    # Проверка на выбросы
    for col in ['open', 'high', 'low', 'close']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        if outliers > 0:
            issues.append(f"Выбросы в {col}: {outliers}")
    
    return issues
```

### 2. Регулярное переобучение

```python
# Настройка регулярного переобучения
def setup_regular_retraining(model_manager):
    """Настройка регулярного переобучения"""
    
    # Ежедневное переобучение
    schedule.every().day.at("02:00").do(
        model_manager._auto_retrain_all_models
    )
    
    # Еженедельная проверка качества
    schedule.every().monday.at("10:00").do(
        check_model_quality, model_manager
    )
    
    # Ежемесячная очистка старых моделей
    schedule.every().month.at("01:00").do(
        model_manager.cleanup_old_models, 30
    )
```

### 3. Мониторинг производительности

```python
# Мониторинг производительности моделей
def monitor_model_performance(model_manager):
    """Мониторинг производительности моделей"""
    
    performance_log = []
    
    for symbol in ["EURUSD", "GBPUSD", "USDJPY"]:
        for timeframe in ["H1", "H4"]:
            try:
                performance = model_manager.get_model_performance(symbol, timeframe)
                
                log_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'accuracy': performance.get('overall_accuracy', 0),
                    'backtest_results': performance.get('backtest_results', {})
                }
                
                performance_log.append(log_entry)
                
                # Алерт при низкой точности
                if log_entry['accuracy'] < 0.6:
                    print(f"🚨 Низкая точность {symbol} {timeframe}: {log_entry['accuracy']:.4f}")
                    
            except Exception as e:
                print(f"❌ Ошибка мониторинга {symbol} {timeframe}: {e}")
    
    # Сохранение лога
    log_path = f"data/reports/performance_log_{datetime.now().strftime('%Y%m%d')}.json"
    with open(log_path, 'w') as f:
        json.dump(performance_log, f, indent=2)
    
    return performance_log
```

---

**🎯 Следуя этому руководству, вы сможете эффективно обучать, сохранять и управлять AI моделями для торгового бота!**