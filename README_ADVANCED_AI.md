# 🤖 Advanced ForexBot AI - Профессиональная торговая система

## 📋 Обзор

Advanced ForexBot AI - это комплексная торговая система с профессиональными AI моделями, предназначенная для автоматизированной торговли на финансовых рынках. Система объединяет передовые технологии машинного обучения с надежной архитектурой для максимальной эффективности.

## 🚀 Ключевые особенности

### 🧠 Продвинутые AI модели
- **LSTM (Long Short-Term Memory)** - для анализа временных рядов
- **Transformer** - для обработки последовательностей с attention механизмом
- **XGBoost** - градиентный бустинг для классификации
- **LightGBM** - быстрый градиентный бустинг
- **Random Forest** - ансамблевый метод
- **Gradient Boosting** - дополнительный бустинг

### 📊 Инженер признаков
- **Технические индикаторы**: SMA, EMA, MACD, RSI, Bollinger Bands, Stochastic, ATR
- **Продвинутые признаки**: Фракталы, дивергенции, мультитаймфреймный анализ
- **Режимы рынка**: Определение трендового/бокового/волатильного рынка
- **Анализ волатильности**: Динамическая оценка рыночных условий

### 🔄 Backtesting система
- **Реалистичное моделирование** торговых операций
- **Метрики производительности**: Sharpe Ratio, Sortino Ratio, Calmar Ratio
- **Анализ просадки** и управления рисками
- **Сравнение моделей** и оптимизация параметров

### 🛡️ Управление рисками
- **Динамическое позиционирование** на основе волатильности
- **Лимиты риска** на сделку и общий портфель
- **Стоп-лоссы** и тейк-профиты
- **Максимальная просадка** контроль

## 📦 Установка

### 1. Клонирование репозитория
```bash
git clone <repository-url>
cd forexbot-ai
```

### 2. Автоматическая установка зависимостей
```bash
python install_advanced_dependencies.py
```

### 3. Ручная установка (альтернатива)
```bash
# Создание виртуального окружения
python3 -m venv venv

# Активация (Linux/Mac)
source venv/bin/activate
# Активация (Windows)
venv\Scripts\activate

# Установка зависимостей
pip install -r requirements.txt
```

## 🚀 Запуск

### Быстрый запуск
```bash
# Linux/Mac
./start_advanced_bot.sh

# Windows
start_advanced_bot.bat
```

### Ручной запуск
```bash
# Активация виртуального окружения
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Запуск бота
python integrated_bot_advanced.py
```

## 🌐 Веб-интерфейс

После запуска откройте браузер и перейдите по адресу:
- **Главная страница**: http://localhost:8000
- **API документация**: http://localhost:8000/docs
- **Альтернативная документация**: http://localhost:8000/redoc

## 📊 API Endpoints

### Основные endpoints
- `GET /api/status` - Статус бота
- `POST /api/start` - Запуск бота
- `POST /api/stop` - Остановка бота
- `GET /api/trades` - История сделок
- `GET /api/positions` - Открытые позиции
- `GET /api/statistics` - Статистика торговли

### AI модели
- `GET /api/ai/models` - Информация о моделях
- `POST /api/ai/retrain` - Переобучение моделей
- `GET /api/backtest` - Запуск backtesting

### WebSocket
- `WS /ws` - Real-time обновления

## ⚙️ Конфигурация

Файл `config.json` содержит все настройки системы:

```json
{
  "initial_capital": 10000,
  "max_risk_per_trade": 0.02,
  "max_positions": 5,
  "confidence_threshold": 0.6,
  "enable_ai_trading": true,
  "enable_risk_management": true,
  "ai_models": {
    "lstm": {
      "enabled": true,
      "sequence_length": 60,
      "n_features": 30
    },
    "transformer": {
      "enabled": true,
      "sequence_length": 60,
      "n_features": 30,
      "n_heads": 8
    }
  }
}
```

## 🧠 AI модели

### LSTM модель
```python
class AdvancedLSTMModel:
    """Продвинутая LSTM модель для анализа временных рядов"""
    
    def __init__(self, sequence_length=60, n_features=30):
        self.sequence_length = sequence_length
        self.n_features = n_features
        
    def create_model(self):
        """Создание архитектуры LSTM"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(sequence_length, n_features)),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(3, activation='softmax')  # BUY, SELL, HOLD
        ])
        return model
```

### Transformer модель
```python
class AdvancedTransformerModel:
    """Продвинутая Transformer модель с attention механизмом"""
    
    def create_model(self):
        """Создание архитектуры Transformer"""
        inputs = Input(shape=(sequence_length, n_features))
        
        # Positional encoding
        pos_encoding = self._positional_encoding(sequence_length, n_features)
        x = inputs + pos_encoding
        
        # Transformer blocks
        for _ in range(4):
            x = self.create_transformer_block(x, 64, n_heads, 128, dropout=0.1)
            
        # Global average pooling
        x = GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = Dense(128, activation="relu")(x)
        x = Dense(64, activation="relu")(x)
        outputs = Dense(3, activation="softmax")(x)
        
        return Model(inputs=inputs, outputs=outputs)
```

### Ансамбль моделей
```python
class AdvancedEnsembleModel:
    """Продвинутый ансамбль моделей с взвешенным голосованием"""
    
    def __init__(self):
        self.models = {}
        self.weights = {
            'lstm': 0.25,
            'transformer': 0.25,
            'xgboost': 0.20,
            'lightgbm': 0.15,
            'random_forest': 0.10,
            'gradient_boosting': 0.05
        }
```

## 📈 Backtesting

### Запуск backtesting
```python
from advanced_backtesting import AdvancedBacktester, PerformanceAnalyzer

# Создание backtester
backtester = AdvancedBacktester(initial_capital=10000)

# Запуск backtesting
results = backtester.run_backtest(market_data, predictions)

# Анализ результатов
analyzer = PerformanceAnalyzer()
analyzer.analyze_model_performance(results, "ensemble")
report = analyzer.generate_report()
```

### Метрики производительности
- **Total Return** - Общая доходность
- **Sharpe Ratio** - Коэффициент Шарпа
- **Max Drawdown** - Максимальная просадка
- **Win Rate** - Процент прибыльных сделок
- **Profit Factor** - Фактор прибыли
- **Calmar Ratio** - Коэффициент Кальмара
- **Sortino Ratio** - Коэффициент Сортино

## 🔧 Оптимизация моделей

### Автоматическая оптимизация
```python
from advanced_backtesting import ModelOptimizer

optimizer = ModelOptimizer(ensemble_model)
best_params = optimizer.optimize_hyperparameters(df, target)
```

### Параметры для оптимизации
- **XGBoost**: n_estimators, max_depth, learning_rate, subsample, colsample_bytree
- **LightGBM**: n_estimators, max_depth, learning_rate, subsample, colsample_bytree
- **Random Forest**: n_estimators, max_depth, min_samples_split, min_samples_leaf

## 📊 Аналитика и отчетность

### Генерация отчетов
```python
from analytics_enhancement import AnalyticsManager

analytics = AnalyticsManager()
report = analytics.generate_report()
```

### Типы отчетов
- **Performance Metrics** - Метрики производительности
- **Strategy Analysis** - Анализ стратегий
- **Symbol Analysis** - Анализ по символам
- **Signal Analysis** - Анализ сигналов
- **Equity Curves** - Кривые доходности
- **Drawdown Analysis** - Анализ просадки

## 🔔 Уведомления

### Настройка уведомлений
```json
{
  "notifications": {
    "email": {
      "enabled": true,
      "smtp_server": "smtp.gmail.com",
      "smtp_port": 587,
      "username": "your-email@gmail.com",
      "password": "your-app-password"
    },
    "telegram": {
      "enabled": true,
      "bot_token": "your-bot-token",
      "chat_id": "your-chat-id"
    }
  }
}
```

### Типы алертов
- **Profit Threshold** - Достижение целевой прибыли
- **Loss Threshold** - Превышение допустимых убытков
- **Drawdown Threshold** - Критическая просадка
- **Trade Count** - Количество сделок
- **Signal Confidence** - Высокая уверенность сигнала

## 🛡️ Безопасность

### Аутентификация
```python
from security_integration import SecurityManager

security_manager = SecurityManager()
user = security_manager.authenticate_user(username, password)
token = security_manager.create_access_token(user)
```

### Роли пользователей
- **Admin** - Полный доступ к системе
- **User** - Ограниченный доступ к торговле
- **Viewer** - Только просмотр статистики

## 🗄️ База данных

### Модели данных
```python
class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    direction = Column(String(10), nullable=False)
    volume = Column(Float, nullable=False)
    open_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=True)
    profit = Column(Float, nullable=True)
    status = Column(String(20), default="OPEN")
```

### Операции с БД
```python
from database_integration import DatabaseManager

db = DatabaseManager()
await db.save_trade(trade_data)
trades = await db.get_trades(limit=100)
stats = await db.get_trade_statistics(days=30)
```

## 📁 Структура проекта

```
forexbot-ai/
├── integrated_bot_advanced.py      # Основной файл бота
├── advanced_ai_models.py           # AI модели
├── advanced_backtesting.py         # Backtesting система
├── analytics_enhancement.py        # Аналитика
├── database_integration.py         # База данных
├── security_integration.py         # Безопасность
├── notifications_system.py         # Уведомления
├── install_advanced_dependencies.py # Установка зависимостей
├── config.json                     # Конфигурация
├── requirements.txt                # Зависимости
├── data/
│   ├── logs/                      # Логи
│   └── models/                    # Сохраненные модели
├── models/                        # Обученные модели
├── backtests/                     # Результаты backtesting
├── reports/                       # Отчеты
└── web_interface/
    └── frontend/                  # Веб-интерфейс
```

## 🚀 Производительность

### Оптимизация для продакшена
- **Асинхронная обработка** - FastAPI для высокой производительности
- **Кэширование** - Redis для кэширования данных
- **Масштабирование** - Docker контейнеризация
- **Мониторинг** - Prometheus + Grafana

### Рекомендуемые требования
- **CPU**: 4+ ядра
- **RAM**: 8+ GB
- **Storage**: 50+ GB SSD
- **OS**: Linux (Ubuntu 20.04+)

## 🔧 Устранение неполадок

### Частые проблемы

#### 1. Ошибка импорта TensorFlow
```bash
# Установка CUDA для GPU поддержки
pip install tensorflow-gpu

# Или использование CPU версии
pip install tensorflow
```

#### 2. Ошибка памяти при обучении
```python
# Уменьшение размера batch
model.fit(X_train, y_train, batch_size=16)

# Использование генераторов данных
from tensorflow.keras.utils import Sequence
```

#### 3. Медленная работа моделей
```python
# Оптимизация XGBoost
xgb_model = xgb.XGBClassifier(
    n_jobs=-1,  # Использование всех ядер
    tree_method='hist'  # Быстрый алгоритм
)
```

## 📚 Дополнительные ресурсы

### Документация
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)

### Сообщество
- [GitHub Issues](https://github.com/your-repo/issues)
- [Discord Server](https://discord.gg/your-server)
- [Telegram Channel](https://t.me/your-channel)

## 📄 Лицензия

MIT License - см. файл [LICENSE](LICENSE) для подробностей.

## 🤝 Вклад в проект

Мы приветствуем вклад в развитие проекта! Пожалуйста, ознакомьтесь с [CONTRIBUTING.md](CONTRIBUTING.md) для получения подробной информации.

## 📞 Поддержка

- **Email**: support@forexbot-ai.com
- **Telegram**: @forexbot_support
- **Discord**: #support

---

**⚠️ Важное предупреждение**: Торговля на финансовых рынках связана с высокими рисками. Используйте эту систему на свой страх и риск. Авторы не несут ответственности за возможные финансовые потери.