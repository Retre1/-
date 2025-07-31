# 🚀 Advanced ForexBot AI - Enhanced Edition

## 📋 Обзор проекта

**Advanced ForexBot AI - Enhanced Edition** - это профессиональная торговая система с продвинутыми AI моделями, комплексным мониторингом, системой тестирования, CI/CD pipeline, кэшированием и горизонтальным масштабированием.

## ✨ Новые возможности

### 🔍 Система мониторинга
- **Prometheus метрики** для отслеживания производительности
- **Grafana дашборды** для визуализации
- **Алертинг система** для критических событий
- **Системные метрики** (CPU, Memory, Disk, Network)

### 🧪 Система тестирования
- **Unit тесты** для всех компонентов
- **Интеграционные тесты** для полной системы
- **Тесты производительности** под нагрузкой
- **Покрытие кода** с отчетами

### 🔄 CI/CD Pipeline
- **Автоматическое тестирование** при каждом коммите
- **Сканирование безопасности** с Bandit и Safety
- **Сборка Docker образов**
- **Автоматическое развертывание**

### 🗄️ Кэширование Redis
- **Кэширование предсказаний** AI моделей
- **Кэширование результатов** backtesting
- **Кэширование аналитических данных**
- **Оптимизация производительности**

### 🚀 Горизонтальное масштабирование
- **Распределенная архитектура** с множественными узлами
- **Балансировщик нагрузки** с различными стратегиями
- **Автоматическое масштабирование**
- **Координация через Redis**

## 🏗️ Архитектура системы

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Monitoring    │    │   CI/CD Pipeline│
│                 │    │   System        │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Node 1        │    │   Node 2        │    │   Node N        │
│   (AI Models)   │    │   (Backtesting) │    │   (Analytics)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Redis Cache   │    │   Database      │    │   Security      │
│                 │    │   (PostgreSQL)  │    │   (JWT)         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
# Клонирование репозитория
git clone https://github.com/your-repo/forexbot-ai-enhanced.git
cd forexbot-ai-enhanced

# Создание виртуального окружения
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# Установка зависимостей
pip install -r requirements.txt
```

### 2. Настройка Redis

```bash
# Установка Redis
sudo apt-get install redis-server  # Ubuntu/Debian
# или
brew install redis  # macOS

# Запуск Redis
redis-server
```

### 3. Запуск системы

```bash
# Запуск основного приложения
python integrated_bot_advanced.py

# Запуск узла кластера
python scaling_system.py

# Запуск системы мониторинга
python monitoring_system.py
```

## 📊 Мониторинг и метрики

### Prometheus метрики

Система предоставляет следующие метрики:

- **Торговые метрики**: `trades_total`, `total_profit`, `win_rate`
- **AI метрики**: `prediction_latency`, `model_accuracy`
- **Системные метрики**: `cpu_usage`, `memory_usage`, `disk_usage`
- **API метрики**: `api_requests_total`, `api_response_time`

### Доступ к метрикам

```bash
# Метрики Prometheus
curl http://localhost:8000/metrics

# Системная информация
curl http://localhost:8000/api/monitoring/system

# Метрики приложения
curl http://localhost:8000/api/monitoring/metrics
```

## 🧪 Тестирование

### Запуск тестов

```bash
# Все тесты
pytest tests/ -v

# Unit тесты
pytest tests/test_ai_models.py -v
pytest tests/test_backtesting.py -v

# Интеграционные тесты
pytest tests/test_integration.py -v

# Тесты с покрытием
pytest tests/ --cov=. --cov-report=html
```

### Тесты производительности

```bash
# Тесты производительности
pytest tests/test_integration.py::TestSystemIntegration::test_performance_under_load -v --benchmark-only
```

## 🔄 CI/CD Pipeline

### GitHub Actions

Система включает автоматизированный CI/CD pipeline:

1. **Тестирование**: Unit и интеграционные тесты
2. **Сканирование безопасности**: Bandit и Safety
3. **Проверка качества кода**: Flake8, Black, MyPy
4. **Сборка Docker образа**
5. **Автоматическое развертывание**

### Локальная проверка

```bash
# Проверка качества кода
flake8 .
black --check .
mypy . --ignore-missing-imports

# Сканирование безопасности
bandit -r .
safety check
```

## 🗄️ Кэширование

### Redis кэширование

Система использует Redis для кэширования:

- **Предсказания AI моделей** (TTL: 5 минут)
- **Результаты backtesting** (TTL: 1 час)
- **Аналитические данные** (TTL: 30 минут)
- **Рыночные данные** (TTL: 1 минута)

### Использование кэша

```python
from cache_system import CacheManager, PredictionCache

# Создание менеджера кэша
cache_manager = CacheManager(host='localhost', port=6379)

# Создание специализированного кэша
prediction_cache = PredictionCache(cache_manager)

# Сохранение предсказания
prediction_cache.set_cached_prediction('EURUSD', 'H1', 100, prediction_data)

# Получение предсказания
cached_prediction = prediction_cache.get_cached_prediction('EURUSD', 'H1', 100)
```

## 🚀 Горизонтальное масштабирование

### Архитектура кластера

Система поддерживает горизонтальное масштабирование:

- **Множественные узлы** для обработки задач
- **Балансировщик нагрузки** с различными стратегиями
- **Автоматическое масштабирование** на основе нагрузки
- **Координация через Redis**

### Запуск кластера

```bash
# Запуск первого узла
python scaling_system.py --port 8001

# Запуск второго узла
python scaling_system.py --port 8002

# Запуск балансировщика
python load_balancer.py
```

### API кластера

```bash
# Статус узла
curl http://localhost:8001/api/status

# Статус кластера
curl http://localhost:8001/api/cluster/status

# Масштабирование
curl -X POST http://localhost:8001/api/cluster/scale \
  -H "Content-Type: application/json" \
  -d '{"action": "up", "target_nodes": 3}'
```

## 🔐 Безопасность

### JWT аутентификация

```python
from security_integration import SecurityManager

# Создание менеджера безопасности
security_manager = SecurityManager()

# Создание пользователя
user = security_manager.create_user({
    'username': 'admin',
    'email': 'admin@example.com',
    'password': 'secure_password',
    'role': 'admin'
})

# Аутентификация
token = security_manager.create_access_token(user)
```

### Защищенные endpoints

```python
from fastapi import Depends
from security_integration import get_current_user

@app.get("/api/protected")
async def protected_endpoint(current_user = Depends(get_current_user)):
    return {"message": "Доступ разрешен", "user": current_user.username}
```

## 📈 Аналитика и отчеты

### Метрики производительности

- **Sharpe Ratio**: Мера риск-скорректированной доходности
- **Sortino Ratio**: Мера доходности с учетом downside risk
- **Calmar Ratio**: Отношение годовой доходности к максимальной просадке
- **Profit Factor**: Отношение общей прибыли к общим убыткам
- **Maximum Drawdown**: Максимальная просадка

### Генерация отчетов

```python
from analytics_enhancement import AnalyticsManager

# Создание менеджера аналитики
analytics_manager = AnalyticsManager()

# Расчет метрик
metrics = analytics_manager.calculate_performance_metrics()

# Генерация отчета
report = analytics_manager.generate_report()
```

## 🔔 Уведомления

### Настройка уведомлений

```python
from notifications_system import NotificationManager

# Конфигурация уведомлений
config = {
    'telegram': {
        'enabled': True,
        'bot_token': 'your_bot_token',
        'chat_id': 'your_chat_id'
    },
    'email': {
        'enabled': True,
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'your_email@gmail.com',
        'password': 'your_password'
    }
}

# Создание менеджера уведомлений
notification_manager = NotificationManager(config)

# Отправка уведомления
await notification_manager.send_trade_notification(trade_data)
```

## 🗄️ База данных

### Модели данных

```python
from database_integration import Trade, Signal, TokenBurn, BotLog

# Сохранение сделки
trade = Trade(
    symbol='EURUSD',
    direction='BUY',
    volume=1000.0,
    open_price=1.0850,
    strategy='AI_Ensemble',
    confidence=0.75
)
```

### Миграции

```bash
# Создание миграции
alembic revision --autogenerate -m "Add new fields"

# Применение миграций
alembic upgrade head
```

## 🐳 Docker развертывание

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "integrated_bot_advanced.py"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  forexbot:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: forexbot
      POSTGRES_USER: forexbot
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

## 📊 Производительность

### Оптимизация

- **Кэширование** часто используемых данных
- **Асинхронная обработка** для улучшения производительности
- **Горизонтальное масштабирование** для обработки больших нагрузок
- **Оптимизация AI моделей** для быстрых предсказаний

### Бенчмарки

```bash
# Тест производительности предсказаний
pytest tests/test_integration.py::TestSystemIntegration::test_performance_under_load -v --benchmark-only

# Тест памяти
python -m memory_profiler performance_test.py
```

## 🔧 Конфигурация

### Основные настройки

```json
{
  "ai": {
    "models": ["lstm", "xgboost", "lightgbm"],
    "timeframes": ["M15", "H1", "H4"],
    "lookback_periods": [50, 100, 200],
    "retrain_interval": 24,
    "min_accuracy_threshold": 0.65
  },
  "risk": {
    "max_risk_per_trade": 0.02,
    "max_daily_loss": 0.05,
    "max_concurrent_trades": 5,
    "stop_loss_pips": 50,
    "take_profit_pips": 100
  },
  "monitoring": {
    "prometheus_enabled": true,
    "metrics_port": 8001,
    "alert_rules": [
      {
        "name": "High CPU Usage",
        "type": "cpu_high",
        "threshold": 80,
        "severity": "warning"
      }
    ]
  },
  "caching": {
    "redis_host": "localhost",
    "redis_port": 6379,
    "prediction_ttl": 300,
    "backtest_ttl": 3600
  }
}
```

## 🚨 Алертинг

### Правила алертов

```python
# Добавление правила алерта
alert_manager.add_alert_rule({
    'name': 'High Drawdown Alert',
    'type': 'drawdown_high',
    'threshold': 15,
    'severity': 'critical',
    'notification_type': 'email',
    'recipients': ['admin@example.com']
})
```

### Уведомления

- **Email уведомления** для критических событий
- **Telegram уведомления** для быстрых алертов
- **Webhook уведомления** для интеграции с внешними системами
- **Discord уведомления** для командных чатов

## 📚 API документация

### Основные endpoints

- `GET /` - Главная страница
- `GET /api/status` - Статус системы
- `GET /api/trades` - Список сделок
- `POST /api/trades` - Создание сделки
- `GET /api/predictions` - Предсказания AI моделей
- `POST /api/backtest` - Запуск backtesting
- `GET /api/analytics` - Аналитические данные
- `GET /metrics` - Prometheus метрики

### WebSocket endpoints

- `WS /ws/trades` - Real-time обновления сделок
- `WS /ws/predictions` - Real-time предсказания
- `WS /ws/alerts` - Real-time алерты

## 🔍 Отладка и логирование

### Логирование

```python
from loguru import logger

# Настройка логирования
logger.add("logs/forexbot.log", rotation="10 MB", retention="7 days")

# Логирование
logger.info("Система запущена")
logger.warning("Высокая загрузка CPU")
logger.error("Ошибка подключения к базе данных")
```

### Отладка

```bash
# Запуск в режиме отладки
python integrated_bot_advanced.py --debug

# Просмотр логов
tail -f logs/forexbot.log

# Мониторинг метрик
curl http://localhost:8000/metrics | grep trades_total
```

## 🤝 Вклад в проект

### Разработка

1. **Fork** репозитория
2. **Создайте** feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** изменения (`git commit -m 'Add amazing feature'`)
4. **Push** в branch (`git push origin feature/amazing-feature`)
5. **Откройте** Pull Request

### Тестирование

```bash
# Запуск всех тестов
pytest tests/ -v

# Проверка покрытия
pytest tests/ --cov=. --cov-report=html

# Проверка качества кода
flake8 .
black --check .
mypy . --ignore-missing-imports
```

## 📄 Лицензия

Этот проект лицензирован под MIT License - см. файл [LICENSE](LICENSE) для деталей.

## ⚠️ Отказ от ответственности

**ВАЖНО**: Этот проект предназначен только для образовательных и исследовательских целей. Торговля на финансовых рынках сопряжена с высокими рисками. Авторы не несут ответственности за любые финансовые потери, связанные с использованием этого программного обеспечения.

## 📞 Поддержка

- **Issues**: [GitHub Issues](https://github.com/your-repo/forexbot-ai-enhanced/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/forexbot-ai-enhanced/discussions)
- **Email**: support@forexbot-ai.com

## 🎯 Roadmap

### Планируемые улучшения

- [ ] Интеграция с дополнительными брокерами
- [ ] Расширенные AI модели (Transformer, BERT)
- [ ] Real-time обработка рыночных данных
- [ ] Интеграция с криптовалютными биржами
- [ ] Мобильное приложение
- [ ] Web интерфейс с React
- [ ] Интеграция с Telegram Bot API
- [ ] Расширенная аналитика с ML
- [ ] Автоматическая оптимизация стратегий
- [ ] Интеграция с внешними API данных

---

**🚀 Advanced ForexBot AI - Enhanced Edition** - Профессиональная торговая система нового поколения!