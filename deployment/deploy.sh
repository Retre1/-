#!/bin/bash
# 🚀 Скрипт автоматического развертывания ForexBot AI

set -e

echo "🚀 Развертывание ForexBot AI..."

# Проверка наличия Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker не установлен. Установите Docker сначала."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose не установлен. Установите Docker Compose сначала."
    exit 1
fi

# Создание директорий
echo "📁 Создание директорий..."
mkdir -p data/logs
mkdir -p data/models
mkdir -p data/backtests
mkdir -p data/reports
mkdir -p logs
mkdir -p ssl
mkdir -p grafana/dashboards
mkdir -p grafana/datasources

# Создание конфигурационного файла
echo "⚙️ Создание config.json..."
if [ ! -f config.json ]; then
    cat > config.json << 'EOF'
{
  "mt5": {
    "server": "YourBroker-Server",
    "login": 0,
    "password": "",
    "symbols": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "NZDUSD", "USDCAD"]
  },
  "ai": {
    "models": ["lstm", "xgboost", "lightgbm"],
    "timeframes": ["M15", "H1", "H4"],
    "lookback_periods": [50, 100, 200],
    "retrain_interval": 24,
    "min_accuracy_threshold": 0.65,
    "ensemble_weights": {
      "lstm": 0.4,
      "xgboost": 0.3,
      "lightgbm": 0.3
    }
  },
  "risk": {
    "max_risk_per_trade": 0.02,
    "max_daily_loss": 0.05,
    "max_concurrent_trades": 5,
    "stop_loss_pips": 50,
    "take_profit_pips": 100,
    "trailing_stop_pips": 30,
    "max_spread": 3.0,
    "min_margin_level": 200,
    "position_sizing_method": "fixed_risk"
  },
  "database": {
    "url": "postgresql://forexbot:forexbot_password@postgres:5432/forexbot",
    "backup_interval": 24,
    "retention_days": 365
  },
  "redis": {
    "host": "redis",
    "port": 6379,
    "db": 0
  },
  "web_interface": {
    "host": "0.0.0.0",
    "port": 8000,
    "cors_origins": ["*"],
    "secret_key": "your-secret-key-here",
    "session_expire": 3600
  },
  "monitoring": {
    "prometheus_port": 8001,
    "metrics_interval": 30
  }
}
EOF
fi

# Создание SSL сертификатов (самоподписанные для тестирования)
echo "🔒 Создание SSL сертификатов..."
if [ ! -f ssl/cert.pem ]; then
    openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
fi

# Создание инициализационного SQL
echo "🗄️ Создание init.sql..."
cat > init.sql << 'EOF'
-- Инициализация базы данных ForexBot AI
CREATE DATABASE IF NOT EXISTS forexbot;
\c forexbot;

-- Таблица пользователей
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Таблица сделок
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    volume DECIMAL(15,2) NOT NULL,
    open_price DECIMAL(15,5) NOT NULL,
    close_price DECIMAL(15,5),
    open_time TIMESTAMP NOT NULL,
    close_time TIMESTAMP,
    profit DECIMAL(15,2),
    status VARCHAR(20) DEFAULT 'OPEN',
    strategy VARCHAR(50),
    confidence DECIMAL(5,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Таблица сигналов
CREATE TABLE IF NOT EXISTS signals (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    model VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Таблица токенов
CREATE TABLE IF NOT EXISTS token_burns (
    id SERIAL PRIMARY KEY,
    amount DECIMAL(15,2) NOT NULL,
    reason VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Таблица логов
CREATE TABLE IF NOT EXISTS bot_logs (
    id SERIAL PRIMARY KEY,
    level VARCHAR(10) NOT NULL,
    message TEXT NOT NULL,
    module VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Индексы для производительности
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_open_time ON trades(open_time);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_created_at ON signals(created_at);

-- Создание пользователя по умолчанию
INSERT INTO users (username, email, password_hash, role) 
VALUES ('admin', 'admin@forexbot.ai', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj3ZxQQxq3Hy', 'admin')
ON CONFLICT (username) DO NOTHING;
EOF

# Создание конфигурации Grafana
echo "📊 Настройка Grafana..."
cat > grafana/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

# Сборка и запуск контейнеров
echo "🐳 Сборка и запуск контейнеров..."
docker-compose build
docker-compose up -d

# Ожидание запуска сервисов
echo "⏳ Ожидание запуска сервисов..."
sleep 30

# Проверка статуса
echo "🔍 Проверка статуса сервисов..."
docker-compose ps

echo "✅ Развертывание завершено!"
echo ""
echo "🌐 Доступные сервисы:"
echo "   - ForexBot AI: http://localhost:8000"
echo "   - Grafana: http://localhost:3000 (admin/admin)"
echo "   - Prometheus: http://localhost:9090"
echo ""
echo "📊 Логи:"
echo "   - docker-compose logs forexbot"
echo "   - docker-compose logs -f forexbot"
echo ""
echo "🛑 Остановка:"
echo "   - docker-compose down"
echo ""
echo "🔄 Перезапуск:"
echo "   - docker-compose restart forexbot"