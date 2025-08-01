#!/bin/bash
# 🔧 Скрипт ручного развертывания ForexBot AI (без Docker)

set -e

echo "🔧 Ручное развертывание ForexBot AI..."

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не установлен"
    exit 1
fi

# Создание виртуального окружения
echo "🐍 Создание виртуального окружения..."
python3 -m venv venv
source venv/bin/activate

# Установка зависимостей
echo "📦 Установка зависимостей..."
pip install --upgrade pip
pip install -r requirements.txt

# Установка системных зависимостей
echo "🔧 Установка системных зависимостей..."
sudo apt update
sudo apt install -y \
    postgresql \
    postgresql-contrib \
    redis-server \
    nginx \
    supervisor

# Настройка PostgreSQL
echo "🗄️ Настройка PostgreSQL..."
sudo systemctl enable postgresql
sudo systemctl start postgresql

# Создание пользователя и базы данных
sudo -u postgres psql << EOF
CREATE USER forexbot WITH PASSWORD 'forexbot_password';
CREATE DATABASE forexbot OWNER forexbot;
GRANT ALL PRIVILEGES ON DATABASE forexbot TO forexbot;
\q
EOF

# Настройка Redis
echo "🔴 Настройка Redis..."
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Создание директорий
echo "📁 Создание директорий..."
mkdir -p data/logs
mkdir -p data/models
mkdir -p data/backtests
mkdir -p data/reports
mkdir -p logs

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
    "url": "postgresql://forexbot:forexbot_password@localhost:5432/forexbot",
    "backup_interval": 24,
    "retention_days": 365
  },
  "redis": {
    "host": "localhost",
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

# Создание systemd сервиса
echo "🔧 Создание systemd сервиса..."
sudo tee /etc/systemd/system/forexbot.service > /dev/null << EOF
[Unit]
Description=ForexBot AI Trading System
After=network.target postgresql.service redis-server.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/python integrated_bot_advanced.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Создание supervisor конфигурации
echo "🔧 Создание supervisor конфигурации..."
sudo tee /etc/supervisor/conf.d/forexbot.conf > /dev/null << EOF
[program:forexbot]
command=$(pwd)/venv/bin/python integrated_bot_advanced.py
directory=$(pwd)
user=$USER
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=$(pwd)/logs/forexbot.log
environment=PATH="$(pwd)/venv/bin"
EOF

# Настройка Nginx
echo "🌐 Настройка Nginx..."
sudo tee /etc/nginx/sites-available/forexbot > /dev/null << EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket поддержка
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location /static/ {
        alias $(pwd)/web_interface/frontend/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
EOF

# Активация Nginx конфигурации
sudo ln -sf /etc/nginx/sites-available/forexbot /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Перезапуск сервисов
echo "🔄 Перезапуск сервисов..."
sudo systemctl daemon-reload
sudo systemctl enable forexbot
sudo systemctl start forexbot
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start forexbot

echo "✅ Ручное развертывание завершено!"
echo ""
echo "🌐 Доступные сервисы:"
echo "   - ForexBot AI: http://localhost:8000"
echo "   - Nginx: http://localhost"
echo ""
echo "📊 Управление сервисом:"
echo "   - sudo systemctl status forexbot"
echo "   - sudo systemctl restart forexbot"
echo "   - sudo supervisorctl status forexbot"
echo ""
echo "📋 Логи:"
echo "   - tail -f logs/forexbot.log"
echo "   - sudo journalctl -u forexbot -f"