# Установка и настройка ForexBot AI Trading System

## Содержание

1. [Требования к системе](#требования-к-системе)
2. [Установка зависимостей](#установка-зависимостей)
3. [Настройка MetaTrader 5](#настройка-metatrader-5)
4. [Настройка Solana](#настройка-solana)
5. [Конфигурация бота](#конфигурация-бота)
6. [Запуск системы](#запуск-системы)
7. [Веб-интерфейс](#веб-интерфейс)
8. [Мониторинг и логи](#мониторинг-и-логи)

## Требования к системе

### Минимальные требования:
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 10.15+
- **CPU**: 4 ядра, 2.5 GHz
- **RAM**: 8 GB
- **Storage**: 20 GB свободного места
- **Internet**: Стабильное соединение

### Рекомендуемые требования:
- **OS**: Windows 11, Ubuntu 22.04+
- **CPU**: 8 ядер, 3.0 GHz
- **RAM**: 16 GB
- **Storage**: 50 GB SSD
- **GPU**: NVIDIA GTX 1060+ (для ускорения ML)

### Программное обеспечение:
- Python 3.9+
- MetaTrader 5
- PostgreSQL 13+
- Node.js 16+ (для веб-интерфейса)
- Git

## Установка зависимостей

### 1. Клонирование репозитория

```bash
git clone https://github.com/your-repo/forex-bot-solana.git
cd forex-bot-solana
```

### 2. Создание виртуального окружения

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Установка Python зависимостей

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Установка TA-Lib (технические индикаторы)

#### Windows:
```bash
# Скачайте wheel файл с https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.28-cp39-cp39-win_amd64.whl
```

#### Linux:
```bash
sudo apt-get install build-essential wget
wget https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib
```

#### macOS:
```bash
brew install ta-lib
pip install TA-Lib
```

## Настройка MetaTrader 5

### 1. Установка MT5

1. Скачайте MetaTrader 5 с официального сайта брокера
2. Установите и настройте подключение к торговому счету
3. Убедитесь, что MT5 может подключаться к Python

### 2. Настройка API

1. Откройте MT5
2. Перейдите в **Tools** → **Options** → **Expert Advisors**
3. Включите опции:
   - ✅ Allow automated trading
   - ✅ Allow DLL imports
   - ✅ Allow import of external experts

### 3. Проверка подключения

```python
import MetaTrader5 as mt5

# Тест подключения
if mt5.initialize():
    print("MT5 инициализирован успешно")
    account_info = mt5.account_info()
    if account_info:
        print(f"Баланс: {account_info.balance}")
    mt5.shutdown()
else:
    print("Ошибка инициализации MT5")
```

## Настройка Solana

### 1. Установка Solana CLI

```bash
# Linux/macOS
sh -c "$(curl -sSfL https://release.solana.com/v1.17.0/install)"

# Windows (PowerShell)
cmd /c "curl https://release.solana.com/v1.17.0/solana-install-init-x86_64-pc-windows-msvc.exe --output C:\solana-install-tmp\solana-install-init.exe --create-dirs"
```

### 2. Создание кошелька

```bash
# Создание нового кошелька
solana-keygen new --outfile ~/.config/solana/id.json

# Или восстановление из seed фразы
solana-keygen recover --outfile ~/.config/solana/id.json
```

### 3. Настройка RPC

```bash
# Mainnet
solana config set --url https://api.mainnet-beta.solana.com

# Devnet (для тестирования)
solana config set --url https://api.devnet.solana.com
```

### 4. Пополнение кошелька

Для работы с токенами необходимо иметь SOL для оплаты комиссий:

```bash
# Проверка баланса
solana balance

# Для devnet можно получить тестовые SOL
solana airdrop 1
```

## Конфигурация бота

### 1. Основной конфиг

Скопируйте и отредактируйте `config.json`:

```bash
cp config.json.example config.json
```

### 2. MT5 настройки

```json
{
  "mt5": {
    "server": "YourBroker-Server",
    "login": 12345,
    "password": "your_password",
    "symbols": ["EURUSD", "GBPUSD", "USDJPY"]
  }
}
```

### 3. Solana настройки

```json
{
  "solana": {
    "rpc_endpoint": "https://api.mainnet-beta.solana.com",
    "token_address": "",
    "burn_percentage": 0.1,
    "min_profit_for_burn": 100
  }
}
```

### 4. AI настройки

```json
{
  "ai": {
    "models": ["lstm", "xgboost", "lightgbm"],
    "retrain_interval": 24,
    "ensemble_weights": {
      "lstm": 0.4,
      "xgboost": 0.3,
      "lightgbm": 0.3
    }
  }
}
```

## Запуск системы

### 1. Создание токена (первый запуск)

```python
from solana_integration.token_manager import SolanaTokenManager
import asyncio

async def create_token():
    config = {
        "rpc_endpoint": "https://api.devnet.solana.com",
        "token_name": "ForexBot Token",
        "token_symbol": "FBT",
        "initial_supply": 1000000
    }
    
    token_manager = SolanaTokenManager(config)
    await token_manager.initialize()
    
    print(f"Токен создан: {token_manager.token_address}")

asyncio.run(create_token())
```

### 2. Запуск торгового бота

```bash
python trading_bot/main.py
```

### 3. Запуск веб-интерфейса

```bash
cd web_interface/backend
python main.py
```

Веб-интерфейс будет доступен по адресу: http://localhost:8000

### 4. Режим демо (без реальной торговли)

```bash
# Установите в config.json
{
  "demo_mode": true,
  "mt5": {
    "demo_account": true
  }
}
```

## Веб-интерфейс

### Основные разделы:

1. **Dashboard** - общая статистика и статус бота
2. **Trading** - активные позиции и сигналы
3. **AI Models** - статус и производительность моделей
4. **Token** - информация о токене и сжигание
5. **Settings** - настройки и конфигурация

### API эндпоинты:

- `GET /api/status` - статус бота
- `POST /api/control` - управление ботом
- `GET /api/predictions` - прогнозы ИИ
- `GET /api/positions` - открытые позиции
- `GET /api/token/info` - информация о токене
- `POST /api/token/burn` - сжигание токенов

## Мониторинг и логи

### 1. Настройка логирования

```python
from loguru import logger

# Настройка в config.json
{
  "logging": {
    "level": "INFO",
    "file_path": "data/logs/forexbot.log",
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
  }
}
```

### 2. Мониторинг производительности

```bash
# Просмотр логов
tail -f data/logs/forexbot.log

# Мониторинг ресурсов
htop

# Проверка процесса
ps aux | grep python
```

### 3. Системный мониторинг

Создайте systemd сервис для автозапуска:

```bash
# /etc/systemd/system/forexbot.service
[Unit]
Description=ForexBot AI Trading System
After=network.target

[Service]
Type=simple
User=forexbot
WorkingDirectory=/path/to/forex-bot-solana
ExecStart=/path/to/venv/bin/python trading_bot/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable forexbot
sudo systemctl start forexbot
sudo systemctl status forexbot
```

## Безопасность

### 1. Защита приватных ключей

```bash
chmod 600 data/solana_wallet.json
chmod 600 config.json
```

### 2. Настройка firewall

```bash
# Ubuntu
sudo ufw allow 8000  # веб-интерфейс
sudo ufw enable
```

### 3. SSL сертификат

Для продакшена настройте HTTPS:

```bash
# Let's Encrypt
sudo certbot --nginx -d yourdomain.com
```

## Резервное копирование

### 1. Автоматический бэкап

```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf "backup_$DATE.tar.gz" data/ config.json
aws s3 cp "backup_$DATE.tar.gz" s3://your-bucket/backups/
```

### 2. Cron задача

```bash
# Каждый день в 2:00
0 2 * * * /path/to/backup.sh
```

## Устранение неполадок

### Частые проблемы:

1. **MT5 не подключается**
   - Проверьте настройки Expert Advisors
   - Убедитесь, что MT5 запущен под администратором

2. **Ошибки Solana RPC**
   - Проверьте интернет-соединение
   - Попробуйте другой RPC endpoint

3. **Недостаточно памяти для AI**
   - Уменьшите количество моделей
   - Используйте batch_size поменьше

4. **Веб-интерфейс недоступен**
   - Проверьте порт 8000
   - Посмотрите логи FastAPI

### Логи и диагностика:

```bash
# Проверка логов
tail -f data/logs/forexbot.log

# Проверка состояния
python -c "from trading_bot.main import ForexTradingBot; bot = ForexTradingBot(); print('OK')"

# Тест MT5
python -c "import MetaTrader5 as mt5; print(mt5.initialize())"

# Тест Solana
solana balance
```

## Обновление

```bash
git pull origin main
pip install -r requirements.txt --upgrade
# Перезапустите сервисы
sudo systemctl restart forexbot
```

---

**Поддержка**: support@forexbot-ai.com  
**Документация**: https://docs.forexbot-ai.com  
**Telegram**: @forexbot_support