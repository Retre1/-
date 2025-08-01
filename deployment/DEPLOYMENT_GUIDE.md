# 🚀 **Руководство по развертыванию ForexBot AI**

## 📋 **Содержание**

1. [Подготовка сервера](#подготовка-сервера)
2. [Docker развертывание](#docker-развертывание)
3. [Ручное развертывание](#ручное-развертывание)
4. [Настройка домена и SSL](#настройка-домена-и-ssl)
5. [Мониторинг и логи](#мониторинг-и-логи)
6. [Резервное копирование](#резервное-копирование)
7. [Обновление системы](#обновление-системы)
8. [Устранение неполадок](#устранение-неполадок)

---

## 🖥️ **Подготовка сервера**

### **Системные требования:**

| Компонент | Минимальные | Рекомендуемые |
|-----------|-------------|---------------|
| **ОС** | Ubuntu 20.04+ | Ubuntu 22.04 LTS |
| **CPU** | 2 ядра | 4+ ядра |
| **RAM** | 4 GB | 8+ GB |
| **Диск** | 20 GB | 50+ GB SSD |
| **Сеть** | Стабильное подключение | Высокоскоростное |

### **Быстрая настройка сервера:**

```bash
# Клонирование проекта
git clone https://github.com/your-repo/forexbot-ai.git
cd forexbot-ai

# Настройка сервера
chmod +x deployment/server_setup.sh
./deployment/server_setup.sh
```

---

## 🐳 **Docker развертывание (рекомендуемый)**

### **Преимущества Docker:**
- ✅ Изоляция зависимостей
- ✅ Простота развертывания
- ✅ Масштабируемость
- ✅ Воспроизводимость

### **Быстрое развертывание:**

```bash
# Переход в директорию развертывания
cd deployment

# Запуск автоматического развертывания
chmod +x deploy.sh
./deploy.sh
```

### **Ручное развертывание Docker:**

```bash
# Создание директорий
mkdir -p data/{logs,models,backtests,reports}
mkdir -p logs ssl grafana/{dashboards,datasources}

# Создание конфигурации
cp config.json.example config.json
# Редактирование config.json

# Сборка и запуск
docker-compose build
docker-compose up -d

# Проверка статуса
docker-compose ps
docker-compose logs -f forexbot
```

### **Управление Docker контейнерами:**

```bash
# Просмотр логов
docker-compose logs forexbot
docker-compose logs -f forexbot

# Перезапуск сервиса
docker-compose restart forexbot

# Остановка всех сервисов
docker-compose down

# Обновление и перезапуск
docker-compose pull
docker-compose up -d --build
```

---

## 🔧 **Ручное развертывание (без Docker)**

### **Установка зависимостей:**

```bash
# Системные зависимости
sudo apt update
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    postgresql \
    postgresql-contrib \
    redis-server \
    nginx \
    supervisor \
    build-essential \
    libpq-dev

# Python зависимости
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### **Настройка базы данных:**

```bash
# Настройка PostgreSQL
sudo systemctl enable postgresql
sudo systemctl start postgresql

# Создание пользователя и БД
sudo -u postgres psql << EOF
CREATE USER forexbot WITH PASSWORD 'forexbot_password';
CREATE DATABASE forexbot OWNER forexbot;
GRANT ALL PRIVILEGES ON DATABASE forexbot TO forexbot;
\q
EOF
```

### **Настройка Redis:**

```bash
# Настройка Redis
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Проверка статуса
redis-cli ping
```

### **Автоматическое развертывание:**

```bash
# Запуск скрипта ручного развертывания
chmod +x deployment/manual_deploy.sh
./deployment/manual_deploy.sh
```

---

## 🌐 **Настройка домена и SSL**

### **Настройка домена:**

1. **Покупка домена** (например, forexbot.yourdomain.com)
2. **Настройка DNS** - указание A-записи на IP сервера
3. **Обновление конфигурации Nginx**

### **SSL сертификаты с Let's Encrypt:**

```bash
# Установка Certbot
sudo apt install certbot python3-certbot-nginx

# Получение SSL сертификата
sudo certbot --nginx -d forexbot.yourdomain.com

# Автоматическое обновление
sudo crontab -e
# Добавить строку: 0 12 * * * /usr/bin/certbot renew --quiet
```

### **Обновление Nginx конфигурации:**

```nginx
server {
    listen 80;
    server_name forexbot.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name forexbot.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/forexbot.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/forexbot.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## 📊 **Мониторинг и логи**

### **Системный мониторинг:**

```bash
# Установка htop для мониторинга ресурсов
sudo apt install htop
htop

# Мониторинг диска
df -h
du -sh /opt/forexbot/*

# Мониторинг памяти
free -h
```

### **Логи приложения:**

```bash
# Docker логи
docker-compose logs -f forexbot

# Systemd логи (ручное развертывание)
sudo journalctl -u forexbot -f

# Supervisor логи
sudo supervisorctl status forexbot
tail -f logs/forexbot.log
```

### **Мониторинг базы данных:**

```bash
# Подключение к PostgreSQL
psql -h localhost -U forexbot -d forexbot

# Проверка размера БД
SELECT pg_size_pretty(pg_database_size('forexbot'));

# Проверка активных подключений
SELECT * FROM pg_stat_activity;
```

### **Grafana дашборды:**

1. Откройте http://your-domain.com:3000
2. Логин: admin/admin
3. Настройте источники данных (Prometheus)
4. Импортируйте готовые дашборды

---

## 💾 **Резервное копирование**

### **Автоматическое резервное копирование:**

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/forexbot"

# Создание резервной копии БД
pg_dump -h localhost -U forexbot forexbot > $BACKUP_DIR/db_$DATE.sql

# Создание резервной копии данных
tar -czf $BACKUP_DIR/data_$DATE.tar.gz data/

# Создание резервной копии конфигурации
tar -czf $BACKUP_DIR/config_$DATE.tar.gz config.json logs/

# Удаление старых резервных копий (старше 30 дней)
find $BACKUP_DIR -name "*.sql" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

### **Настройка cron для автоматического резервного копирования:**

```bash
# Редактирование crontab
crontab -e

# Добавить строку для ежедневного резервного копирования в 2:00
0 2 * * * /path/to/backup.sh
```

---

## 🔄 **Обновление системы**

### **Обновление с Docker:**

```bash
# Остановка сервисов
docker-compose down

# Обновление кода
git pull origin main

# Пересборка и запуск
docker-compose build --no-cache
docker-compose up -d

# Проверка статуса
docker-compose ps
docker-compose logs -f forexbot
```

### **Обновление ручного развертывания:**

```bash
# Остановка сервиса
sudo systemctl stop forexbot

# Обновление кода
git pull origin main

# Обновление зависимостей
source venv/bin/activate
pip install -r requirements.txt

# Запуск сервиса
sudo systemctl start forexbot

# Проверка статуса
sudo systemctl status forexbot
```

---

## 🔧 **Устранение неполадок**

### **Частые проблемы:**

#### **1. Проблемы с подключением к БД:**
```bash
# Проверка статуса PostgreSQL
sudo systemctl status postgresql

# Проверка подключения
psql -h localhost -U forexbot -d forexbot -c "SELECT 1;"
```

#### **2. Проблемы с Redis:**
```bash
# Проверка статуса Redis
sudo systemctl status redis-server

# Проверка подключения
redis-cli ping
```

#### **3. Проблемы с портами:**
```bash
# Проверка занятых портов
sudo netstat -tlnp | grep :8000
sudo lsof -i :8000
```

#### **4. Проблемы с правами доступа:**
```bash
# Проверка прав на директории
ls -la data/
sudo chown -R $USER:$USER data/
```

#### **5. Проблемы с логами:**
```bash
# Просмотр последних ошибок
tail -f logs/forexbot.log | grep ERROR

# Проверка размера логов
du -sh logs/
```

### **Команды диагностики:**

```bash
# Проверка системных ресурсов
htop
df -h
free -h

# Проверка сетевых подключений
netstat -tlnp
ss -tlnp

# Проверка процессов
ps aux | grep python
ps aux | grep forexbot

# Проверка логов системы
sudo journalctl -f
```

---

## 📞 **Поддержка**

### **Полезные команды:**

```bash
# Статус всех сервисов
docker-compose ps
sudo systemctl status forexbot postgresql redis-server nginx

# Логи в реальном времени
docker-compose logs -f
tail -f logs/forexbot.log

# Перезапуск всех сервисов
docker-compose restart
sudo systemctl restart forexbot postgresql redis-server nginx
```

### **Контакты для поддержки:**
- 📧 Email: support@forexbot.ai
- 💬 Telegram: @forexbot_support
- 📖 Документация: https://docs.forexbot.ai

---

## ✅ **Проверка развертывания**

После завершения развертывания проверьте:

1. ✅ Веб-интерфейс доступен: http://your-domain.com
2. ✅ API работает: http://your-domain.com/api/status
3. ✅ База данных подключена
4. ✅ Redis работает
5. ✅ Логи записываются
6. ✅ Мониторинг активен

**🎉 Поздравляем! ForexBot AI успешно развернут на сервере!**