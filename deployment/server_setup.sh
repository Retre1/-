#!/bin/bash
# 🚀 Скрипт настройки сервера для ForexBot AI

set -e

echo "🚀 Настройка сервера для ForexBot AI..."

# Обновление системы
echo "📦 Обновление системы..."
sudo apt update && sudo apt upgrade -y

# Установка базовых пакетов
echo "📦 Установка базовых пакетов..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    unzip \
    build-essential \
    libpq-dev \
    redis-server \
    nginx \
    supervisor \
    htop \
    tmux \
    nano \
    vim

# Установка Docker (опционально)
echo "🐳 Установка Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Установка Docker Compose
echo "🐳 Установка Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Настройка Redis
echo "🔴 Настройка Redis..."
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Настройка Nginx
echo "🌐 Настройка Nginx..."
sudo systemctl enable nginx
sudo systemctl start nginx

# Создание пользователя для приложения
echo "👤 Создание пользователя forexbot..."
sudo useradd -m -s /bin/bash forexbot
sudo usermod -aG docker forexbot

# Создание директорий
echo "📁 Создание директорий..."
sudo mkdir -p /opt/forexbot
sudo mkdir -p /var/log/forexbot
sudo mkdir -p /etc/forexbot
sudo chown -R forexbot:forexbot /opt/forexbot
sudo chown -R forexbot:forexbot /var/log/forexbot
sudo chown -R forexbot:forexbot /etc/forexbot

echo "✅ Настройка сервера завершена!"
echo "🔧 Следующий шаг: клонирование проекта"