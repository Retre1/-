# 🐳 Dockerfile для ForexBot AI
FROM python:3.9-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование файлов зависимостей
COPY requirements.txt .
COPY install_advanced_dependencies.py .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Создание необходимых директорий
RUN mkdir -p /app/data/logs \
    /app/data/models \
    /app/data/backtests \
    /app/data/reports \
    /app/web_interface/frontend

# Создание пользователя для безопасности
RUN useradd -m -u 1000 forexbot && \
    chown -R forexbot:forexbot /app

USER forexbot

# Открытие портов
EXPOSE 8000 8001

# Переменные окружения
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Команда запуска
CMD ["python", "integrated_bot_advanced.py"]