#!/usr/bin/env python3
"""
Install Advanced Dependencies for ForexBot AI
Установка зависимостей для продвинутых AI моделей
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command: str, description: str) -> bool:
    """Выполнение команды с выводом"""
    print(f"\n🔄 {description}...")
    print(f"Команда: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} завершено успешно")
        if result.stdout:
            print(f"Вывод: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при {description.lower()}: {e}")
        if e.stderr:
            print(f"Ошибка: {e.stderr}")
        return False

def check_python_version():
    """Проверка версии Python"""
    print("🐍 Проверка версии Python...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Требуется Python 3.8+, текущая версия: {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    return True

def create_virtual_environment():
    """Создание виртуального окружения"""
    venv_path = Path("venv")
    if venv_path.exists():
        print("✅ Виртуальное окружение уже существует")
        return True
    
    return run_command("python3 -m venv venv", "Создание виртуального окружения")

def activate_venv_and_install():
    """Активация venv и установка зависимостей"""
    # Определение команды активации в зависимости от ОС
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Linux/Mac
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    # Обновление pip
    if not run_command(f"{pip_cmd} install --upgrade pip", "Обновление pip"):
        return False
    
    # Основные зависимости для FastAPI и веб-интерфейса
    basic_deps = [
        "fastapi",
        "uvicorn[standard]",
        "loguru",
        "pandas",
        "numpy",
        "requests",
        "websockets"
    ]
    
    for dep in basic_deps:
        if not run_command(f"{pip_cmd} install {dep}", f"Установка {dep}"):
            return False
    
    # Зависимости для машинного обучения
    ml_deps = [
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "joblib"
    ]
    
    for dep in ml_deps:
        if not run_command(f"{pip_cmd} install {dep}", f"Установка {dep}"):
            return False
    
    # TensorFlow для глубокого обучения
    if not run_command(f"{pip_cmd} install tensorflow", "Установка TensorFlow"):
        return False
    
    # Дополнительные зависимости для аналитики
    analytics_deps = [
        "matplotlib",
        "seaborn",
        "plotly"
    ]
    
    for dep in analytics_deps:
        if not run_command(f"{pip_cmd} install {dep}", f"Установка {dep}"):
            return False
    
    # Зависимости для безопасности
    security_deps = [
        "pyjwt",
        "bcrypt",
        "python-multipart"
    ]
    
    for dep in security_deps:
        if not run_command(f"{pip_cmd} install {dep}", f"Установка {dep}"):
            return False
    
    # Зависимости для базы данных
    db_deps = [
        "sqlalchemy",
        "alembic"
    ]
    
    for dep in db_deps:
        if not run_command(f"{pip_cmd} install {dep}", f"Установка {dep}"):
            return False
    
    return True

def create_directories():
    """Создание необходимых директорий"""
    directories = [
        "data/logs",
        "data/models",
        "web_interface/frontend",
        "models",
        "backtests",
        "reports"
    ]
    
    print("\n📁 Создание директорий...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Создана директория: {directory}")

def create_config_file():
    """Создание конфигурационного файла"""
    config = {
        "initial_capital": 10000,
        "max_risk_per_trade": 0.02,
        "max_positions": 5,
        "confidence_threshold": 0.6,
        "enable_ai_trading": True,
        "enable_risk_management": True,
        "ai_models": {
            "lstm": {
                "enabled": True,
                "sequence_length": 60,
                "n_features": 30
            },
            "transformer": {
                "enabled": True,
                "sequence_length": 60,
                "n_features": 30,
                "n_heads": 8
            },
            "xgboost": {
                "enabled": True,
                "n_estimators": 1000,
                "max_depth": 6,
                "learning_rate": 0.01
            },
            "lightgbm": {
                "enabled": True,
                "n_estimators": 1000,
                "max_depth": 6,
                "learning_rate": 0.01
            },
            "random_forest": {
                "enabled": True,
                "n_estimators": 500,
                "max_depth": 10
            },
            "gradient_boosting": {
                "enabled": True,
                "n_estimators": 500,
                "max_depth": 6
            }
        },
        "backtesting": {
            "initial_capital": 10000,
            "commission": 0.001,
            "confidence_threshold": 0.6
        },
        "notifications": {
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "recipient": []
            },
            "telegram": {
                "enabled": False,
                "bot_token": "",
                "chat_id": ""
            }
        },
        "database": {
            "enabled": True,
            "url": "sqlite:///forexbot.db"
        },
        "security": {
            "enabled": True,
            "jwt_secret": "",
            "password_hash_rounds": 12
        }
    }
    
    import json
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Создан файл конфигурации: config.json")

def create_startup_script():
    """Создание скрипта запуска"""
    if os.name == 'nt':  # Windows
        script_content = """@echo off
echo Starting Advanced ForexBot AI...
call venv\\Scripts\\activate
python integrated_bot_advanced.py
pause
"""
        with open("start_advanced_bot.bat", "w") as f:
            f.write(script_content)
        print("✅ Создан скрипт запуска: start_advanced_bot.bat")
    else:  # Linux/Mac
        script_content = """#!/bin/bash
echo "Starting Advanced ForexBot AI..."
source venv/bin/activate
python integrated_bot_advanced.py
"""
        with open("start_advanced_bot.sh", "w") as f:
            f.write(script_content)
        os.chmod("start_advanced_bot.sh", 0o755)
        print("✅ Создан скрипт запуска: start_advanced_bot.sh")

def test_installation():
    """Тестирование установки"""
    print("\n🧪 Тестирование установки...")
    
    test_script = """
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from fastapi import FastAPI
from loguru import logger

print("✅ Все основные зависимости установлены успешно!")
print(f"Python версия: {sys.version}")
print(f"Pandas версия: {pd.__version__}")
print(f"TensorFlow версия: {tf.__version__}")
print(f"XGBoost версия: {xgb.__version__}")
"""
    
    try:
        if os.name == 'nt':
            result = subprocess.run("venv\\Scripts\\python -c \"" + test_script.replace('\n', '; ') + "\"", 
                                 shell=True, check=True, capture_output=True, text=True)
        else:
            result = subprocess.run("venv/bin/python -c \"" + test_script.replace('\n', '; ') + "\"", 
                                 shell=True, check=True, capture_output=True, text=True)
        
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка тестирования: {e}")
        return False

def main():
    """Основная функция установки"""
    print("🚀 Установка зависимостей для Advanced ForexBot AI")
    print("=" * 50)
    
    # Проверка Python
    if not check_python_version():
        return False
    
    # Создание виртуального окружения
    if not create_virtual_environment():
        return False
    
    # Установка зависимостей
    if not activate_venv_and_install():
        return False
    
    # Создание директорий
    create_directories()
    
    # Создание конфигурации
    create_config_file()
    
    # Создание скрипта запуска
    create_startup_script()
    
    # Тестирование
    if not test_installation():
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Установка завершена успешно!")
    print("\n📋 Следующие шаги:")
    print("1. Активируйте виртуальное окружение:")
    if os.name == 'nt':
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Запустите бота:")
    if os.name == 'nt':
        print("   python integrated_bot_advanced.py")
    else:
        print("   python integrated_bot_advanced.py")
    print("3. Откройте веб-интерфейс: http://localhost:8000")
    print("\n📚 Документация:")
    print("- API документация: http://localhost:8000/docs")
    print("- Конфигурация: config.json")
    print("- Логи: data/logs/")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Установка завершена с ошибками")
        sys.exit(1)
    else:
        print("\n✅ Установка завершена успешно!")