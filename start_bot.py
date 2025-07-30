#!/usr/bin/env python3
"""
ForexBot AI Trading System - Quick Start
Простой скрипт для запуска интегрированного бота
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Проверка зависимостей"""
    print("🔍 Проверка зависимостей...")
    
    required_packages = [
        "fastapi",
        "uvicorn",
        "loguru",
        "pandas",
        "numpy"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Отсутствуют пакеты: {', '.join(missing_packages)}")
        print("📦 Установка зависимостей...")
        subprocess.run([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("✅ Зависимости установлены")
    else:
        print("✅ Все зависимости найдены")

def create_directories():
    """Создание необходимых директорий"""
    print("📁 Создание директорий...")
    
    directories = [
        "data/logs",
        "web_interface/frontend"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Директории созданы")

def main():
    """Главная функция"""
    print("=" * 50)
    print("🤖 ForexBot AI Trading System")
    print("=" * 50)
    
    # Проверка зависимостей
    check_dependencies()
    
    # Создание директорий
    create_directories()
    
    # Проверка конфигурации
    if not Path("config.json").exists():
        print("⚠️  Файл config.json не найден")
        print("📝 Создание базовой конфигурации...")
        
        basic_config = {
            "mt5": {
                "server": "",
                "login": 0,
                "password": "",
                "timeout": 5000,
                "symbols": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
            },
            "ai": {
                "models": ["lstm", "xgboost", "lightgbm"],
                "timeframes": ["M15", "H1", "H4"],
                "lookback_periods": [50, 100, 200],
                "retrain_interval": 24
            },
            "strategies": {
                "trend_following": {"enabled": True, "weight": 0.4},
                "mean_reversion": {"enabled": True, "weight": 0.3},
                "breakout": {"enabled": True, "weight": 0.3}
            },
            "risk": {
                "max_risk_per_trade": 0.02,
                "max_daily_loss": 0.05,
                "max_concurrent_trades": 5,
                "stop_loss_pips": 50,
                "take_profit_pips": 100
            },
            "solana": {
                "token_address": "",
                "burn_percentage": 0.1,
                "min_profit_for_burn": 100
            },
            "web_interface": {
                "host": "0.0.0.0",
                "port": 8000
            }
        }
        
        import json
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(basic_config, f, indent=2, ensure_ascii=False)
        
        print("✅ Базовая конфигурация создана")
        print("📝 Отредактируйте config.json для настройки бота")
    
    print("\n🚀 Запуск интегрированного бота...")
    print("🌐 Веб-интерфейс будет доступен по адресу: http://localhost:8000")
    print("⏹️  Нажмите Ctrl+C для остановки")
    print("-" * 50)
    
    # Запуск интегрированного бота
    try:
        subprocess.run([
            sys.executable, 
            "integrated_bot.py",
            "--mode", "demo",
            "--port", "8000"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Остановка бота...")
    except Exception as e:
        print(f"❌ Ошибка запуска: {e}")

if __name__ == "__main__":
    main()