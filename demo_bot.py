#!/usr/bin/env python3
"""
ForexBot AI Trading System - Demo Script
Демонстрационный скрипт для показа работы интегрированного бота
"""

import requests
import json
import time
import sys
from datetime import datetime

# Базовый URL API
API_BASE = "http://localhost:8000"

def print_header():
    """Вывод заголовка"""
    print("=" * 60)
    print("🤖 ForexBot AI Trading System - Demo")
    print("=" * 60)
    print()

def check_bot_status():
    """Проверка статуса бота"""
    print("🔍 Проверка статуса бота...")
    try:
        response = requests.get(f"{API_BASE}/api/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Бот доступен")
            print(f"   Статус: {'🟢 Работает' if data['is_running'] else '🔴 Остановлен'}")
            print(f"   Позиции: {data['open_positions']}")
            print(f"   Баланс: ${data['balance']:.2f}")
            print(f"   Капитал: ${data['equity']:.2f}")
            return True
        else:
            print(f"❌ Ошибка подключения: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return False

def start_bot():
    """Запуск бота"""
    print("\n🚀 Запуск бота...")
    try:
        response = requests.post(
            f"{API_BASE}/api/control",
            json={"action": "start"}
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {data['message']}")
            return True
        else:
            print(f"❌ Ошибка запуска: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ошибка запуска: {e}")
        return False

def get_account_info():
    """Получение информации о счете"""
    print("\n💰 Информация о счете:")
    try:
        response = requests.get(f"{API_BASE}/api/account")
        if response.status_code == 200:
            data = response.json()
            print(f"   Баланс: ${data.get('balance', 0):.2f}")
            print(f"   Собственный капитал: ${data.get('equity', 0):.2f}")
            print(f"   Маржа: ${data.get('margin', 0):.2f}")
            print(f"   Свободная маржа: ${data.get('margin_free', 0):.2f}")
        else:
            print(f"❌ Ошибка получения данных: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка получения данных: {e}")

def get_token_info():
    """Получение информации о токенах"""
    print("\n🪙 Информация о токенах:")
    try:
        response = requests.get(f"{API_BASE}/api/token/info")
        if response.status_code == 200:
            data = response.json()
            print(f"   Название: {data.get('token_name', 'N/A')}")
            print(f"   Символ: {data.get('token_symbol', 'N/A')}")
            print(f"   Баланс: {data.get('token_balance', 0):.2f}")
        else:
            print(f"❌ Ошибка получения данных: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка получения данных: {e}")

def get_statistics():
    """Получение статистики"""
    print("\n📈 Статистика торговли:")
    try:
        response = requests.get(f"{API_BASE}/api/statistics")
        if response.status_code == 200:
            data = response.json()
            stats = data.get('statistics', {})
            print(f"   Всего сделок: {stats.get('total_trades', 0)}")
            print(f"   Прибыльные сделки: {stats.get('winning_trades', 0)}")
            print(f"   Убыточные сделки: {stats.get('losing_trades', 0)}")
            print(f"   Общая прибыль: ${stats.get('total_profit', 0):.2f}")
            print(f"   Просадка: {stats.get('current_drawdown', 0):.2f}%")
        else:
            print(f"❌ Ошибка получения данных: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка получения данных: {e}")

def get_predictions():
    """Получение предсказаний"""
    print("\n🎯 Торговые сигналы:")
    try:
        response = requests.get(f"{API_BASE}/api/predictions")
        if response.status_code == 200:
            data = response.json()
            signals = data.get('signals', [])
            if signals:
                for signal in signals:
                    direction_emoji = "📈" if signal['direction'] == 'BUY' else "📉" if signal['direction'] == 'SELL' else "➡️"
                    print(f"   {direction_emoji} {signal['symbol']}: {signal['direction']} (уверенность: {signal['confidence']*100:.1f}%)")
            else:
                print("   Нет активных сигналов")
        else:
            print(f"❌ Ошибка получения данных: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка получения данных: {e}")

def get_positions():
    """Получение позиций"""
    print("\n📋 Открытые позиции:")
    try:
        response = requests.get(f"{API_BASE}/api/positions")
        if response.status_code == 200:
            data = response.json()
            positions = data.get('positions', [])
            if positions:
                for pos in positions:
                    profit_emoji = "📈" if pos.get('profit', 0) >= 0 else "📉"
                    print(f"   {profit_emoji} {pos.get('symbol', 'N/A')}: {pos.get('type', 'N/A')} - ${pos.get('profit', 0):.2f}")
            else:
                print("   Нет открытых позиций")
        else:
            print(f"❌ Ошибка получения данных: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка получения данных: {e}")

def burn_tokens(amount):
    """Сжигание токенов"""
    print(f"\n🔥 Сжигание {amount} токенов...")
    try:
        response = requests.post(
            f"{API_BASE}/api/token/burn",
            json={"amount": amount, "reason": "Demo burn"}
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {data['message']}")
        else:
            print(f"❌ Ошибка сжигания: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка сжигания: {e}")

def monitor_bot():
    """Мониторинг бота в реальном времени"""
    print("\n🔄 Мониторинг бота (нажмите Ctrl+C для остановки):")
    print("-" * 60)
    
    try:
        while True:
            # Получение статуса
            response = requests.get(f"{API_BASE}/api/status")
            if response.status_code == 200:
                data = response.json()
                status_emoji = "🟢" if data['is_running'] else "🔴"
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] {status_emoji} Бот: {'Работает' if data['is_running'] else 'Остановлен'} | "
                      f"Позиции: {data['open_positions']} | "
                      f"Баланс: ${data['balance']:.2f}")
            
            time.sleep(5)  # Обновление каждые 5 секунд
            
    except KeyboardInterrupt:
        print("\n⏹️ Мониторинг остановлен")

def main():
    """Главная функция"""
    print_header()
    
    # Проверка доступности бота
    if not check_bot_status():
        print("❌ Бот недоступен. Убедитесь, что он запущен на http://localhost:8000")
        sys.exit(1)
    
    # Запуск бота
    if start_bot():
        time.sleep(2)  # Ждем запуска
        
        # Получение всей информации
        get_account_info()
        get_token_info()
        get_statistics()
        get_predictions()
        get_positions()
        
        # Демонстрация сжигания токенов
        burn_tokens(100)
        
        # Мониторинг
        monitor_bot()
    else:
        print("❌ Не удалось запустить бота")

if __name__ == "__main__":
    main()