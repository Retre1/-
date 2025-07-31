#!/usr/bin/env python3
"""
Example: Single Pair Model System
Пример использования системы "одна пара - одна модель"
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from single_pair_model import SinglePairModelManager, SinglePairModel

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def example_single_pair_training():
    """Пример обучения моделей для отдельных пар"""
    
    print("🎯 Запуск примера системы 'Одна пара - одна модель'")
    
    # Конфигурация
    config = {
        "ai": {
            "models": ["lstm", "xgboost", "lightgbm"],
            "timeframes": ["H1", "H4"],
            "min_accuracy_threshold": 0.65
        }
    }
    
    # Создание менеджера
    manager = SinglePairModelManager(config)
    
    # Определение пар для обучения
    pairs = [
        ("EURUSD", "H1"),
        ("GBPUSD", "H1"),
        ("USDJPY", "H1"),
        ("EURUSD", "H4"),
        ("GBPUSD", "H4")
    ]
    
    print(f"\n📚 Обучение моделей для {len(pairs)} пар...")
    
    # Обучение всех моделей
    results = manager.train_all_models(
        pairs=pairs,
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    # Анализ результатов
    print("\n📊 Результаты обучения:")
    successful_models = 0
    
    for pair_key, result in results.items():
        if 'error' not in result:
            successful_models += 1
            validation = result.get('validation_results', {})
            backtest = result.get('backtest_results', {})
            
            print(f"\n✅ {pair_key}:")
            print(f"   Точность: {validation.get('ensemble_accuracy', 0):.4f}")
            print(f"   Сделки: {backtest.get('total_trades', 0)}")
            print(f"   Винрейт: {backtest.get('win_rate', 0):.2f}%")
            print(f"   Прибыль: {backtest.get('total_profit', 0):.2f}")
            print(f"   Sharpe Ratio: {backtest.get('sharpe_ratio', 0):.4f}")
        else:
            print(f"\n❌ {pair_key}: Ошибка - {result['error']}")
    
    print(f"\n📈 Успешно обучено: {successful_models}/{len(pairs)} моделей")
    
    # Получение статуса всех моделей
    status = manager.get_model_status()
    print(f"\n📋 Статус моделей:")
    for pair_key, info in status.items():
        status_text = "✅ Обучена" if info['trained'] else "❌ Не обучена"
        accuracy_text = f"Точность: {info['accuracy']:.4f}" if info['trained'] else "Точность: N/A"
        print(f"   {pair_key}: {status_text} | {accuracy_text}")
    
    return manager

async def example_prediction_workflow(manager: SinglePairModelManager):
    """Пример workflow предсказаний"""
    
    print("\n🔮 Тестирование предсказаний...")
    
    # Создание тестовых данных для разных пар
    import numpy as np
    import pandas as pd
    
    test_data = {}
    
    for symbol in ["EURUSD", "GBPUSD", "USDJPY"]:
        for timeframe in ["H1", "H4"]:
            # Создание тестовых данных для каждой пары
            dates = pd.date_range('2023-12-01', '2023-12-31', freq='1H')
            np.random.seed(42)
            
            # Разные базовые цены для разных пар
            if symbol == 'EURUSD':
                base_price = 1.0850
            elif symbol == 'GBPUSD':
                base_price = 1.2650
            elif symbol == 'USDJPY':
                base_price = 150.0
            else:
                base_price = 1.0
            
            price = base_price + np.cumsum(np.random.randn(len(dates)) * 0.001)
            volume = np.random.randint(1000, 10000, len(dates))
            
            df = pd.DataFrame({
                'open': price * (1 + np.random.randn(len(dates)) * 0.0005),
                'high': price * (1 + abs(np.random.randn(len(dates)) * 0.001)),
                'low': price * (1 - abs(np.random.randn(len(dates)) * 0.001)),
                'close': price,
                'volume': volume
            }, index=dates)
            
            test_data[f"{symbol}_{timeframe}"] = df
    
    # Получение предсказаний
    predictions = manager.predict_all(test_data)
    
    print("\n📊 Результаты предсказаний:")
    for pair_key, prediction in predictions.items():
        if 'error' not in prediction:
            print(f"\n✅ {pair_key}:")
            print(f"   Уверенность: {prediction.get('confidence', 0):.4f}")
            print(f"   Направление: {prediction.get('ensemble_prediction', [])}")
            print(f"   Модель: {prediction.get('model_info', {}).get('symbol', 'N/A')}")
        else:
            print(f"\n❌ {pair_key}: Ошибка - {prediction['error']}")

async def example_model_management(manager: SinglePairModelManager):
    """Пример управления моделями"""
    
    print("\n🗄️ Управление моделями...")
    
    # Получение информации о конкретной модели
    eurusd_h1_model = manager.create_model("EURUSD", "H1")
    model_info = eurusd_h1_model.get_model_info()
    
    print(f"\n📋 Информация о модели EURUSD H1:")
    print(f"   Статус: {'Обучена' if model_info['trained'] else 'Не обучена'}")
    print(f"   Точность: {model_info['accuracy']:.4f}")
    print(f"   Последнее обучение: {model_info['last_trained'] or 'Нет'}")
    
    if model_info['trained']:
        backtest = model_info.get('backtest_results', {})
        print(f"   Backtesting результаты:")
        print(f"     Сделки: {backtest.get('total_trades', 0)}")
        print(f"     Винрейт: {backtest.get('win_rate', 0):.2f}%")
        print(f"     Прибыль: {backtest.get('total_profit', 0):.2f}")
    
    # Переобучение модели
    print(f"\n🔄 Переобучение модели EURUSD H1...")
    retrain_result = eurusd_h1_model.retrain_model(days_back=180)
    
    if 'error' not in retrain_result:
        print("✅ Переобучение завершено успешно!")
        
        # Обновленная информация
        updated_info = eurusd_h1_model.get_model_info()
        print(f"   Новая точность: {updated_info['accuracy']:.4f}")
        print(f"   Время обучения: {updated_info['last_trained']}")
    else:
        print(f"❌ Ошибка переобучения: {retrain_result['error']}")

async def example_performance_comparison(manager: SinglePairModelManager):
    """Пример сравнения производительности моделей"""
    
    print("\n📈 Сравнение производительности моделей...")
    
    # Получение статуса всех моделей
    status = manager.get_model_status()
    
    # Анализ по парам
    pair_performance = {}
    timeframe_performance = {}
    
    for pair_key, info in status.items():
        if info['trained']:
            symbol, timeframe = pair_key.split('_')
            
            # Группировка по парам
            if symbol not in pair_performance:
                pair_performance[symbol] = []
            pair_performance[symbol].append({
                'timeframe': timeframe,
                'accuracy': info['accuracy'],
                'backtest': info.get('backtest_results', {})
            })
            
            # Группировка по таймфреймам
            if timeframe not in timeframe_performance:
                timeframe_performance[timeframe] = []
            timeframe_performance[timeframe].append({
                'symbol': symbol,
                'accuracy': info['accuracy'],
                'backtest': info.get('backtest_results', {})
            })
    
    # Анализ по парам
    print("\n📊 Производительность по парам:")
    for symbol, models in pair_performance.items():
        avg_accuracy = sum(m['accuracy'] for m in models) / len(models)
        total_trades = sum(m['backtest'].get('total_trades', 0) for m in models)
        avg_win_rate = sum(m['backtest'].get('win_rate', 0) for m in models) / len(models)
        
        print(f"\n   {symbol}:")
        print(f"     Средняя точность: {avg_accuracy:.4f}")
        print(f"     Общее количество сделок: {total_trades}")
        print(f"     Средний винрейт: {avg_win_rate:.2f}%")
        
        for model in models:
            print(f"       {model['timeframe']}: {model['accuracy']:.4f} точность")
    
    # Анализ по таймфреймам
    print("\n📊 Производительность по таймфреймам:")
    for timeframe, models in timeframe_performance.items():
        avg_accuracy = sum(m['accuracy'] for m in models) / len(models)
        total_trades = sum(m['backtest'].get('total_trades', 0) for m in models)
        avg_win_rate = sum(m['backtest'].get('win_rate', 0) for m in models) / len(models)
        
        print(f"\n   {timeframe}:")
        print(f"     Средняя точность: {avg_accuracy:.4f}")
        print(f"     Общее количество сделок: {total_trades}")
        print(f"     Средний винрейт: {avg_win_rate:.2f}%")
        
        for model in models:
            print(f"       {model['symbol']}: {model['accuracy']:.4f} точность")

async def example_automated_trading_simulation(manager: SinglePairModelManager):
    """Пример симуляции автоматической торговли"""
    
    print("\n🤖 Симуляция автоматической торговли...")
    
    # Создание торгового симулятора
    class TradingSimulator:
        def __init__(self, initial_capital=10000):
            self.capital = initial_capital
            self.positions = {}
            self.trades = []
            self.equity_curve = []
        
        def execute_trade(self, symbol, direction, price, confidence):
            """Выполнение торговой операции"""
            
            if direction == 'HOLD':
                return
            
            position_size = self.capital * 0.1  # 10% капитала на сделку
            
            if direction == 'BUY':
                if symbol in self.positions:
                    # Закрытие существующей позиции
                    old_position = self.positions[symbol]
                    profit = (price - old_position['price']) * old_position['size']
                    self.capital += profit
                    
                    self.trades.append({
                        'symbol': symbol,
                        'direction': 'SELL',
                        'entry_price': old_position['price'],
                        'exit_price': price,
                        'profit': profit,
                        'confidence': confidence,
                        'timestamp': datetime.now().isoformat()
                    })
                
                # Открытие новой позиции
                self.positions[symbol] = {
                    'direction': 'BUY',
                    'price': price,
                    'size': position_size / price,
                    'confidence': confidence
                }
                
            elif direction == 'SELL':
                if symbol in self.positions:
                    # Закрытие существующей позиции
                    old_position = self.positions[symbol]
                    profit = (old_position['price'] - price) * old_position['size']
                    self.capital += profit
                    
                    self.trades.append({
                        'symbol': symbol,
                        'direction': 'BUY',
                        'entry_price': old_position['price'],
                        'exit_price': price,
                        'profit': profit,
                        'confidence': confidence,
                        'timestamp': datetime.now().isoformat()
                    })
                
                # Открытие новой позиции
                self.positions[symbol] = {
                    'direction': 'SELL',
                    'price': price,
                    'size': position_size / price,
                    'confidence': confidence
                }
            
            self.equity_curve.append(self.capital)
        
        def get_statistics(self):
            """Получение торговой статистики"""
            if not self.trades:
                return {}
            
            profitable_trades = len([t for t in self.trades if t['profit'] > 0])
            total_profit = sum(t['profit'] for t in self.trades)
            
            return {
                'total_trades': len(self.trades),
                'profitable_trades': profitable_trades,
                'win_rate': (profitable_trades / len(self.trades)) * 100,
                'total_profit': total_profit,
                'final_capital': self.capital,
                'return_percent': ((self.capital - 10000) / 10000) * 100
            }
    
    # Создание симулятора
    simulator = TradingSimulator()
    
    # Симуляция торговли на основе предсказаний моделей
    print("📈 Симуляция торговли на основе предсказаний...")
    
    # Создание тестовых данных для симуляции
    import numpy as np
    import pandas as pd
    
    for symbol in ["EURUSD", "GBPUSD"]:
        for timeframe in ["H1"]:
            # Создание 100 точек данных для симуляции
            dates = pd.date_range('2023-12-01', '2023-12-31', freq='1H')
            np.random.seed(42)
            
            base_price = 1.0850 if symbol == 'EURUSD' else 1.2650
            price = base_price + np.cumsum(np.random.randn(len(dates)) * 0.001)
            
            df = pd.DataFrame({
                'open': price * (1 + np.random.randn(len(dates)) * 0.0005),
                'high': price * (1 + abs(np.random.randn(len(dates)) * 0.001)),
                'low': price * (1 - abs(np.random.randn(len(dates)) * 0.001)),
                'close': price,
                'volume': np.random.randint(1000, 10000, len(dates))
            }, index=dates)
            
            # Получение предсказания
            model = manager.create_model(symbol, timeframe)
            prediction = model.predict(df)
            
            if 'error' not in prediction:
                # Определение направления
                ensemble_pred = prediction.get('ensemble_prediction', [])
                if len(ensemble_pred) > 0:
                    direction = ['HOLD', 'BUY', 'SELL'][np.argmax(ensemble_pred[-1])]
                    confidence = prediction.get('confidence', 0.5)
                    current_price = df['close'].iloc[-1]
                    
                    # Выполнение торговой операции
                    simulator.execute_trade(symbol, direction, current_price, confidence)
                    
                    print(f"   {symbol} {timeframe}: {direction} @ {current_price:.5f} (уверенность: {confidence:.4f})")
    
    # Получение результатов симуляции
    stats = simulator.get_statistics()
    
    print(f"\n📊 Результаты симуляции:")
    print(f"   Общее количество сделок: {stats['total_trades']}")
    print(f"   Прибыльные сделки: {stats['profitable_trades']}")
    print(f"   Винрейт: {stats['win_rate']:.2f}%")
    print(f"   Общая прибыль: {stats['total_profit']:.2f}")
    print(f"   Финальный капитал: {stats['final_capital']:.2f}")
    print(f"   Доходность: {stats['return_percent']:.2f}%")

if __name__ == "__main__":
    # Запуск всех примеров
    async def main():
        print("🚀 Запуск примеров системы 'Одна пара - одна модель'")
        
        # Обучение моделей
        manager = await example_single_pair_training()
        
        # Тестирование предсказаний
        await example_prediction_workflow(manager)
        
        # Управление моделями
        await example_model_management(manager)
        
        # Сравнение производительности
        await example_performance_comparison(manager)
        
        # Симуляция торговли
        await example_automated_trading_simulation(manager)
        
        print("\n🎉 Все примеры выполнены успешно!")
        print("📚 Система 'Одна пара - одна модель' готова к использованию!")
    
    asyncio.run(main())