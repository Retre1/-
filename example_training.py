#!/usr/bin/env python3
"""
Example: Training and Saving AI Models
Пример: Обучение и сохранение AI моделей
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from training_pipeline import ModelTrainingPipeline
from model_manager import ModelManager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def example_training_workflow():
    """Пример полного workflow обучения моделей"""
    
    print("🚀 Запуск примера обучения AI моделей")
    
    # 1. Конфигурация
    config = {
        "mt5": {
            "symbols": ["EURUSD", "GBPUSD", "USDJPY"]
        },
        "ai": {
            "models": ["lstm", "xgboost", "lightgbm"],
            "timeframes": ["M15", "H1", "H4"],
            "lookback_periods": [50, 100, 200],
            "retrain_interval": 24,
            "min_accuracy_threshold": 0.65,
            "auto_training": True
        }
    }
    
    # 2. Создание пайплайна обучения
    print("\n📚 Создание пайплайна обучения...")
    pipeline = ModelTrainingPipeline(config)
    
    # 3. Обучение моделей для EURUSD H1
    print("\n🎯 Обучение моделей для EURUSD H1...")
    
    try:
        results = pipeline.train_models(
            symbol="EURUSD",
            timeframe="H1",
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        
        print("✅ Обучение завершено!")
        
        # Анализ результатов
        print("\n📊 Результаты валидации:")
        for model_name, result in results['validation_results'].items():
            if 'accuracy' in result:
                print(f"   {model_name}: {result['accuracy']:.4f}")
        
        print("\n📈 Результаты backtesting:")
        backtest = results['backtest_results']
        print(f"   Общее количество сделок: {backtest.get('total_trades', 0)}")
        print(f"   Винрейт: {backtest.get('win_rate', 0):.2f}%")
        print(f"   Общая прибыль: {backtest.get('total_profit', 0):.2f}")
        print(f"   Sharpe Ratio: {backtest.get('sharpe_ratio', 0):.4f}")
        print(f"   Максимальная просадка: {backtest.get('max_drawdown', 0):.2f}%")
        
    except Exception as e:
        print(f"❌ Ошибка обучения: {e}")
        return
    
    # 4. Загрузка обученных моделей
    print("\n📂 Загрузка обученных моделей...")
    success = pipeline.load_models("EURUSD", "H1")
    
    if success:
        print("✅ Модели загружены успешно!")
        
        # Информация о моделях
        model_info = pipeline.get_model_info("EURUSD", "H1")
        print(f"\n📋 Информация о моделях:")
        for model_name, info in model_info.items():
            print(f"   {model_name}: {info.get('training_date', 'N/A')}")
    else:
        print("❌ Ошибка загрузки моделей")
    
    # 5. Создание менеджера моделей
    print("\n🤖 Создание менеджера моделей...")
    model_manager = ModelManager(config)
    
    # 6. Инициализация всех моделей
    print("\n🚀 Инициализация всех моделей...")
    success = model_manager.initialize_models(
        symbols=["EURUSD", "GBPUSD"],
        timeframes=["H1", "H4"]
    )
    
    if success:
        print("✅ Все модели инициализированы!")
        
        # Статус моделей
        status = model_manager.get_models_status()
        print(f"\n📊 Статус моделей:")
        print(f"   Всего моделей: {status['total_models']}")
        print(f"   Загружено: {status['loaded_models']}")
        print(f"   Обучено: {status['trained_models']}")
        print(f"   Переобучено: {status['retrained_models']}")
        
    else:
        print("❌ Проблемы с инициализацией моделей")
    
    # 7. Тестирование предсказаний
    print("\n🔮 Тестирование предсказаний...")
    
    # Создание тестовых данных
    import numpy as np
    import pandas as pd
    
    dates = pd.date_range('2023-12-01', '2023-12-31', freq='1H')
    np.random.seed(42)
    
    price = 100 + np.cumsum(np.random.randn(len(dates)) * 0.1
    volume = np.random.randint(1000, 10000, len(dates))
    
    test_data = pd.DataFrame({
        'open': price * (1 + np.random.randn(len(dates)) * 0.001),
        'high': price * (1 + abs(np.random.randn(len(dates)) * 0.002)),
        'low': price * (1 - abs(np.random.randn(len(dates)) * 0.002)),
        'close': price,
        'volume': volume
    }, index=dates)
    
    # Получение предсказания
    prediction = model_manager.get_prediction("EURUSD", "H1", test_data)
    
    if 'error' not in prediction:
        print("✅ Предсказание получено успешно!")
        print(f"   Уверенность: {prediction.get('confidence', 0):.4f}")
        print(f"   Класс: {prediction.get('ensemble_prediction', [])}")
    else:
        print(f"❌ Ошибка предсказания: {prediction['error']}")
    
    # 8. Тестирование производительности
    print("\n📈 Тестирование производительности...")
    performance = model_manager.get_model_performance("EURUSD", "H1")
    
    if 'error' not in performance:
        print("✅ Производительность получена!")
        print(f"   Общая точность: {performance.get('overall_accuracy', 0):.4f}")
        print(f"   Модели: {list(performance.get('models', {}).keys())}")
    else:
        print(f"❌ Ошибка получения производительности: {performance['error']}")
    
    # 9. Переобучение моделей
    print("\n🔄 Тестирование переобучения...")
    retrain_results = model_manager.retrain_models("EURUSD", "H1", force=False)
    
    if 'error' not in retrain_results:
        print("✅ Переобучение выполнено!")
        if 'status' in retrain_results and retrain_results['status'] == 'not_required':
            print("   Переобучение не требовалось")
        else:
            print("   Модели переобучены")
    else:
        print(f"❌ Ошибка переобучения: {retrain_results['error']}")
    
    print("\n🎉 Пример обучения завершен!")

async def example_advanced_training():
    """Пример продвинутого обучения с настройками"""
    
    print("\n🔧 Запуск продвинутого примера обучения...")
    
    # Расширенная конфигурация
    config = {
        "mt5": {
            "symbols": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
        },
        "ai": {
            "models": ["lstm", "xgboost", "lightgbm", "random_forest"],
            "timeframes": ["M15", "H1", "H4", "D1"],
            "lookback_periods": [50, 100, 200, 500],
            "retrain_interval": 12,  # Каждые 12 часов
            "min_accuracy_threshold": 0.70,  # Более высокий порог
            "auto_training": True,
            "model_params": {
                "lstm": {
                    "sequence_length": 30,
                    "units": 256,
                    "dropout": 0.3,
                    "epochs": 150,
                    "batch_size": 64
                },
                "xgboost": {
                    "n_estimators": 2000,
                    "max_depth": 8,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8
                },
                "lightgbm": {
                    "n_estimators": 2000,
                    "max_depth": 8,
                    "learning_rate": 0.05,
                    "num_leaves": 63,
                    "feature_fraction": 0.8
                }
            }
        }
    }
    
    # Создание пайплайна
    pipeline = ModelTrainingPipeline(config)
    
    # Обучение для нескольких символов и таймфреймов
    symbols = ["EURUSD", "GBPUSD"]
    timeframes = ["H1", "H4"]
    
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\n🎯 Обучение {symbol} {timeframe}...")
            
            try:
                results = pipeline.train_models(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date="2023-01-01",
                    end_date="2023-12-31"
                )
                
                # Проверка качества
                validation_results = results.get('validation_results', {})
                avg_accuracy = sum(
                    result.get('accuracy', 0) 
                    for result in validation_results.values()
                ) / len(validation_results) if validation_results else 0
                
                if avg_accuracy >= config['ai']['min_accuracy_threshold']:
                    print(f"✅ {symbol} {timeframe}: точность {avg_accuracy:.4f}")
                else:
                    print(f"⚠️ {symbol} {timeframe}: низкая точность {avg_accuracy:.4f}")
                    
            except Exception as e:
                print(f"❌ Ошибка обучения {symbol} {timeframe}: {e}")
    
    print("\n🎉 Продвинутое обучение завершено!")

async def example_model_management():
    """Пример управления моделями"""
    
    print("\n🗄️ Запуск примера управления моделями...")
    
    config = {
        "ai": {
            "models": ["lstm", "xgboost", "lightgbm"],
            "timeframes": ["H1", "H4"],
            "auto_training": True,
            "retrain_interval": 24,
            "min_accuracy_threshold": 0.65
        }
    }
    
    # Создание менеджера
    model_manager = ModelManager(config)
    
    # Инициализация
    success = model_manager.initialize_models(
        symbols=["EURUSD", "GBPUSD"],
        timeframes=["H1", "H4"]
    )
    
    if success:
        print("✅ Менеджер моделей инициализирован!")
        
        # Получение статуса
        status = model_manager.get_models_status()
        print(f"📊 Статус: {status['total_models']} моделей")
        
        # Мониторинг производительности
        for symbol in ["EURUSD", "GBPUSD"]:
            for timeframe in ["H1", "H4"]:
                performance = model_manager.get_model_performance(symbol, timeframe)
                
                if 'error' not in performance:
                    accuracy = performance.get('overall_accuracy', 0)
                    print(f"📈 {symbol} {timeframe}: {accuracy:.4f}")
                    
                    # Проверка необходимости переобучения
                    if accuracy < config['ai']['min_accuracy_threshold']:
                        print(f"🔄 Переобучение {symbol} {timeframe}...")
                        model_manager.retrain_models(symbol, timeframe, force=True)
        
        # Очистка старых моделей
        print("\n🧹 Очистка старых моделей...")
        model_manager.cleanup_old_models(days_to_keep=7)
        
    else:
        print("❌ Ошибка инициализации менеджера")
    
    print("\n🎉 Управление моделями завершено!")

if __name__ == "__main__":
    # Запуск примеров
    asyncio.run(example_training_workflow())
    asyncio.run(example_advanced_training())
    asyncio.run(example_model_management())
    
    print("\n🎯 Все примеры выполнены успешно!")
    print("📚 Дополнительную информацию см. в TRAINING_GUIDE.md")