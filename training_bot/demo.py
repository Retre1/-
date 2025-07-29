#!/usr/bin/env python3
"""
🎯 ДЕМОНСТРАЦИЯ ПОЭТАПНОГО ОБУЧЕНИЯ
Быстрый тест системы: XGBoost → LSTM → Ensemble
"""

import asyncio
import sys
import time
from datetime import datetime
from loguru import logger

# Настройка простого логирования
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>", level="INFO")

from progressive_trainer import ProgressiveForexTrainer


async def demo_progressive_training():
    """Демонстрация поэтапного обучения"""
    
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                           🎯 ДЕМО ОБУЧЕНИЯ                                     ║
║                   XGBoost → LSTM → Ensemble                                   ║
║                                                                               ║
║  Это быстрая демонстрация поэтапного обучения AI моделей                     ║
║  для торговли на форекс с интеграцией Solana токена                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    logger.info("🚀 Начинаем демонстрацию поэтапного обучения!")
    
    # Создание тренера
    trainer = ProgressiveForexTrainer("EURUSD_DEMO")
    
    try:
        # =================== ДЕМО ПАРАМЕТРЫ ===================
        logger.info("⚡ ДЕМО РЕЖИМ: ускоренные параметры для быстрого тестирования")
        logger.info("🎲 Используются синтетические данные для демонстрации")
        
        # Загрузка данных
        df = trainer.load_data()
        X, y, feature_columns = trainer.prepare_data(df)
        
        logger.info(f"📊 Данные готовы: {len(X)} образцов, {len(feature_columns)} признаков")
        
        # =================== ЭТАП 1: XGBoost ===================
        print("\n" + "="*80)
        logger.info("🎯 ЭТАП 1: БЫСТРОЕ ОБУЧЕНИЕ XGBoost")
        logger.info("   Цель: Получить базовую модель за минимальное время")
        logger.info("   Параметры: 5 trials оптимизации")
        
        start_time = time.time()
        
        xgb_results = trainer.train_xgboost_phase(
            X, y, 
            optimize=True, 
            trials=5  # Быстро для демо
        )
        
        xgb_time = time.time() - start_time
        logger.info(f"✅ XGBoost завершен за {xgb_time:.1f} секунд")
        logger.info(f"📈 Результат: MSE={xgb_results['test_mse']:.6f}, Точность={xgb_results['directional_accuracy']:.1f}%")
        
        # =================== ЭТАП 2: LSTM ===================
        print("\n" + "="*80)
        logger.info("🧠 ЭТАП 2: БЫСТРОЕ ОБУЧЕНИЕ LSTM")
        logger.info("   Цель: Улучшить точность через нейронные сети")
        logger.info("   Параметры: 3 trials оптимизации, 30 эпох")
        
        start_time = time.time()
        
        lstm_results = trainer.train_lstm_phase(
            X, y,
            sequence_length=30,  # Короче для демо
            optimize=True,
            trials=3  # Быстро для демо
        )
        
        lstm_time = time.time() - start_time
        logger.info(f"✅ LSTM завершен за {lstm_time:.1f} секунд")
        logger.info(f"📈 Результат: MSE={lstm_results['test_mse']:.6f}, Точность={lstm_results['directional_accuracy']:.1f}%")
        
        # =================== ЭТАП 3: ENSEMBLE ===================
        print("\n" + "="*80)
        logger.info("🏆 ЭТАП 3: СОЗДАНИЕ ENSEMBLE")
        logger.info("   Цель: Объединить лучшее от XGBoost и LSTM")
        logger.info("   Параметры: Автоматическая оптимизация весов")
        
        start_time = time.time()
        
        ensemble_results = trainer.train_ensemble_phase(X, y)
        
        ensemble_time = time.time() - start_time
        logger.info(f"✅ Ensemble завершен за {ensemble_time:.1f} секунд")
        logger.info(f"📈 Результат: MSE={ensemble_results['test_mse']:.6f}, Точность={ensemble_results['directional_accuracy']:.1f}%")
        
        # =================== РЕЗУЛЬТАТЫ ===================
        print("\n" + "="*80)
        logger.info("📊 ФИНАЛЬНОЕ СРАВНЕНИЕ МОДЕЛЕЙ")
        print("="*80)
        
        # Таблица результатов
        results_table = f"""
╔════════════════╦══════════════╦══════════════════╦═══════════════╗
║ Модель         ║ MSE          ║ Точность (%)     ║ Время (сек)   ║
╠════════════════╬══════════════╬══════════════════╬═══════════════╣
║ XGBoost        ║ {xgb_results['test_mse']:.6f}   ║ {xgb_results['directional_accuracy']:>14.1f}% ║ {xgb_time:>11.1f}   ║
║ LSTM           ║ {lstm_results['test_mse']:.6f}   ║ {lstm_results['directional_accuracy']:>14.1f}% ║ {lstm_time:>11.1f}   ║
║ Ensemble       ║ {ensemble_results['test_mse']:.6f}   ║ {ensemble_results['directional_accuracy']:>14.1f}% ║ {ensemble_time:>11.1f}   ║
╚════════════════╩══════════════╩══════════════════╩═══════════════╝
        """
        print(results_table)
        
        # Лучшая модель
        models = {
            'XGBoost': xgb_results,
            'LSTM': lstm_results, 
            'Ensemble': ensemble_results
        }
        
        best_model = min(models.keys(), key=lambda x: models[x]['test_mse'])
        best_mse = models[best_model]['test_mse']
        best_acc = models[best_model]['directional_accuracy']
        
        logger.info(f"🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model}")
        logger.info(f"   📊 MSE: {best_mse:.6f}")
        logger.info(f"   🎯 Точность: {best_acc:.1f}%")
        
        # Веса ансамбля
        weights = ensemble_results['weights']
        logger.info(f"⚖️ Веса в ансамбле: XGBoost={weights[0]:.2f}, LSTM={weights[1]:.2f}")
        
        total_time = xgb_time + lstm_time + ensemble_time
        logger.info(f"⏱️ Общее время демо: {total_time:.1f} секунд ({total_time/60:.1f} минут)")
        
        # =================== СОЗДАНИЕ ГРАФИКОВ ===================
        print("\n" + "="*80)
        logger.info("📊 Создание графиков сравнения...")
        
        try:
            trainer.create_comparison_plots()
            logger.info("✅ Графики созданы успешно!")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось создать графики: {e}")
        
        # =================== ЗАКЛЮЧЕНИЕ ===================
        print("\n" + "="*80)
        logger.info("🎉 ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА!")
        print("="*80)
        
        conclusion = f"""
💡 ВЫВОДЫ ДЕМОНСТРАЦИИ:

✅ Поэтапное обучение работает!
   • Каждый этап добавляет ценность к общему результату
   • XGBoost дает быструю базовую модель
   • LSTM улучшает качество прогнозов
   • Ensemble объединяет лучшее от обеих моделей

📈 Качество моделей:
   • Directional Accuracy > 50% считается хорошим результатом
   • MSE показывает точность прогноза цены
   • Ensemble обычно дает лучшие результаты

⚡ Производительность:
   • XGBoost: Быстрое обучение (~{xgb_time:.0f}с)
   • LSTM: Среднее время (~{lstm_time:.0f}с) 
   • Ensemble: Мгновенное создание (~{ensemble_time:.0f}с)

🚀 Следующие шаги:
   1. Попробуйте с реальными данными
   2. Увеличьте количество trials для лучшего качества
   3. Экспериментируйте с разными валютными парами
   4. Интегрируйте лучшую модель в торгового бота

💾 Модели сохранены в: progressive_models/EURUSD_DEMO/
        """
        
        print(conclusion)
        
        return {
            'xgboost': xgb_results,
            'lstm': lstm_results,
            'ensemble': ensemble_results
        }
        
    except Exception as e:
        logger.error(f"❌ Ошибка в демонстрации: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


async def quick_comparison_demo():
    """Быстрое сравнение всех трех подходов"""
    
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        ⚡ БЫСТРОЕ СРАВНЕНИЕ                                    ║
║            Какая модель лучше для форекс торговли?                           ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    logger.info("⚡ Запускаем ускоренное сравнение моделей...")
    
    results = await demo_progressive_training()
    
    if results:
        # Анализ результатов
        xgb = results['xgboost']
        lstm = results['lstm'] 
        ensemble = results['ensemble']
        
        print(f"""
🎯 КРАТКИЕ ВЫВОДЫ:

1️⃣ XGBoost:
   • Скорость: ⭐⭐⭐⭐⭐ (очень быстро)
   • Точность: {'⭐⭐⭐⭐⭐' if xgb['directional_accuracy'] > 55 else '⭐⭐⭐⭐' if xgb['directional_accuracy'] > 50 else '⭐⭐⭐'}
   • Простота: ⭐⭐⭐⭐⭐ (легко настроить)
   
2️⃣ LSTM:  
   • Скорость: ⭐⭐⭐ (требует времени)
   • Точность: {'⭐⭐⭐⭐⭐' if lstm['directional_accuracy'] > 55 else '⭐⭐⭐⭐' if lstm['directional_accuracy'] > 50 else '⭐⭐⭐'}
   • Гибкость: ⭐⭐⭐⭐⭐ (много настроек)
   
3️⃣ Ensemble:
   • Скорость: ⭐⭐⭐⭐ (быстро после обучения)
   • Точность: {'⭐⭐⭐⭐⭐' if ensemble['directional_accuracy'] > 55 else '⭐⭐⭐⭐' if ensemble['directional_accuracy'] > 50 else '⭐⭐⭐'}
   • Стабильность: ⭐⭐⭐⭐⭐ (самая надежная)

🏆 РЕКОМЕНДАЦИЯ: {'Ensemble' if ensemble['directional_accuracy'] == max(xgb['directional_accuracy'], lstm['directional_accuracy'], ensemble['directional_accuracy']) else 'XGBoost' if xgb['directional_accuracy'] >= lstm['directional_accuracy'] else 'LSTM'}
        """)
    
    logger.info("✅ Быстрое сравнение завершено!")


def main():
    """Главная функция демо"""
    
    print("🎯 Выберите режим демонстрации:")
    print("1. Полная демонстрация поэтапного обучения")
    print("2. Быстрое сравнение моделей")
    print("3. Выход")
    
    try:
        choice = input("\nВведите номер (1-3): ").strip()
        
        if choice == "1":
            logger.info("🚀 Запускаем полную демонстрацию...")
            asyncio.run(demo_progressive_training())
            
        elif choice == "2":
            logger.info("⚡ Запускаем быстрое сравнение...")
            asyncio.run(quick_comparison_demo())
            
        elif choice == "3":
            logger.info("👋 До свидания!")
            
        else:
            logger.warning("❌ Неверный выбор. Попробуйте снова.")
            main()
            
    except KeyboardInterrupt:
        logger.info("\n⚠️ Демонстрация прервана пользователем")
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()