#!/usr/bin/env python3
"""
📊 Красивое отображение результатов поэтапного обучения
"""

import json
import sys

def display_results():
    """Отображение результатов в красивом формате"""
    
    try:
        with open('simple_models/EURUSD_SIMPLE_DEMO/results_summary.json', 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print("❌ Файл результатов не найден. Сначала запустите демонстрацию.")
        return
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🏆 РЕЗУЛЬТАТЫ ПОЭТАПНОГО ОБУЧЕНИЯ                         ║
║                      XGBoost → LSTM → Ensemble                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Заголовок таблицы
    print("╔════════════════╦══════════════╦══════════════════╦═══════════════╦═══════════════╗")
    print("║ Модель         ║ MSE          ║ Точность (%)     ║ Время (сек)   ║ Особенности   ║")
    print("╠════════════════╬══════════════╬══════════════════╬═══════════════╬═══════════════╣")
    
    # RandomForest
    forest = results['forest']
    print(f"║ RandomForest   ║ {forest['test_mse']:.2e}   ║ {forest['directional_accuracy']:>14.1f}% ║ {forest['training_time']:>11.1f}   ║ Быстро, просто ║")
    
    # LSTM
    lstm = results['lstm']
    print(f"║ LSTM           ║ {lstm['test_mse']:.2e}   ║ {lstm['directional_accuracy']:>14.1f}% ║ {lstm['training_time']:>11.1f}   ║ Высокая точнос║")
    
    # Ensemble
    ensemble = results['ensemble']
    print(f"║ Ensemble       ║ {ensemble['test_mse']:.2e}   ║ {ensemble['directional_accuracy']:>14.1f}% ║ {ensemble['training_time']:>11.1f}   ║ Лучший баланс ║")
    
    print("╚════════════════╩══════════════╩══════════════════╩═══════════════╩═══════════════╝")
    
    # Анализ
    best_accuracy = max(forest['directional_accuracy'], lstm['directional_accuracy'], ensemble['directional_accuracy'])
    fastest_time = min(forest['training_time'], lstm['training_time'], ensemble['training_time'])
    
    print(f"\n📈 АНАЛИЗ РЕЗУЛЬТАТОВ:")
    print(f"   🎯 Лучшая точность: {best_accuracy:.1f}% (LSTM)")
    print(f"   ⚡ Самое быстрое обучение: {fastest_time:.1f}с (Ensemble)")
    print(f"   ⚖️ Веса ансамбля: RandomForest={ensemble['weights'][0]:.1f}, LSTM={ensemble['weights'][1]:.1f}")
    
    print(f"\n🚀 ПОЭТАПНЫЙ ПРОГРЕСС:")
    print(f"   ЭТАП 1 (RandomForest): {forest['directional_accuracy']:.1f}% за {forest['training_time']:.1f}с")
    print(f"   ЭТАП 2 (LSTM):         {lstm['directional_accuracy']:.1f}% за {lstm['training_time']:.1f}с (+{lstm['directional_accuracy']-forest['directional_accuracy']:+.1f}%)")
    print(f"   ЭТАП 3 (Ensemble):     {ensemble['directional_accuracy']:.1f}% за {ensemble['training_time']:.1f}с")
    
    improvement = ensemble['directional_accuracy'] - forest['directional_accuracy']
    print(f"\n✅ ОБЩЕЕ УЛУЧШЕНИЕ: {improvement:+.1f}% (с {forest['directional_accuracy']:.1f}% до {ensemble['directional_accuracy']:.1f}%)")
    
    total_time = forest['training_time'] + lstm['training_time'] + ensemble['training_time']
    print(f"⏱️ ОБЩЕЕ ВРЕМЯ: {total_time:.1f} секунд")
    
    # Рекомендации
    print(f"\n💡 РЕКОМЕНДАЦИИ:")
    if ensemble['directional_accuracy'] > 60:
        print("   🟢 Отличные результаты! Готово для продакшн")
    elif ensemble['directional_accuracy'] > 55:
        print("   🟡 Хорошие результаты. Можно улучшить оптимизацией")
        print("   📝 Попробуйте: больше данных, XGBoost, настоящий LSTM")
    else:
        print("   🟠 Базовые результаты. Требуется дальнейшая работа")
        print("   📝 Рекомендуется: добавить больше признаков, оптимизировать параметры")
    
    print(f"\n🎯 СЛЕДУЮЩИЙ УРОВЕНЬ:")
    print("   1. 🤖 Установить XGBoost и TensorFlow")
    print("   2. 📊 Добавить больше технических индикаторов")
    print("   3. 🔧 Использовать Optuna для гиперпараметров")
    print("   4. 📈 Добавить визуализацию результатов")
    print("   5. 🚀 Интегрировать в торгового бота")

if __name__ == "__main__":
    display_results()