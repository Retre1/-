#!/usr/bin/env python3
"""
🚀 Быстрый старт обучения ForexBot AI
Автоматизированный процесс обучения моделей
"""

import json
import logging
from pathlib import Path
from datetime import datetime
import sys
import os

# Добавление пути к модулям
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection.mt5_data_collector import MT5DataCollector
from training.advanced_training_system import AdvancedTrainingSystem

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quick_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def quick_training_pipeline():
    """Быстрый пайплайн обучения"""
    
    logger.info("🚀 Запуск быстрого обучения ForexBot AI")
    
    # 1. Загрузка конфигурации
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        logger.info("✅ Конфигурация загружена")
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки конфигурации: {e}")
        return False
    
    # 2. Сбор данных из MT5
    logger.info("📊 Этап 1: Сбор данных из MetaTrader5")
    try:
        data_collector = MT5DataCollector(config)
        
        # Подключение к MT5
        if not data_collector.connect_mt5():
            logger.error("❌ Не удалось подключиться к MT5")
            return False
        
        # Сбор данных для всех символов
        symbols = config.get('mt5', {}).get('symbols', ['EURUSD'])
        timeframes = config.get('mt5', {}).get('timeframes', ['H1'])
        
        for symbol in symbols:
            for timeframe in timeframes:
                logger.info(f"🔄 Сбор данных для {symbol} {timeframe}")
                
                # Получение последних данных (1 год)
                from datetime import datetime, timedelta
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)
                
                data = data_collector.collect_multiple_timeframes(symbol, start_date, end_date)
                
                if data:
                    logger.info(f"✅ Данные собраны для {symbol} {timeframe}")
                else:
                    logger.warning(f"⚠️ Нет данных для {symbol} {timeframe}")
        
        # Генерация отчета о данных
        report = data_collector.generate_data_report()
        logger.info(f"📊 Отчет о данных: {report['total_files']} файлов, {report['total_size_mb']:.2f} MB")
        
    except Exception as e:
        logger.error(f"❌ Ошибка сбора данных: {e}")
        return False
    
    # 3. Обучение моделей
    logger.info("🎓 Этап 2: Обучение AI моделей")
    try:
        training_system = AdvancedTrainingSystem(config)
        
        # Обучение для всех символов и таймфреймов
        for symbol in symbols:
            for timeframe in timeframes:
                logger.info(f"🎓 Обучение для {symbol} {timeframe}")
                
                success = training_system.run_complete_training(symbol, timeframe)
                
                if success:
                    logger.info(f"✅ Обучение {symbol} {timeframe} завершено успешно")
                else:
                    logger.error(f"❌ Ошибка обучения {symbol} {timeframe}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка обучения: {e}")
        return False
    
    # 4. Генерация финального отчета
    logger.info("📊 Этап 3: Генерация финального отчета")
    try:
        generate_final_report(config)
    except Exception as e:
        logger.error(f"❌ Ошибка генерации отчета: {e}")
    
    logger.info("🎉 Быстрое обучение завершено!")
    return True

def generate_final_report(config):
    """Генерация финального отчета"""
    try:
        from pathlib import Path
        import json
        
        report = {
            'training_date': datetime.now().isoformat(),
            'config': config,
            'summary': {
                'symbols_trained': config.get('mt5', {}).get('symbols', []),
                'timeframes_trained': config.get('mt5', {}).get('timeframes', []),
                'models_trained': config.get('ai', {}).get('models', []),
                'total_models': len(config.get('ai', {}).get('models', [])) * 
                               len(config.get('mt5', {}).get('symbols', [])) * 
                               len(config.get('mt5', {}).get('timeframes', []))
            },
            'next_steps': [
                "1. Проверьте качество моделей в data/reports/",
                "2. Настройте торговые параметры в config.json",
                "3. Запустите бота: python integrated_bot_advanced.py",
                "4. Откройте веб-интерфейс: http://localhost:8000"
            ]
        }
        
        # Сохранение отчета
        report_path = Path('data/reports/quick_training_report.json')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Финальный отчет сохранен: {report_path}")
        
        # Вывод краткой информации
        print("\n" + "="*50)
        print("🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("="*50)
        print(f"📊 Обучено моделей: {report['summary']['total_models']}")
        print(f"💱 Символы: {', '.join(report['summary']['symbols_trained'])}")
        print(f"⏰ Таймфреймы: {', '.join(report['summary']['timeframes_trained'])}")
        print(f"🤖 Модели: {', '.join(report['summary']['models_trained'])}")
        print("\n📋 Следующие шаги:")
        for step in report['next_steps']:
            print(f"   {step}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"❌ Ошибка генерации финального отчета: {e}")

def check_prerequisites():
    """Проверка предварительных требований"""
    logger.info("🔍 Проверка предварительных требований...")
    
    # Проверка наличия файлов
    required_files = ['config.json', 'advanced_ai_models.py', 'advanced_backtesting.py']
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"❌ Отсутствуют файлы: {', '.join(missing_files)}")
        return False
    
    # Проверка директорий
    required_dirs = ['data', 'data/market_data', 'data/models', 'data/reports']
    
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ Предварительные требования выполнены")
    return True

def main():
    """Основная функция"""
    print("🚀 Быстрый старт обучения ForexBot AI")
    print("="*50)
    
    # Проверка предварительных требований
    if not check_prerequisites():
        print("❌ Предварительные требования не выполнены")
        return
    
    # Запуск пайплайна обучения
    success = quick_training_pipeline()
    
    if success:
        print("\n🎉 Обучение завершено успешно!")
        print("📊 Проверьте результаты в директории data/")
    else:
        print("\n❌ Обучение завершено с ошибками")
        print("📋 Проверьте логи для деталей")

if __name__ == "__main__":
    main()