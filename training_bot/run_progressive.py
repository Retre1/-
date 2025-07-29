#!/usr/bin/env python3
"""
🚀 Запуск поэтапного обучения: XGBoost → LSTM → Ensemble
Простой интерфейс для прогрессивного обучения моделей
"""

import asyncio
import argparse
import sys
import os
from datetime import datetime
from loguru import logger

# Добавляем путь к проекту
sys.path.append('..')

from progressive_trainer import ProgressiveForexTrainer


def setup_logging():
    """Настройка логирования"""
    
    # Удаляем стандартный обработчик
    logger.remove()
    
    # Консольный вывод с цветами
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
        colorize=True
    )
    
    # Файл лога
    log_file = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    os.makedirs("logs", exist_ok=True)
    
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        rotation="10 MB"
    )
    
    logger.info(f"📝 Логи сохраняются в {log_file}")


def print_banner():
    """Красивый баннер"""
    
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        🤖 PROGRESSIVE FOREX TRAINER                           ║
║                      Поэтапное обучение AI моделей                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🎯 ЭТАП 1: XGBoost     → Быстро и эффективно (10-30 мин)                  ║
║  🧠 ЭТАП 2: LSTM        → Улучшение точности (1-4 часа)                    ║  
║  🏆 ЭТАП 3: Ensemble    → Максимальная точность (автоматически)             ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    
    print(banner)


def print_options():
    """Показать доступные опции"""
    
    options = """
📋 ДОСТУПНЫЕ ОПЦИИ:

🚀 Режимы запуска:
   --quick         Быстрое обучение (10-15 мин)
   --standard      Стандартное обучение (1-3 часа) [по умолчанию]
   --professional  Профессиональное обучение (4-8 часов)

🎯 Символы:
   --symbol EURUSD    Валютная пара для обучения [по умолчанию: EURUSD]
   
📂 Данные:
   --data path/to/data.csv    Путь к CSV файлу с данными
                              (если не указан - используются синтетические)

🤖 Модели:
   --xgboost-only     Только XGBoost
   --lstm-only        Только LSTM  
   --ensemble-only    Только Ensemble (требует уже обученных моделей)

💻 Примеры:
   python run_progressive.py --quick
   python run_progressive.py --symbol GBPUSD --standard
   python run_progressive.py --data ../data/collected/EURUSD.csv --professional
   python run_progressive.py --xgboost-only --quick
    """
    
    print(options)


async def run_training(args):
    """Запуск обучения с заданными параметрами"""
    
    logger.info(f"🎯 Начинаем обучение для {args.symbol}")
    logger.info(f"📊 Режим: {args.mode}")
    
    if args.data:
        logger.info(f"📂 Данные: {args.data}")
    else:
        logger.info("🎲 Используются синтетические данные")
    
    # Создание тренера
    trainer = ProgressiveForexTrainer(args.symbol)
    
    # Определение параметров по режиму
    quick_mode = (args.mode == 'quick')
    
    try:
        if args.xgboost_only:
            logger.info("🎯 Режим: только XGBoost")
            
            # Загрузка данных
            df = trainer.load_data(args.data)
            X, y, _ = trainer.prepare_data(df)
            
            # Обучение XGBoost
            trials = 10 if quick_mode else 50
            results = trainer.train_xgboost_phase(X, y, optimize=True, trials=trials)
            
        elif args.lstm_only:
            logger.info("🧠 Режим: только LSTM")
            
            # Загрузка данных
            df = trainer.load_data(args.data)
            X, y, _ = trainer.prepare_data(df)
            
            # Обучение LSTM
            trials = 5 if quick_mode else 30
            results = trainer.train_lstm_phase(X, y, optimize=True, trials=trials)
            
        elif args.ensemble_only:
            logger.info("🏆 Режим: только Ensemble")
            
            # Проверка наличия обученных моделей
            model_dir = f"progressive_models/{args.symbol}"
            if not (os.path.exists(f"{model_dir}/xgboost_model.pkl") and 
                   os.path.exists(f"{model_dir}/lstm_model.h5")):
                logger.error("❌ Для Ensemble нужны обученные XGBoost и LSTM модели")
                logger.info("💡 Сначала запустите полное обучение или отдельно XGBoost и LSTM")
                return
            
            # Загрузка моделей и создание ансамбля
            # TODO: Добавить загрузку сохраненных моделей
            logger.warning("⚠️ Загрузка сохраненных моделей пока не реализована")
            
        else:
            logger.info("🚀 Режим: полное поэтапное обучение")
            
            # Полное обучение
            results = await trainer.run_progressive_training(
                data_path=args.data,
                quick_mode=quick_mode
            )
        
        logger.info("✅ Обучение успешно завершено!")
        
        return results
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Обучение прервано пользователем")
        
    except Exception as e:
        logger.error(f"❌ Ошибка при обучении: {e}")
        import traceback
        logger.debug(traceback.format_exc())


def main():
    """Главная функция"""
    
    setup_logging()
    print_banner()
    
    # Парсинг аргументов
    parser = argparse.ArgumentParser(
        description="🤖 Progressive Forex Trainer - Поэтапное обучение AI моделей",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Режимы
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--quick', action='store_const', dest='mode', const='quick',
                           help='⚡ Быстрое обучение (10-15 мин)')
    mode_group.add_argument('--standard', action='store_const', dest='mode', const='standard', 
                           help='🔥 Стандартное обучение (1-3 часа)')
    mode_group.add_argument('--professional', action='store_const', dest='mode', const='professional',
                           help='🏆 Профессиональное обучение (4-8 часов)')
    
    # Модели
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('--xgboost-only', action='store_true',
                            help='🎯 Обучить только XGBoost')
    model_group.add_argument('--lstm-only', action='store_true',
                            help='🧠 Обучить только LSTM')
    model_group.add_argument('--ensemble-only', action='store_true',
                            help='🏆 Создать только Ensemble (требует готовых моделей)')
    
    # Данные
    parser.add_argument('--symbol', default='EURUSD',
                       help='💱 Валютная пара (по умолчанию: EURUSD)')
    parser.add_argument('--data', 
                       help='📂 Путь к CSV файлу с данными (опционально)')
    
    # Дополнительные опции
    parser.add_argument('--help-options', action='store_true',
                       help='📋 Показать подробные опции и примеры')
    
    args = parser.parse_args()
    
    # Показать опции
    if args.help_options:
        print_options()
        return
    
    # Установка режима по умолчанию
    if not args.mode:
        args.mode = 'standard'
    
    # Валидация
    if args.data and not os.path.exists(args.data):
        logger.error(f"❌ Файл данных не найден: {args.data}")
        return
    
    # Приветствие
    logger.info("🎉 Добро пожаловать в Progressive Forex Trainer!")
    
    # Запуск обучения
    results = asyncio.run(run_training(args))
    
    if results:
        logger.info("🎊 Все готово! Ваши AI модели обучены и готовы к использованию!")
        
        # Показать где сохранены модели
        model_dir = f"progressive_models/{args.symbol}"
        logger.info(f"💾 Модели сохранены в: {model_dir}")
        logger.info(f"📊 Графики результатов: {model_dir}/comparison_results.png")


if __name__ == "__main__":
    main()