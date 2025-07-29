#!/usr/bin/env python3
"""
ForexBot AI Trading System - Quick Start Script
Скрипт для быстрого запуска всей системы
"""

import asyncio
import argparse
import subprocess
import sys
import os
import signal
import time
from pathlib import Path

# Добавляем текущую директорию в путь
sys.path.append(str(Path(__file__).parent))

from trading_bot.main import ForexTradingBot
from loguru import logger


class ForexBotRunner:
    """Управление запуском и остановкой компонентов системы"""
    
    def __init__(self):
        self.processes = {}
        self.bot = None
        self.running = False
    
    def setup_logging(self):
        """Настройка логирования"""
        log_dir = Path("data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            "data/logs/runner.log",
            rotation="10 MB",
            retention="30 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
    
    def check_dependencies(self):
        """Проверка зависимостей"""
        logger.info("Проверка зависимостей...")
        
        # Проверка Python пакетов
        required_packages = [
            "MetaTrader5",
            "pandas",
            "numpy",
            "tensorflow",
            "xgboost",
            "lightgbm",
            "solana",
            "fastapi",
            "uvicorn"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Отсутствуют пакеты: {', '.join(missing_packages)}")
            logger.info("Запустите: pip install -r requirements.txt")
            return False
        
        # Проверка конфигурации
        if not Path("config.json").exists():
            logger.error("Файл config.json не найден")
            logger.info("Скопируйте config.json.example в config.json и настройте")
            return False
        
        logger.info("Все зависимости найдены")
        return True
    
    def start_web_interface(self):
        """Запуск веб-интерфейса"""
        logger.info("Запуск веб-интерфейса...")
        
        try:
            cmd = [
                sys.executable, 
                "web_interface/backend/main.py"
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path.cwd()
            )
            
            self.processes['web'] = process
            logger.info(f"Веб-интерфейс запущен (PID: {process.pid})")
            logger.info("Доступен по адресу: http://localhost:8000")
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка запуска веб-интерфейса: {e}")
            return False
    
    async def start_trading_bot(self):
        """Запуск торгового бота"""
        logger.info("Запуск торгового бота...")
        
        try:
            self.bot = ForexTradingBot()
            
            # Инициализация бота
            if not await self.bot.initialize():
                logger.error("Не удалось инициализировать торгового бота")
                return False
            
            logger.info("Торговый бот инициализирован")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка запуска торгового бота: {e}")
            return False
    
    async def run_demo_mode(self):
        """Запуск в демо режиме"""
        logger.info("=== ДЕМО РЕЖИМ ===")
        logger.info("Бот запущен в демонстрационном режиме")
        logger.info("Никаких реальных сделок не будет совершено")
        
        # Имитация работы бота
        while self.running:
            try:
                # Получение статуса
                if self.bot:
                    status = await self.bot.get_status()
                    logger.info(f"Статус: {status.get('is_running', False)}")
                
                await asyncio.sleep(30)  # Обновления каждые 30 секунд
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ошибка в демо режиме: {e}")
                await asyncio.sleep(60)
    
    async def run_trading_mode(self):
        """Запуск в торговом режиме"""
        logger.warning("=== ТОРГОВЫЙ РЕЖИМ ===")
        logger.warning("ВНИМАНИЕ: Бот будет совершать РЕАЛЬНЫЕ сделки!")
        
        # Подтверждение от пользователя
        confirmation = input("Вы уверены? Введите 'YES' для продолжения: ")
        if confirmation != "YES":
            logger.info("Запуск отменен пользователем")
            return
        
        # Запуск торговли
        await self.bot.start_trading()
    
    def stop_all(self):
        """Остановка всех компонентов"""
        logger.info("Остановка всех компонентов...")
        
        self.running = False
        
        # Остановка торгового бота
        if self.bot:
            try:
                asyncio.create_task(self.bot.stop_trading())
            except:
                pass
        
        # Остановка процессов
        for name, process in self.processes.items():
            try:
                logger.info(f"Остановка {name} (PID: {process.pid})")
                process.terminate()
                
                # Ждем завершения
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    
            except Exception as e:
                logger.error(f"Ошибка остановки {name}: {e}")
        
        logger.info("Все компоненты остановлены")
    
    def signal_handler(self, signum, frame):
        """Обработчик сигналов"""
        logger.info(f"Получен сигнал {signum}")
        self.stop_all()
        sys.exit(0)
    
    async def run(self, mode="demo"):
        """Главный цикл запуска"""
        self.setup_logging()
        
        # Установка обработчиков сигналов
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("=" * 50)
        logger.info("ForexBot AI Trading System")
        logger.info("=" * 50)
        
        # Проверка зависимостей
        if not self.check_dependencies():
            return False
        
        self.running = True
        
        try:
            # Запуск веб-интерфейса
            if not self.start_web_interface():
                return False
            
            # Ждем запуска веб-сервера
            await asyncio.sleep(3)
            
            # Запуск торгового бота
            if not await self.start_trading_bot():
                return False
            
            # Выбор режима работы
            if mode == "demo":
                await self.run_demo_mode()
            elif mode == "trade":
                await self.run_trading_mode()
            else:
                logger.error(f"Неизвестный режим: {mode}")
                return False
                
        except KeyboardInterrupt:
            logger.info("Получено прерывание от пользователя")
        except Exception as e:
            logger.error(f"Критическая ошибка: {e}")
        finally:
            self.stop_all()
        
        return True


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description="ForexBot AI Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python run.py --mode demo          # Демо режим
  python run.py --mode trade         # Торговый режим
  python run.py --web-only           # Только веб-интерфейс
  python run.py --check              # Проверка зависимостей
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["demo", "trade"],
        default="demo",
        help="Режим работы (demo/trade)"
    )
    
    parser.add_argument(
        "--web-only",
        action="store_true",
        help="Запуск только веб-интерфейса"
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="Проверка зависимостей и конфигурации"
    )
    
    args = parser.parse_args()
    
    runner = ForexBotRunner()
    
    # Только проверка
    if args.check:
        runner.setup_logging()
        success = runner.check_dependencies()
        sys.exit(0 if success else 1)
    
    # Только веб-интерфейс
    if args.web_only:
        runner.setup_logging()
        if runner.check_dependencies():
            runner.start_web_interface()
            print("Веб-интерфейс запущен: http://localhost:8000")
            print("Нажмите Ctrl+C для остановки")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                runner.stop_all()
        sys.exit(0)
    
    # Полный запуск
    try:
        success = asyncio.run(runner.run(args.mode))
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nОстановка системы...")
        sys.exit(0)


if __name__ == "__main__":
    main()