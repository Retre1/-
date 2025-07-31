#!/usr/bin/env python3
"""
Enhanced ForexBot AI System Launcher
Запуск улучшенной системы ForexBot AI
"""

import os
import sys
import subprocess
import time
import signal
import psutil
import asyncio
import json
from pathlib import Path
from datetime import datetime
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system_launcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedSystemLauncher:
    """Запуск улучшенной системы ForexBot AI"""
    
    def __init__(self):
        self.processes = {}
        self.config = self.load_config()
        self.services = [
            'redis',
            'postgres',
            'main_app',
            'monitoring',
            'scaling_node',
            'cache_manager'
        ]
        
    def load_config(self):
        """Загрузка конфигурации"""
        try:
            with open('config.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Файл config.json не найден, используются настройки по умолчанию")
            return self.get_default_config()
    
    def get_default_config(self):
        """Получение конфигурации по умолчанию"""
        return {
            "main_app": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 4
            },
            "monitoring": {
                "host": "0.0.0.0",
                "port": 8001
            },
            "scaling": {
                "host": "0.0.0.0",
                "port": 8002
            },
            "redis": {
                "host": "localhost",
                "port": 6379
            },
            "postgres": {
                "host": "localhost",
                "port": 5432,
                "database": "forexbot",
                "user": "forexbot",
                "password": "password"
            }
        }
    
    def check_dependencies(self):
        """Проверка зависимостей"""
        logger.info("🔍 Проверка зависимостей...")
        
        # Проверка Python версии
        if sys.version_info < (3, 8):
            logger.error("❌ Требуется Python 3.8 или выше")
            return False
        
        # Проверка необходимых пакетов
        required_packages = [
            'fastapi', 'uvicorn', 'redis', 'psutil', 
            'prometheus_client', 'pytest', 'aiohttp'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"❌ Отсутствуют пакеты: {', '.join(missing_packages)}")
            logger.info("💡 Установите зависимости: pip install -r requirements.txt")
            return False
        
        logger.info("✅ Все зависимости установлены")
        return True
    
    def create_directories(self):
        """Создание необходимых директорий"""
        directories = [
            'logs',
            'data',
            'data/models',
            'data/backtests',
            'data/reports',
            'tests',
            'web_interface/frontend'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ Директории созданы")
    
    def start_redis(self):
        """Запуск Redis"""
        try:
            # Проверка, запущен ли Redis
            result = subprocess.run(['redis-cli', 'ping'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info("✅ Redis уже запущен")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        try:
            logger.info("🚀 Запуск Redis...")
            process = subprocess.Popen(['redis-server'], 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE)
            self.processes['redis'] = process
            
            # Ожидание запуска
            time.sleep(3)
            
            # Проверка запуска
            result = subprocess.run(['redis-cli', 'ping'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info("✅ Redis запущен успешно")
                return True
            else:
                logger.error("❌ Ошибка запуска Redis")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка запуска Redis: {e}")
            return False
    
    def start_postgres(self):
        """Запуск PostgreSQL (если доступен)"""
        try:
            # Проверка подключения к PostgreSQL
            import psycopg2
            conn = psycopg2.connect(
                host=self.config['postgres']['host'],
                port=self.config['postgres']['port'],
                database=self.config['postgres']['database'],
                user=self.config['postgres']['user'],
                password=self.config['postgres']['password']
            )
            conn.close()
            logger.info("✅ PostgreSQL подключен")
            return True
        except Exception as e:
            logger.warning(f"⚠️ PostgreSQL недоступен: {e}")
            logger.info("💡 Система будет использовать SQLite")
            return False
    
    def start_main_app(self):
        """Запуск основного приложения"""
        try:
            logger.info("🚀 Запуск основного приложения...")
            
            cmd = [
                sys.executable, 'integrated_bot_advanced.py',
                '--host', self.config['main_app']['host'],
                '--port', str(self.config['main_app']['port'])
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.processes['main_app'] = process
            
            # Ожидание запуска
            time.sleep(5)
            
            if process.poll() is None:
                logger.info("✅ Основное приложение запущено")
                return True
            else:
                logger.error("❌ Ошибка запуска основного приложения")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка запуска основного приложения: {e}")
            return False
    
    def start_monitoring(self):
        """Запуск системы мониторинга"""
        try:
            logger.info("🚀 Запуск системы мониторинга...")
            
            cmd = [
                sys.executable, 'monitoring_system.py',
                '--host', self.config['monitoring']['host'],
                '--port', str(self.config['monitoring']['port'])
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.processes['monitoring'] = process
            
            # Ожидание запуска
            time.sleep(3)
            
            if process.poll() is None:
                logger.info("✅ Система мониторинга запущена")
                return True
            else:
                logger.error("❌ Ошибка запуска системы мониторинга")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка запуска системы мониторинга: {e}")
            return False
    
    def start_scaling_node(self):
        """Запуск узла масштабирования"""
        try:
            logger.info("🚀 Запуск узла масштабирования...")
            
            cmd = [
                sys.executable, 'scaling_system.py',
                '--host', self.config['scaling']['host'],
                '--port', str(self.config['scaling']['port'])
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.processes['scaling_node'] = process
            
            # Ожидание запуска
            time.sleep(3)
            
            if process.poll() is None:
                logger.info("✅ Узел масштабирования запущен")
                return True
            else:
                logger.error("❌ Ошибка запуска узла масштабирования")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка запуска узла масштабирования: {e}")
            return False
    
    def run_tests(self):
        """Запуск тестов"""
        try:
            logger.info("🧪 Запуск тестов...")
            
            cmd = [sys.executable, '-m', 'pytest', 'tests/', '-v']
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✅ Все тесты прошли успешно")
                return True
            else:
                logger.warning(f"⚠️ Некоторые тесты не прошли: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка запуска тестов: {e}")
            return False
    
    def check_system_health(self):
        """Проверка здоровья системы"""
        logger.info("🏥 Проверка здоровья системы...")
        
        health_status = {}
        
        # Проверка основного приложения
        if 'main_app' in self.processes:
            process = self.processes['main_app']
            if process.poll() is None:
                health_status['main_app'] = 'healthy'
            else:
                health_status['main_app'] = 'unhealthy'
        
        # Проверка Redis
        try:
            result = subprocess.run(['redis-cli', 'ping'], 
                                  capture_output=True, text=True, timeout=5)
            health_status['redis'] = 'healthy' if result.returncode == 0 else 'unhealthy'
        except:
            health_status['redis'] = 'unhealthy'
        
        # Проверка мониторинга
        if 'monitoring' in self.processes:
            process = self.processes['monitoring']
            if process.poll() is None:
                health_status['monitoring'] = 'healthy'
            else:
                health_status['monitoring'] = 'unhealthy'
        
        # Вывод статуса
        for service, status in health_status.items():
            if status == 'healthy':
                logger.info(f"✅ {service}: {status}")
            else:
                logger.error(f"❌ {service}: {status}")
        
        return all(status == 'healthy' for status in health_status.values())
    
    def show_system_info(self):
        """Показать информацию о системе"""
        logger.info("📊 Информация о системе:")
        logger.info(f"   🌐 Основное приложение: http://localhost:{self.config['main_app']['port']}")
        logger.info(f"   📈 Мониторинг: http://localhost:{self.config['monitoring']['port']}")
        logger.info(f"   🔗 Масштабирование: http://localhost:{self.config['scaling']['port']}")
        logger.info(f"   📊 Prometheus метрики: http://localhost:{self.config['main_app']['port']}/metrics")
        logger.info(f"   🧪 API документация: http://localhost:{self.config['main_app']['port']}/docs")
        logger.info(f"   📁 Логи: logs/")
        logger.info(f"   🗄️ База данных: data/")
    
    def stop_all_services(self):
        """Остановка всех сервисов"""
        logger.info("🛑 Остановка всех сервисов...")
        
        for name, process in self.processes.items():
            try:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
                    logger.info(f"✅ {name} остановлен")
                else:
                    logger.info(f"⚠️ {name} уже остановлен")
            except Exception as e:
                logger.error(f"❌ Ошибка остановки {name}: {e}")
        
        # Принудительная остановка Redis
        try:
            subprocess.run(['redis-cli', 'shutdown'], timeout=5)
            logger.info("✅ Redis остановлен")
        except:
            pass
    
    def signal_handler(self, signum, frame):
        """Обработчик сигналов для корректного завершения"""
        logger.info(f"📡 Получен сигнал {signum}, завершение работы...")
        self.stop_all_services()
        sys.exit(0)
    
    def launch_system(self):
        """Запуск всей системы"""
        logger.info("🚀 Запуск Enhanced ForexBot AI System...")
        
        # Регистрация обработчика сигналов
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            # Проверка зависимостей
            if not self.check_dependencies():
                return False
            
            # Создание директорий
            self.create_directories()
            
            # Запуск Redis
            if not self.start_redis():
                logger.error("❌ Не удалось запустить Redis")
                return False
            
            # Запуск PostgreSQL (опционально)
            self.start_postgres()
            
            # Запуск основного приложения
            if not self.start_main_app():
                logger.error("❌ Не удалось запустить основное приложение")
                return False
            
            # Запуск системы мониторинга
            if not self.start_monitoring():
                logger.warning("⚠️ Не удалось запустить систему мониторинга")
            
            # Запуск узла масштабирования
            if not self.start_scaling_node():
                logger.warning("⚠️ Не удалось запустить узел масштабирования")
            
            # Запуск тестов (опционально)
            if len(sys.argv) > 1 and sys.argv[1] == '--test':
                self.run_tests()
            
            # Проверка здоровья системы
            if self.check_system_health():
                logger.info("🎉 Система запущена успешно!")
                self.show_system_info()
                
                # Ожидание завершения
                try:
                    while True:
                        time.sleep(60)
                        if not self.check_system_health():
                            logger.warning("⚠️ Обнаружены проблемы в системе")
                except KeyboardInterrupt:
                    logger.info("📡 Получен сигнал завершения")
            else:
                logger.error("❌ Система не работает корректно")
                return False
                
        except Exception as e:
            logger.error(f"❌ Критическая ошибка: {e}")
            return False
        finally:
            self.stop_all_services()

def main():
    """Главная функция"""
    launcher = EnhancedSystemLauncher()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            logger.info("🧪 Запуск в режиме тестирования")
            launcher.launch_system()
        elif sys.argv[1] == '--stop':
            logger.info("🛑 Остановка системы")
            launcher.stop_all_services()
        elif sys.argv[1] == '--status':
            logger.info("📊 Статус системы")
            launcher.check_system_health()
        else:
            print("Использование:")
            print("  python start_enhanced_system.py          # Запуск системы")
            print("  python start_enhanced_system.py --test   # Запуск с тестами")
            print("  python start_enhanced_system.py --stop   # Остановка системы")
            print("  python start_enhanced_system.py --status # Статус системы")
    else:
        launcher.launch_system()

if __name__ == "__main__":
    main()