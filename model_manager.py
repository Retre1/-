#!/usr/bin/env python3
"""
Model Manager for ForexBot AI
Менеджер моделей для интеграции с основным ботом
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import schedule
import threading
import time

from training_pipeline import ModelTrainingPipeline
from advanced_ai_models import create_advanced_models
from cache_system import CacheManager, PredictionCache

logger = logging.getLogger(__name__)

class ModelManager:
    """Менеджер моделей для автоматического обучения и управления"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.training_pipeline = ModelTrainingPipeline(config)
        self.cache_manager = CacheManager()
        self.prediction_cache = PredictionCache(self.cache_manager)
        
        # Состояние моделей
        self.models_status = {}
        self.training_schedule = {}
        
        # Автоматическое обучение
        self.auto_training_enabled = config.get('ai', {}).get('auto_training', True)
        self.retrain_interval = config.get('ai', {}).get('retrain_interval', 24)  # часы
        
        # Минимальные требования к точности
        self.min_accuracy_threshold = config.get('ai', {}).get('min_accuracy_threshold', 0.65)
        
        # Запуск планировщика
        if self.auto_training_enabled:
            self._start_scheduler()
    
    def _start_scheduler(self):
        """Запуск планировщика автоматического обучения"""
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)  # Проверка каждую минуту
        
        # Настройка расписания
        schedule.every(self.retrain_interval).hours.do(self._auto_retrain_all_models)
        
        # Запуск в отдельном потоке
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logger.info(f"🕐 Планировщик автоматического обучения запущен (интервал: {self.retrain_interval} часов)")
    
    def initialize_models(self, symbols: List[str], timeframes: List[str]) -> bool:
        """
        Инициализация моделей для всех символов и таймфреймов
        
        Args:
            symbols: Список торговых инструментов
            timeframes: Список временных интервалов
            
        Returns:
            bool: Успешность инициализации
        """
        logger.info(f"🚀 Инициализация моделей для {len(symbols)} символов и {len(timeframes)} таймфреймов")
        
        success_count = 0
        total_count = len(symbols) * len(timeframes)
        
        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    # Попытка загрузки существующих моделей
                    if self.training_pipeline.load_models(symbol, timeframe):
                        self.models_status[f"{symbol}_{timeframe}"] = {
                            'status': 'loaded',
                            'last_loaded': datetime.now().isoformat(),
                            'accuracy': self._get_model_accuracy(symbol, timeframe)
                        }
                        success_count += 1
                        logger.info(f"✅ Модели загружены для {symbol} {timeframe}")
                    else:
                        # Если модели не найдены, обучаем новые
                        logger.info(f"📚 Обучение новых моделей для {symbol} {timeframe}")
                        if self._train_models_for_symbol(symbol, timeframe):
                            success_count += 1
                            logger.info(f"✅ Модели обучены для {symbol} {timeframe}")
                        else:
                            logger.error(f"❌ Ошибка обучения моделей для {symbol} {timeframe}")
                            
                except Exception as e:
                    logger.error(f"❌ Ошибка инициализации моделей для {symbol} {timeframe}: {e}")
        
        success_rate = success_count / total_count * 100
        logger.info(f"📊 Инициализация завершена: {success_count}/{total_count} ({success_rate:.1f}%)")
        
        return success_rate > 80  # Успех если более 80% моделей загружены
    
    def _train_models_for_symbol(self, symbol: str, timeframe: str) -> bool:
        """Обучение моделей для конкретного символа и таймфрейма"""
        try:
            # Определение периода обучения (последние 365 дней)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            # Обучение моделей
            results = self.training_pipeline.train_models(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            # Проверка качества моделей
            validation_results = results.get('validation_results', {})
            avg_accuracy = 0
            valid_models = 0
            
            for model_name, result in validation_results.items():
                if 'accuracy' in result:
                    avg_accuracy += result['accuracy']
                    valid_models += 1
            
            if valid_models > 0:
                avg_accuracy /= valid_models
                
                if avg_accuracy >= self.min_accuracy_threshold:
                    self.models_status[f"{symbol}_{timeframe}"] = {
                        'status': 'trained',
                        'last_trained': datetime.now().isoformat(),
                        'accuracy': avg_accuracy,
                        'backtest_results': results.get('backtest_results', {})
                    }
                    return True
                else:
                    logger.warning(f"⚠️ Низкая точность моделей для {symbol} {timeframe}: {avg_accuracy:.4f}")
                    return False
            else:
                logger.error(f"❌ Нет валидных моделей для {symbol} {timeframe}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка обучения моделей для {symbol} {timeframe}: {e}")
            return False
    
    def _get_model_accuracy(self, symbol: str, timeframe: str) -> float:
        """Получение точности моделей"""
        try:
            model_info = self.training_pipeline.get_model_info(symbol, timeframe)
            accuracies = []
            
            for model_name, info in model_info.items():
                if 'training_metrics' in info and 'accuracy' in info['training_metrics']:
                    accuracies.append(info['training_metrics']['accuracy'])
            
            return sum(accuracies) / len(accuracies) if accuracies else 0.0
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения точности моделей: {e}")
            return 0.0
    
    def get_prediction(self, symbol: str, timeframe: str, market_data: pd.DataFrame) -> Dict:
        """
        Получение предсказания от обученных моделей
        
        Args:
            symbol: Торговый инструмент
            timeframe: Временной интервал
            market_data: Рыночные данные
            
        Returns:
            Dict: Результат предсказания
        """
        try:
            # Проверка статуса моделей
            model_key = f"{symbol}_{timeframe}"
            if model_key not in self.models_status:
                logger.error(f"❌ Модели не инициализированы для {symbol} {timeframe}")
                return {'error': 'Models not initialized'}
            
            # Проверка кэша
            cache_key = f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H')}"
            cached_prediction = self.prediction_cache.get_cached_prediction(symbol, timeframe, 100)
            
            if cached_prediction:
                logger.info(f"📋 Использовано кэшированное предсказание для {symbol} {timeframe}")
                return cached_prediction
            
            # Получение предсказания от ансамбля
            ensemble_predictions = self.training_pipeline.ensemble_model.predict_ensemble(market_data)
            
            # Сохранение в кэш
            self.prediction_cache.set_cached_prediction(symbol, timeframe, 100, ensemble_predictions)
            
            # Обновление статистики
            self._update_prediction_stats(symbol, timeframe, ensemble_predictions)
            
            return ensemble_predictions
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения предсказания для {symbol} {timeframe}: {e}")
            return {'error': str(e)}
    
    def _update_prediction_stats(self, symbol: str, timeframe: str, prediction: Dict):
        """Обновление статистики предсказаний"""
        model_key = f"{symbol}_{timeframe}"
        
        if model_key in self.models_status:
            if 'prediction_count' not in self.models_status[model_key]:
                self.models_status[model_key]['prediction_count'] = 0
                self.models_status[model_key]['last_prediction'] = None
            
            self.models_status[model_key]['prediction_count'] += 1
            self.models_status[model_key]['last_prediction'] = datetime.now().isoformat()
    
    def retrain_models(self, symbol: str, timeframe: str, force: bool = False) -> Dict:
        """
        Переобучение моделей
        
        Args:
            symbol: Торговый инструмент
            timeframe: Временной интервал
            force: Принудительное переобучение
            
        Returns:
            Dict: Результаты переобучения
        """
        try:
            logger.info(f"🔄 Переобучение моделей для {symbol} {timeframe}")
            
            # Проверка необходимости переобучения
            if not force:
                model_key = f"{symbol}_{timeframe}"
                if model_key in self.models_status:
                    last_trained = datetime.fromisoformat(self.models_status[model_key].get('last_trained', '2020-01-01'))
                    hours_since_training = (datetime.now() - last_trained).total_seconds() / 3600
                    
                    if hours_since_training < self.retrain_interval:
                        logger.info(f"⏰ Переобучение не требуется для {symbol} {timeframe} (прошло {hours_since_training:.1f} часов)")
                        return {'status': 'not_required', 'hours_since_training': hours_since_training}
            
            # Переобучение моделей
            results = self.training_pipeline.retrain_models(symbol, timeframe)
            
            # Обновление статуса
            model_key = f"{symbol}_{timeframe}"
            self.models_status[model_key] = {
                'status': 'retrained',
                'last_trained': datetime.now().isoformat(),
                'accuracy': self._get_model_accuracy(symbol, timeframe),
                'backtest_results': results.get('backtest_results', {})
            }
            
            # Очистка кэша предсказаний
            self.prediction_cache.clear_old_predictions(symbol)
            
            logger.info(f"✅ Переобучение завершено для {symbol} {timeframe}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Ошибка переобучения моделей для {symbol} {timeframe}: {e}")
            return {'error': str(e)}
    
    def _auto_retrain_all_models(self):
        """Автоматическое переобучение всех моделей"""
        logger.info("🤖 Запуск автоматического переобучения всех моделей")
        
        symbols = self.config.get('mt5', {}).get('symbols', ['EURUSD'])
        timeframes = self.config.get('ai', {}).get('timeframes', ['H1'])
        
        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    self.retrain_models(symbol, timeframe, force=False)
                except Exception as e:
                    logger.error(f"❌ Ошибка автоматического переобучения {symbol} {timeframe}: {e}")
    
    def get_models_status(self) -> Dict:
        """Получение статуса всех моделей"""
        return {
            'models_status': self.models_status,
            'auto_training_enabled': self.auto_training_enabled,
            'retrain_interval': self.retrain_interval,
            'min_accuracy_threshold': self.min_accuracy_threshold,
            'total_models': len(self.models_status),
            'loaded_models': len([m for m in self.models_status.values() if m['status'] == 'loaded']),
            'trained_models': len([m for m in self.models_status.values() if m['status'] == 'trained']),
            'retrained_models': len([m for m in self.models_status.values() if m['status'] == 'retrained'])
        }
    
    def get_model_performance(self, symbol: str, timeframe: str) -> Dict:
        """Получение производительности моделей"""
        try:
            model_info = self.training_pipeline.get_model_info(symbol, timeframe)
            
            performance = {
                'symbol': symbol,
                'timeframe': timeframe,
                'models': {},
                'overall_accuracy': 0.0,
                'backtest_results': {}
            }
            
            accuracies = []
            
            for model_name, info in model_info.items():
                model_perf = {
                    'training_date': info.get('training_date', 'N/A'),
                    'file_size': info.get('file_size', 0),
                    'accuracy': info.get('training_metrics', {}).get('accuracy', 0.0)
                }
                
                performance['models'][model_name] = model_perf
                accuracies.append(model_perf['accuracy'])
            
            if accuracies:
                performance['overall_accuracy'] = sum(accuracies) / len(accuracies)
            
            # Получение результатов backtesting
            model_key = f"{symbol}_{timeframe}"
            if model_key in self.models_status:
                performance['backtest_results'] = self.models_status[model_key].get('backtest_results', {})
            
            return performance
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения производительности моделей: {e}")
            return {'error': str(e)}
    
    def cleanup_old_models(self, days_to_keep: int = 30):
        """Очистка старых версий моделей"""
        try:
            logger.info(f"🧹 Очистка старых моделей (старше {days_to_keep} дней)")
            
            models_dir = Path("data/models")
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            deleted_count = 0
            
            for model_file in models_dir.rglob("*.joblib"):
                if model_file.stat().st_mtime < cutoff_date.timestamp():
                    try:
                        model_file.unlink()
                        deleted_count += 1
                        logger.info(f"🗑️ Удален старый файл модели: {model_file}")
                    except Exception as e:
                        logger.error(f"❌ Ошибка удаления файла {model_file}: {e}")
            
            logger.info(f"✅ Очистка завершена: удалено {deleted_count} файлов")
            
        except Exception as e:
            logger.error(f"❌ Ошибка очистки старых моделей: {e}")

# Интеграция с FastAPI
def create_model_endpoints(app, model_manager: ModelManager):
    """Создание API endpoints для управления моделями"""
    
    @app.post("/api/models/train")
    async def train_models_endpoint(request: Dict):
        """Обучение моделей"""
        symbol = request.get('symbol', 'EURUSD')
        timeframe = request.get('timeframe', 'H1')
        force = request.get('force', False)
        
        results = model_manager.retrain_models(symbol, timeframe, force)
        return results
    
    @app.get("/api/models/status")
    async def get_models_status():
        """Получение статуса моделей"""
        return model_manager.get_models_status()
    
    @app.get("/api/models/performance/{symbol}/{timeframe}")
    async def get_model_performance(symbol: str, timeframe: str):
        """Получение производительности моделей"""
        return model_manager.get_model_performance(symbol, timeframe)
    
    @app.post("/api/models/initialize")
    async def initialize_models_endpoint():
        """Инициализация всех моделей"""
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        timeframes = ["M15", "H1", "H4"]
        
        success = model_manager.initialize_models(symbols, timeframes)
        return {"success": success, "message": "Models initialized" if success else "Initialization failed"}
    
    @app.post("/api/models/cleanup")
    async def cleanup_models_endpoint(request: Dict):
        """Очистка старых моделей"""
        days_to_keep = request.get('days_to_keep', 30)
        model_manager.cleanup_old_models(days_to_keep)
        return {"message": f"Cleanup completed, keeping models from last {days_to_keep} days"}

# Пример использования
if __name__ == "__main__":
    # Конфигурация
    config = {
        "mt5": {
            "symbols": ["EURUSD", "GBPUSD", "USDJPY"]
        },
        "ai": {
            "models": ["lstm", "xgboost", "lightgbm"],
            "timeframes": ["M15", "H1", "H4"],
            "auto_training": True,
            "retrain_interval": 24,
            "min_accuracy_threshold": 0.65
        }
    }
    
    # Создание менеджера моделей
    model_manager = ModelManager(config)
    
    # Инициализация моделей
    print("🚀 Инициализация моделей...")
    success = model_manager.initialize_models(
        symbols=["EURUSD", "GBPUSD"],
        timeframes=["H1", "H4"]
    )
    
    if success:
        print("✅ Модели инициализированы успешно!")
        
        # Получение статуса
        status = model_manager.get_models_status()
        print(f"📊 Статус моделей: {status['total_models']} моделей")
        print(f"   Загружено: {status['loaded_models']}")
        print(f"   Обучено: {status['trained_models']}")
        print(f"   Переобучено: {status['retrained_models']}")
        
        # Получение производительности
        performance = model_manager.get_model_performance("EURUSD", "H1")
        print(f"📈 Производительность EURUSD H1:")
        print(f"   Общая точность: {performance.get('overall_accuracy', 0):.4f}")
        
    else:
        print("❌ Ошибка инициализации моделей")