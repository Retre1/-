#!/usr/bin/env python3
"""
Cache System for ForexBot AI
Система кэширования с Redis для улучшения производительности
"""

import redis
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import asyncio
import logging
from functools import wraps
import time

logger = logging.getLogger(__name__)

class CacheManager:
    """Менеджер кэширования с Redis"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, 
                 password: str = None, max_connections: int = 10):
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            decode_responses=False  # Для работы с pickle
        )
        self.default_ttl = 3600  # 1 час по умолчанию
        
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Генерация ключа кэша"""
        # Создаем строку из аргументов
        key_parts = [prefix]
        key_parts.extend([str(arg) for arg in args])
        key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
        
        # Создаем хеш
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Установка значения в кэш"""
        try:
            serialized_value = pickle.dumps(value)
            ttl = ttl or self.default_ttl
            return self.redis_client.setex(key, ttl, serialized_value)
        except Exception as e:
            logger.error(f"Ошибка установки кэша: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Получение значения из кэша"""
        try:
            value = self.redis_client.get(key)
            if value is not None:
                return pickle.loads(value)
            return None
        except Exception as e:
            logger.error(f"Ошибка получения кэша: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Удаление значения из кэша"""
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"Ошибка удаления кэша: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Проверка существования ключа"""
        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"Ошибка проверки кэша: {e}")
            return False
    
    def expire(self, key: str, ttl: int) -> bool:
        """Установка времени жизни ключа"""
        try:
            return bool(self.redis_client.expire(key, ttl))
        except Exception as e:
            logger.error(f"Ошибка установки TTL: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Очистка ключей по паттерну"""
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Ошибка очистки кэша: {e}")
            return 0

class PredictionCache:
    """Кэш для предсказаний AI моделей"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.prefix = "prediction"
        self.ttl = 300  # 5 минут для предсказаний
    
    def get_cached_prediction(self, symbol: str, timeframe: str, 
                            lookback_period: int, model_name: str = None) -> Optional[Dict]:
        """Получение кэшированного предсказания"""
        key = self._generate_prediction_key(symbol, timeframe, lookback_period, model_name)
        return self.cache.get(key)
    
    def set_cached_prediction(self, symbol: str, timeframe: str, lookback_period: int,
                            prediction: Dict, model_name: str = None) -> bool:
        """Сохранение предсказания в кэш"""
        key = self._generate_prediction_key(symbol, timeframe, lookback_period, model_name)
        return self.cache.set(key, prediction, self.ttl)
    
    def _generate_prediction_key(self, symbol: str, timeframe: str, 
                               lookback_period: int, model_name: str = None) -> str:
        """Генерация ключа для предсказания"""
        return self.cache._generate_key(
            self.prefix, symbol, timeframe, lookback_period, model=model_name
        )
    
    def clear_old_predictions(self, symbol: str = None) -> int:
        """Очистка старых предсказаний"""
        pattern = f"{self.prefix}:*"
        if symbol:
            pattern = f"{self.prefix}:{symbol}:*"
        return self.cache.clear_pattern(pattern)

class MarketDataCache:
    """Кэш для рыночных данных"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.prefix = "market_data"
        self.ttl = 60  # 1 минута для рыночных данных
    
    def get_cached_data(self, symbol: str, timeframe: str, 
                       start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        """Получение кэшированных рыночных данных"""
        key = self._generate_data_key(symbol, timeframe, start_time, end_time)
        return self.cache.get(key)
    
    def set_cached_data(self, symbol: str, timeframe: str, 
                       start_time: datetime, end_time: datetime, data: pd.DataFrame) -> bool:
        """Сохранение рыночных данных в кэш"""
        key = self._generate_data_key(symbol, timeframe, start_time, end_time)
        return self.cache.set(key, data, self.ttl)
    
    def _generate_data_key(self, symbol: str, timeframe: str, 
                          start_time: datetime, end_time: datetime) -> str:
        """Генерация ключа для рыночных данных"""
        return self.cache._generate_key(
            self.prefix, symbol, timeframe, 
            start_time.isoformat(), end_time.isoformat()
        )

class BacktestCache:
    """Кэш для результатов backtesting"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.prefix = "backtest"
        self.ttl = 3600  # 1 час для результатов backtesting
    
    def get_cached_results(self, model_name: str, symbol: str, 
                          start_date: str, end_date: str, 
                          parameters: Dict) -> Optional[Dict]:
        """Получение кэшированных результатов backtesting"""
        key = self._generate_results_key(model_name, symbol, start_date, end_date, parameters)
        return self.cache.get(key)
    
    def set_cached_results(self, model_name: str, symbol: str, 
                          start_date: str, end_date: str, 
                          parameters: Dict, results: Dict) -> bool:
        """Сохранение результатов backtesting в кэш"""
        key = self._generate_results_key(model_name, symbol, start_date, end_date, parameters)
        return self.cache.set(key, results, self.ttl)
    
    def _generate_results_key(self, model_name: str, symbol: str, 
                            start_date: str, end_date: str, parameters: Dict) -> str:
        """Генерация ключа для результатов backtesting"""
        return self.cache._generate_key(
            self.prefix, model_name, symbol, start_date, end_date, **parameters
        )

class AnalyticsCache:
    """Кэш для аналитических данных"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.prefix = "analytics"
        self.ttl = 1800  # 30 минут для аналитических данных
    
    def get_cached_metrics(self, user_id: str, period: str) -> Optional[Dict]:
        """Получение кэшированных метрик"""
        key = self._generate_metrics_key(user_id, period)
        return self.cache.get(key)
    
    def set_cached_metrics(self, user_id: str, period: str, metrics: Dict) -> bool:
        """Сохранение метрик в кэш"""
        key = self._generate_metrics_key(user_id, period)
        return self.cache.set(key, metrics, self.ttl)
    
    def _generate_metrics_key(self, user_id: str, period: str) -> str:
        """Генерация ключа для метрик"""
        return self.cache._generate_key(self.prefix, user_id, period)
    
    def get_cached_chart(self, chart_type: str, parameters: Dict) -> Optional[str]:
        """Получение кэшированного графика"""
        key = self._generate_chart_key(chart_type, parameters)
        return self.cache.get(key)
    
    def set_cached_chart(self, chart_type: str, parameters: Dict, chart_data: str) -> bool:
        """Сохранение графика в кэш"""
        key = self._generate_chart_key(chart_type, parameters)
        return self.cache.set(key, chart_data, self.ttl)
    
    def _generate_chart_key(self, chart_type: str, parameters: Dict) -> str:
        """Генерация ключа для графика"""
        return self.cache._generate_key(self.prefix, chart_type, **parameters)

# Декораторы для кэширования
def cache_prediction(ttl: int = 300):
    """Декоратор для кэширования предсказаний"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Создание ключа кэша
            cache_key = f"prediction:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Попытка получить из кэша
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                logger.info(f"Получено из кэша: {cache_key}")
                return cached_result
            
            # Выполнение функции
            result = func(self, *args, **kwargs)
            
            # Сохранение в кэш
            self.cache_manager.set(cache_key, result, ttl)
            logger.info(f"Сохранено в кэш: {cache_key}")
            
            return result
        return wrapper
    return decorator

def cache_backtest_results(ttl: int = 3600):
    """Декоратор для кэширования результатов backtesting"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Создание ключа кэша
            cache_key = f"backtest:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Попытка получить из кэша
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                logger.info(f"Получено из кэша: {cache_key}")
                return cached_result
            
            # Выполнение функции
            result = func(self, *args, **kwargs)
            
            # Сохранение в кэш
            self.cache_manager.set(cache_key, result, ttl)
            logger.info(f"Сохранено в кэш: {cache_key}")
            
            return result
        return wrapper
    return decorator

def cache_analytics(ttl: int = 1800):
    """Декоратор для кэширования аналитических данных"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Создание ключа кэша
            cache_key = f"analytics:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Попытка получить из кэша
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                logger.info(f"Получено из кэша: {cache_key}")
                return cached_result
            
            # Выполнение функции
            result = func(self, *args, **kwargs)
            
            # Сохранение в кэш
            self.cache_manager.set(cache_key, result, ttl)
            logger.info(f"Сохранено в кэш: {cache_key}")
            
            return result
        return wrapper
    return decorator

# Интеграция с существующими классами
class CachedEnsembleModel:
    """Ансамбль моделей с кэшированием"""
    
    def __init__(self, ensemble_model, cache_manager: CacheManager):
        self.ensemble_model = ensemble_model
        self.cache_manager = cache_manager
        self.prediction_cache = PredictionCache(cache_manager)
    
    @cache_prediction(ttl=300)
    def predict_ensemble(self, df: pd.DataFrame) -> Dict:
        """Предсказание с кэшированием"""
        return self.ensemble_model.predict_ensemble(df)
    
    def get_cached_prediction(self, symbol: str, timeframe: str, 
                            lookback_period: int) -> Optional[Dict]:
        """Получение кэшированного предсказания"""
        return self.prediction_cache.get_cached_prediction(symbol, timeframe, lookback_period)
    
    def set_cached_prediction(self, symbol: str, timeframe: str, 
                            lookback_period: int, prediction: Dict) -> bool:
        """Сохранение предсказания в кэш"""
        return self.prediction_cache.set_cached_prediction(symbol, timeframe, lookback_period, prediction)

class CachedBacktester:
    """Backtester с кэшированием"""
    
    def __init__(self, backtester, cache_manager: CacheManager):
        self.backtester = backtester
        self.cache_manager = cache_manager
        self.backtest_cache = BacktestCache(cache_manager)
    
    @cache_backtest_results(ttl=3600)
    def run_backtest(self, df: pd.DataFrame, predictions: np.ndarray, 
                    confidence_threshold: float = 0.6) -> Dict:
        """Backtesting с кэшированием"""
        return self.backtester.run_backtest(df, predictions, confidence_threshold)
    
    def get_cached_results(self, model_name: str, symbol: str, 
                          start_date: str, end_date: str, 
                          parameters: Dict) -> Optional[Dict]:
        """Получение кэшированных результатов"""
        return self.backtest_cache.get_cached_results(model_name, symbol, start_date, end_date, parameters)

# Пример использования
if __name__ == "__main__":
    # Создание менеджера кэша
    cache_manager = CacheManager(host='localhost', port=6379)
    
    # Создание специализированных кэшей
    prediction_cache = PredictionCache(cache_manager)
    market_data_cache = MarketDataCache(cache_manager)
    backtest_cache = BacktestCache(cache_manager)
    analytics_cache = AnalyticsCache(cache_manager)
    
    # Тестирование кэширования предсказаний
    test_prediction = {
        'ensemble_prediction': np.array([[0.3, 0.4, 0.3]]),
        'confidence': 0.75,
        'model_predictions': {
            'lstm': np.array([[0.2, 0.5, 0.3]]),
            'xgboost': np.array([[0.4, 0.3, 0.3]])
        }
    }
    
    # Сохранение предсказания
    success = prediction_cache.set_cached_prediction('EURUSD', 'H1', 100, test_prediction)
    print(f"Предсказание сохранено: {success}")
    
    # Получение предсказания
    cached_prediction = prediction_cache.get_cached_prediction('EURUSD', 'H1', 100)
    print(f"Предсказание получено: {cached_prediction is not None}")
    
    # Очистка старых предсказаний
    cleared = prediction_cache.clear_old_predictions('EURUSD')
    print(f"Очищено предсказаний: {cleared}")
    
    print("✅ Система кэширования протестирована успешно!")