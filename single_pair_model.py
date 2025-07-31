#!/usr/bin/env python3
"""
Single Pair Model System
Система модели для одной валютной пары
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from advanced_ai_models import (
    AdvancedFeatureEngineer,
    AdvancedLSTMModel,
    AdvancedTransformerModel,
    AdvancedEnsembleModel,
    create_advanced_models
)
from advanced_backtesting import AdvancedBacktester

logger = logging.getLogger(__name__)

class SinglePairModel:
    """Модель для одной валютной пары"""
    
    def __init__(self, symbol: str, timeframe: str, config: Dict):
        self.symbol = symbol
        self.timeframe = timeframe
        self.config = config
        
        # Создание директории для модели
        self.model_dir = Path(f"data/models/{symbol}/{timeframe}")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Инженер признаков
        self.feature_engineer = AdvancedFeatureEngineer()
        
        # Ансамбль моделей
        self.ensemble_model = create_advanced_models()
        
        # Backtester
        self.backtester = AdvancedBacktester()
        
        # Статус модели
        self.model_status = {
            "symbol": symbol,
            "timeframe": timeframe,
            "trained": False,
            "last_trained": None,
            "accuracy": 0.0,
            "backtest_results": {},
            "model_info": {}
        }
        
        # Загрузка существующей модели
        self._load_existing_model()
    
    def _load_existing_model(self):
        """Загрузка существующей модели"""
        try:
            model_files = list(self.model_dir.glob("*.joblib"))
            if model_files:
                latest_file = max(model_files, key=lambda x: x.stat().st_mtime)
                
                # Загрузка модели
                self.ensemble_model = joblib.load(latest_file)
                
                # Загрузка метаданных
                metadata_path = latest_file.with_suffix('_metadata.json')
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    self.model_status.update({
                        "trained": True,
                        "last_trained": metadata.get('training_date'),
                        "accuracy": metadata.get('training_metrics', {}).get('accuracy', 0.0),
                        "model_info": metadata
                    })
                
                logger.info(f"✅ Загружена существующая модель для {self.symbol} {self.timeframe}")
                
        except Exception as e:
            logger.warning(f"⚠️ Не удалось загрузить существующую модель: {e}")
    
    def prepare_data(self, start_date: str, end_date: str) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """Подготовка данных для обучения"""
        
        logger.info(f"📊 Подготовка данных для {self.symbol} {self.timeframe}")
        
        # Загрузка исторических данных
        df = self._load_market_data(start_date, end_date)
        
        if df.empty:
            raise ValueError(f"Нет данных для {self.symbol} {self.timeframe}")
        
        # Создание технических индикаторов
        df = self.feature_engineer.create_technical_indicators(df)
        
        # Создание продвинутых признаков
        df = self.feature_engineer.create_advanced_features(df)
        
        # Удаление NaN значений
        df = df.dropna()
        
        # Подготовка признаков
        features, feature_names = self.feature_engineer.prepare_features(df)
        
        # Создание целевой переменной
        target = self.ensemble_model._create_target_variable(df['close'])
        
        logger.info(f"✅ Подготовлено {len(features)} записей с {len(feature_names)} признаками")
        
        return df, features, target, feature_names
    
    def _load_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Загрузка рыночных данных для конкретной пары"""
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Определение частоты данных
        if self.timeframe == 'M15':
            freq = '15T'
        elif self.timeframe == 'H1':
            freq = 'H'
        elif self.timeframe == 'H4':
            freq = '4H'
        elif self.timeframe == 'D1':
            freq = 'D'
        else:
            freq = 'H'
        
        dates = pd.date_range(start_dt, end_dt, freq=freq)
        
        # Создание синтетических данных (замените на реальную загрузку)
        np.random.seed(42)
        
        # Базовый тренд в зависимости от пары
        if self.symbol == 'EURUSD':
            base_price = 1.0850
            volatility = 0.001
        elif self.symbol == 'GBPUSD':
            base_price = 1.2650
            volatility = 0.0015
        elif self.symbol == 'USDJPY':
            base_price = 150.0
            volatility = 0.5
        else:
            base_price = 1.0
            volatility = 0.001
        
        # Создание цены с учетом специфики пары
        price = base_price + np.cumsum(np.random.randn(len(dates)) * volatility)
        volume = np.random.randint(1000, 10000, len(dates))
        
        df = pd.DataFrame({
            'open': price * (1 + np.random.randn(len(dates)) * 0.0005),
            'high': price * (1 + abs(np.random.randn(len(dates)) * 0.001)),
            'low': price * (1 - abs(np.random.randn(len(dates)) * 0.001)),
            'close': price,
            'volume': volume
        }, index=dates)
        
        return df
    
    def train_model(self, start_date: str, end_date: str, validation_split: float = 0.2) -> Dict:
        """Обучение модели для конкретной пары"""
        
        logger.info(f"🚀 Обучение модели для {self.symbol} {self.timeframe}")
        
        try:
            # Подготовка данных
            df, features, target, feature_names = self.prepare_data(start_date, end_date)
            
            # Разделение на обучающую и валидационную выборки
            split_idx = int(len(features) * (1 - validation_split))
            
            X_train = features[:split_idx]
            y_train = target[:split_idx]
            X_val = features[split_idx:]
            y_val = target[split_idx:]
            
            logger.info(f"📈 Обучающая выборка: {len(X_train)} записей")
            logger.info(f"📊 Валидационная выборка: {len(X_val)} записей")
            
            # Обучение ансамбля моделей
            training_results = self.ensemble_model.train_models(df.iloc[:split_idx])
            
            # Валидация моделей
            validation_results = self._validate_models(X_val, y_val)
            
            # Backtesting на валидационных данных
            backtest_results = self._backtest_validation(df.iloc[split_idx:], X_val, y_val)
            
            # Сохранение модели
            model_info = self._save_model(training_results, feature_names)
            
            # Обновление статуса
            self.model_status.update({
                "trained": True,
                "last_trained": datetime.now().isoformat(),
                "accuracy": validation_results.get('ensemble_accuracy', 0.0),
                "backtest_results": backtest_results,
                "model_info": model_info
            })
            
            # Сохранение статуса
            self._save_model_status()
            
            logger.info(f"✅ Обучение завершено для {self.symbol} {self.timeframe}")
            
            return {
                'training_results': training_results,
                'validation_results': validation_results,
                'backtest_results': backtest_results,
                'model_info': model_info
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка обучения модели {self.symbol} {self.timeframe}: {e}")
            return {'error': str(e)}
    
    def _validate_models(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Валидация моделей"""
        validation_results = {}
        
        for model_name, model in self.ensemble_model.models.items():
            try:
                if hasattr(model, 'predict'):
                    predictions = model.predict(X_val)
                    
                    # Расчет точности
                    if len(predictions.shape) > 1:
                        y_pred = np.argmax(predictions, axis=1)
                    else:
                        y_pred = predictions
                    
                    accuracy = np.mean(y_pred == y_val)
                    validation_results[model_name] = {
                        'accuracy': accuracy,
                        'predictions_shape': predictions.shape
                    }
                    
                    logger.info(f"📊 {model_name}: точность = {accuracy:.4f}")
                    
            except Exception as e:
                logger.error(f"❌ Ошибка валидации {model_name}: {e}")
                validation_results[model_name] = {'error': str(e)}
        
        # Расчет средней точности ансамбля
        accuracies = [result.get('accuracy', 0) for result in validation_results.values() if 'accuracy' in result]
        if accuracies:
            validation_results['ensemble_accuracy'] = sum(accuracies) / len(accuracies)
        
        return validation_results
    
    def _backtest_validation(self, df_val: pd.DataFrame, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Backtesting на валидационных данных"""
        try:
            # Получение предсказаний ансамбля
            ensemble_predictions = self.ensemble_model.predict_ensemble(df_val)
            predictions = ensemble_predictions['ensemble_prediction']
            
            # Запуск backtesting
            backtest_results = self.backtester.run_backtest(
                df_val, predictions, confidence_threshold=0.6
            )
            
            logger.info(f"📈 Backtesting результаты для {self.symbol} {self.timeframe}:")
            logger.info(f"   Общее количество сделок: {backtest_results['total_trades']}")
            logger.info(f"   Винрейт: {backtest_results['win_rate']:.2f}%")
            logger.info(f"   Общая прибыль: {backtest_results['total_profit']:.2f}")
            logger.info(f"   Sharpe Ratio: {backtest_results['sharpe_ratio']:.4f}")
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"❌ Ошибка backtesting: {e}")
            return {'error': str(e)}
    
    def _save_model(self, training_results: Dict, feature_names: List[str]) -> Dict:
        """Сохранение модели"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохранение модели
        model_path = self.model_dir / f"{self.symbol}_{self.timeframe}_{timestamp}.joblib"
        joblib.dump(self.ensemble_model, model_path)
        
        # Сохранение метаданных
        metadata = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'training_date': datetime.now().isoformat(),
            'model_path': str(model_path),
            'training_metrics': training_results,
            'feature_names': feature_names,
            'feature_count': len(feature_names),
            'model_type': 'SinglePairEnsemble'
        }
        
        metadata_path = self.model_dir / f"{self.symbol}_{self.timeframe}_{timestamp}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Сохранение информации о признаках
        feature_info_path = self.model_dir / "feature_info.json"
        with open(feature_info_path, 'w') as f:
            json.dump({'feature_names': feature_names}, f, indent=2)
        
        logger.info(f"💾 Модель сохранена: {model_path}")
        
        return {
            'model_path': str(model_path),
            'metadata_path': str(metadata_path),
            'feature_count': len(feature_names)
        }
    
    def _save_model_status(self):
        """Сохранение статуса модели"""
        status_path = self.model_dir / "model_status.json"
        with open(status_path, 'w') as f:
            json.dump(self.model_status, f, indent=2)
    
    def predict(self, market_data: pd.DataFrame) -> Dict:
        """Получение предсказания"""
        
        if not self.model_status['trained']:
            return {'error': 'Model not trained'}
        
        try:
            # Подготовка данных
            df = self.feature_engineer.create_technical_indicators(market_data)
            df = self.feature_engineer.create_advanced_features(df)
            df = df.dropna()
            
            if df.empty:
                return {'error': 'No valid data after preprocessing'}
            
            # Получение предсказания
            prediction = self.ensemble_model.predict_ensemble(df)
            
            # Добавление информации о модели
            prediction['model_info'] = {
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'last_trained': self.model_status['last_trained'],
                'accuracy': self.model_status['accuracy']
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"❌ Ошибка предсказания: {e}")
            return {'error': str(e)}
    
    def get_model_info(self) -> Dict:
        """Получение информации о модели"""
        return self.model_status
    
    def retrain_model(self, days_back: int = 365) -> Dict:
        """Переобучение модели"""
        
        logger.info(f"🔄 Переобучение модели {self.symbol} {self.timeframe}")
        
        # Определение периода обучения
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Обучение на новых данных
        results = self.train_model(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        logger.info(f"✅ Переобучение завершено для {self.symbol} {self.timeframe}")
        
        return results

class SinglePairModelManager:
    """Менеджер моделей для отдельных пар"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.models_dir = Path("data/models")
        
    def create_model(self, symbol: str, timeframe: str) -> SinglePairModel:
        """Создание модели для пары"""
        
        model_key = f"{symbol}_{timeframe}"
        
        if model_key not in self.models:
            self.models[model_key] = SinglePairModel(symbol, timeframe, self.config)
            logger.info(f"✅ Создана модель для {symbol} {timeframe}")
        
        return self.models[model_key]
    
    def train_all_models(self, pairs: List[Tuple[str, str]], 
                        start_date: str, end_date: str) -> Dict:
        """Обучение всех моделей"""
        
        results = {}
        
        for symbol, timeframe in pairs:
            logger.info(f"🎯 Обучение модели {symbol} {timeframe}")
            
            model = self.create_model(symbol, timeframe)
            result = model.train_model(start_date, end_date)
            
            results[f"{symbol}_{timeframe}"] = result
            
            if 'error' not in result:
                logger.info(f"✅ Модель {symbol} {timeframe} обучена успешно")
            else:
                logger.error(f"❌ Ошибка обучения {symbol} {timeframe}: {result['error']}")
        
        return results
    
    def get_model_status(self) -> Dict:
        """Получение статуса всех моделей"""
        status = {}
        
        for model_key, model in self.models.items():
            status[model_key] = model.get_model_info()
        
        return status
    
    def predict_all(self, market_data: Dict) -> Dict:
        """Предсказания для всех моделей"""
        predictions = {}
        
        for model_key, model in self.models.items():
            if model_key in market_data:
                prediction = model.predict(market_data[model_key])
                predictions[model_key] = prediction
        
        return predictions

# Пример использования
if __name__ == "__main__":
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
    
    # Обучение всех моделей
    print("🚀 Обучение моделей для отдельных пар...")
    
    results = manager.train_all_models(
        pairs=pairs,
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    # Анализ результатов
    print("\n📊 Результаты обучения:")
    for pair_key, result in results.items():
        if 'error' not in result:
            validation = result.get('validation_results', {})
            backtest = result.get('backtest_results', {})
            
            print(f"\n{pair_key}:")
            print(f"  Точность: {validation.get('ensemble_accuracy', 0):.4f}")
            print(f"  Сделки: {backtest.get('total_trades', 0)}")
            print(f"  Винрейт: {backtest.get('win_rate', 0):.2f}%")
            print(f"  Прибыль: {backtest.get('total_profit', 0):.2f}")
        else:
            print(f"\n{pair_key}: Ошибка - {result['error']}")
    
    # Получение статуса
    status = manager.get_model_status()
    print(f"\n📋 Статус моделей:")
    for pair_key, info in status.items():
        print(f"  {pair_key}: {'Обучена' if info['trained'] else 'Не обучена'}")
    
    print("\n🎉 Обучение моделей завершено!")