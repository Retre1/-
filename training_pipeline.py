#!/usr/bin/env python3
"""
Training Pipeline for ForexBot AI Models
Система обучения и сохранения AI моделей
"""

import numpy as np
import pandas as pd
import joblib
import pickle
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

# Импорт наших модулей
from advanced_ai_models import (
    AdvancedFeatureEngineer,
    AdvancedLSTMModel,
    AdvancedTransformerModel,
    AdvancedEnsembleModel,
    create_advanced_models
)
from advanced_backtesting import AdvancedBacktester

logger = logging.getLogger(__name__)

class ModelTrainingPipeline:
    """Пайплайн обучения моделей"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.models_dir = Path("data/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Создание инженера признаков
        self.feature_engineer = AdvancedFeatureEngineer()
        
        # Создание ансамбля моделей
        self.ensemble_model = create_advanced_models()
        
        # Backtester для валидации
        self.backtester = AdvancedBacktester()
        
        # История обучения
        self.training_history = {}
        
    def prepare_training_data(self, symbol: str, timeframe: str, 
                            start_date: str, end_date: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Подготовка данных для обучения
        
        Args:
            symbol: Торговый инструмент (например, 'EURUSD')
            timeframe: Временной интервал ('M15', 'H1', 'H4')
            start_date: Дата начала в формате 'YYYY-MM-DD'
            end_date: Дата окончания в формате 'YYYY-MM-DD'
            
        Returns:
            Tuple[pd.DataFrame, np.ndarray]: Подготовленные данные и целевая переменная
        """
        logger.info(f"📊 Подготовка данных для {symbol} {timeframe}")
        
        # Загрузка исторических данных
        df = self._load_market_data(symbol, timeframe, start_date, end_date)
        
        if df.empty:
            raise ValueError(f"Нет данных для {symbol} {timeframe}")
        
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
    
    def _load_market_data(self, symbol: str, timeframe: str, 
                         start_date: str, end_date: str) -> pd.DataFrame:
        """Загрузка рыночных данных"""
        # Здесь должна быть реальная загрузка данных
        # Для демонстрации создаем синтетические данные
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Создание временного ряда
        if timeframe == 'M15':
            freq = '15T'
        elif timeframe == 'H1':
            freq = 'H'
        elif timeframe == 'H4':
            freq = '4H'
        else:
            freq = 'D'
        
        dates = pd.date_range(start_dt, end_dt, freq=freq)
        
        # Создание синтетических данных
        np.random.seed(42)
        price = 100 + np.cumsum(np.random.randn(len(dates)) * 0.1
        volume = np.random.randint(1000, 10000, len(dates))
        
        df = pd.DataFrame({
            'open': price * (1 + np.random.randn(len(dates)) * 0.001),
            'high': price * (1 + abs(np.random.randn(len(dates)) * 0.002)),
            'low': price * (1 - abs(np.random.randn(len(dates)) * 0.002)),
            'close': price,
            'volume': volume
        }, index=dates)
        
        return df
    
    def train_models(self, symbol: str, timeframe: str, 
                    start_date: str, end_date: str,
                    validation_split: float = 0.2) -> Dict:
        """
        Обучение всех моделей
        
        Args:
            symbol: Торговый инструмент
            timeframe: Временной интервал
            start_date: Дата начала обучения
            end_date: Дата окончания обучения
            validation_split: Доля данных для валидации
            
        Returns:
            Dict: Результаты обучения
        """
        logger.info(f"🚀 Начало обучения моделей для {symbol} {timeframe}")
        
        # Подготовка данных
        df, features, target, feature_names = self.prepare_training_data(
            symbol, timeframe, start_date, end_date
        )
        
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
        
        # Сохранение моделей
        model_info = self._save_models(symbol, timeframe, training_results)
        
        # Сохранение результатов обучения
        self._save_training_results(symbol, timeframe, {
            'training_results': training_results,
            'validation_results': validation_results,
            'backtest_results': backtest_results,
            'model_info': model_info,
            'feature_names': feature_names,
            'training_date': datetime.now().isoformat(),
            'data_info': {
                'symbol': symbol,
                'timeframe': timeframe,
                'start_date': start_date,
                'end_date': end_date,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'features_count': len(feature_names)
            }
        })
        
        logger.info(f"✅ Обучение завершено для {symbol} {timeframe}")
        
        return {
            'training_results': training_results,
            'validation_results': validation_results,
            'backtest_results': backtest_results,
            'model_info': model_info
        }
    
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
        
        return validation_results
    
    def _backtest_validation(self, df_val: pd.DataFrame, 
                           X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Backtesting на валидационных данных"""
        try:
            # Получение предсказаний ансамбля
            ensemble_predictions = self.ensemble_model.predict_ensemble(df_val)
            predictions = ensemble_predictions['ensemble_prediction']
            
            # Запуск backtesting
            backtest_results = self.backtester.run_backtest(
                df_val, predictions, confidence_threshold=0.6
            )
            
            logger.info(f"📈 Backtesting результаты:")
            logger.info(f"   Общее количество сделок: {backtest_results['total_trades']}")
            logger.info(f"   Винрейт: {backtest_results['win_rate']:.2f}%")
            logger.info(f"   Общая прибыль: {backtest_results['total_profit']:.2f}")
            logger.info(f"   Sharpe Ratio: {backtest_results['sharpe_ratio']:.4f}")
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"❌ Ошибка backtesting: {e}")
            return {'error': str(e)}
    
    def _save_models(self, symbol: str, timeframe: str, training_results: Dict) -> Dict:
        """Сохранение обученных моделей"""
        model_info = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, result in training_results.items():
            try:
                model = self.ensemble_model.models.get(model_name)
                if model is not None:
                    # Создание директории для модели
                    model_dir = self.models_dir / symbol / timeframe / model_name
                    model_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Сохранение модели
                    model_path = model_dir / f"{model_name}_{timestamp}.joblib"
                    
                    if hasattr(model, 'save_model'):
                        # Для TensorFlow моделей
                        tf_model_path = model_dir / f"{model_name}_{timestamp}.h5"
                        model.save_model(str(tf_model_path))
                        joblib.dump(model, model_path)
                    else:
                        # Для sklearn моделей
                        joblib.dump(model, model_path)
                    
                    # Сохранение метаданных
                    metadata = {
                        'model_name': model_name,
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'training_date': datetime.now().isoformat(),
                        'model_path': str(model_path),
                        'training_metrics': result.get('metrics', {}),
                        'model_type': type(model).__name__
                    }
                    
                    metadata_path = model_dir / f"{model_name}_{timestamp}_metadata.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    model_info[model_name] = {
                        'path': str(model_path),
                        'metadata_path': str(metadata_path),
                        'training_metrics': result.get('metrics', {})
                    }
                    
                    logger.info(f"💾 Модель {model_name} сохранена: {model_path}")
                    
            except Exception as e:
                logger.error(f"❌ Ошибка сохранения модели {model_name}: {e}")
                model_info[model_name] = {'error': str(e)}
        
        return model_info
    
    def _save_training_results(self, symbol: str, timeframe: str, results: Dict):
        """Сохранение результатов обучения"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Создание директории для результатов
        results_dir = self.models_dir / symbol / timeframe / "training_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Сохранение результатов
        results_path = results_dir / f"training_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Сохранение краткого отчета
        summary = {
            'symbol': symbol,
            'timeframe': timeframe,
            'training_date': results['training_date'],
            'models_trained': list(results['model_info'].keys()),
            'validation_accuracy': {
                name: result.get('accuracy', 0) 
                for name, result in results['validation_results'].items()
                if 'accuracy' in result
            },
            'backtest_summary': {
                'total_trades': results['backtest_results'].get('total_trades', 0),
                'win_rate': results['backtest_results'].get('win_rate', 0),
                'total_profit': results['backtest_results'].get('total_profit', 0),
                'sharpe_ratio': results['backtest_results'].get('sharpe_ratio', 0)
            }
        }
        
        summary_path = results_dir / f"summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"💾 Результаты обучения сохранены: {results_path}")
        logger.info(f"📋 Краткий отчет: {summary_path}")
    
    def load_models(self, symbol: str, timeframe: str, model_version: str = None) -> bool:
        """
        Загрузка обученных моделей
        
        Args:
            symbol: Торговый инструмент
            timeframe: Временной интервал
            model_version: Версия модели (если None, загружается последняя)
            
        Returns:
            bool: Успешность загрузки
        """
        logger.info(f"📂 Загрузка моделей для {symbol} {timeframe}")
        
        try:
            model_dir = self.models_dir / symbol / timeframe
            
            if not model_dir.exists():
                logger.error(f"❌ Директория моделей не найдена: {model_dir}")
                return False
            
            # Поиск доступных моделей
            available_models = {}
            for model_name in self.ensemble_model.models.keys():
                model_subdir = model_dir / model_name
                if model_subdir.exists():
                    model_files = list(model_subdir.glob("*.joblib"))
                    if model_files:
                        # Выбор версии модели
                        if model_version:
                            target_file = model_subdir / f"{model_name}_{model_version}.joblib"
                            if target_file.exists():
                                available_models[model_name] = target_file
                        else:
                            # Последняя версия
                            latest_file = max(model_files, key=lambda x: x.stat().st_mtime)
                            available_models[model_name] = latest_file
            
            if not available_models:
                logger.error(f"❌ Не найдены модели для {symbol} {timeframe}")
                return False
            
            # Загрузка моделей
            for model_name, model_path in available_models.items():
                try:
                    model = joblib.load(model_path)
                    self.ensemble_model.models[model_name] = model
                    logger.info(f"✅ Загружена модель {model_name}: {model_path}")
                    
                    # Загрузка метаданных
                    metadata_path = model_path.with_suffix('_metadata.json')
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        logger.info(f"📋 Метаданные {model_name}: {metadata.get('training_date', 'N/A')}")
                    
                except Exception as e:
                    logger.error(f"❌ Ошибка загрузки модели {model_name}: {e}")
            
            # Загрузка информации о признаках
            feature_info_path = model_dir / "feature_info.json"
            if feature_info_path.exists():
                with open(feature_info_path, 'r') as f:
                    feature_info = json.load(f)
                self.ensemble_model.feature_engineer.feature_names = feature_info.get('feature_names', [])
                logger.info(f"📊 Загружена информация о {len(self.ensemble_model.feature_engineer.feature_names)} признаках")
            
            logger.info(f"✅ Загружено {len(available_models)} моделей")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки моделей: {e}")
            return False
    
    def get_model_info(self, symbol: str, timeframe: str) -> Dict:
        """Получение информации о моделях"""
        model_dir = self.models_dir / symbol / timeframe
        
        if not model_dir.exists():
            return {}
        
        model_info = {}
        
        for model_subdir in model_dir.iterdir():
            if model_subdir.is_dir() and model_subdir.name != "training_results":
                model_name = model_subdir.name
                model_files = list(model_subdir.glob("*.joblib"))
                
                if model_files:
                    latest_file = max(model_files, key=lambda x: x.stat().st_mtime)
                    metadata_path = latest_file.with_suffix('_metadata.json')
                    
                    info = {
                        'model_path': str(latest_file),
                        'last_modified': datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat(),
                        'file_size': latest_file.stat().st_size
                    }
                    
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        info.update(metadata)
                    
                    model_info[model_name] = info
        
        return model_info
    
    def retrain_models(self, symbol: str, timeframe: str, 
                      days_back: int = 365) -> Dict:
        """
        Переобучение моделей на новых данных
        
        Args:
            symbol: Торговый инструмент
            timeframe: Временной интервал
            days_back: Количество дней назад для обучения
            
        Returns:
            Dict: Результаты переобучения
        """
        logger.info(f"🔄 Переобучение моделей для {symbol} {timeframe}")
        
        # Определение периода обучения
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Обучение на новых данных
        results = self.train_models(
            symbol, timeframe,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        logger.info(f"✅ Переобучение завершено для {symbol} {timeframe}")
        
        return results

# Пример использования
if __name__ == "__main__":
    # Конфигурация
    config = {
        "ai": {
            "models": ["lstm", "xgboost", "lightgbm"],
            "timeframes": ["M15", "H1", "H4"],
            "lookback_periods": [50, 100, 200],
            "retrain_interval": 24,
            "min_accuracy_threshold": 0.65
        }
    }
    
    # Создание пайплайна обучения
    pipeline = ModelTrainingPipeline(config)
    
    # Обучение моделей
    print("🚀 Начало обучения моделей...")
    
    results = pipeline.train_models(
        symbol="EURUSD",
        timeframe="H1",
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    print("✅ Обучение завершено!")
    print(f"📊 Результаты валидации:")
    for model_name, result in results['validation_results'].items():
        if 'accuracy' in result:
            print(f"   {model_name}: {result['accuracy']:.4f}")
    
    print(f"📈 Backtesting результаты:")
    backtest = results['backtest_results']
    print(f"   Общее количество сделок: {backtest.get('total_trades', 0)}")
    print(f"   Винрейт: {backtest.get('win_rate', 0):.2f}%")
    print(f"   Общая прибыль: {backtest.get('total_profit', 0):.2f}")
    print(f"   Sharpe Ratio: {backtest.get('sharpe_ratio', 0):.4f}")
    
    # Загрузка моделей
    print("\n📂 Загрузка моделей...")
    success = pipeline.load_models("EURUSD", "H1")
    if success:
        print("✅ Модели загружены успешно!")
    else:
        print("❌ Ошибка загрузки моделей")
    
    # Информация о моделях
    model_info = pipeline.get_model_info("EURUSD", "H1")
    print(f"\n📋 Информация о моделях:")
    for model_name, info in model_info.items():
        print(f"   {model_name}: {info.get('training_date', 'N/A')}")